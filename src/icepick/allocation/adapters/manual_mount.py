"""Manual-mount adapter.

Scans a mounted path (file or directory) and converts everything it
recognises into canonical handoff JSONL the pipeline can consume. The
mount path itself is **never modified** — derived JSONL is written under
``out/intake/runs/<timestamp>/handoff/``.

Supported inputs:

- ``.jsonl`` — one JSON record per line
- ``.json``  — either a JSON array of records, or a JSON object with a
              ``records`` / ``data`` array key
- ``.csv``   — column-mapped table; requires a ``column_map`` arg
- ``.tsv``   — same as CSV but tab-separated
- directory — recurses one level, processes every supported file

A ``column_map`` is ``{canonical_field: source_column}`` and is applied
to CSV/TSV. It projects the source columns onto icepick's canonical
field names without forking the schema. Unmapped columns are preserved
under ``raw_columns`` so nothing from the operator's drop is silently
dropped.

Provenance and truth_policy are stamped on every output record. Records
that already carry these fields are left alone (the operator's stamping
wins).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

SUPPORTED_FILE_SUFFIXES = (".jsonl", ".json", ".csv", ".tsv")


@dataclass
class ScanResult:
    """One file discovered under the mount path."""

    path: Path
    file_type: str  # "jsonl" | "json" | "csv" | "tsv"


@dataclass
class MountResult:
    """What ``mount`` returns once a scan + write completes."""

    records_path: Path
    record_count: int
    files_scanned: list  # list[ScanResult]
    files_skipped: list  # list[(Path, reason)]
    warnings: list


def scan(path) -> list:
    """Find every supported file under ``path`` (recursive, one level).

    Single-file mounts return a one-element list. Directory mounts return
    every supported file. Unknown extensions are silently skipped — the
    caller decides whether to report them.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"mount path not found: {path}")
    if p.is_file():
        kind = _suffix_to_kind(p.suffix)
        if kind is None:
            return []
        return [ScanResult(path=p, file_type=kind)]
    out: list = []
    for child in sorted(p.iterdir()):
        if child.is_dir():
            continue
        kind = _suffix_to_kind(child.suffix)
        if kind is not None:
            out.append(ScanResult(path=child, file_type=kind))
    return out


def mount(
    *,
    path,
    source: str,
    provenance: str,
    truth_policy: str = "unknown",
    column_map: Optional[dict] = None,
    output_dir,
    family: Optional[str] = None,
) -> MountResult:
    """Walk the mount path, produce canonical handoff JSONL, never touch source.

    Returns a ``MountResult`` describing every file the runner saw, with
    skips and warnings annotated for the operator. The caller is
    responsible for writing the accompanying ``ApprovedManifest``.
    """
    scanned = scan(path)
    output_dir = Path(output_dir)
    handoff_dir = output_dir / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    records_path = handoff_dir / "records.jsonl"

    files_skipped: list = []
    warnings: list = []
    record_count = 0
    seen_uids: set = set()

    with records_path.open("w", encoding="utf-8") as out_fh:
        for entry in scanned:
            try:
                for raw in _iter_records(entry, column_map=column_map):
                    record = _stamp(
                        raw,
                        source=source,
                        provenance=provenance,
                        truth_policy=truth_policy,
                        family=family,
                    )
                    if not record.get("statement"):
                        files_skipped.append((entry.path, "record missing 'statement'"))
                        continue
                    out_fh.write(json.dumps(record) + "\n")
                    record_count += 1
                    uid = record.get("uid")
                    if uid:
                        if uid in seen_uids:
                            warnings.append(f"duplicate uid {uid} from {entry.path}")
                        seen_uids.add(uid)
            except Exception as exc:  # noqa: BLE001 — surface per-file, don't abort the mount
                files_skipped.append((entry.path, f"{type(exc).__name__}: {exc}"))

    return MountResult(
        records_path=records_path,
        record_count=record_count,
        files_scanned=scanned,
        files_skipped=files_skipped,
        warnings=warnings,
    )


# --- internals ---------------------------------------------------------------


def _suffix_to_kind(suffix: str) -> Optional[str]:
    s = suffix.lower()
    if s == ".jsonl":
        return "jsonl"
    if s == ".json":
        return "json"
    if s == ".csv":
        return "csv"
    if s == ".tsv":
        return "tsv"
    return None


def _iter_records(entry: ScanResult, *, column_map: Optional[dict]) -> Iterator[dict]:
    if entry.file_type == "jsonl":
        yield from _iter_jsonl(entry.path)
    elif entry.file_type == "json":
        yield from _iter_json(entry.path)
    elif entry.file_type == "csv":
        yield from _iter_csv(entry.path, delimiter=",", column_map=column_map)
    elif entry.file_type == "tsv":
        yield from _iter_csv(entry.path, delimiter="\t", column_map=column_map)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not a JSON object")
            yield row


def _iter_json(path: Path) -> Iterator[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("records") or payload.get("data")
        if rows is None:
            raise ValueError(
                f"{path}: JSON object has neither 'records' nor 'data' array"
            )
    else:
        raise ValueError(f"{path}: top-level JSON must be array or object")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: extracted rows not a list")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {i} is not a JSON object")
        yield row


def _iter_csv(path: Path, *, delimiter: str, column_map: Optional[dict]) -> Iterator[dict]:
    if not column_map:
        raise ValueError(
            f"{path}: CSV/TSV files require --column-map (e.g. statement=question,answer=gold)"
        )
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            projected: dict = {}
            for canonical, source_col in column_map.items():
                if source_col in row:
                    projected[canonical] = row[source_col]
            unmapped = {k: v for k, v in row.items() if k not in column_map.values()}
            if unmapped:
                projected.setdefault("raw_columns", unmapped)
            yield projected


def _stamp(
    raw: dict,
    *,
    source: str,
    provenance: str,
    truth_policy: str,
    family: Optional[str],
) -> dict:
    """Apply mount-level stamps without clobbering record-level overrides."""
    record = dict(raw)
    record.setdefault("source", source)
    record.setdefault("provenance", provenance)
    record.setdefault("truth_policy", truth_policy)
    if family is not None:
        record.setdefault("family", family)
    return record

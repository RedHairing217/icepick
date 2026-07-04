"""File IO for well-posedness scoring."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import PassKRecord


def load_records(paths: Iterable[Path], default_source: str | None = None) -> tuple[list[PassKRecord], list[dict[str, Any]]]:
    records: list[PassKRecord] = []
    inputs: list[dict[str, Any]] = []
    rid = 0
    for path in paths:
        source = default_source or path.stem
        raw_rows = list(_read_rows(path))
        start_count = len(records)
        for raw in raw_rows:
            records.append(PassKRecord.from_raw(raw, rid=rid, default_source=source))
            rid += 1
        inputs.append(
            {
                "path": str(path),
                "source_default": source,
                "records": len(records) - start_count,
            }
        )
    return records, inputs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rid",
        "uid",
        "source",
        "provenance",
        "family",
        "label",
        "pass_at_k",
        "n_correct",
        "n_wrong",
        "n_degenerate",
        "well_posedness_status",
        "well_posedness_score",
        "well_posedness_detail",
        "signals",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["signals"] = json.dumps(row.get("signals") or {}, sort_keys=True)
            writer.writerow(csv_row)


def infer_format(path: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if path.suffix.lower() == ".csv":
        return "csv"
    return "json"


def _read_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _read_jsonl(path)
        return
    if suffix == ".json":
        yield from _read_json(path)
        return
    if suffix == ".csv":
        yield from _read_csv(path)
        return
    raise ValueError(f"unsupported input format for {path}")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def _read_json(path: Path) -> Iterable[dict[str, Any]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        rows = loaded
    elif isinstance(loaded, dict) and isinstance(loaded.get("records"), list):
        rows = loaded["records"]
    else:
        raise ValueError(f"{path} must be a JSON array or an object with a records array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: record {index} is not an object")
        yield row


def _read_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)

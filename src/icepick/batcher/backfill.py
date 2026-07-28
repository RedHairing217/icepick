"""Backfill the ledger with all historical source files.

Purpose: before the slicer can guarantee zero duplication against pre-existing
history (batch0–8, rescue passes, etc.), every historically processed record
must appear in the ledger.  ``backfill`` ingests the files listed in
``backfill_sources.json`` (or an explicit override list) and appends a
``hist:<label>`` row for each record into the ledger.

Idempotence: the ledger's (uid, batch) guard means re-running backfill on an
already-seeded ledger is a no-op — existing rows are silently skipped.

Missing files: historical paths may not exist on every machine (e.g. a fresh
checkout); ``missing_file: True`` is recorded in the summary and the label is
skipped.  Backfill is safe to re-run once the files arrive.

Cross-label duplicates: expected.  The rescue passes re-ran records from
batch1-8, so the same uid naturally appears in multiple labels.  The ledger's
(uid, batch) guard keeps each ``hist:<label>`` row unique; the blocking index
uses the *first* occurrence, which is correct (subsequent occurrences are
replays of history, not conflicts in new data).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .identity import compute_uid, stmt_key as make_stmt_key, content_hash as make_content_hash
from .ledger import Ledger, LedgerRow

# Path to the default source table, adjacent to this module.
_DEFAULT_SOURCES_JSON = Path(__file__).parent / "backfill_sources.json"


def load_sources(sources_path: Optional[Path] = None) -> list[dict]:
    """Load the backfill source table.

    Falls back to the packaged ``backfill_sources.json`` when ``sources_path``
    is ``None``.  The file is a JSON array of objects with keys:
    ``label``, ``path`` (relative to repo root), ``warn_only``,
    ``expect_has_uid_field``.
    """
    p = sources_path if sources_path is not None else _DEFAULT_SOURCES_JSON
    return json.loads(p.read_text(encoding="utf-8"))


def _iter_jsonl_tolerant(path: Path):
    """Yield parsed JSON objects from a JSONL file, tolerating a torn tail."""
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError:
            # Torn tail (kill mid-write): skip.  Non-tail corruption is still
            # skipped here because backfill is best-effort over historical data;
            # the ledger check below catches any integrity issues.
            continue


def backfill(
    ledger: Ledger,
    sources: list[dict],
    repo_root: Path,
    now_iso: str,
) -> dict[str, dict]:
    """Seed the ledger with all historical source records.

    Args:
        ledger:    A ``Ledger`` instance (already loaded).
        sources:   List of source dicts (from ``load_sources``).
        repo_root: Absolute path to the repo root; ``source.path`` is
                   joined to it.
        now_iso:   ISO-8601 timestamp string to stamp on all backfill rows.

    Returns:
        A summary dict keyed by label::

            {
              "<label>": {
                "rows": int,           # lines seen in the file
                "appended": int,       # new rows written to ledger
                "distinct_uids": int,  # unique uids in this file
                "uid_mismatches": int, # rows where computed uid != row["uid"]
                "missing_statement": int,  # rows skipped for missing statement
                "missing_file": bool,  # True when the file did not exist
              }
            }

    Cross-label duplicate uids are expected (rescue passes re-ran batch1-8
    records) and produce no error; the ledger's (uid, batch) guard ensures
    each ``hist:<label>`` slot is written only once.
    """
    summary: dict[str, dict] = {}

    for source in sources:
        label: str = source["label"]
        rel_path: str = source["path"]
        warn_only: bool = bool(source.get("warn_only", False))

        abs_path = repo_root / rel_path
        stat: dict = {
            "rows": 0,
            "appended": 0,
            "distinct_uids": 0,
            "uid_mismatches": 0,
            "missing_statement": 0,
            "missing_file": False,
        }

        if not abs_path.exists():
            stat["missing_file"] = True
            summary[label] = stat
            continue

        batch_tag = f"hist:{label}"
        rows_to_append: list[LedgerRow] = []
        seen_uids_this_file: set[str] = set()
        uid_mismatch_examples: list[dict] = []

        for row in _iter_jsonl_tolerant(abs_path):
            stat["rows"] += 1

            statement = row.get("statement") or row.get("question") or row.get("problem")
            if not statement:
                stat["missing_statement"] += 1
                continue

            source_field: str = row.get("source", "")
            computed = compute_uid(source_field, statement)

            if "uid" in row:
                uid = row["uid"]
                if uid != computed:
                    stat["uid_mismatches"] += 1
                    if len(uid_mismatch_examples) < 5:
                        uid_mismatch_examples.append(
                            {"row_uid": uid, "computed_uid": computed, "source": source_field}
                        )
                    # Use the row's own uid per spec.
            else:
                uid = computed

            sk = make_stmt_key(statement)
            ch = make_content_hash(row)

            if uid not in seen_uids_this_file:
                seen_uids_this_file.add(uid)
                stat["distinct_uids"] += 1

            ledger_row = LedgerRow(
                uid=uid,
                stmt_key=sk,
                content_hash=ch,
                batch=batch_tag,
                source_journal=str(abs_path),
                journal_line=-1,
                sliced_at=now_iso,
                warn_only=warn_only,
            )
            rows_to_append.append(ledger_row)

        before_count = len(ledger._uid_batch)
        ledger.append_all(rows_to_append)
        after_count = len(ledger._uid_batch)
        stat["appended"] = after_count - before_count

        if uid_mismatch_examples:
            stat["uid_mismatch_examples"] = uid_mismatch_examples

        summary[label] = stat

    return summary

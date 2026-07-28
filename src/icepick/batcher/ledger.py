"""Append-only consumed-uid ledger for the bulk-batcher subsystem.

``consumed_uids.jsonl`` is the dedup source of truth for the whole batcher
pipeline.  Every row the slicer commits must be appended here and fsync'd
before any downstream step proceeds.  The ledger is loaded into memory at
startup; all checks are O(1) in-memory dict lookups.

Design choices:
- fsync is deliberate (see ``append_all``).  The rest of the repo uses
  flush-only for audit journals; the ledger is upgraded because it is the
  only automatic dup gate and a partial write would allow duplicate records
  into the funnel.
- Torn final lines (from a kill mid-write) are tolerated on load; torn
  non-final lines indicate a deeper corruption and raise ``LedgerCorruption``.
- warn-only rows are indexed separately: they participate in verdict
  reporting ('warn') but never block a new record.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


class LedgerCorruption(Exception):
    """Raised when a non-final JSONL line in the ledger is unparseable."""


@dataclass
class LedgerRow:
    """One record in ``consumed_uids.jsonl``."""

    uid: str
    stmt_key: str
    content_hash: str
    batch: str           # "hist:<label>" for backfill; "batch<N>" for slices
    source_journal: str
    journal_line: int    # -1 for backfill rows
    sliced_at: str       # ISO-8601 timestamp; caller supplies
    warn_only: bool = False


@dataclass
class Verdict:
    """Result of ``Ledger.check``."""

    kind: str            # 'new' | 'replay' | 'uid_conflict' | 'stmt_conflict' | 'warn'
    prior: Optional[LedgerRow] = field(default=None)


class Ledger:
    """In-memory view of ``consumed_uids.jsonl`` with append + check API."""

    _FILENAME = "consumed_uids.jsonl"
    _SKIPS_FILENAME = "cross_source_skips.jsonl"

    def __init__(self, ledger_dir: Path) -> None:
        self._dir = ledger_dir
        self._path = ledger_dir / self._FILENAME
        self._skips_path = ledger_dir / self._SKIPS_FILENAME
        # blocking indices: uid → first LedgerRow, stmt_key → first LedgerRow
        self._by_uid: dict[str, LedgerRow] = {}
        self._by_stmt: dict[str, LedgerRow] = {}
        # per-uid occurrence count (blocking only)
        self._uid_count: dict[str, int] = {}
        # warn-only index: keys are uids and stmt_keys seen in warn-only rows
        self._warn_uids: dict[str, LedgerRow] = {}
        self._warn_stmts: dict[str, LedgerRow] = {}
        # set of (uid, batch) pairs already in the ledger — idempotence guard
        self._uid_batch: set[tuple[str, str]] = set()
        # torn-tail warning flag (set during load if the last line was corrupt)
        self.torn_tail_skipped: bool = False

    @classmethod
    def load(cls, ledger_dir: Path) -> "Ledger":
        """Load (or create) the ledger from ``ledger_dir/consumed_uids.jsonl``.

        Tolerates a torn final line (partial write from a kill mid-append) by
        skipping it with a runtime warning and setting ``torn_tail_skipped``.
        A corrupt non-final line raises ``LedgerCorruption`` because it
        indicates a write that clobbered intact earlier data.
        """
        ledger = cls(ledger_dir)
        path = ledger._path
        if not path.exists():
            return ledger

        raw_lines: list[str] = path.read_text(encoding="utf-8").splitlines()
        # strip blank lines
        lines = [ln for ln in raw_lines if ln.strip()]
        for i, line in enumerate(lines):
            is_last = i == len(lines) - 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                if is_last:
                    ledger.torn_tail_skipped = True
                    warnings.warn(
                        f"Torn tail line in {path} (index {i}); skipped. "
                        "This is normal after a crash mid-write.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                raise LedgerCorruption(
                    f"Unparseable non-final line {i} in {path}: {line!r}"
                ) from None

            row = LedgerRow(
                uid=data["uid"],
                stmt_key=data["stmt_key"],
                content_hash=data["content_hash"],
                batch=data["batch"],
                source_journal=data["source_journal"],
                journal_line=data["journal_line"],
                sliced_at=data["sliced_at"],
                warn_only=data.get("warn_only", False),
            )
            ledger._index_row(row)

        return ledger

    def _index_row(self, row: LedgerRow) -> None:
        """Add a row to the appropriate in-memory index."""
        self._uid_batch.add((row.uid, row.batch))
        if row.warn_only:
            self._warn_uids.setdefault(row.uid, row)
            self._warn_stmts.setdefault(row.stmt_key, row)
        else:
            if row.uid not in self._by_uid:
                self._by_uid[row.uid] = row
            self._uid_count[row.uid] = self._uid_count.get(row.uid, 0) + 1
            self._by_stmt.setdefault(row.stmt_key, row)

    def check(self, uid: str, stmt_key_: str, content_hash_: str) -> Verdict:
        """Return a Verdict for a candidate (uid, stmt_key, content_hash) triple.

        Precedence (highest to lowest):
        1. uid in blocking index → 'replay' if content_hash matches, else
           'uid_conflict'.
        2. stmt_key in blocking index (uid differs) → 'stmt_conflict'.
        3. uid or stmt_key in warn-only index → 'warn'.
        4. Otherwise → 'new'.
        """
        if uid in self._by_uid:
            prior = self._by_uid[uid]
            kind = "replay" if prior.content_hash == content_hash_ else "uid_conflict"
            return Verdict(kind=kind, prior=prior)
        if stmt_key_ in self._by_stmt:
            return Verdict(kind="stmt_conflict", prior=self._by_stmt[stmt_key_])
        if uid in self._warn_uids or stmt_key_ in self._warn_stmts:
            prior = self._warn_uids.get(uid) or self._warn_stmts.get(stmt_key_)
            return Verdict(kind="warn", prior=prior)
        return Verdict(kind="new")

    def append_all(self, rows: list[LedgerRow]) -> None:
        """Append rows to the ledger file and update in-memory indices.

        Each row is written as a single JSON line.  After writing all rows the
        file is flushed and then fsync'd before close.  fsync is deliberate:
        this file is the dedup source of truth; without it a kernel crash
        between the OS flush and the physical write could leave the file
        shorter than the process believes, causing duplicate records in the
        next run.

        Idempotence guard: a row whose (uid, batch) pair is already in the
        ledger is silently skipped so that crash-recovery re-appends are safe.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        to_write = [r for r in rows if (r.uid, r.batch) not in self._uid_batch]
        if not to_write:
            return
        with self._path.open("a", encoding="utf-8") as fh:
            for row in to_write:
                fh.write(json.dumps(asdict(row)) + "\n")
            fh.flush()
            # fsync ensures the OS page cache is flushed to durable storage
            # before we return.  A process kill after flush but before fsync
            # could lose the append; the ledger would then allow re-processing
            # of records we believed consumed.  This is the one place in the
            # batcher that upgrades flush-only to flush+fsync.
            os.fsync(fh.fileno())
        for row in to_write:
            self._index_row(row)

    def log_skip(self, skip_row: dict) -> None:
        """Append a cross-source skip entry to ``cross_source_skips.jsonl``.

        Flush + fsync mirrors ``append_all``; skips are evidence of policy
        decisions and should be as durable as the ledger itself.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._skips_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(skip_row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

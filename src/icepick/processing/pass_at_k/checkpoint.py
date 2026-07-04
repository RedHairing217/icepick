"""Durable pass@k progress: pause/restart acceptable, full kill unacceptable.

``PassAtKCheckpoint`` gives a production scoring run per-rollout disk
commits so any exit — crash, network failure, Ctrl-C — leaves enough state
on disk to resume by re-running the same ``processing pass_at_k`` command.
Nothing already scored is re-run, and cached model outputs are never
re-billed; the restartability contract is identical in spirit to the
scraper's (:mod:`icepick.allocation.scrape.checkpoint`).

Layout under ``<output_dir>/_progress/``::

    records_done.jsonl   one line per finished record: the full stamped output row
    rollouts.jsonl       append-only audit: one line per rollout (uid, rollout_uid,
                         sample_idx, output, candidate, verdict, from_cache)
    llm_cache.jsonl      one line per PAID model call: {key, output}
    INCOMPLETE           marker; present while a run is unfinished

Every write is append + flush, committed per model call (and per record),
so a kill loses at most the in-flight call. The files double as an audit
trail; ``mark_complete`` removes only the marker, never the data.

The runner scores up to ``max_concurrent`` records in parallel, so every
append and index mutation sits behind one lock.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional


class PassAtKCheckpoint:
    """Per-record progress store for resumable scoring runs."""

    def __init__(self, progress_dir):
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self._records_path = self.progress_dir / "records_done.jsonl"
        self._rollouts_path = self.progress_dir / "rollouts.jsonl"
        self._llm_cache_path = self.progress_dir / "llm_cache.jsonl"
        self._incomplete_path = self.progress_dir / "INCOMPLETE"

        self.resuming = self._incomplete_path.exists()
        self._records: dict = {}
        self._rollouts_by_uid: dict = {}
        self._llm_cache: dict = {}
        self._lock = threading.Lock()
        self._load()
        self.resumed_records = len(self._records)

    def begin(self) -> None:
        """Mark the run in-flight. Removed by ``mark_complete``."""
        self._incomplete_path.write_text("run in progress; safe to resume by re-running\n")

    def mark_complete(self) -> None:
        """The run finished; only the marker goes — the audit data stays."""
        self._incomplete_path.unlink(missing_ok=True)

    def stored_record(self, uid: str) -> Optional[dict]:
        """The committed stamped row for a done record, or ``None`` if not done."""
        with self._lock:
            row = self._records.get(uid)
            return dict(row) if row is not None else None

    def commit_record(self, uid: str, stamped_row: dict) -> None:
        """Durably record one finished record. Called once per record."""
        with self._lock:
            with self._records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(stamped_row) + "\n")
                fh.flush()
            self._records[uid] = dict(stamped_row)

    def append_rollout(self, row_dict: dict) -> None:
        """Append one rollout to the audit trail, flushed immediately."""
        with self._lock:
            with self._rollouts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row_dict) + "\n")
                fh.flush()
            self._rollouts_by_uid.setdefault(row_dict.get("uid"), []).append(dict(row_dict))

    def cached_output(self, key: str) -> Optional[str]:
        """The raw output a previous run already paid for, or ``None``.

        A hit costs nothing on resume. Empty-string outputs ARE cached — the
        model really returned nothing, and we really paid for it. Failures
        are NEVER stored here, so transient errors retry on the next run.
        """
        with self._lock:
            return self._llm_cache.get(key)

    def store_output(self, key: str, output: str) -> None:
        """Durably cache one PAID model output. Never called on failure."""
        with self._lock:
            with self._llm_cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "output": output}) + "\n")
                fh.flush()
            self._llm_cache[key] = output

    # --- internals -------------------------------------------------------------

    def _load(self) -> None:
        for row in _iter_jsonl(self._records_path):
            self._records[row["uid"]] = row
        for row in _iter_jsonl(self._rollouts_path):
            self._rollouts_by_uid.setdefault(row.get("uid"), []).append(row)
        for row in _iter_jsonl(self._llm_cache_path):
            self._llm_cache[row["key"]] = row["output"]


def rollout_key(model: str, question: str, temperature: float, think: bool, sample_idx: int) -> str:
    """Cache key for one PAID model call, mirroring the scraper's ``_statement_key``.

    Keyed per-sample so a kill mid-record resumes at the exact rollout it
    died on rather than redoing the whole record. ``model`` and ``think``
    are included in the key (a deviation from the sketchier spec) so
    switching models — or toggling reasoning — never serves stale cache.
    """
    joined = "\x1f".join(
        [str(model), str(question), str(temperature), str(think), str(sample_idx)]
    )
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A kill mid-write can truncate the final line; everything
                # before it is intact, so skip the torn tail rather than fail.
                continue

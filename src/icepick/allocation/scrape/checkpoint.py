"""Durable scrape progress: pause/restart acceptable, full kill unacceptable.

``ScrapeCheckpoint`` gives a production scrape per-item disk commits so any
exit — crash, network failure, Ctrl-C — leaves enough state on disk to
resume by re-running the same ``allocation run`` command. Nothing already
acquired is refetched, and cached QA answers are never re-billed.

Layout under ``<run_dir>/_progress/``::

    papers_done.jsonl        one line per completed paper: arxiv_id + counts
    candidates.jsonl         append-only raw candidate rows, keyed by arxiv_id
    qa_cache.jsonl           one line per QA-generation call: statement hash -> result
    rate_limited_at          ISO timestamp of the last 429/503 throttle response
    rate_limit_events.jsonl  one line per 429/503: timestamp, status, backoff slept
    INCOMPLETE               marker; present while a run is unfinished

Every write is append + flush, committed per paper (and per LLM call), so
a kill loses at most the in-flight item. The files double as an audit
trail; ``mark_complete`` removes only the marker, never the data.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

_RATE_LIMIT_MARKER = "rate_limited_at"
_RATE_LIMIT_EVENTS = "rate_limit_events.jsonl"
_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 20 * 60


class RateLimitCooldownError(OSError):
    """Raised before scraping when arXiv is still in the cooldown window."""


class ScrapeCheckpoint:
    """Per-paper progress store for resumable acquisition runs."""

    def __init__(self, progress_dir):
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self._papers_path = self.progress_dir / "papers_done.jsonl"
        self._candidates_path = self.progress_dir / "candidates.jsonl"
        self._qa_cache_path = self.progress_dir / "qa_cache.jsonl"
        self._rate_limit_path = self.progress_dir / _RATE_LIMIT_MARKER
        self._rate_limit_events_path = self.progress_dir / _RATE_LIMIT_EVENTS
        self._incomplete_path = self.progress_dir / "INCOMPLETE"

        self.resuming = self._incomplete_path.exists()
        self._done: set = set()
        self._candidates_by_paper: dict = {}
        self._qa_cache: dict = {}
        self._rate_limit_telemetry: dict = {"events": 0, "backoff_seconds": 0.0, "statuses": {}}
        self._load()
        self.resumed_papers = len(self._done)

    def begin(self) -> None:
        """Mark the run in-flight. Removed by ``mark_complete``."""
        self._incomplete_path.write_text("run in progress; safe to resume by re-running\n")

    def mark_complete(self) -> None:
        """The run finished; only the marker goes — the audit data stays."""
        self._incomplete_path.unlink(missing_ok=True)

    def enforce_rate_limit_cooldown(self, *, now: Optional[datetime] = None) -> None:
        """Refuse to hit arXiv again while a fresh throttle marker is present."""
        if not self._rate_limit_path.exists():
            return
        cooldown = _cooldown_seconds()
        if cooldown <= 0:
            return
        stamped = _parse_iso(self._rate_limit_path.read_text(encoding="utf-8").strip())
        if stamped is None:
            return
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elapsed = now - stamped
        remaining = timedelta(seconds=cooldown) - elapsed
        if remaining.total_seconds() <= 0:
            return
        retry_at = now + remaining
        raise RateLimitCooldownError(
            "arXiv is cooling down after a recent 429/503; "
            f"retry after {retry_at.strftime('%H:%M UTC')} "
            f"(set ICEPICK_ARXIV_COOLDOWN_SECONDS=0 to override)"
        )

    def stamp_rate_limited(self, *, now: Optional[datetime] = None) -> None:
        """Record that arXiv returned a throttle response."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        self._rate_limit_path.write_text(_iso_z(now) + "\n", encoding="utf-8")

    def clear_rate_limit(self) -> None:
        """A successful arXiv request proves the cooldown marker is stale."""
        self._rate_limit_path.unlink(missing_ok=True)

    def record_rate_limit(self, status, sleep_seconds, *, now: Optional[datetime] = None) -> None:
        """Durably log one 429/503 throttle event, as it happens.

        The cooldown marker is transient (cleared by the next success); this
        log is history. Appending per event means an invocation the limiter
        kills before its first paper commit still leaves its throttle events
        on disk, so the final report shows the run's lifetime throttling —
        not just the invocation that happened to finish.
        """
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        row = {"at": _iso_z(now), "status": int(status), "backoff_seconds": float(sleep_seconds or 0.0)}
        with self._rate_limit_events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        self._accumulate_rate_limit(row["status"], row["backoff_seconds"])

    def rate_limit_telemetry(self) -> dict:
        """Run-lifetime throttle totals: every invocation's events, this one included."""
        telemetry = self._rate_limit_telemetry
        return {
            "events": telemetry["events"],
            "backoff_seconds": telemetry["backoff_seconds"],
            "statuses": dict(telemetry["statuses"]),
        }

    def stored_candidates(self, arxiv_id: str) -> Optional[list]:
        """The committed candidates for a done paper, or ``None`` if not done."""
        if arxiv_id not in self._done:
            return None
        return list(self._candidates_by_paper.get(arxiv_id, []))

    def commit(self, arxiv_id: str, candidates: list) -> None:
        """Durably record one finished paper. Called once per paper."""
        with self._candidates_path.open("a", encoding="utf-8") as fh:
            for candidate in candidates:
                fh.write(json.dumps({"arxiv_id": arxiv_id, "candidate": candidate}) + "\n")
            fh.flush()
        with self._papers_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"arxiv_id": arxiv_id, "candidates": len(candidates)}) + "\n")
            fh.flush()
        self._done.add(arxiv_id)
        self._candidates_by_paper[arxiv_id] = list(candidates)

    def caching_generator(self, generate: Callable) -> Callable:
        """Wrap a QA generator with the disk-backed cache.

        A cache hit returns without calling ``generate`` at all — a resumed
        qa run re-bills nothing it already paid for. ``None`` results
        (theorem states no answer) are cached too. Failures are not cached,
        so transient errors retry on the next run.
        """

        def cached(statement, **kwargs):
            key = _statement_key(statement)
            if key in self._qa_cache:
                return self._qa_cache[key]
            result = generate(statement, **kwargs)
            with self._qa_cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "result": result}) + "\n")
                fh.flush()
            self._qa_cache[key] = result
            return result

        return cached

    # --- internals -------------------------------------------------------------

    def _load(self) -> None:
        for row in _iter_jsonl(self._candidates_path):
            self._candidates_by_paper.setdefault(row["arxiv_id"], []).append(row["candidate"])
        for row in _iter_jsonl(self._papers_path):
            self._done.add(row["arxiv_id"])
        for row in _iter_jsonl(self._qa_cache_path):
            self._qa_cache[row["key"]] = row["result"]
        for row in _iter_jsonl(self._rate_limit_events_path):
            self._accumulate_rate_limit(row["status"], row["backoff_seconds"])

    def _accumulate_rate_limit(self, status, backoff_seconds) -> None:
        telemetry = self._rate_limit_telemetry
        telemetry["events"] += 1
        telemetry["backoff_seconds"] += float(backoff_seconds or 0.0)
        status_key = str(status)
        telemetry["statuses"][status_key] = telemetry["statuses"].get(status_key, 0) + 1


def _statement_key(statement: str) -> str:
    return hashlib.sha1(str(statement).encode("utf-8")).hexdigest()[:16]


def _cooldown_seconds() -> int:
    raw = os.environ.get("ICEPICK_ARXIV_COOLDOWN_SECONDS")
    if raw is None:
        return _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    try:
        return int(float(raw))
    except ValueError:
        return _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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

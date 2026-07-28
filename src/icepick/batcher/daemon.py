"""BatcherDaemon — orchestration layer for the bulk-batcher subsystem.

System ships DISARMED. Every state transition is disk-visible. A kill at any
moment resumes cleanly.

Entry point: `icepick batcher run` → cli_glue builds a BatcherConfig, creates
a BatcherDaemon, calls run_forever().

Signal handling:
  SIGTERM / SIGINT set a stop flag checked between ticks. In-flight stage
  subprocess calls are blocking within tick — signals take effect at the next
  tick boundary, never mid-stage. Pass@k and cascade both have their own
  checkpoint mechanisms, so a mid-flight kill is safe.

Lock discipline:
  daemon.lock is held via fcntl.flock(LOCK_EX | LOCK_NB) on an open fd kept
  for process lifetime. A second daemon attempting to acquire the same lock
  gets EWOULDBLOCK → clean refusal 'lock_held'. On graceful exit the fd is
  closed (releasing the lock) and the lock file is removed (best-effort).
  flock is the actual guard; file removal is courtesy cleanup.

Watch-journal asymmetry:
  Watch journal rows that cannot be parsed (no candidate/statement) are
  counted and noted in STATUS but do NOT abort the queue. Watch journals
  are advisory blockers (batch9 dedup); the primary journal being corrupt
  would be fatal, but a supplemental ledger-feed having malformed rows
  should not halt the whole queue. This asymmetry is intentional.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from icepick.batcher.config import BatcherConfig
from icepick.batcher.journal import (
    CursorStore,
    JournalCorruption,
    JournalTailer,
    journal_quiet_seconds,
    read_manifest_source_name,
    run_concluded,
)
from icepick.batcher.ledger import Ledger, LedgerRow
from icepick.batcher.slicer import (
    SliceConfig,
    cut_slice,
    recover_pending_slice,
)
from icepick.batcher.stages import (
    StageOutcome,
    build_stage_env,
    cascade_slot_free,
    qwen_slot_free,
    run_cascade,
    run_mount,
    run_passk,
    with_retries,
)
from icepick.batcher.state import (
    STATES_LINEAR,
    load_all_states,
    load_state,
    transition,
)
from icepick.batcher.status import render_status, write_status
from icepick.batcher.identity import compute_uid, stmt_key as make_stmt_key, content_hash


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QUEUE_STATE_FILENAME = "queue_state.json"
_LOCK_FILENAME = "daemon.lock"
_ARMED_FILENAME = "ARMED"
_EVENTS_FILENAME = "events.jsonl"
_BATCHES_DIR = "batches"

# States that are "done" for the purposes of in-flight detection
_DONE_STATES = {"READY", "FROZEN"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_runner(argv, env=None, capture_output=True, text=True, timeout=None):
    import subprocess
    return subprocess.run(argv, env=env, capture_output=capture_output, text=text, timeout=timeout)


def _default_slot_checker() -> bool:
    return qwen_slot_free()


def _default_cascade_slot_checker() -> bool:
    return cascade_slot_free()


def _read_queue_state(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_queue_state(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_event(events_path: Path, event: dict) -> None:
    """Append one event JSON line to events.jsonl (append+flush)."""
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
        fh.flush()


def _max_batch_number(batches_root: Path) -> Optional[int]:
    """Return the highest N seen in batch<N> dirs, or None if none exist."""
    if not batches_root.exists():
        return None
    max_n = None
    for d in batches_root.iterdir():
        if d.is_dir() and d.name.startswith("batch"):
            try:
                n = int(d.name[len("batch"):])
                if max_n is None or n > max_n:
                    max_n = n
            except ValueError:
                continue
    return max_n


def _find_in_flight(batch_states: dict) -> Optional[str]:
    """Return the lowest-numbered batch not in DONE_STATES, or None."""
    def _batch_n(name: str) -> int:
        try:
            return int(name[len("batch"):])
        except (ValueError, IndexError):
            return 0

    candidates = [
        name for name, st in batch_states.items()
        if st.get("state") not in _DONE_STATES
    ]
    if not candidates:
        return None
    return min(candidates, key=_batch_n)


# ---------------------------------------------------------------------------
# BatcherDaemon
# ---------------------------------------------------------------------------


class BatcherDaemon:
    """Orchestrates batch slicing → mount → cascade → pass@k lifecycle.

    All side-effecting collaborators are injectable for testing:
      ledger_cls, runner, slot_checker, sleep_fn, now_iso_fn, clock

    The real defaults wire to the actual icepick stage runners and system
    clock. Tests pass fakes.
    """

    def __init__(
        self,
        config: BatcherConfig,
        *,
        ledger_cls=None,
        runner=None,
        slot_checker=None,
        cascade_slot_checker=None,
        sleep_fn=None,
        now_iso_fn=None,
        clock=None,
    ):
        self.config = config
        self._ledger_cls = ledger_cls or Ledger
        self._runner = runner or _default_runner
        self._slot_checker = slot_checker or _default_slot_checker
        self._cascade_slot_checker = cascade_slot_checker or _default_cascade_slot_checker
        self._sleep_fn = sleep_fn or time.sleep
        self._now_iso_fn = now_iso_fn or _now_iso
        self._clock = clock or time.monotonic

        self._root = Path(config.root)
        self._batches_root = self._root / _BATCHES_DIR
        self._queue_state_path = self._root / _QUEUE_STATE_FILENAME
        self._lock_path = self._root / _LOCK_FILENAME
        self._events_path = self._root / _EVENTS_FILENAME

        self._lock_fd: Optional[int] = None
        self._stop = False
        self._ledger: Optional[Ledger] = None
        self._cursor: Optional[CursorStore] = None
        self._queue_state: Optional[dict] = None

        # Per-tick accumulators reset each startup
        self._skip_counts: dict = {"replay": 0, "stmt": 0, "warns": 0}
        self._watch_counters: dict = {}
        self._cursor_reset: bool = False
        self._watch_source_cache: dict = {}  # label → source_name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Armed check
    # ------------------------------------------------------------------

    def armed(self) -> bool:
        return (self._root / _ARMED_FILENAME).exists()

    # ------------------------------------------------------------------
    # Lock
    # ------------------------------------------------------------------

    def _acquire_lock(self) -> str:
        """Acquire daemon.lock; returns 'ok' or 'lock_held'."""
        self._root.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_path), os.O_CREAT | os.O_WRONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return "lock_held"

        self._lock_fd = fd
        # Write PID + timestamp to the lock file body (informational).
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        body = json.dumps({"pid": os.getpid(), "started_at": self._now_iso_fn()})
        os.write(fd, body.encode("utf-8"))
        return "ok"

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Queue state
    # ------------------------------------------------------------------

    def _load_or_create_queue_state(self) -> dict:
        """Load queue_state.json (creating if absent) with all checks.

        Identity-field cross-check: if queue_state exists and identity fields
        differ → raise RuntimeError('identity_mismatch').

        Numbering cross-check: max N in batches/ must be < next_batch_number
        → else raise RuntimeError('numbering_mismatch').
        """
        now = self._now_iso_fn()
        existing = _read_queue_state(self._queue_state_path)

        if existing is None:
            # First run: create queue state.
            qs = {
                "config": self.config.to_dict(),
                "next_batch_number": 10,
                "halt": None,
                "spend_usd_total": 0.0,
                "created_at": now,
                "updated_at": now,
            }
            _write_queue_state(self._queue_state_path, qs)
            return qs

        # Verify identity fields.
        persisted_cfg = existing.get("config") or {}
        mismatches = self.config.check_identity(persisted_cfg)
        if mismatches:
            raise RuntimeError(
                "identity_mismatch: persisted config identity fields differ from "
                f"current config: {'; '.join(mismatches)}"
            )

        # Log note if non-identity fields differ.
        all_current = self.config.to_dict()
        non_identity_diffs = []
        for k, v in all_current.items():
            if k in BatcherConfig.IDENTITY_FIELDS:
                continue
            if persisted_cfg.get(k) != v:
                non_identity_diffs.append(k)
        if non_identity_diffs:
            # Update persisted config with current non-identity fields.
            existing["config"] = all_current
            existing["updated_at"] = now
            _write_queue_state(self._queue_state_path, existing)

        # Numbering cross-check.
        next_n = existing.get("next_batch_number", 10)
        max_n = _max_batch_number(self._batches_root)
        if max_n is not None and max_n >= next_n:
            raise RuntimeError(
                f"numbering_mismatch: max batch number on disk (batch{max_n}) "
                f">= next_batch_number ({next_n}). Inspect batches/ directory."
            )

        return existing

    def _save_queue_state(self) -> None:
        if self._queue_state is not None:
            _write_queue_state(self._queue_state_path, self._queue_state)

    # ------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------

    def startup(self) -> dict:
        """Acquire lock, load/create queue state, load ledger, recover pending.

        Returns a startup report dict. Raises RuntimeError on lock or config
        failures.
        """
        self._root.mkdir(parents=True, exist_ok=True)

        lock_result = self._acquire_lock()
        if lock_result != "ok":
            raise RuntimeError("lock_held: another daemon instance holds the lock")

        self._queue_state = self._load_or_create_queue_state()

        # Load or create cursor store.
        cursor_path = self._root / "ledger" / "cursor.json"
        self._cursor = CursorStore(cursor_path)
        self._cursor.load()

        # Load ledger.
        ledger_dir = self._root / "ledger"
        self._ledger = self._ledger_cls.load(ledger_dir)

        # Recover any pending slice.
        def _tailer_factory(jp: Path) -> JournalTailer:
            return JournalTailer(jp, self._cursor)

        recovery = recover_pending_slice(
            self._batches_root,
            self._ledger,
            self._cursor,
            _tailer_factory,
        )

        return {
            "lock": "acquired",
            "queue_state": self._queue_state,
            "recovery": recovery,
        }

    # ------------------------------------------------------------------
    # Internal event helpers
    # ------------------------------------------------------------------

    def _emit_event(self, kind: str, detail: dict) -> None:
        now = self._now_iso_fn()
        event = {"at": now, "kind": kind, **detail}
        _append_event(self._events_path, event)

    def _set_halt(self, reason: str, now: Optional[str] = None) -> None:
        if now is None:
            now = self._now_iso_fn()
        if self._queue_state is None:
            return
        self._queue_state["halt"] = {"active": True, "reason": reason, "at": now}
        self._queue_state["updated_at"] = now
        self._save_queue_state()
        self._emit_event("halt_set", {"reason": reason})

    def _add_spend(self, cost_usd: float) -> None:
        if self._queue_state is None:
            return
        self._queue_state["spend_usd_total"] = (
            self._queue_state.get("spend_usd_total", 0.0) + cost_usd
        )
        self._queue_state["updated_at"] = self._now_iso_fn()
        self._save_queue_state()

    def _inc_batch_number(self) -> None:
        if self._queue_state is None:
            return
        self._queue_state["next_batch_number"] = (
            self._queue_state.get("next_batch_number", 10) + 1
        )
        self._queue_state["updated_at"] = self._now_iso_fn()
        self._save_queue_state()

    # ------------------------------------------------------------------
    # Watch-journal ingestion
    # ------------------------------------------------------------------

    def _ingest_watch_journals(self) -> None:
        """Ingest batch9 (and other watch journals) into the ledger only.

        Watch-journal rows block future slices by populating the ledger but
        are never auto-batched. Malformed rows (no candidate/statement) are
        counted and noted in STATUS but do NOT abort the queue.

        Asymmetry vs primary journal: the primary journal being corrupt
        would raise JournalCorruption and surface to the operator. Watch
        journals are advisory; malformed rows are silently counted here
        because watch journals may be produced by different writers whose
        format we cannot fully control.
        """
        if not self.config.watch_journals:
            return

        for wj in self.config.watch_journals:
            label = wj["label"]
            jp = Path(wj["journal_path"])
            run_dir = Path(wj["run_dir"])

            if label not in self._watch_counters:
                self._watch_counters[label] = {"ingested": 0, "malformed": 0}

            # Cache source name (read once per label).
            if label not in self._watch_source_cache:
                try:
                    source_name = read_manifest_source_name(run_dir)
                    self._watch_source_cache[label] = source_name
                except (KeyError, OSError, json.JSONDecodeError) as exc:
                    self._watch_counters[label]["malformed"] += 1
                    continue
            source_name = self._watch_source_cache[label]

            tailer = JournalTailer(jp, self._cursor)
            try:
                rows = tailer.read_new()
            except JournalCorruption:
                self._watch_counters[label]["malformed"] += 1
                continue

            for jrow in rows:
                raw_row = jrow.row
                candidate = raw_row.get("candidate")
                if not isinstance(candidate, dict):
                    self._watch_counters[label]["malformed"] += 1
                    # Advance cursor past malformed row — it is advisory.
                    self._cursor.advance(jp, jrow)
                    continue
                statement = candidate.get("statement")
                if not statement or not isinstance(statement, str):
                    self._watch_counters[label]["malformed"] += 1
                    self._cursor.advance(jp, jrow)
                    continue

                uid = compute_uid(source_name, statement)
                sk = make_stmt_key(statement)
                ch = content_hash(raw_row)

                verdict = self._ledger.check(uid, sk, ch)
                if verdict.kind in ("new", "warn"):
                    # Append to ledger (blocks future slices).
                    row = LedgerRow(
                        uid=uid,
                        stmt_key=sk,
                        content_hash=ch,
                        batch=f"watch:{label}",
                        source_journal=str(jp),
                        journal_line=jrow.line_no,
                        sliced_at=self._now_iso_fn(),
                        warn_only=False,
                    )
                    self._ledger.append_all([row])
                    self._watch_counters[label]["ingested"] += 1
                else:
                    # Already known — skip silently.
                    pass

                # Advance cursor AFTER ledger append (ledger-then-cursor invariant).
                self._cursor.advance(jp, jrow)
                self._cursor.save()

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    def _advance_batch(self, batch_name: str) -> str:
        """Advance a batch by one stage. Returns action tag."""
        batch_dir = self._batches_root / batch_name
        st = load_state(batch_dir)
        current_state = st.get("state")
        now = self._now_iso_fn()

        if current_state == "SLICED":
            return self._do_mount(batch_name, batch_dir, now)
        elif current_state == "MOUNTED":
            return self._do_cascade(batch_name, batch_dir, st, now)
        elif current_state == "CASCADE_DONE":
            return self._do_passk(batch_name, batch_dir, now)
        elif current_state == "PASSK_DONE":
            return self._do_ready(batch_name, batch_dir, now)
        else:
            return f"unknown_state:{current_state}"

    def _do_mount(self, batch_name: str, batch_dir: Path, now: str) -> str:
        # Load expected UIDs from slice manifest.
        manifest_path = batch_dir / "slice_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reason = f"mount_setup_failed: cannot read slice_manifest: {exc}"
            transition(batch_dir, "FROZEN", note=reason, now_iso=now)
            self._emit_event("frozen", {"batch": batch_name, "reason": reason})
            return "frozen"

        expected_uids = [e["uid"] for e in manifest.get("entries", [])]
        slice_records = str(batch_dir / "slice_records.jsonl")

        outcome = run_mount(
            runner=self._runner,
            batch_dir=str(batch_dir),
            slice_records=slice_records,
            campaign_source=self.config.campaign_source,
            expected_uids=expected_uids,
            mode=self.config.mode,
            key_path=self.config.key_path,
            calibration_sheet=self.config.calibration_sheet,
            icepick_bin=self.config.icepick_bin,
        )

        if outcome.ok:
            run_dir = outcome.data.get("run_dir", "")
            transition(
                batch_dir, "MOUNTED",
                note=f"mount_ok: {outcome.detail}",
                now_iso=now,
                extra={"mount_run_dir": run_dir},
            )
            return "mounted"
        else:
            reason = f"{outcome.kind}: {outcome.detail}"
            transition(batch_dir, "FROZEN", note=reason, now_iso=now)
            self._emit_event("frozen", {"batch": batch_name, "reason": reason, "kind": outcome.kind})
            return "frozen"

    def _do_cascade(self, batch_name: str, batch_dir: Path, st: dict, now: str) -> str:
        # Check cascade slot before attempting.
        try:
            cascade_free = self._cascade_slot_checker()
        except Exception:
            return "waiting_cascade"

        if not cascade_free:
            return "waiting_cascade"

        # Find handoff records path from mount run dir.
        mount_run_dir = st.get("mount_run_dir", "")
        if mount_run_dir:
            handoff_records = str(Path(mount_run_dir) / "handoff" / "records.jsonl")
        else:
            # Fallback: search intake/runs for a run dir.
            intake_runs = batch_dir / "intake" / "runs"
            if intake_runs.exists():
                runs = [d for d in intake_runs.iterdir() if d.is_dir()]
                if runs:
                    handoff_records = str(sorted(runs)[-1] / "handoff" / "records.jsonl")
                else:
                    handoff_records = ""
            else:
                handoff_records = ""

        if not handoff_records:
            reason = "cascade_setup_failed: cannot locate handoff records"
            transition(batch_dir, "FROZEN", note=reason, now_iso=now)
            self._emit_event("frozen", {"batch": batch_name, "reason": reason})
            return "frozen"

        def _run_cascade_once() -> StageOutcome:
            return run_cascade(
                runner=self._runner,
                batch_dir=str(batch_dir),
                handoff_records=handoff_records,
                key_path=self.config.key_path,
                cost_limit_usd=self.config.cost_limit_usd,
                mode=self.config.mode,
                calibration_sheet=self.config.calibration_sheet,
                icepick_bin=self.config.icepick_bin,
            )

        outcome = with_retries(_run_cascade_once, sleep_fn=self._sleep_fn)

        if outcome.ok:
            cost = outcome.data.get("cost_usd", 0.0)
            self._add_spend(cost)
            transition(
                batch_dir, "CASCADE_DONE",
                note=f"cascade_ok: {outcome.detail}",
                now_iso=now,
                extra={
                    "cascade_data": {
                        "cost_usd": cost,
                        "initial_record_count": outcome.data.get("initial_record_count"),
                        "final_corpus_count": outcome.data.get("final_corpus_count"),
                    }
                },
            )
            return "cascade_done"

        elif outcome.kind == "cost_guard_tripped":
            cost = outcome.data.get("cost_usd", 0.0)
            reason = f"cost_guard:{batch_name} ${cost:.4f}>{self.config.cost_limit_usd}"
            transition(batch_dir, "FROZEN", note=reason, now_iso=now)
            self._set_halt(reason)
            self._emit_event("frozen", {
                "batch": batch_name,
                "reason": reason,
                "cost_usd": cost,
                "limit_usd": self.config.cost_limit_usd,
            })
            return "frozen"

        else:
            reason = f"retries_exhausted: {outcome.kind}: {outcome.detail}"
            transition(batch_dir, "FROZEN", note=reason, now_iso=now)
            self._emit_event("frozen", {
                "batch": batch_name,
                "reason": reason,
                "kind": outcome.kind,
            })
            return "frozen"

    def _do_passk(self, batch_name: str, batch_dir: Path, now: str) -> str:
        # Check slot before attempting.
        try:
            slot_free = self._slot_checker()
        except Exception as exc:
            return f"waiting_qwen"

        if not slot_free:
            return "waiting_qwen"

        def _run_passk_once() -> StageOutcome:
            return run_passk(
                runner=self._runner,
                batch_dir=str(batch_dir),
                mode=self.config.mode,
                calibration_sheet=self.config.calibration_sheet,
                slot_checker=None,  # We already checked slot above; don't recheck inside
                icepick_bin=self.config.icepick_bin,
            )

        outcome = with_retries(_run_passk_once, sleep_fn=self._sleep_fn)

        if outcome.ok:
            passk_counts = outcome.data.get("counts", {})
            transition(
                batch_dir, "PASSK_DONE",
                note=f"passk_ok: {outcome.detail}",
                now_iso=now,
                extra={"passk_counts": passk_counts},
            )
            return "passk_done"

        elif outcome.kind == "qwen_slot_busy":
            return "waiting_qwen"

        else:
            reason = f"passk_interrupted: {outcome.kind}: {outcome.detail}"
            transition(batch_dir, "FROZEN", note=reason, now_iso=now)
            self._emit_event("frozen", {
                "batch": batch_name,
                "reason": reason,
                "kind": outcome.kind,
            })
            return "frozen"

    def _do_ready(self, batch_name: str, batch_dir: Path, now: str) -> str:
        transition(batch_dir, "READY", note="all stages complete", now_iso=now)
        self._emit_event("ready", {"batch": batch_name})
        return "ready"

    # ------------------------------------------------------------------
    # Status writer
    # ------------------------------------------------------------------

    def _write_status(
        self,
        batch_states: dict,
        *,
        in_flight: Optional[str] = None,
        held: Optional[dict] = None,
    ) -> None:
        if self._queue_state is None:
            return

        # Collect last-10 events from events.jsonl.
        events_log: list[str] = []
        if self._events_path.exists():
            try:
                lines = self._events_path.read_text(encoding="utf-8").splitlines()
                events_log = lines[-10:]
            except OSError:
                pass

        extras = {
            "armed": self.armed(),
            "pid": os.getpid(),
            "in_flight": in_flight,
            "held_remainder": held,
            "skip_counts": dict(self._skip_counts),
            "watch_counters": dict(self._watch_counters),
            "events_log": events_log,
            "cursor_reset": self._cursor_reset,
        }

        content = render_status(self._queue_state, batch_states, extras)
        write_status(self._root, content)

    # ------------------------------------------------------------------
    # tick
    # ------------------------------------------------------------------

    def tick(self) -> str:
        """Single loop body. Returns action tag. Unit-testable.

        Order:
          a) Not armed → 'disarmed'
          b) Watch-journal ingestion
          c) Halt active → 'halted'
          d) In-flight batch → advance one stage → return its tag
          e) Frozen-only (no in-flight) → slice-cutting may continue
          f) No in-flight → try slice
          g) Rewrite STATUS.md after any change
        """
        # (a) Armed check — first thing every tick.
        if not self.armed():
            return "disarmed"

        assert self._ledger is not None
        assert self._cursor is not None
        assert self._queue_state is not None

        now = self._now_iso_fn()
        changed = False

        # (b) Watch-journal ingestion.
        self._ingest_watch_journals()

        # (c) Halt check.
        halt = self._queue_state.get("halt") or {}
        if halt.get("active"):
            batch_states = load_all_states(self._batches_root)
            self._write_status(batch_states)
            return "halted"

        batch_states = load_all_states(self._batches_root)

        # (d) In-flight batch.
        in_flight = _find_in_flight(batch_states)
        if in_flight and batch_states[in_flight].get("state") != "FROZEN":
            tag = self._advance_batch(in_flight)
            # Reload states after mutation.
            batch_states = load_all_states(self._batches_root)
            self._write_status(batch_states, in_flight=in_flight)
            return tag

        # (e) Only frozen batches in flight: slicing may continue but nothing
        # advances past FROZEN. Processing is serialised: we only cut new slices;
        # no stage dispatch until the frozen batch is cleared.
        any_frozen = any(
            st.get("state") == "FROZEN" for st in batch_states.values()
        )

        # (f) No processable in-flight → try slice.
        slice_cfg = SliceConfig(
            campaign_source=self.config.campaign_source,
            slice_size=self.config.slice_size,
            cross_source_statement_policy=self.config.cross_source_statement_policy,
        )

        journal_path = Path(self.config.journal_path)
        tailer = JournalTailer(journal_path, self._cursor)
        next_batch_no = self._queue_state.get("next_batch_number", 10)

        try:
            outcome = cut_slice(
                tailer=tailer,
                cursor=self._cursor,
                ledger=self._ledger,
                batches_root=self._batches_root,
                batch_no=next_batch_no,
                config=slice_cfg,
                now_iso=now,
            )
        except JournalCorruption as exc:
            reason = f"slice_abort:journal_corruption:{exc}"
            self._set_halt(reason)
            self._emit_event("halt_set", {"reason": reason, "kind": "journal_corruption"})
            batch_states = load_all_states(self._batches_root)
            self._write_status(batch_states)
            return "slice_aborted"

        counts = outcome.counts or {}
        # Accumulate skip tallies.
        self._skip_counts["replay"] += counts.get("replay_skips", 0)
        self._skip_counts["stmt"] += counts.get("stmt_skips", 0)
        self._skip_counts["warns"] += counts.get("warns", 0)

        if outcome.kind == "sliced":
            self._inc_batch_number()
            # Initialize state machine for new batch.
            batch_dir = self._batches_root / f"batch{next_batch_no}"
            transition(batch_dir, "SLICED", note="slice committed", now_iso=now)
            batch_states = load_all_states(self._batches_root)
            self._write_status(batch_states)
            return "sliced"

        elif outcome.kind == "insufficient":
            # Check whether run has concluded with a remainder.
            pending_count = counts.get("pending_size", 0)
            if run_concluded(Path(self.config.run_dir)) and pending_count > 0:
                # Remainder HELD: listed in STATUS, never auto-batched.
                held = {"count": pending_count, "uids": []}
                self._emit_event("held_remainder", {
                    "count": pending_count,
                    "reason": "run concluded with remainder below slice_size",
                })
                batch_states = load_all_states(self._batches_root)
                self._write_status(batch_states, held=held)
                return "held_remainder"
            else:
                batch_states = load_all_states(self._batches_root)
                self._write_status(batch_states)
                return "waiting_journal"

        elif outcome.kind == "aborted":
            # slice_abort: set queue halt (no batch dir created on abort).
            abort_info = outcome.abort_info or {}
            reason = f"slice_abort:{outcome.detail}"
            self._set_halt(reason)
            self._emit_event("slice_aborted", {
                "reason": reason,
                "abort_info": abort_info,
                "detail": outcome.detail,
            })
            batch_states = load_all_states(self._batches_root)
            self._write_status(batch_states)
            return "slice_aborted"

        # Should not reach here, but cover defensively.
        return "waiting_journal"

    # ------------------------------------------------------------------
    # run_forever
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """Main daemon loop. Runs until DISARMED or signalled.

        Signal handling:
          SIGTERM and SIGINT set a stop flag checked at the top of each
          iteration. Signals take effect between ticks — never mid-stage.
          In-flight stage subprocess calls are blocking; killing the
          daemon mid-stage leaves the stage in its own checkpoint state,
          which is safe to resume on restart.
        """
        def _handle_signal(signum, frame):
            self._stop = True

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        try:
            while True:
                if self._stop:
                    break

                tag = self.tick()

                if tag == "disarmed":
                    # ARMED flag is absent — graceful exit.
                    break

                # Determine sleep duration based on tag.
                if tag in ("waiting_qwen", "waiting_cascade"):
                    sleep_s = self.config.qwen_recheck_interval_s
                else:
                    sleep_s = self.config.poll_interval_s

                self._sleep_fn(sleep_s)

        finally:
            if self._stop:
                # Write a stopped-by-signal note to STATUS.
                try:
                    batch_states = load_all_states(self._batches_root)
                    qs = self._queue_state or {}
                    extras = {
                        "armed": self.armed(),
                        "pid": os.getpid(),
                        "events_log": ["daemon stopped by signal"],
                        "skip_counts": dict(self._skip_counts),
                        "watch_counters": dict(self._watch_counters),
                        "cursor_reset": self._cursor_reset,
                    }
                    content = render_status(qs, batch_states, extras)
                    write_status(self._root, content)
                except Exception:
                    pass

            self._release_lock()

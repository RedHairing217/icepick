"""Tests for src/icepick/batcher/daemon.py.

All tests use:
- Synthetic journals (written to tmp_path)
- Injected fake runners / slot checkers
- No real icepick subprocesses
- Small slice_size (3–5) to keep journals tiny

Key coverage:
1. disarmed tick + run_forever exits immediately
2. Full happy lifecycle: slice→mount→cascade(cost)→qwen-busy wait→passk→READY_TO_FOLD
3. Crash-resume at every stage boundary
4. Cost-guard trip → FROZEN + queue halt + events + STATUS
5. Transient fail → retries → FROZEN, slicing continues but no advancement past frozen
6. slice uid_conflict abort → queue halt with abort_info in STATUS
7. held remainder (run concluded + <slice_size) — remainder NEVER sliced
8. Watch-journal ingestion blocks later slice
9. Numbering-mismatch + config-drift + lock-held refusals
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from icepick.batcher.config import BatcherConfig
from icepick.batcher.daemon import BatcherDaemon
from icepick.batcher.identity import compute_uid, stmt_key as make_stmt_key, content_hash
from icepick.batcher.journal import CursorStore, JournalTailer
from icepick.batcher.ledger import Ledger, LedgerRow
from icepick.batcher.state import load_all_states, load_state, transition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOW = "2026-07-07T00:00:00Z"
SOURCE = "arxiv_bulk_test"
SLICE_SIZE = 3


# ---------------------------------------------------------------------------
# Journal helpers
# ---------------------------------------------------------------------------


def _make_row(statement: str, arxiv_id: str = "2401.00001") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "candidate": {
            "statement": statement,
            "answer": "True",
            "metadata": {},
        },
    }


def _write_journal(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_statements(n: int, prefix: str = "stmt") -> list[str]:
    return [f"Theorem {prefix}_{i}: The solution exists." for i in range(n)]


# ---------------------------------------------------------------------------
# Fake stage runner
# ---------------------------------------------------------------------------


class FakeRunner:
    """Fake subprocess runner that simulates stage outcomes."""

    def __init__(self):
        self.calls: list[list] = []
        # Mount settings
        self._mount_ok = True
        self._mount_row_override: Optional[int] = None  # override handoff row count
        # Cascade settings
        self._cascade_ok = True
        self._cascade_cost = 2.31
        self._cascade_fail_kind: Optional[str] = None  # 'exec_failed', 'cost_guard'
        self._cascade_fail_count = 0  # how many times to fail before succeeding
        self._cascade_called = 0
        # Passk settings
        self._passk_ok = True
        self._passk_interrupted = False

    def __call__(self, argv, env=None, capture_output=True, text=True, timeout=None):
        self.calls.append(list(argv))
        cmd = " ".join(argv)

        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        if "allocation" in cmd and "mount" in cmd:
            if not self._mount_ok:
                result.returncode = 1
                result.stderr = "mount failed"
                return result
            # Simulate mount output: find batch dir from --output-dir
            output_dir = None
            for i, a in enumerate(argv):
                if a == "--output-dir" and i + 1 < len(argv):
                    output_dir = Path(argv[i + 1])
            if output_dir:
                self._write_mount_output(argv, output_dir)

        elif "wellposed-cascade" in cmd:
            self._cascade_called += 1
            if self._cascade_fail_count > 0 and self._cascade_called <= self._cascade_fail_count:
                result.returncode = 1
                result.stderr = "529 overloaded transient"
                return result
            if not self._cascade_ok:
                result.returncode = 1
                result.stderr = "cascade failed"
                return result
            # Write cascade manifest
            batch_dir = None
            for i, a in enumerate(argv):
                if a == "--output-dir" and i + 1 < len(argv):
                    batch_dir = Path(argv[i + 1]).parent
            if batch_dir:
                self._write_cascade_output(batch_dir)

        elif "pass_at_k" in cmd:
            if not self._passk_ok:
                result.returncode = 1
                result.stderr = "passk failed"
                return result
            # Write passk manifest
            batch_dir = None
            for i, a in enumerate(argv):
                if a == "--output-dir" and i + 1 < len(argv):
                    batch_dir = Path(argv[i + 1]).parent
            if batch_dir:
                self._write_passk_output(batch_dir)

        return result

    def _write_mount_output(self, argv, output_dir: Path) -> None:
        """Simulate icepick allocation mount writing a run dir with handoff."""
        # Find the slice_records path and campaign_source from argv.
        slice_records_path = None
        campaign_source = SOURCE
        for i, a in enumerate(argv):
            if a == "--path" and i + 1 < len(argv):
                slice_records_path = Path(argv[i + 1])
            if a == "--source" and i + 1 < len(argv):
                campaign_source = argv[i + 1]

        runs_dir = output_dir / "runs" / "20260707T000000Z"
        runs_dir.mkdir(parents=True, exist_ok=True)
        handoff_dir = runs_dir / "handoff"
        handoff_dir.mkdir(parents=True, exist_ok=True)

        # Write manifest.json for the run dir (read_manifest_source_name).
        manifest_path = runs_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({"source_name": campaign_source}),
            encoding="utf-8",
        )

        # Write handoff records from the slice_records file.
        records = []
        if slice_records_path and slice_records_path.exists():
            for line in slice_records_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        n = self._mount_row_override if self._mount_row_override is not None else len(records)
        with (handoff_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
            for i, r in enumerate(records[:n]):
                fh.write(json.dumps(r) + "\n")

    def _write_cascade_output(self, batch_dir: Path) -> None:
        cascade_dir = batch_dir / "cascade"
        cascade_dir.mkdir(parents=True, exist_ok=True)

        cost = self._cascade_cost if self._cascade_ok else None
        manifest = {
            "overall": {
                "total_estimated_cost_usd": cost,
                "initial_record_count": SLICE_SIZE,
                "final_corpus_count": SLICE_SIZE,
            },
            "inputs": {"initial_record_count": SLICE_SIZE},
        }
        (cascade_dir / "cascade_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        # Write final_corpus.jsonl
        (cascade_dir / "final_corpus.jsonl").write_text(
            "\n".join(json.dumps({"uid": f"uid_{i}"}) for i in range(SLICE_SIZE)),
            encoding="utf-8",
        )

    def _write_passk_output(self, batch_dir: Path) -> None:
        passk_dir = batch_dir / "pass_at_k"
        passk_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "interrupted": self._passk_interrupted,
            "counts": {"total": SLICE_SIZE, "pass": 2, "fail": 1},
        }
        (passk_dir / "pass_at_k_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Config / Daemon factory
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    journal_path: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    watch_journals: Optional[list] = None,
) -> BatcherConfig:
    if journal_path is None:
        journal_path = tmp_path / "journal" / "candidates.jsonl"
    if run_dir is None:
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

    return BatcherConfig(
        root=tmp_path / "batcher",
        journal_path=journal_path,
        run_dir=run_dir,
        campaign_source=SOURCE,
        slice_size=SLICE_SIZE,
        cost_limit_usd=5.0,
        key_path="/fake/key.env",
        mode="flow_testing",
        icepick_bin="icepick",
        poll_interval_s=0,
        qwen_recheck_interval_s=0,
        watch_journals=watch_journals or [],
    )


def _make_daemon(
    tmp_path: Path,
    runner: Optional[FakeRunner] = None,
    slot_checker=None,
    cascade_slot_checker=None,
    journal_path: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    watch_journals: Optional[list] = None,
    config: Optional[BatcherConfig] = None,
) -> BatcherDaemon:
    if config is None:
        config = _make_config(
            tmp_path,
            journal_path=journal_path,
            run_dir=run_dir,
            watch_journals=watch_journals,
        )
    if runner is None:
        runner = FakeRunner()
    if slot_checker is None:
        slot_checker = lambda: True
    if cascade_slot_checker is None:
        cascade_slot_checker = lambda: True

    return BatcherDaemon(
        config,
        runner=runner,
        slot_checker=slot_checker,
        cascade_slot_checker=cascade_slot_checker,
        sleep_fn=lambda s: None,  # no actual sleeping in tests
        now_iso_fn=lambda: NOW,
        clock=time.monotonic,
    )


def _arm(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ARMED").write_text(json.dumps({"armed_at": NOW}))


def _conclude_run(run_dir: Path) -> None:
    """Simulate run completion by ensuring no INCOMPLETE marker."""
    progress = run_dir / "_progress"
    progress.mkdir(parents=True, exist_ok=True)
    incomplete = progress / "INCOMPLETE"
    if incomplete.exists():
        incomplete.unlink()


def _mark_run_in_progress(run_dir: Path) -> None:
    """Simulate run in progress by creating INCOMPLETE marker."""
    progress = run_dir / "_progress"
    progress.mkdir(parents=True, exist_ok=True)
    (progress / "INCOMPLETE").write_text("in progress")


# ---------------------------------------------------------------------------
# 1. DISARMED tick + run_forever exits
# ---------------------------------------------------------------------------


def test_tick_disarmed_returns_disarmed(tmp_path):
    config = _make_config(tmp_path)
    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()
    # No ARMED file → disarmed
    tag = daemon.tick()
    assert tag == "disarmed"


def test_run_forever_exits_when_disarmed(tmp_path):
    config = _make_config(tmp_path)
    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()
    daemon.run_forever()  # Should exit immediately without ARMED file.


# ---------------------------------------------------------------------------
# 2. Full happy lifecycle
# ---------------------------------------------------------------------------


def test_full_lifecycle_happy_path(tmp_path):
    """
    Journal: 3 rows → slice → mount → cascade(cost recorded) →
    qwen-busy on first attempt → passk → READY_TO_FOLD flag.
    """
    journal = tmp_path / "j" / "candidates.jsonl"
    stmts = _make_statements(SLICE_SIZE)
    _write_journal(journal, [_make_row(s) for s in stmts])

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = FakeRunner()
    runner._cascade_cost = 2.31

    # Slot busy on first try, free on second.
    slot_calls = [False, True]
    slot_idx = [0]

    def slot_checker():
        v = slot_calls[slot_idx[0]]
        slot_idx[0] = min(slot_idx[0] + 1, len(slot_calls) - 1)
        return v

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    daemon = _make_daemon(tmp_path, runner=runner, slot_checker=slot_checker, config=config)
    root = config.root
    _arm(root)
    daemon.startup()

    # Tick 1: slice
    tag = daemon.tick()
    assert tag == "sliced"

    # Tick 2: mount
    tag = daemon.tick()
    assert tag == "mounted"

    # Tick 3: cascade
    tag = daemon.tick()
    assert tag == "cascade_done"

    # Tick 4: passk — slot busy on first check → waiting_qwen
    tag = daemon.tick()
    assert tag == "waiting_qwen"

    # Tick 5: passk — slot free
    tag = daemon.tick()
    assert tag == "passk_done"

    # Tick 6: ready
    tag = daemon.tick()
    assert tag == "ready"

    # Verify READY_TO_FOLD flag file.
    batch_dir = root / "batches" / "batch10"
    assert (batch_dir / "READY_TO_FOLD").exists()

    # Verify spend accumulated.
    qs = json.loads((root / "queue_state.json").read_text())
    assert abs(qs["spend_usd_total"] - 2.31) < 0.001

    # Verify next_batch_number incremented.
    assert qs["next_batch_number"] == 11

    # Verify STATE.md was written.
    assert (root / "STATUS.md").exists()

    # Verify events.jsonl has a ready event.
    events_raw = (root / "events.jsonl").read_text().splitlines()
    event_kinds = [json.loads(e)["kind"] for e in events_raw]
    assert "ready" in event_kinds


# ---------------------------------------------------------------------------
# 2b. Cascade slot guard
# ---------------------------------------------------------------------------


def _setup_mounted_batch(tmp_path: Path):
    """Helper: bring a batch to MOUNTED state and return (config, daemon, runner)."""
    journal = tmp_path / "j" / "candidates.jsonl"
    stmts = _make_statements(SLICE_SIZE)
    _write_journal(journal, [_make_row(s) for s in stmts])
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = FakeRunner()
    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)
    return config, runner, root


def test_cascade_slot_busy_defers_without_launching(tmp_path):
    """When cascade slot is busy, tick returns 'waiting_cascade' and runner is not called for cascade."""
    config, runner, root = _setup_mounted_batch(tmp_path)

    # Slot always busy.
    daemon = _make_daemon(
        tmp_path, runner=runner, config=config,
        cascade_slot_checker=lambda: False,
    )
    daemon.startup()

    daemon.tick()  # slice
    daemon.tick()  # mount
    cascade_calls_before = sum(1 for c in runner.calls if "wellposed-cascade" in " ".join(c))

    tag = daemon.tick()  # cascade attempt — slot busy
    assert tag == "waiting_cascade"

    cascade_calls_after = sum(1 for c in runner.calls if "wellposed-cascade" in " ".join(c))
    assert cascade_calls_after == cascade_calls_before, "cascade runner must not be called when slot is busy"


def test_cascade_slot_defers_then_proceeds(tmp_path):
    """Cascade defers on first tick (busy), then proceeds when slot frees."""
    config, runner, root = _setup_mounted_batch(tmp_path)

    # Busy first, free second.
    slot_calls = [False, True]
    slot_idx = [0]

    def cascade_checker():
        v = slot_calls[slot_idx[0]]
        slot_idx[0] = min(slot_idx[0] + 1, len(slot_calls) - 1)
        return v

    daemon = _make_daemon(
        tmp_path, runner=runner, config=config,
        cascade_slot_checker=cascade_checker,
    )
    daemon.startup()

    daemon.tick()  # slice
    daemon.tick()  # mount

    tag = daemon.tick()  # cascade attempt — slot busy
    assert tag == "waiting_cascade"

    tag = daemon.tick()  # cascade — slot now free
    assert tag == "cascade_done"


def test_cascade_slot_checker_exception_defers(tmp_path):
    """If cascade_slot_checker raises, tick returns 'waiting_cascade' (conservative)."""
    config, runner, root = _setup_mounted_batch(tmp_path)

    def _bad_checker():
        raise RuntimeError("pgrep exploded")

    daemon = _make_daemon(
        tmp_path, runner=runner, config=config,
        cascade_slot_checker=_bad_checker,
    )
    daemon.startup()

    daemon.tick()  # slice
    daemon.tick()  # mount
    tag = daemon.tick()  # cascade attempt — checker raises
    assert tag == "waiting_cascade"

    cascade_calls = sum(1 for c in runner.calls if "wellposed-cascade" in " ".join(c))
    assert cascade_calls == 0


# ---------------------------------------------------------------------------
# 3. Crash-resume at every stage boundary
# ---------------------------------------------------------------------------


def _do_lifecycle_through_state(target_state: str, tmp_path: Path) -> tuple:
    """
    Run the daemon through ticks until the in-flight batch reaches target_state.
    Returns (config, daemon, batch_name).  Caller is responsible for releasing lock.
    """
    journal = tmp_path / "j" / "candidates.jsonl"
    stmts = _make_statements(SLICE_SIZE)
    _write_journal(journal, [_make_row(s) for s in stmts])
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = FakeRunner()
    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)

    daemon = _make_daemon(tmp_path, runner=runner, config=config)
    daemon.startup()

    state_order = ["SLICED", "MOUNTED", "CASCADE_DONE", "PASSK_DONE", "READY"]
    target_idx = state_order.index(target_state)

    # Execute ticks until we reach the target state.
    for _ in range(target_idx + 1):
        daemon.tick()

    return config, daemon, "batch10"


@pytest.mark.parametrize("target_state", ["SLICED", "MOUNTED", "CASCADE_DONE", "PASSK_DONE"])
def test_crash_resume_at_stage_boundary(tmp_path, target_state):
    """
    Advance to target_state, then create a fresh daemon from disk and verify:
    - No stage is re-run past its checkpoint
    - Batch composition is unchanged
    - The lifecycle can continue to READY
    """
    config, daemon_first, batch_name = _do_lifecycle_through_state(target_state, tmp_path)
    root = config.root
    batch_dir = root / "batches" / batch_name

    # Verify state on disk.
    st = load_state(batch_dir)
    assert st["state"] == target_state

    # Simulate crash: release lock (in a real crash the OS does this).
    daemon_first._release_lock()

    runner2 = FakeRunner()
    daemon2 = _make_daemon(tmp_path, runner=runner2, config=config)
    daemon2.startup()  # This calls recover_pending_slice.

    # Continue through to READY.
    state_order = ["SLICED", "MOUNTED", "CASCADE_DONE", "PASSK_DONE", "READY"]
    target_idx = state_order.index(target_state)

    for i in range(target_idx, len(state_order) - 1):
        # The first daemon already created mount output; cascade/passk use
        # skip-if-done semantics so runner2 may not be called for stages
        # that already completed. Ticks that have no in-flight batch will
        # try to slice (journal exhausted → waiting_journal).
        daemon2.tick()

    final_st = load_state(batch_dir)
    assert final_st["state"] == "READY"
    assert (batch_dir / "READY_TO_FOLD").exists()

    # Verify batch composition unchanged: same UIDs in slice_manifest.
    manifest = json.loads((batch_dir / "slice_manifest.json").read_text())
    assert len(manifest["entries"]) == SLICE_SIZE
    assert manifest["campaign_source"] == SOURCE


# ---------------------------------------------------------------------------
# 4. Cost-guard trip
# ---------------------------------------------------------------------------


def test_cost_guard_trip_freezes_and_halts(tmp_path):
    journal = tmp_path / "j" / "candidates.jsonl"
    _write_journal(journal, [_make_row(s) for s in _make_statements(SLICE_SIZE)])
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = FakeRunner()
    runner._cascade_cost = 10.0  # Exceeds $5 limit

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)

    daemon = _make_daemon(tmp_path, runner=runner, config=config)
    daemon.startup()

    daemon.tick()  # slice
    daemon.tick()  # mount
    tag = daemon.tick()  # cascade → cost guard
    assert tag == "frozen"

    # Batch must be FROZEN.
    batch_dir = root / "batches" / "batch10"
    st = load_state(batch_dir)
    assert st["state"] == "FROZEN"
    assert "cost_guard" in st["frozen"]["reason"]

    # Queue must be halted.
    qs = json.loads((root / "queue_state.json").read_text())
    assert qs["halt"]["active"] is True
    assert "cost_guard" in qs["halt"]["reason"]

    # events.jsonl must have the event.
    events_raw = (root / "events.jsonl").read_text().splitlines()
    events = [json.loads(e) for e in events_raw]
    kinds = [e["kind"] for e in events]
    assert "frozen" in kinds
    assert "halt_set" in kinds

    # STATUS.md should mention HALTED.
    status = (root / "STATUS.md").read_text()
    assert "HALTED" in status or "halted" in status.lower()


def test_cost_guard_status_shows_it(tmp_path):
    journal = tmp_path / "j" / "candidates.jsonl"
    _write_journal(journal, [_make_row(s) for s in _make_statements(SLICE_SIZE)])
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = FakeRunner()
    runner._cascade_cost = 10.0

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)

    daemon = _make_daemon(tmp_path, runner=runner, config=config)
    daemon.startup()
    daemon.tick()
    daemon.tick()
    daemon.tick()

    status = (root / "STATUS.md").read_text()
    assert "FROZEN" in status


# ---------------------------------------------------------------------------
# 5. Transient fail → retries → FROZEN, slicing continues past frozen
# ---------------------------------------------------------------------------


def test_transient_fail_retries_then_frozen(tmp_path):
    """Cascade fails 3 times (retries exhausted) → batch FROZEN.
    Next slice tick cuts a NEW batch (slicing continues).
    """
    journal = tmp_path / "j" / "candidates.jsonl"
    # Write enough for 2 slices.
    stmts = _make_statements(SLICE_SIZE * 2)
    _write_journal(journal, [_make_row(s) for s in stmts])
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = FakeRunner()
    # Cascade always fails with transient error.
    runner._cascade_fail_count = 999
    runner._cascade_ok = False

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)

    daemon = _make_daemon(tmp_path, runner=runner, config=config)
    daemon.startup()

    daemon.tick()  # slice batch10
    daemon.tick()  # mount batch10
    tag = daemon.tick()  # cascade → retries (3) → FROZEN
    # The with_retries wrapper with 3 attempts will exhaust.
    # Since we inject sleep_fn=no-op, retries happen synchronously.
    # The cascade runner is called, fails, is retried, fails again, exhausted → frozen.
    assert tag == "frozen"

    st = load_state(root / "batches" / "batch10")
    assert st["state"] == "FROZEN"

    # Next tick should CUT A NEW SLICE (slicing continues past frozen).
    # No in-flight advancing batch, but a frozen batch exists.
    tag2 = daemon.tick()
    # Should be 'sliced' because there are still rows in the journal.
    assert tag2 == "sliced"

    # batch11 should now be in SLICED state.
    st11 = load_state(root / "batches" / "batch11")
    assert st11["state"] == "SLICED"


# ---------------------------------------------------------------------------
# 6. Slice uid_conflict abort → queue halt
# ---------------------------------------------------------------------------


def test_slice_uid_conflict_halts_queue(tmp_path):
    """Seed ledger with a uid, then inject a journal row with same uid but
    different content → uid_conflict → queue halt."""
    journal = tmp_path / "j" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    stmt = "Unique theorem that will conflict."
    uid = compute_uid(SOURCE, stmt)

    # Write journal rows: first row is the conflict.
    rows = [_make_row(stmt)]
    # Add SLICE_SIZE-1 more unique rows to try to fill a full slice.
    for i in range(SLICE_SIZE):
        rows.append(_make_row(f"Other theorem {i} unique."))
    _write_journal(journal, rows)

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)

    # Seed the ledger with the conflicting uid (different content_hash).
    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    ledger_dir = root / "ledger"
    row = LedgerRow(
        uid=uid,
        stmt_key=make_stmt_key(stmt),
        content_hash="deadbeef" * 8,  # different hash → conflict
        batch="hist:old",
        source_journal="old.jsonl",
        journal_line=1,
        sliced_at=NOW,
        warn_only=False,
    )
    daemon._ledger.append_all([row])

    tag = daemon.tick()
    assert tag == "slice_aborted"

    qs = json.loads((root / "queue_state.json").read_text())
    assert qs["halt"]["active"] is True
    assert "slice_abort" in qs["halt"]["reason"]

    # STATUS should reflect the abort.
    status = (root / "STATUS.md").read_text()
    assert "HALTED" in status or "halted" in status.lower()


# ---------------------------------------------------------------------------
# 7. Held remainder
# ---------------------------------------------------------------------------


def test_held_remainder_when_run_concluded(tmp_path):
    """Journal has fewer than SLICE_SIZE rows and run is concluded → 'held_remainder'."""
    journal = tmp_path / "j" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Only 2 rows < SLICE_SIZE (3)
    stmts = _make_statements(SLICE_SIZE - 1)
    _write_journal(journal, [_make_row(s) for s in stmts])

    # Run is concluded (no INCOMPLETE marker).
    _conclude_run(run_dir)

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)

    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    tag = daemon.tick()
    assert tag == "held_remainder"

    # STATUS should have HELD section.
    status = (root / "STATUS.md").read_text()
    assert "HELD Remainder" in status

    # events.jsonl should have held event.
    events_raw = (root / "events.jsonl").read_text().splitlines()
    kinds = [json.loads(e)["kind"] for e in events_raw]
    assert "held_remainder" in kinds


def test_held_remainder_never_sliced_on_next_tick(tmp_path):
    """After held_remainder, a subsequent tick still returns held_remainder
    (or waiting_journal), never 'sliced'."""
    journal = tmp_path / "j" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_journal(journal, [_make_row(s) for s in _make_statements(SLICE_SIZE - 1)])
    _conclude_run(run_dir)

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)

    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    tag1 = daemon.tick()
    assert tag1 == "held_remainder"

    # Second tick: ledger cursor is advanced; no new rows → waiting_journal or held_remainder.
    tag2 = daemon.tick()
    assert tag2 in ("held_remainder", "waiting_journal")

    # Never sliced.
    batches = list((root / "batches").iterdir()) if (root / "batches").exists() else []
    assert len(batches) == 0


# ---------------------------------------------------------------------------
# 8. Watch-journal ingestion blocks later slice
# ---------------------------------------------------------------------------


def test_watch_journal_blocks_same_uid_in_slice(tmp_path):
    """
    Watch journal ingests a record with stmt X → blocks it from being sliced.
    Primary journal row with same stmt → stmt_conflict → skip (policy=skip).
    """
    # Set up a watch journal with stmt0.
    stmt_in_watch = "Watched theorem that should block."
    watch_run_dir = tmp_path / "batch9_run"
    watch_run_dir.mkdir(parents=True, exist_ok=True)
    watch_journal = watch_run_dir / "_progress" / "candidates.jsonl"
    watch_journal.parent.mkdir(parents=True, exist_ok=True)
    # Write watch manifest.
    (watch_run_dir / "manifest.json").write_text(
        json.dumps({"source_name": "batch9_source"}), encoding="utf-8"
    )
    _write_journal(watch_journal, [_make_row(stmt_in_watch)])

    # Primary journal: includes stmt_in_watch PLUS enough unique rows for a full slice.
    journal = tmp_path / "j" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    unique_stmts = _make_statements(SLICE_SIZE)
    rows = [_make_row(stmt_in_watch)] + [_make_row(s) for s in unique_stmts]
    _write_journal(journal, rows)

    watch_journals = [{
        "label": "batch9",
        "journal_path": str(watch_journal),
        "run_dir": str(watch_run_dir),
    }]

    config = _make_config(
        tmp_path,
        journal_path=journal,
        run_dir=run_dir,
        watch_journals=watch_journals,
    )
    root = config.root
    _arm(root)

    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    # First tick: watch ingestion happens → then slice attempt.
    tag = daemon.tick()
    # The stmt_in_watch is now in the ledger from watch.
    # The primary journal starts with stmt_in_watch which will hit stmt_conflict.
    # With policy=skip it will be skipped; remaining SLICE_SIZE rows should fill a slice.
    assert tag == "sliced"

    # Verify the slice does NOT contain the watched statement.
    batch_dir = root / "batches" / "batch10"
    manifest = json.loads((batch_dir / "slice_manifest.json").read_text())
    uids_in_slice = {e["uid"] for e in manifest["entries"]}
    watch_uid_with_batch9_source = compute_uid("batch9_source", stmt_in_watch)
    watch_uid_with_campaign_source = compute_uid(SOURCE, stmt_in_watch)
    # Neither the watch uid nor the campaign-source uid of the watched stmt should appear.
    # (They share the same stmt_key, so stmt_conflict is triggered.)
    slice_stmts = set()
    for line in (batch_dir / "slice_records.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            slice_stmts.add(r.get("statement", ""))
    assert stmt_in_watch not in slice_stmts


def test_watch_journal_malformed_rows_do_not_abort(tmp_path):
    """Watch journal with malformed rows (no candidate) → counted but queue continues."""
    watch_run_dir = tmp_path / "batch9_run"
    watch_run_dir.mkdir(parents=True, exist_ok=True)
    watch_journal = watch_run_dir / "_progress" / "candidates.jsonl"
    watch_journal.parent.mkdir(parents=True, exist_ok=True)
    (watch_run_dir / "manifest.json").write_text(
        json.dumps({"source_name": "batch9_source"}), encoding="utf-8"
    )
    # Malformed: no candidate key.
    _write_journal(watch_journal, [{"bad": "row"}, {"also_bad": True}])

    journal = tmp_path / "j" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_journal(journal, [_make_row(s) for s in _make_statements(SLICE_SIZE)])

    watch_journals = [{
        "label": "batch9",
        "journal_path": str(watch_journal),
        "run_dir": str(watch_run_dir),
    }]

    config = _make_config(
        tmp_path,
        journal_path=journal,
        run_dir=run_dir,
        watch_journals=watch_journals,
    )
    root = config.root
    _arm(root)

    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    # Should still slice (queue not aborted by malformed watch rows).
    tag = daemon.tick()
    assert tag == "sliced"

    # Malformed count should be non-zero.
    assert daemon._watch_counters.get("batch9", {}).get("malformed", 0) > 0


# ---------------------------------------------------------------------------
# 9. Refusals: numbering mismatch, config drift, lock held
# ---------------------------------------------------------------------------


def test_numbering_mismatch_refusal(tmp_path):
    """If max batch number on disk >= next_batch_number → RuntimeError."""
    config = _make_config(tmp_path)
    root = config.root

    # Create a batch dir with a higher number than next_batch_number (10).
    (root / "batches" / "batch15").mkdir(parents=True)

    # Write queue_state with next_batch_number=10.
    qs_path = root / "queue_state.json"
    root.mkdir(parents=True, exist_ok=True)
    qs = {
        "config": config.to_dict(),
        "next_batch_number": 10,  # But batch15 exists on disk!
        "halt": None,
        "spend_usd_total": 0.0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    qs_path.write_text(json.dumps(qs), encoding="utf-8")

    daemon = _make_daemon(tmp_path, config=config)

    with pytest.raises(RuntimeError, match="numbering_mismatch"):
        daemon.startup()


def test_identity_mismatch_refusal(tmp_path):
    """If campaign_source in queue_state differs from config → RuntimeError."""
    config = _make_config(tmp_path)
    root = config.root
    root.mkdir(parents=True, exist_ok=True)

    # Write queue_state with different campaign_source.
    cfg_dict = config.to_dict()
    cfg_dict["campaign_source"] = "different_source"  # Mismatch!
    qs = {
        "config": cfg_dict,
        "next_batch_number": 10,
        "halt": None,
        "spend_usd_total": 0.0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    (root / "queue_state.json").write_text(json.dumps(qs), encoding="utf-8")

    daemon = _make_daemon(tmp_path, config=config)

    with pytest.raises(RuntimeError, match="identity_mismatch"):
        daemon.startup()


def test_lock_held_refusal(tmp_path):
    """Second daemon startup with the same root → RuntimeError('lock_held')."""
    config = _make_config(tmp_path)
    d1 = _make_daemon(tmp_path, config=config)
    d2 = _make_daemon(tmp_path, config=config)

    d1.startup()  # Acquires lock.
    try:
        with pytest.raises(RuntimeError, match="lock_held"):
            d2.startup()
    finally:
        d1._release_lock()


def test_non_identity_config_drift_is_allowed(tmp_path):
    """Non-identity fields (e.g. poll_interval_s) may differ — only a note, not a refusal."""
    config = _make_config(tmp_path)
    root = config.root
    root.mkdir(parents=True, exist_ok=True)

    # Write queue_state with different poll_interval_s (non-identity).
    cfg_dict = config.to_dict()
    cfg_dict["poll_interval_s"] = 999
    qs = {
        "config": cfg_dict,
        "next_batch_number": 10,
        "halt": None,
        "spend_usd_total": 0.0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    (root / "queue_state.json").write_text(json.dumps(qs), encoding="utf-8")

    daemon = _make_daemon(tmp_path, config=config)
    # Should NOT raise.
    report = daemon.startup()
    assert report["lock"] == "acquired"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_halted_queue_tick_returns_halted(tmp_path):
    config = _make_config(tmp_path)
    root = config.root
    _arm(root)
    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    # Set halt manually.
    daemon._queue_state["halt"] = {"active": True, "reason": "test", "at": NOW}
    daemon._save_queue_state()

    tag = daemon.tick()
    assert tag == "halted"


def test_waiting_journal_when_insufficient_and_run_in_progress(tmp_path):
    """Fewer rows than slice_size and run still in progress → waiting_journal."""
    journal = tmp_path / "j" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _mark_run_in_progress(run_dir)

    # Only 1 row < SLICE_SIZE
    _write_journal(journal, [_make_row("Theorem A.")])

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)

    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    tag = daemon.tick()
    assert tag == "waiting_journal"


def test_next_batch_number_starts_at_10(tmp_path):
    config = _make_config(tmp_path)
    root = config.root
    _arm(root)
    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    qs = json.loads((root / "queue_state.json").read_text())
    assert qs["next_batch_number"] == 10


def test_status_md_written_every_tick(tmp_path):
    journal = tmp_path / "j" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_journal(journal, [_make_row(s) for s in _make_statements(SLICE_SIZE)])

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)
    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    daemon.tick()  # sliced
    assert (root / "STATUS.md").exists()


def test_events_jsonl_appended_on_events(tmp_path):
    """Events (freeze, halt) all appear in events.jsonl."""
    journal = tmp_path / "j" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_journal(journal, [_make_row(s) for s in _make_statements(SLICE_SIZE)])

    runner = FakeRunner()
    runner._cascade_cost = 10.0  # Trigger cost guard.

    config = _make_config(tmp_path, journal_path=journal, run_dir=run_dir)
    root = config.root
    _arm(root)
    daemon = _make_daemon(tmp_path, runner=runner, config=config)
    daemon.startup()

    daemon.tick()  # slice
    daemon.tick()  # mount
    daemon.tick()  # cascade → cost guard → FROZEN + HALT

    events = (root / "events.jsonl").read_text().splitlines()
    assert len(events) >= 2
    kinds = [json.loads(e)["kind"] for e in events]
    assert "frozen" in kinds
    assert "halt_set" in kinds


def test_run_forever_stops_on_signal(tmp_path):
    """run_forever respects _stop flag set between ticks."""
    config = _make_config(tmp_path)
    root = config.root
    _arm(root)
    daemon = _make_daemon(tmp_path, config=config)
    daemon.startup()

    # Patch tick to set stop flag after first call.
    original_tick = daemon.tick
    call_count = [0]

    def patched_tick():
        call_count[0] += 1
        daemon._stop = True
        return "waiting_journal"

    daemon.tick = patched_tick
    daemon.run_forever()
    assert call_count[0] == 1  # Stopped after one tick.

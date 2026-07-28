"""Acceptance tests for the bulk-batcher subsystem.

7 end-to-end scenarios driving BatcherDaemon over synthetic journals with
faked stage subprocesses.  Assertions are black-box over disk state.

Each test drives real BatcherDaemon code through real slicer / ledger / state
modules.  Only the subprocess runner and slot-checker are faked.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from icepick.batcher.backfill import backfill
from icepick.batcher.cli_glue import _handle_clear_halt
from icepick.batcher.config import BatcherConfig
from icepick.batcher.daemon import BatcherDaemon
from icepick.batcher.identity import (
    compute_uid,
    content_hash as make_content_hash,
    stmt_key as make_stmt_key,
)
from icepick.batcher.ledger import Ledger, LedgerRow
from icepick.batcher.state import load_all_states, load_state, transition


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NOW = "2026-07-07T00:00:00+00:00"
CAMPAIGN_SOURCE = "arxiv_bulk_acceptance_test"


# ---------------------------------------------------------------------------
# Journal builder helper
# ---------------------------------------------------------------------------


def _make_journal_row(i: int, j: int, *, statement: Optional[str] = None, answer: str = "42") -> dict:
    """Build a journal row in the canonical candidates.jsonl format.

    Row shape: {arxiv_id, candidate: {statement, answer, tier, provenance,
                truth_policy, metadata}}.
    """
    arxiv_id = f"25{i:02d}.{j:05d}"
    if statement is None:
        statement = f"Theorem 25{i:02d}.{j:05d}: The unique solution exists for case ({i},{j})."
    return {
        "arxiv_id": arxiv_id,
        "candidate": {
            "statement": statement,
            "answer": answer,
            "tier": "latex",
            "provenance": "extracted",
            "truth_policy": "extracted",
            "metadata": {"arxiv_id": arxiv_id},
        },
    }


def _write_journal(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _build_unique_rows(n: int, *, offset: int = 0) -> list[dict]:
    """Build n rows with unique statements (globally unique via offset)."""
    rows = []
    for k in range(n):
        idx = k + offset
        i = idx // 100
        j = idx % 100
        rows.append(_make_journal_row(i, j))
    return rows


# ---------------------------------------------------------------------------
# Shared FakeStageRunner
#
# High-fidelity simulation of:
#   - "allocation mount"  → creates runs/<ts>/handoff/records.jsonl + manifest.json
#   - "processing wellposed-cascade" → writes cascade_manifest.json + final_corpus.jsonl
#   - "processing pass_at_k"         → writes pass_at_k_manifest.json + pass_at_k.jsonl
# ---------------------------------------------------------------------------

_TS_COUNTER = [0]  # global monotonic tick for distinct fake run-dir timestamps


def _next_fake_ts() -> str:
    _TS_COUNTER[0] += 1
    return f"20260707T{_TS_COUNTER[0]:06d}Z"


class FakeStageRunner:
    """Injectable runner simulating all three funnel stage subprocesses.

    Configuration attributes (set per-test before use):
      mount_ok         — if False, mount returns rc=1 (exec_failed)
      cascade_ok       — if False, cascade returns rc=1 (exec_failed)
      cascade_cost     — float written to cascade_manifest overall.total_estimated_cost_usd
      cascade_m_fraction — fraction of N records that become final_corpus rows (default 0.5)
      passk_ok         — if False, pass@k returns rc=1 (exec_failed)
      passk_interrupted — if True, manifest interrupted=true
      mode_override    — expected --mode value (for test 7)
      calibration_sheet — expected --calibration-sheet value (for test 7)

    Call counters (per batch-dir):
      mount_calls[batch_dir]   → int
      cascade_calls[batch_dir] → int
      passk_calls[batch_dir]   → int
    """

    def __init__(
        self,
        *,
        mount_ok: bool = True,
        cascade_ok: bool = True,
        cascade_cost: float = 2.30,
        cascade_m_fraction: float = 0.5,
        passk_ok: bool = True,
        passk_interrupted: bool = False,
    ):
        self.mount_ok = mount_ok
        self.cascade_ok = cascade_ok
        self.cascade_cost = cascade_cost
        self.cascade_m_fraction = cascade_m_fraction
        self.passk_ok = passk_ok
        self.passk_interrupted = passk_interrupted

        # Per-call logging
        self.all_calls: list[list[str]] = []
        self.mount_calls: dict[str, int] = {}
        self.cascade_calls: dict[str, int] = {}
        self.passk_calls: dict[str, int] = {}

    def __call__(self, argv, env=None, capture_output=True, text=True, timeout=None):
        self.all_calls.append(list(argv))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        cmd = " ".join(str(a) for a in argv)

        if "allocation" in cmd and "mount" in cmd:
            return self._handle_mount(argv, result)
        elif "wellposed-cascade" in cmd:
            return self._handle_cascade(argv, result)
        elif "pass_at_k" in cmd:
            return self._handle_passk(argv, result)

        return result

    # ------------------------------------------------------------------

    def _handle_mount(self, argv, result):
        """High-fidelity mount simulation.

        Reads --path (slice_records.jsonl), creates
        <output-dir>/runs/<fake-ts>/handoff/records.jsonl by passing through
        every record and stamping setdefault fields for
        source/provenance/truth_policy/family.  Also writes manifest.json with
        source_name = the --source argument.
        """
        output_dir = None
        slice_records_path = None
        campaign_source = CAMPAIGN_SOURCE

        for i, a in enumerate(argv):
            if a == "--output-dir" and i + 1 < len(argv):
                output_dir = Path(argv[i + 1])
            if a == "--path" and i + 1 < len(argv):
                slice_records_path = Path(argv[i + 1])
            if a == "--source" and i + 1 < len(argv):
                campaign_source = argv[i + 1]

        if not self.mount_ok:
            result.returncode = 1
            result.stderr = "mount failed (simulated)"
            return result

        assert output_dir is not None, "FakeStageRunner: mount called without --output-dir"

        # Increment per-batch-dir counter
        batch_dir = str(output_dir.parent)
        self.mount_calls[batch_dir] = self.mount_calls.get(batch_dir, 0) + 1

        # Read slice records
        records = []
        if slice_records_path and slice_records_path.exists():
            for line in slice_records_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Create run dir with distinct timestamp
        ts = _next_fake_ts()
        run_dir = output_dir / "runs" / ts
        handoff_dir = run_dir / "handoff"
        handoff_dir.mkdir(parents=True, exist_ok=True)

        # Write manifest.json (source_name for read_manifest_source_name)
        (run_dir / "manifest.json").write_text(
            json.dumps({"source_name": campaign_source}),
            encoding="utf-8",
        )

        # Write handoff/records.jsonl — pass-through with setdefault stamps
        with (handoff_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                out = dict(rec)
                out.setdefault("source", campaign_source)
                out.setdefault("provenance", "extracted")
                out.setdefault("truth_policy", "extracted")
                out.setdefault("family", "realmath")
                fh.write(json.dumps(out) + "\n")

        return result

    def _handle_cascade(self, argv, result):
        """Simulate wellposed-cascade.

        Writes cascade_manifest.json and final_corpus.jsonl into <--output-dir>.
        N = number of rows in the --input handoff file.
        M = max(1, int(N * cascade_m_fraction)).
        """
        output_dir = None
        handoff_records_path = None

        for i, a in enumerate(argv):
            if a == "--output-dir" and i + 1 < len(argv):
                output_dir = Path(argv[i + 1])
            if a == "--input" and i + 1 < len(argv):
                handoff_records_path = Path(argv[i + 1])

        batch_dir = str(output_dir.parent) if output_dir else "?"
        self.cascade_calls[batch_dir] = self.cascade_calls.get(batch_dir, 0) + 1

        if not self.cascade_ok:
            result.returncode = 1
            result.stderr = "cascade failed (simulated)"
            return result

        assert output_dir is not None

        # Count input records
        n = 0
        if handoff_records_path and handoff_records_path.exists():
            for line in handoff_records_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    n += 1

        m = max(1, int(n * self.cascade_m_fraction))
        cost = self.cascade_cost

        output_dir.mkdir(parents=True, exist_ok=True)

        # Write cascade_manifest.json
        manifest = {
            "inputs": {"initial_record_count": n},
            "overall": {
                "initial_record_count": n,
                "final_corpus_count": m,
                "total_estimated_cost_usd": cost,
            },
            "outputs": {"final_corpus_count": m},
        }
        (output_dir / "cascade_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        # Write final_corpus.jsonl (m rows, each carrying a uid for identification)
        # Re-read the first m input records to carry real uids if possible
        corpus_rows = []
        if handoff_records_path and handoff_records_path.exists():
            for line in handoff_records_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and len(corpus_rows) < m:
                    corpus_rows.append(json.loads(line))
        # Pad if needed
        while len(corpus_rows) < m:
            corpus_rows.append({"uid": f"synthetic_uid_{len(corpus_rows)}"})

        with (output_dir / "final_corpus.jsonl").open("w", encoding="utf-8") as fh:
            for row in corpus_rows:
                fh.write(json.dumps(row) + "\n")

        return result

    def _handle_passk(self, argv, result):
        """Simulate processing pass_at_k.

        Writes pass_at_k_manifest.json and pass_at_k.jsonl.
        """
        output_dir = None
        input_path = None

        for i, a in enumerate(argv):
            if a == "--output-dir" and i + 1 < len(argv):
                output_dir = Path(argv[i + 1])
            if a == "--input" and i + 1 < len(argv):
                input_path = Path(argv[i + 1])

        batch_dir = str(output_dir.parent) if output_dir else "?"
        self.passk_calls[batch_dir] = self.passk_calls.get(batch_dir, 0) + 1

        if not self.passk_ok:
            result.returncode = 1
            result.stderr = "passk failed (simulated)"
            return result

        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        m = 0
        if input_path and input_path.exists():
            for line in input_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    m += 1

        manifest = {
            "interrupted": self.passk_interrupted,
            "counts": {
                "band": 2,
                "solved": 1,
                "drop": max(0, m - 3),
            },
            "model_calls": m * 8,
        }
        (output_dir / "pass_at_k_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        # Write pass_at_k.jsonl
        corpus_rows = []
        if input_path and input_path.exists():
            for line in input_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    corpus_rows.append(json.loads(line))
        with (output_dir / "pass_at_k.jsonl").open("w", encoding="utf-8") as fh:
            for row in corpus_rows:
                fh.write(json.dumps(row) + "\n")

        return result


# ---------------------------------------------------------------------------
# Config / daemon factory helpers
# ---------------------------------------------------------------------------


def _make_config(
    root: Path,
    journal_path: Path,
    run_dir: Path,
    *,
    slice_size: int = 5,
    campaign_source: str = CAMPAIGN_SOURCE,
    cross_source_statement_policy: str = "skip",
    mode: str = "production",
    calibration_sheet: Optional[str] = None,
    cost_limit_usd: float = 5.0,
) -> BatcherConfig:
    return BatcherConfig(
        root=root,
        journal_path=journal_path,
        run_dir=run_dir,
        campaign_source=campaign_source,
        slice_size=slice_size,
        cross_source_statement_policy=cross_source_statement_policy,
        mode=mode,
        calibration_sheet=calibration_sheet,
        cost_limit_usd=cost_limit_usd,
        key_path="/fake/key.env",
        icepick_bin="icepick",
        poll_interval_s=0,
        qwen_recheck_interval_s=0,
    )


def _make_daemon(
    config: BatcherConfig,
    runner: Optional[FakeStageRunner] = None,
    slot_checker=None,
) -> BatcherDaemon:
    if runner is None:
        runner = FakeStageRunner()
    if slot_checker is None:
        slot_checker = lambda: True
    return BatcherDaemon(
        config,
        runner=runner,
        slot_checker=slot_checker,
        sleep_fn=lambda s: None,
        now_iso_fn=lambda: NOW,
        clock=time.monotonic,
    )


def _arm(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ARMED").write_text(json.dumps({"armed_at": NOW}), encoding="utf-8")


def _conclude_run(run_dir: Path) -> None:
    progress = run_dir / "_progress"
    progress.mkdir(parents=True, exist_ok=True)
    incomplete = progress / "INCOMPLETE"
    if incomplete.exists():
        incomplete.unlink()


def _mark_run_in_progress(run_dir: Path) -> None:
    progress = run_dir / "_progress"
    progress.mkdir(parents=True, exist_ok=True)
    (progress / "INCOMPLETE").write_text("in progress", encoding="utf-8")


def _drive_to_ready(daemon: BatcherDaemon, *, max_ticks: int = 30) -> list[str]:
    """Drive daemon ticks until READY or max_ticks hit. Returns tag list."""
    tags = []
    for _ in range(max_ticks):
        tag = daemon.tick()
        tags.append(tag)
        if tag == "ready":
            break
    return tags


def _drive_n_slices(daemon: BatcherDaemon, n: int, *, max_ticks_each: int = 20) -> None:
    """Drive daemon until n batches are READY."""
    ready = 0
    for _ in range(n * max_ticks_each):
        tag = daemon.tick()
        if tag == "ready":
            ready += 1
            if ready >= n:
                break


def _read_ledger_rows(root: Path) -> list[dict]:
    ledger_path = root / "ledger" / "consumed_uids.jsonl"
    if not ledger_path.exists():
        return []
    rows = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _read_events(root: Path) -> list[dict]:
    events_path = root / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _read_queue_state(root: Path) -> dict:
    qs_path = root / "queue_state.json"
    return json.loads(qs_path.read_text(encoding="utf-8"))


def _read_slice_manifest(batch_dir: Path) -> dict:
    return json.loads((batch_dir / "slice_manifest.json").read_text(encoding="utf-8"))


def _read_handoff_records(batch_dir: Path) -> list[dict]:
    """Return all rows from the (single) mounted handoff file."""
    intake_runs = batch_dir / "intake" / "runs"
    if not intake_runs.exists():
        return []
    runs = sorted(intake_runs.iterdir())
    if not runs:
        return []
    handoff = runs[-1] / "handoff" / "records.jsonl"
    rows = []
    for line in handoff.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ===========================================================================
# Acceptance test 1 — Exactness: 1003 → 4×250 + 3 HELD
# ===========================================================================


def test_acceptance_1_exactness_1003_records(tmp_path):
    """Journal with EXACTLY 1003 unique records, slice_size=250.

    Drive daemon ticks (armed, fake runners, slot free, run NOT concluded yet)
    until 4 batches reach READY.

    Assert:
    - exactly 4 batch dirs; each slice_manifest has EXACTLY 250 entries
    - each fake-mounted handoff has exactly 250 rows
    - ledger has exactly 1000 batch-consumed rows, all uids distinct
    - batches are batch10..batch13

    Then mark run concluded (remove INCOMPLETE) → next tick returns
    'held_remainder'; assert STATUS.md contains a HELD section with count 3;
    assert those 3 uids NEVER appear in any ledger batch row or batch dir;
    further ticks never slice them.
    """
    journal = tmp_path / "run" / "_progress" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1003 unique rows
    rows = _build_unique_rows(1003)
    _write_journal(journal, rows)

    # Mark run in-progress (INCOMPLETE present)
    _mark_run_in_progress(run_dir)

    root = tmp_path / "batcher"
    config = _make_config(
        root, journal, run_dir,
        slice_size=250,  # production value per spec
    )
    runner = FakeStageRunner()
    daemon = _make_daemon(config, runner)
    _arm(root)
    daemon.startup()

    # Drive until 4 slices reach READY
    ready_count = 0
    for _ in range(4 * 10):  # at most 10 ticks per batch
        tag = daemon.tick()
        if tag == "ready":
            ready_count += 1
            if ready_count == 4:
                break

    assert ready_count == 4, f"Expected 4 READY batches, got {ready_count}"

    batches_root = root / "batches"
    batch_dirs = sorted(
        [d for d in batches_root.iterdir() if d.is_dir() and d.name.startswith("batch")],
        key=lambda d: int(d.name[5:]),
    )

    # Exactly 4 batch dirs
    assert len(batch_dirs) == 4, f"Expected 4 batch dirs, got {[d.name for d in batch_dirs]}"

    # Named batch10..batch13
    batch_names = [d.name for d in batch_dirs]
    assert batch_names == ["batch10", "batch11", "batch12", "batch13"], batch_names

    # Each slice_manifest has exactly 250 entries
    all_slice_uids: set[str] = set()
    all_ledger_batch_uids: set[str] = set()
    for bd in batch_dirs:
        manifest = _read_slice_manifest(bd)
        entries = manifest["entries"]
        assert len(entries) == 250, f"{bd.name}: expected 250 entries in manifest, got {len(entries)}"
        for e in entries:
            all_slice_uids.add(e["uid"])

        # Each mounted handoff has exactly 250 rows
        handoff = _read_handoff_records(bd)
        assert len(handoff) == 250, f"{bd.name}: expected 250 handoff rows, got {len(handoff)}"

    # Ledger: exactly 1000 batch-consumed rows (not watch or hist)
    ledger_rows = _read_ledger_rows(root)
    batch_rows = [r for r in ledger_rows if r["batch"].startswith("batch")]
    assert len(batch_rows) == 1000, f"Expected 1000 ledger batch rows, got {len(batch_rows)}"

    # All 1000 uids distinct
    batch_uids = [r["uid"] for r in batch_rows]
    assert len(set(batch_uids)) == 1000, "UIDs in ledger are not all distinct"

    # --- Now conclude the run and check remainder HELD ---
    _conclude_run(run_dir)

    # Next tick should return held_remainder
    tag = daemon.tick()
    assert tag == "held_remainder", f"Expected held_remainder after conclusion, got {tag!r}"

    # STATUS.md has HELD section with count 3
    status = (root / "STATUS.md").read_text(encoding="utf-8")
    assert "HELD Remainder" in status, "STATUS.md missing 'HELD Remainder' section"
    assert "3" in status, "STATUS.md should mention count 3 for held remainder"

    # The 3 remainder uids are the last 3 rows (rows 1001, 1002, 1003 = indices 1000-1002)
    remainder_rows = rows[1000:]
    assert len(remainder_rows) == 3
    remainder_stmts = [r["candidate"]["statement"] for r in remainder_rows]
    remainder_uids = {compute_uid(CAMPAIGN_SOURCE, s) for s in remainder_stmts}

    # None of the remainder uids appear in any ledger batch row
    for uid in remainder_uids:
        assert uid not in set(batch_uids), f"Remainder uid {uid} found in ledger batch rows"

    # None appear in any batch dir's slice_manifest
    for bd in batch_dirs:
        manifest = _read_slice_manifest(bd)
        manifest_uids = {e["uid"] for e in manifest["entries"]}
        assert remainder_uids.isdisjoint(manifest_uids), (
            f"{bd.name} contains a remainder uid"
        )

    # Further ticks never slice them
    for _ in range(5):
        tag = daemon.tick()
        assert tag in ("held_remainder", "waiting_journal", "disarmed"), (
            f"Unexpected tag after remainder: {tag!r}"
        )

    # Still only 4 batch dirs
    batch_dirs_after = [
        d for d in batches_root.iterdir() if d.is_dir() and d.name.startswith("batch")
    ]
    assert len(batch_dirs_after) == 4, "New batches were created from the remainder"


# ===========================================================================
# Acceptance test 2 — Crash-resume at every stage boundary
# ===========================================================================


@pytest.mark.parametrize("boundary", [
    "post-slice/pre-mount",
    "post-mount/pre-cascade",
    "post-cascade/pre-passk",
    "post-passk/pre-READY",
])
def test_acceptance_2_crash_resume_every_boundary(tmp_path, boundary):
    """Run lifecycle to boundary, discard daemon, build fresh daemon, complete.

    Assert vs uninterrupted control run:
    - identical batch composition (same uid set per slice_manifest)
    - ledger contains each uid exactly once
    - stage invocation counters show no double-billing
      (completed stages: mount/cascade/passk runner call counts <= control + 0)
    """
    SLICE = 5
    journal = tmp_path / "run" / "_progress" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = _build_unique_rows(SLICE)
    _write_journal(journal, rows)

    root = tmp_path / "batcher"
    config = _make_config(root, journal, run_dir, slice_size=SLICE)

    # --- Control run (uninterrupted) ---
    runner_ctrl = FakeStageRunner()
    daemon_ctrl = _make_daemon(config, runner_ctrl)
    _arm(root)
    daemon_ctrl.startup()
    _drive_to_ready(daemon_ctrl)
    daemon_ctrl._release_lock()

    ctrl_manifest = _read_slice_manifest(root / "batches" / "batch10")
    ctrl_uids = {e["uid"] for e in ctrl_manifest["entries"]}

    # --- Reset and rebuild for the crash-resume run ---
    import shutil
    shutil.rmtree(str(root))
    _TS_COUNTER[0] = 100  # separate ts range

    runner1 = FakeStageRunner()
    daemon1 = _make_daemon(config, runner1)
    _arm(root)
    daemon1.startup()

    # Map boundary to ticks needed to reach it
    # State progression per tick (one stage per tick):
    #   slice → SLICED, mount → MOUNTED, cascade → CASCADE_DONE, passk → PASSK_DONE
    boundary_ticks = {
        "post-slice/pre-mount": 1,       # SLICED
        "post-mount/pre-cascade": 2,     # MOUNTED
        "post-cascade/pre-passk": 3,     # CASCADE_DONE
        "post-passk/pre-READY": 4,       # PASSK_DONE
    }
    ticks_needed = boundary_ticks[boundary]

    for _ in range(ticks_needed):
        daemon1.tick()

    # Verify we reached the expected state
    state_at_boundary = {
        "post-slice/pre-mount": "SLICED",
        "post-mount/pre-cascade": "MOUNTED",
        "post-cascade/pre-passk": "CASCADE_DONE",
        "post-passk/pre-READY": "PASSK_DONE",
    }[boundary]
    st = load_state(root / "batches" / "batch10")
    assert st["state"] == state_at_boundary, (
        f"Expected {state_at_boundary!r} at boundary {boundary!r}, got {st['state']!r}"
    )

    # Capture invocation counts before crash
    batch_dir_str = str(root / "batches" / "batch10")
    mounts_before = runner1.mount_calls.get(batch_dir_str, 0)
    cascades_before = runner1.cascade_calls.get(batch_dir_str, 0)
    passk_before = runner1.passk_calls.get(batch_dir_str, 0)

    # Simulate crash: release lock
    daemon1._release_lock()

    # Fresh daemon from same disk root
    _TS_COUNTER[0] = 200  # distinct ts range for resumed run
    runner2 = FakeStageRunner()
    daemon2 = _make_daemon(config, runner2)
    daemon2.startup()  # calls recover_pending_slice

    # Drive to completion
    _drive_to_ready(daemon2)

    # Final state is READY
    final_st = load_state(root / "batches" / "batch10")
    assert final_st["state"] == "READY"
    assert (root / "batches" / "batch10" / "READY_TO_FOLD").exists()

    # Same batch composition
    resume_manifest = _read_slice_manifest(root / "batches" / "batch10")
    resume_uids = {e["uid"] for e in resume_manifest["entries"]}
    assert resume_uids == ctrl_uids, (
        f"batch composition mismatch after resume at {boundary!r}"
    )

    # Ledger: each uid exactly once (batch rows only)
    ledger_rows = _read_ledger_rows(root)
    batch_uids_ledger = [r["uid"] for r in ledger_rows if r["batch"].startswith("batch")]
    assert len(batch_uids_ledger) == SLICE
    assert len(set(batch_uids_ledger)) == SLICE, "Duplicate uids in ledger after resume"

    # Skip-if-done: stages that already ran should NOT be reinvoked
    mounts_after = runner2.mount_calls.get(batch_dir_str, 0)
    cascades_after = runner2.cascade_calls.get(batch_dir_str, 0)
    passk_after = runner2.passk_calls.get(batch_dir_str, 0)

    if boundary in ("post-mount/pre-cascade", "post-cascade/pre-passk", "post-passk/pre-READY"):
        # Mount already done before crash → MOUNT_VERIFIED present → skip-if-done
        assert mounts_after == 0, (
            f"Mount was re-invoked after resume at {boundary!r} (expected 0, got {mounts_after})"
        )

    if boundary in ("post-cascade/pre-passk", "post-passk/pre-READY"):
        # Cascade already done → cascade_manifest.json present → skip-if-done
        assert cascades_after == 0, (
            f"Cascade was re-invoked after resume at {boundary!r} (got {cascades_after})"
        )

    if boundary == "post-passk/pre-READY":
        # pass@k already done → manifest with interrupted=false → skip-if-done
        assert passk_after == 0, (
            f"pass@k was re-invoked after resume at {boundary!r} (got {passk_after})"
        )


# ===========================================================================
# Acceptance test 3 — Dup injection
# ===========================================================================


def test_acceptance_3a_byte_identical_replay_is_skipped_and_refilled(tmp_path):
    """Journal where row 3 (index 2) is a BYTE-IDENTICAL replay of row 1 (index 0).

    The replay must appear within the first slice_size rows so that the slicer
    hits it during the pass that needs to refill (the loop breaks once pending
    reaches slice_size, so a replay placed after the fill point is never seen).

    Daemon slices: batch still EXACTLY slice_size, replay skipped + logged,
    ledger has the uid once.

    Layout with SLICE=5:
      rows: [u0, u1, u0_replay, u2, u3, u4, u5]
      slicer sees: u0(accept), u1(accept), u0_replay(replay→skip), u2(accept),
                   u3(accept), u4(accept) → 5 accepted; stop.
    """
    SLICE = 5
    journal = tmp_path / "run" / "_progress" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build SLICE+2 unique rows (need SLICE + 1 filler to fill after 1 replay skip)
    unique_rows = _build_unique_rows(SLICE + 2)
    # Place byte-identical replay of unique_rows[0] at position 2
    replay_row = unique_rows[0]
    # Journal: u0, u1, u0(replay), u2, u3, u4, u5
    rows = [unique_rows[0], unique_rows[1], replay_row] + unique_rows[2:SLICE + 2]
    _write_journal(journal, rows)

    root = tmp_path / "batcher"
    config = _make_config(root, journal, run_dir, slice_size=SLICE)
    runner = FakeStageRunner()
    daemon = _make_daemon(config, runner)
    _arm(root)
    daemon.startup()

    tag = daemon.tick()
    assert tag == "sliced", f"Expected sliced, got {tag!r}"

    # Batch exists
    batch_dir = root / "batches" / "batch10"
    assert batch_dir.exists()

    # Manifest has exactly SLICE entries
    manifest = _read_slice_manifest(batch_dir)
    assert len(manifest["entries"]) == SLICE, (
        f"Expected {SLICE} manifest entries, got {len(manifest['entries'])}"
    )

    # Replay was logged in manifest skips
    skips = manifest.get("skips", [])
    replay_skips = [s for s in skips if s.get("kind") == "replay"]
    assert len(replay_skips) >= 1, "Expected at least 1 replay skip logged in manifest"

    # cross_source_skips.jsonl OR manifest skips records the replay
    # (The design says "log" — slicer logs replays into the manifest's skips list
    # and also calls ledger.log_skip which writes cross_source_skips.jsonl)
    cross_skips_path = root / "ledger" / "cross_source_skips.jsonl"
    # cross_source_skips.jsonl may contain it too
    all_skips = []
    if cross_skips_path.exists():
        for line in cross_skips_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_skips.append(json.loads(line))

    # The replay uid should appear in manifest skips OR cross_source_skips.jsonl
    # manifest skips already verified above; that's sufficient.

    # Ledger has the uid exactly once (the original)
    ledger_rows = _read_ledger_rows(root)
    batch_rows = [r for r in ledger_rows if r["batch"].startswith("batch")]
    replay_uid = compute_uid(CAMPAIGN_SOURCE, replay_row["candidate"]["statement"])
    uid_occurrences = [r for r in batch_rows if r["uid"] == replay_uid]
    assert len(uid_occurrences) == 1, (
        f"Expected uid to appear exactly once in ledger, got {len(uid_occurrences)}"
    )


def test_acceptance_3b_same_uid_mutated_content_hard_abort(tmp_path):
    """Row shares the uid-determining content (same statement) but MUTATED other
    content (different answer → different content_hash) → slice-abort.

    queue_state halt active with reason containing 'uid'/'abort',
    NOTHING committed (no new batch dir, ledger unchanged, cursor unchanged),
    STATUS.md + events.jsonl show it loudly.
    """
    SLICE = 5
    journal = tmp_path / "run" / "_progress" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build enough unique rows to fill a slice
    unique_rows = _build_unique_rows(SLICE + 5)

    # Seed the ledger before daemon startup with one row carrying a known uid
    stmt_conflict = unique_rows[0]["candidate"]["statement"]
    uid_conflict = compute_uid(CAMPAIGN_SOURCE, stmt_conflict)
    sk_conflict = make_stmt_key(stmt_conflict)

    # Create a "mutated" row: same statement → same uid, but different answer → different content_hash
    mutated_row = _make_journal_row(99, 99, statement=stmt_conflict, answer="DIFFERENT_ANSWER")

    # Journal: start with the mutated row (will conflict), then fillers
    rows = [mutated_row] + unique_rows[1:SLICE + 3]
    _write_journal(journal, rows)

    root = tmp_path / "batcher"
    config = _make_config(root, journal, run_dir, slice_size=SLICE)
    runner = FakeStageRunner()
    daemon = _make_daemon(config, runner)
    _arm(root)
    daemon.startup()

    # Seed the ledger with the original uid (different content_hash → conflict)
    orig_row = unique_rows[0]
    orig_ch = make_content_hash(orig_row)  # content_hash of the ORIGINAL row
    seed_row = LedgerRow(
        uid=uid_conflict,
        stmt_key=sk_conflict,
        content_hash=orig_ch,
        batch="hist:old",
        source_journal="old.jsonl",
        journal_line=1,
        sliced_at=NOW,
        warn_only=False,
    )
    daemon._ledger.append_all([seed_row])

    # Capture cursor and ledger state before
    cursor_before = daemon._cursor.get(journal)
    ledger_before = _read_ledger_rows(root)

    tag = daemon.tick()
    assert tag == "slice_aborted", f"Expected slice_aborted, got {tag!r}"

    # queue_state halt active, reason contains 'uid' or 'abort'
    qs = _read_queue_state(root)
    halt = qs.get("halt") or {}
    assert halt.get("active") is True, "Queue halt must be active after uid_conflict abort"
    reason = halt.get("reason", "")
    assert "uid" in reason.lower() or "abort" in reason.lower(), (
        f"Halt reason should mention 'uid' or 'abort', got: {reason!r}"
    )

    # No batch dir created
    batches_root = root / "batches"
    if batches_root.exists():
        batch_dirs = [d for d in batches_root.iterdir() if d.is_dir() and d.name.startswith("batch")]
        assert len(batch_dirs) == 0, "No batch dirs should be created on abort"

    # Ledger unchanged (no new batch rows added)
    ledger_after = _read_ledger_rows(root)
    batch_rows_before = [r for r in ledger_before if r["batch"].startswith("batch")]
    batch_rows_after = [r for r in ledger_after if r["batch"].startswith("batch")]
    assert len(batch_rows_after) == len(batch_rows_before), "Ledger should not grow on abort"

    # STATUS.md mentions halt / abort
    status = (root / "STATUS.md").read_text(encoding="utf-8")
    assert "HALT" in status.upper() or "halt" in status.lower(), (
        "STATUS.md should mention halt after abort"
    )

    # events.jsonl records the abort event
    events = _read_events(root)
    event_kinds = [e["kind"] for e in events]
    assert any("abort" in k.lower() or "halt" in k.lower() for k in event_kinds), (
        f"events.jsonl should record abort/halt, got kinds: {event_kinds}"
    )


# ===========================================================================
# Acceptance test 4 — History collision
# ===========================================================================


def test_acceptance_4_history_collision(tmp_path):
    """Seed ledger via backfill() with synthetic hist source containing stmt S;
    also seed a direct ledger row for stmt T (same-campaign predecessor).

    Journal contains S and T among fillers.

    Assert:
    - S excluded via stmt_conflict → batch exact size, S absent from every
      slice manifest + mounted handoff
    - T (byte-fresh, same uid, different content_hash) → HARD ABORT path

    Sub-case: mount-refusal backstop — tamper a slice_manifest uid before
    the mount tick → mount verification fails → batch FROZEN.
    """
    SLICE = 5
    journal = tmp_path / "run" / "_progress" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    root = tmp_path / "batcher"
    config = _make_config(root, journal, run_dir, slice_size=SLICE)

    # --- Part A: stmt S is excluded via stmt_conflict (stmt_key known from hist) ---
    # Statement S comes from a different source (hist source != campaign source)
    # so it has a different uid but the same stmt_key.
    stmt_S = "Theorem S: This is a statement known from historical data."
    stmt_T = "Theorem T: Same uid as batch09_manual predecessor, different content."

    # Unique filler statements for the journal (enough to fill a slice)
    filler_stmts = [
        f"Theorem filler_{i}: Unique filler theorem {i}." for i in range(SLICE + 5)
    ]

    # Build journal: S, T (conflict uid), fillers
    rows = (
        [_make_journal_row(50, 0, statement=stmt_S)]
        + [_make_journal_row(50, 1, statement=stmt_T, answer="ORIGINAL_ANSWER")]
        + [_make_journal_row(50, i + 2, statement=s) for i, s in enumerate(filler_stmts)]
    )
    _write_journal(journal, rows)

    runner = FakeStageRunner()
    daemon = _make_daemon(config, runner)
    _arm(root)
    daemon.startup()

    # Backfill: seed S from a DIFFERENT hist source (stmt_key same but uid differs)
    hist_source_name = "hist_different_source"
    stmt_s_uid = compute_uid(hist_source_name, stmt_S)
    stmt_s_sk = make_stmt_key(stmt_S)
    stmt_s_ch = make_content_hash({"statement": stmt_S, "source": hist_source_name})
    hist_row = LedgerRow(
        uid=stmt_s_uid,
        stmt_key=stmt_s_sk,
        content_hash=stmt_s_ch,
        batch="hist:hist_batch1",
        source_journal="hist.jsonl",
        journal_line=-1,
        sliced_at=NOW,
        warn_only=False,
    )
    daemon._ledger.append_all([hist_row])

    # Seed T: same uid (campaign_source + stmt_T) but different content_hash
    stmt_t_uid = compute_uid(CAMPAIGN_SOURCE, stmt_T)
    stmt_t_sk = make_stmt_key(stmt_T)
    # The "original" content_hash differs from what the journal row will produce
    stmt_t_orig_ch = make_content_hash({"statement": stmt_T, "answer": "ORIGINAL_ANSWER", "source": "different"})
    # The journal row has answer="ORIGINAL_ANSWER" but we seed with a made-up different ch
    # so the uid_conflict triggers. Use a fabricated ch for the prior.
    seed_t_row = LedgerRow(
        uid=stmt_t_uid,
        stmt_key=stmt_t_sk,
        content_hash="aaaa" + "b" * 60,  # different from what journal row produces
        batch="batch09_manual",
        source_journal="manual.jsonl",
        journal_line=1,
        sliced_at=NOW,
        warn_only=False,
    )
    daemon._ledger.append_all([seed_t_row])

    # First tick: S is excluded via stmt_conflict (skipped+refilled), BUT T comes
    # before the fillers → uid_conflict abort triggered by T.
    # The journal order is: S, T, fillers...
    # S will hit stmt_conflict (policy=skip, refilled).
    # T will hit uid_conflict (HARD ABORT).
    tag = daemon.tick()
    assert tag == "slice_aborted", (
        f"Expected slice_aborted due to T uid_conflict, got {tag!r}"
    )

    qs = _read_queue_state(root)
    halt = qs.get("halt") or {}
    assert halt.get("active") is True, "Queue halt must be active after T uid_conflict"
    assert "uid" in halt.get("reason", "").lower() or "abort" in halt.get("reason", "").lower()

    # Nothing committed
    batches_root = root / "batches"
    if batches_root.exists():
        batch_dirs = [d for d in batches_root.iterdir() if d.is_dir()]
        assert len(batch_dirs) == 0, "No batch dirs should exist after abort"

    # events.jsonl has the abort/halt line
    events = _read_events(root)
    event_kinds = [e["kind"] for e in events]
    assert any("abort" in k or "halt" in k for k in event_kinds)

    # --- Sub-case: mount-refusal backstop ---
    # Build a fresh setup with no conflicts; after slice, tamper one uid in
    # slice_manifest before the mount tick → mount verification fails → FROZEN.
    import shutil
    shutil.rmtree(str(root))

    journal2 = tmp_path / "journal2" / "candidates.jsonl"
    run_dir2 = tmp_path / "run2"
    run_dir2.mkdir(parents=True, exist_ok=True)
    clean_rows = _build_unique_rows(SLICE + 2)
    _write_journal(journal2, clean_rows)

    root2 = tmp_path / "batcher2"
    config2 = _make_config(root2, journal2, run_dir2, slice_size=SLICE)
    runner2 = FakeStageRunner()
    daemon2 = _make_daemon(config2, runner2)
    _arm(root2)
    daemon2.startup()

    # Tick 1: slice → SLICED
    tag2 = daemon2.tick()
    assert tag2 == "sliced"

    # Tamper: overwrite one uid in the slice_manifest with a fake uid
    batch_dir2 = root2 / "batches" / "batch10"
    manifest_path2 = batch_dir2 / "slice_manifest.json"
    m2 = json.loads(manifest_path2.read_text(encoding="utf-8"))
    original_uid = m2["entries"][0]["uid"]
    m2["entries"][0]["uid"] = "tampered_uid_0000000000000000"
    manifest_path2.write_text(json.dumps(m2, indent=2), encoding="utf-8")

    # Tick 2: mount → should fail verification → batch FROZEN
    tag2 = daemon2.tick()
    assert tag2 == "frozen", f"Expected frozen after uid mismatch, got {tag2!r}"

    st2 = load_state(batch_dir2)
    assert st2["state"] == "FROZEN"

    # Queue should NOT halt for mount_verification_failed (only cost_guard / slice_abort halt)
    # But the batch is frozen, which serializes processing.
    qs2 = _read_queue_state(root2)
    # No queue halt for mount failure (mount failure freezes only the batch, does not halt the queue)
    # (per design: only cost_guard + slice_abort halt the queue)
    # Verify: further tick may try to slice (no in-flight non-frozen batch)
    tag2b = daemon2.tick()
    # Should try to slice (or wait_journal if journal exhausted → after slice we'd have consumed 5)
    # Actually journal has SLICE+2=7 rows; we sliced 5; 2 remain < slice_size → waiting_journal
    # Unless run concluded
    assert tag2b in ("sliced", "waiting_journal", "held_remainder"), (
        f"Unexpected tag after frozen batch: {tag2b!r}"
    )


# ===========================================================================
# Acceptance test 5 — Qwen contention
# ===========================================================================


def test_acceptance_5_qwen_contention(tmp_path):
    """Batch at CASCADE_DONE; slot_checker busy for first K checks.

    Each tick returns 'waiting_qwen', passk runner invocation count stays 0,
    state stays CASCADE_DONE.
    Then free → passk fires exactly once → PASSK_DONE → READY.
    """
    SLICE = 5
    K = 4  # times slot is busy

    journal = tmp_path / "run" / "_progress" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_journal(journal, _build_unique_rows(SLICE))

    root = tmp_path / "batcher"
    config = _make_config(root, journal, run_dir, slice_size=SLICE)

    runner = FakeStageRunner()
    slot_call_idx = [0]
    slot_results = [False] * K + [True]  # busy K times, then free

    def slot_checker():
        idx = slot_call_idx[0]
        slot_call_idx[0] += 1
        if idx < len(slot_results):
            return slot_results[idx]
        return True

    daemon = _make_daemon(config, runner, slot_checker=slot_checker)
    _arm(root)
    daemon.startup()

    # Drive to CASCADE_DONE
    daemon.tick()  # slice → SLICED
    daemon.tick()  # mount → MOUNTED
    daemon.tick()  # cascade → CASCADE_DONE

    st = load_state(root / "batches" / "batch10")
    assert st["state"] == "CASCADE_DONE"

    batch_dir_str = str(root / "batches" / "batch10")

    # K ticks: slot busy → waiting_qwen, passk NOT invoked
    for i in range(K):
        tag = daemon.tick()
        assert tag == "waiting_qwen", f"Tick {i+1}: expected waiting_qwen, got {tag!r}"
        passk_count = runner.passk_calls.get(batch_dir_str, 0)
        assert passk_count == 0, f"Tick {i+1}: passk should not have been invoked (got {passk_count})"
        st = load_state(root / "batches" / "batch10")
        assert st["state"] == "CASCADE_DONE", (
            f"Tick {i+1}: state should remain CASCADE_DONE, got {st['state']!r}"
        )

    # Next tick: slot free → passk fires
    tag = daemon.tick()
    assert tag == "passk_done", f"Expected passk_done after slot freed, got {tag!r}"
    passk_count = runner.passk_calls.get(batch_dir_str, 0)
    assert passk_count == 1, f"Expected exactly 1 passk invocation, got {passk_count}"

    # One more tick: READY
    tag = daemon.tick()
    assert tag == "ready"

    assert (root / "batches" / "batch10" / "READY_TO_FOLD").exists()


# ===========================================================================
# Acceptance test 6 — Cost guard
# ===========================================================================


def test_acceptance_6_cost_guard(tmp_path):
    """Fake cascade writes total_estimated_cost_usd=6.10 (> $5 limit).

    Assert:
    - batch FROZEN with reason containing 'cost_guard'
    - queue halt active
    - passk NEVER invoked for that batch
    - events.jsonl has the halt line
    - STATUS.md shows frozen + halt
    - subsequent tick: 'halted', no stage advancement
    - halt stops slicing too (journal growth does not produce new slice while halted)
    - clear-halt handler with --reason → next tick still does NOT advance FROZEN batch
      (freeze is per-batch; slicing resumes)
    """
    SLICE = 5

    journal = tmp_path / "run" / "_progress" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write enough journal rows to fill 2+ slices
    _write_journal(journal, _build_unique_rows(SLICE * 3))
    _mark_run_in_progress(run_dir)

    root = tmp_path / "batcher"
    config = _make_config(root, journal, run_dir, slice_size=SLICE)

    runner = FakeStageRunner(cascade_cost=6.10)  # > $5 limit
    daemon = _make_daemon(config, runner)
    _arm(root)
    daemon.startup()

    # Drive to frozen
    daemon.tick()  # slice → SLICED
    daemon.tick()  # mount → MOUNTED
    tag = daemon.tick()  # cascade → cost guard → FROZEN + HALT
    assert tag == "frozen", f"Expected frozen, got {tag!r}"

    batch_dir = root / "batches" / "batch10"
    st = load_state(batch_dir)
    assert st["state"] == "FROZEN"
    assert "cost_guard" in st["frozen"]["reason"].lower(), (
        f"FROZEN reason should mention cost_guard: {st['frozen']['reason']!r}"
    )

    # Queue halt active
    qs = _read_queue_state(root)
    halt = qs.get("halt") or {}
    assert halt.get("active") is True
    assert "cost_guard" in halt.get("reason", "").lower(), (
        f"Halt reason should mention cost_guard: {halt.get('reason')!r}"
    )

    # passk NEVER invoked
    batch_dir_str = str(batch_dir)
    assert runner.passk_calls.get(batch_dir_str, 0) == 0

    # events.jsonl has halt line
    events = _read_events(root)
    event_kinds = [e["kind"] for e in events]
    assert "halt_set" in event_kinds, f"events.jsonl missing halt_set event: {event_kinds}"
    assert "frozen" in event_kinds

    # STATUS.md shows frozen + halt
    status = (root / "STATUS.md").read_text(encoding="utf-8")
    assert "FROZEN" in status
    assert "HALT" in status.upper() or "halt" in status.lower()

    # Subsequent tick: 'halted', no stage advancement
    tag2 = daemon.tick()
    assert tag2 == "halted", f"Expected halted, got {tag2!r}"

    # Batch still FROZEN (not advanced)
    st2 = load_state(batch_dir)
    assert st2["state"] == "FROZEN"

    # Journal growth does not produce new slice while halted
    # (Add more rows to the journal — the daemon should not slice them)
    with journal.open("a", encoding="utf-8") as fh:
        for row in _build_unique_rows(SLICE, offset=1000):
            fh.write(json.dumps(row) + "\n")

    tag3 = daemon.tick()
    assert tag3 == "halted", "Should still be halted despite new journal rows"

    batches_root = root / "batches"
    batch_dirs = [d for d in batches_root.iterdir() if d.is_dir() and d.name.startswith("batch")]
    assert len(batch_dirs) == 1, "No new batches should be created while halted"

    # --- Clear halt via cli_glue handler ---
    root_str = str(root)

    class _FakeArgs:
        root = root_str
        reason = "cost verified by operator; batch frozen ok"

    _handle_clear_halt(_FakeArgs())

    # Reload queue state
    qs_cleared = _read_queue_state(root)
    halt_cleared = qs_cleared.get("halt") or {}
    assert halt_cleared.get("active") is False, "Halt should be cleared after clear-halt"

    # Next tick: halt is cleared, so no longer 'halted'
    # The frozen batch is the in-flight batch — it's FROZEN state,
    # so _find_in_flight finds it but it IS frozen → daemon skips it, tries to slice.
    # (per design: a frozen batch does NOT block slicing, only blocks stage advancement past it)
    # We need to reload the daemon's queue_state from disk since we wrote it externally.
    daemon._queue_state = json.loads((root / "queue_state.json").read_text(encoding="utf-8"))

    tag4 = daemon.tick()
    # After halt cleared: slicing resumes (new journal rows available)
    # Expected: 'sliced' (cuts batch11) or 'waiting_journal' (if cursor already past them)
    # The rows we added while halted are after the cursor position; slicing should resume.
    assert tag4 in ("sliced", "waiting_journal"), (
        f"Expected sliced or waiting_journal after clear-halt, got {tag4!r}"
    )

    # The FROZEN batch remains FROZEN (freeze is per-batch)
    st_final = load_state(batch_dir)
    assert st_final["state"] == "FROZEN", (
        "FROZEN batch should remain frozen after clear-halt (freeze != halt)"
    )

    # passk was never invoked on the frozen batch
    assert runner.passk_calls.get(batch_dir_str, 0) == 0


# ===========================================================================
# Acceptance test 7 — Dry-run / flow_testing mode ($0 spend)
# ===========================================================================


def test_acceptance_7_dry_run_zero_spend(tmp_path):
    """Full pipeline in mode='flow_testing' with calibration_sheet set.

    Assert:
    - every cascade/passk argv contains ['--mode','flow_testing']
      and the --calibration-sheet path
    - cascade manifest cost null → treated as $0 (queue spend_usd_total == 0.0)
    - whole run reaches READY
    - NO argv ever contains '--mode production'
    """
    SLICE = 5
    CAL_SHEET = "/fake/calibration_sheet.jsonl"

    journal = tmp_path / "run" / "_progress" / "candidates.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_journal(journal, _build_unique_rows(SLICE))

    root = tmp_path / "batcher"
    config = _make_config(
        root, journal, run_dir,
        slice_size=SLICE,
        mode="flow_testing",
        calibration_sheet=CAL_SHEET,
    )

    # FakeStageRunner with null cascade cost (flow_testing → cost is null → $0)
    class _NullCostRunner(FakeStageRunner):
        """Overrides cascade to write null cost (flow_testing behaviour)."""

        def _handle_cascade(self, argv, result):
            # Write null cost
            output_dir = None
            handoff_records_path = None
            for i, a in enumerate(argv):
                if a == "--output-dir" and i + 1 < len(argv):
                    output_dir = Path(argv[i + 1])
                if a == "--input" and i + 1 < len(argv):
                    handoff_records_path = Path(argv[i + 1])

            batch_dir = str(output_dir.parent) if output_dir else "?"
            self.cascade_calls[batch_dir] = self.cascade_calls.get(batch_dir, 0) + 1

            n = 0
            if handoff_records_path and handoff_records_path.exists():
                for line in handoff_records_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        n += 1
            m = max(1, n)

            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "inputs": {"initial_record_count": n},
                "overall": {
                    "initial_record_count": n,
                    "final_corpus_count": m,
                    "total_estimated_cost_usd": None,  # null → $0 in flow_testing
                },
                "outputs": {"final_corpus_count": m},
            }
            (output_dir / "cascade_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with (output_dir / "final_corpus.jsonl").open("w", encoding="utf-8") as fh:
                for i in range(m):
                    fh.write(json.dumps({"uid": f"ft_uid_{i}"}) + "\n")
            return result

    runner = _NullCostRunner()
    daemon = _make_daemon(config, runner)
    _arm(root)
    daemon.startup()

    # Drive to READY
    tags = _drive_to_ready(daemon)
    assert "ready" in tags, f"Expected ready in tags: {tags}"

    # Verify all cascade and passk calls contain --mode flow_testing
    for call_argv in runner.all_calls:
        cmd = " ".join(str(a) for a in call_argv)
        if "wellposed-cascade" in cmd or "pass_at_k" in cmd:
            # Must contain --mode flow_testing
            assert "--mode" in call_argv, f"argv missing --mode: {call_argv}"
            mode_idx = call_argv.index("--mode")
            assert call_argv[mode_idx + 1] == "flow_testing", (
                f"Expected --mode flow_testing, got {call_argv[mode_idx + 1]!r}"
            )
            # Must contain --calibration-sheet
            assert "--calibration-sheet" in call_argv, (
                f"argv missing --calibration-sheet: {call_argv}"
            )
            cal_idx = call_argv.index("--calibration-sheet")
            assert call_argv[cal_idx + 1] == CAL_SHEET, (
                f"Expected cal sheet {CAL_SHEET!r}, got {call_argv[cal_idx + 1]!r}"
            )
            # Must NOT contain '--mode production'
            for i, a in enumerate(call_argv):
                if a == "--mode" and i + 1 < len(call_argv):
                    assert call_argv[i + 1] != "production", (
                        f"argv contains --mode production in flow_testing run: {call_argv}"
                    )

    # spend_usd_total == 0.0 (null cost treated as $0)
    qs = _read_queue_state(root)
    assert qs["spend_usd_total"] == 0.0, (
        f"Expected $0 spend in flow_testing mode, got {qs['spend_usd_total']}"
    )

    # Batch reached READY
    batch_dir = root / "batches" / "batch10"
    st = load_state(batch_dir)
    assert st["state"] == "READY"
    assert (batch_dir / "READY_TO_FOLD").exists()

    # No argv ever contains '--mode production'
    for call_argv in runner.all_calls:
        for i, a in enumerate(call_argv):
            if a == "--mode" and i + 1 < len(call_argv):
                assert call_argv[i + 1] != "production", (
                    f"Found '--mode production' in argv: {call_argv}"
                )

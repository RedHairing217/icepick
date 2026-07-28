"""
tests/batcher/test_stages.py

Thorough unit tests for src/icepick/batcher/stages.py.
All tests use synthetic fixtures and fake runners — no real subprocesses to
icepick binaries ever.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from icepick.batcher.stages import (
    DEFAULT_TRANSIENT_MARKERS,
    StageOutcome,
    _default_is_retryable,
    _uid_set_sha,
    build_cascade_cmd,
    build_mount_cmd,
    build_passk_cmd,
    build_stage_env,
    cascade_slot_free,
    qwen_slot_free,
    run_cascade,
    run_mount,
    run_passk,
    with_retries,
)


# ---------------------------------------------------------------------------
# Helpers / FakeRunner
# ---------------------------------------------------------------------------


@dataclass
class FakeResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeRunner:
    """Records every (argv, env) call and returns scripted results."""

    def __init__(self, results: list[FakeResult] | None = None, side_effect=None):
        self.calls: list[dict] = []
        self._results = list(results or [])
        self._side_effect = side_effect
        self._index = 0
        # file_actions: list of callables executed BEFORE the result is returned
        self._file_actions: list[callable] = []

    def add_result(self, result: FakeResult, file_action=None) -> None:
        self._results.append(result)
        self._file_actions.append(file_action)

    def __call__(self, argv, env=None, capture_output=True, text=True, timeout=None):
        self.calls.append({"argv": argv, "env": env, "timeout": timeout})
        if self._side_effect is not None:
            raise self._side_effect
        if self._index < len(self._results):
            result = self._results[self._index]
            if self._index < len(self._file_actions) and self._file_actions[self._index]:
                self._file_actions[self._index]()
        else:
            result = FakeResult()
        self._index += 1
        return result


# ---------------------------------------------------------------------------
# Command builder tests
# ---------------------------------------------------------------------------


class TestBuildMountCmd:
    def test_exact_argv_production(self):
        cmd = build_mount_cmd(
            slice_records="/data/slice.jsonl",
            campaign_source="arxiv_bulk_pde625",
            batch_dir="/batches/batch10",
        )
        assert cmd == [
            "icepick",
            "allocation",
            "mount",
            "--path", "/data/slice.jsonl",
            "--source", "arxiv_bulk_pde625",
            "--provenance", "extracted",
            "--truth-policy", "extracted",
            "--family", "realmath",
            "--output-dir", "/batches/batch10/intake",
        ]

    def test_custom_bin(self):
        cmd = build_mount_cmd("/s.jsonl", "src", "/bdir", icepick_bin="myicepick")
        assert cmd[0] == "myicepick"

    def test_output_dir_is_intake_subdir(self):
        cmd = build_mount_cmd("/s.jsonl", "src", "/x/y")
        idx = cmd.index("--output-dir")
        assert cmd[idx + 1] == "/x/y/intake"

    def test_calibration_sheet_ignored_for_mount(self):
        # mount has no calibration-sheet arg; verify it doesn't bleed in
        cmd = build_mount_cmd("/s.jsonl", "src", "/bd", calibration_sheet="/cs.json")
        assert "--calibration-sheet" not in cmd


class TestBuildCascadeCmd:
    def test_exact_argv_production(self):
        cmd = build_cascade_cmd(
            handoff_records="/batch/intake/runs/x/handoff/records.jsonl",
            batch_dir="/batch",
            key_path="/keys/anthro.env",
        )
        assert cmd == [
            "icepick",
            "processing",
            "wellposed-cascade",
            "--mode", "production",
            "--stages", "codex:anthropic",
            "--input", "/batch/intake/runs/x/handoff/records.jsonl",
            "--output-dir", "/batch/cascade",
            "--anthro-key-file", "/keys/anthro.env",
            "--cost-per-input-mtok", "3",
            "--cost-per-output-mtok", "15",
        ]

    def test_flow_testing_mode(self):
        cmd = build_cascade_cmd("/h.jsonl", "/bd", "/k.env", mode="flow_testing")
        assert "--mode" in cmd
        assert cmd[cmd.index("--mode") + 1] == "flow_testing"

    def test_calibration_sheet_appended(self):
        cmd = build_cascade_cmd("/h.jsonl", "/bd", "/k.env", calibration_sheet="/cal.json")
        assert "--calibration-sheet" in cmd
        assert cmd[cmd.index("--calibration-sheet") + 1] == "/cal.json"

    def test_no_calibration_sheet_by_default(self):
        cmd = build_cascade_cmd("/h.jsonl", "/bd", "/k.env")
        assert "--calibration-sheet" not in cmd

    def test_cost_flags_are_strings(self):
        cmd = build_cascade_cmd("/h.jsonl", "/bd", "/k.env")
        assert cmd[cmd.index("--cost-per-input-mtok") + 1] == "3"
        assert cmd[cmd.index("--cost-per-output-mtok") + 1] == "15"


class TestBuildPasskCmd:
    def test_exact_argv_production(self):
        cmd = build_passk_cmd(batch_dir="/batch")
        assert cmd == [
            "icepick",
            "processing",
            "pass_at_k",
            "--mode", "production",
            "--input", "/batch/cascade/final_corpus.jsonl",
            "--output-dir", "/batch/pass_at_k",
            "--backend", "qwen_http",
            "--backend-url", "http://127.0.0.1:1234/v1/chat/completions",
            "--model", "qwen/qwen3-8b",
            "--k", "8",
            "--temperature", "0.7",
            "--max-tokens", "2048",
            "--think", "off",
            "--max-concurrent", "1",
        ]

    def test_calibration_sheet_appended(self):
        cmd = build_passk_cmd("/batch", calibration_sheet="/cal.json")
        assert "--calibration-sheet" in cmd
        assert cmd[cmd.index("--calibration-sheet") + 1] == "/cal.json"

    def test_flow_testing(self):
        cmd = build_passk_cmd("/batch", mode="flow_testing")
        assert cmd[cmd.index("--mode") + 1] == "flow_testing"

    def test_custom_bin(self):
        cmd = build_passk_cmd("/batch", icepick_bin="mybin")
        assert cmd[0] == "mybin"


# ---------------------------------------------------------------------------
# build_stage_env
# ---------------------------------------------------------------------------


class TestBuildStageEnv:
    def test_sets_key_file(self):
        env = build_stage_env("/path/to/key.env")
        assert env["ANTHROPIC_KEY_FILE"] == "/path/to/key.env"

    def test_is_copy_of_os_environ(self):
        env = build_stage_env("/k.env")
        for k, v in os.environ.items():
            if k != "ANTHROPIC_KEY_FILE":
                assert env.get(k) == v

    def test_does_not_mutate_os_environ(self):
        original = os.environ.copy()
        build_stage_env("/fake.env")
        assert os.environ == original


# ---------------------------------------------------------------------------
# run_mount tests
# ---------------------------------------------------------------------------


def _write_handoff(run_dir: Path, uids: list[str]) -> None:
    handoff_dir = run_dir / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps({"uid": uid, "statement": f"s{i}"})
                      for i, uid in enumerate(uids))
    (handoff_dir / "records.jsonl").write_text(lines)


def _make_run_dir(intake_path: Path, name: str = "20260707T120000Z") -> Path:
    run_dir = intake_path / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class TestRunMount:
    UIDS = ["uid_a", "uid_b", "uid_c"]

    def _happy_runner(self, batch_dir: str):
        """Returns a FakeRunner that creates the expected run dir + handoff on exec."""
        intake = Path(batch_dir) / "intake"

        def _action():
            run_dir = _make_run_dir(intake)
            _write_handoff(run_dir, self.UIDS)

        runner = FakeRunner(results=[FakeResult(returncode=0)])
        runner._file_actions = [_action]
        return runner

    def test_happy_path(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        outcome = run_mount(
            runner, str(tmp_path), "/slice.jsonl", "src", self.UIDS
        )
        assert outcome.ok
        assert outcome.kind == "mount_ok"
        assert len(runner.calls) == 1

    def test_argv_passed_to_runner(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        run_mount(runner, str(tmp_path), "/s.jsonl", "mysrc", self.UIDS)
        argv = runner.calls[0]["argv"]
        assert "allocation" in argv
        assert "mount" in argv
        assert "--source" in argv
        assert argv[argv.index("--source") + 1] == "mysrc"

    def test_marker_written_on_success(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        run_mount(runner, str(tmp_path), "/s.jsonl", "src", self.UIDS)
        marker = tmp_path / "intake" / "MOUNT_VERIFIED"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert "uid_set_sha" in data

    def test_idempotent_skips_rerun(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        # First run
        outcome1 = run_mount(runner, str(tmp_path), "/s.jsonl", "src", self.UIDS)
        assert outcome1.ok
        # Second run — should skip
        runner2 = FakeRunner()
        outcome2 = run_mount(runner2, str(tmp_path), "/s.jsonl", "src", self.UIDS)
        assert outcome2.ok
        assert outcome2.data.get("resumed") is True
        assert len(runner2.calls) == 0  # no subprocess called

    def test_idempotent_uid_set_mismatch_in_marker(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        run_mount(runner, str(tmp_path), "/s.jsonl", "src", self.UIDS)
        # Now call with different uids
        runner2 = FakeRunner()
        outcome = run_mount(runner2, str(tmp_path), "/s.jsonl", "src", ["uid_x"])
        assert not outcome.ok
        assert outcome.kind == "mount_dirty"

    def test_count_mismatch(self, tmp_path):
        """Handoff has fewer rows than expected_uids."""
        intake = tmp_path / "intake"

        def _action():
            run_dir = _make_run_dir(intake)
            _write_handoff(run_dir, ["uid_a"])  # only 1 row, expected 3

        runner = FakeRunner(results=[FakeResult(returncode=0)])
        runner._file_actions = [_action]
        outcome = run_mount(runner, str(tmp_path), "/s.jsonl", "src", self.UIDS)
        assert not outcome.ok
        assert outcome.kind == "mount_verification_failed"
        assert "mismatch" in outcome.detail.lower()

    def test_uid_mismatch(self, tmp_path):
        """Handoff has correct count but wrong uids."""
        intake = tmp_path / "intake"

        def _action():
            run_dir = _make_run_dir(intake)
            _write_handoff(run_dir, ["uid_x", "uid_y", "uid_z"])

        runner = FakeRunner(results=[FakeResult(returncode=0)])
        runner._file_actions = [_action]
        outcome = run_mount(runner, str(tmp_path), "/s.jsonl", "src", self.UIDS)
        assert not outcome.ok
        assert outcome.kind == "mount_verification_failed"
        assert "uid" in outcome.detail.lower()

    def test_dirty_existing_unverified_run(self, tmp_path):
        """Run dir exists but no MOUNT_VERIFIED marker → mount_dirty."""
        intake = tmp_path / "intake"
        _make_run_dir(intake, "20260707T111111Z")
        runner = FakeRunner()
        outcome = run_mount(runner, str(tmp_path), "/s.jsonl", "src", self.UIDS)
        assert not outcome.ok
        assert outcome.kind == "mount_dirty"
        assert len(runner.calls) == 0  # did not attempt to re-run

    def test_exec_failure(self, tmp_path):
        runner = FakeRunner(results=[FakeResult(returncode=1, stderr="error msg")])
        runner._file_actions = [None]
        outcome = run_mount(runner, str(tmp_path), "/s.jsonl", "src", self.UIDS)
        assert not outcome.ok
        assert outcome.kind == "exec_failed"

    def test_subprocess_exception(self, tmp_path):
        runner = FakeRunner(side_effect=RuntimeError("boom"))
        outcome = run_mount(runner, str(tmp_path), "/s.jsonl", "src", self.UIDS)
        assert not outcome.ok
        assert outcome.kind == "exec_failed"


# ---------------------------------------------------------------------------
# run_cascade tests
# ---------------------------------------------------------------------------


def _make_cascade_manifest(
    cascade_dir: Path,
    cost: Optional[float] = 2.30,
    initial: int = 250,
    final: int = 180,
    also_final_corpus: bool = True,
) -> None:
    cascade_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "wellposed_cascade",
        "inputs": {"initial_record_count": initial},
        "overall": {
            "initial_record_count": initial,
            "final_corpus_count": final,
            "total_estimated_cost_usd": cost,
        },
    }
    (cascade_dir / "cascade_manifest.json").write_text(json.dumps(manifest))
    if also_final_corpus:
        rows = "\n".join(json.dumps({"uid": f"u{i}"}) for i in range(final))
        (cascade_dir / "final_corpus.jsonl").write_text(rows)


class TestRunCascade:
    KEY = "/keys/a.env"

    def _happy_runner(self, batch_dir: str, cost: float = 2.30):
        cascade_dir = Path(batch_dir) / "cascade"

        def _action():
            _make_cascade_manifest(cascade_dir, cost=cost)

        runner = FakeRunner(results=[FakeResult(returncode=0)])
        runner._file_actions = [_action]
        return runner

    def test_happy_path(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        outcome = run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        assert outcome.ok
        assert outcome.kind == "cascade_ok"
        assert outcome.data["cost_usd"] == pytest.approx(2.30)

    def test_skip_if_done(self, tmp_path):
        """Manifest already exists → skip re-run."""
        cascade_dir = tmp_path / "cascade"
        _make_cascade_manifest(cascade_dir, cost=1.50)
        runner = FakeRunner()
        outcome = run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        assert outcome.ok
        assert len(runner.calls) == 0
        assert "skipped" in outcome.detail.lower()

    def test_cost_guard_trip_at_5_01(self, tmp_path):
        runner = self._happy_runner(str(tmp_path), cost=5.01)
        outcome = run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        assert not outcome.ok
        assert outcome.kind == "cost_guard_tripped"
        assert outcome.data["cost_usd"] == pytest.approx(5.01)

    def test_cost_guard_ok_at_5_00(self, tmp_path):
        """Exactly at limit is still ok."""
        runner = self._happy_runner(str(tmp_path), cost=5.00)
        outcome = run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        assert outcome.ok

    def test_null_cost_in_flow_testing(self, tmp_path):
        cascade_dir = tmp_path / "cascade"
        _make_cascade_manifest(cascade_dir, cost=None)
        runner = FakeRunner()
        outcome = run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY, mode="flow_testing")
        assert outcome.ok
        assert outcome.data["cost_usd"] == pytest.approx(0.0)
        assert "cost_note" in outcome.data

    def test_missing_final_corpus(self, tmp_path):
        cascade_dir = tmp_path / "cascade"
        _make_cascade_manifest(cascade_dir, also_final_corpus=False)
        runner = FakeRunner()
        outcome = run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        assert not outcome.ok
        assert outcome.kind == "cascade_verification_failed"

    def test_env_carries_key_file(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        assert len(runner.calls) == 1
        env = runner.calls[0]["env"]
        assert env is not None
        assert env["ANTHROPIC_KEY_FILE"] == self.KEY

    def test_argv_shape(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        argv = runner.calls[0]["argv"]
        assert "wellposed-cascade" in argv
        assert "--stages" in argv
        assert argv[argv.index("--stages") + 1] == "codex:anthropic"

    def test_empty_corpus_is_ok_with_flag(self, tmp_path):
        cascade_dir = tmp_path / "cascade"
        _make_cascade_manifest(cascade_dir, cost=1.0, final=0)
        runner = FakeRunner()
        outcome = run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        assert outcome.ok
        assert "empty_corpus_note" in outcome.data

    def test_exec_failure(self, tmp_path):
        runner = FakeRunner(results=[FakeResult(returncode=1, stderr="api error")])
        runner._file_actions = [None]
        outcome = run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        assert not outcome.ok
        assert outcome.kind == "exec_failed"

    def test_calibration_sheet_in_argv(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY,
                    calibration_sheet="/cal.json", mode="flow_testing")
        argv = runner.calls[0]["argv"]
        assert "--calibration-sheet" in argv
        assert argv[argv.index("--calibration-sheet") + 1] == "/cal.json"

    def test_initial_record_count_from_inputs_section(self, tmp_path):
        cascade_dir = tmp_path / "cascade"
        _make_cascade_manifest(cascade_dir, initial=250, final=180)
        runner = FakeRunner()
        outcome = run_cascade(runner, str(tmp_path), "/h.jsonl", self.KEY)
        assert outcome.ok
        assert outcome.data["initial_record_count"] == 250
        assert outcome.data["final_corpus_count"] == 180


# ---------------------------------------------------------------------------
# run_passk tests
# ---------------------------------------------------------------------------


def _make_passk_manifest(
    passk_dir: Path,
    interrupted: bool = False,
    counts: Optional[dict] = None,
) -> None:
    passk_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "pass_at_k",
        "interrupted": interrupted,
        "counts": counts or {"solved": 10, "band": 5},
    }
    (passk_dir / "pass_at_k_manifest.json").write_text(json.dumps(manifest))


class TestRunPassk:
    def _happy_runner(self, batch_dir: str, interrupted: bool = False):
        passk_dir = Path(batch_dir) / "pass_at_k"

        def _action():
            _make_passk_manifest(passk_dir, interrupted=interrupted)

        runner = FakeRunner(results=[FakeResult(returncode=0)])
        runner._file_actions = [_action]
        return runner

    def test_happy_path(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        outcome = run_passk(runner, str(tmp_path))
        assert outcome.ok
        assert outcome.kind == "passk_ok"
        assert "counts" in outcome.data

    def test_slot_busy_blocks_exec(self, tmp_path):
        runner = FakeRunner()
        outcome = run_passk(
            runner, str(tmp_path), slot_checker=lambda: False
        )
        assert not outcome.ok
        assert outcome.kind == "qwen_slot_busy"
        assert len(runner.calls) == 0

    def test_slot_free_allows_exec(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        outcome = run_passk(runner, str(tmp_path), slot_checker=lambda: True)
        assert outcome.ok

    def test_skip_if_done(self, tmp_path):
        passk_dir = tmp_path / "pass_at_k"
        _make_passk_manifest(passk_dir, interrupted=False)
        runner = FakeRunner()
        outcome = run_passk(runner, str(tmp_path))
        assert outcome.ok
        assert outcome.data.get("resumed") is True
        assert len(runner.calls) == 0

    def test_interrupted_resume(self, tmp_path):
        """interrupted=true in manifest means re-run (resume semantics)."""
        passk_dir = tmp_path / "pass_at_k"
        _make_passk_manifest(passk_dir, interrupted=True)
        # Now the runner will produce a completed manifest
        completed_runner = self._happy_runner(str(tmp_path), interrupted=False)
        outcome = run_passk(completed_runner, str(tmp_path))
        assert outcome.ok
        assert len(completed_runner.calls) == 1

    def test_interrupted_manifest_after_run(self, tmp_path):
        """If the completed run still shows interrupted=true → passk_interrupted."""
        runner = self._happy_runner(str(tmp_path), interrupted=True)
        outcome = run_passk(runner, str(tmp_path))
        assert not outcome.ok
        assert outcome.kind == "passk_interrupted"

    def test_exec_failure(self, tmp_path):
        runner = FakeRunner(results=[FakeResult(returncode=1, stderr="fail")])
        runner._file_actions = [None]
        outcome = run_passk(runner, str(tmp_path))
        assert not outcome.ok
        assert outcome.kind == "exec_failed"

    def test_argv_shape(self, tmp_path):
        runner = self._happy_runner(str(tmp_path))
        run_passk(runner, str(tmp_path))
        argv = runner.calls[0]["argv"]
        assert "pass_at_k" in argv
        assert "--backend" in argv
        assert argv[argv.index("--backend") + 1] == "qwen_http"
        assert "--max-concurrent" in argv
        assert argv[argv.index("--max-concurrent") + 1] == "1"

    def test_slot_checker_exception_returns_busy(self, tmp_path):
        def _bad_checker():
            raise RuntimeError("pgrep broke")

        runner = FakeRunner()
        outcome = run_passk(runner, str(tmp_path), slot_checker=_bad_checker)
        assert not outcome.ok
        assert outcome.kind == "qwen_slot_busy"


# ---------------------------------------------------------------------------
# qwen_slot_free tests
# ---------------------------------------------------------------------------


class TestQwenSlotFree:
    def _make_pgrep(self, rc: int, out: str = ""):
        def _runner(argv):
            return FakeResult(returncode=rc, stdout=out)
        return _runner

    def _make_lsof(self, out: str = ""):
        def _runner(argv):
            return FakeResult(returncode=0 if out else 1, stdout=out)
        return _runner

    def test_both_clear_is_free(self):
        result = qwen_slot_free(
            pgrep_runner=self._make_pgrep(rc=1, out=""),
            lsof_runner=self._make_lsof(out=""),
        )
        assert result is True

    def test_pgrep_hit_is_busy(self):
        result = qwen_slot_free(
            pgrep_runner=self._make_pgrep(rc=0, out="12345\n"),
            lsof_runner=self._make_lsof(out=""),
        )
        assert result is False

    def test_lsof_hit_is_busy(self):
        result = qwen_slot_free(
            pgrep_runner=self._make_pgrep(rc=1, out=""),
            lsof_runner=self._make_lsof(out="54321\n"),
        )
        assert result is False

    def test_both_hit_is_busy(self):
        result = qwen_slot_free(
            pgrep_runner=self._make_pgrep(rc=0, out="111\n"),
            lsof_runner=self._make_lsof(out="222\n"),
        )
        assert result is False

    def test_pgrep_exception_returns_false(self):
        def _bad(argv):
            raise OSError("no pgrep")

        result = qwen_slot_free(pgrep_runner=_bad, lsof_runner=self._make_lsof())
        assert result is False

    def test_lsof_exception_returns_false(self):
        def _bad(argv):
            raise OSError("no lsof")

        result = qwen_slot_free(
            pgrep_runner=self._make_pgrep(rc=1),
            lsof_runner=_bad,
        )
        assert result is False

    def test_pgrep_rc0_but_empty_stdout_is_free(self):
        """pgrep rc=0 with empty stdout is edge case — treat as free."""
        result = qwen_slot_free(
            pgrep_runner=self._make_pgrep(rc=0, out=""),
            lsof_runner=self._make_lsof(out=""),
        )
        assert result is True


# ---------------------------------------------------------------------------
# cascade_slot_free tests
# ---------------------------------------------------------------------------


class TestCascadeSlotFree:
    def _make_pgrep(self, rc: int, out: str = ""):
        def _runner(argv):
            return FakeResult(returncode=rc, stdout=out)
        return _runner

    def test_no_match_is_free(self):
        """pgrep rc=1 (no match) → slot is free."""
        result = cascade_slot_free(
            pgrep_runner=self._make_pgrep(rc=1, out=""),
        )
        assert result is True

    def test_pgrep_hit_is_busy(self):
        """pgrep rc=0 with a pid in stdout → slot is busy."""
        result = cascade_slot_free(
            pgrep_runner=self._make_pgrep(rc=0, out="12345\n"),
        )
        assert result is False

    def test_pgrep_exception_returns_false(self):
        """If pgrep itself raises, conservatively report busy."""
        def _bad(argv):
            raise OSError("no pgrep")

        result = cascade_slot_free(pgrep_runner=_bad)
        assert result is False

    def test_pgrep_rc0_but_empty_stdout_is_free(self):
        """pgrep rc=0 with empty stdout is edge case — treat as free (no real process)."""
        result = cascade_slot_free(
            pgrep_runner=self._make_pgrep(rc=0, out=""),
        )
        assert result is True

    def test_pgrep_pattern_targets_wellposed_cascade(self):
        """Verify the pgrep call uses the correct process pattern."""
        captured_argv = []

        def _capture(argv):
            captured_argv.extend(argv)
            return FakeResult(returncode=1, stdout="")

        cascade_slot_free(pgrep_runner=_capture)
        assert "icepick processing wellposed-cascade" in captured_argv


# ---------------------------------------------------------------------------
# with_retries tests
# ---------------------------------------------------------------------------


class TestWithRetries:
    def test_success_first_attempt(self):
        fn = MagicMock(return_value=StageOutcome(ok=True, kind="ok", detail=""))
        outcome = with_retries(fn, max_attempts=3)
        assert outcome.ok
        assert fn.call_count == 1
        assert outcome.data["attempts"] == 1

    def test_retries_on_retryable_failure(self):
        calls = [
            StageOutcome(ok=False, kind="exec_failed", detail="529 overloaded"),
            StageOutcome(ok=False, kind="exec_failed", detail="529 overloaded"),
            StageOutcome(ok=True, kind="ok", detail=""),
        ]
        fn = MagicMock(side_effect=calls)
        slept: list[float] = []
        outcome = with_retries(fn, max_attempts=3, backoffs=(30, 120), sleep_fn=slept.append)
        assert outcome.ok
        assert fn.call_count == 3
        assert slept == [30, 120]
        assert outcome.data["attempts"] == 3

    def test_stops_on_non_retryable(self):
        fn = MagicMock(return_value=StageOutcome(
            ok=False, kind="cost_guard_tripped", detail="cost too high"
        ))
        slept: list[float] = []
        outcome = with_retries(fn, max_attempts=3, sleep_fn=slept.append)
        assert not outcome.ok
        assert fn.call_count == 1
        assert slept == []

    def test_exhaustion_returns_last_outcome(self):
        fn = MagicMock(return_value=StageOutcome(
            ok=False, kind="exec_failed", detail="connection refused"
        ))
        slept: list[float] = []
        outcome = with_retries(fn, max_attempts=3, backoffs=(1, 2), sleep_fn=slept.append)
        assert not outcome.ok
        assert fn.call_count == 3
        assert len(slept) == 2

    def test_backoff_schedule(self):
        fn = MagicMock(return_value=StageOutcome(
            ok=False, kind="exec_failed", detail="timeout error"
        ))
        slept: list[float] = []
        with_retries(fn, max_attempts=4, backoffs=(30, 120, 480), sleep_fn=slept.append)
        assert slept == [30, 120, 480]

    def test_data_attempts_attached(self):
        fn = MagicMock(return_value=StageOutcome(
            ok=False, kind="exec_failed", detail="rate limited"
        ))
        outcome = with_retries(fn, max_attempts=2, backoffs=(1,), sleep_fn=lambda _: None)
        assert outcome.data["attempts"] == 2

    def test_mount_dirty_not_retried(self):
        fn = MagicMock(return_value=StageOutcome(
            ok=False, kind="mount_dirty", detail="dirty"
        ))
        with_retries(fn, max_attempts=3, sleep_fn=lambda _: None)
        assert fn.call_count == 1

    def test_mount_verification_failed_not_retried(self):
        fn = MagicMock(return_value=StageOutcome(
            ok=False, kind="mount_verification_failed", detail="uid mismatch"
        ))
        with_retries(fn, max_attempts=3, sleep_fn=lambda _: None)
        assert fn.call_count == 1

    def test_qwen_slot_busy_not_retried(self):
        fn = MagicMock(return_value=StageOutcome(
            ok=False, kind="qwen_slot_busy", detail="busy"
        ))
        with_retries(fn, max_attempts=3, sleep_fn=lambda _: None)
        assert fn.call_count == 1

    def test_non_transient_exec_failed_not_retried(self):
        """exec_failed without transient marker text is not retried."""
        fn = MagicMock(return_value=StageOutcome(
            ok=False, kind="exec_failed", detail="permission denied"
        ))
        with_retries(fn, max_attempts=3, sleep_fn=lambda _: None)
        assert fn.call_count == 1

    def test_passk_interrupted_retried_when_transient(self):
        calls = [
            StageOutcome(ok=False, kind="passk_interrupted", detail="timeout during rollouts"),
            StageOutcome(ok=True, kind="passk_ok", detail="done"),
        ]
        fn = MagicMock(side_effect=calls)
        slept: list[float] = []
        outcome = with_retries(fn, max_attempts=3, backoffs=(5,), sleep_fn=slept.append)
        assert outcome.ok
        assert fn.call_count == 2
        assert slept == [5]


# ---------------------------------------------------------------------------
# DEFAULT_TRANSIENT_MARKERS
# ---------------------------------------------------------------------------


class TestDefaultTransientMarkers:
    def test_all_markers_present(self):
        for marker in ("529", "overloaded", "timeout", "connection", "rate"):
            assert marker in DEFAULT_TRANSIENT_MARKERS

    def test_retryable_matches_all_markers(self):
        for marker in DEFAULT_TRANSIENT_MARKERS:
            outcome = StageOutcome(
                ok=False, kind="exec_failed", detail=f"Error: {marker} occurred"
            )
            assert _default_is_retryable(outcome), f"Expected {marker!r} to be retryable"

    def test_case_insensitive(self):
        outcome = StageOutcome(
            ok=False, kind="exec_failed", detail="RATE LIMIT HIT"
        )
        assert _default_is_retryable(outcome)


# ---------------------------------------------------------------------------
# StageOutcome dataclass
# ---------------------------------------------------------------------------


class TestStageOutcome:
    def test_defaults(self):
        o = StageOutcome(ok=True, kind="x", detail="y")
        assert o.data == {}

    def test_data_not_shared(self):
        a = StageOutcome(ok=True, kind="x", detail="y")
        b = StageOutcome(ok=True, kind="x", detail="y")
        a.data["k"] = 1
        assert "k" not in b.data

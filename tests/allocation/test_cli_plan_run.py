"""End-to-end CLI tests for `icepick allocation plan` and `allocation run`.

Uses the realmath_scrape adapter in flow_testing mode with the checked-in
QA fixture so no network / API calls occur.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick import cli

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "realmath" / "qa_candidates.jsonl"


def _run_plan(tmp_path, extra_args=()):
    output_dir = tmp_path / "intake"
    argv = [
        "allocation", "plan",
        "--source-type", "realmath_scrape",
        "--source", "test_realmath",
        "--target-count", "5",
        "--output-dir", str(output_dir),
        "--requested-by", "alice",
    ]
    argv.extend(extra_args)
    rc = cli.main(argv)
    return rc, output_dir


def test_cli_plan_writes_proposed_plan_without_auto_approve(tmp_path, capsys):
    rc, output_dir = _run_plan(tmp_path)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["stage"] == "allocation.plan"
    assert out["source_type"] == "realmath_scrape"
    assert out["source_name"] == "test_realmath"
    assert out["manifest"] is None
    assert "review the plan" in out["next"]
    plan_path = Path(out["plan_path"])
    assert plan_path.exists()
    # Filename is timestamp + source stamp
    assert plan_path.name.endswith("_test_realmath_proposed_plan.json")
    plan = json.loads(plan_path.read_text())
    assert plan["source_type"] == "realmath_scrape"
    assert plan["target_count"] == 5
    assert out["estimate"]["expected_handoff_records"] == 5


def test_cli_plan_auto_approve_flow_testing_emits_manifest(tmp_path, capsys):
    rc, output_dir = _run_plan(tmp_path, extra_args=[
        "--auto-approve",
        "--mode", "flow_testing",
        "--calibration-sheet", str(_FIXTURE),
        "--approved-by", "alice",
        "--approval-notes", "flow_testing pilot",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["manifest"] is not None
    manifest_path = Path(out["manifest"]["run_id"])
    # summary carries run_id; actual manifest.json lives under runs/<run_id>/
    actual_manifest = output_dir / "runs" / out["manifest"]["run_id"] / "manifest.json"
    assert actual_manifest.exists()
    assert f"allocation run --manifest {actual_manifest}" in out["next"]
    manifest = json.loads(actual_manifest.read_text())
    assert manifest["processor_mode"] == "flow_testing"
    assert manifest["approved_by"] == "alice"
    assert manifest["source_name"] == "test_realmath"
    assert manifest["calibration_sheet"] == str(_FIXTURE)


def test_cli_plan_auto_approve_production_refused(tmp_path, capsys):
    rc, _ = _run_plan(tmp_path, extra_args=[
        "--auto-approve",
        "--mode", "production",
        "--approved-by", "alice",
    ])
    assert rc == 1  # production auto-approval refused


def test_cli_plan_auto_approve_flow_testing_requires_calibration_sheet(tmp_path):
    rc, _ = _run_plan(tmp_path, extra_args=[
        "--auto-approve",
        "--mode", "flow_testing",
        "--approved-by", "alice",
    ])
    assert rc == 1


def test_cli_plan_auto_approve_requires_mode(tmp_path):
    rc, _ = _run_plan(tmp_path, extra_args=[
        "--auto-approve",
        "--approved-by", "alice",
    ])
    assert rc == 1


def test_cli_plan_auto_approve_requires_approver(tmp_path):
    rc, _ = _run_plan(tmp_path, extra_args=[
        "--auto-approve",
        "--mode", "flow_testing",
        "--calibration-sheet", str(_FIXTURE),
    ])
    assert rc == 1


def test_cli_run_dispatches_to_realmath_scrape_adapter(tmp_path, capsys):
    """Plan+approve, then run — full plumb-through end-to-end via CLI."""
    rc, output_dir = _run_plan(tmp_path, extra_args=[
        "--auto-approve",
        "--mode", "flow_testing",
        "--calibration-sheet", str(_FIXTURE),
        "--approved-by", "alice",
    ])
    assert rc == 0
    plan_summary = json.loads(capsys.readouterr().out)
    manifest_path = output_dir / "runs" / plan_summary["manifest"]["run_id"] / "manifest.json"

    rc = cli.main(["allocation", "run", "--manifest", str(manifest_path)])
    assert rc == 0
    run_out = json.loads(capsys.readouterr().out)
    assert run_out["stage"] == "allocation.run"
    assert run_out["processor_mode"] == "flow_testing"
    assert run_out["calibration_replay"] is True
    assert run_out["counts"]["handoff_records"] > 0
    handoff = Path(run_out["outputs"]["handoff"])
    assert handoff.exists()
    lines = [l for l in handoff.read_text().splitlines() if l.strip()]
    assert len(lines) == run_out["counts"]["handoff_records"]
    # "next" hint points at the processing pipeline
    assert "processing pipeline" in run_out["next"]


def test_cli_run_refuses_unapproved_manifest(tmp_path, capsys):
    """A ProposedPlan on disk (missing approved_by/approved_at) → E_INVALID."""
    rc, output_dir = _run_plan(tmp_path)
    assert rc == 0
    plan_summary = json.loads(capsys.readouterr().out)
    # Feed the proposed_plan.json to `run` — it lacks the approval fields.
    rc = cli.main(["allocation", "run", "--manifest", plan_summary["plan_path"]])
    assert rc == 1

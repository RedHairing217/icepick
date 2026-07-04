"""`icepick allocation approve` — the human gate from proposed plan to manifest.

Turns a proposed_plan.json into an ApprovedManifest so operators (and a
fresh session) never hand-craft manifest JSON. flow_testing here so the
subsequent run replays the fixture with no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from icepick import cli
from icepick.allocation.manifests import load_manifest

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "realmath" / "qa_candidates.jsonl"


def _plan(tmp_path, capsys, extra=()):
    output_dir = tmp_path / "intake"
    argv = [
        "allocation", "plan",
        "--source-type", "realmath_scrape",
        "--source", "pde_2026Q2",
        "--target-count", "5",
        "--category", "math.AP",
        "--output-dir", str(output_dir),
    ]
    argv.extend(extra)
    assert cli.main(argv) == 0
    plan_path = json.loads(capsys.readouterr().out)["plan_path"]
    return output_dir, plan_path


def test_approve_flow_testing_then_run(tmp_path, capsys):
    output_dir, plan_path = _plan(tmp_path, capsys)

    rc = cli.main([
        "allocation", "approve", "--plan", plan_path,
        "--mode", "flow_testing", "--calibration-sheet", str(_FIXTURE),
        "--approved-by", "alice", "--output-dir", str(output_dir),
    ])
    assert rc == 0
    approved = json.loads(capsys.readouterr().out)
    assert approved["stage"] == "allocation.approve"
    assert approved["processor_mode"] == "flow_testing"
    assert f"allocation run --manifest {approved['manifest']}" in approved["next"]

    rc = cli.main(["allocation", "run", "--manifest", approved["manifest"]])
    assert rc == 0
    run_out = json.loads(capsys.readouterr().out)
    assert run_out["calibration_replay"] is True
    assert run_out["counts"]["handoff_records"] > 0
    assert Path(run_out["outputs"]["handoff"]).exists()


def test_approve_production_writes_an_approved_manifest_carrying_the_window(tmp_path, capsys):
    output_dir, plan_path = _plan(tmp_path, capsys, extra=["--primary-only"])

    rc = cli.main([
        "allocation", "approve", "--plan", plan_path,
        "--mode", "production", "--approved-by", "alice",
        "--call-budget", "100", "--output-dir", str(output_dir),
    ])
    assert rc == 0
    manifest_path = json.loads(capsys.readouterr().out)["manifest"]

    manifest = load_manifest(manifest_path)
    assert manifest.is_approved()
    assert manifest.processor_mode == "production"
    assert manifest.requires_calls() is True
    assert manifest.call_budget == 100
    assert manifest.scrape_window == {"category": "math.AP", "primary_only": True}


def test_approve_production_requires_a_call_budget(tmp_path, capsys):
    output_dir, plan_path = _plan(tmp_path, capsys)
    rc = cli.main([
        "allocation", "approve", "--plan", plan_path,
        "--mode", "production", "--approved-by", "alice", "--output-dir", str(output_dir),
    ])
    assert rc == 1  # no --call-budget


def test_approve_production_refuses_a_budget_below_the_estimate(tmp_path, capsys):
    # qa extraction estimates LLM calls, so its budget must be substantial.
    output_dir, plan_path = _plan(tmp_path, capsys, extra=["--extraction", "qa"])
    rc = cli.main([
        "allocation", "approve", "--plan", plan_path,
        "--mode", "production", "--approved-by", "alice",
        "--call-budget", "10", "--output-dir", str(output_dir),
    ])
    assert rc == 1  # qa plan estimates many LLM calls; budget 10 is too low


def test_approve_flow_testing_requires_a_calibration_sheet(tmp_path, capsys):
    output_dir, plan_path = _plan(tmp_path, capsys)
    rc = cli.main([
        "allocation", "approve", "--plan", plan_path,
        "--mode", "flow_testing", "--approved-by", "alice", "--output-dir", str(output_dir),
    ])
    assert rc == 1  # no --calibration-sheet

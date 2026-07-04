"""`allocation plan` records the arXiv scrape window (category, primary-only).

The scrape window is the acquisition filter — e.g. pull only PDE papers
with ``--category math.AP``. It is recorded on the ProposedPlan (and, when
auto-approved, on the manifest) so a run acquires the right papers.
"""

from __future__ import annotations

import json
from pathlib import Path

from icepick import cli

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "realmath" / "qa_candidates.jsonl"


def _run_plan(tmp_path, extra_args):
    output_dir = tmp_path / "intake"
    argv = [
        "allocation", "plan",
        "--source-type", "realmath_scrape",
        "--source", "pde_2026Q2",
        "--target-count", "5",
        "--output-dir", str(output_dir),
        "--requested-by", "alice",
    ]
    argv.extend(extra_args)
    rc = cli.main(argv)
    return rc, output_dir


def _plan_json(capsys):
    out = json.loads(capsys.readouterr().out)
    return out, json.loads(Path(out["plan_path"]).read_text())


def test_plan_records_pde_category_and_primary_only(tmp_path, capsys):
    rc, _ = _run_plan(tmp_path, [
        "--category", "math.AP",
        "--primary-only",
        "--year", "2026",
        "--month", "4",
        "--max-papers", "200",
        "--family", "pde",
    ])
    assert rc == 0
    _, plan = _plan_json(capsys)
    assert plan["scrape_window"] == {
        "category": "math.AP",
        "year": 2026,
        "month": 4,
        "max_papers": 200,
        "primary_only": True,
    }
    assert plan["families"] == ["pde"]


def test_plan_without_window_flags_records_no_window(tmp_path, capsys):
    rc, _ = _run_plan(tmp_path, [])
    assert rc == 0
    _, plan = _plan_json(capsys)
    assert plan["scrape_window"] is None


def test_plan_category_only_records_just_the_category(tmp_path, capsys):
    rc, _ = _run_plan(tmp_path, ["--category", "math.AP"])
    assert rc == 0
    _, plan = _plan_json(capsys)
    assert plan["scrape_window"] == {"category": "math.AP"}


def test_plan_records_latex_extraction_mode(tmp_path, capsys):
    rc, _ = _run_plan(tmp_path, ["--category", "math.AP", "--extraction", "latex"])
    assert rc == 0
    _, plan = _plan_json(capsys)
    assert plan["scrape_window"] == {"category": "math.AP", "extraction": "latex"}


def test_plan_records_qa_extraction_mode(tmp_path, capsys):
    rc, _ = _run_plan(tmp_path, ["--category", "math.AP", "--extraction", "qa"])
    assert rc == 0
    _, plan = _plan_json(capsys)
    assert plan["scrape_window"] == {"category": "math.AP", "extraction": "qa"}


def test_auto_approved_manifest_carries_the_scrape_window(tmp_path, capsys):
    rc, output_dir = _run_plan(tmp_path, [
        "--category", "math.AP",
        "--primary-only",
        "--auto-approve",
        "--mode", "flow_testing",
        "--calibration-sheet", str(_FIXTURE),
        "--approved-by", "alice",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    manifest_path = output_dir / "runs" / out["manifest"]["run_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["scrape_window"]["category"] == "math.AP"
    assert manifest["scrape_window"]["primary_only"] is True

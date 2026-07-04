"""End-to-end CLI for `icepick allocation mount` and `validate-manifest`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick import cli


def _seed_csv(tmp_path: Path) -> Path:
    src = tmp_path / "src.csv"
    src.write_text("question,gold,arxiv\nq1,a1,2403.11111\nq2,a2,2403.22222\n")
    return src


def test_cli_mount_with_column_args(tmp_path, capsys):
    src = _seed_csv(tmp_path)
    output_dir = tmp_path / "intake"

    rc = cli.main(
        [
            "allocation", "mount",
            "--path", str(src),
            "--source", "csv_batch",
            "--provenance", "external",
            "--truth-policy", "unknown",
            "--output-dir", str(output_dir),
            "--column", "statement=question",
            "--column", "answer=gold",
            "--column", "arxiv_id=arxiv",
            "--requested-by", "alice",
        ]
    )
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["stage"] == "allocation.mount"
    assert out["record_count"] == 2
    handoff = Path(out["outputs"]["handoff"])
    manifest = Path(out["outputs"]["manifest"])
    assert handoff.exists()
    assert manifest.exists()

    # The "next" hint points at the pipeline command.
    assert "processing pipeline" in out["next"]


def test_cli_validate_manifest_after_mount(tmp_path, capsys):
    src = tmp_path / "src.jsonl"
    src.write_text(json.dumps({"statement": "q", "arxiv_id": "2403.11111"}) + "\n")
    output_dir = tmp_path / "intake"

    cli.main([
        "allocation", "mount",
        "--path", str(src),
        "--source", "s",
        "--provenance", "manual",
        "--output-dir", str(output_dir),
        "--requested-by", "alice",
    ])
    mount_out = json.loads(capsys.readouterr().out)
    manifest_path = mount_out["outputs"]["manifest"]

    rc = cli.main([
        "allocation", "validate-manifest",
        "--manifest", manifest_path,
    ])
    assert rc == 0
    val = json.loads(capsys.readouterr().out)
    assert val["status"] == "approved"
    assert val["requires_calls"] is False


def test_cli_mount_rejects_malformed_column_spec(tmp_path):
    src = _seed_csv(tmp_path)
    rc = cli.main(
        [
            "allocation", "mount",
            "--path", str(src),
            "--source", "s",
            "--provenance", "external",
            "--output-dir", str(tmp_path / "intake"),
            "--column", "no_equals_sign",  # malformed
        ]
    )
    assert rc == 1


def test_cli_mount_to_pipeline_smoke(tmp_path, monkeypatch, capsys):
    """Mount a CSV, then run the pipeline on the handoff. End-to-end shape check."""
    src = _seed_csv(tmp_path)
    intake_out = tmp_path / "intake"

    cli.main([
        "allocation", "mount",
        "--path", str(src),
        "--source", "csv",
        "--provenance", "external",
        "--output-dir", str(intake_out),
        "--column", "statement=question",
        "--column", "answer=gold",
        "--column", "arxiv_id=arxiv",
    ])
    mount_summary = json.loads(capsys.readouterr().out)
    handoff_path = mount_summary["outputs"]["handoff"]

    # Substitute fake adapters in the pipeline so no Anthropic / poser
    # subprocess actually runs.
    from icepick.processing.groundtruth import runner as gt_runner_mod
    from icepick.processing.groundtruth.base import GroundtruthVerdict, STATUS_PUBLISHED
    from icepick.processing.poser import runner as poser_runner_mod
    from icepick.processing.poser.base import (
        PoserRequest, PoserRunResult, PoserVerdict, STATUS_WELL_POSED,
    )

    class _FakeGT:
        def lookup_paper(self, *, arxiv_id, paper_title, uid_for_error_attribution):
            return GroundtruthVerdict(
                uid=uid_for_error_attribution, source="", verdict_status=STATUS_PUBLISHED,
                arxiv_id=arxiv_id, judge_model="fake", judge_votes=["published"] * 3,
                judge_majority="published", reasoning="x", confidence="high",
            )

    class _FakePoser:
        build = "claude"
        def plan(self, records, cfg, combo, work_dir):
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            input_path = Path(work_dir) / f"{combo.slug()}_input.jsonl"
            with input_path.open("w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
            return PoserRequest(
                argv=["fake"], env={}, input_path=input_path,
                output_path=Path(work_dir) / f"{combo.slug()}_out.json",
                cache_path=None, poser_name=combo.key(),
            )
        def run(self, request):
            return PoserRunResult(exit_code=0, stdout="", stderr="",
                                  output_path=request.output_path, wall_clock_seconds=0.01)
        def normalise(self, raw_output_path, input_uids, *, combo):
            return [
                PoserVerdict(uid=u, source="", verdict_status=STATUS_WELL_POSED,
                             verdict_score=1.0, poser_name=combo.key(), poser_model="fake")
                for u in input_uids
            ]

    monkeypatch.setattr(gt_runner_mod, "_build_adapter", lambda cfg: _FakeGT())
    monkeypatch.setattr(poser_runner_mod, "ClaudePoserAdapter", _FakePoser)

    pipeline_out = tmp_path / "pipeline"
    rc = cli.main([
        "processing", "pipeline",
        "--mode", "flow_testing",
        "--calibration-sheet", str(tmp_path / "sheet.jsonl"),
        "--input", handoff_path,
        "--output-dir", str(pipeline_out),
        "--combo", "claude:anthropic",
        "--no-judge",
    ])
    assert rc == 0
    final = pipeline_out / "final_corpus.jsonl"
    assert final.exists()
    final_records = [json.loads(l) for l in final.read_text().splitlines() if l.strip()]
    assert len(final_records) == 2  # both CSV rows survived both stages

"""End-to-end CLI for `icepick processing groundtruth`.

Uses monkeypatching to substitute a fake adapter — no real Anthropic
calls anywhere in this test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick import cli
from icepick.processing.groundtruth.base import (
    STATUS_PUBLISHED,
    STATUS_UNPUBLISHED,
    GroundtruthVerdict,
)


class _FakeAdapter:
    def lookup_paper(self, *, arxiv_id, paper_title, uid_for_error_attribution):
        status = STATUS_PUBLISHED if arxiv_id.endswith("11111") else STATUS_UNPUBLISHED
        return GroundtruthVerdict(
            uid=uid_for_error_attribution, source="",
            verdict_status=status, arxiv_id=arxiv_id,
            judge_model="fake-model", judge_votes=[status] * 3,
            judge_majority=status, reasoning="fake", confidence="high",
        )


def _seed_input(tmp_path) -> Path:
    records = [
        {"source": "realmath", "statement": "X", "arxiv_id": "2403.11111",
         "provenance": "extracted", "uid": "uid_a"},
        {"source": "realmath", "statement": "Y", "arxiv_id": "2403.22222",
         "provenance": "extracted", "uid": "uid_b"},
    ]
    path = tmp_path / "records.jsonl"
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def test_cli_runs_end_to_end_with_fake_adapter(tmp_path, monkeypatch, capsys):
    input_path = _seed_input(tmp_path)
    output_dir = tmp_path / "out"

    # Substitute the adapter at the runner build site.
    from icepick.processing.groundtruth import runner as runner_mod
    monkeypatch.setattr(runner_mod, "_build_adapter", lambda cfg: _FakeAdapter())

    rc = cli.main(
        [
            "processing", "groundtruth",
            "--mode", "flow_testing",
            "--calibration-sheet", str(tmp_path / "sheet.jsonl"),
            "--input", str(input_path),
            "--output-dir", str(output_dir),
        ]
    )
    assert rc == 0

    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "verdicts.jsonl").exists()
    assert (output_dir / "published.jsonl").exists()
    assert (output_dir / "discarded.jsonl").exists()

    pubs = [json.loads(l) for l in (output_dir / "published.jsonl").read_text().splitlines() if l.strip()]
    assert {p["uid"] for p in pubs} == {"uid_a"}

    out = json.loads(capsys.readouterr().out)
    assert out["stage"] == "groundtruth"
    assert out["counts"][STATUS_PUBLISHED] == 1
    assert out["counts"][STATUS_UNPUBLISHED] == 1


def test_cli_rejects_missing_mode(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([
            "processing", "groundtruth",
            "--input", str(tmp_path / "x.jsonl"),
            "--output-dir", str(tmp_path),
        ])


def test_cli_rejects_production_without_key_file(tmp_path):
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps({"source": "s", "statement": "q"}) + "\n")
    rc = cli.main(
        [
            "processing", "groundtruth",
            "--mode", "production",
            "--input", str(input_path),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 1

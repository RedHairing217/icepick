"""End-to-end check on the only working CLI command: processing ingest-check."""

from __future__ import annotations

import json

from icepick.cli import main


def test_ingest_check_runs_and_writes_summary(mixed_jsonl, tmp_path, capsys):
    out = tmp_path / "out"
    rc = main(
        [
            "processing",
            "ingest-check",
            "--input",
            str(mixed_jsonl),
            "mixed",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    summary_path = out / "ingest_check_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["run"]["total_records"] == 2
    by_source = summary["counts"]["by_source"]["mixed"]
    assert by_source["records"] == 2
    assert by_source["computed"] == 1
    assert by_source["extracted"] == 1
    assert by_source["uid_collisions"] == 0


def test_unwired_stage_fails_cleanly(tmp_path, capsys):
    """The remaining stub (stage-tests) must surface a clean error code,
    not crash. This catches accidental wiring regressions on stubs.
    """
    rc = main(
        [
            "processing",
            "stage-tests",
            "--mode",
            "production",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "E_NOT_IMPLEMENTED" in err

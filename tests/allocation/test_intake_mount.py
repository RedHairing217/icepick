"""Intake-level mount: produces handoff JSONL + auto-approved manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from icepick.allocation.intake import mount as intake_mount
from icepick.allocation.manifests import load_manifest
from icepick.contracts.manifests import SOURCE_MANUAL_MOUNT


def _seed_jsonl(tmp_path: Path, n: int = 2) -> Path:
    src = tmp_path / "src.jsonl"
    with src.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "statement": f"q{i}",
                "answer": str(i),
                "arxiv_id": f"2403.{10000 + i:05d}",
            }) + "\n")
    return src


def test_mount_writes_handoff_and_manifest_at_expected_paths(tmp_path):
    src = _seed_jsonl(tmp_path)
    now = datetime(2026, 6, 30, 14, 0, 0, tzinfo=timezone.utc)
    outcome = intake_mount(
        path=src, source="cust_2026Q2", provenance="manual",
        requested_by="alice", output_dir=tmp_path / "intake", now=now,
    )

    expected_run_id = "20260630T140000Z"
    assert outcome.run_id == expected_run_id
    assert outcome.handoff_path == tmp_path / "intake" / "runs" / expected_run_id / "handoff" / "records.jsonl"
    assert outcome.manifest_path == tmp_path / "intake" / "runs" / expected_run_id / "manifest.json"
    assert outcome.handoff_path.exists()
    assert outcome.manifest_path.exists()
    assert outcome.record_count == 2


def test_mount_manifest_is_auto_approved_with_zero_call_budget(tmp_path):
    src = _seed_jsonl(tmp_path)
    outcome = intake_mount(
        path=src, source="s", provenance="manual",
        requested_by="alice", output_dir=tmp_path / "intake",
    )
    manifest = load_manifest(outcome.manifest_path)
    assert manifest.source_type == SOURCE_MANUAL_MOUNT
    assert manifest.approved_by == "alice"
    assert manifest.approved_at  # non-empty
    assert manifest.call_budget == 0
    assert manifest.is_approved()
    assert manifest.requires_calls() is False
    assert manifest.target_count == 2


def test_mount_records_are_pipeline_ready(tmp_path):
    """The handoff JSONL should drop into pipeline --input unchanged."""
    src = _seed_jsonl(tmp_path)
    outcome = intake_mount(
        path=src, source="src1", provenance="manual",
        truth_policy="unknown", requested_by="alice",
        output_dir=tmp_path / "intake",
    )
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    for r in records:
        assert "statement" in r
        assert "source" in r
        assert "provenance" in r
        assert "truth_policy" in r
        assert "arxiv_id" in r  # downstream groundtruth needs this


def test_mount_csv_with_column_map_via_intake(tmp_path):
    src = tmp_path / "src.csv"
    src.write_text("question,gold,arxiv\nq1,a1,2403.11111\n")
    outcome = intake_mount(
        path=src, source="csv_batch", provenance="external",
        requested_by="alice", output_dir=tmp_path / "intake",
        column_map={"statement": "question", "answer": "gold", "arxiv_id": "arxiv"},
    )
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert len(records) == 1
    assert records[0]["statement"] == "q1"
    assert records[0]["arxiv_id"] == "2403.11111"
    assert records[0]["provenance"] == "external"

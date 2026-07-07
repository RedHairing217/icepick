"""Deterministic flow-testing replay: no chunks, no calls, full run layout.

The fixture (``tests/fixtures/arxiv_bulk/qa_candidates.jsonl``) carries nine
upstream-shaped candidates: six clean ones, a v1/v2 repost pair (same statement,
collapsed by statement dedup), one explicitly computed row, a row with no link
or arxiv_id, and a same-title repost pair exercising the paper-pool title dedup.
Replay needs no manifest_path and spends nothing, so it is auto-approvable by
its creator.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone

import pytest

from icepick.allocation.adapters import arxiv_bulk
from icepick.allocation.manifests import load_manifest
from icepick.contracts.manifests import ApprovedManifest, SOURCE_ARXIV_BULK

FIXTURE = "arxiv_bulk/qa_candidates.jsonl"
NOW = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during flow_testing replay")

    monkeypatch.setattr(socket, "socket", _blocked)


def _manifest(output_dir, fixture, **overrides):
    base = dict(
        run_id="20260706T120000Z",
        source_type=SOURCE_ARXIV_BULK,
        processor_mode="flow_testing",
        requested_by="alice",
        requested_at="2026-07-06T00:00:00Z",
        approved_by="alice",
        approved_at="2026-07-06T00:00:00Z",
        source_name="arxiv_bulk_2025Q1",
        target_count=5,
        call_budget=0,
        judge_enabled=False,
        confirmation_enabled=False,
        enable_leakage=False,
        enable_duplication=True,
        enable_robustness=False,
        output_dir=str(output_dir),
        calibration_sheet=str(fixture),
        approval_notes="auto-approved: flow_testing replay spends no calls",
    )
    base.update(overrides)
    return ApprovedManifest(**base)


def _run(tmp_path, fixtures_dir, subdir="intake", **overrides):
    manifest = _manifest(tmp_path / subdir, fixtures_dir / FIXTURE, **overrides)
    return arxiv_bulk.run(manifest, now=NOW)


def test_replay_writes_the_documented_run_layout(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    run_dir = tmp_path / "intake" / "runs" / "20260706T120000Z"
    assert outcome.handoff_path == run_dir / "handoff" / "records.jsonl"
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "handoff" / "records.jsonl").is_file()
    assert (run_dir / "raw" / "papers.jsonl").is_file()
    assert (run_dir / "raw" / "extracted_candidates.jsonl").is_file()
    assert (run_dir / "raw" / "qa_candidates.jsonl").is_file()
    assert (run_dir / "reports" / "source_report.md").is_file()


def test_replay_creates_no_progress_or_chunk_dirs(tmp_path, fixtures_dir):
    """flow_testing touches no S3 chunk store and no checkpoint."""
    outcome = _run(tmp_path, fixtures_dir)
    run_dir = tmp_path / "intake" / "runs" / "20260706T120000Z"
    assert outcome.interrupted is False
    assert not (run_dir / "_progress").exists()
    assert not (run_dir / "_chunks").exists()
    assert outcome.acquisition is None  # replay spends nothing


def test_replay_is_deterministic(tmp_path, fixtures_dir):
    first = _run(tmp_path, fixtures_dir, subdir="first")
    second = _run(tmp_path, fixtures_dir, subdir="second")
    assert first.handoff_path.read_bytes() == second.handoff_path.read_bytes()
    assert first.record_count == second.record_count == 8


def test_replay_never_mutates_the_fixture(tmp_path, fixtures_dir):
    fixture = fixtures_dir / FIXTURE
    before = fixture.read_bytes()
    _run(tmp_path, fixtures_dir)
    assert fixture.read_bytes() == before


def test_replay_counts_papers_candidates_dupes_and_drops(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    assert outcome.candidate_count == 9
    # v1/v2 repost collapses by statement dedup; the same-title pair by title.
    assert outcome.paper_count == 6
    assert outcome.duplicates_dropped == 1
    assert outcome.record_count == 8
    assert outcome.calibration_replay is True
    assert any("duplicate paper titles" in w for w in outcome.warnings)


def test_replay_records_are_canonical_and_stamped_as_replay(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert len(records) == 8
    for record in records:
        assert record["source"] == "arxiv_bulk_2025Q1"
        assert record["statement"]
        assert record["provenance"] in ("extracted", "computed")
        assert record["truth_policy"]
        assert record["metadata"]["calibration_replay"] is True
    # The computed candidate is handed off honestly, not relabelled.
    assert [r["provenance"] for r in records].count("computed") == 1


def test_replay_stamps_the_family_when_one_is_requested(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir, families=["pde"])
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert all(r["family"] == "pde" for r in records)


def test_replay_manifest_roundtrips_through_the_loader(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    manifest = load_manifest(outcome.manifest_path)
    assert manifest.is_approved()
    assert manifest.source_type == SOURCE_ARXIV_BULK
    assert manifest.processor_mode == "flow_testing"


def test_replay_report_shows_counts_and_handoff_path(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    report = outcome.report_path.read_text()
    assert report.startswith("# arXiv bulk source report")  # honest, source-stamped title
    assert str(outcome.handoff_path) in report
    assert "calibration_replay: true" in report
    assert "| handoff records | 8 |" in report
    assert "no arxiv_id" in report  # warning surfaced for the operator


def test_replay_refuses_a_missing_fixture(tmp_path, fixtures_dir):
    manifest = _manifest(tmp_path / "intake", tmp_path / "nope.jsonl")
    with pytest.raises(FileNotFoundError):
        arxiv_bulk.run(manifest, now=NOW)


def test_replay_needs_no_manifest_path(tmp_path, fixtures_dir):
    """A flow_testing manifest with no scrape_window at all still replays."""
    outcome = _run(tmp_path, fixtures_dir, scrape_window=None)
    assert outcome.record_count == 8

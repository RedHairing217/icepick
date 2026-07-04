"""Deterministic flow-testing replay: no scraping, no calls, full run layout.

The fixture (``tests/fixtures/realmath/qa_candidates.jsonl``) carries nine
upstream-shaped candidates: six clean ones, a duplicate statement, a row
with no statement, one explicitly computed row, and a same-title repost
pair exercising the paper-pool title dedup.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone

import pytest

from icepick.allocation.adapters import realmath_scrape
from icepick.allocation.manifests import load_manifest
from icepick.contracts.manifests import ApprovedManifest, SOURCE_REALMATH_SCRAPE

FIXTURE = "realmath/qa_candidates.jsonl"
NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during flow_testing replay")

    monkeypatch.setattr(socket, "socket", _blocked)


def _manifest(output_dir, fixture, **overrides):
    base = dict(
        run_id="20260701T120000Z",
        source_type=SOURCE_REALMATH_SCRAPE,
        processor_mode="flow_testing",
        requested_by="alice",
        requested_at="2026-07-01T00:00:00Z",
        approved_by="alice",
        approved_at="2026-07-01T00:00:00Z",
        source_name="realmath_2026Q2",
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
    return realmath_scrape.run(manifest, now=NOW)


def test_replay_writes_the_documented_run_layout(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    run_dir = tmp_path / "intake" / "runs" / "20260701T120000Z"
    assert outcome.handoff_path == run_dir / "handoff" / "records.jsonl"
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "handoff" / "records.jsonl").is_file()
    assert (run_dir / "raw" / "papers.jsonl").is_file()
    assert (run_dir / "raw" / "extracted_candidates.jsonl").is_file()
    assert (run_dir / "raw" / "qa_candidates.jsonl").is_file()
    assert (run_dir / "raw" / "quarantined.jsonl").is_file()
    assert (run_dir / "reports" / "source_report.md").is_file()


def test_replay_is_deterministic(tmp_path, fixtures_dir):
    first = _run(tmp_path, fixtures_dir, subdir="first")
    second = _run(tmp_path, fixtures_dir, subdir="second")
    assert first.handoff_path.read_bytes() == second.handoff_path.read_bytes()
    assert first.record_count == second.record_count == 7


def test_replay_never_mutates_the_fixture(tmp_path, fixtures_dir):
    fixture = fixtures_dir / FIXTURE
    before = fixture.read_bytes()
    _run(tmp_path, fixtures_dir)
    assert fixture.read_bytes() == before


def test_replay_counts_papers_candidates_dupes_and_drops(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    assert outcome.candidate_count == 9
    # v1/v2 repost collapses by arxiv id; the same-title pair by title dedup.
    assert outcome.paper_count == 6
    assert outcome.duplicates_dropped == 1
    assert outcome.quarantined_count == 1
    assert outcome.record_count == 7
    assert outcome.calibration_replay is True
    assert any("duplicate paper titles" in w for w in outcome.warnings)


def test_replay_records_are_canonical_and_stamped_as_replay(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert len(records) == 7
    for record in records:
        assert record["source"] == "realmath_2026Q2"
        assert record["statement"]
        assert record["provenance"] in ("extracted", "computed")
        assert record["truth_policy"]
        assert record["family"] == "realmath"
        assert record["metadata"]["calibration_replay"] is True
    # The computed candidate is handed off honestly, not relabelled.
    assert [r["provenance"] for r in records].count("computed") == 1


def test_replay_manifest_roundtrips_through_the_manifest_loader(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    manifest = load_manifest(outcome.manifest_path)
    assert manifest.is_approved()
    assert manifest.source_type == SOURCE_REALMATH_SCRAPE
    assert manifest.processor_mode == "flow_testing"


def test_replay_report_shows_counts_drops_warnings_and_handoff_path(tmp_path, fixtures_dir):
    outcome = _run(tmp_path, fixtures_dir)
    report = outcome.report_path.read_text()
    assert str(outcome.handoff_path) in report
    assert "calibration_replay: true" in report
    assert "| handoff records | 7 |" in report
    assert "| quarantined | 1 |" in report
    assert "missing statement" in report
    assert "no arxiv_id" in report  # warning surfaced for the operator


def test_replay_refuses_a_missing_fixture(tmp_path, fixtures_dir):
    manifest = _manifest(tmp_path / "intake", tmp_path / "nope.jsonl")
    with pytest.raises(FileNotFoundError):
        realmath_scrape.run(manifest, now=NOW)


def test_rerun_clears_a_stale_quarantine_file(tmp_path, fixtures_dir):
    """Re-running the same run_id with a clean fixture must not leave the
    previous run's quarantine file claiming drops that never happened."""
    first = _run(tmp_path, fixtures_dir)
    assert (first.raw_dir / "quarantined.jsonl").is_file()

    clean_fixture = tmp_path / "clean.jsonl"
    rows = (fixtures_dir / FIXTURE).read_text().splitlines()
    clean_fixture.write_text(
        "\n".join(row for row in rows if json.loads(row).get("question")) + "\n"
    )
    manifest = _manifest(tmp_path / "intake", clean_fixture)
    second = realmath_scrape.run(manifest, now=NOW)
    assert second.quarantined_count == 0
    assert not (second.raw_dir / "quarantined.jsonl").exists()

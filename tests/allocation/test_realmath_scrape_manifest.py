"""Manifest gates for realmath_scrape runs: fail closed, never guess."""

from __future__ import annotations

import pytest

from icepick.allocation.adapters import realmath_scrape
from icepick.contracts.manifests import ApprovedManifest, SOURCE_REALMATH_SCRAPE


def _manifest(tmp_path, **overrides):
    base = dict(
        run_id="20260701T000000Z",
        source_type=SOURCE_REALMATH_SCRAPE,
        processor_mode="production",
        requested_by="alice",
        requested_at="2026-07-01T00:00:00Z",
        approved_by="bob",
        approved_at="2026-07-01T01:00:00Z",
        source_name="realmath_2026Q2",
        target_count=5,
        call_budget=1000,
        judge_enabled=False,
        confirmation_enabled=False,
        enable_leakage=False,
        enable_duplication=True,
        enable_robustness=False,
        truth_policy="extracted",
        output_dir=str(tmp_path / "intake"),
    )
    base.update(overrides)
    return ApprovedManifest(**base)


def test_unapproved_production_run_is_refused(tmp_path):
    manifest = _manifest(tmp_path, approved_by="", approved_at="")
    with pytest.raises(ValueError, match="not approved"):
        realmath_scrape.run(manifest)


def test_unapproved_flow_testing_run_is_refused(tmp_path):
    """Runs execute only from an approved manifest — replay included."""
    manifest = _manifest(
        tmp_path, processor_mode="flow_testing",
        calibration_sheet="fixture.jsonl", approved_by="", approved_at="",
    )
    with pytest.raises(ValueError, match="not approved"):
        realmath_scrape.run(manifest)


def test_foreign_source_type_is_refused(tmp_path):
    manifest = _manifest(tmp_path, source_type="manual_mount")
    with pytest.raises(ValueError, match="source_type"):
        realmath_scrape.run(manifest)


def test_unknown_processor_mode_is_refused(tmp_path):
    manifest = _manifest(tmp_path, processor_mode="dry_run")
    with pytest.raises(ValueError, match="processor_mode"):
        realmath_scrape.run(manifest)


def test_missing_call_budget_is_refused(tmp_path):
    manifest = _manifest(tmp_path, call_budget=None)
    with pytest.raises(ValueError, match="call_budget"):
        realmath_scrape.run(manifest)


def test_exceeded_call_budget_is_refused_before_any_work(tmp_path):
    manifest = _manifest(tmp_path, target_count=500, call_budget=10)
    with pytest.raises(ValueError, match="call_budget"):
        realmath_scrape.run(manifest)


def test_missing_output_dir_is_refused(tmp_path):
    manifest = _manifest(tmp_path, output_dir=None)
    with pytest.raises(ValueError, match="output_dir"):
        realmath_scrape.run(manifest)


def test_run_id_escaping_the_output_dir_is_refused(tmp_path):
    manifest = _manifest(tmp_path, run_id="../../escape")
    with pytest.raises(ValueError, match="escapes"):
        realmath_scrape.run(manifest)


def test_degenerate_run_ids_breaking_the_layout_are_refused(tmp_path):
    for run_id in (".", "a/b", "runs/../rid2"):
        manifest = _manifest(tmp_path, run_id=run_id)
        with pytest.raises(ValueError, match="layout"):
            realmath_scrape.run(manifest)


def test_flow_testing_without_a_fixture_is_refused(tmp_path):
    manifest = _manifest(tmp_path, processor_mode="flow_testing", calibration_sheet=None)
    with pytest.raises(ValueError, match="calibration_sheet"):
        realmath_scrape.run(manifest)


def test_unknown_truth_policy_is_refused(tmp_path):
    manifest = _manifest(tmp_path, truth_policy="verified_by_vibes")
    with pytest.raises(ValueError, match="truth_policy"):
        realmath_scrape.run(manifest)


def test_approved_production_run_scrapes_in_house(tmp_path, monkeypatch):
    """Gates pass, then production reaches the in-house scraper (fetcher injected, no network)."""
    from icepick.allocation.scrape import realmath as realmath_source

    empty_feed = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    monkeypatch.setattr(
        realmath_source, "default_arxiv_fetcher",
        lambda query, *, start, max_results: empty_feed,
    )
    manifest = _manifest(tmp_path, scrape_window={"category": "math.AP"})
    outcome = realmath_scrape.run(manifest)
    assert outcome.calibration_replay is False
    assert outcome.processor_mode == "production"
    assert outcome.handoff_path.exists()  # gates passed, run layout written

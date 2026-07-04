"""Estimates describe the work without performing any of it."""

from __future__ import annotations

import socket

import pytest

from icepick.allocation.adapters import realmath_scrape
from icepick.contracts.manifests import SOURCE_REALMATH_SCRAPE


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during estimation")

    monkeypatch.setattr(socket, "socket", _blocked)


def _plan(**overrides):
    request = dict(
        source_name="realmath_2026Q2",
        target_count=500,
        requested_by="alice",
        requested_at="2026-07-01T00:00:00Z",
    )
    request.update(overrides)
    return realmath_scrape.plan(request)


def test_estimate_describes_the_expected_work():
    estimate = realmath_scrape.estimate(_plan())
    assert estimate["source_type"] == SOURCE_REALMATH_SCRAPE
    assert estimate["expected_papers"] >= estimate["target_count"]
    assert estimate["expected_candidates"] >= estimate["expected_papers"]
    assert estimate["expected_handoff_records"] == 500
    assert estimate["estimated_calls"] > 0
    assert estimate["call_kinds"]
    assert estimate["local_prerequisites"]


def test_estimate_is_conservative_about_paper_volume():
    """More papers than records: the scrape yield is well below 1:1."""
    estimate = realmath_scrape.estimate(_plan(target_count=10))
    assert estimate["expected_papers"] > estimate["expected_handoff_records"]


def test_estimate_matches_the_plan_estimated_calls():
    plan = _plan(target_count=25)
    assert realmath_scrape.estimate(plan)["estimated_calls"] == plan.estimated_calls


def test_estimate_is_extraction_aware():
    """qa mode budgets LLM calls; abstract does not — the budget must reflect it."""
    abstract = realmath_scrape.estimate(_plan(target_count=10))
    latex = realmath_scrape.estimate(_plan(target_count=10, scrape_window={"extraction": "latex"}))
    qa = realmath_scrape.estimate(_plan(target_count=10, scrape_window={"extraction": "qa"}))

    assert abstract["expected_llm_calls"] == 0
    assert "qa_generation" not in abstract["call_kinds"]
    assert "latex_source_fetch" in latex["call_kinds"]
    assert qa["expected_llm_calls"] > 0
    assert "qa_generation" in qa["call_kinds"]
    # More work per mode ⇒ a higher budgeted call count.
    assert qa["estimated_calls"] > latex["estimated_calls"] > abstract["estimated_calls"]


def test_qa_plan_budget_gate_covers_llm_calls():
    """A production qa manifest whose budget covers arXiv but not the LLM calls is refused."""
    from icepick.contracts.manifests import ApprovedManifest

    qa_plan = _plan(target_count=10, scrape_window={"extraction": "qa"})
    manifest = ApprovedManifest(
        run_id="r", source_type="realmath_scrape", processor_mode="production",
        requested_by="a", requested_at="t", approved_by="b", approved_at="t",
        source_name="s", target_count=10, call_budget=5,  # covers arXiv, not the LLM calls
        judge_enabled=False, confirmation_enabled=False, enable_leakage=False,
        enable_duplication=False, enable_robustness=False,
        scrape_window={"extraction": "qa"}, output_dir="out",
    )
    with pytest.raises(ValueError, match="call_budget"):
        realmath_scrape.run(manifest)
    assert qa_plan.estimated_calls > 5


def test_estimate_refuses_foreign_source_types():
    plan = _plan()
    plan.source_type = "manual_mount"
    with pytest.raises(ValueError, match="source_type"):
        realmath_scrape.estimate(plan)


def test_estimate_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    realmath_scrape.estimate(_plan())
    assert list(tmp_path.iterdir()) == []

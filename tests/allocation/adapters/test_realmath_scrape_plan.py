"""Plan creation is pure: no calls, no scraping, no writes."""

from __future__ import annotations

import socket

import pytest

from icepick.allocation.adapters import realmath_scrape
from icepick.contracts.manifests import SOURCE_REALMATH_SCRAPE, ProposedPlan


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during plan creation")

    monkeypatch.setattr(socket, "socket", _blocked)


def _request(**overrides):
    base = dict(
        source_name="realmath_2026Q2",
        target_count=500,
        requested_by="alice",
        requested_at="2026-07-01T00:00:00Z",
    )
    base.update(overrides)
    return base


def test_plan_builds_a_proposed_plan():
    plan = realmath_scrape.plan(_request())
    assert isinstance(plan, ProposedPlan)
    assert plan.source_type == SOURCE_REALMATH_SCRAPE
    assert plan.source_name == "realmath_2026Q2"
    assert plan.target_count == 500
    assert plan.requested_by == "alice"


def test_plan_estimates_calls_scaling_with_target_count():
    small = realmath_scrape.plan(_request(target_count=10))
    large = realmath_scrape.plan(_request(target_count=100))
    assert small.estimated_calls > 0
    assert large.estimated_calls > small.estimated_calls


def test_plan_records_families_and_scrape_window():
    plan = realmath_scrape.plan(
        _request(
            families=["number_theory"],
            scrape_window={"year": 2026, "month": 4, "category": "math.NT"},
        )
    )
    assert plan.families == ["number_theory"]
    assert plan.scrape_window == {"year": 2026, "month": 4, "category": "math.NT"}


def test_plan_records_expected_fixture_path_in_notes():
    plan = realmath_scrape.plan(
        _request(fixture_path="tests/fixtures/realmath/qa_candidates.jsonl", notes="dry run")
    )
    assert "dry run" in plan.notes
    assert "tests/fixtures/realmath/qa_candidates.jsonl" in plan.notes


def test_plan_refuses_unknown_request_fields():
    with pytest.raises(ValueError, match="unknown plan request fields"):
        realmath_scrape.plan(_request(scrape_now=True))


def test_plan_refuses_missing_request_fields():
    request = _request()
    del request["source_name"]
    with pytest.raises(ValueError, match="missing plan request fields"):
        realmath_scrape.plan(request)


def test_plan_refuses_non_positive_target_count():
    with pytest.raises(ValueError, match="target_count"):
        realmath_scrape.plan(_request(target_count=0))


def test_plan_refuses_unknown_scrape_window_fields():
    with pytest.raises(ValueError, match="unknown scrape_window fields"):
        realmath_scrape.plan(_request(scrape_window={"until_forever": True}))


def test_plan_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    realmath_scrape.plan(_request())
    assert list(tmp_path.iterdir()) == []

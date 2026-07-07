"""Plan creation for arxiv_bulk is pure: no manifest parsing, no calls, no writes."""

from __future__ import annotations

import socket

import pytest

from icepick.allocation.adapters import arxiv_bulk
from icepick.contracts.manifests import SOURCE_ARXIV_BULK, ProposedPlan


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during plan creation")

    monkeypatch.setattr(socket, "socket", _blocked)


def _request(**overrides):
    base = dict(
        source_name="arxiv_bulk_2025Q1",
        target_count=500,
        requested_by="alice",
        requested_at="2026-07-06T00:00:00Z",
    )
    base.update(overrides)
    return base


def test_plan_builds_a_proposed_plan():
    plan = arxiv_bulk.plan(_request())
    assert isinstance(plan, ProposedPlan)
    assert plan.source_type == SOURCE_ARXIV_BULK
    assert plan.source_name == "arxiv_bulk_2025Q1"
    assert plan.target_count == 500
    assert plan.requested_by == "alice"
    assert plan.estimated_calls > 0


def test_plan_estimates_calls_scaling_with_target_count():
    small = arxiv_bulk.plan(_request(target_count=10))
    large = arxiv_bulk.plan(_request(target_count=100))
    assert large.estimated_calls > small.estimated_calls


def test_plan_records_families_and_bulk_scrape_window():
    plan = arxiv_bulk.plan(
        _request(
            families=["pde"],
            scrape_window={
                "year": 2025, "month": 1, "category": "math.AP",
                "extraction": "latex", "manifest_path": "/tmp/src_manifest.xml",
            },
        )
    )
    assert plan.families == ["pde"]
    assert plan.scrape_window["manifest_path"] == "/tmp/src_manifest.xml"
    assert plan.scrape_window["category"] == "math.AP"


def test_plan_accepts_cache_dir_in_window():
    plan = arxiv_bulk.plan(
        _request(scrape_window={"extraction": "qa", "cache_dir": "/tmp/oai_cache"})
    )
    assert plan.scrape_window["cache_dir"] == "/tmp/oai_cache"


def test_plan_records_expected_fixture_path_in_notes():
    plan = arxiv_bulk.plan(
        _request(fixture_path="tests/fixtures/arxiv_bulk/qa_candidates.jsonl", notes="dry run")
    )
    assert "dry run" in plan.notes
    assert "tests/fixtures/arxiv_bulk/qa_candidates.jsonl" in plan.notes


def test_plan_refuses_unknown_request_fields():
    with pytest.raises(ValueError, match="unknown plan request fields"):
        arxiv_bulk.plan(_request(download_now=True))


def test_plan_refuses_missing_request_fields():
    request = _request()
    del request["source_name"]
    with pytest.raises(ValueError, match="missing plan request fields"):
        arxiv_bulk.plan(request)


def test_plan_refuses_non_positive_target_count():
    with pytest.raises(ValueError, match="target_count"):
        arxiv_bulk.plan(_request(target_count=0))


def test_plan_refuses_unknown_scrape_window_fields():
    with pytest.raises(ValueError, match="unknown scrape_window fields"):
        arxiv_bulk.plan(_request(scrape_window={"until_forever": True}))


def test_plan_refuses_bad_exclude_arxiv_ids():
    with pytest.raises(ValueError, match="exclude_arxiv_ids"):
        arxiv_bulk.plan(_request(scrape_window={"exclude_arxiv_ids": [123]}))


def test_plan_refuses_non_string_manifest_path():
    with pytest.raises(ValueError, match="manifest_path"):
        arxiv_bulk.plan(_request(scrape_window={"manifest_path": 5}))


def test_plan_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arxiv_bulk.plan(_request())
    assert list(tmp_path.iterdir()) == []

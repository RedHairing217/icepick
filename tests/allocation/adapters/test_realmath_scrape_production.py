"""Production run: realmath_scrape.run scrapes arXiv in-house.

The arXiv fetcher is monkeypatched to return canned Atom XML, so these
tests exercise the whole production path — gates → scrape → normalise →
handoff → report — with no network.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone

import pytest

from icepick.allocation.adapters import realmath_scrape
from icepick.allocation.scrape import realmath as source
from icepick.contracts.manifests import ApprovedManifest, SOURCE_REALMATH_SCRAPE

NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2604.00001v1</id>
    <title>On a nonlinear PDE</title>
    <summary>We prove existence of solutions to a nonlinear PDE.</summary>
    <published>2026-04-01T00:00:00Z</published>
    <arxiv:primary_category term="math.AP"/>
    <category term="math.AP"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2604.00002v1</id>
    <title>Number theory meets PDE</title>
    <summary>A cross-listed paper.</summary>
    <published>2026-04-02T00:00:00Z</published>
    <arxiv:primary_category term="math.NT"/>
    <category term="math.NT"/>
    <category term="math.AP"/>
  </entry>
</feed>"""

_EMPTY = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during production scrape test")

    monkeypatch.setattr(socket, "socket", _blocked)


@pytest.fixture(autouse=True)
def _canned_arxiv(monkeypatch):
    def fetcher(query, *, start, max_results):
        return _FEED if start == 0 else _EMPTY

    monkeypatch.setattr(source, "default_arxiv_fetcher", fetcher)


def _manifest(tmp_path, **overrides):
    base = dict(
        run_id="20260703T120000Z",
        source_type=SOURCE_REALMATH_SCRAPE,
        processor_mode="production",
        requested_by="alice",
        requested_at="2026-07-03T00:00:00Z",
        approved_by="bob",
        approved_at="2026-07-03T01:00:00Z",
        source_name="pde_2026Q2",
        target_count=5,
        call_budget=1000,
        judge_enabled=False,
        confirmation_enabled=False,
        enable_leakage=False,
        enable_duplication=True,
        enable_robustness=False,
        families=["pde"],
        scrape_window={"category": "math.AP"},
        truth_policy="extracted",
        output_dir=str(tmp_path / "intake"),
    )
    base.update(overrides)
    return ApprovedManifest(**base)


def test_production_run_scrapes_and_writes_the_handoff(tmp_path):
    outcome = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    assert outcome.calibration_replay is False
    assert outcome.processor_mode == "production"
    assert outcome.paper_count == 2
    assert outcome.record_count == 2
    assert outcome.handoff_path.exists()

    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert len(records) == 2
    for record in records:
        assert record["source"] == "pde_2026Q2"
        assert record["provenance"] == "extracted"
        assert record["family"] == "pde"
        assert record["arxiv_id"]
        assert "calibration_replay" not in record.get("metadata", {})
    assert {r["arxiv_id"] for r in records} == {"2604.00001", "2604.00002"}

    # raw/papers.jsonl keeps titles for production (pulled from candidate metadata).
    papers = [json.loads(l) for l in (outcome.raw_dir / "papers.jsonl").read_text().splitlines() if l.strip()]
    assert all(p.get("title") for p in papers)
    assert {"On a nonlinear PDE", "Number theory meets PDE"} == {p["title"] for p in papers}


def test_production_run_honours_primary_only(tmp_path):
    manifest = _manifest(tmp_path, scrape_window={"category": "math.AP", "primary_only": True})
    outcome = realmath_scrape.run(manifest, now=NOW)
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    # The cross-listed math.NT paper is dropped; only the primary-math.AP one survives.
    assert [r["arxiv_id"] for r in records] == ["2604.00001"]


def test_production_run_reports_acquisition_spend(tmp_path):
    outcome = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    assert outcome.acquisition is not None
    assert outcome.acquisition["arxiv_queries"] >= 1
    assert outcome.acquisition["qa_calls"] == 0  # abstract mode: no LLM spend
    assert outcome.acquisition["call_budget"] == 1000
    report = outcome.report_path.read_text()
    assert "## Spend (acquisition calls)" in report
    assert "arxiv_query" in report


def test_production_report_is_not_marked_calibration_replay(tmp_path):
    outcome = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    report = outcome.report_path.read_text()
    assert "calibration_replay: false" in report
    assert "| handoff records | 2 |" in report


def test_production_run_with_no_results_writes_empty_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "default_arxiv_fetcher", lambda q, *, start, max_results: _EMPTY)
    outcome = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    assert outcome.record_count == 0
    assert outcome.handoff_path.exists()
    assert outcome.handoff_path.read_text() == ""
    assert any("no candidates" in w for w in outcome.warnings)

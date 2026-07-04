"""Surplus preservation: cap overflow is written mount-ready, never dropped.

Same no-network production harness as test_realmath_scrape_production; a
dense fake extractor makes the breadth cap bite so accepted rows overflow
the handoff. The invariant under test: every extracted (paid-for) row ends
up in the handoff or in ``handoff/surplus_records.jsonl`` — good theorems
are never rejected.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone

import pytest

from icepick.allocation.adapters import realmath_scrape
from icepick.allocation.scrape import realmath as source
from icepick.contracts.manifests import ApprovedManifest, SOURCE_REALMATH_SCRAPE

NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)

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
    <title>A second PDE paper</title>
    <summary>More solutions to more PDEs.</summary>
    <published>2026-04-02T00:00:00Z</published>
    <arxiv:primary_category term="math.AP"/>
    <category term="math.AP"/>
  </entry>
</feed>"""

_EMPTY = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during surplus test")

    monkeypatch.setattr(socket, "socket", _blocked)


@pytest.fixture(autouse=True)
def _canned_arxiv(monkeypatch):
    def fetcher(query, *, start, max_results):
        return _FEED if start == 0 else _EMPTY

    monkeypatch.setattr(source, "default_arxiv_fetcher", fetcher)


def _dense(paper, *, family=None):
    """Three accepted rows per paper, upstream candidate shape."""
    return [
        {
            "arxiv_id": paper.arxiv_id,
            "link": f"http://arxiv.org/abs/{paper.arxiv_id}v1",
            "statement": f"{paper.arxiv_id} theorem {i}",
            "answer": f"${i}$",
            "provenance": "extracted",
        }
        for i in range(3)
    ]


@pytest.fixture
def _dense_extractor(monkeypatch):
    monkeypatch.setattr(source, "extractor_for", lambda mode: _dense)


def _manifest(tmp_path, **overrides):
    base = dict(
        run_id="20260704T120000Z",
        source_type=SOURCE_REALMATH_SCRAPE,
        processor_mode="production",
        requested_by="alice",
        requested_at="2026-07-04T00:00:00Z",
        approved_by="bob",
        approved_at="2026-07-04T01:00:00Z",
        source_name="pde_surplus_test",
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


def test_cap_overflow_is_preserved_canonical_and_mount_ready(tmp_path, _dense_extractor):
    manifest = _manifest(
        tmp_path, scrape_window={"category": "math.AP", "max_per_paper": 1}
    )
    outcome = realmath_scrape.run(manifest, now=NOW)

    assert outcome.record_count == 2  # one kept per paper
    assert outcome.surplus_count == 4  # two preserved per paper
    assert outcome.surplus_path is not None
    assert outcome.surplus_path == outcome.handoff_path.parent / "surplus_records.jsonl"
    assert outcome.surplus_path.is_file()

    surplus = [
        json.loads(l) for l in outcome.surplus_path.read_text().splitlines() if l.strip()
    ]
    assert len(surplus) == 4
    for row in surplus:
        # Canonical record shape — directly mountable, same funnel as the handoff.
        assert row["source"] == "pde_surplus_test"
        assert row["provenance"] == "extracted"
        assert row["family"] == "pde"
        assert row["statement"]
        assert row["truth_policy"]

    # Nothing lost, nothing double-counted: kept + surplus == every accepted row.
    handoff = [
        json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()
    ]
    kept = {r["statement"] for r in handoff}
    preserved = {r["statement"] for r in surplus}
    assert kept.isdisjoint(preserved)
    assert len(kept | preserved) == 6


def test_surplus_is_counted_in_the_report_with_a_mount_hint(tmp_path, _dense_extractor):
    manifest = _manifest(
        tmp_path, scrape_window={"category": "math.AP", "max_per_paper": 1}
    )
    outcome = realmath_scrape.run(manifest, now=NOW)
    report = outcome.report_path.read_text()
    assert "| surplus records (cap overflow, preserved) | 4 |" in report
    assert "## Surplus — accepted past the caps" in report
    assert "allocation mount" in report
    assert str(outcome.surplus_path) in report


def test_no_surplus_means_no_file_and_a_zero_count(tmp_path):
    outcome = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    assert outcome.surplus_count == 0
    assert outcome.surplus_path is None
    assert not (outcome.handoff_path.parent / "surplus_records.jsonl").exists()
    report = outcome.report_path.read_text()
    assert "| surplus records (cap overflow, preserved) | 0 |" in report
    assert "## Surplus" not in report


def test_run_clears_a_planted_stale_surplus_file(tmp_path):
    """A re-run with no surplus must not leave an old surplus file claiming
    preserved rows this run never produced (same hygiene as quarantine)."""
    run_dir = tmp_path / "intake" / "runs" / "20260704T120000Z"
    (run_dir / "handoff").mkdir(parents=True)
    stale = run_dir / "handoff" / "surplus_records.jsonl"
    stale.write_text('{"stale": true}\n', encoding="utf-8")

    outcome = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    assert outcome.surplus_count == 0
    assert not stale.exists()

"""In-house RealMath scraper: query building, Atom parsing, orchestration.

No network: every test injects a ``fetcher`` returning canned Atom XML.
"""

from __future__ import annotations

import socket

import pytest

from icepick.allocation.scrape import realmath as source

# Two entries: a primary math.AP paper and one that only cross-lists into
# math.AP (primary math.NT). The second id is v2 to exercise version-strip.
_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2604.00001v1</id>
    <title>On a nonlinear   PDE</title>
    <summary>We prove existence of solutions to a nonlinear PDE.</summary>
    <published>2026-04-01T00:00:00Z</published>
    <arxiv:primary_category term="math.AP"/>
    <category term="math.AP"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2604.00002v2</id>
    <title>Number theory meets PDE</title>
    <summary>A cross-listed paper.</summary>
    <published>2026-04-02T00:00:00Z</published>
    <arxiv:primary_category term="math.NT"/>
    <category term="math.NT"/>
    <category term="math.AP"/>
  </entry>
</feed>"""

_EMPTY = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _one_page_fetcher(feed=_FEED):
    """Return ``feed`` on the first page, then empty so paging terminates."""
    def fetcher(query, *, start, max_results):
        return feed if start == 0 else _EMPTY
    return fetcher


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a scraper unit test")

    monkeypatch.setattr(socket, "socket", _blocked)


# --- build_query --------------------------------------------------------------


def test_build_query_exact_match_for_subcategory():
    assert source.build_query({"category": "math.AP"}) == "cat:math.AP"


def test_build_query_wildcards_a_bare_main_category():
    assert source.build_query({"category": "math"}) == "cat:math.*"


def test_build_query_adds_a_submitted_date_lower_bound():
    q = source.build_query({"category": "math.AP", "year": 2026, "month": 4})
    assert "cat:math.AP" in q
    assert "submittedDate:[202604010000 TO" in q


def test_build_query_defaults_to_all_math():
    assert source.build_query(None) == "cat:math.*"


# --- parse_atom ---------------------------------------------------------------


def test_parse_atom_reads_entries_and_strips_versions():
    papers = source.parse_atom(_FEED)
    assert [p.arxiv_id for p in papers] == ["2604.00001", "2604.00002"]
    first = papers[0]
    assert first.title == "On a nonlinear PDE"  # whitespace collapsed
    assert first.primary_category == "math.AP"
    assert first.link == "http://arxiv.org/abs/2604.00001v1"
    assert papers[1].categories == ["math.NT", "math.AP"]


def test_parse_atom_rejects_non_xml():
    with pytest.raises(ValueError, match="not valid Atom XML"):
        source.parse_atom("<<not xml")


# --- default_extractor --------------------------------------------------------


def test_default_extractor_emits_a_metadata_candidate():
    paper = source.parse_atom(_FEED)[0]
    [candidate] = source.default_extractor(paper, family="pde")
    assert candidate["statement"] == "We prove existence of solutions to a nonlinear PDE."
    assert candidate["arxiv_id"] == "2604.00001"
    assert candidate["provenance"] == "extracted"
    assert candidate["family"] == "pde"
    assert candidate["metadata"]["primary_category"] == "math.AP"
    assert candidate["metadata"]["extraction"] == "abstract"


# --- scrape orchestration -----------------------------------------------------


def test_scrape_returns_candidates_from_the_feed():
    result = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="pde", target_count=10,
        fetcher=_one_page_fetcher(),
    )
    assert result.papers_seen == 2
    assert len(result.candidates) == 2
    assert {c["arxiv_id"] for c in result.candidates} == {"2604.00001", "2604.00002"}
    assert result.surplus == []  # nothing capped, nothing to preserve


def test_scrape_primary_only_drops_cross_listed_papers():
    result = source.scrape(
        scrape_window={"category": "math.AP", "primary_only": True},
        source_name="pde", target_count=10, fetcher=_one_page_fetcher(),
    )
    assert [c["arxiv_id"] for c in result.candidates] == ["2604.00001"]


def test_scrape_stops_at_target_count():
    result = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="pde", target_count=1,
        fetcher=_one_page_fetcher(),
    )
    assert len(result.candidates) == 1


def test_scrape_passes_the_built_query_to_the_fetcher():
    seen = {}

    def capturing(query, *, start, max_results):
        seen["query"] = query
        return _EMPTY

    source.scrape(
        scrape_window={"category": "math.AP", "year": 2026, "month": 4},
        source_name="pde", target_count=5, fetcher=capturing,
    )
    assert seen["query"].startswith("cat:math.AP AND submittedDate:")


def test_scrape_warns_when_no_candidates():
    result = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="pde", target_count=5,
        fetcher=lambda q, *, start, max_results: _EMPTY,
    )
    assert result.candidates == []
    assert any("no candidates" in w for w in result.warnings)


def test_scrape_respects_max_papers():
    result = source.scrape(
        scrape_window={"category": "math.AP", "max_papers": 1},
        source_name="pde", target_count=10, fetcher=_one_page_fetcher(),
    )
    assert result.papers_seen == 1
    assert len(result.candidates) == 1


def test_scrape_max_per_paper_forces_breadth(monkeypatch):
    """One theorem-dense paper must not fill the whole target under a per-paper cap."""
    def dense(paper, *, family=None):
        return [{"arxiv_id": paper.arxiv_id, "statement": f"{paper.arxiv_id} thm {i}"}
                for i in range(10)]

    result = source.scrape(
        scrape_window={"category": "math.AP", "max_per_paper": 2},
        source_name="pde", target_count=10, fetcher=_one_page_fetcher(), extractor=dense,
    )
    # Two papers in the feed × 2 kept each = 4, not 10-from-one-paper.
    from collections import Counter
    per_paper = Counter(c["arxiv_id"] for c in result.candidates)
    assert all(n <= 2 for n in per_paper.values())
    assert len(per_paper) == 2  # breadth: both papers contributed
    # The cap shapes the corpus, it does not reject: the other 8 accepted
    # rows of each paper are preserved as surplus, nothing on the floor.
    assert len(result.surplus) == 16
    per_paper_surplus = Counter(c["arxiv_id"] for c in result.surplus)
    assert per_paper_surplus == {"2604.00001": 8, "2604.00002": 8}
    kept_and_surplus = {c["statement"] for c in result.candidates} | {
        c["statement"] for c in result.surplus
    }
    assert len(kept_and_surplus) == 20  # every extracted row accounted for


def test_scrape_excluded_ids_are_skipped_without_counting():
    """Continuation: papers a prior run consumed are passed over for free —
    never extracted, never counted toward max_papers — so a follow-up run
    starts spending at the first unseen paper."""
    result = source.scrape(
        scrape_window={"category": "math.AP", "max_papers": 1,
                       "exclude_arxiv_ids": ["2604.00001"]},
        source_name="pde", target_count=10, fetcher=_one_page_fetcher(),
    )
    # The single max_papers slot goes to the next unseen paper, not the
    # excluded one (contrast: test_scrape_respects_max_papers takes 00001).
    assert result.papers_seen == 1
    assert [c["arxiv_id"] for c in result.candidates] == ["2604.00002"]


def test_scrape_target_count_overflow_lands_in_surplus():
    """Hitting the target mid-paper preserves that paper's remaining rows."""
    def dense(paper, *, family=None):
        return [{"arxiv_id": paper.arxiv_id, "statement": f"{paper.arxiv_id} thm {i}"}
                for i in range(10)]

    result = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="pde", target_count=3,
        fetcher=_one_page_fetcher(), extractor=dense,
    )
    assert len(result.candidates) == 3
    # Paper one is already extracted (paid for) in full: 3 kept, 7 preserved.
    # Paper two is never extracted once the target is met — nothing paid,
    # nothing owed, so no surplus from it.
    assert len(result.surplus) == 7
    assert {c["arxiv_id"] for c in result.surplus} == {"2604.00001"}


def test_scrape_call_budget_is_a_hard_cap_on_paid_calls():
    result = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="pde", target_count=10,
        fetcher=_one_page_fetcher(), call_budget=1,
    )
    # The single budgeted call buys page one; extracting its papers is free
    # (abstract mode), and the next paid call (page two) is refused.
    assert result.queries == 1
    assert len(result.candidates) == 2  # both page-one papers, extraction unpaid
    assert result.interrupted is True  # paused, resumable — not dead
    assert any("call budget 1 exhausted" in w for w in result.warnings)


def test_scrape_reports_call_counts():
    result = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="pde", target_count=10,
        fetcher=_one_page_fetcher(),
    )
    assert result.queries >= 1
    assert result.latex_fetches == 0  # abstract mode makes no e-print/LLM calls
    assert result.qa_calls == 0


def test_scrape_pages_and_dedups_across_pages():
    """Second page re-lists paper 1 (a repost) and adds a new one; start advances."""
    page_two = _FEED.replace("2604.00002", "2604.00003").replace(
        "Number theory meets PDE", "Another PDE result"
    ).replace('term="math.NT"', 'term="math.AP"')
    starts = []

    def paging_fetcher(query, *, start, max_results):
        starts.append(start)
        if start == 0:
            return _FEED                      # 2604.00001, 2604.00002
        if start == source._PAGE_SIZE:
            return page_two                   # 2604.00001 (dup), 2604.00003 (new)
        return _EMPTY

    result = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="pde", target_count=10,
        fetcher=paging_fetcher,
    )
    # start advances by the page size (offset into the result set), not the
    # post-filter parsed count — so pages never overlap.
    assert starts[:3] == [0, source._PAGE_SIZE, 2 * source._PAGE_SIZE]
    assert [c["arxiv_id"] for c in result.candidates] == ["2604.00001", "2604.00002", "2604.00003"]
    assert result.papers_seen == 3  # the repost of 2604.00001 was not recounted

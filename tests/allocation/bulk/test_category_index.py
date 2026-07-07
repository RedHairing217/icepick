"""OAI-PMH category index: paging, 503 backoff, page-cache resume, queries.

No network anywhere: every test injects a scripted ``fetcher`` returning
canned ``OAIResponse`` objects (bodies from tests/fixtures/arxiv_bulk/) and
a spy ``sleeper``, so backoff is asserted without waiting.
"""

from __future__ import annotations

import dataclasses
import socket
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from icepick.allocation.bulk.category_index import (
    DEFAULT_BACKOFF_SCHEDULE,
    MAX_ATTEMPTS_PER_PAGE,
    CategoryIndex,
    OAIError,
    OAIResponse,
    PaperMeta,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "arxiv_bulk"
PAGE1 = (FIXTURES / "oai_page1.xml").read_text(encoding="utf-8")
PAGE2 = (FIXTURES / "oai_page2.xml").read_text(encoding="utf-8")
PAGE2_TOKEN = "token-page-2-abc123"  # page 1's resumptionToken

PAGE1_IDS = {"2501.00101", "2501.00202", "2501.00303", "2501.00404"}
PAGE2_IDS = {"2502.00111", "2502.00222", "2502.00333", "2502.00444",
             "2502.00555", "2502.00666"}
ALL_IDS = PAGE1_IDS | PAGE2_IDS


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a category-index unit test")

    monkeypatch.setattr(socket, "socket", _blocked)


def _ok(text):
    return OAIResponse(status=200, retry_after=None, text=text)


def _503(retry_after=None):
    return OAIResponse(status=503, retry_after=retry_after, text="")


class ScriptedFetcher:
    """Plays back a fixed response script; records every requested URL."""

    def __init__(self, script):
        self.script = list(script)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        if not self.script:
            raise AssertionError("fetcher called more times than scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class SpySleeper:
    def __init__(self):
        self.sleeps = []

    def __call__(self, seconds):
        self.sleeps.append(seconds)


def _build(cache_dir, script):
    """Build a fresh index over ``script``; return (index, fetcher, sleeper)."""
    fetcher = ScriptedFetcher(script)
    sleeper = SpySleeper()
    index = CategoryIndex(cache_dir)
    index.build(oai_set="math", fetcher=fetcher, sleeper=sleeper)
    return index, fetcher, sleeper


def _params(url):
    return parse_qs(urlsplit(url).query)


# --- paging + token chaining --------------------------------------------------


def test_paging_follows_resumption_token_serially(tmp_path):
    index, fetcher, sleeper = _build(tmp_path, [_ok(PAGE1), _ok(PAGE2)])

    assert len(fetcher.urls) == 2
    first, second = map(_params, fetcher.urls)
    assert first == {"verb": ["ListRecords"], "set": ["math"],
                     "metadataPrefix": ["arXiv"]}
    # A token request carries ONLY verb + resumptionToken (OAI exclusivity).
    assert second == {"verb": ["ListRecords"], "resumptionToken": [PAGE2_TOKEN]}
    assert index.oai_requests == 2
    assert sleeper.sleeps == []
    assert {p for p in ALL_IDS if index.lookup(p)} == ALL_IDS


def test_single_page_feed_stops_on_empty_token(tmp_path):
    # PAGE2 carries an empty <resumptionToken/> — a complete one-page feed.
    index, fetcher, _ = _build(tmp_path, [_ok(PAGE2)])
    assert len(fetcher.urls) == 1
    assert index.oai_requests == 1
    assert {p for p in PAGE2_IDS if index.lookup(p)} == PAGE2_IDS


# --- from_date window bound (W3 H2) -------------------------------------------


def test_from_date_bounds_initial_request_only(tmp_path):
    fetcher = ScriptedFetcher([_ok(PAGE1), _ok(PAGE2)])
    index = CategoryIndex(tmp_path)
    index.build(oai_set="math", fetcher=fetcher, sleeper=SpySleeper(),
                from_date="2025-01-01")

    first, second = map(_params, fetcher.urls)
    # Initial request carries the `from` datestamp bound alongside set+prefix.
    assert first == {"verb": ["ListRecords"], "set": ["math"],
                     "metadataPrefix": ["arXiv"], "from": ["2025-01-01"]}
    # Token-continuation request carries ONLY verb + resumptionToken —
    # no from=, no set=, no metadataPrefix= (re-sending them is an OAI error).
    assert second == {"verb": ["ListRecords"], "resumptionToken": [PAGE2_TOKEN]}
    assert "from" not in second and "set" not in second
    assert {p for p in ALL_IDS if index.lookup(p)} == ALL_IDS


def test_from_date_none_preserves_urls_byte_for_byte(tmp_path):
    # from_date=None (default) must reproduce the exact URLs of the
    # no-from_date build — the bound is purely additive.
    baseline = ScriptedFetcher([_ok(PAGE1), _ok(PAGE2)])
    CategoryIndex(tmp_path / "a").build(
        oai_set="math", fetcher=baseline, sleeper=SpySleeper())

    explicit_none = ScriptedFetcher([_ok(PAGE1), _ok(PAGE2)])
    CategoryIndex(tmp_path / "b").build(
        oai_set="math", fetcher=explicit_none, sleeper=SpySleeper(),
        from_date=None)

    assert explicit_none.urls == baseline.urls
    assert all("from=" not in url for url in baseline.urls)


# --- record parsing -----------------------------------------------------------


def test_lookup_hit_parses_id_categories_and_collapsed_title(tmp_path):
    index, _, _ = _build(tmp_path, [_ok(PAGE1), _ok(PAGE2)])

    meta = index.lookup("2501.00101")
    assert meta == PaperMeta(
        arxiv_id="2501.00101",
        primary_category="math.AP",
        categories=("math.AP", "math.FA"),
        title="Regularity of solutions to the Navier-Stokes equations",
    )
    # Three-category record parses in listed order, first = primary.
    meta666 = index.lookup("2502.00666")
    assert meta666.primary_category == "math.AP"
    assert meta666.categories == ("math.AP", "math.FA", "cs.LG")


def test_lookup_miss_returns_none(tmp_path):
    index, _, _ = _build(tmp_path, [_ok(PAGE2)])
    assert index.lookup("2501.00101") is None
    assert index.lookup("9999.99999") is None


def test_paper_meta_stores_no_abstract(tmp_path):
    # Fixture records DO carry <abstract>; the index must not keep it.
    index, _, _ = _build(tmp_path, [_ok(PAGE2)])
    field_names = {f.name for f in dataclasses.fields(PaperMeta)}
    assert field_names == {"arxiv_id", "primary_category", "categories", "title"}
    meta = index.lookup("2502.00111")
    assert not hasattr(meta, "abstract")


def test_deleted_record_is_skipped(tmp_path):
    page = """<?xml version="1.0" encoding="UTF-8"?>
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
      <ListRecords>
        <record>
          <header status="deleted">
            <identifier>oai:arXiv.org:2501.00001</identifier>
          </header>
        </record>
        <record>
          <header><identifier>oai:arXiv.org:2501.00002</identifier></header>
          <metadata>
            <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
              <id>2501.00002</id>
              <title>Survivor</title>
              <categories>math.AP</categories>
            </arXiv>
          </metadata>
        </record>
        <resumptionToken/>
      </ListRecords>
    </OAI-PMH>"""
    index, _, _ = _build(tmp_path, [_ok(page)])
    assert index.lookup("2501.00001") is None
    assert index.lookup("2501.00002").title == "Survivor"


# --- ids_for ------------------------------------------------------------------


def test_ids_for_filters_by_yymm(tmp_path):
    index, _, _ = _build(tmp_path, [_ok(PAGE1), _ok(PAGE2)])

    jan = index.ids_for(category="math.AP", yymm="2501", primary_only=True)
    feb = index.ids_for(category="math.AP", yymm="2502", primary_only=True)
    assert jan == ["2501.00101", "2501.00404"]
    assert feb == ["2502.00111", "2502.00666"]
    assert index.ids_for(category="math.AP", yymm="2412", primary_only=True) == []


def test_ids_for_primary_only_true_excludes_cross_listed(tmp_path):
    index, _, _ = _build(tmp_path, [_ok(PAGE1), _ok(PAGE2)])
    # 00101/00303 cross-list math.FA but their primaries differ.
    assert index.ids_for(category="math.FA", yymm="2501",
                         primary_only=True) == ["2501.00202"]
    assert index.ids_for(category="cs.LG", yymm="2502",
                         primary_only=True) == ["2502.00333", "2502.00555"]


def test_ids_for_primary_only_false_includes_cross_listed(tmp_path):
    index, _, _ = _build(tmp_path, [_ok(PAGE1), _ok(PAGE2)])
    assert index.ids_for(category="math.FA", yymm="2501", primary_only=False) == [
        "2501.00101", "2501.00202", "2501.00303",
    ]
    assert index.ids_for(category="cs.LG", yymm="2502", primary_only=False) == [
        "2502.00111", "2502.00333", "2502.00555", "2502.00666",
    ]


# --- 503 handling ---------------------------------------------------------------


def test_503_with_retry_after_sleeps_exactly_that_long(tmp_path):
    index, fetcher, sleeper = _build(
        tmp_path, [_503(retry_after=7.5), _ok(PAGE2)])

    assert sleeper.sleeps == [7.5]           # honored exactly, not rounded
    assert index.oai_requests == 2           # retry counted
    assert fetcher.urls[0] == fetcher.urls[1]  # same page re-requested
    assert index.lookup("2502.00111") is not None
    # journaled telemetry (realmath shapes)
    assert index.rate_limit_events == 1
    assert index.rate_limit_backoff_seconds == 7.5
    assert index.rate_limit_statuses == {"503": 1}


def test_503_without_retry_after_uses_default_schedule(tmp_path):
    index, _, sleeper = _build(tmp_path, [_503(), _503(), _ok(PAGE2)])

    assert sleeper.sleeps == [DEFAULT_BACKOFF_SCHEDULE[0],
                              DEFAULT_BACKOFF_SCHEDULE[1]]
    assert index.oai_requests == 3
    assert index.rate_limit_events == 2
    assert index.rate_limit_backoff_seconds == sum(DEFAULT_BACKOFF_SCHEDULE[:2])


def test_gives_up_after_max_attempts_with_clear_error(tmp_path):
    fetcher = ScriptedFetcher([_503()] * MAX_ATTEMPTS_PER_PAGE)
    sleeper = SpySleeper()
    index = CategoryIndex(tmp_path)

    with pytest.raises(OAIError, match="giving up .* after 5 attempts"):
        index.build(oai_set="math", fetcher=fetcher, sleeper=sleeper)

    assert index.oai_requests == MAX_ATTEMPTS_PER_PAGE
    # One fewer sleep than attempts; schedule is bounded and capped.
    assert sleeper.sleeps == list(DEFAULT_BACKOFF_SCHEDULE)
    assert index.rate_limit_statuses == {"503": MAX_ATTEMPTS_PER_PAGE - 1}


def test_non_200_non_503_raises_immediately(tmp_path):
    fetcher = ScriptedFetcher([OAIResponse(status=500, retry_after=None, text="")])
    sleeper = SpySleeper()
    index = CategoryIndex(tmp_path)

    with pytest.raises(OAIError, match="HTTP 500"):
        index.build(oai_set="math", fetcher=fetcher, sleeper=sleeper)
    assert index.oai_requests == 1
    assert sleeper.sleeps == []


def test_oai_requests_counts_retries_across_pages(tmp_path):
    index, _, sleeper = _build(
        tmp_path,
        [_503(retry_after=1.0), _ok(PAGE1), _503(), _ok(PAGE2)],
    )
    assert index.oai_requests == 4
    # Default-backoff attempt counting resets per page: page 2's first
    # no-header 503 sleeps schedule[0] again.
    assert sleeper.sleeps == [1.0, DEFAULT_BACKOFF_SCHEDULE[0]]
    assert {p for p in ALL_IDS if index.lookup(p)} == ALL_IDS


# --- page cache + resume --------------------------------------------------------


def test_page_cached_before_next_request(tmp_path):
    cached_when_called = []

    def fetcher(url):
        cached_when_called.append(len(list(tmp_path.glob("*.xml"))))
        return _ok(PAGE1) if len(cached_when_called) == 1 else _ok(PAGE2)

    CategoryIndex(tmp_path).build(oai_set="math", fetcher=fetcher,
                                  sleeper=SpySleeper())
    # Page 1 was on disk before the page-2 request went out.
    assert cached_when_called == [0, 1]
    assert len(list(tmp_path.glob("*.xml"))) == 2


def test_warm_cache_rebuild_issues_zero_requests(tmp_path):
    first, _, _ = _build(tmp_path, [_ok(PAGE1), _ok(PAGE2)])

    def exploding_fetcher(url):
        raise AssertionError("warm-cache rebuild must not issue any request")

    rebuilt = CategoryIndex(tmp_path)
    rebuilt.build(oai_set="math", fetcher=exploding_fetcher, sleeper=SpySleeper())

    assert rebuilt.oai_requests == 0
    assert rebuilt.lookup("2501.00101") == first.lookup("2501.00101")
    assert rebuilt.ids_for(category="math.AP", yymm="2502", primary_only=False) \
        == first.ids_for(category="math.AP", yymm="2502", primary_only=False)


def test_partial_cache_resumes_at_first_uncached_page(tmp_path):
    # Build killed between page 1 and page 2: page 1 is cached, then the
    # process dies (fetcher raises on the second request).
    fetcher = ScriptedFetcher([_ok(PAGE1), RuntimeError("killed mid-build")])
    interrupted = CategoryIndex(tmp_path)
    with pytest.raises(RuntimeError, match="killed mid-build"):
        interrupted.build(oai_set="math", fetcher=fetcher, sleeper=SpySleeper())
    assert len(list(tmp_path.glob("*.xml"))) == 1

    # Resume: page 1 replays from cache (zero requests for it); the only
    # request issued is page 2, addressed by the token stored in page 1.
    resumed_fetcher = ScriptedFetcher([_ok(PAGE2)])
    resumed = CategoryIndex(tmp_path)
    resumed.build(oai_set="math", fetcher=resumed_fetcher, sleeper=SpySleeper())

    assert resumed.oai_requests == 1
    assert _params(resumed_fetcher.urls[0]) == {
        "verb": ["ListRecords"], "resumptionToken": [PAGE2_TOKEN]}
    assert {p for p in ALL_IDS if resumed.lookup(p)} == ALL_IDS


# --- malformed pages ------------------------------------------------------------


def test_malformed_xml_raises_clear_error_and_is_not_cached(tmp_path):
    fetcher = ScriptedFetcher([_ok("this is not XML <<<")])
    index = CategoryIndex(tmp_path)
    with pytest.raises(OAIError, match="malformed OAI XML on page 1"):
        index.build(oai_set="math", fetcher=fetcher, sleeper=SpySleeper())
    # Parse-before-persist: a bad page never poisons the cache.
    assert list(tmp_path.iterdir()) == []


def test_oai_protocol_error_element_raises(tmp_path):
    page = """<?xml version="1.0" encoding="UTF-8"?>
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
      <error code="badResumptionToken">The token has expired.</error>
    </OAI-PMH>"""
    fetcher = ScriptedFetcher([_ok(page)])
    with pytest.raises(OAIError, match="badResumptionToken"):
        CategoryIndex(tmp_path).build(oai_set="math", fetcher=fetcher,
                                      sleeper=SpySleeper())

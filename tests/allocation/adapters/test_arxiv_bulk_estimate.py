"""Estimates describe the work without performing it.

The rollup path (expected_chunk_bytes / expected_egress_usd) parses a src
manifest LOCALLY via ``icepick.allocation.bulk.manifest``. That sibling module
may not be on disk while this adapter is built in parallel; the rollup tests
skip cleanly if it is absent, and the pure-ratio tests always run. When it IS
present, the tests use an OWN tiny manifest XML written to tmp_path — never the
sibling-owned ``tests/fixtures/arxiv_bulk/src_manifest_sample.xml``.
"""

from __future__ import annotations

import math
import socket

import pytest

from icepick.allocation.adapters import arxiv_bulk
from icepick.contracts.manifests import SOURCE_ARXIV_BULK

# An OWN minimal src manifest: two 2025-01 chunks + one 2025-02 chunk, shaped
# per the frozen §1 schema (ten fields, yymm as string, decimal-GB sizes).
_OWN_MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<arXivSRC>
  <file>
    <filename>src/arXiv_src_2501_001.tar</filename>
    <yymm>2501</yymm>
    <seq_num>1</seq_num>
    <first_item>2501.00001</first_item>
    <last_item>2501.09999</last_item>
    <num_items>9999</num_items>
    <size>2000000000</size>
    <md5sum>aaaa1111bbbb2222cccc3333dddd4444</md5sum>
    <content_md5sum>1111aaaa2222bbbb3333cccc4444dddd</content_md5sum>
    <timestamp>2025-02-04 09:22:11</timestamp>
  </file>
  <file>
    <filename>src/arXiv_src_2501_002.tar</filename>
    <yymm>2501</yymm>
    <seq_num>2</seq_num>
    <first_item>2501.10000</first_item>
    <last_item>2501.18442</last_item>
    <num_items>8443</num_items>
    <size>1000000000</size>
    <md5sum>bbbb2222cccc3333dddd4444eeee5555</md5sum>
    <content_md5sum>2222bbbb3333cccc4444dddd5555eeee</content_md5sum>
    <timestamp>2025-02-04 09:45:00</timestamp>
  </file>
  <file>
    <filename>src/arXiv_src_2502_001.tar</filename>
    <yymm>2502</yymm>
    <seq_num>1</seq_num>
    <first_item>2502.00001</first_item>
    <last_item>2502.10211</last_item>
    <num_items>10211</num_items>
    <size>3000000000</size>
    <md5sum>cccc3333dddd4444eeee5555ffff6666</md5sum>
    <content_md5sum>3333cccc4444dddd5555eeee6666ffff</content_md5sum>
    <timestamp>2025-03-05 11:30:45</timestamp>
  </file>
</arXivSRC>
"""


def _bulk_manifest_available() -> bool:
    try:
        import icepick.allocation.bulk.manifest  # noqa: F401
    except Exception:
        return False
    return True


_needs_manifest = pytest.mark.skipif(
    not _bulk_manifest_available(),
    reason="icepick.allocation.bulk.manifest not on disk yet (sibling module)",
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during estimation")

    monkeypatch.setattr(socket, "socket", _blocked)


def _own_manifest(tmp_path):
    path = tmp_path / "my_src_manifest.xml"
    path.write_text(_OWN_MANIFEST_XML, encoding="utf-8")
    return path


def _plan(**overrides):
    request = dict(
        source_name="arxiv_bulk_2025Q1",
        target_count=500,
        requested_by="alice",
        requested_at="2026-07-06T00:00:00Z",
    )
    request.update(overrides)
    return arxiv_bulk.plan(request)


def test_estimate_describes_the_expected_work():
    estimate = arxiv_bulk.estimate(_plan())
    assert estimate["source_type"] == SOURCE_ARXIV_BULK
    assert estimate["expected_papers"] >= estimate["target_count"]
    assert estimate["expected_candidates"] >= estimate["expected_papers"]
    assert estimate["expected_handoff_records"] == 500
    assert estimate["estimated_calls"] > 0
    assert estimate["call_kinds"]
    assert estimate["local_prerequisites"]


def test_estimate_call_kinds_cover_the_bulk_seams():
    latex = arxiv_bulk.estimate(_plan(scrape_window={"extraction": "latex"}))
    assert latex["call_kinds"] == ["oai_requests", "chunk_downloads"]
    assert latex["expected_llm_calls"] == 0

    qa = arxiv_bulk.estimate(_plan(scrape_window={"extraction": "qa"}))
    assert "qa_calls" in qa["call_kinds"]
    assert qa["expected_llm_calls"] > 0


def test_estimate_defaults_to_latex_not_abstract():
    """Bulk exists to mine LaTeX; a window with no extraction is latex, not abstract."""
    est = arxiv_bulk.estimate(_plan(scrape_window=None))
    assert est["extraction"] == "latex"
    assert est["call_kinds"] == ["oai_requests", "chunk_downloads"]


def test_estimate_qa_is_more_expensive_than_latex():
    latex = arxiv_bulk.estimate(_plan(target_count=10, scrape_window={"extraction": "latex"}))
    qa = arxiv_bulk.estimate(_plan(target_count=10, scrape_window={"extraction": "qa"}))
    assert qa["estimated_calls"] > latex["estimated_calls"]


def test_estimate_matches_plan_estimated_calls():
    plan = _plan(target_count=25, scrape_window={"extraction": "latex"})
    assert arxiv_bulk.estimate(plan)["estimated_calls"] == plan.estimated_calls


def test_estimate_rounds_against_the_operator_with_safety_multiplier():
    qa = arxiv_bulk.estimate(_plan(target_count=10, scrape_window={"extraction": "qa"}))
    expected_papers = qa["expected_papers"]
    oai_pages = max(1, -(-expected_papers // 1000))
    central = oai_pages + expected_papers + qa["expected_llm_calls"]
    assert qa["estimated_calls"] == math.ceil(central * arxiv_bulk.ESTIMATE_SAFETY_MULTIPLIER)


def test_estimate_refuses_abstract_extraction():
    plan = _plan(scrape_window={"extraction": "abstract"})
    with pytest.raises(ValueError, match="extraction"):
        arxiv_bulk.estimate(plan)


def test_estimate_refuses_foreign_source_types():
    plan = _plan()
    plan.source_type = "manual_mount"
    with pytest.raises(ValueError, match="source_type"):
        arxiv_bulk.estimate(plan)


def test_estimate_reports_zero_egress_without_a_manifest_path():
    """An early plan before the operator fetched the src manifest still estimates."""
    est = arxiv_bulk.estimate(_plan(scrape_window={"extraction": "latex"}))
    assert est["expected_chunk_bytes"] == 0
    assert est["expected_egress_usd"] == 0.0


@_needs_manifest
def test_estimate_prices_the_window_egress_from_the_manifest(tmp_path):
    """expected_chunk_bytes / expected_egress_usd come from the window's chunk rollup."""
    manifest_path = _own_manifest(tmp_path)
    est = arxiv_bulk.estimate(
        _plan(scrape_window={
            "year": 2025, "month": 1, "extraction": "latex",
            "manifest_path": str(manifest_path),
        })
    )
    # Jan 2025 = the two 2501 chunks: 2e9 + 1e9 = 3e9 bytes; egress 3 GB * 0.09.
    assert est["expected_chunk_bytes"] == 3_000_000_000
    assert est["expected_egress_usd"] == pytest.approx(0.27)


@_needs_manifest
def test_estimate_whole_year_window_rolls_up_all_chunks(tmp_path):
    manifest_path = _own_manifest(tmp_path)
    est = arxiv_bulk.estimate(
        _plan(scrape_window={
            "year": 2025, "extraction": "latex", "manifest_path": str(manifest_path),
        })
    )
    # year=2025, month=None -> whole year: all three chunks, 6e9 bytes.
    assert est["expected_chunk_bytes"] == 6_000_000_000
    assert est["expected_egress_usd"] == pytest.approx(0.54)


def test_estimate_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arxiv_bulk.estimate(_plan(scrape_window={"extraction": "latex"}))
    assert list(tmp_path.iterdir()) == []

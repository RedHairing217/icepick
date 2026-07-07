"""Unit tests for icepick.allocation.bulk.manifest.

No network access anywhere in this file: the socket-guard autouse fixture
blocks all socket creation.  Every test is pure (string/fixture inputs only).
"""

from __future__ import annotations

import math
import socket
import warnings
from pathlib import Path

import pytest

from icepick.allocation.bulk.manifest import (
    EGRESS_USD_PER_GB,
    ChunkRollup,
    ManifestEntry,
    ManifestError,
    chunks_for_ids,
    id_in_range,
    newstyle_key,
    parse_manifest,
    rollup,
    select_chunks,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_FIXTURE_XML = (
    Path(__file__).parent.parent.parent
    / "fixtures"
    / "arxiv_bulk"
    / "src_manifest_sample.xml"
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a manifest test")

    monkeypatch.setattr(socket, "socket", _blocked)


def _minimal_xml(*, extra_fields: str = "", replace: dict | None = None) -> str:
    """Return a single-entry manifest XML.  *extra_fields* is injected verbatim
    inside the ``<file>`` block; *replace* maps field names to replacement text
    (use empty string to omit the element entirely).
    """
    defaults = {
        "filename": "src/arXiv_src_2501_001.tar",
        "yymm": "2501",
        "seq_num": "1",
        "first_item": "2501.00001",
        "last_item": "2501.09999",
        "num_items": "9999",
        "size": "511345678",
        "md5sum": "aabbccdd",
        "content_md5sum": "eeff0011",
        "timestamp": "2025-02-04 09:22:11",
    }
    if replace:
        defaults.update(replace)
    inner = "\n".join(
        f"    <{k}>{v}</{k}>" for k, v in defaults.items() if v != ""
    )
    if replace:
        for k, v in replace.items():
            if v == "":
                inner = inner.replace(f"    <{k}>{defaults.get(k, '')}</{k}>\n", "")
    inner += extra_fields
    return f"<arXivSRC>\n  <file>\n{inner}\n  </file>\n</arXivSRC>"


def _make_entry(**kwargs) -> ManifestEntry:
    defaults = dict(
        filename="src/arXiv_src_2501_001.tar",
        yymm="2501",
        seq_num=1,
        first_item="2501.00001",
        last_item="2501.09999",
        num_items=9999,
        size_bytes=511345678,
        md5sum="aabbccdd",
        content_md5sum="eeff0011",
        timestamp="2025-02-04 09:22:11",
    )
    defaults.update(kwargs)
    return ManifestEntry(**defaults)


# ---------------------------------------------------------------------------
# parse_manifest — happy path
# ---------------------------------------------------------------------------


def test_parse_manifest_reads_fixture_file():
    """The hand-written fixture should parse to exactly six entries."""
    xml_text = _FIXTURE_XML.read_text()
    entries = parse_manifest(xml_text)
    assert len(entries) == 6


def test_parse_manifest_all_field_types():
    """Every field of ManifestEntry is populated with the correct Python type."""
    xml_text = _minimal_xml()
    entries = parse_manifest(xml_text)
    assert len(entries) == 1
    e = entries[0]
    assert isinstance(e.filename, str)
    assert isinstance(e.yymm, str)
    assert isinstance(e.seq_num, int)
    assert isinstance(e.first_item, str)
    assert isinstance(e.last_item, str)
    assert isinstance(e.num_items, int)
    assert isinstance(e.size_bytes, int)
    assert isinstance(e.md5sum, str)
    assert isinstance(e.content_md5sum, str)
    assert isinstance(e.timestamp, str)


def test_parse_manifest_field_values():
    """Field values match the fixture XML exactly (no trimming surprises)."""
    xml_text = _minimal_xml()
    e = parse_manifest(xml_text)[0]
    assert e.filename == "src/arXiv_src_2501_001.tar"
    assert e.yymm == "2501"
    assert e.seq_num == 1
    assert e.first_item == "2501.00001"
    assert e.last_item == "2501.09999"
    assert e.num_items == 9999
    assert e.size_bytes == 511345678
    assert e.md5sum == "aabbccdd"
    assert e.content_md5sum == "eeff0011"
    assert e.timestamp == "2025-02-04 09:22:11"


def test_parse_manifest_frozen_dataclass():
    """ManifestEntry is frozen; attribute assignment must raise."""
    e = parse_manifest(_minimal_xml())[0]
    with pytest.raises((AttributeError, TypeError)):
        e.yymm = "9999"  # type: ignore[misc]


def test_parse_manifest_empty_manifest_returns_empty_list():
    """A manifest with no <file> elements is valid and returns []."""
    entries = parse_manifest("<arXivSRC></arXivSRC>")
    assert entries == []


def test_parse_manifest_multiple_entries_count():
    """All six sample-fixture entries are present."""
    entries = parse_manifest(_FIXTURE_XML.read_text())
    yymms = [e.yymm for e in entries]
    assert yymms.count("2501") == 2  # two chunks for January 2025


def test_parse_manifest_yymm_stays_string():
    """yymm is stored as a string; leading-zero month values are preserved."""
    xml = _minimal_xml(replace={"yymm": "0001"})
    e = parse_manifest(xml)[0]
    assert e.yymm == "0001"


# ---------------------------------------------------------------------------
# parse_manifest — error cases
# ---------------------------------------------------------------------------


def test_parse_manifest_raises_on_malformed_xml():
    with pytest.raises(ManifestError, match="malformed"):
        parse_manifest("<arXivSRC><file><unclosed></arXivSRC>")


def test_parse_manifest_raises_on_completely_invalid_xml():
    with pytest.raises(ManifestError):
        parse_manifest("not xml at all <<<>>>")


def test_parse_manifest_raises_when_filename_missing():
    """A <file> missing <filename> must raise ManifestError naming the field."""
    xml = _minimal_xml(replace={"filename": ""})
    with pytest.raises(ManifestError, match="filename"):
        parse_manifest(xml)


def test_parse_manifest_raises_when_yymm_missing():
    xml = _minimal_xml(replace={"yymm": ""})
    with pytest.raises(ManifestError, match="yymm"):
        parse_manifest(xml)


def test_parse_manifest_raises_when_size_missing():
    xml = _minimal_xml(replace={"size": ""})
    with pytest.raises(ManifestError, match="size"):
        parse_manifest(xml)


def test_parse_manifest_raises_when_md5sum_missing():
    xml = _minimal_xml(replace={"md5sum": ""})
    with pytest.raises(ManifestError, match="md5sum"):
        parse_manifest(xml)


def test_parse_manifest_raises_when_seq_num_not_integer():
    """Non-integer <seq_num> text must raise ManifestError."""
    xml = _minimal_xml(replace={"seq_num": "one"})
    with pytest.raises(ManifestError, match="seq_num"):
        parse_manifest(xml)


def test_parse_manifest_raises_when_size_not_integer():
    xml = _minimal_xml(replace={"size": "big"})
    with pytest.raises(ManifestError, match="size"):
        parse_manifest(xml)


def test_parse_manifest_raises_when_num_items_not_integer():
    xml = _minimal_xml(replace={"num_items": "many"})
    with pytest.raises(ManifestError, match="num_items"):
        parse_manifest(xml)


# ---------------------------------------------------------------------------
# select_chunks
# ---------------------------------------------------------------------------


def _sample_entries() -> list[ManifestEntry]:
    """Return the six sample-fixture entries."""
    return parse_manifest(_FIXTURE_XML.read_text())


def test_select_chunks_year_and_month_exact():
    entries = _sample_entries()
    result = select_chunks(entries, year=2501 // 100 + 2000, month=1)
    # year=2025, month=1 → yymm "2501"
    assert len(result) == 2
    assert all(e.yymm == "2501" for e in result)


def test_select_chunks_year_only_returns_all_matching_months():
    entries = _sample_entries()
    result = select_chunks(entries, year=2025)
    # fixture has 2501×2, 2502×1, 2503×1, 2506×1 → 5 entries in 2025
    yymms = {e.yymm for e in result}
    assert "2501" in yymms
    assert "2502" in yymms
    assert "2503" in yymms
    assert "2506" in yymms
    assert len(result) == 5


def test_select_chunks_year_only_excludes_other_years():
    entries = _sample_entries()
    result = select_chunks(entries, year=2025)
    # 2412 = December 2024 → excluded from 2025 window
    result_yymms = [e.yymm for e in result]
    assert "2412" not in result_yymms  # December 2024 → year 2024


def test_select_chunks_2026_year():
    entries = _sample_entries()
    # 2506 = June 2025, not 2026
    result = select_chunks(entries, year=2026)
    assert result == []


def test_select_chunks_empty_result_is_valid():
    """An empty result is valid and not an error."""
    entries = _sample_entries()
    result = select_chunks(entries, year=1999, month=1)
    assert result == []


def test_select_chunks_month_with_no_chunks_is_valid():
    entries = _sample_entries()
    result = select_chunks(entries, year=2025, month=4)  # no April 2025 in fixture
    assert result == []


def test_select_chunks_preserves_entry_order():
    entries = _sample_entries()
    result = select_chunks(entries, year=2025)
    # Should come back in the order they appear in the manifest
    for i in range(len(result) - 1):
        assert entries.index(result[i]) < entries.index(result[i + 1])


def test_select_chunks_single_month_returns_one_chunk():
    entries = _sample_entries()
    result = select_chunks(entries, year=2025, month=2)  # 2502
    assert len(result) == 1
    assert result[0].yymm == "2502"


def test_select_chunks_december_2024():
    entries = _sample_entries()
    result = select_chunks(entries, year=2024, month=12)  # 2412
    assert len(result) == 1
    assert result[0].yymm == "2412"


# ---------------------------------------------------------------------------
# chunks_for_ids
# ---------------------------------------------------------------------------


def test_chunks_for_ids_single_id_in_first_chunk():
    entries = _sample_entries()
    wanted = {"2501.00050"}
    result = chunks_for_ids(entries, wanted)
    assert len(result) == 1
    assert result[0].yymm == "2501"
    assert result[0].seq_num == 1


def test_chunks_for_ids_id_in_second_chunk():
    entries = _sample_entries()
    # 2501.10000 is the first item of chunk 2 (seq_num=2)
    wanted = {"2501.10000"}
    result = chunks_for_ids(entries, wanted)
    assert len(result) == 1
    assert result[0].seq_num == 2


def test_chunks_for_ids_spans_two_chunks_same_month():
    entries = _sample_entries()
    # one id in each of the two 2501 chunks
    wanted = {"2501.00001", "2501.18000"}
    result = chunks_for_ids(entries, wanted)
    assert len(result) == 2
    seq_nums = {e.seq_num for e in result}
    assert seq_nums == {1, 2}


def test_chunks_for_ids_spans_different_months():
    entries = _sample_entries()
    wanted = {"2412.00001", "2502.00500"}
    result = chunks_for_ids(entries, wanted)
    assert len(result) == 2
    yymms = {e.yymm for e in result}
    assert yymms == {"2412", "2502"}


def test_chunks_for_ids_minimal_cover_no_duplicates():
    """Two ids in the same chunk should yield exactly one chunk entry."""
    entries = _sample_entries()
    wanted = {"2501.00001", "2501.05000"}  # both in chunk 1 of 2501
    result = chunks_for_ids(entries, wanted)
    assert len(result) == 1


def test_chunks_for_ids_id_not_in_any_chunk_is_silently_skipped():
    """An id outside all chunk ranges is skipped; no error is raised."""
    entries = _sample_entries()
    # 2504 not in fixture at all
    wanted = {"2504.00001"}
    result = chunks_for_ids(entries, wanted)
    assert result == []


def test_chunks_for_ids_empty_wanted_returns_empty():
    entries = _sample_entries()
    result = chunks_for_ids(entries, set())
    assert result == []


def test_chunks_for_ids_old_style_id_issues_warning():
    """Old-style ids (math/YYYYNNN) must produce a UserWarning and be skipped."""
    entries = _sample_entries()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = chunks_for_ids(entries, {"math/0501123"})
    assert result == []
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "old-style" in str(caught[0].message).lower()


def test_chunks_for_ids_old_style_mixed_with_valid_still_returns_valid():
    """A mix of old-style and new-style ids: old-style warned+skipped, new-style resolved."""
    entries = _sample_entries()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = chunks_for_ids(entries, {"math/0501123", "2501.00001"})
    assert len(result) == 1
    assert result[0].yymm == "2501"
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_chunks_for_ids_result_is_in_manifest_order():
    entries = _sample_entries()
    wanted = {"2506.00001", "2412.00001"}  # reverse of manifest order
    result = chunks_for_ids(entries, wanted)
    assert len(result) == 2
    # 2412 comes before 2506 in the manifest
    assert result[0].yymm == "2412"
    assert result[1].yymm == "2506"


def test_chunks_for_ids_boundary_first_item():
    """An id equal to first_item of a chunk must be included."""
    entries = _sample_entries()
    e0 = entries[0]  # 2412, first_item = "2412.00001"
    result = chunks_for_ids(entries, {e0.first_item})
    assert e0 in result


def test_chunks_for_ids_boundary_last_item():
    """An id equal to last_item of a chunk must be included."""
    entries = _sample_entries()
    e0 = entries[0]  # 2412, last_item = "2412.08754"
    result = chunks_for_ids(entries, {e0.last_item})
    assert e0 in result


# ---------------------------------------------------------------------------
# rollup
# ---------------------------------------------------------------------------


def test_rollup_empty_list():
    r = rollup([])
    assert r.chunk_count == 0
    assert r.total_bytes == 0
    assert r.egress_usd == 0.0


def test_rollup_single_entry():
    e = _make_entry(size_bytes=1_000_000_000)  # exactly 1 GB
    r = rollup([e])
    assert r.chunk_count == 1
    assert r.total_bytes == 1_000_000_000
    assert math.isclose(r.egress_usd, EGRESS_USD_PER_GB, rel_tol=1e-9)


def test_rollup_two_entries_sum():
    entries = [_make_entry(size_bytes=500_000_000), _make_entry(size_bytes=500_000_000)]
    r = rollup(entries)
    assert r.chunk_count == 2
    assert r.total_bytes == 1_000_000_000
    assert math.isclose(r.egress_usd, EGRESS_USD_PER_GB, rel_tol=1e-9)


def test_rollup_uses_decimal_gb_not_binary():
    """1 000 000 000 bytes must cost exactly EGRESS_USD_PER_GB; 1 GiB would be wrong."""
    e = _make_entry(size_bytes=1_000_000_000)
    r = rollup([e])
    binary_cost = 1_000_000_000 / (1024 ** 3) * EGRESS_USD_PER_GB
    decimal_cost = EGRESS_USD_PER_GB
    # The two differ by ~7%; the implementation must use decimal
    assert abs(r.egress_usd - decimal_cost) < abs(r.egress_usd - binary_cost)


def test_rollup_egress_usd_rounded_to_six_decimal_places():
    """egress_usd should be rounded to at most 6 decimal places."""
    e = _make_entry(size_bytes=511_345_678)
    r = rollup([e])
    as_str = f"{r.egress_usd:.10f}"
    # The value after 6 dp should be all zeros (or close enough that repr is clean)
    assert round(r.egress_usd, 6) == r.egress_usd


def test_rollup_all_fixture_entries():
    entries = _sample_entries()
    r = rollup(entries)
    assert r.chunk_count == 6
    expected_bytes = sum(e.size_bytes for e in entries)
    assert r.total_bytes == expected_bytes
    expected_usd = round(expected_bytes / 1e9 * EGRESS_USD_PER_GB, 6)
    assert r.egress_usd == expected_usd


def test_rollup_result_is_frozen_dataclass():
    r = rollup([])
    with pytest.raises((AttributeError, TypeError)):
        r.chunk_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# id_in_range / newstyle_key — public range helpers (reused by the adapter)
# ---------------------------------------------------------------------------


def test_id_in_range_month_straddling_chunk_includes_next_month():
    """The exact case the adapter's own range logic got wrong.

    A chunk that begins in one month and ends in the next must include ids
    from the later month that fall before last_item, and exclude ids from a
    prior month that fall before first_item.
    """
    first, last = "2504.20000", "2505.00050"
    assert id_in_range("2505.00010", first, last) is True
    assert id_in_range("2503.99999", first, last) is False


def test_id_in_range_month_straddling_excludes_beyond_last():
    """An id past last_item in the later month is out of range."""
    first, last = "2504.20000", "2505.00050"
    assert id_in_range("2505.00051", first, last) is False


def test_id_in_range_month_straddling_includes_early_month_within_range():
    """An id in the earlier month at/after first_item is in range."""
    first, last = "2504.20000", "2505.00050"
    assert id_in_range("2504.20000", first, last) is True  # boundary first_item
    assert id_in_range("2504.99999", first, last) is True


def test_id_in_range_inclusive_boundaries():
    first, last = "2501.00001", "2501.09999"
    assert id_in_range(first, first, last) is True
    assert id_in_range(last, first, last) is True


def test_id_in_range_outside_single_month():
    first, last = "2501.00010", "2501.00090"
    assert id_in_range("2501.00009", first, last) is False
    assert id_in_range("2501.00091", first, last) is False


def test_id_in_range_ignores_zero_padding_via_numeric_suffix():
    """Suffix comparison is numeric, so 2 sorts between 1 and 10 despite padding."""
    first, last = "2501.00001", "2501.00010"
    assert id_in_range("2501.00002", first, last) is True


def test_id_in_range_returns_false_when_bounds_not_newstyle():
    """Old-style bounds are out of scope: membership is always False."""
    assert id_in_range("2501.00001", "math/0501001", "math/0501999") is False


def test_newstyle_key_orders_across_month_boundary():
    assert newstyle_key("2504.20000") < newstyle_key("2505.00010")
    assert newstyle_key("2503.99999") < newstyle_key("2504.20000")


def test_newstyle_key_suffix_is_numeric():
    """The suffix component is an int, so padding does not affect ordering."""
    assert newstyle_key("2501.00002") == ("2501", 2)
    assert newstyle_key("2501.00002") < newstyle_key("2501.00010")


# ---------------------------------------------------------------------------
# EGRESS_USD_PER_GB constant
# ---------------------------------------------------------------------------


def test_egress_usd_per_gb_value():
    assert EGRESS_USD_PER_GB == 0.09

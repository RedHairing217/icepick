"""Normalisation onto the canonical handoff record shape."""

from __future__ import annotations

import pytest

from icepick.allocation.adapters import realmath_scrape

# Every top-level key normalise may emit. Source-specific extras must
# land under ``metadata`` instead of growing this set.
CANONICAL_TOP_LEVEL = {
    "source", "provenance", "truth_policy", "statement", "answer", "truth",
    "truth_strings", "arxiv_id", "family", "tier", "label", "pass_at_k",
    "n_correct", "correct", "n_wrong", "wrong", "wrong_complete",
    "n_degenerate", "degenerate", "modal_wrong", "top_wrong_share",
    "params", "metadata",
}


def _normalise(candidates, **overrides):
    raw_outputs = dict(source_name="realmath_2026Q2", candidates=candidates)
    raw_outputs.update(overrides)
    return realmath_scrape.normalise(raw_outputs)


def _upstream_row(**overrides):
    base = {
        "link": "http://arxiv.org/abs/2412.02902v1",
        "question": "Determine the number of rising-continuous functions.",
        "answer": "1",
        "tier": "number",
        "truth": "1",
    }
    base.update(overrides)
    return base


def test_normalise_maps_the_upstream_shape_onto_the_canonical_record():
    result = _normalise([_upstream_row()])
    assert len(result.records) == 1
    record = result.records[0]
    assert record["source"] == "realmath_2026Q2"
    assert record["provenance"] == "extracted"
    assert record["truth_policy"] == "extracted"
    assert record["statement"] == "Determine the number of rising-continuous functions."
    assert record["answer"] == "1"
    assert record["family"] == "realmath"
    assert record["arxiv_id"] == "2412.02902"  # derived from the link, version stripped
    assert record["metadata"]["link"] == "http://arxiv.org/abs/2412.02902v1"


def test_normalise_keeps_a_distinct_truth_field():
    row = _upstream_row(answer="$$x^2$$", truth="x**2")
    record = _normalise([row]).records[0]
    assert record["answer"] == "$$x^2$$"
    assert record["truth"] == "x**2"


def test_normalise_emits_no_new_top_level_fields():
    row = _upstream_row(k=8, samples_used=3, wrong_dist={"0": 2}, scrape_batch=7)
    record = _normalise([row]).records[0]
    assert set(record) <= CANONICAL_TOP_LEVEL
    assert record["metadata"]["k"] == 8
    assert record["metadata"]["wrong_dist"] == {"0": 2}
    assert record["metadata"]["scrape_batch"] == 7


def test_normalise_rejects_records_without_usable_statements():
    result = _normalise([{"answer": "42"}, _upstream_row()])
    assert len(result.records) == 1
    assert len(result.quarantined) == 1
    assert result.quarantined[0]["reason"] == "missing statement"


def test_normalise_deduplicates_statements_before_handoff():
    result = _normalise([_upstream_row(), _upstream_row(link="http://arxiv.org/abs/2412.02902v2")])
    assert len(result.records) == 1
    assert result.duplicates_dropped == 1


def test_normalise_never_labels_computed_records_as_extracted():
    rows = [
        _upstream_row(provenance="computed"),
        _upstream_row(question="A generated variant.", generated=True),
    ]
    result = _normalise(rows)
    assert [r["provenance"] for r in result.records] == ["computed", "computed"]
    # Computed truth was produced at harvest: trusted, not extracted.
    assert all(r["truth_policy"] == "trusted" for r in result.records)


def test_normalise_quarantines_unknown_provenance_instead_of_guessing():
    result = _normalise([_upstream_row(provenance="scraped_maybe")])
    assert result.records == []
    assert "unknown provenance" in result.quarantined[0]["reason"]


def test_normalise_stamps_a_narrower_family_when_one_is_requested():
    record = _normalise([_upstream_row()], families=["number_theory"]).records[0]
    assert record["family"] == "number_theory"


def test_normalise_defaults_to_realmath_when_families_are_ambiguous():
    record = _normalise([_upstream_row()], families=["algebra", "analysis"]).records[0]
    assert record["family"] == "realmath"


def test_normalise_prefers_the_record_level_family():
    record = _normalise([_upstream_row(family="p_adic")], families=["number_theory"]).records[0]
    assert record["family"] == "p_adic"


def test_normalise_honours_a_truth_policy_override():
    record = _normalise([_upstream_row()], truth_policy="unknown").records[0]
    assert record["truth_policy"] == "unknown"


def test_normalise_warns_on_missing_arxiv_id_and_junk_answers():
    rows = [
        _upstream_row(link="", question="No paper reference here."),
        _upstream_row(question="Junk answer form.", answer="\\mathbb{Z}[i]"),
    ]
    result = _normalise(rows)
    assert len(result.records) == 2
    assert any("no arxiv_id" in w for w in result.warnings)
    assert any("answer" in w and "form" in w for w in result.warnings)


def test_normalise_refuses_unknown_raw_outputs_fields():
    with pytest.raises(ValueError, match="unknown raw_outputs fields"):
        _normalise([], output_dir="out")


def test_normalise_takes_the_truth_as_answer_when_no_answer_is_given():
    row = {"question": "Sum the first five primes.", "truth": "28",
           "link": "http://arxiv.org/abs/2501.99999v1"}
    record = _normalise([row]).records[0]
    assert record["answer"] == "28"
    assert "truth" not in record


def test_normalise_drops_a_truth_that_only_differs_in_type():
    record = _normalise([_upstream_row(answer=1, truth="1")]).records[0]
    assert record["answer"] == 1
    assert "truth" not in record


def test_normalise_quarantines_non_object_metadata_without_aborting_the_batch():
    result = _normalise([_upstream_row(metadata=5), _upstream_row(question="A survivor.")])
    assert len(result.records) == 1
    assert "metadata must be a JSON object" in result.quarantined[0]["reason"]


def test_normalise_falls_through_a_whitespace_only_statement():
    row = {"statement": "  ", "question": "The real question?", "answer": "1",
           "link": "http://arxiv.org/abs/2501.99999v1"}
    record = _normalise([row]).records[0]
    assert record["statement"] == "The real question?"


def test_normalise_strips_url_fragments_from_derived_arxiv_ids():
    row = _upstream_row(link="http://arxiv.org/abs/2412.02902v1#page=3")
    assert _normalise([row]).records[0]["arxiv_id"] == "2412.02902"


def test_normalise_ignores_arxiv_lookalike_hosts():
    row = _upstream_row(link="http://fakearxiv.org/abs/1234.5678v1")
    result = _normalise([row])
    assert "arxiv_id" not in result.records[0]
    assert any("no arxiv_id" in w for w in result.warnings)

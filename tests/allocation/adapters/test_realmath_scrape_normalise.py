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


# --- qa_ref_guard (E2) + elision_signals (E4) ----------------------------------


def _ref_row(metadata, **overrides):
    row = _upstream_row(metadata=metadata)
    row.update(overrides)
    return row


def test_normalise_defaults_qa_ref_guard_to_advisory():
    row = _ref_row({"has_external_refs": True, "unresolved_refs": ["eq:x"]})
    result = _normalise([row])  # no qa_ref_guard override
    guard = result.records[0]["metadata"]["quality_guard"]
    assert guard == {"unresolved_external_refs": True, "policy": "advisory"}
    assert result.guard_flagged == 1


def test_normalise_advisory_annotates_and_counts_unresolved_refs():
    row = _ref_row({"has_external_refs": True, "unresolved_refs": ["eq:x"]})
    result = realmath_scrape.normalise(
        {"source_name": "s", "candidates": [row]}, qa_ref_guard="advisory",
    )
    assert len(result.records) == 1
    assert result.quarantined == []
    guard = result.records[0]["metadata"]["quality_guard"]
    assert guard["unresolved_external_refs"] is True
    assert guard["policy"] == "advisory"
    assert result.guard_flagged == 1


def test_normalise_advisory_flags_a_row_with_no_resolution_attempted_at_all():
    """has_external_refs true, no resolved_refs key at all (e.g. a row from
    before E5, or where resolution failed to run) — still flagged."""
    row = _ref_row({"has_external_refs": True})
    result = realmath_scrape.normalise(
        {"source_name": "s", "candidates": [row]}, qa_ref_guard="advisory",
    )
    assert result.records[0]["metadata"]["quality_guard"]["unresolved_external_refs"] is True


def test_normalise_strict_quarantines_unresolved_refs_with_quality_guard_tag():
    row = _ref_row({"has_external_refs": True, "unresolved_refs": ["eq:x"]})
    result = realmath_scrape.normalise(
        {"source_name": "s", "candidates": [row]}, qa_ref_guard="strict",
    )
    assert result.records == []
    assert len(result.quarantined) == 1
    assert result.quarantined[0]["reason"] == "[quality-guard] unresolved external refs in source"
    assert result.guard_flagged == 1


def test_normalise_off_policy_ignores_unresolved_refs():
    row = _ref_row({"has_external_refs": True, "unresolved_refs": ["eq:x"]})
    result = realmath_scrape.normalise(
        {"source_name": "s", "candidates": [row]}, qa_ref_guard="off",
    )
    assert len(result.records) == 1
    assert "quality_guard" not in result.records[0].get("metadata", {})
    assert result.guard_flagged == 0


def test_normalise_fully_resolved_refs_row_is_not_flagged():
    """resolution cured it: resolved_refs non-empty, unresolved_refs empty."""
    row = _ref_row({"has_external_refs": True, "resolved_refs": {"eq:x": "E=mc^2"}})
    result = realmath_scrape.normalise(
        {"source_name": "s", "candidates": [row]}, qa_ref_guard="advisory",
    )
    assert len(result.records) == 1
    assert "quality_guard" not in result.records[0].get("metadata", {})
    assert result.guard_flagged == 0


def test_normalise_partially_resolved_refs_row_is_still_flagged():
    row = _ref_row({
        "has_external_refs": True,
        "resolved_refs": {"eq:x": "E=mc^2"},
        "unresolved_refs": ["eq:y"],
    })
    result = realmath_scrape.normalise(
        {"source_name": "s", "candidates": [row]}, qa_ref_guard="advisory",
    )
    assert result.records[0]["metadata"]["quality_guard"]["unresolved_external_refs"] is True


def test_normalise_refuses_unknown_qa_ref_guard_value():
    with pytest.raises(ValueError, match="qa_ref_guard"):
        realmath_scrape.normalise(
            {"source_name": "s", "candidates": []}, qa_ref_guard="paranoid",
        )


def test_normalise_elision_signals_attach_advisory_on_a_hole_bigram_row():
    row = _upstream_row(question="The solution of, must vanish at infinity.")
    result = _normalise([row])
    signals = result.records[0]["metadata"]["quality_guard"]["elision_signals"]
    assert signals.get("solution_of_comma") == 1
    assert result.guard_flagged == 1


def test_normalise_elision_signals_apply_even_when_ref_guard_is_off():
    """E4 is independent of the qa_ref_guard policy: always-on advisory."""
    row = _upstream_row(question="The solution of, must vanish at infinity.")
    result = realmath_scrape.normalise(
        {"source_name": "s", "candidates": [row]}, qa_ref_guard="off",
    )
    assert result.records[0]["metadata"]["quality_guard"]["elision_signals"]
    assert result.guard_flagged == 1


def test_normalise_elision_signals_never_quarantine_on_their_own():
    row = _upstream_row(question="The solution of, must vanish at infinity.")
    result = realmath_scrape.normalise(
        {"source_name": "s", "candidates": [row]}, qa_ref_guard="strict",
    )
    assert result.quarantined == []
    assert len(result.records) == 1


def test_normalise_merges_ref_guard_and_elision_signals_in_one_block():
    row = _ref_row(
        {"has_external_refs": True, "unresolved_refs": ["eq:x"]},
        question="The solution of, is unique.",
    )
    result = realmath_scrape.normalise(
        {"source_name": "s", "candidates": [row]}, qa_ref_guard="advisory",
    )
    guard = result.records[0]["metadata"]["quality_guard"]
    assert guard["unresolved_external_refs"] is True
    assert guard["elision_signals"]["solution_of_comma"] == 1
    assert result.guard_flagged == 1  # one row flagged, not double-counted


def test_normalise_clean_row_gets_no_quality_guard_block():
    result = _normalise([_upstream_row()])
    assert "quality_guard" not in result.records[0].get("metadata", {})
    assert result.guard_flagged == 0


@pytest.mark.parametrize(
    "text,pattern_name",
    [
        ("The solution of, must vanish.", "solution_of_comma"),
        ("A subsolution to; is bounded.", "solution_of_comma"),
        ("A solution of such that data is given.", "solution_of_conjunction"),
        ("A solution to with prescribed data.", "solution_of_conjunction"),
        ("Consider the system - with initial data.", "system_dash"),
        ("Consider the system — and boundary data.", "system_dash"),
        ("The assumptions and hold throughout.", "assumptions_hold"),
        ("The assumption be satisfied everywhere.", "assumptions_hold"),
        (r"a weak solution of, \begin{equation}", "dangling_preposition_equation"),
    ],
)
def test_elision_signals_detects_each_hole_bigram(text, pattern_name):
    signals = realmath_scrape.elision_signals(text)
    assert signals.get(pattern_name) == 1


def test_elision_signals_is_empty_for_clean_prose():
    assert realmath_scrape.elision_signals("A clean, self-contained statement.") == {}


def test_elision_signals_handles_none_and_empty_input():
    assert realmath_scrape.elision_signals("") == {}
    assert realmath_scrape.elision_signals(None) == {}

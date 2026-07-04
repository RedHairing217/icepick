from claude_poser.config import WellposedConfig
from claude_poser.schema import normalise_record
from claude_poser.wellposed import check_record


def _norm(rid, **kw):
    return normalise_record(kw, rid)


def test_computed_provenance_short_circuits_to_pass():
    rec = _norm(0, source="calc", provenance="computed",
                statement="See Theorem 3.2 from above")  # dangling text ignored
    result = check_record(rec, WellposedConfig())
    assert result["tier"] == "code"
    assert result["wellposed_status"] == "pass"
    assert result["wellposed_score"] == 1.0


def test_trusted_manual_short_circuits_to_pass():
    rec = _norm(0, source="manual", provenance="manual", truth_policy="trusted",
                statement="see Equation (3.2)")
    result = check_record(rec, WellposedConfig())
    assert result["wellposed_status"] == "pass"


def test_extracted_clean_passes():
    rec = _norm(0, source="rm", provenance="extracted",
                statement="Prove that n^2 - n is even for all positive integers n.")
    result = check_record(rec, WellposedConfig())
    assert result["wellposed_status"] == "pass"


def test_extracted_dangling_without_judge_flags():
    rec = _norm(0, source="rm", provenance="extracted",
                statement="Using Theorem 3.2 from the previous section, deduce A.")
    cfg = WellposedConfig(enable_judge=False)
    result = check_record(rec, cfg)
    assert result["tier"] == "code"
    assert result["wellposed_status"] == "flag"
    assert result["wellposed_score"] == 0.0
    assert result["code_hits"]


def test_unknown_provenance_is_conservative():
    rec = _norm(0, source="???", provenance="unknown",
                statement="see Equation (4.7)")
    result = check_record(rec, WellposedConfig(enable_judge=False))
    assert result["wellposed_status"] == "flag"

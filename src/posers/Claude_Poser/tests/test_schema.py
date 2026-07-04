from claude_poser.schema import compute_uid, normalise_record


def test_uid_stable_across_order():
    a = compute_uid("rm", "Prove n^2 - n is even.")
    b = compute_uid("rm", "Prove n^2 - n is even.")
    assert a == b


def test_uid_depends_on_source():
    assert compute_uid("rm", "x") != compute_uid("calc", "x")


def test_unknown_provenance_normalised():
    rec = normalise_record({"source": "x", "statement": "y", "provenance": "bogus"}, rid=0)
    assert rec["provenance"] == "unknown"


def test_missing_source_defaults_to_unknown():
    rec = normalise_record({"statement": "x"}, rid=0)
    assert rec["source"] == "unknown"

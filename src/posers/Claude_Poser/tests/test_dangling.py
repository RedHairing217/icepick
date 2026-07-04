from claude_poser import dangling


def test_named_reference_trips():
    assert dangling.scan("Using Theorem 3.2 from earlier, deduce x.")


def test_directional_anaphor_trips():
    assert dangling.scan("As defined above, compute phi(1).")


def test_previous_item_trips():
    assert dangling.scan("In the previous problem, we set n = 5.")


def test_bare_equation_label_trips():
    assert dangling.scan("Substitute into (3.2) and simplify.")


def test_clean_statement_does_not_trip():
    statement = "Let n be a positive integer. Prove that n^2 - n is even."
    assert dangling.scan(statement) == []


def test_directional_value_does_not_trip():
    statement = "Find the smallest integer above 7 that is prime."
    assert dangling.scan(statement) == []


# -------------------------------------------------------------------------- #
# Coverage added after the 0/70-hit realmath regression. Each of these was a
# false-negative under the old regex.
# -------------------------------------------------------------------------- #


def test_bare_theorem_integer_trips():
    """Old scanner required a decimal on _NOUN_NUMBER — 'Theorem 3' slipped."""
    hits = dangling.scan("Prove that Theorem 3 holds.")
    assert hits
    assert any(h.pattern == "noun_number" for h in hits)


def test_bare_integer_paren_at_sentence_end_trips():
    """'equation (3).' is an eq ref; 'f(3)' mid-statement is not."""
    assert dangling.scan("Substitute into equation (3).")
    # Not-a-hit control: bare f(3) mid-sentence must NOT flag.
    assert dangling.scan("Compute the value of f(3) explicitly.") == []


def test_latex_ref_macro_trips():
    for macro in (r"\ref{thm:main}", r"\cref{lem:aux}", r"\eqref{eq:key}", r"\Cref{sec:setup}"):
        stmt = f"See {macro} for details."
        hits = dangling.scan(stmt)
        assert hits, f"expected {macro} to trip"
        assert any(h.pattern == "latex_ref" for h in hits)


def test_unresolved_ref_marker_trips():
    assert dangling.scan("As shown in [?], we conclude.")
    assert dangling.scan("Using {?}, we have...")


def test_broader_directional_verbs_trip():
    for verb in ("constructed", "obtained", "assumed", "derived", "considered"):
        stmt = f"As {verb} above, we have x = 0."
        hits = dangling.scan(stmt)
        assert hits, f"expected 'as {verb} above' to trip"


def test_meta_source_reference_trips():
    for stmt in (
        "The main result of the paper implies...",
        "In our paper, we prove...",
        "The following theorem holds:",
        "Using standard notation...",
    ):
        assert dangling.scan(stmt), f"expected meta ref to trip in: {stmt!r}"


def test_latex_math_delimiters_do_not_leak_false_positives():
    """LaTeX math like $\\mathbb{F}$ or $\\ref$ inside math mode shouldn't
    accidentally trip the ref detector when there's no actual \\ref{...}."""
    # A statement using LaTeX for math but no cross-refs should be clean.
    stmt = r"Let $\mathbb{F}$ be a field. Compute $l(\mathbb{F})$ for $k \geq 2$."
    hits = dangling.scan(stmt)
    # We accept zero hits here — this is a paper-notation problem the scanner
    # genuinely can't catch (semantic, not lexical). The judge will handle it.
    assert hits == []

"""Verifier audit, ported from ModelBreaker ``realmath/verifier_audit.py``.

Verifier integrity splits in two. The form-equivalence direction — the
verifier scoring an equivalent answer form as wrong — is a property of
``verifier.py`` rather than of any problem: if its symbolic equality is sound,
no per-problem pass can exhibit a form-equivalence flip, so that direction is
discharged once by the form-equivalence suite here. The numeric-sampling
fallback direction — two different expressions coinciding at the sampled
points so a wrong answer scores correct — is characterised by the coincidence
suite: ``proposed_numeric_equiv`` lives IN THIS TEST FILE because it is an
audit artifact (a proposal MB never adopted), not production code; production
``verify()`` is fully symbolic and must keep distinguishing the pairs the
numeric sampler false-accepts.

Edge-case tests pin the classify/verify contract the pass@k runner leans on:
unverifiable tiers, delimiter stripping, rhs extraction, tuple arity.
"""

from __future__ import annotations

import random

import pytest
import sympy as sp

from icepick.processing.pass_at_k import verifier as V

# --- audit cases, unmodified in substance from MB's verifier_audit.py --------

FORM_EQUIVALENCE_CASES = [
    ("number_unreduced", "expr", "1/2", "2/4"),
    ("number_sqrt", "number", "2", "sqrt(4)"),
    ("expr_factored", "expr", "x**2 - 1", "(x - 1)*(x + 1)"),
    ("expr_reordered", "expr", "a + b", "b + a"),
    ("expr_distributed", "expr", "2*x + 2*y", "2*(x + y)"),
    ("tuple_unreduced", "tuple", "(1, 2)", "(2/2, 4/2)"),
]

FALLBACK_COINCIDENCE_CASES = [
    ("vanishes_on_2_to_9", "x",
     "x + (x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)"),
    ("scaled_vanisher", "2*x",
     "2*x + (x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)"),
]


def proposed_numeric_equiv(a, b, lo, hi, trials, seed, tol=1e-6):
    """Proposed seeded numeric fallback. Not called by production verify().

    Samples integer points in [lo, hi] for the shared free symbols and reports
    whether the two expressions agree at every sampled point. Agreement is
    necessary but not sufficient for symbolic equality, so this can
    false-accept a coincident pair. The seed keeps the verdict reproducible.
    """
    if a is None or b is None:
        return None
    rng = random.Random(seed)
    syms = sorted(a.free_symbols | b.free_symbols, key=str)
    seen = 0
    for _ in range(trials * 4):
        if seen >= trials:
            break
        subs = {s: rng.randint(lo, hi) for s in syms}
        try:
            va = complex(a.evalf(subs=subs))
            vb = complex(b.evalf(subs=subs))
        except Exception:
            continue
        if va != va or vb != vb:
            continue
        seen += 1
        if abs(va - vb) > tol * (1 + abs(va)):
            return False
    return True if seen else None


# --- form-equivalence suite ---------------------------------------------------


@pytest.mark.parametrize(
    "label, tier, truth_src, candidate",
    FORM_EQUIVALENCE_CASES,
    ids=[c[0] for c in FORM_EQUIVALENCE_CASES],
)
def test_form_equivalence(label, tier, truth_src, candidate):
    """Equivalent forms must verify as equal across the verifier tiers."""
    classified_tier, obj = V.classify(truth_src)
    assert obj is not None, f"{label}: classify({truth_src!r}) gave no object"
    assert V.verify(candidate, obj, tier) is True


# --- numeric-fallback coincidence characterisation ----------------------------


@pytest.mark.parametrize(
    "label, truth_src, candidate_src",
    FALLBACK_COINCIDENCE_CASES,
    ids=[c[0] for c in FALLBACK_COINCIDENCE_CASES],
)
def test_numeric_fallback_false_accepts_where_symbolic_holds(
    label, truth_src, candidate_src
):
    """Numeric sampling on [2, 9] false-accepts; simplify and verify() do not."""
    a = V.parse_expr(truth_src)
    b = V.parse_expr(candidate_src)
    assert a is not None and b is not None

    numeric = proposed_numeric_equiv(a, b, lo=2, hi=9, trials=6, seed=0)
    symbolic = bool(sp.simplify(a - b) == 0)
    assert numeric is True, f"{label}: sampler should coincide on [2, 9]"
    assert symbolic is False, f"{label}: simplify should distinguish the pair"

    tier, obj = V.classify(truth_src)
    assert tier == "expr"
    assert V.verify(candidate_src, obj, tier) is False


# --- edge cases ---------------------------------------------------------------


def test_empty_candidate_is_false():
    tier, obj = V.classify("5")
    assert tier == "number"
    assert V.verify("", obj, tier) is False
    assert V.verify(None, obj, tier) is False


def test_unverifiable_truth_tiers_classify():
    tier, obj = V.classify("the answer is undefined")
    assert (tier, obj) == ("prose", None)

    tier, obj = V.classify(r"\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}")
    assert (tier, obj) == ("piecewise", None)

    tier, obj = V.classify("(a) $x = 1$ (b) $y = 2$")
    assert (tier, obj) == ("multi", None)


def test_tuple_length_mismatch_is_false():
    tier, obj = V.classify("(1, 2)")
    assert tier == "tuple"
    assert V.verify("(1, 2, 3)", obj, tier) is False
    assert V.verify("(1,)", obj, tier) is False


def test_strip_delims_variants():
    assert V.strip_delims("$x + 1$") == "x + 1"
    assert V.strip_delims("$$x + 1$$") == "x + 1"
    assert V.strip_delims(r"\[x + 1\]") == "x + 1"
    assert V.strip_delims(r"\(x + 1\)") == "x + 1"
    # And end-to-end: a delimited candidate still verifies.
    tier, obj = V.classify("5")
    assert V.verify("$5$", obj, tier) is True


def test_rhs_extraction_verifies():
    tier, obj = V.classify("5")
    assert tier == "number"
    assert V.verify("x = 5", obj, tier) is True


def test_verify_false_for_nonverifiable_tiers():
    for tier in ("prose", "piecewise", "multi", "empty", "reject"):
        assert V.verify("5", None, tier) is False


# --- infinity guard (release checkbox R4) --------------------------------
#
# simplify(candidate - truth) == 0 breaks when either side is infinite:
# oo - oo is nan, and nan == 0 is False, so an infinity-valued answer
# silently failed to self-verify (censused: 21 non-drop corpus records, 18
# misdirection + 3 collapse, all `fail` labels that were grader artifacts —
# see out/analysis/verifier_fix_20260801/). The corpus-observed answer
# strings below are pinned verbatim from a grep of
# wellposed_all_with_passk.json, not invented.

INFINITY_SELF_VERIFY_CASES = [
    ("unsigned", r"$\infty$"),
    ("positive", r"$+\infty$"),
    ("negative", r"$-\infty$"),
    ("positive_with_rhs_prefix", r"$p_N = +\infty$"),
]


@pytest.mark.parametrize(
    "label, answer",
    INFINITY_SELF_VERIFY_CASES,
    ids=[c[0] for c in INFINITY_SELF_VERIFY_CASES],
)
def test_infinity_self_verifies(label, answer):
    """An infinity-valued truth must verify against itself (the R4 defect)."""
    tier, obj = V.classify(answer)
    assert tier == "number"
    assert V.verify(answer, obj, tier) is True


def test_infinity_vs_finite_is_false():
    tier, obj = V.classify(r"$+\infty$")
    assert V.verify("5", obj, tier) is False

    tier, obj = V.classify("5")
    assert V.verify(r"$+\infty$", obj, tier) is False


def test_positive_vs_negative_infinity_is_false():
    tier, obj = V.classify(r"$+\infty$")
    assert V.verify(r"$-\infty$", obj, tier) is False

    tier, obj = V.classify(r"$-\infty$")
    assert V.verify(r"$+\infty$", obj, tier) is False


def test_infinity_minus_infinity_nan_resolves_false_not_exception():
    """Pin the historical failure mode directly, then confirm verify() no
    longer inherits it: oo - oo is nan and nan == 0 is False, so comparing
    two *unequal* infinities must still come back False, cleanly."""
    assert sp.simplify(sp.oo - sp.oo) is sp.nan
    assert bool(sp.nan == 0) is False

    tier, obj = V.classify(r"$+\infty$")
    assert obj is sp.oo
    assert V.verify(r"$-\infty$", obj, tier) is False


def test_nan_candidate_against_finite_truth_is_false():
    """A candidate that parses directly to nan (not via oo - oo) must not
    crash verify() either — nan is neither infinite nor equal to anything."""
    tier, obj = V.classify("5")
    assert V.verify("nan", obj, tier) is False


# --- finite-answer regressions (unchanged by the R4 guard) ----------------


def test_finite_number_tier_regression():
    tier, obj = V.classify("5")
    assert tier == "number"
    assert V.verify("5", obj, tier) is True
    assert V.verify("6", obj, tier) is False


def test_finite_latex_number_tier_regression():
    """A LaTeX-marked finite answer routes through parse_latex, not
    sympify; the R4 guard must not touch this path."""
    tier, obj = V.classify(r"$\frac{1}{2}$")
    assert tier == "number"
    assert V.verify(r"$\frac{1}{2}$", obj, tier) is True
    assert V.verify(r"$\frac{1}{3}$", obj, tier) is False


def test_finite_expr_tier_regression():
    tier, obj = V.classify("x**2 - 1")
    assert tier == "expr"
    assert V.verify("(x - 1)*(x + 1)", obj, tier) is True
    assert V.verify("x**2 + 1", obj, tier) is False

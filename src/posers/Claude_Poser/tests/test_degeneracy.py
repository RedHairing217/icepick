"""Degeneracy scanner tests.

The three positive fixtures are lightly trimmed versions of the real
records the 2026-07-04 stage-3 kill analysis ruled degenerate_circular
(uids 05ed3495d2c3, 7eccee453a96, b4ef435b1e8a in
out/wellposed_pde625_claude_anthropic/stage3_kill_analysis.jsonl).
"""

from claude_poser import degeneracy
from claude_poser.degeneracy import normalise_math, scan


def _patterns(hits):
    return {h.pattern for h in hits}


def test_answer_in_statement_eigenvalues():
    stmt = (
        "Let $T$ be a matrix such that the eigenvalues of $T^{-H}T^{-1}$ are "
        "$(3\\pm\\sqrt{5})/2$, each with multiplicity two. What are these eigenvalues?"
    )
    ans = "$\\dfrac{3\\pm\\sqrt{5}}{2}$"
    assert "answer_in_statement" in _patterns(scan(stmt, ans))


def test_defined_then_asked_ratio():
    stmt = (
        "Let $R(p/q)$ denote the quantity defined so that $R(p/q) = \\frac{1}{L(p/q)}$, "
        "where $L(p/q) = \\sqrt{p^2 + q^2}$ for integers $p, q$. What is $R(p/q)$?"
    )
    ans = "$\\dfrac{1}{\\sqrt{p^2+q^2}}$"
    assert "defined_then_asked" in _patterns(scan(stmt, ans))


def test_defined_then_asked_alpha():
    stmt = (
        "Let $1 < p < N$ and set $\\alpha := \\frac{N-p}{p-1}$. Suppose $U$ decays at "
        "the rate of the fundamental solution. What is the exact decay exponent "
        "$\\alpha$ (i.e., the value such that $U(x) \\sim |x|^{-\\alpha}$)?"
    )
    ans = "$\\alpha = \\dfrac{N-p}{p-1}$"
    hits = _patterns(scan(stmt, ans))
    assert hits & {"defined_then_asked", "answer_in_statement"}


def test_ubiquitous_answers_never_fire():
    stmt = "If $\\mathrm{Tax}(D)=0$ and $\\mathrm{Tax}(D) \\ge c\\sum_k \\|D_k\\|$ with $c>0$, what is $\\|D_k\\|$?"
    assert scan(stmt, "$0$") == []
    # \infty appears in most analysis statements ("T^* < \infty").
    stmt2 = "Let $T^* < \\infty$ be maximal. What is $\\limsup_{t \\nearrow T^*} \\int_0^t \\|\\omega\\|_{L^\\infty} d\\tau$?"
    assert scan(stmt2, "$\\infty$") == []


def test_ordinary_record_no_hits():
    stmt = (
        "Let $\\Omega \\subseteq \\mathbb{R}^3$ and $f \\in H^1_0(\\Omega)$. Into which "
        "Lorentz space $L^{p,q}(\\Omega)$ does every such $f$ embed?"
    )
    assert scan(stmt, "$L^{6,2}(\\Omega)$") == []


def test_short_constraint_symbol_does_not_false_fire():
    # "K" appears next to "=" in a constraint-flavoured display; single-letter
    # targets must not fire on bare "=".
    stmt = (
        "Suppose $\\|z\\| \\le C\\|Oz\\|$ and the error satisfies "
        "$\\varepsilon \\le \\frac{1}{2C}$. Then $\\|z\\| \\le K \\|O_R z\\|$. "
        "What is the constant $K$?"
    )
    assert scan(stmt, "$2C$") == []


def test_missing_answer_or_statement_is_quiet():
    assert scan("", "$1$") == []
    assert scan("What is $x$?", "") == []


def test_normalise_math_frac_variants_collide():
    assert normalise_math("$\\dfrac{3\\pm\\sqrt{5}}{2}$") == normalise_math("(3\\pm\\sqrt{5})/2")
    assert normalise_math("\\frac{N-p}{p-1}") == normalise_math("$\\dfrac{N-p}{p-1}$")


def test_scan_results_serialisable():
    stmt = "Set $\\beta := x^2+y^2$. What is $\\beta$?"
    hits = scan(stmt, "$x^2 + y^2$")
    assert hits
    d = hits[0].to_dict()
    assert set(d) == {"pattern", "snippet"}

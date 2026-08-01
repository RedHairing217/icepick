"""Deterministic answer verifier: classify the truth once, verify candidates against it.

Ported from ModelBreaker ``realmath/verifier.py`` with the argparse/JSONL dataset
plumbing dropped — this is a library module consumed by the pass@k runner, not a
corpus-building CLI.

The shape is classify-then-verify on purpose. A record's truth answer is parsed
exactly once by :func:`classify` into ``(tier, obj)``; each of the k rollout
candidates is then verified against that already-parsed sympy object. Parsing
the truth per-candidate would repeat the expensive step k times and — worse —
could disagree with itself across rollouts if parsing were ever ambiguous. The
tiers split verifiable answers (``number``, ``tuple``, ``expr``) from shapes no
symbolic check can score (``empty``, ``piecewise``, ``multi``, ``prose``,
``reject``); :func:`verify` returns False for anything outside the verifiable
tiers, and the runner drops such records up front (``DROP_UNVERIFIABLE``).

Verification is fully symbolic — ``simplify(candidate - truth) == 0`` — with no
numeric sampling. That is deliberate: this verifier replaces RealMath's LLM
judge, and MB's audit (see ``tests/processing/pass_at_k/test_verifier.py``)
characterises why a numeric-sampling fallback false-accepts expressions that
merely coincide at the sampled points.

One deliberate deviation from the MB source. MB's ``parse_expr`` tried
``parse_latex`` first and fell back to ``sympify``; in MB's runtime the ANTLR
LaTeX backend was absent (``HAVE_LATEX`` False) so everything went through
``sympify`` and the audit passed. In icepick's environment ``parse_latex``
works — and on plain sympy syntax it mis-parses *without raising*:
``sqrt(4)`` becomes the product ``s*q*r*t*(4)`` and ``x**2 - 1`` silently
truncates to ``x``, flipping MB's form-equivalence audit cases from pass to
fail. ``parse_expr`` therefore routes on surface syntax: strings carrying LaTeX
markers (backslash, ``^``, ``{``) go to ``parse_latex`` first, plain strings go
to ``sympify`` first, and each parser remains the other's fallback.

A second deviation, found by census rather than by port: ``simplify(candidate
- truth) == 0`` breaks when either side is infinite. ``oo - oo`` (and
``-oo - -oo``) is ``nan``, and ``nan == 0`` is ``False`` — not an exception —
so an infinity-valued answer silently failed to verify even against itself.
``verify`` special-cases infinite operands: when either side's ``is_infinite``
is true, it compares the two sympy objects directly (``oo == oo``,
``oo != -oo``, ``oo != zoo``, ``oo != 5``) instead of subtracting. Finite
answers are untouched — the subtraction path still runs unchanged for
everything that isn't infinite.
"""

from __future__ import annotations

import re

from sympy import Tuple, nsimplify, simplify, sympify

try:
    from sympy.parsing.latex import parse_latex

    HAVE_LATEX = True
except Exception:  # pragma: no cover - environment-dependent
    HAVE_LATEX = False


PROSE_RE = re.compile(r"[A-Za-z]{3,}")
TUPLE_RE = re.compile(r"^\(\s*[-+0-9.,\s/]+\)$")

# Surface markers that mean "this string is LaTeX, not sympy syntax".
_LATEX_MARKER_RE = re.compile(r"[\\^{]")


def strip_delims(s):
    s = (s or "").strip()
    for a, b in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if s.startswith(a) and s.endswith(b):
            s = s[len(a):-len(b)].strip()
    s = s.strip("$").strip()
    return s


def english_words(s):
    s2 = re.sub(r"\\text\{[^}]*\}", " ", s)
    s2 = re.sub(r"\\[a-zA-Z]+", " ", s2)
    return PROSE_RE.findall(s2)


def has_cases(s):
    return "\\begin{cases}" in s or "\\begin{dcases" in s


def multi_part(s):
    parts = re.findall(r"\(\s*[a-d]\s*\)", s)
    return len(parts) >= 2


def rhs(s):
    s2 = re.sub(r"\\(leq|geq|neq|equiv|approx|sim|le|ge)\b", " ", s)
    if "=" in s2:
        tail = s2.rsplit("=", 1)[1].strip()
        if tail:
            return tail
    return s


def parse_expr(s):
    s = s.strip()
    looks_latex = bool(_LATEX_MARKER_RE.search(s))
    if HAVE_LATEX and looks_latex:
        try:
            return parse_latex(s)
        except Exception:
            pass
    try:
        return sympify(s)
    except Exception:
        pass
    if HAVE_LATEX and not looks_latex:
        try:
            return parse_latex(s)
        except Exception:
            pass
    return None


def as_number(s):
    e = parse_expr(s)
    if e is None:
        return None
    try:
        if e.is_number:
            return e
    except Exception:
        pass
    return None


def as_tuple(s):
    if not TUPLE_RE.match(s):
        return None
    inner = s[1:-1]
    items = [p.strip() for p in inner.split(",") if p.strip()]
    vals = []
    for it in items:
        try:
            vals.append(sympify(it))
        except Exception:
            return None
    return Tuple(*vals)


def classify(answer):
    raw = answer or ""
    s = strip_delims(raw)
    if not s:
        return "empty", None
    if has_cases(raw):
        return "piecewise", None
    if multi_part(raw) or raw.count("$$") >= 3:
        return "multi", None

    t = as_tuple(s)
    if t is not None:
        return "tuple", t

    r = rhs(s)
    n = as_number(r)
    if n is not None:
        return "number", n

    wc = len(english_words(s))
    if wc >= 2:
        return "prose", None

    e = parse_expr(r)
    if e is not None:
        return "expr", e
    return "reject", None


def verify(candidate, truth_obj, tier):
    c = strip_delims(candidate or "")
    if tier in ("number", "expr"):
        ce = parse_expr(rhs(c))
        if ce is None:
            return False
        try:
            infinite = bool(ce.is_infinite or truth_obj.is_infinite)
        except Exception:
            infinite = False
        if infinite:
            # oo - oo is nan, not 0, so the subtraction below can't be used
            # for infinite operands; compare them directly instead.
            return bool(ce == truth_obj)
        try:
            return bool(simplify(ce - truth_obj) == 0)
        except Exception:
            try:
                return bool(nsimplify(ce) == nsimplify(truth_obj))
            except Exception:
                return False
    if tier == "tuple":
        ct = as_tuple(c)
        if ct is None or len(ct) != len(truth_obj):
            return False
        return all(simplify(a - b) == 0 for a, b in zip(ct, truth_obj))
    return False

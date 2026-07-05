"""Code-tier detector for degenerate (self-answering) problem statements.

The stage-3 kill analysis (2026-07-04, out/wellposed_pde625_claude_anthropic/
stage3_kill_analysis.md) found three records whose statement literally
contains its own answer — e.g. "the eigenvalues ... are (3±√5)/2 ... What
are these eigenvalues?", or "set α := (N−p)/(p−1) ... What is the exact
decay exponent α?". Such records are trivially answerable, so the
well-posedness judge passes them, but they are worthless benchmark items.

Two conservative checks:

  answer_in_statement — the normalised answer string appears verbatim in
      the normalised statement. Guarded so ubiquitous short answers
      ("0", "1", "\\infty") never fire.

  defined_then_asked — the question asks "what is X?" for a symbol X that
      the statement itself directly defines ("X := <expr>" or "X = <expr>").

Hits are evidence for review routing, not a gate: the caller records them
as ``review_flags`` and leaves the pass/flag verdict to the judge, so a
false positive costs a human glance, never a record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Answers that appear in virtually every mathematical statement — matching
# them proves nothing. Compared post-normalisation.
_UBIQUITOUS_ANSWERS = {
    "0", "1", "-1", "2", "\\infty", "infty", "+\\infty", "-\\infty",
    "\\pi", "e", "yes", "no", "true", "false",
}

# Minimum normalised-answer length for the substring check. Below this the
# collision rate on ordinary statements is too high to be meaningful.
_MIN_SUBSTRING_LEN = 7

# One level of brace nesting per argument ("\frac{3\pm\sqrt{5}}{2}") —
# repeated passes in normalise_math() handle deeper frac-in-frac nesting.
_FRAC_ARG = r"\{((?:[^{}]|\{[^{}]*\})*)\}"
_FRAC = re.compile(r"\\[dtc]?frac\s*" + _FRAC_ARG + r"\s*" + _FRAC_ARG)

# "what is X" / "what are X" — capture a short trailing target expression.
_WHAT_IS = re.compile(
    r"what\s+(?:is|are)\s+(?:the\s+(?:value\s+of|exact\s+[a-z ]{1,30})\s+)?"
    r"([^?.,;]{1,80})[?.]",
    re.IGNORECASE,
)

# Trailing parenthetical aside in a question target: "α (i.e., the value ...)".
# Only space-preceded parens are asides — "R(p/q)" is part of the symbol.
_TRAILING_PAREN = re.compile(r"\s+\(.*$", re.DOTALL)


@dataclass
class DegeneracyHit:
    pattern: str  # "answer_in_statement" | "defined_then_asked"
    snippet: str

    def to_dict(self) -> dict:
        return {"pattern": self.pattern, "snippet": self.snippet}


def normalise_math(text: str) -> str:
    """Aggressive lexical normalisation for LaTeX-ish math comparison.

    Lowercases, rewrites simple \\frac{A}{B} to A/B (three passes for mild
    nesting), and strips math-mode dollars, sizing/spacing macros, braces,
    parens, and whitespace. NOT an equivalence check — only enough to make
    cosmetic variants of the same expression collide.
    """
    s = (text or "").lower()
    s = s.replace("$", "")
    s = s.replace("\\coloneqq", ":=")
    for macro in ("\\left", "\\right", "\\big", "\\bigg", "\\!", "\\,", "\\;", "\\:", "\\ "):
        s = s.replace(macro, "")
    for _ in range(3):
        new = _FRAC.sub(r"(\1)/(\2)", s)
        if new == s:
            break
        s = new
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\cfrac", "\\frac")
    s = re.sub(r"[\s{}()]+", "", s)
    return s.strip(" .,;:$")


def _snippet(text: str, needle_hint: str, limit: int = 60) -> str:
    return (needle_hint or text)[:limit]


def scan(statement: str, answer: str) -> list[DegeneracyHit]:
    """Return degeneracy hits for a (statement, answer) pair (possibly empty)."""
    if not statement or not answer:
        return []
    hits: list[DegeneracyHit] = []

    norm_stmt = normalise_math(statement)
    norm_ans = normalise_math(answer)

    # Check 1: the answer itself is written out in the statement.
    if (
        len(norm_ans) >= _MIN_SUBSTRING_LEN
        and not norm_ans.isdigit()
        and norm_ans not in _UBIQUITOUS_ANSWERS
        and norm_ans in norm_stmt
    ):
        hits.append(DegeneracyHit(
            pattern="answer_in_statement",
            snippet=_snippet(statement, answer),
        ))

    # Check 2: the asked-for symbol is directly defined in the statement.
    for m in _WHAT_IS.finditer(statement):
        raw_target = _TRAILING_PAREN.sub("", m.group(1)).strip()
        target = normalise_math(raw_target)
        # A short symbol-like target ("r(p/q)", "\alpha"), not a prose
        # question ("the asymptotic scale λ(t) as t → ∞").
        if not target or len(target) > 16:
            continue
        # ":=" and "is defined" are unambiguous definitions at any target
        # length; bare "=" only counts for longer targets — single-letter
        # "K=" style matches are too often constraints, not definitions.
        needles = [target + ":=", target + "isdefined"]
        if len(target) >= 3:
            needles.append(target + "=")
        if any(n in norm_stmt for n in needles):
            hits.append(DegeneracyHit(
                pattern="defined_then_asked",
                snippet=_snippet(statement, m.group(1)),
            ))
            break

    return hits

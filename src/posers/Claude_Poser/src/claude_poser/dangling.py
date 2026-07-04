"""Code-tier detector for dangling cross-references in a problem statement.

The check is conservative: it should fire only on text patterns that name
external material a self-contained problem statement could not satisfy on its
own (Section/Theorem/Equation/Figure references by number, "above"/"below"
without a local definition, bare equation labels like (3.2), etc.).

Computed-provenance records skip this check entirely (see schema.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


REFERENCE_NOUNS = (
    "section",
    "subsection",
    "chapter",
    "appendix",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "equation",
    "eq",
    "eqn",
    "figure",
    "fig",
    "table",
    "tab",
    "example",
    "exercise",
    "problem",
    "exhibit",
    "remark",
    "claim",
    "fact",
    "axiom",
    "assumption",
    "conjecture",
    "page",
    "part",
    "thm",
    "lem",
    "prop",
    "cor",
    "def",
)

_NOUN_PATTERN = r"(?:" + "|".join(REFERENCE_NOUNS) + r")"

# "see Section 3", "as in Theorem 4.2", "from Eq. (12)", "by Lemma 1"
_NAMED_REF = re.compile(
    rf"\b(?:see|cf\.?|refer\s+to|as\s+in|from|by|recall|using)\s+{_NOUN_PATTERN}s?\.?\s*"
    r"(?:\(?\s*\d+(?:[.\-]\d+)*\s*\)?|[ivxlcdmIVXLCDM]+)\b",
    re.IGNORECASE,
)

# "Theorem 3", "Theorem 3.2", "Equation (12)", "Chapter 5", "Section 3.1"
# Bare noun+number, no trigger verb required. Decimal is optional now — this
# is the fix for the "0/70 realmath hits" bug: dropping the `+` (was: at least
# one decimal) to `*` (any number of decimals) catches integer-only refs like
# "Theorem 3" that dominate arXiv-extracted text.
_NOUN_NUMBER = re.compile(
    rf"\b{_NOUN_PATTERN}s?\.?\s*\(?\s*\d+(?:[.\-]\d+)*\s*\)?",
    re.IGNORECASE,
)

# Bare parenthesised equation labels. Two variants:
#   - decimal form "(3.2)", "(2.1.5)" — almost always an equation ref
#   - integer form "(3)" at sentence end, so it's less likely to be a function
#     call like "f(3)". Requires a period/newline/end-of-string right after.
_BARE_EQ_LABEL = re.compile(
    r"\(\s*\d+\.\d+(?:\.\d+)*\s*\)|\(\s*\d+\s*\)(?=\s*[.\n]|\s*$)"
)

# LaTeX cross-reference macros left over from imperfect extraction. Any of
# these means the extractor failed to resolve the target — the reader would
# see "as shown in [?]" in the rendered problem.
_LATEX_REF = re.compile(
    r"\\(?:c?ref|eqref|pageref|autoref|nameref|Cref|Ref)\s*\{[^}]*\}"
)

# Unresolved-reference marker that LaTeX inserts when \ref fails.
_UNRESOLVED_REF = re.compile(r"\[\?\]|\{\?\}")

# "above", "below", "previously" used as anaphoric references to material
# not present in the statement. Broader verb list than before — arXiv text
# frequently uses "constructed above", "obtained above", etc.
_DIRECTIONAL_REF = re.compile(
    r"\b(?:"
    r"(?:as\s+)?(?:defined|shown|given|stated|noted|proved|introduced|"
    r"described|established|constructed|obtained|assumed|considered|"
    r"discussed|mentioned|derived|presented)"
    r")\s+(?:above|below|earlier|previously|before)\b",
    re.IGNORECASE,
)

# "the previous problem", "the preceding theorem"
_PREVIOUS_ITEM = re.compile(
    r"\bthe\s+(?:previous|preceding|prior|earlier|last)\s+"
    rf"(?:{_NOUN_PATTERN}|part|question|result|paper|work|paragraph)s?\b",
    re.IGNORECASE,
)

# Meta references to source material — a self-contained problem should not
# mention "the paper", "our paper", "the main result", "the following section".
_META_SOURCE_REF = re.compile(
    r"\b(?:"
    r"(?:in|of|from)\s+(?:the|our|this)\s+paper|"
    r"the\s+(?:main|following|preceding)\s+(?:result|theorem|section|lemma|"
    r"proposition|corollary|equation|proof|claim)|"
    r"(?:following|standard|usual)\s+notation"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class DanglingHit:
    pattern: str
    snippet: str

    def to_dict(self) -> dict:
        return {"pattern": self.pattern, "snippet": self.snippet}


def _snippet(text: str, span: tuple[int, int], pad: int = 20) -> str:
    a = max(0, span[0] - pad)
    b = min(len(text), span[1] + pad)
    s = text[a:b].replace("\n", " ").strip()
    return s


def scan(statement: str) -> list[DanglingHit]:
    """Return every dangling-reference hit in the statement (possibly empty)."""
    if not statement:
        return []
    hits: list[DanglingHit] = []
    for label, pat in (
        ("named_ref", _NAMED_REF),
        ("noun_number", _NOUN_NUMBER),
        ("bare_eq_label", _BARE_EQ_LABEL),
        ("latex_ref", _LATEX_REF),
        ("unresolved_ref", _UNRESOLVED_REF),
        ("directional_ref", _DIRECTIONAL_REF),
        ("previous_item", _PREVIOUS_ITEM),
        ("meta_source_ref", _META_SOURCE_REF),
    ):
        for m in pat.finditer(statement):
            hits.append(DanglingHit(pattern=label, snippet=_snippet(statement, m.span())))
    return hits

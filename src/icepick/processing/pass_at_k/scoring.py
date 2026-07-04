"""Pure text/label helpers for the pass@k stage.

Candidate extraction, junk detection and rollout tallying are ported
verbatim from ModelBreaker's ``realmath/harvest_realmath.py`` so that
measurements made here reproduce the Qwen-era harvest exactly. Label
derivation reimplements the derivation branch of
``processing/schema.py:_normalise_label`` — reimplemented rather than
imported because ``_normalise_label`` is private to the ingest schema;
parity is pinned by a test instead.

Deliberately free of sympy and of ``verifier``: everything here is
string-in/string-out or counts-in/label-out, so these functions (and
their tests) run without any LaTeX parsing. The runner glues extraction
to verification.
"""

from __future__ import annotations

import collections
import re
from typing import Optional

from icepick.contracts.records import BAND_HI, BAND_LO
from icepick.processing.pass_at_k.base import (
    MISDIRECTION_THRESHOLD,
    ROLLOUT_CORRECT,
    ROLLOUT_DEGENERATE,
    ROLLOUT_WRONG,
)

# LaTeX macros that only ever show up in truths too mangled to verify
# (formatting wrappers, ellipses, extraction artefacts). Ported verbatim
# from ModelBreaker's harvest.
JUNK = ["mathrm", "mathbb", "mathcal", "mathbf", "mathsf", "operatorname",
        "Bigl", "Bigr", "bigl", "bigr", "widetilde", "widehat", "cdots",
        "ldots", "efsub", "displaystyle", "text", "boldsymbol", "mathfrak",
        "mathscr"]


def strip_think(text: str) -> str:
    """Drop ``<think>...</think>`` blocks so extraction sees only the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST ``\\boxed{...}`` via brace matching.

    Regex cannot balance braces, so this walks the string: rfind the last
    ``\\boxed``, then track depth from its opening brace. Nested groups
    (``\\boxed{\\frac{1}{2}}``) come back intact.
    """
    i = text.rfind("\\boxed")
    if i == -1:
        return None
    j = text.find("{", i)
    if j == -1:
        return None
    depth = 0
    out = []
    for c in text[j:]:
        if c == "{":
            depth += 1
            if depth == 1:
                continue
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
    return "".join(out).strip()


def extract_candidate(text: str) -> Optional[str]:
    """MB's fallback chain: boxed, else last ``$...$``, else "answer:" prose."""
    b = extract_boxed(text)
    if b:
        return b
    m = re.findall(r"\$([^$]+)\$", text)
    if m:
        return m[-1].strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def truth_garbage(truth) -> bool:
    """True when the truth string carries junk macros — unscoreable as-is."""
    s = str(truth)
    return any(j in s for j in JUNK)


def in_band(p: Optional[float], lo: float = BAND_LO, hi: float = BAND_HI) -> bool:
    """Is ``p`` inside the difficulty band (inclusive both ends)?

    Defaults come from ``icepick.contracts.records`` — never re-declared
    here. NOTE: ModelBreaker's harvest used (0.125, 0.875); icepick's
    contract band (0.125, 0.75) is authoritative for this stage.
    """
    return p is not None and lo <= p <= hi


def derive_label(pass_at_k: Optional[float], top_wrong_share: float) -> str:
    """Label from pass rate; mirrors ``schema._normalise_label``'s derivation.

    The confident-wrong attractor (``misdirection``) only sub-sorts the
    below-band region. Parity with the ingest schema's private labeler is
    pinned by a test, not an import.
    """
    if pass_at_k is None:
        return "other"
    if pass_at_k > BAND_HI:
        return "solved"
    if pass_at_k >= BAND_LO:
        return "band"
    return "misdirection" if top_wrong_share >= MISDIRECTION_THRESHOLD else "collapse"


def tally_rollouts(verdicts: list, candidates: list) -> dict:
    """Aggregate per-rollout verdicts into the counts the label needs.

    MB semantics exactly: ``top_wrong_share`` is the modal wrong
    candidate's count over ALL rollouts used (``len(verdicts)``), not
    over ``n_wrong`` — a misdirection call means at least half of every
    rollout hit the same wrong answer. Degenerate rollouts (candidate
    ``None``) never enter the wrong counter but do stay in the
    denominator. Ties on the modal wrong go to the first-seen candidate
    (``Counter.most_common`` is insertion-stable).
    """
    if len(verdicts) != len(candidates):
        raise ValueError(
            f"verdicts/candidates length mismatch: {len(verdicts)} vs {len(candidates)}"
        )
    n_correct = n_wrong = n_degenerate = 0
    wrongs: collections.Counter = collections.Counter()
    for verdict, candidate in zip(verdicts, candidates):
        if verdict == ROLLOUT_CORRECT:
            n_correct += 1
        elif verdict == ROLLOUT_WRONG:
            n_wrong += 1
            if candidate is not None:
                wrongs[candidate] += 1
        elif verdict == ROLLOUT_DEGENERATE:
            n_degenerate += 1
        else:
            raise ValueError(f"unknown rollout verdict: {verdict!r}")
    if wrongs:
        modal_wrong, modal_n = wrongs.most_common(1)[0]
    else:
        modal_wrong, modal_n = None, 0
    total = len(verdicts)
    return {
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_degenerate": n_degenerate,
        "modal_wrong": modal_wrong,
        "top_wrong_share": modal_n / total if total else 0.0,
    }

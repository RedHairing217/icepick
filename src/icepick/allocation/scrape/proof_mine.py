"""Proof mining for the ``proof-import`` mission (P3): pair paper proofs with
the theorem-like environments realmath's miner already extracted.

Context: ``realmath.py``'s ``_THEOREM_ENVS`` never included ``proof`` — every
QA record is (question, answer) with the paper's derivation left in the
``.tex``. This module recovers the missing middle for train-split records:
locate the theorem instance a record's ``source_statement`` came from, find
the paper's proof of it, and resolve any ``\\ref``-family cross-references in
that proof body against paper content (never stripping the body itself — the
E5 philosophy: resolve alongside, don't rewrite).

**Import-only reuse.** This module never edits ``realmath.py``. It imports
``realmath._THEOREM_ENVS`` (the theorem-like environment names) and
``realmath._label_content_index`` (the E5 ref resolver, indexing labels
inside display/hypothesis environments to their content) and otherwise
rebuilds whatever else it needs — its own env regex, its own ref-macro
regex — as plain, independent code. Nothing here can change
``extract_theorem_candidates``'s behavior because nothing here is called
from anywhere in realmath's own code paths.

Pipeline (see ``mine_paper`` for the per-paper orchestration):

  1. ``mine_proof_envs`` — every ``\\begin{proof}...\\end{proof}`` block in a
     paper, with its optional ``[Proof of ...]`` argument split out.
  2. ``locate_theorem`` — find the theorem-env instance in the paper whose
     body corresponds to a record's ``source_statement`` (exact, then
     ref-stripped, then fuzzy).
  3. ``match_proof`` — pick which mined proof belongs to a located theorem
     (explicit opt-arg cross-match first, then nested containment, then
     positional adjacency).
  4. ``resolve_proof_refs`` — resolve the winning proof's ``\\ref`` family
     against paper content, without touching the proof text.
  5. ``classify_proof_body`` — flag stub/omitted proofs ("see [12]", "the
     proof is standard", ...) so the reformulation stage can count, not
     force, them.

Pure, offline, deterministic: no network, no randomness, no filesystem
access. Everything takes ``tex`` (already-read file content) and returns
plain dicts/lists.
"""

from __future__ import annotations

import difflib
import re
from typing import Optional

from icepick.allocation.scrape import realmath

# ---------------------------------------------------------------------------
# Regexes. All independent of realmath's compiled patterns (only the plain
# ``_THEOREM_ENVS`` name tuple is imported — see module docstring) so nothing
# here can be affected by, or affect, realmath's own matching.
# ---------------------------------------------------------------------------

# Same construction realmath._ENV_RE uses, rebuilt from the imported name
# tuple: behavior-identical (star optional independently on each side, same
# non-greedy DOTALL body capture) without touching realmath's compiled
# pattern object.
_THEOREM_ENV_RE = re.compile(
    r"\\begin\{(" + "|".join(realmath._THEOREM_ENVS) + r")\*?\}(.*?)\\end\{\1\*?\}",
    re.DOTALL,
)

_PROOF_BEGIN_RE = re.compile(r"\\begin\{proof\*?\}")
_PROOF_END_RE = re.compile(r"\\end\{proof\*?\}")

_LABEL_NAME_RE = re.compile(r"\\label\s*\{([^}]*)\}")

# The ref-macro family this module resolves/strips: \ref, \eqref, \cref,
# \Cref. (realmath's own _REF_MACROS also covers \autoref/\pageref for
# _clean_tex's stripping and the E5 resolver; this module's contract names
# exactly these four for both the locate_theorem ref-stripped retry and
# resolve_proof_refs — see the real-data smoke test note in the mission
# report about the one \autoref instance this narrower set doesn't catch,
# which falls through to the fuzzy tier instead and still resolves.)
_REF_TARGET_RE = re.compile(r"\\(?:ref|eqref|cref|Cref)\s*\{([^}]*)\}")

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")

# Stub/omitted-proof phrases, checked only when the whitespace-collapsed
# body is short (< ~200 chars) — see classify_proof_body.
_STUB_PATTERNS = (
    "omitted",
    "left to the reader",
    "standard",
    "follows from",
    "immediate from",
    "see [",
    "see \\cite",
)

_OMITTED_MIN_LEN = 40
_STUB_PATTERN_MAX_LEN = 200
_ADJACENCY_HIGH_CONFIDENCE_GAP = 2000  # chars


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def _norm_ws(text: str) -> str:
    """Collapse whitespace runs to single spaces, strip."""
    return " ".join(text.split()).strip()


def _strip_refs(text: str) -> str:
    """Delete every \\ref/\\eqref/\\cref/\\Cref{...} occurrence bare."""
    return _REF_TARGET_RE.sub("", text)


def _norm_ws_ref_stripped(text: str) -> str:
    return _norm_ws(_strip_refs(text))


def _first_label(body: str) -> Optional[str]:
    """The first \\label{...} target inside ``body``, if any."""
    match = _LABEL_NAME_RE.search(body)
    return match.group(1) if match else None


def _guess_number(label: Optional[str]) -> Optional[str]:
    """Best-effort numeric guess from a label's text (e.g. ``lemma 3.1.2``
    -> ``"3.1.2"``). ``None`` when the label carries no digits at all —
    callers must treat that as "unknown", never as a mismatch signal."""
    if not label:
        return None
    match = _NUMBER_RE.search(label)
    return match.group(0) if match else None


def _first_ref_target(text: str) -> Optional[str]:
    """The first label named by a ref-family macro in ``text`` (cleveref
    comma-lists split, first entry wins), or ``None`` if none is present."""
    match = _REF_TARGET_RE.search(text)
    if not match:
        return None
    parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
    return parts[0] if parts else None


def _first_number(text: str) -> Optional[str]:
    match = _NUMBER_RE.search(text)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# Theorem-env scanning (shared by locate_theorem and match_proof)
# ---------------------------------------------------------------------------


def _theorem_envs(tex: str) -> list:
    """Every theorem-like env in ``tex`` (realmath's _THEOREM_ENVS family),
    in document order: ``[{start, end, env_name, body_raw}, ...]``."""
    out = []
    for match in _THEOREM_ENV_RE.finditer(tex):
        out.append({
            "start": match.start(),
            "end": match.end(),
            "env_name": match.group(1),
            "body_raw": match.group(2),
        })
    return out


def _next_theorem_start(tex: str, after: int) -> Optional[int]:
    """The start offset of the nearest theorem-like env beginning at or
    after ``after``, or ``None`` if this is the last one in the paper."""
    for env in _theorem_envs(tex):
        if env["start"] >= after:
            return env["start"]
    return None


# ---------------------------------------------------------------------------
# 1. mine_proof_envs
# ---------------------------------------------------------------------------


def _scan_balanced_opt_arg(tex: str, pos: int) -> tuple:
    """If ``tex[pos:]`` opens with (optional whitespace then) ``[``, scan a
    bracket/brace-balanced optional argument and return
    ``(content, index_after_closing_bracket)``. Otherwise ``(None, pos)``.

    Treats ``[``/``{`` uniformly as "open" and ``]``/``}`` as "close" so a
    nested ``\\ref{...}`` or a citation's own ``[...]`` group inside the
    optional argument (e.g. ``[Proof of Theorem~\\ref{thm:x}, cf.~\\cite[Rem.
    2]{Foo99}]``) doesn't prematurely close the scan at the first ``]``.
    A blank line (paragraph break) before any ``[`` is found means there is
    no optional argument — LaTeX itself stops looking for one there, and
    without that guard a proof beginning with a bracketed citation many
    lines down would otherwise get misread as the argument.
    """
    n = len(tex)
    i = pos
    while i < n and tex[i] in " \t\r\n":
        i += 1
    gap = tex[pos:i]
    if gap.count("\n") >= 2 or i >= n or tex[i] != "[":
        return None, pos
    start_content = i + 1
    depth = 0
    j = i
    while j < n:
        char = tex[j]
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return tex[start_content:j], j + 1
        j += 1
    # Unterminated optional argument (malformed tex) — treat as absent
    # rather than consuming to end of file.
    return None, pos


def mine_proof_envs(tex: str) -> list:
    """Every ``\\begin{proof}...\\end{proof}`` block in ``tex``.

    Non-greedy: each ``\\begin{proof}`` (or starred ``\\begin{proof*}``)
    pairs with the nearest following ``\\end{proof}``/``\\end{proof*}``, no
    nesting awareness (proofs never nest in practice — realmath's own
    theorem-env matching makes the same simplifying assumption). The
    optional argument, e.g. ``\\begin{proof}[Proof of Theorem 3.2]``, is
    split out (bracket/brace-balanced) into ``opt_arg`` rather than left in
    ``body_raw``.

    Returns dicts in document order:
    ``{start, end, body_raw, opt_arg}`` — ``start``/``end`` span the full
    ``\\begin{proof}...\\end{proof}`` (inclusive), matching how the
    theorem-env scanner reports its own spans, so callers can compare
    positions directly. ``opt_arg`` is ``None`` when no optional argument
    is present. ``body_raw`` is untouched proof text — never cleaned,
    never stripped.
    """
    envs = []
    for begin in _PROOF_BEGIN_RE.finditer(tex):
        opt_arg, body_start = _scan_balanced_opt_arg(tex, begin.end())
        end = _PROOF_END_RE.search(tex, body_start)
        if end is None:
            continue  # unterminated \begin{proof} — malformed tex, skip
        envs.append({
            "start": begin.start(),
            "end": end.end(),
            "body_raw": tex[body_start:end.start()],
            "opt_arg": opt_arg,
        })
    return envs


# ---------------------------------------------------------------------------
# 2. locate_theorem
# ---------------------------------------------------------------------------


def _finalize_theorem(candidate: dict, method: str, confidence: str, ambiguous: bool) -> dict:
    label = _first_label(candidate["body_raw"])
    return {
        "start": candidate["start"],
        "end": candidate["end"],
        "body": candidate["body_raw"],
        "env_name": candidate["env_name"],
        "label": label,
        "theorem_number_guess": _guess_number(label),
        "confidence": confidence,
        "method": method,
        "ambiguous": ambiguous,
    }


def locate_theorem(tex: str, source_statement: str, environment: str) -> Optional[dict]:
    """Find the theorem-env instance in ``tex`` whose body corresponds to
    ``source_statement``.

    Mines every instance of realmath's theorem-like env family (not just
    ``environment`` — the fuzzy tier explicitly compares against every env
    body, and an exact/ref-stripped textual hit is unambiguous regardless of
    env type; ``environment`` is accepted for interface parity with the
    record schema but the current matching is purely text-driven).

    Three tiers, whitespace-collapsed comparison throughout:

    1. **exact** — ``source_statement`` equals or is a substring of a
       candidate body. Confidence "high".
    2. **ref_stripped** — same check after additionally deleting every
       ``\\ref``/``\\eqref``/``\\cref``/``\\Cref{...}`` from both sides (old
       corpus rows had refs bare-deleted from ``source_statement``, leaving
       holes where the tex still has them). Confidence "high".
    3. **fuzzy** — ``difflib.SequenceMatcher`` ratio (on the ref-stripped
       normalization, the best available) against every candidate; highest
       ratio wins. >=0.90 "high", >=0.75 "medium", >=0.60 "low", below ->
       no match at all (returns ``None``).

    When multiple candidates tie at the top match quality within a tier,
    the first in document order wins and ``ambiguous`` is set ``True``.
    Returns ``None`` when the paper has no theorem-like env, or nothing
    clears the fuzzy floor.
    """
    candidates = _theorem_envs(tex)
    if not candidates:
        return None

    for norm_fn, method in ((_norm_ws, "exact"), (_norm_ws_ref_stripped, "ref_stripped")):
        needle = norm_fn(source_statement)
        if not needle:
            continue  # an empty needle is trivially "in" everything — never a real hit
        hits = [c for c in candidates if needle in norm_fn(c["body_raw"])]
        if hits:
            return _finalize_theorem(hits[0], method, "high", len(hits) > 1)

    needle = _norm_ws_ref_stripped(source_statement)
    scored = [
        (difflib.SequenceMatcher(None, needle, _norm_ws_ref_stripped(c["body_raw"])).ratio(), c)
        for c in candidates
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)  # stable: ties keep document order
    best_ratio = scored[0][0]
    if best_ratio >= 0.90:
        confidence = "high"
    elif best_ratio >= 0.75:
        confidence = "medium"
    elif best_ratio >= 0.60:
        confidence = "low"
    else:
        return None
    tied = [c for ratio, c in scored if ratio == best_ratio]
    return _finalize_theorem(tied[0], "fuzzy", confidence, len(tied) > 1)


# ---------------------------------------------------------------------------
# 3. match_proof
# ---------------------------------------------------------------------------


def _finalize_proof(proof: dict, method: str, confidence: str, multi_proof: bool, notes: str) -> dict:
    return {
        "start": proof["start"],
        "end": proof["end"],
        "body_raw": proof["body_raw"],
        "opt_arg": proof.get("opt_arg"),
        "match_method": method,
        "match_confidence": confidence,
        "multi_proof": multi_proof,
        "notes": notes,
    }


def match_proof(tex: str, theorem: dict, proofs: list, label_index: dict) -> Optional[dict]:
    """Pick the proof belonging to a located ``theorem``.

    ``label_index`` (built with ``realmath._label_content_index(tex)`` at
    the call site, same as ``resolve_proof_refs``) is accepted for
    signature symmetry and because ``mine_paper`` computes it once per
    paper regardless; the E5 index only covers display/hypothesis
    environments (``_LABELED_ENV_NAMES`` — equation/align/.../assumption/
    definition/hypothesis), never theorem-like ones, so it can't tell us
    what a ``\\ref`` inside an opt-arg names among *theorems*. Opt-arg
    cross-matching below instead compares directly against
    ``theorem['label']``/``theorem['theorem_number_guess']``.

    Priority:

    1. **Explicit opt-arg cross-match.** A proof whose optional argument
       names this theorem's own ``\\label`` wins outright
       (``match_method="proof_of_label"``, confidence "high"), searched
       across *every* mined proof regardless of position. Failing that, a
       proof whose opt-arg carries a bare number equal to this theorem's
       best-effort ``theorem_number_guess`` wins
       (``"proof_of_number"``, confidence "medium") — only when both sides
       have an inferable number; when this theorem's number can't be
       guessed, a numbered opt-arg is neither a match nor a mismatch (do
       not guess) and the proof stays adjacency/nested-eligible.
       A proof whose opt-arg names a *different* label or number is
       excluded from nested/adjacency for this theorem — it has explicitly
       claimed something else and must never be silently absorbed.
    2. **Nested.** Among proofs not excluded above, a proof env whose span
       lies textually INSIDE this theorem env's own span
       (``proof.start > theorem.start`` and ``proof.end <= theorem.end``)
       — the common paper structure
       ``\\begin{theorem}...\\begin{proof}...\\end{proof}...\\end{theorem}``,
       which never leaves adjacency anything to find since the proof never
       starts *after* the theorem ends. ``match_method="nested"``,
       confidence "high". Loses to tier 1 (an explicit opt-arg match for
       *this* theorem is more explicit than mere textual containment).
       Among several nested proofs, the one nearest the statement end
       (first in document order, i.e. smallest start) wins.
    3. **Adjacency.** Among proofs not excluded above and not nested in any
       theorem-like env, the nearest one that starts at or after this
       theorem's env ends and before the next theorem-like env begins.
       Confidence "high" if it starts within ~2000 chars of the theorem's
       end, else "medium". A nested candidate always wins over adjacency —
       structurally the two candidate sets never overlap (a nested proof's
       end can't be at/after this theorem's own end, so it can never also
       be adjacency-eligible), so this tier only fires when tier 2 found
       nothing.

    ``multi_proof`` is set when more than one proof plausibly claims this
    theorem — multiple opt-arg hits, multiple nested proofs, an opt-arg or
    nested winner whose positional nested/adjacency pick would have been a
    *different* proof, or (pure adjacency) more than one unclaimed
    candidate in the window — and ``notes`` records which one won. Returns
    ``None`` when nothing qualifies under any tier.
    """
    label_index = label_index or {}  # accepted per docstring; not load-bearing here
    theorem_label = theorem.get("label")
    theorem_number = theorem.get("theorem_number_guess")
    t_start = theorem["start"]
    t_end = theorem["end"]

    label_hits: list = []
    number_hits: list = []
    claimed_elsewhere: set = set()
    for proof in proofs:
        opt_arg = proof.get("opt_arg")
        if not opt_arg:
            continue
        target = _first_ref_target(opt_arg)
        if target is not None:
            if theorem_label is not None and target == theorem_label:
                label_hits.append(proof)
            else:
                claimed_elsewhere.add(id(proof))
            continue
        number = _first_number(opt_arg)
        if number is not None and theorem_number is not None:
            if number == theorem_number:
                number_hits.append(proof)
            else:
                claimed_elsewhere.add(id(proof))
        # number found but theorem_number unknown: don't guess either way

    next_start = _next_theorem_start(tex, t_end)

    def _adjacency_candidates() -> list:
        eligible = [
            p for p in proofs
            if id(p) not in claimed_elsewhere
            and p["start"] >= t_end  # ">=" not ">": a proof immediately abutting the
            # theorem's closing tag with zero intervening characters (no whitespace at
            # all between "\end{...}" and "\begin{proof}") still starts "after" it.
            and (next_start is None or p["start"] < next_start)
        ]
        eligible.sort(key=lambda p: p["start"])
        return eligible

    def _nested_candidates() -> list:
        # proof.start > t_start (not >=): the two spans start at literally
        # different \begin{...} tags so this is never tight in practice, but
        # ">" is the correct "strictly inside" reading. proof.end <= t_end
        # ("at/inside" the theorem's own end, per the containment contract) —
        # a nested proof's end can never legitimately exceed the enclosing
        # theorem's own \end{...}, since realmath's env regex is non-greedy
        # and theorem-env spans never overlap each other.
        eligible = [
            p for p in proofs
            if id(p) not in claimed_elsewhere
            and p["start"] > t_start
            and p["end"] <= t_end
        ]
        eligible.sort(key=lambda p: p["start"])
        return eligible

    if label_hits:
        winner = label_hits[0]
        multi_proof = len(label_hits) > 1
        notes = [f"{len(label_hits)} proofs opt-arg-labeled for this theorem; first wins"] if multi_proof else []
        nested = _nested_candidates()
        if nested and nested[0] is not winner:
            multi_proof = True
            notes.append("nested candidate differs from proof_of_label winner; proof_of_label kept")
        adjacent = _adjacency_candidates()
        if adjacent and adjacent[0] is not winner:
            multi_proof = True
            notes.append("adjacency candidate differs from proof_of_label winner; proof_of_label kept")
        return _finalize_proof(winner, "proof_of_label", "high", multi_proof, "; ".join(notes))

    if number_hits:
        winner = number_hits[0]
        multi_proof = len(number_hits) > 1
        notes = [f"{len(number_hits)} proofs opt-arg-numbered for this theorem; first wins"] if multi_proof else []
        nested = _nested_candidates()
        if nested and nested[0] is not winner:
            multi_proof = True
            notes.append("nested candidate differs from proof_of_number winner; proof_of_number kept")
        adjacent = _adjacency_candidates()
        if adjacent and adjacent[0] is not winner:
            multi_proof = True
            notes.append("adjacency candidate differs from proof_of_number winner; proof_of_number kept")
        return _finalize_proof(winner, "proof_of_number", "medium", multi_proof, "; ".join(notes))

    nested = _nested_candidates()
    if nested:
        winner = nested[0]
        multi_proof = len(nested) > 1
        notes = [f"{len(nested)} proofs nested inside this theorem env; nearest-to-statement-end wins"] if multi_proof else []
        adjacent = _adjacency_candidates()
        if adjacent:
            # Structurally disjoint from `nested` (see docstring), so this can
            # never be the same object as `winner` — no "is not winner" guard
            # needed here, unlike the tier-1 checks above.
            multi_proof = True
            notes.append(f"{len(adjacent)} adjacency candidate(s) displaced by nested proof; nested kept")
        return _finalize_proof(winner, "nested", "high", multi_proof, "; ".join(notes))

    adjacent = _adjacency_candidates()
    if not adjacent:
        return None
    winner = adjacent[0]
    gap = winner["start"] - t_end
    confidence = "high" if gap <= _ADJACENCY_HIGH_CONFIDENCE_GAP else "medium"
    multi_proof = len(adjacent) > 1
    notes = f"{len(adjacent)} unclaimed proof(s) in the adjacency window; nearest wins" if multi_proof else ""
    return _finalize_proof(winner, "adjacency", confidence, multi_proof, notes)


# ---------------------------------------------------------------------------
# 4. resolve_proof_refs
# ---------------------------------------------------------------------------


def resolve_proof_refs(proof_body: str, label_index: dict) -> tuple:
    """Resolve every \\ref/\\eqref/\\cref/\\Cref{label} in ``proof_body``
    against ``label_index`` (build with ``realmath._label_content_index(tex)``
    at the call site). Returns ``(resolved, unresolved)``:
    ``resolved`` maps label -> content for every label ``label_index`` had;
    ``unresolved`` lists every label (in order, duplicates kept — same
    shape as realmath's own ``_referenced_labels`` + resolution loop) it
    didn't. ``proof_body`` is only ever read — this is metadata built
    *alongside* the proof, never a rewrite of it (the E5 philosophy).
    """
    label_index = label_index or {}
    resolved: dict = {}
    unresolved: list = []
    for match in _REF_TARGET_RE.finditer(proof_body):
        parts = [part.strip() for part in match.group(1).split(",")]
        labels = [part for part in parts if part] or [""]
        for label in labels:
            content = label_index.get(label) if label else None
            if content is not None:
                resolved[label] = content
            else:
                unresolved.append(label)
    return resolved, unresolved


# ---------------------------------------------------------------------------
# 5. classify_proof_body
# ---------------------------------------------------------------------------


def classify_proof_body(body: str) -> str:
    """Returns "substantive" or "omitted".

    "omitted" when the whitespace-collapsed body is shorter than 40 chars,
    or (only when it's shorter than ~200 chars) it contains a stub phrase
    ("left to the reader", "standard", "follows from", "see [", ...) —
    the length gate means a long real proof that merely *contains* "follows
    from" mid-derivation stays "substantive"; these patterns only fire when
    they plausibly ARE most of a short body, not a phrase buried in one.
    """
    collapsed = _norm_ws(body)
    if len(collapsed) < _OMITTED_MIN_LEN:
        return "omitted"
    if len(collapsed) < _STUB_PATTERN_MAX_LEN:
        lowered = collapsed.lower()
        if any(pattern in lowered for pattern in _STUB_PATTERNS):
            return "omitted"
    return "substantive"


# ---------------------------------------------------------------------------
# 6. mine_paper
# ---------------------------------------------------------------------------


def mine_paper(tex: str, records: list) -> list:
    """Orchestrate P3 mining for one paper's ``tex`` against its records
    (``{uid, arxiv_id, statement, source_statement, environment}`` rows —
    the ``train_records.jsonl`` shape).

    Returns exactly one result dict per input record, in the same order,
    never dropping any. Every result carries ``uid``, ``arxiv_id``, and
    ``class`` (one of ``matched``, ``proof_omitted``,
    ``unmatched_statement_not_found``, ``unmatched_no_proof``,
    ``paper_has_no_proof_env``) plus ``theorem_located`` (bool) and
    ``notes`` (str).

    ``matched`` and ``proof_omitted`` results additionally carry the full
    ``proofs_raw.jsonl`` field set (CONTRACTS.md): ``proof_raw``,
    ``resolved_refs``, ``unresolved_refs``, ``match_method``,
    ``match_confidence``, ``multi_proof`` — omitted rows still carry
    ``proof_raw`` (a real proof was found and classified, just as a stub;
    count it, don't drop it).

    A paper with zero ``\\begin{proof}`` environments short-circuits every
    record straight to ``paper_has_no_proof_env`` without attempting
    theorem location (cheap, and the outcome for every record is already
    determined: there is no proof anywhere to find).
    """
    proofs = mine_proof_envs(tex)
    no_proof_env = not proofs
    label_index = realmath._label_content_index(tex)

    results = []
    for record in records:
        uid = record["uid"]
        arxiv_id = record.get("arxiv_id")

        if no_proof_env:
            results.append({
                "uid": uid,
                "arxiv_id": arxiv_id,
                "class": "paper_has_no_proof_env",
                "theorem_located": False,
                "notes": "tex has zero \\begin{proof} environments",
            })
            continue

        theorem = locate_theorem(tex, record.get("source_statement", ""), record.get("environment", ""))
        if theorem is None:
            results.append({
                "uid": uid,
                "arxiv_id": arxiv_id,
                "class": "unmatched_statement_not_found",
                "theorem_located": False,
                "notes": "",
            })
            continue

        proof = match_proof(tex, theorem, proofs, label_index)
        if proof is None:
            results.append({
                "uid": uid,
                "arxiv_id": arxiv_id,
                "class": "unmatched_no_proof",
                "theorem_located": True,
                "notes": "",
            })
            continue

        resolved_refs, unresolved_refs = resolve_proof_refs(proof["body_raw"], label_index)
        classification = classify_proof_body(proof["body_raw"])
        cls = "matched" if classification == "substantive" else "proof_omitted"
        notes = proof["notes"]
        if cls == "proof_omitted":
            stub_note = "classify_proof_body: omitted (short or stub-pattern body)"
            notes = f"{notes}; {stub_note}" if notes else stub_note
        results.append({
            "uid": uid,
            "arxiv_id": arxiv_id,
            "class": cls,
            "proof_raw": proof["body_raw"],
            "resolved_refs": resolved_refs,
            "unresolved_refs": unresolved_refs,
            "match_method": proof["match_method"],
            "match_confidence": proof["match_confidence"],
            "theorem_located": True,
            "multi_proof": proof["multi_proof"],
            "notes": notes,
        })

    return results

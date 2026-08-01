"""Proof mining (P3, mission proof-import) — synthetic-fixture tests only.

No network, no repo data files: every fixture below is inline LaTeX built
for this suite. (The real-data smoke test against cached arXiv papers for
this mission is a manual, one-off check reported separately — not part of
this file, per the mission brief: this suite must run standalone, offline,
deterministic.)

``proof_mine`` is import-only reuse of ``realmath`` (``_THEOREM_ENVS``,
``_label_content_index``); it must never be able to change realmath's own
behavior. The last section of this file checks exactly that.
"""

from __future__ import annotations

import collections
import inspect
import re

from icepick.allocation.scrape import proof_mine as pm
from icepick.allocation.scrape import realmath


def _record(uid, source_statement, environment, arxiv_id="9999.00001", statement=None):
    return {
        "uid": uid,
        "arxiv_id": arxiv_id,
        "statement": statement or f"Question text for {uid}.",
        "source_statement": source_statement,
        "environment": environment,
    }


# --- mine_proof_envs -----------------------------------------------------


def test_mine_proof_envs_plain_proof_no_opt_arg():
    tex = r"\begin{proof}plain body\end{proof}"
    [env] = pm.mine_proof_envs(tex)
    assert env["opt_arg"] is None
    assert env["body_raw"] == "plain body"
    assert env["start"] == 0
    assert env["end"] == len(tex)


def test_mine_proof_envs_opt_arg_with_nested_ref_braces():
    """The optional argument's own \\ref{...} has braces of its own —
    the opt-arg scan must not stop at the first '}' it meets."""
    tex = r"\begin{proof}[Proof of Theorem~\ref{thm:main}]body\end{proof}"
    [env] = pm.mine_proof_envs(tex)
    assert env["opt_arg"] == r"Proof of Theorem~\ref{thm:main}"
    assert env["body_raw"] == "body"


def test_mine_proof_envs_opt_arg_tolerates_nested_brackets():
    """A citation's own [...] group nested inside the optional argument
    (e.g. \\cite[Rem. 2]{Foo99}) must not truncate the scan early."""
    tex = r"\begin{proof}[Proof of~\ref{thm:x}, cf.~\cite[Rem.~2]{Foo99}]body two\end{proof}"
    [env] = pm.mine_proof_envs(tex)
    assert env["opt_arg"] == r"Proof of~\ref{thm:x}, cf.~\cite[Rem.~2]{Foo99}"
    assert env["body_raw"] == "body two"


def test_mine_proof_envs_paragraph_break_before_bracket_is_not_opt_arg():
    """A blank line before a '[' means no optional argument — otherwise a
    proof that opens with a bracketed citation (very common) would get its
    citation misread as the \\begin{proof}[...] title."""
    tex = "\\begin{proof}\n\nWe cite [12, Thm 3] for this fact.\\end{proof}"
    [env] = pm.mine_proof_envs(tex)
    assert env["opt_arg"] is None
    assert "[12, Thm 3]" in env["body_raw"]


def test_mine_proof_envs_single_newline_before_bracket_is_opt_arg():
    """A single newline (no blank line) before '[' is still consumed as
    leading whitespace — matches how LaTeX itself would parse it."""
    tex = "\\begin{proof}\n[Proof of Theorem 1]\nbody\\end{proof}"
    [env] = pm.mine_proof_envs(tex)
    assert env["opt_arg"] == "Proof of Theorem 1"


def test_mine_proof_envs_starred_environment():
    tex = r"\begin{proof*}starred body\end{proof*}"
    [env] = pm.mine_proof_envs(tex)
    assert env["body_raw"] == "starred body"


def test_mine_proof_envs_multiple_in_document_order():
    tex = r"\begin{proof}first\end{proof} some prose \begin{proof}second\end{proof}"
    envs = pm.mine_proof_envs(tex)
    assert [e["body_raw"] for e in envs] == ["first", "second"]
    assert envs[0]["start"] < envs[1]["start"]


def test_mine_proof_envs_empty_when_none_present():
    assert pm.mine_proof_envs("just prose, no proof environments here") == []


def test_mine_proof_envs_does_not_capture_a_theorem_environment():
    tex = r"\begin{theorem}not a proof\end{theorem}"
    assert pm.mine_proof_envs(tex) == []


# --- locate_theorem --------------------------------------------------------


def test_locate_theorem_exact_hit_substring_after_label_prefix():
    """The realistic case: source_statement is the _clean_tex'd body (label
    stripped), so it's a substring of the raw candidate body, not equal to
    it. Exact tier must use substring, not just equality."""
    tex = (
        r"\begin{theorem}\label{thm:one}"
        r"Statement one is true."
        r"\end{theorem}"
    )
    theorem = pm.locate_theorem(tex, "Statement one is true.", "theorem")
    assert theorem is not None
    assert theorem["method"] == "exact"
    assert theorem["confidence"] == "high"
    assert theorem["label"] == "thm:one"
    assert theorem["env_name"] == "theorem"
    assert theorem["ambiguous"] is False


def test_locate_theorem_ref_stripped_retry():
    """Old corpus rows had \\ref bare-deleted from source_statement, leaving
    a hole where the tex still has the \\eqref — exact tier must fail and
    the ref-stripped retry must catch it."""
    tex = (
        r"\begin{lemma}\label{lem:x}"
        r"The bound holds by \eqref{eq:main} and follows."
        r"\end{lemma}"
    )
    # Mirrors what realmath._clean_tex actually produces: the eqref deleted
    # bare, leaving "by  and follows" collapsed to "by and follows".
    holed_statement = "The bound holds by and follows."
    exact_attempt = pm.locate_theorem(tex, holed_statement, "lemma")
    assert exact_attempt is not None
    assert exact_attempt["method"] == "ref_stripped"
    assert exact_attempt["confidence"] == "high"
    assert exact_attempt["label"] == "lem:x"


def test_locate_theorem_fuzzy_high_confidence():
    tex = (
        r"\begin{theorem}\label{thm:f}"
        r"The solution converges to zero as time tends to infinity under mild assumptions on the initial data."
        r"\end{theorem}"
    )
    reworded = "The solution converges to zero as time goes to infinity under mild assumptions on the initial data."
    theorem = pm.locate_theorem(tex, reworded, "theorem")
    assert theorem["method"] == "fuzzy"
    assert theorem["confidence"] == "high"


def test_locate_theorem_fuzzy_medium_confidence():
    tex = (
        r"\begin{theorem}\label{thm:f}"
        r"The solution converges to zero as time tends to infinity under mild assumptions on the initial data."
        r"\end{theorem}"
    )
    reworded = "The computed solution converges to zero as time tends to infinity under mild assumptions placed on the data."
    theorem = pm.locate_theorem(tex, reworded, "theorem")
    assert theorem["method"] == "fuzzy"
    assert theorem["confidence"] == "medium"


def test_locate_theorem_fuzzy_low_confidence():
    tex = (
        r"\begin{theorem}\label{thm:f}"
        r"The solution converges to zero as time tends to infinity under mild assumptions on the initial data."
        r"\end{theorem}"
    )
    reworded = "The solution tends toward zero as time increases without bound under mild hypotheses on the initial data."
    theorem = pm.locate_theorem(tex, reworded, "theorem")
    assert theorem["method"] == "fuzzy"
    assert theorem["confidence"] == "low"


def test_locate_theorem_below_fuzzy_floor_returns_none():
    tex = (
        r"\begin{theorem}\label{thm:f}"
        r"The solution converges to zero as time tends to infinity under mild assumptions on the initial data."
        r"\end{theorem}"
    )
    unrelated = "Bananas are a good source of potassium for athletes."
    assert pm.locate_theorem(tex, unrelated, "theorem") is None


def test_locate_theorem_starred_environment():
    tex = (
        r"\begin{theorem*}\label{thm:star}"
        r"An unnumbered starred statement."
        r"\end{theorem*}"
    )
    theorem = pm.locate_theorem(tex, "An unnumbered starred statement.", "theorem")
    assert theorem is not None
    # realmath's own _ENV_RE excludes the star from the captured group name
    # (it's outside group 1); a behavior-identical rebuild must match.
    assert theorem["env_name"] == "theorem"


def test_locate_theorem_no_theorem_envs_in_paper_returns_none():
    assert pm.locate_theorem("just prose, no envs", "anything", "theorem") is None


def test_locate_theorem_exact_hit_when_body_contains_a_nested_proof():
    """Real-corpus pattern: a paper writes \\begin{proof}...\\end{proof}
    INSIDE the theorem env itself, so the env body is statement+proof, not
    statement alone. The exact tier's substring check must still find the
    right theorem — it only asks whether the (clean) statement appears
    somewhere in the body, and doesn't care what else (a nested proof, in
    this case) the body also contains after/around it."""
    tex = (
        r"\begin{theorem}\label{thm:x}"
        r"Statement X holds under the standing hypotheses of this section."
        r"\begin{proof}Nested proof body, with its own citations and content that"
        r" has nothing lexically in common with the statement above.\end{proof}"
        r"\end{theorem}"
    )
    theorem = pm.locate_theorem(
        tex, "Statement X holds under the standing hypotheses of this section.", "theorem"
    )
    assert theorem is not None
    assert theorem["method"] == "exact"
    assert theorem["confidence"] == "high"
    assert theorem["label"] == "thm:x"
    assert theorem["ambiguous"] is False


def test_locate_theorem_ambiguous_tie_returns_first_in_document_order():
    tex = (
        r"\begin{lemma}\label{lem:first}Repeated identical statement text.\end{lemma}"
        r"\begin{lemma}\label{lem:second}Repeated identical statement text.\end{lemma}"
    )
    theorem = pm.locate_theorem(tex, "Repeated identical statement text.", "lemma")
    assert theorem["ambiguous"] is True
    assert theorem["label"] == "lem:first"


# --- match_proof -------------------------------------------------------


def test_match_proof_plain_adjacency():
    tex = (
        r"\begin{lemma}\label{lem:c}Statement C.\end{lemma}"
        r"\begin{proof}Adjacent proof of C.\end{proof}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement C.", "lemma")
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "adjacency"
    assert result["match_confidence"] == "high"
    assert result["multi_proof"] is False
    assert result["body_raw"] == "Adjacent proof of C."


def test_match_proof_adjacency_medium_confidence_when_far():
    filler = "x " * 1500  # pushes the proof well past the 2000-char gap floor
    tex = (
        r"\begin{lemma}\label{lem:c}Statement C.\end{lemma}"
        + filler
        + r"\begin{proof}Distant proof of C.\end{proof}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement C.", "lemma")
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "adjacency"
    assert result["match_confidence"] == "medium"


def test_match_proof_proof_of_label_via_opt_arg_ref():
    tex = (
        r"\begin{theorem}\label{thm:main}Statement.\end{theorem}"
        r"Some intervening prose that is not a proof."
        r"\begin{proof}[Proof of Theorem~\ref{thm:main}]"
        r"The labeled proof body."
        r"\end{proof}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement.", "theorem")
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "proof_of_label"
    assert result["match_confidence"] == "high"
    assert result["body_raw"] == "The labeled proof body."


def test_match_proof_opt_arg_naming_a_different_theorem_is_never_adjacency_stolen():
    """A proof whose opt-arg explicitly claims theorem B must not be
    silently absorbed by theorem A's adjacency search, even though it is
    positionally the nearest proof after A."""
    tex = (
        r"\begin{theorem}\label{thm:a}Statement A.\end{theorem}"
        r"\begin{proof}[Proof of Theorem~\ref{thm:b}]"
        r"This proof explicitly claims theorem B, not A."
        r"\end{proof}"
        r"\begin{theorem}\label{thm:b}Statement B.\end{theorem}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem_a = pm.locate_theorem(tex, "Statement A.", "theorem")
    theorem_b = pm.locate_theorem(tex, "Statement B.", "theorem")

    result_a = pm.match_proof(tex, theorem_a, proofs, {})
    assert result_a is None  # the only nearby proof is claimed by B, not adjacency-eligible for A

    result_b = pm.match_proof(tex, theorem_b, proofs, {})
    assert result_b["match_method"] == "proof_of_label"


def test_match_proof_opt_arg_different_theorem_leaves_a_real_adjacency_candidate_free():
    """Same exclusion as above, but this time a second, unclaimed proof
    sits in A's window and should win adjacency for A."""
    tex = (
        r"\begin{theorem}\label{thm:a}Statement A.\end{theorem}"
        r"\begin{proof}[Proof of Theorem~\ref{thm:b}]claims B, not A.\end{proof}"
        r"\begin{proof}unlabeled proof, actually adjacent to A.\end{proof}"
        r"\begin{theorem}\label{thm:b}Statement B.\end{theorem}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem_a = pm.locate_theorem(tex, "Statement A.", "theorem")
    result_a = pm.match_proof(tex, theorem_a, proofs, {})
    assert result_a["match_method"] == "adjacency"
    assert result_a["body_raw"] == "unlabeled proof, actually adjacent to A."


def test_match_proof_proof_of_number():
    tex = (
        r"\begin{theorem}\label{thm:3.2}Statement numbered 3.2.\end{theorem}"
        r"\begin{proof}[Proof of Theorem 3.2]Numbered opt-arg proof.\end{proof}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement numbered 3.2.", "theorem")
    assert theorem["theorem_number_guess"] == "3.2"
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "proof_of_number"
    assert result["match_confidence"] == "medium"


def test_match_proof_unlabeled_theorem_does_not_guess_at_a_numbered_opt_arg():
    """theorem_number_guess is None (label has no digits) — a numbered
    opt-arg elsewhere must not be treated as a match OR a mismatch; the
    proof stays eligible for adjacency."""
    tex = (
        r"\begin{theorem}\label{thm:nodigits}Statement with an unnumbered label.\end{theorem}"
        r"\begin{proof}[Proof of Theorem 3.2]Body.\end{proof}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement with an unnumbered label.", "theorem")
    assert theorem["theorem_number_guess"] is None
    result = pm.match_proof(tex, theorem, proofs, {})
    # Not proof_of_number (can't confirm), falls through to adjacency instead.
    assert result["match_method"] == "adjacency"


def test_match_proof_multi_proof_when_opt_arg_and_adjacency_disagree():
    tex = (
        r"\begin{theorem}\label{thm:d}Statement D.\end{theorem}"
        r"\begin{proof}An unlabeled proof sitting right after D.\end{proof}"
        r"\begin{proof}[Proof of Theorem~\ref{thm:d}]The actual opt-arg-labeled proof of D.\end{proof}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement D.", "theorem")
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "proof_of_label"
    assert result["multi_proof"] is True
    assert "proof_of_label" in result["notes"]


def test_match_proof_multi_proof_pure_adjacency_window():
    tex = (
        r"\begin{lemma}\label{lem:e}Statement E.\end{lemma}"
        r"\begin{proof}first unclaimed candidate.\end{proof}"
        r"\begin{proof}second unclaimed candidate.\end{proof}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement E.", "lemma")
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "adjacency"
    assert result["multi_proof"] is True
    assert result["body_raw"] == "first unclaimed candidate."  # nearest wins


def test_match_proof_returns_none_when_no_candidates_qualify():
    tex = r"\begin{theorem}\label{thm:lonely}Statement with no proof.\end{theorem}"
    proofs = pm.mine_proof_envs(tex)  # empty
    theorem = pm.locate_theorem(tex, "Statement with no proof.", "theorem")
    assert pm.match_proof(tex, theorem, proofs, {}) is None


# --- match_proof: nested containment --------------------------------------
#
# The dominant unmatched_no_proof cause the smoke round found: many papers
# write \begin{proof}...\end{proof} INSIDE the theorem env itself
# (\begin{theorem}...\begin{proof}...\end{proof}...\end{theorem}), which
# structurally defeats the adjacency rule (the proof never starts *after*
# the theorem ends, since it's nested inside the theorem's own span).


def test_match_proof_nested_basic():
    tex = (
        r"\begin{theorem}\label{thm:x}Statement X."
        r"\begin{proof}Nested proof of X.\end{proof}"
        r"\end{theorem}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement X.", "theorem")
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "nested"
    assert result["match_confidence"] == "high"
    assert result["multi_proof"] is False
    assert result["body_raw"] == "Nested proof of X."


def test_match_proof_nested_beats_adjacency_with_multi_proof():
    """A nested proof always wins over a pure-adjacency candidate, but the
    displaced adjacency candidate must still surface via multi_proof/notes
    — it's real signal that more than one proof plausibly claims this
    theorem, even though the structural (nested) signal is the stronger
    one and decides the winner."""
    tex = (
        r"\begin{theorem}\label{thm:x}Statement X."
        r"\begin{proof}Nested proof of X.\end{proof}"
        r"\end{theorem}"
        r"\begin{proof}A later, unclaimed adjacency candidate.\end{proof}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement X.", "theorem")
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "nested"
    assert result["body_raw"] == "Nested proof of X."
    assert result["multi_proof"] is True
    assert "nested" in result["notes"]


def test_match_proof_nested_excluded_when_opt_arg_names_a_different_theorem():
    """Same principle as adjacency's opt-arg exclusion, reused verbatim for
    nested: a proof nested inside A's span that explicitly claims theorem B
    via its opt-arg must never be silently absorbed by A."""
    tex = (
        r"\begin{theorem}\label{thm:a}Statement A."
        r"\begin{proof}[Proof of Theorem~\ref{thm:b}]"
        r"This proof explicitly claims theorem B, not A, even though it is"
        r" textually nested inside A."
        r"\end{proof}"
        r"\end{theorem}"
        r"\begin{theorem}\label{thm:b}Statement B.\end{theorem}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem_a = pm.locate_theorem(tex, "Statement A.", "theorem")
    theorem_b = pm.locate_theorem(tex, "Statement B.", "theorem")

    result_a = pm.match_proof(tex, theorem_a, proofs, {})
    assert result_a is None  # the only proof nested in A's span is explicitly claimed by B

    result_b = pm.match_proof(tex, theorem_b, proofs, {})
    assert result_b["match_method"] == "proof_of_label"
    assert "claims theorem B" in result_b["body_raw"]


def test_match_proof_nested_multiple_first_wins():
    tex = (
        r"\begin{theorem}\label{thm:x}Statement X."
        r"\begin{proof}First nested proof.\end{proof}"
        r"Some intervening remark text between the two proofs."
        r"\begin{proof}Second nested proof.\end{proof}"
        r"\end{theorem}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement X.", "theorem")
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "nested"
    assert result["body_raw"] == "First nested proof."  # nearest-to-statement-end wins
    assert result["multi_proof"] is True
    assert "2 proofs nested" in result["notes"]


def test_match_proof_nested_loses_to_proof_of_label():
    """An explicit opt-arg cross-match for THIS theorem is more explicit
    than mere textual containment, so proof_of_label still wins even
    though a nested candidate also exists."""
    tex = (
        r"\begin{theorem}\label{thm:x}Statement X."
        r"\begin{proof}An unlabeled nested proof, not the real one.\end{proof}"
        r"\end{theorem}"
        r"some prose in between"
        r"\begin{proof}[Proof of Theorem~\ref{thm:x}]The explicit opt-arg-labeled proof.\end{proof}"
    )
    proofs = pm.mine_proof_envs(tex)
    theorem = pm.locate_theorem(tex, "Statement X.", "theorem")
    result = pm.match_proof(tex, theorem, proofs, {})
    assert result["match_method"] == "proof_of_label"
    assert result["body_raw"] == "The explicit opt-arg-labeled proof."
    assert result["multi_proof"] is True
    assert "nested" in result["notes"]


# --- resolve_proof_refs --------------------------------------------------


def test_resolve_proof_refs_resolved_and_unresolved_split():
    label_index = {"eq:main": "E = mc^2", "lem:aux": "Auxiliary lemma content."}
    body = r"By \eqref{eq:main} and \ref{lem:aux}, and also \ref{eq:ghost}, we conclude."
    resolved, unresolved = pm.resolve_proof_refs(body, label_index)
    assert resolved == {"eq:main": "E = mc^2", "lem:aux": "Auxiliary lemma content."}
    assert unresolved == ["eq:ghost"]


def test_resolve_proof_refs_never_modifies_the_body():
    """E5 philosophy: resolution is metadata alongside, never a rewrite."""
    body = r"By \eqref{eq:main} we conclude the bound."
    label_index = {"eq:main": "content"}
    before = body
    pm.resolve_proof_refs(body, label_index)
    assert body == before  # the string passed in is never mutated
    resolved, _ = pm.resolve_proof_refs(body, label_index)
    assert resolved  # sanity: it did resolve something, this isn't a no-op check


def test_resolve_proof_refs_empty_label_index_leaves_everything_unresolved():
    body = r"See \ref{a} and \cref{b}."
    resolved, unresolved = pm.resolve_proof_refs(body, {})
    assert resolved == {}
    assert unresolved == ["a", "b"]


def test_resolve_proof_refs_no_refs_at_all():
    resolved, unresolved = pm.resolve_proof_refs("No references in this proof body.", {"x": "y"})
    assert resolved == {}
    assert unresolved == []


# --- classify_proof_body --------------------------------------------------


def test_classify_proof_body_short_body_is_omitted():
    assert pm.classify_proof_body("Trivial.") == "omitted"


def test_classify_proof_body_stub_pattern_is_omitted():
    assert pm.classify_proof_body("The proof is left to the reader as an easy exercise.") == "omitted"
    assert pm.classify_proof_body("This is standard.") == "omitted"
    assert pm.classify_proof_body("See [12] for the argument.") == "omitted"
    assert pm.classify_proof_body(r"See \cite{Foo99} for details.") == "omitted"


def test_classify_proof_body_long_proof_containing_follows_from_stays_substantive():
    """A real, long derivation that happens to use the phrase 'follows
    from' mid-proof must NOT be misclassified as a stub."""
    long_proof = (
        "This follows from a long and careful chain of estimates that we now spell out in full: "
        "first we bound the energy norm using the standard Gronwall argument, then we combine "
        "this with the Sobolev embedding to control the nonlinear term, and finally we pass to "
        "the limit using weak compactness, which together establish the claimed convergence rate."
    )
    assert len(" ".join(long_proof.split())) >= 200
    assert pm.classify_proof_body(long_proof) == "substantive"


def test_classify_proof_body_ordinary_substantive_proof():
    body = (
        "We proceed by induction on n. The base case n=0 is immediate from direct computation of "
        "both sides. For the inductive step, assume the claim holds for n; we then verify it for "
        "n+1 by expanding the recursive definition and applying the inductive hypothesis together "
        "with a routine estimate on the remainder term, which completes the induction."
    )
    assert pm.classify_proof_body(body) == "substantive"


# --- mine_paper ----------------------------------------------------------


def test_mine_paper_end_to_end_three_theorems_two_proofs_exact_census():
    tex = r"""
\begin{theorem}\label{thm:one}
Statement one is true and holds under the standing hypotheses of this section.
\end{theorem}
\begin{proof}
Direct proof of statement one, spelled out in full detail across many words to be
clearly substantive and not a stub in any way; we derive the result carefully here.
\end{proof}

\begin{lemma}\label{lem:two}
Statement two also holds under the same hypotheses as above with additional detail.
\end{lemma}
\begin{proof}[Proof of Lemma~\ref{lem:two}]
This is the proof of statement two, written out with enough substantive mathematical
content and reasoning to not be classified as a stub proof at all by any measure.
\end{proof}

\begin{theorem}\label{thm:three}
Statement three is a standalone claim with no proof anywhere in this paper at all.
\end{theorem}
"""
    records = [
        _record("u1", "Statement one is true and holds under the standing hypotheses of this section.", "theorem"),
        _record("u2", "Statement two also holds under the same hypotheses as above with additional detail.", "lemma"),
        _record("u3", "Statement three is a standalone claim with no proof anywhere in this paper at all.", "theorem"),
    ]
    results = pm.mine_paper(tex, records)
    assert len(results) == 3
    by_uid = {r["uid"]: r for r in results}

    assert by_uid["u1"]["class"] == "matched"
    assert by_uid["u1"]["match_method"] == "adjacency"
    assert by_uid["u2"]["class"] == "matched"
    assert by_uid["u2"]["match_method"] == "proof_of_label"
    assert by_uid["u3"]["class"] == "unmatched_no_proof"
    assert by_uid["u3"]["theorem_located"] is True

    census = collections.Counter(r["class"] for r in results)
    assert dict(census) == {"matched": 2, "unmatched_no_proof": 1}
    assert sum(census.values()) == len(records) == 3


def test_mine_paper_zero_proof_envs_short_circuits_every_record():
    tex = r"\begin{theorem}\label{thm:x}A claim with a proof nowhere in this document.\end{theorem}"
    records = [
        _record("u1", "A claim with a proof nowhere in this document.", "theorem"),
        _record("u2", "A second, unrelated claim not even present in the tex.", "theorem"),
    ]
    results = pm.mine_paper(tex, records)
    assert len(results) == 2
    assert all(r["class"] == "paper_has_no_proof_env" for r in results)
    assert all(r["theorem_located"] is False for r in results)


def test_mine_paper_unmatched_statement_not_found():
    tex = (
        r"\begin{theorem}\label{thm:x}Statement that exists in the paper.\end{theorem}"
        r"\begin{proof}Some proof body.\end{proof}"
    )
    records = [_record("u1", "A statement that was never mined from this paper at all.", "theorem")]
    results = pm.mine_paper(tex, records)
    [result] = results
    assert result["class"] == "unmatched_statement_not_found"
    assert result["theorem_located"] is False


def test_mine_paper_proof_omitted_class_still_carries_proof_raw():
    tex = (
        r"\begin{theorem}\label{thm:x}A theorem whose proof is a stub.\end{theorem}"
        r"\begin{proof}This is standard.\end{proof}"
    )
    records = [_record("u1", "A theorem whose proof is a stub.", "theorem")]
    [result] = pm.mine_paper(tex, records)
    assert result["class"] == "proof_omitted"
    assert result["proof_raw"] == "This is standard."
    assert result["match_method"] == "adjacency"


def test_mine_paper_never_drops_a_record_even_when_nothing_matches():
    # A real (non-empty) proof env, so the zero-proof-env fast path does not
    # fire — this exercises the per-record unmatched_statement_not_found
    # path instead (already covered on its own above; here the point is
    # that every one of several non-matching records still comes back).
    tex = r"\begin{theorem}\label{thm:x}The only real statement in this paper.\end{theorem}\begin{proof}proof body\end{proof}"
    records = [_record(f"u{i}", f"statement {i} matches nothing in the paper", "theorem") for i in range(5)]
    results = pm.mine_paper(tex, records)
    assert len(results) == 5
    assert {r["uid"] for r in results} == {f"u{i}" for i in range(5)}
    assert all(r["class"] == "unmatched_statement_not_found" for r in results)


# --- realmath must stay byte-identical after importing proof_mine --------


def test_realmath_source_unchanged_before_and_after_importing_proof_mine():
    before = inspect.getsource(realmath.extract_theorem_candidates)
    import icepick.allocation.scrape.proof_mine  # noqa: F401,F811 — re-import, exercises any import-time side effects
    after = inspect.getsource(realmath.extract_theorem_candidates)
    assert before == after


def test_proof_mine_defines_its_own_regex_and_reuses_the_theorem_envs_tuple():
    # proof_mine has its own module-level compiled pattern — it is not
    # merely re-exporting/aliasing realmath's private _ENV_RE attribute.
    assert hasattr(pm, "_THEOREM_ENV_RE")
    assert isinstance(pm._THEOREM_ENV_RE, re.Pattern)
    # It is genuinely the *same* name tuple realmath uses (imported, not
    # hand-copied) — the one piece of state this module is allowed to share.
    assert pm.realmath._THEOREM_ENVS is realmath._THEOREM_ENVS

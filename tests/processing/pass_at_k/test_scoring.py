"""Pure extraction/tally/label helpers — no backend, no sympy parsing."""

from __future__ import annotations

import pytest

from icepick.contracts.records import BAND_HI, BAND_LO
from icepick.processing.pass_at_k.base import (
    ROLLOUT_CORRECT,
    ROLLOUT_DEGENERATE,
    ROLLOUT_WRONG,
)
from icepick.processing.pass_at_k.scoring import (
    JUNK,
    derive_label,
    extract_boxed,
    extract_candidate,
    in_band,
    strip_think,
    tally_rollouts,
    truth_garbage,
)


# --- extract_boxed -----------------------------------------------------------


def test_extract_boxed_simple():
    assert extract_boxed("The answer is \\boxed{42}.") == "42"


def test_extract_boxed_nested_braces():
    assert extract_boxed("so \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"


def test_extract_boxed_absent_or_braceless():
    assert extract_boxed("no boxed here") is None
    assert extract_boxed("dangling \\boxed with no brace") is None


def test_extract_boxed_takes_the_last():
    text = "first \\boxed{1} then later \\boxed{2}"
    assert extract_boxed(text) == "2"


# --- extract_candidate -------------------------------------------------------


def test_extract_candidate_fallback_chain():
    # Tier 1: boxed wins even when dollars are present.
    assert extract_candidate("$x$ and \\boxed{7}") == "7"
    # Tier 2: last $...$ block.
    assert extract_candidate("we get $x=1$ hence $ 3 $") == "3"
    # Tier 3: answer-prose regex, trailing period rstripped.
    assert extract_candidate("Final Answer: 42.") == "42"
    assert extract_candidate("the ANSWER = 9") == "9"
    # Nothing extractable.
    assert extract_candidate("I give up") is None


# --- strip_think -------------------------------------------------------------


def test_strip_think_variants():
    assert strip_think("<think>hmm</think>42") == "42"
    assert strip_think("<think>line1\nline2\n</think>\n7") == "7"
    assert strip_think("no tags at all") == "no tags at all"
    assert strip_think("<think>a</think>x<think>b</think>y") == "xy"


# --- truth_garbage / in_band -------------------------------------------------


def test_truth_garbage_flags_macros_and_passes_clean():
    assert truth_garbage("\\mathrm{deg}")
    assert truth_garbage("\\text{no solution}")
    assert truth_garbage("\\displaystyle\\frac{1}{2}")
    for junk in JUNK:
        assert truth_garbage(f"\\{junk}{{x}}")
    assert not truth_garbage("3/4")
    assert not truth_garbage("\\frac{\\pi}{2}")
    assert not truth_garbage(17)  # non-str truths are stringified


def test_in_band_boundaries_inclusive():
    assert BAND_LO == 0.125 and BAND_HI == 0.75  # contract, not MB's 0.875
    assert in_band(0.125)
    assert in_band(0.75)
    assert not in_band(0.1249)
    assert not in_band(0.7501)
    assert not in_band(None)


# --- tally_rollouts ----------------------------------------------------------


def test_tally_rollouts_mixed_verdicts():
    verdicts = [ROLLOUT_CORRECT, ROLLOUT_WRONG, ROLLOUT_WRONG,
                ROLLOUT_DEGENERATE, ROLLOUT_WRONG]
    candidates = ["2", "3", "3", None, "5"]
    t = tally_rollouts(verdicts, candidates)
    assert t["n_correct"] == 1
    assert t["n_wrong"] == 3
    assert t["n_degenerate"] == 1
    assert t["modal_wrong"] == "3"
    assert t["top_wrong_share"] == pytest.approx(2 / 5)  # over ALL 5, not n_wrong


def test_tally_rollouts_modal_wrong_tie_goes_first_seen():
    verdicts = [ROLLOUT_WRONG] * 4
    candidates = ["a", "b", "b", "a"]
    t = tally_rollouts(verdicts, candidates)
    assert t["modal_wrong"] == "a"
    assert t["top_wrong_share"] == pytest.approx(0.5)


def test_tally_rollouts_degenerates_dilute_the_share():
    verdicts = [ROLLOUT_WRONG, ROLLOUT_DEGENERATE, ROLLOUT_DEGENERATE,
                ROLLOUT_DEGENERATE]
    candidates = ["x", None, None, None]
    t = tally_rollouts(verdicts, candidates)
    assert t["top_wrong_share"] == pytest.approx(1 / 4)  # not 1/1


def test_tally_rollouts_no_wrongs_and_empty():
    t = tally_rollouts([ROLLOUT_CORRECT, ROLLOUT_DEGENERATE], ["1", None])
    assert t["modal_wrong"] is None
    assert t["top_wrong_share"] == 0.0
    t = tally_rollouts([], [])
    assert t == {"n_correct": 0, "n_wrong": 0, "n_degenerate": 0,
                 "modal_wrong": None, "top_wrong_share": 0.0}


def test_tally_rollouts_rejects_bad_input():
    with pytest.raises(ValueError):
        tally_rollouts([ROLLOUT_CORRECT], [])
    with pytest.raises(ValueError):
        tally_rollouts(["maybe"], ["1"])


# --- derive_label parity (the critical pin) ----------------------------------


def test_derive_label_spot_checks():
    assert derive_label(None, 0.0) == "other"
    assert derive_label(1.0, 0.0) == "solved"
    assert derive_label(0.75, 0.0) == "band"     # BAND_HI inclusive on band side
    assert derive_label(0.125, 0.0) == "band"    # BAND_LO inclusive
    assert derive_label(0.0, 0.5) == "misdirection"
    assert derive_label(0.0, 0.49) == "collapse"


def test_derive_label_parity_with_schema_normalise_label():
    from icepick.processing.schema import _normalise_label

    for p in [None] + [i / 16 for i in range(17)]:
        for tws in [0.0, 0.4, 0.5, 0.6, 1.0]:
            assert derive_label(p, tws) == _normalise_label({}, p, tws), (p, tws)

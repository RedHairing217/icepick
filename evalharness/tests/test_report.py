"""Tests for evalharness.report.

Covers, per the task's required list:
  * McNemar exact values on hand-computed cases.
  * report golden-file smoke (full markdown, byte-for-byte against a
    checked-in fixture).

Plus focused unit tests for the paired table and the Wald CI helper,
since those are easy to get subtly wrong (off-by-one on which cell is
"b" vs "c", CI blowing up on n_pairs == 0, etc.).
"""

from __future__ import annotations

import pytest

from evalharness.report import (
    UNDERPOWERED_THRESHOLD,
    generate_report,
    mcnemar_exact,
    paired_table,
    wald_ci_paired_diff,
)

FIXED_TIMESTAMP = "2099-01-01T00:00:00Z"  # keeps golden-file comparisons byte-stable


# --- McNemar exact: hand-computed cases -----------------------------------------


@pytest.mark.parametrize(
    "b, c, expected_p",
    [
        # n=0: no discordant pairs at all -- no directional evidence either way.
        (0, 0, 1.0),
        # n=10, k=1: tail = C(10,0)+C(10,1) = 1+10 = 11; p = 2*11/1024.
        (1, 9, 22 / 1024),
        # n=10, k=2: tail = C(10,0)+C(10,1)+C(10,2) = 1+10+45 = 56; p = 2*56/1024.
        (2, 8, 112 / 1024),
        # n=5, k=0: tail = C(5,0) = 1; p = 2*1/32.
        (0, 5, 2 / 32),
        # b == c must always be non-significant (the observed split IS the
        # null's expectation); the raw formula would give > 1, so this also
        # pins the min(1.0, ...) cap.
        (3, 3, 1.0),
        (1, 1, 1.0),
    ],
)
def test_mcnemar_exact_hand_computed(b, c, expected_p):
    assert mcnemar_exact(b, c) == pytest.approx(expected_p, abs=1e-12)


def test_mcnemar_exact_symmetric_in_b_c():
    assert mcnemar_exact(1, 9) == pytest.approx(mcnemar_exact(9, 1))


def test_mcnemar_exact_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


# --- paired table ----------------------------------------------------------------


def test_paired_table_cells():
    base = {"u1": {"n_correct": 1}, "u2": {"n_correct": 0}, "u3": {"n_correct": 0}, "u4": {"n_correct": 1}}
    tuned = {"u1": {"n_correct": 1}, "u2": {"n_correct": 1}, "u3": {"n_correct": 0}, "u4": {"n_correct": 0}}
    table = paired_table(["u1", "u2", "u3", "u4"], base, tuned)
    assert (table.a, table.b, table.c, table.d) == (1, 1, 1, 1)
    assert table.n_pairs == 4
    assert table.base_solved_n == 2
    assert table.tuned_solved_n == 2
    assert table.missing_uids == []


def test_paired_table_excludes_unscored_uids():
    base = {"u1": {"n_correct": 1}}
    tuned = {"u1": {"n_correct": 1}}
    table = paired_table(["u1", "u2"], base, tuned)  # u2 absent from both
    assert table.n_pairs == 1
    assert table.missing_uids == ["u2"]


def test_wald_ci_zero_pairs_does_not_crash():
    assert wald_ci_paired_diff(0, 0, 0) == (0.0, 0.0)


def test_wald_ci_widens_around_zero_when_balanced():
    lo, hi = wald_ci_paired_diff(2, 2, 10)
    assert lo < 0 < hi


# --- golden-file smoke -----------------------------------------------------------


def test_report_golden_file_smoke(fixtures_dir):
    report_dir = fixtures_dir / "report"
    md = generate_report(
        eval_set_path=report_dir / "eval_set.jsonl",
        baseline_path=report_dir / "baseline_greedy.jsonl",
        post_path=report_dir / "post_greedy.jsonl",
        secondary_base_paths=[report_dir / f"secondary_base_rep{i}.jsonl" for i in range(3)],
        secondary_post_paths=[report_dir / f"secondary_tuned_rep{i}.jsonl" for i in range(3)],
        generated_at=FIXED_TIMESTAMP,
    )
    golden_path = report_dir / "expected_report.md"
    expected = golden_path.read_text(encoding="utf-8")
    assert md == expected


def test_report_without_secondary_says_so_explicitly(fixtures_dir):
    report_dir = fixtures_dir / "report"
    md = generate_report(
        eval_set_path=report_dir / "eval_set.jsonl",
        baseline_path=report_dir / "baseline_greedy.jsonl",
        post_path=report_dir / "post_greedy.jsonl",
        generated_at=FIXED_TIMESTAMP,
    )
    assert "(secondary distributional comparison not provided)" in md
    assert "UNDERPOWERED" in md  # this fixture's eval-band n=6 < 25


def test_report_reflects_hand_verified_eval_band_numbers(fixtures_dir):
    """Cross-check the golden report's headline against the hand-derived
    a/b/c/d for this fixture (see the module docstring's fixture design):
    a=1 (e1), b=1 (e5, base-only/regression), c=3 (e2,e3,e4, tuned-only),
    d=1 (e6) -> n=6, base_solved=2, tuned_solved=4, McNemar(b=1,c=3)."""
    report_dir = fixtures_dir / "report"
    md = generate_report(
        eval_set_path=report_dir / "eval_set.jsonl",
        baseline_path=report_dir / "baseline_greedy.jsonl",
        post_path=report_dir / "post_greedy.jsonl",
        generated_at=FIXED_TIMESTAMP,
    )
    assert mcnemar_exact(1, 3) == pytest.approx(0.625)
    assert "n (eval-band, paired) = 6" in md
    assert "base solved: 2 / 6" in md
    assert "tuned solved: 4 / 6" in md
    assert "exact McNemar p = 0.625" in md
    assert "UNDERPOWERED" in md and str(UNDERPOWERED_THRESHOLD) in md
    # anchor drift red flags: s2 regresses (solved->unsolved), f2 contaminates (unsolved->solved)
    assert "regressed (base solved, tuned did not): 1 **(RED FLAG)**" in md
    assert "contaminated (base failed, tuned solved): 1 **(RED FLAG)**" in md


def test_report_hard_fails_on_empty_eval_band(tmp_path):
    from evalharness.report import ReportError

    eval_set = tmp_path / "eval_set.jsonl"
    eval_set.write_text('{"uid": "a1", "eval_slice": "anchor_solved", "statement": "s", "answer": "a"}\n')
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text('{"uid": "a1", "n_correct": 1}\n')
    post = tmp_path / "post.jsonl"
    post.write_text('{"uid": "a1", "n_correct": 1}\n')

    with pytest.raises(ReportError, match="zero eval_slice"):
        generate_report(eval_set_path=eval_set, baseline_path=baseline, post_path=post)

"""N-way comparator: agreement classification, pairwise kappa, markdown rollup."""

from __future__ import annotations

from icepick.processing.poser.base import (
    STATUS_DEFER,
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserVerdict,
)
from icepick.processing.poser.comparator import (
    HAS_MISSING,
    SPLIT,
    UNANIMOUS_DEFER,
    UNANIMOUS_FAIL,
    UNANIMOUS_PASS,
    compare_verdicts,
    write_comparison_report,
)


def _v(uid, status, combo_key):
    return PoserVerdict(uid=uid, source="s", verdict_status=status,
                        verdict_score=1.0, poser_name=combo_key, poser_model="m")


def test_two_combo_buckets_cover_unanimous_and_split():
    a, b = "claude:anthropic", "codex:openai"
    verdicts_by_combo = {
        a: [_v("u1", STATUS_WELL_POSED, a), _v("u2", STATUS_ILL_POSED, a),
            _v("u3", STATUS_DEFER, a),       _v("u4", STATUS_WELL_POSED, a)],
        b: [_v("u1", STATUS_WELL_POSED, b), _v("u2", STATUS_ILL_POSED, b),
            _v("u3", STATUS_DEFER, b),       _v("u4", STATUS_ILL_POSED, b),
            _v("u5", STATUS_WELL_POSED, b)],
    }
    cmp = compare_verdicts(verdicts_by_combo=verdicts_by_combo, statements_by_uid={})
    counts = cmp.counts
    assert counts[UNANIMOUS_PASS] == 1
    assert counts[UNANIMOUS_FAIL] == 1
    assert counts[UNANIMOUS_DEFER] == 1
    assert counts[SPLIT] == 1
    assert counts[HAS_MISSING] == 1
    assert cmp.total == 5


def test_four_combo_fleet_classifies_unanimous_and_split():
    keys = ["claude:anthropic", "claude:openai", "codex:anthropic", "codex:openai"]
    # u1: all four pass → unanimous_pass
    # u2: 3 pass, 1 fail → split
    verdicts_by_combo = {
        keys[0]: [_v("u1", STATUS_WELL_POSED, keys[0]), _v("u2", STATUS_WELL_POSED, keys[0])],
        keys[1]: [_v("u1", STATUS_WELL_POSED, keys[1]), _v("u2", STATUS_WELL_POSED, keys[1])],
        keys[2]: [_v("u1", STATUS_WELL_POSED, keys[2]), _v("u2", STATUS_WELL_POSED, keys[2])],
        keys[3]: [_v("u1", STATUS_WELL_POSED, keys[3]), _v("u2", STATUS_ILL_POSED, keys[3])],
    }
    cmp = compare_verdicts(verdicts_by_combo=verdicts_by_combo, statements_by_uid={})
    assert cmp.counts[UNANIMOUS_PASS] == 1
    assert cmp.counts[SPLIT] == 1


def test_pairwise_kappa_table_has_one_entry_per_pair():
    keys = ["claude:anthropic", "claude:openai", "codex:openai"]
    verdicts_by_combo = {k: [_v("u1", STATUS_WELL_POSED, k)] for k in keys}
    cmp = compare_verdicts(verdicts_by_combo=verdicts_by_combo, statements_by_uid={})
    # 3 combos → C(3,2) = 3 pairs
    assert len(cmp.pairwise_kappa) == 3
    expected_pairs = {(keys[0], keys[1]), (keys[0], keys[2]), (keys[1], keys[2])}
    assert set(cmp.pairwise_kappa.keys()) == expected_pairs


def test_markdown_report_lists_fleet_and_pairwise_section(tmp_path):
    keys = ["claude:anthropic", "codex:openai"]
    verdicts_by_combo = {
        keys[0]: [_v("u1", STATUS_WELL_POSED, keys[0])],
        keys[1]: [_v("u1", STATUS_ILL_POSED, keys[1])],
    }
    cmp = compare_verdicts(
        verdicts_by_combo=verdicts_by_combo,
        statements_by_uid={"u1": "the statement under test"},
    )
    out = tmp_path / "report.md"
    write_comparison_report(cmp, out, combo_keys=keys)
    text = out.read_text()
    assert "Fleet size" in text
    assert text.index("Agreement counts") < text.index("Pairwise")
    assert "the statement under test" in text
    # Every combo column appears in the disagreement table header
    for k in keys:
        assert k in text

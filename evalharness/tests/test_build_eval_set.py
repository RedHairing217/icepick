"""Tests for evalharness.build_eval_set.

Covers, per the task's required list:
  * split-immutability: the sha pin matches the real, committed
    eval_paper_split.json (and a mismatch is hard-failed).
  * leakage guard trips on a planted violation.
  * build_eval_set end-to-end on a small synthetic fixture.

Plus a few cheap adjacent cases (missing tier-output path, missing
statement/answer, empty --tier-outputs) that exercise the same hard-fail
surface with negligible extra cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalharness.build_eval_set import (
    EVAL_SLICE_ANCHOR_FAIL,
    EVAL_SLICE_ANCHOR_SOLVED,
    EVAL_SLICE_BAND,
    EXPECTED_SPLIT_SHA256_16,
    EvalSetError,
    assert_has_statement_and_answer,
    assert_no_leakage,
    build,
    load_split,
    sha256_16,
)

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../icepick
REAL_SPLIT_PATH = REPO_ROOT / "evalharness" / "data" / "eval_paper_split.json"


def _write_jsonl(path: Path, rows: list) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


# --- split immutability --------------------------------------------------------


def test_pinned_constant_matches_the_real_frozen_split():
    """The module constant must match the real, committed artifact's hash.

    This is the tripwire for the frozen-artifact invariant: if someone
    regenerates or reformats evalharness/data/eval_paper_split.json, this
    test (and build_eval_set's own load_split hard-fail) catches it.
    """
    assert REAL_SPLIT_PATH.exists(), f"frozen split missing: {REAL_SPLIT_PATH}"
    assert sha256_16(REAL_SPLIT_PATH) == EXPECTED_SPLIT_SHA256_16 == "110a4bf27320f2b1"


def test_load_split_succeeds_against_the_real_file():
    data = load_split(REAL_SPLIT_PATH)
    assert isinstance(data["eval_papers"], list)
    assert len(data["eval_papers"]) == data.get("eval_papers_n", len(data["eval_papers"]))


def test_load_split_hard_fails_on_sha_mismatch(fixtures_dir):
    tiny_split = fixtures_dir / "tiny_split.json"
    with pytest.raises(EvalSetError, match="integrity check FAILED"):
        load_split(tiny_split, expected_sha16="0000000000000000")


def test_load_split_hard_fails_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path / "does_not_exist.json")


# --- leakage guard --------------------------------------------------------------


def test_leakage_guard_trips_on_planted_violation():
    """Plant a train record whose arxiv_id is an eval paper; the guard must refuse."""
    eval_papers = {"9999.00001"}
    planted_train_records = {
        "innocent-uid": {"arxiv_id": "1000.00001", "label": "band"},
        "leaked-uid": {"arxiv_id": "9999.00001", "label": "band"},  # planted violation
    }
    with pytest.raises(EvalSetError, match="LEAKAGE GUARD TRIPPED"):
        assert_no_leakage(planted_train_records, eval_papers)


def test_leakage_guard_passes_on_clean_train_set():
    eval_papers = {"9999.00001"}
    clean_train_records = {"innocent-uid": {"arxiv_id": "1000.00001", "label": "band"}}
    assert_no_leakage(clean_train_records, eval_papers) is None  # does not raise


# --- statement/answer guard ------------------------------------------------------


def test_statement_answer_guard_trips_on_blank_answer():
    records = {
        "ok-uid": {"statement": "S", "answer": "A"},
        "bad-uid": {"statement": "S", "answer": ""},
    }
    with pytest.raises(EvalSetError, match="missing statement/answer"):
        assert_has_statement_and_answer(records)


def test_statement_answer_guard_trips_on_missing_key():
    records = {"bad-uid": {"statement": "S"}}  # no "answer" key at all
    with pytest.raises(EvalSetError, match="missing statement/answer"):
        assert_has_statement_and_answer(records)


# --- end-to-end on a small synthetic fixture -------------------------------------


def test_build_end_to_end_synthetic_fixture(fixtures_dir, tmp_path):
    split_path = fixtures_dir / "tiny_split.json"
    tier1 = fixtures_dir / "tiny_tier1.jsonl"
    tier2 = fixtures_dir / "tiny_tier2.jsonl"
    output_dir = tmp_path / "out"

    result = build(
        split_path=split_path,
        tier_output_paths=[tier1, tier2],
        output_dir=output_dir,
        expected_split_sha16=sha256_16(split_path),
    )

    assert result.counts["eval_papers_n"] == 2
    assert result.counts["tier_files"] == 2
    assert result.counts["eval_band"] == 2  # u1, u9
    assert result.counts["anchor_solved"] == 1  # u2 (8/8); u3 (7/8) excluded
    assert result.counts["anchor_fail"] == 1  # u4 (0/8 collapse); u5 (0/8 misdirection) excluded
    assert result.counts["eval_set_total"] == 4
    assert result.counts["train_band_total"] == 1  # u7 only; u8 is solved not band
    assert result.counts["duplicate_uids"] == 1  # u1 repeated in tier2

    assert result.eval_set_path == output_dir / "eval_set.jsonl"
    assert result.train_uids_path == output_dir / "train_uids.txt"
    assert result.eval_set_path.exists()
    assert result.train_uids_path.exists()

    by_uid = {}
    with result.eval_set_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            by_uid[row["uid"]] = row
    assert set(by_uid) == {"u1", "u9", "u2", "u4"}
    assert by_uid["u1"]["eval_slice"] == EVAL_SLICE_BAND
    assert by_uid["u9"]["eval_slice"] == EVAL_SLICE_BAND
    assert by_uid["u2"]["eval_slice"] == EVAL_SLICE_ANCHOR_SOLVED
    assert by_uid["u4"]["eval_slice"] == EVAL_SLICE_ANCHOR_FAIL
    # Kept the FIRST-seen occurrence of the duplicated uid (tier1's u1, not
    # tier2's corrupted duplicate).
    assert by_uid["u1"]["answer"] == "A1"

    train_uids = result.train_uids_path.read_text().split()
    assert train_uids == ["u7"]

    assert any("duplicate uid" in w and "u1" in w for w in result.warnings)


def test_build_end_to_end_is_deterministic_across_runs(fixtures_dir, tmp_path):
    """Re-running build() with the same inputs must produce byte-identical outputs."""
    split_path = fixtures_dir / "tiny_split.json"
    tiers = [fixtures_dir / "tiny_tier1.jsonl", fixtures_dir / "tiny_tier2.jsonl"]
    sha = sha256_16(split_path)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    build(split_path=split_path, tier_output_paths=tiers, output_dir=out1, expected_split_sha16=sha)
    build(split_path=split_path, tier_output_paths=tiers, output_dir=out2, expected_split_sha16=sha)

    assert (out1 / "eval_set.jsonl").read_text() == (out2 / "eval_set.jsonl").read_text()
    assert (out1 / "train_uids.txt").read_text() == (out2 / "train_uids.txt").read_text()


# --- missing / malformed input -----------------------------------------------------


def test_missing_tier_output_gives_a_clear_actionable_error(fixtures_dir, tmp_path):
    split_path = fixtures_dir / "tiny_split.json"
    missing_path = tmp_path / "out" / "remote_rescore" / "tier3_misdirection" / "pass_at_k.jsonl"

    with pytest.raises(FileNotFoundError) as exc_info:
        build(
            split_path=split_path,
            tier_output_paths=[fixtures_dir / "tiny_tier1.jsonl", missing_path],
            output_dir=tmp_path / "out_dir",
            expected_split_sha16=sha256_16(split_path),
        )
    message = str(exc_info.value)
    assert str(missing_path) in message
    assert "cascade incomplete" in message


def test_empty_tier_outputs_raises():
    with pytest.raises(EvalSetError, match="at least one"):
        build(
            split_path=REAL_SPLIT_PATH,
            tier_output_paths=[],
            output_dir=Path("/tmp/should-not-be-created"),
        )


def test_missing_statement_or_answer_hard_fails_end_to_end(fixtures_dir, tmp_path):
    split_path = fixtures_dir / "tiny_split.json"
    bad_tier = _write_jsonl(
        tmp_path / "bad_tier.jsonl",
        [
            {
                "uid": "bad-uid",
                "arxiv_id": "9999.00001",
                "label": "band",
                "statement": "has a statement",
                "answer": "",  # blank -- must hard-fail
                "n_correct": 3,
                "n_wrong": 5,
                "n_degenerate": 0,
                "pass_at_k": 0.375,
            }
        ],
    )
    with pytest.raises(EvalSetError, match="missing statement/answer"):
        build(
            split_path=split_path,
            tier_output_paths=[bad_tier],
            output_dir=tmp_path / "out",
            expected_split_sha16=sha256_16(split_path),
        )


def test_tier_output_with_no_label_field_warns_but_does_not_crash(fixtures_dir, tmp_path):
    """A pre-scoring pass_at_k_input.jsonl (no 'label' yet) shouldn't crash --
    it should just contribute nothing, with a warning pointing at the likely mistake."""
    split_path = fixtures_dir / "tiny_split.json"
    input_only_tier = _write_jsonl(
        tmp_path / "pass_at_k_input.jsonl",
        [{"uid": "x1", "arxiv_id": "9999.00001", "statement": "S", "answer": "A"}],
    )
    result = build(
        split_path=split_path,
        tier_output_paths=[input_only_tier],
        output_dir=tmp_path / "out",
        expected_split_sha16=sha256_16(split_path),
    )
    assert result.counts["eval_band"] == 0
    assert result.counts["train_band_total"] == 0
    assert any("pass_at_k_input.jsonl" in w or "no record had a 'label'" in w for w in result.warnings)

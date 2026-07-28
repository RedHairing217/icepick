"""Tests for loratrain.build_dataset's guard functions (leakage, dedupe, pins).

Synthetic data only -- no real corpus, no network. See README "Split &
corpus" for the invariants these guards enforce.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from loratrain.build_dataset import (
    DuplicateRecordError,
    LeakageError,
    PinMismatchError,
    assert_corpus_pinned,
    assert_no_cross_uid_statement_dups,
    assert_no_leakage,
    dedupe_examples,
    load_eval_papers,
)

# --- leakage: paper-level / uid-level / clean / nested ------------------------


def test_paper_level_leak_raises():
    eval_papers = {"9999.00001"}
    eval_uids = set()
    examples = [
        {"uid": "u1", "arxiv_id": "1000.00001"},
        {"uid": "u2", "arxiv_id": "9999.00001"},  # leak: eval paper
    ]
    with pytest.raises(LeakageError):
        assert_no_leakage(examples, eval_papers, eval_uids)


def test_uid_level_leak_raises():
    eval_papers = set()
    eval_uids = {"eval-uid-1"}
    examples = [
        {"uid": "u1", "arxiv_id": "1000.00001"},
        {"uid": "eval-uid-1", "arxiv_id": "1000.00002"},  # leak: eval uid
    ]
    with pytest.raises(LeakageError):
        assert_no_leakage(examples, eval_papers, eval_uids)


def test_clean_set_passes():
    eval_papers = {"9999.00001"}
    eval_uids = {"eval-uid-1"}
    examples = [
        {"uid": "u1", "arxiv_id": "1000.00001"},
        {"uid": "u2", "arxiv_id": "1000.00002"},
    ]
    assert assert_no_leakage(examples, eval_papers, eval_uids) is None


def test_nested_provenance_form_works():
    eval_papers = {"9999.00001"}
    eval_uids = set()

    leaking = [{"provenance": {"uid": "u1", "arxiv_id": "9999.00001"}}]
    with pytest.raises(LeakageError):
        assert_no_leakage(leaking, eval_papers, eval_uids)

    clean = [{"provenance": {"uid": "u1", "arxiv_id": "1000.00001"}}]
    assert assert_no_leakage(clean, eval_papers, eval_uids) is None


# --- dedupe ---------------------------------------------------------------------


def test_dedupe_drops_uid_rollout_uid_dup_and_keeps_order():
    examples = [
        {"uid": "a", "rollout_uid": "r1", "tag": "first"},
        {"uid": "a", "rollout_uid": "r1", "tag": "duplicate"},
        {"uid": "a", "rollout_uid": "r2", "tag": "second"},
        {"uid": "b", "rollout_uid": "r1", "tag": "third"},
    ]
    result = dedupe_examples(examples)
    assert [e["tag"] for e in result] == ["first", "second", "third"]


# --- cross-uid statement dups -----------------------------------------------------


def test_cross_uid_identical_statement_raises():
    records = [
        {"uid": "a", "statement": "Prove X."},
        {"uid": "b", "statement": "Prove X."},
    ]
    with pytest.raises(DuplicateRecordError):
        assert_no_cross_uid_statement_dups(records)


def test_same_uid_repeats_are_fine():
    records = [
        {"uid": "a", "statement": "Prove X."},
        {"uid": "a", "statement": "Prove X."},
        {"uid": "a", "statement": "Prove X."},
    ]
    assert assert_no_cross_uid_statement_dups(records) is None


# --- load_eval_papers sha16 pin -----------------------------------------------


def test_load_eval_papers_correct_sha_passes(tmp_path):
    split_path = tmp_path / "split.json"
    payload = {"eval_papers": ["1111.00001", "2222.00002"]}
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    expected_sha16 = hashlib.sha256(split_path.read_bytes()).hexdigest()[:16]

    result = load_eval_papers(split_path, expected_sha16)
    assert result == {"1111.00001", "2222.00002"}


def test_load_eval_papers_tampered_bytes_raises(tmp_path):
    split_path = tmp_path / "split.json"
    payload = {"eval_papers": ["1111.00001"]}
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    expected_sha16 = hashlib.sha256(split_path.read_bytes()).hexdigest()[:16]

    # Tamper AFTER computing the expected hash.
    split_path.write_text(json.dumps({"eval_papers": ["1111.00001", "extra"]}), encoding="utf-8")

    with pytest.raises(PinMismatchError):
        load_eval_papers(split_path, expected_sha16)


# --- assert_corpus_pinned --------------------------------------------------------


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_corpus_pinned_matching_sha_and_rows_passes(tmp_path):
    corpus_path = tmp_path / "band_corpus.jsonl"
    rows = [{"uid": f"u{i}"} for i in range(5)]
    _write_jsonl(corpus_path, rows)
    expected_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    assert assert_corpus_pinned(corpus_path, expected_sha256, 5) is None


def test_corpus_pinned_wrong_rows_raises(tmp_path):
    corpus_path = tmp_path / "band_corpus.jsonl"
    rows = [{"uid": f"u{i}"} for i in range(5)]
    _write_jsonl(corpus_path, rows)
    expected_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    with pytest.raises(PinMismatchError):
        assert_corpus_pinned(corpus_path, expected_sha256, 999)


def test_corpus_pinned_wrong_sha_raises(tmp_path):
    corpus_path = tmp_path / "band_corpus.jsonl"
    rows = [{"uid": f"u{i}"} for i in range(5)]
    _write_jsonl(corpus_path, rows)

    with pytest.raises(PinMismatchError):
        assert_corpus_pinned(corpus_path, "0" * 64, 5)

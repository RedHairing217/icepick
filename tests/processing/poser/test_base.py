"""Contract surface for the poser stage."""

from __future__ import annotations

from icepick.processing.poser.base import (
    CANONICAL_STATUSES,
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserVerdict,
    compute_uid,
    inject_uid,
)


def test_canonical_status_space_is_four_values():
    assert CANONICAL_STATUSES == (
        STATUS_WELL_POSED,
        STATUS_ILL_POSED,
        STATUS_DEFER,
        STATUS_ERROR,
    )


def test_uid_is_deterministic_and_truncated():
    uid = compute_uid("src", "statement")
    assert len(uid) == 32
    assert all(c in "0123456789abcdef" for c in uid)
    assert uid == compute_uid("src", "statement")


def test_uid_differs_by_source_and_by_statement():
    assert compute_uid("a", "s") != compute_uid("b", "s")
    assert compute_uid("a", "s") != compute_uid("a", "t")


def test_inject_uid_does_not_overwrite_existing():
    out = inject_uid([{"uid": "pre-set", "source": "x", "statement": "y"}])
    assert out[0]["uid"] == "pre-set"


def test_inject_uid_fills_when_absent():
    out = inject_uid([{"source": "x", "statement": "y"}])
    assert out[0]["uid"] == compute_uid("x", "y")


def test_inject_uid_handles_question_and_prompt_aliases():
    a = inject_uid([{"source": "x", "question": "y"}])
    b = inject_uid([{"source": "x", "prompt": "y"}])
    c = inject_uid([{"source": "x", "statement": "y"}])
    assert a[0]["uid"] == b[0]["uid"] == c[0]["uid"]


def test_verdict_serialises_to_jsonl_row_with_canonical_keys():
    v = PoserVerdict(
        uid="abc", source="s", verdict_status=STATUS_WELL_POSED,
        verdict_score=1.0, poser_name="claude", poser_model="opus",
    )
    row = v.to_jsonl_row()
    for key in ("uid", "source", "verdict_status", "verdict_score",
                "poser_name", "poser_model", "verdict_detail",
                "verdict_signals", "raw_payload"):
        assert key in row

"""Contracts + arxiv-id extraction."""

from __future__ import annotations

from icepick.processing.groundtruth.base import (
    CANONICAL_STATUSES,
    STATUS_DEFER,
    STATUS_DISCARDED,
    STATUS_ERROR,
    STATUS_PUBLISHED,
    STATUS_UNPUBLISHED,
    GroundtruthVerdict,
    extract_arxiv_id,
)


def test_canonical_statuses_are_five_values():
    assert CANONICAL_STATUSES == (
        STATUS_PUBLISHED,
        STATUS_UNPUBLISHED,
        STATUS_DEFER,
        STATUS_ERROR,
        STATUS_DISCARDED,
    )


def test_verdict_serialises_to_jsonl_row():
    v = GroundtruthVerdict(
        uid="abc", source="s", verdict_status=STATUS_PUBLISHED,
        arxiv_id="2403.12345", venue="NeurIPS 2024", publication_year=2024,
        indexed_in=["DBLP"], judge_votes=["published"] * 3,
    )
    row = v.to_jsonl_row()
    for key in ("uid", "source", "verdict_status", "arxiv_id", "venue",
                "publication_year", "indexed_in", "evidence_urls",
                "judge_model", "judge_votes", "judge_majority",
                "discarded_reason", "error_reason", "reasoning",
                "confidence", "raw_payload"):
        assert key in row


def test_extract_arxiv_from_canonical_field():
    assert extract_arxiv_id({"arxiv_id": "2403.12345"}) == "2403.12345"
    assert extract_arxiv_id({"arxivId": "2403.12345v2"}) == "2403.12345"
    assert extract_arxiv_id({"arxiv": "arXiv:2403.12345"}) == "2403.12345"


def test_extract_arxiv_from_url():
    assert extract_arxiv_id({"url": "https://arxiv.org/abs/2403.12345"}) == "2403.12345"
    assert extract_arxiv_id({"paper_url": "https://arxiv.org/pdf/2403.12345v2.pdf"}) == "2403.12345"


def test_extract_arxiv_handles_old_style_ids():
    assert extract_arxiv_id({"arxiv_id": "math.AG/0601001"}) == "math.AG/0601001"


def test_extract_arxiv_returns_none_when_absent():
    assert extract_arxiv_id({"source": "manual", "statement": "..."}) is None


def test_extract_arxiv_rejects_garbage():
    assert extract_arxiv_id({"arxiv_id": "not-an-id"}) is None
    assert extract_arxiv_id({"arxiv_id": ""}) is None

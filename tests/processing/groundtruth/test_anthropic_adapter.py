"""Anthropic adapter unit tests — no real API calls.

We test the response parsing and majority-uphold logic against a
hand-built fake client that returns the exact response shape the
Anthropic SDK produces.
"""

from __future__ import annotations

from types import SimpleNamespace

from icepick.processing.groundtruth.anthropic_adapter import (
    AnthropicGroundtruthAdapter,
    _majority_vote,
)
from icepick.processing.groundtruth.base import (
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_PUBLISHED,
    STATUS_UNPUBLISHED,
)
from icepick.processing.groundtruth.config import GroundtruthConfig


def _fake_response(report_verdict_input):
    """Mimic the SDK's Message response — content is a list of blocks."""
    blocks = []
    if report_verdict_input is not None:
        blocks.append(SimpleNamespace(
            type="tool_use",
            name="report_verdict",
            input=report_verdict_input,
        ))
    return SimpleNamespace(
        content=blocks,
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


class _FakeClient:
    """Substitute for anthropic.Anthropic(). Returns wired-in responses per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        if self.calls >= len(self._responses):
            raise IndexError("FakeClient ran out of wired responses")
        resp = self._responses[self.calls]
        self.calls += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


def _cfg():
    return GroundtruthConfig(
        mode="production",
        anthropic_key_file=None,  # irrelevant — we inject the client
        judge_samples=3,
        judge_uphold=2,
    )


# --- _majority_vote unit tests -----------------------------------------------

def test_majority_vote_uphold_when_two_of_three_agree():
    votes = [
        {"verdict_status": STATUS_PUBLISHED},
        {"verdict_status": STATUS_PUBLISHED},
        {"verdict_status": STATUS_DEFER},
    ]
    assert _majority_vote(votes, uphold=2) == STATUS_PUBLISHED


def test_majority_vote_defers_when_no_majority():
    votes = [
        {"verdict_status": STATUS_PUBLISHED},
        {"verdict_status": STATUS_UNPUBLISHED},
        {"verdict_status": STATUS_DEFER},
    ]
    assert _majority_vote(votes, uphold=2) == STATUS_DEFER


def test_majority_vote_errors_when_majority_errored():
    votes = [
        {"verdict_status": STATUS_ERROR},
        {"verdict_status": STATUS_ERROR},
        {"verdict_status": STATUS_PUBLISHED},
    ]
    assert _majority_vote(votes, uphold=2) == STATUS_ERROR


def test_majority_vote_one_error_doesnt_block_two_published():
    """A single transient error should not poison a strong 2-1 majority."""
    votes = [
        {"verdict_status": STATUS_PUBLISHED},
        {"verdict_status": STATUS_PUBLISHED},
        {"verdict_status": STATUS_ERROR},
    ]
    assert _majority_vote(votes, uphold=2) == STATUS_PUBLISHED


# --- adapter integration via fake client -------------------------------------

def test_adapter_returns_published_when_all_three_agree():
    fake = _FakeClient([
        _fake_response({"verdict_status": "published", "reasoning": "ok",
                        "confidence": "high", "venue": "NeurIPS 2024",
                        "publication_year": 2024, "indexed_in": ["DBLP"]}),
        _fake_response({"verdict_status": "published", "reasoning": "ok",
                        "confidence": "high"}),
        _fake_response({"verdict_status": "published", "reasoning": "ok",
                        "confidence": "medium"}),
    ])
    adapter = AnthropicGroundtruthAdapter(_cfg(), client=fake)

    verdict = adapter.lookup_paper(
        arxiv_id="2403.12345",
        paper_title="A Test Paper",
        uid_for_error_attribution="uid_x",
    )

    assert verdict.verdict_status == STATUS_PUBLISHED
    assert verdict.judge_votes == ["published"] * 3
    assert verdict.venue == "NeurIPS 2024"  # picked from high-confidence sample
    assert verdict.publication_year == 2024
    assert fake.calls == 3


def test_adapter_majority_overrides_minority():
    fake = _FakeClient([
        _fake_response({"verdict_status": "published", "reasoning": "found in DBLP",
                        "confidence": "high", "venue": "NeurIPS"}),
        _fake_response({"verdict_status": "published", "reasoning": "DOI resolves",
                        "confidence": "medium"}),
        _fake_response({"verdict_status": "defer", "reasoning": "unclear",
                        "confidence": "low"}),
    ])
    adapter = AnthropicGroundtruthAdapter(_cfg(), client=fake)

    verdict = adapter.lookup_paper(
        arxiv_id="2403.12345", paper_title=None, uid_for_error_attribution="uid_x",
    )
    assert verdict.verdict_status == STATUS_PUBLISHED
    assert verdict.venue == "NeurIPS"


def test_adapter_returns_defer_on_split_vote():
    fake = _FakeClient([
        _fake_response({"verdict_status": "published", "reasoning": "x", "confidence": "low"}),
        _fake_response({"verdict_status": "unpublished", "reasoning": "y", "confidence": "low"}),
        _fake_response({"verdict_status": "defer", "reasoning": "z", "confidence": "low"}),
    ])
    adapter = AnthropicGroundtruthAdapter(_cfg(), client=fake)

    verdict = adapter.lookup_paper(
        arxiv_id="2403.12345", paper_title=None, uid_for_error_attribution="uid_x",
    )
    assert verdict.verdict_status == STATUS_DEFER


def test_adapter_handles_missing_tool_call_as_defer():
    """If Claude exits without calling report_verdict, that single sample defers."""
    fake = _FakeClient([
        _fake_response(None),  # no tool_use block
        _fake_response({"verdict_status": "published", "reasoning": "x", "confidence": "high"}),
        _fake_response({"verdict_status": "published", "reasoning": "y", "confidence": "high"}),
    ])
    adapter = AnthropicGroundtruthAdapter(_cfg(), client=fake)

    verdict = adapter.lookup_paper(
        arxiv_id="2403.12345", paper_title=None, uid_for_error_attribution="uid_x",
    )
    # Two published + one defer (the missing-tool sample) → published.
    assert verdict.verdict_status == STATUS_PUBLISHED
    assert STATUS_DEFER in verdict.judge_votes


def test_adapter_handles_api_exception_as_error_vote():
    fake = _FakeClient([
        RuntimeError("simulated 500"),
        _fake_response({"verdict_status": "published", "reasoning": "x", "confidence": "high"}),
        _fake_response({"verdict_status": "published", "reasoning": "y", "confidence": "high"}),
    ])
    adapter = AnthropicGroundtruthAdapter(_cfg(), client=fake)

    verdict = adapter.lookup_paper(
        arxiv_id="2403.12345", paper_title=None, uid_for_error_attribution="uid_x",
    )
    # One error + two published → published (error doesn't reach majority).
    assert verdict.verdict_status == STATUS_PUBLISHED
    assert STATUS_ERROR in verdict.judge_votes


def test_adapter_records_error_when_all_three_fail():
    fake = _FakeClient([RuntimeError("err1"), RuntimeError("err2"), RuntimeError("err3")])
    adapter = AnthropicGroundtruthAdapter(_cfg(), client=fake)

    verdict = adapter.lookup_paper(
        arxiv_id="2403.12345", paper_title=None, uid_for_error_attribution="uid_x",
    )
    assert verdict.verdict_status == STATUS_ERROR
    assert verdict.error_reason is not None

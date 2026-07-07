"""Tests for the OpenAI judge backend.

We don't hit the real API — urllib.request.urlopen is monkeypatched to a
fake that returns a canned chat-completions payload.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from claude_poser.config import WellposedConfig
from claude_poser.judge import (
    JUDGE_SYSTEM,
    JudgeOutcome,
    JudgeSample,
    _call_openai_once,
    judge_wellposed,
)
from claude_poser.schema import normalise_record
from claude_poser.wellposed import check_record


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok(body: dict) -> _FakeResp:
    return _FakeResp(json.dumps(body).encode("utf-8"))


def _openai_payload(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_openai_caller_returns_parsed_pass(monkeypatch):
    cfg = WellposedConfig(judge_provider="openai", openai_api_key="sk-test")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = dict(req.headers)
        return _ok(_openai_payload(
            '{"verdict": "pass", "insufficient_context": false, "reason": "ok"}'
        ))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    reply = _call_openai_once(cfg, "test prompt")
    assert reply == {"verdict": "pass", "insufficient_context": False, "reason": "ok",
                     "derived_answer": None}
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-4o-mini"  # default for openai
    assert captured["body"]["messages"][0]["role"] == "system"


def test_openai_caller_no_key_returns_error():
    cfg = WellposedConfig(judge_provider="openai", openai_api_key=None)
    reply = _call_openai_once(cfg, "test prompt")
    assert reply["verdict"] == "error"
    assert "openai api key" in reply["reason"].lower()


def test_openai_caller_http_error(monkeypatch):
    cfg = WellposedConfig(judge_provider="openai", openai_api_key="sk-test")

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"rate limited")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    reply = _call_openai_once(cfg, "test prompt")
    assert reply["verdict"] == "error"
    assert "openai http 429" in reply["reason"]


def test_openai_caller_url_error(monkeypatch):
    cfg = WellposedConfig(judge_provider="openai", openai_api_key="sk-test")

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    reply = _call_openai_once(cfg, "test prompt")
    assert reply["verdict"] == "error"
    assert "openai url error" in reply["reason"]


def test_openai_caller_unexpected_payload(monkeypatch):
    cfg = WellposedConfig(judge_provider="openai", openai_api_key="sk-test")
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _ok({"unexpected": "shape"}),
    )
    reply = _call_openai_once(cfg, "test prompt")
    assert reply["verdict"] == "error"
    assert "unexpected payload" in reply["reason"]


def test_openai_caller_honours_base_url(monkeypatch):
    """Pointing OPENAI_BASE_URL at a local OpenAI-compatible server works."""
    cfg = WellposedConfig(
        judge_provider="openai",
        openai_api_key="sk-test",
        openai_base_url="http://127.0.0.1:1234/v1",
    )
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _ok(_openai_payload(
            '{"verdict": "flag", "insufficient_context": true, "reason": "missing"}'
        ))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _call_openai_once(cfg, "test prompt")
    assert captured["url"] == "http://127.0.0.1:1234/v1/chat/completions"


def test_openai_caller_honours_explicit_model(monkeypatch):
    cfg = WellposedConfig(
        judge_provider="openai",
        openai_api_key="sk-test",
        judge_model="gpt-5-thinking",
    )
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _ok(_openai_payload(
            '{"verdict": "pass", "insufficient_context": false, "reason": "ok"}'
        ))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _call_openai_once(cfg, "test prompt")
    assert captured["body"]["model"] == "gpt-5-thinking"


def test_openai_non_reasoning_wire_format_byte_identical(monkeypatch):
    """gpt-4o-mini requests must be byte-for-byte what shipped before the
    reasoning-model branch, so old runs' disk caches stay comparable."""
    cfg = WellposedConfig(judge_provider="openai", openai_api_key="sk-test")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        captured["timeout"] = timeout
        return _ok(_openai_payload(
            '{"verdict": "pass", "insufficient_context": false, "reason": "ok"}'
        ))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _call_openai_once(cfg, "test prompt")
    expected = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": "test prompt"},
        ],
        "max_tokens": 400,
        "temperature": 1.0,
    }).encode("utf-8")
    assert captured["data"] == expected
    assert captured["timeout"] == cfg.judge_timeout_s  # no reasoning floor


def test_openai_reasoning_model_param_surface(monkeypatch):
    """gpt-5.x requests carry reasoning_effort + max_completion_tokens and
    must not carry temperature or max_tokens (the API 400s on both)."""
    cfg = WellposedConfig(
        judge_provider="openai",
        openai_api_key="sk-test",
        judge_model="gpt-5.5",
    )
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _ok(_openai_payload(
            '{"verdict": "pass", "insufficient_context": false, "reason": "ok"}'
        ))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _call_openai_once(cfg, "test prompt")
    body = captured["body"]
    assert body["model"] == "gpt-5.5"
    assert body["reasoning_effort"] == "high"  # default effort
    assert body["max_completion_tokens"] == 4000  # reasoning burns the old 400
    assert "temperature" not in body
    assert "max_tokens" not in body
    # High-effort reasoning overruns the 30s default; the floor prevents
    # systematic timeout->error votes. An explicit larger value would win.
    assert captured["timeout"] == 120.0


def test_openai_reasoning_effort_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
    cfg = WellposedConfig(
        judge_provider="openai",
        openai_api_key="sk-test",
        judge_model="o3-mini",
    )
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _ok(_openai_payload(
            '{"verdict": "pass", "insufficient_context": false, "reason": "ok"}'
        ))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _call_openai_once(cfg, "test prompt")
    assert captured["body"]["reasoning_effort"] == "medium"


def test_openai_usage_records_reasoning_tokens(monkeypatch):
    """completion_tokens_details.reasoning_tokens is captured per sample and
    summed by usage_total() alongside the existing counters."""
    cfg = WellposedConfig(
        judge_provider="openai",
        openai_api_key="sk-test",
        judge_model="gpt-5.5",
    )
    payload = _openai_payload(
        '{"verdict": "pass", "insufficient_context": false, "reason": "ok"}'
    )
    payload["usage"] = {
        "prompt_tokens": 700,
        "completion_tokens": 900,
        "completion_tokens_details": {"reasoning_tokens": 850},
    }
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=None: _ok(payload)
    )
    reply = _call_openai_once(cfg, "test prompt")
    assert reply["usage"] == {
        "input_tokens": 700,
        "output_tokens": 900,
        "reasoning_tokens": 850,
    }
    outcome = JudgeOutcome(
        samples=[
            JudgeSample(sample_id=0, verdict="pass", insufficient_context=False,
                        reason="ok", usage=reply["usage"]),
            JudgeSample(sample_id=1, verdict="pass", insufficient_context=False,
                        reason="ok", usage=reply["usage"]),
        ],
        majority_verdict="pass", wellposed_votes=2, flag_votes=0,
        ic_votes=0, error_votes=0, provider="openai", model="gpt-5.5",
    )
    totals = outcome.usage_total()
    assert totals["reasoning_tokens"] == 1700
    assert totals["output_tokens"] == 1800
    assert totals["samples_with_usage"] == 2


def test_judge_wellposed_dispatches_to_openai(monkeypatch):
    """End-to-end: judge_wellposed picks the OpenAI caller when provider=openai."""
    cfg = WellposedConfig(
        judge_provider="openai",
        openai_api_key="sk-test",
        judge_samples=3,
        judge_uphold=2,
    )
    replies = iter([
        '{"verdict": "pass", "insufficient_context": false, "reason": "ok"}',
        '{"verdict": "pass", "insufficient_context": false, "reason": "ok"}',
        '{"verdict": "flag", "insufficient_context": false, "reason": "?"}',
    ])
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _ok(_openai_payload(next(replies))),
    )
    out = judge_wellposed("Using Theorem 3.2, deduce A.", cfg)
    assert out.provider == "openai"
    assert out.model == "gpt-4o-mini"
    assert out.majority_verdict == "pass"


def test_check_record_via_openai(monkeypatch):
    cfg = WellposedConfig(
        enable_judge=True,
        judge_provider="openai",
        openai_api_key="sk-test",
        judge_samples=3,
        judge_uphold=2,
    )
    rec = normalise_record({
        "source": "rm",
        "provenance": "extracted",
        "statement": "Using Theorem 3.2 from the previous section, deduce A.",
    }, rid=0)
    replies = iter([
        '{"verdict": "flag", "insufficient_context": true, "reason": "missing"}',
        '{"verdict": "flag", "insufficient_context": true, "reason": "missing"}',
        '{"verdict": "pass", "insufficient_context": false, "reason": "?"}',
    ])
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _ok(_openai_payload(next(replies))),
    )
    result = check_record(rec, cfg)
    assert result["tier"] == "judge"
    assert result["wellposed_status"] == "insufficient_context"
    assert result["judge"]["provider"] == "openai"
    assert result["judge"]["model"] == "gpt-4o-mini"

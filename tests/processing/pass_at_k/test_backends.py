"""Backend tests: payload shape, kill switch, key loading. No network,
no real SDKs — ``requests`` / ``anthropic`` / ``openai`` are faked via
``monkeypatch.setitem(sys.modules, ...)`` (the test_pacing.py pattern).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from icepick.processing.pass_at_k.backends import build_backend
from icepick.processing.pass_at_k.backends.anthropic import AnthropicBackend
from icepick.processing.pass_at_k.backends.openai import OpenAIBackend
from icepick.processing.pass_at_k.backends.qwen_http import QwenHttpBackend
from icepick.processing.pass_at_k.config import SYSTEM_PROMPT, PassAtKConfig

URL = "http://127.0.0.1:1234/v1/chat/completions"


# --- fake requests (qwen_http) ------------------------------------------------


class _Resp:
    def __init__(self, status=200, content="\\boxed{42}", usage=None):
        self.status_code = status
        self._payload = {"choices": [{"message": {"content": content}}]}
        if usage is not None:
            self._payload["usage"] = usage

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _FakeRequests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


class _FakeRequests(types.SimpleNamespace):
    """Just enough of the requests API for QwenHttpBackend.call."""

    class RequestException(Exception):
        pass

    class HTTPError(RequestException):
        pass

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture
def fake_requests(monkeypatch):
    def _install(responses):
        fake = _FakeRequests(responses)
        monkeypatch.setitem(sys.modules, "requests", fake)
        return fake

    return _install


def test_qwen_payload_carries_system_prompt_and_knobs(fake_requests):
    fake = fake_requests([_Resp()])
    backend = QwenHttpBackend(url=URL, model="qwen/qwen3-8b")
    out = backend.call(
        "What is 6*7?", k=1, temperature=0.7, max_tokens=512, think=True, timeout=180.0
    )
    assert out == ["\\boxed{42}"]
    call = fake.calls[0]
    assert call["url"] == URL
    assert call["timeout"] == 180.0
    payload = call["json"]
    assert payload["model"] == "qwen/qwen3-8b"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 512
    assert payload["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    # think=True sends the bare question — no /no_think suffix.
    assert payload["messages"][1] == {"role": "user", "content": "What is 6*7?"}


def test_qwen_no_think_suffix_when_think_false(fake_requests):
    fake = fake_requests([_Resp()])
    backend = QwenHttpBackend(url=URL, model="qwen/qwen3-8b")
    backend.call("What is 6*7?", k=1, temperature=0.0, max_tokens=64, think=False, timeout=5.0)
    user = fake.calls[0]["json"]["messages"][1]["content"]
    assert user == "What is 6*7? /no_think"


def test_qwen_k3_makes_three_posts_and_returns_three_outputs(fake_requests):
    fake = fake_requests([_Resp(content="a"), _Resp(content="b"), _Resp(content="c")])
    backend = QwenHttpBackend(url=URL, model="m")
    out = backend.call("q", k=3, temperature=0.7, max_tokens=64, think=False, timeout=5.0)
    assert out == ["a", "b", "c"]
    assert len(fake.calls) == 3


def test_qwen_http_error_raises_no_partial_list(fake_requests):
    fake = fake_requests([_Resp(content="ok"), _Resp(status=500)])
    backend = QwenHttpBackend(url=URL, model="m")
    with pytest.raises(_FakeRequests.HTTPError):
        backend.call("q", k=2, temperature=0.7, max_tokens=64, think=False, timeout=5.0)
    assert len(fake.calls) == 2  # failed on the second sample, aborted the call


def test_qwen_usage_accumulates_and_tolerates_missing_usage(fake_requests):
    fake_requests([
        _Resp(usage={"prompt_tokens": 10, "completion_tokens": 5}),
        _Resp(),  # no usage block: some local servers omit it
        _Resp(usage={"prompt_tokens": 3, "completion_tokens": 2}),
    ])
    backend = QwenHttpBackend(url=URL, model="m")
    backend.call("q", k=2, temperature=0.7, max_tokens=64, think=False, timeout=5.0)
    backend.call("q2", k=1, temperature=0.7, max_tokens=64, think=False, timeout=5.0)
    assert backend.usage() == {"input_tokens": 13, "output_tokens": 7}


# --- fake SDKs (anthropic / openai) -------------------------------------------


def _fake_anthropic_module(responses):
    calls = []

    class _Client:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = types.SimpleNamespace(
                create=lambda **kwargs: (calls.append(kwargs), responses.pop(0))[1]
            )

    return types.SimpleNamespace(Anthropic=_Client, calls=calls)


def _anthropic_response(blocks, input_tokens=10, output_tokens=5):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type=t, text=x) for t, x in blocks],
        usage=types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_anthropic_call_joins_text_blocks_and_accumulates_usage(monkeypatch):
    fake = _fake_anthropic_module([
        _anthropic_response([("text", "\\boxed"), ("text", "{9}")]),
        _anthropic_response([("text", "\\boxed{8}")], input_tokens=7, output_tokens=3),
    ])
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    backend = AnthropicBackend(model="claude-haiku-4-5", api_key="sk-test")
    out = backend.call("q", k=2, temperature=0.7, max_tokens=1024, think=False, timeout=30.0)
    assert out == ["\\boxed{9}", "\\boxed{8}"]
    assert len(fake.calls) == 2
    kwargs = fake.calls[0]
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 1024
    assert kwargs["messages"] == [{"role": "user", "content": "q"}]
    assert backend.usage() == {"input_tokens": 17, "output_tokens": 8}


def _fake_openai_module(responses):
    calls = []

    class _OpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kwargs: (calls.append(kwargs), responses.pop(0))[1]
                )
            )

    return types.SimpleNamespace(OpenAI=_OpenAI, calls=calls)


def _openai_response(content, prompt_tokens=11, completion_tokens=4):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
        usage=types.SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def test_openai_reasoning_models_omit_temperature(monkeypatch):
    fake = _fake_openai_module([_openai_response("\\boxed{1}")])
    monkeypatch.setitem(sys.modules, "openai", fake)
    backend = OpenAIBackend(model="o3-mini", api_key="sk-test")
    out = backend.call("q", k=1, temperature=0.7, max_tokens=256, think=False, timeout=30.0)
    assert out == ["\\boxed{1}"]
    kwargs = fake.calls[0]
    assert "temperature" not in kwargs  # o1/o3/o4 family rejects it
    assert kwargs["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert kwargs["messages"][1] == {"role": "user", "content": "q"}
    assert backend.usage() == {"input_tokens": 11, "output_tokens": 4}


def test_openai_non_reasoning_models_forward_temperature(monkeypatch):
    fake = _fake_openai_module([_openai_response("x"), _openai_response("y")])
    monkeypatch.setitem(sys.modules, "openai", fake)
    backend = OpenAIBackend(model="gpt-4.1-mini", api_key="sk-test")
    out = backend.call("q", k=2, temperature=0.3, max_tokens=256, think=False, timeout=30.0)
    assert out == ["x", "y"]
    assert all(c["temperature"] == 0.3 for c in fake.calls)
    assert backend.usage() == {"input_tokens": 22, "output_tokens": 8}


# --- build_backend ------------------------------------------------------------


def _clean_env(monkeypatch, var):
    """Guarantee ``var`` is absent AND restored after the test, even though
    _load_env_file writes os.environ directly (setenv-then-delenv makes
    monkeypatch record the original state before the loader mutates it)."""
    monkeypatch.setenv(var, "sentinel")
    monkeypatch.delenv(var)


def test_kill_switch_off_uses_placeholder_and_never_reads_key_files(monkeypatch, tmp_path):
    for backend_name, key_file_field, var, model in (
        ("anthropic", "anthropic_key_file", "ANTHROPIC_API_KEY", "claude-haiku-4-5"),
        ("openai", "openai_key_file", "OPENAI_API_KEY", "o3-mini"),
    ):
        _clean_env(monkeypatch, var)
        # Key file path does NOT exist — must not be read, must not raise.
        # ``model`` is required for paid backends (policy default is qwen_http).
        cfg = PassAtKConfig(
            backend=backend_name,
            model=model,
            allow_live_calls=False,
            **{key_file_field: tmp_path / "does-not-exist.env"},
        )
        backend = build_backend(cfg)
        assert backend.api_key == "[API key]"
        assert backend.name == backend_name


def test_kill_switch_on_loads_real_key_from_file(monkeypatch, tmp_path):
    _clean_env(monkeypatch, "ANTHROPIC_API_KEY")
    key_file = tmp_path / "keys.env"
    key_file.write_text("# comment\nANTHROPIC_API_KEY=sk-real-123\n", encoding="utf-8")
    cfg = PassAtKConfig(
        backend="anthropic", model="claude-haiku-4-5",
        allow_live_calls=True, anthropic_key_file=key_file,
    )
    backend = build_backend(cfg)
    assert backend.api_key == "sk-real-123"


def test_kill_switch_on_without_any_key_raises(monkeypatch, tmp_path):
    _clean_env(monkeypatch, "OPENAI_API_KEY")
    key_file = tmp_path / "keys.env"
    key_file.write_text("SOMETHING_ELSE=nope\n", encoding="utf-8")
    cfg = PassAtKConfig(
        backend="openai", model="o3-mini",
        allow_live_calls=True, openai_key_file=key_file,
    )
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_backend(cfg)


def test_build_backend_qwen_http_gets_url_and_default_model():
    cfg = PassAtKConfig(backend="qwen_http", backend_url=URL)
    backend = build_backend(cfg)
    assert isinstance(backend, QwenHttpBackend)
    assert backend.url == URL
    assert backend.model == "qwen/qwen3-8b"  # DEFAULT_MODELS fallback via resolved_model


def test_build_backend_flow_testing_raises():
    cfg = PassAtKConfig(mode="flow_testing", calibration_sheet=Path("sheet.jsonl"))
    with pytest.raises(RuntimeError, match="calibration sheet"):
        build_backend(cfg)

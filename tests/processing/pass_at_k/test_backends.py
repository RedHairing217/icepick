"""Backend tests: payload shape, kill switch, key loading. No network,
no real SDKs — ``requests`` / ``anthropic`` / ``openai`` are faked via
``monkeypatch.setitem(sys.modules, ...)`` (the test_pacing.py pattern).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from icepick.processing.pass_at_k.backends import _read_raw_or_env_key, build_backend
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

    def post(self, url, json=None, timeout=None, **kwargs):
        # **kwargs (rather than a declared headers=None) means "headers"
        # only shows up in the recorded call when the caller actually
        # passed it — lets tests assert its total absence, not just None.
        call = {"url": url, "json": json, "timeout": timeout, **kwargs}
        self.calls.append(call)
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


def test_qwen_sends_bearer_header_when_api_key_set(fake_requests):
    """Remote-gateway path: a configured api_key rides as a bearer header
    on every rollout request."""
    fake = fake_requests([_Resp(), _Resp()])
    backend = QwenHttpBackend(url=URL, model="m", api_key="tok-secret-123")
    backend.call("q", k=2, temperature=0.7, max_tokens=64, think=False, timeout=5.0)
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["headers"] == {"Authorization": "Bearer tok-secret-123"}


def test_qwen_sends_no_auth_header_when_api_key_is_none(fake_requests):
    """Backward-compat: the default (no api_key) call is byte-for-byte the
    pre-auth shape — no headers kwarg reaches requests.post at all, so a
    local/keyless endpoint sees zero behavior change."""
    fake = fake_requests([_Resp()])
    backend = QwenHttpBackend(url=URL, model="m")  # api_key defaults to None
    backend.call("q", k=1, temperature=0.7, max_tokens=64, think=False, timeout=5.0)
    assert "headers" not in fake.calls[0]


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
    assert all(c["max_tokens"] == 256 for c in fake.calls)
    assert all("max_completion_tokens" not in c for c in fake.calls)
    assert backend.usage() == {"input_tokens": 22, "output_tokens": 8}


def test_openai_reasoning_models_use_max_completion_tokens(monkeypatch):
    """gpt-5.x / o-series reject both temperature and the legacy max_tokens
    parameter; the cap must travel as max_completion_tokens instead."""
    for model in ("gpt-5.5", "o3-mini"):
        fake = _fake_openai_module([_openai_response("\\boxed{1}")])
        monkeypatch.setitem(sys.modules, "openai", fake)
        backend = OpenAIBackend(model=model, api_key="sk-test")
        out = backend.call("q", k=1, temperature=0.7, max_tokens=256, think=False, timeout=30.0)
        assert out == ["\\boxed{1}"]
        kwargs = fake.calls[0]
        assert kwargs["max_completion_tokens"] == 256, model
        assert "max_tokens" not in kwargs, model
        assert "temperature" not in kwargs, model


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


# --- qwen_http optional bearer auth (remote gateway) --------------------------


def test_passatk_config_qwen_key_file_defaults_to_none():
    """Local endpoints never set this — the field must default to None so
    existing configs/CLIs that omit it are unaffected."""
    cfg = PassAtKConfig()
    assert cfg.qwen_key_file is None


def test_build_backend_qwen_http_api_key_none_when_no_key_file():
    cfg = PassAtKConfig(backend="qwen_http", backend_url=URL)
    backend = build_backend(cfg)
    assert backend.api_key is None


def test_build_backend_qwen_http_wires_key_file_to_api_key(tmp_path):
    key_file = tmp_path / "tangerine_api.env"
    key_file.write_text("sk-remote-tangerine-999", encoding="utf-8")
    cfg = PassAtKConfig(backend="qwen_http", backend_url=URL, qwen_key_file=key_file)
    backend = build_backend(cfg)
    assert isinstance(backend, QwenHttpBackend)
    assert backend.api_key == "sk-remote-tangerine-999"


def test_build_backend_qwen_http_key_file_not_gated_by_allow_live_calls(tmp_path):
    """The qwen key is an AUTH requirement, not a spend risk: unlike the
    paid backends (see test_kill_switch_off_uses_placeholder_and_never_reads_key_files),
    it must resolve to the real token even with allow_live_calls left at
    its False default — kill switch #1 does not apply to qwen_http."""
    key_file = tmp_path / "tangerine_api.env"
    key_file.write_text("sk-remote-tangerine-999", encoding="utf-8")
    cfg = PassAtKConfig(
        backend="qwen_http",
        backend_url=URL,
        qwen_key_file=key_file,
        allow_live_calls=False,
    )
    backend = build_backend(cfg)
    assert backend.api_key == "sk-remote-tangerine-999"  # not the "[API key]" placeholder


# --- _read_raw_or_env_key: raw-token or dotenv key file reader ----------------


def test_read_raw_or_env_key_bare_token_no_trailing_newline(tmp_path):
    """Mimics the real tangerine_api.env: a single raw token, no VAR=
    prefix, no trailing newline."""
    key_file = tmp_path / "tangerine_api.env"
    key_file.write_bytes(b"sk-tangerine-raw-token-xyz")
    assert _read_raw_or_env_key(key_file) == "sk-tangerine-raw-token-xyz"


def test_read_raw_or_env_key_dotenv_style_returns_value(tmp_path):
    key_file = tmp_path / "keys.env"
    key_file.write_text("# comment\nQWEN_API_KEY=sk-qwen-abc\n", encoding="utf-8")
    assert _read_raw_or_env_key(key_file) == "sk-qwen-abc"


def test_read_raw_or_env_key_dotenv_prefers_tangerine_var_over_others(tmp_path):
    key_file = tmp_path / "keys.env"
    key_file.write_text(
        "SOME_OTHER_VAR=nope\nTANGERINE_API_KEY=sk-tangerine-456\n", encoding="utf-8"
    )
    assert _read_raw_or_env_key(key_file) == "sk-tangerine-456"


def test_read_raw_or_env_key_dotenv_prefers_qwen_var_over_first_key(tmp_path):
    key_file = tmp_path / "keys.env"
    key_file.write_text(
        "FIRST_VAR=ignored\nQWEN_API_KEY=sk-qwen-preferred\n", encoding="utf-8"
    )
    assert _read_raw_or_env_key(key_file) == "sk-qwen-preferred"


def test_read_raw_or_env_key_dotenv_falls_back_to_first_key(tmp_path):
    """No QWEN_API_KEY/TANGERINE_API_KEY var present: the first KEY=VALUE
    line in the file wins."""
    key_file = tmp_path / "keys.env"
    key_file.write_text("RANDOM_VAR=sk-first-789\nANOTHER=ignored\n", encoding="utf-8")
    assert _read_raw_or_env_key(key_file) == "sk-first-789"


def test_read_raw_or_env_key_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        _read_raw_or_env_key(tmp_path / "does-not-exist.env")


def test_read_raw_or_env_key_empty_file_raises(tmp_path):
    key_file = tmp_path / "empty.env"
    key_file.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        _read_raw_or_env_key(key_file)


def test_read_raw_or_env_key_whitespace_only_file_raises(tmp_path):
    key_file = tmp_path / "whitespace.env"
    key_file.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        _read_raw_or_env_key(key_file)

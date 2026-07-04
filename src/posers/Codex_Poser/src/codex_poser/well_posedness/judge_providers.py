"""API judge providers for semantic well-posedness residue."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

JudgeProvider = Literal["anthropic", "openai"]

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TOKENS = 512
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_KEY_ENV = {
    "anthropic": Path("../anthro_key.env"),
    "openai": Path("../openai_key.env"),
}


@dataclass(frozen=True)
class JudgeConfig:
    provider: JudgeProvider
    api_key: str
    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    api_url: str = ""
    timeout_seconds: float = 60.0


class JudgeCache:
    """JSONL reply cache keyed by provider, model, and exact prompt."""

    def __init__(self, path: Path | None, provider: JudgeProvider, model: str):
        self.path = path
        self.provider = provider
        self.model = model
        self.store: dict[str, str] = {}
        self.dirty = False
        if path and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_provider = row.get("provider", "anthropic")
                    if (
                        row_provider == provider
                        and row.get("model") == model
                        and row.get("key")
                        and "reply" in row
                    ):
                        self.store[str(row["key"])] = str(row["reply"])

    def get(self, prompt: str) -> str | None:
        return self.store.get(self._key(prompt))

    def put(self, prompt: str, reply: str) -> None:
        self.store[self._key(prompt)] = reply
        self.dirty = True

    def save(self) -> None:
        if not self.path or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for key, reply in sorted(self.store.items()):
                handle.write(
                    json.dumps(
                        {
                            "provider": self.provider,
                            "model": self.model,
                            "key": key,
                            "reply": reply,
                        }
                    )
                    + "\n"
                )
        self.dirty = False

    def _key(self, prompt: str) -> str:
        identity = f"{self.provider}:{self.model}"
        return hashlib.sha256(f"{identity}\x00{prompt}".encode("utf-8")).hexdigest()


class AnthropicJudge:
    """Small stdlib-only client for Anthropic Messages API."""

    def __init__(self, config: JudgeConfig):
        self.config = config

    def __call__(self, prompt: str) -> tuple[str, dict]:
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = _post_json(
            url=self.config.api_url or ANTHROPIC_API_URL,
            payload=payload,
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            timeout_seconds=self.config.timeout_seconds,
            provider="Anthropic",
        )
        return _anthropic_text(data), _anthropic_usage(data)


class OpenAIJudge:
    """Small stdlib-only client for OpenAI Responses API."""

    def __init__(self, config: JudgeConfig):
        self.config = config

    def __call__(self, prompt: str) -> tuple[str, dict]:
        payload = {
            "model": self.config.model,
            "input": prompt,
            "temperature": 0.2,
            "max_output_tokens": self.config.max_tokens,
        }
        data = _post_json(
            url=self.config.api_url or OPENAI_API_URL,
            payload=payload,
            headers={"authorization": f"Bearer {self.config.api_key}"},
            timeout_seconds=self.config.timeout_seconds,
            provider="OpenAI",
        )
        return _openai_text(data), _openai_usage(data)


def make_cached_judge(
    provider: JudgeProvider,
    key_env_path: Path,
    cache_path: Path | None,
    model_override: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: float = 60.0,
) -> tuple[Callable[[str], str], JudgeCache, str]:
    config = load_judge_config(
        provider=provider,
        key_env_path=key_env_path,
        model_override=model_override,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    raw_judge: Callable[[str], str]
    if provider == "anthropic":
        raw_judge = AnthropicJudge(config)
    elif provider == "openai":
        raw_judge = OpenAIJudge(config)
    else:
        raise ValueError(f"unsupported judge provider: {provider}")

    cache = JudgeCache(cache_path, provider=provider, model=config.model)

    def judge(prompt: str) -> tuple[str, dict]:
        """Returns (text, usage). Cache hits contribute empty usage."""
        cached = cache.get(prompt)
        if cached is not None:
            return cached, {}
        reply, usage = raw_judge(prompt)
        cache.put(prompt, reply)
        return reply, usage

    return judge, cache, config.model


def default_key_env_path(provider: JudgeProvider) -> Path:
    try:
        return DEFAULT_KEY_ENV[provider]
    except KeyError as exception:
        raise ValueError(f"unsupported judge provider: {provider}") from exception


def load_judge_config(
    provider: JudgeProvider,
    key_env_path: Path,
    model_override: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: float = 60.0,
) -> JudgeConfig:
    values = load_key_env(key_env_path)
    if provider == "anthropic":
        api_key = values.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(f"ANTHROPIC_API_KEY not found in {key_env_path}")
        model = (
            model_override
            or values.get("ANTHROPIC_MODEL")
            or os.getenv("ANTHROPIC_MODEL")
            or DEFAULT_ANTHROPIC_MODEL
        )
        return JudgeConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            api_url=ANTHROPIC_API_URL,
            timeout_seconds=timeout_seconds,
        )
    if provider == "openai":
        api_key = values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY not found in {key_env_path}")
        model = (
            model_override
            or values.get("OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_OPENAI_MODEL
        )
        return JudgeConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            api_url=OPENAI_API_URL,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"unsupported judge provider: {provider}")


def load_key_env(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].strip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key:
                values[key] = value
    return values


def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout_seconds: float,
    provider: str,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "content-type": "application/json",
            **headers,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exception:
        detail = exception.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider} judge HTTP {exception.code}: {detail}") from exception
    except urllib.error.URLError as exception:
        raise RuntimeError(f"{provider} judge unavailable: {exception.reason}") from exception


def _anthropic_text(data: dict) -> str:
    blocks = data.get("content")
    if not isinstance(blocks, list):
        return ""
    text = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text.append(str(block.get("text", "")))
    return "".join(text)


def _openai_text(data: dict) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    text = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text.append(str(content.get("text", "")))
    return "".join(text)


def _anthropic_usage(data: dict) -> dict:
    """Normalized usage dict from an Anthropic Messages API response."""
    usage = data.get("usage") or {}
    out = {}
    for field in ("input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens"):
        if field in usage:
            out[field] = usage[field]
    return out


def _openai_usage(data: dict) -> dict:
    """Normalized usage dict from an OpenAI Responses API payload.

    OpenAI's Responses API uses ``input_tokens``/``output_tokens`` natively;
    older completions APIs would use ``prompt_tokens``/``completion_tokens``.
    Handle both for safety.
    """
    usage = data.get("usage") or {}
    out = {}
    out["input_tokens"] = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    out["output_tokens"] = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    return {k: v for k, v in out.items() if v}

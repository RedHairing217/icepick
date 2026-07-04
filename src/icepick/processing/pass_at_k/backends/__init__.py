"""Subject-model backends for the pass@k stage.

One class per backend, each implementing the ``ModelBackend`` protocol
from ``base.py``. ``build_backend`` is the single construction point the
runner uses; it owns key resolution so the backend classes stay dumb
transports that are trivial to fake in tests.

KILL SWITCH (mirrors groundtruth's ``_build_anthropic_client``): the
paid backends (anthropic, openai) only receive a real API key when
``cfg.allow_live_calls`` is True. Otherwise the client is constructed
with the placeholder literal ``"[API key]"`` so any accidental
invocation returns 401 from the provider without spending money — and
the key files are never even read. ``qwen_http`` targets a local,
per-call-free endpoint and is exempt.

``_load_env_file`` is copied from the groundtruth adapter rather than
imported: per project direction each stage's secret handling stays
self-contained so a stage can be lifted out without dragging siblings.
"""

from __future__ import annotations

import os

from icepick.config import ConfigError
from icepick.processing.pass_at_k.config import (
    BACKEND_ANTHROPIC,
    BACKEND_OPENAI,
    BACKEND_QWEN_HTTP,
    PassAtKConfig,
)

# The placeholder that makes an un-opted-in paid backend fail with 401
# instead of billing. Same literal as groundtruth's kill switch.
_PLACEHOLDER_KEY = "[API key]"


def build_backend(cfg: PassAtKConfig):
    """Construct the ModelBackend for ``cfg``.

    flow_testing never builds one — that mode replays a calibration
    sheet and must stay runnable with zero SDKs and zero network.
    """
    if cfg.mode == "flow_testing":
        raise RuntimeError(
            "flow_testing replays a calibration sheet; no backend is built"
        )
    if cfg.backend == BACKEND_QWEN_HTTP:
        from icepick.processing.pass_at_k.backends.qwen_http import QwenHttpBackend

        return QwenHttpBackend(url=cfg.backend_url, model=cfg.resolved_model)
    if cfg.backend == BACKEND_ANTHROPIC:
        from icepick.processing.pass_at_k.backends.anthropic import AnthropicBackend

        api_key = _resolve_api_key(
            cfg, env_var="ANTHROPIC_API_KEY", key_file=cfg.anthropic_key_file
        )
        return AnthropicBackend(model=cfg.resolved_model, api_key=api_key)
    if cfg.backend == BACKEND_OPENAI:
        from icepick.processing.pass_at_k.backends.openai import OpenAIBackend

        api_key = _resolve_api_key(
            cfg, env_var="OPENAI_API_KEY", key_file=cfg.openai_key_file
        )
        return OpenAIBackend(model=cfg.resolved_model, api_key=api_key)
    raise ConfigError(f"unknown pass_at_k backend: {cfg.backend!r}")


def _resolve_api_key(cfg: PassAtKConfig, *, env_var: str, key_file) -> str:
    """KILL SWITCH: only load a real key behind ``allow_live_calls``.

    Without the flag the placeholder goes into the client and the key
    file is never read (it need not even exist) — any accidental call
    401s without spending. With the flag, load the key file into the
    environment (existing env always wins) and demand the variable.
    """
    if not cfg.allow_live_calls:
        return _PLACEHOLDER_KEY
    if not os.environ.get(env_var) and key_file:
        _load_env_file(key_file)
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"{env_var} not set and could not be loaded from the key file"
        )
    return key


def _load_env_file(path) -> None:
    """Minimal KEY=VALUE loader. Lines starting with # are ignored."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"key file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Don't overwrite anything already set in the environment.
        os.environ.setdefault(key, value)

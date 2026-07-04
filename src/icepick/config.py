"""Run-mode validation and host-role guard.

Processor mode is required on every call-bearing command. A run fails if
the mode is omitted. ``flow_testing`` runs additionally fail if the
calibration sheet is missing or incomplete.

Host-role validation lives here so any future caller — sampling code, an
agent controller — reads the same enforcement code. The function takes
a ``HostsConfig`` and either returns silently or raises ``ConfigError``.

The per-stage knobs (judge_samples, judge_uphold, comparison_policy,
etc.) live on the stage configs (``GroundtruthConfig``,
``WellposedConfig``) — not here. Stage-specific knobs belong with the
stage that consumes them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The proxy variable: an env var holding the PATH to a provider key file
# (e.g. anthro_key.env). Callers set this instead of exporting the raw key,
# so the secret stays in a gitignored file and never enters the repo, a
# command line, or a log.
ANTHROPIC_KEY_FILE_ENV = "ANTHROPIC_KEY_FILE"


class ConfigError(ValueError):
    """Raised when a run config violates a hard invariant."""


def load_env_file(path) -> dict:
    """Parse a ``KEY=VALUE`` env file (e.g. anthro_key.env) into a dict.

    Blank lines and ``#`` comments are ignored; surrounding quotes are
    stripped. Raises ``FileNotFoundError`` if the path does not exist.
    """
    out: dict = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def resolve_anthropic_credentials(env: Optional[dict] = None) -> tuple:
    """Return ``(api_key, model)`` without ever placing the key in the repo.

    Resolution order:
      1. ``ANTHROPIC_API_KEY`` already in the environment.
      2. otherwise the file that ``ANTHROPIC_KEY_FILE`` points at — the
         proxy variable holds a path, not the secret.

    ``model`` comes from ``ANTHROPIC_MODEL`` (env or key file), or ``None``.
    Raises ``ConfigError`` when no key can be resolved.
    """
    env = env if env is not None else os.environ
    api_key = env.get("ANTHROPIC_API_KEY")
    model = env.get("ANTHROPIC_MODEL")
    if not api_key:
        key_file = env.get(ANTHROPIC_KEY_FILE_ENV)
        if key_file:
            parsed = load_env_file(key_file)
            api_key = parsed.get("ANTHROPIC_API_KEY")
            model = model or parsed.get("ANTHROPIC_MODEL")
    if not api_key:
        raise ConfigError(
            "no Anthropic key: set ANTHROPIC_API_KEY, or point "
            f"{ANTHROPIC_KEY_FILE_ENV} at an anthro_key.env file (the key stays out of the repo)"
        )
    return api_key, model


@dataclass
class HostConfig:
    role: str
    base_url: str
    model: str
    log_path: str
    cache_path: str


@dataclass
class HostsConfig:
    subject: Optional[HostConfig] = None
    manager: Optional[HostConfig] = None


@dataclass
class RunConfig:
    processor_mode: str
    calibration_sheet: Optional[str] = None
    hosts: HostsConfig = field(default_factory=HostsConfig)


def validate_mode(processor_mode: str, calibration_sheet: Optional[str]) -> None:
    """Mode must be explicit. flow_testing requires a calibration sheet."""
    if processor_mode is None or processor_mode == "":
        raise ConfigError("processor_mode is required (production | flow_testing)")
    if processor_mode not in ("production", "flow_testing"):
        raise ConfigError(
            f"processor_mode must be 'production' or 'flow_testing', got {processor_mode!r}"
        )
    if processor_mode == "flow_testing" and not calibration_sheet:
        raise ConfigError(
            "flow_testing mode requires --calibration-sheet"
        )


def validate_host_roles(hosts: HostsConfig) -> None:
    """Subject and manager hosts must be programmatically separable.

    Both must be present, and they must not share a base URL, a model id
    plus base URL pair, a log path, or a cache path. This is the only
    place that decision is made — every call site should reach it.
    """
    if hosts.subject is None or hosts.manager is None:
        raise ConfigError(
            "host roles 'subject' and 'manager' must both be configured"
        )
    s, m = hosts.subject, hosts.manager
    if s.base_url == m.base_url:
        raise ConfigError(
            "subject and manager hosts must not share base_url "
            f"(got {s.base_url!r} for both)"
        )
    if s.model == m.model and s.base_url == m.base_url:
        raise ConfigError("subject and manager must not share both model id and base_url")
    if s.log_path == m.log_path:
        raise ConfigError("subject and manager hosts must not share log_path")
    if s.cache_path == m.cache_path:
        raise ConfigError("subject and manager hosts must not share cache_path")

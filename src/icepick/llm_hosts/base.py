"""Shared LLM host contract and role-separated wrappers.

The two roles are different *types*, not the same type with different
config. ``SubjectLLMHost`` and ``ManagerLLMHost`` are wrappers that own
their respective config and never share underlying clients, logs, caches,
or budgets. This makes misuse a type error rather than a convention
violation.

Sampling code accepts only ``SubjectLLMHost``. The agent controller
accepts only ``ManagerLLMHost``. The wiring layer (``config.validate_host_roles``)
guarantees both are present and that they don't collide on base URL,
model id, log path, or cache path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

ROLE_SUBJECT = "subject"
ROLE_MANAGER = "manager"


@dataclass
class CompletionRequest:
    system: Optional[str]
    user: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 60.0
    no_think: bool = False


@dataclass
class CompletionResult:
    text: str
    finish_reason: str
    latency_s: float
    model: str
    base_url: str
    error: Optional[str] = None


class LLMHost(Protocol):
    role: str

    def health(self) -> dict: ...
    def complete(self, request: CompletionRequest) -> CompletionResult: ...


class _RoleHost:
    """Concrete role wrapper. Concrete clients (LM Studio) plug in here."""

    role: str = ""

    def __init__(self, base_url: str, model: str, log_path: str, cache_path: str):
        self.base_url = base_url
        self.model = model
        self.log_path = log_path
        self.cache_path = cache_path

    def health(self) -> dict:
        raise NotImplementedError("LLMHost.health is not yet implemented")

    def complete(self, request: CompletionRequest) -> CompletionResult:
        raise NotImplementedError("LLMHost.complete is not yet implemented")


class SubjectLLMHost(_RoleHost):
    """Subject under test. Pass@k and confirmation only."""

    role = ROLE_SUBJECT


class ManagerLLMHost(_RoleHost):
    """Manager / chat controller. Agent only."""

    role = ROLE_MANAGER

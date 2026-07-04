"""The agent controller must refuse a subject host by type."""

from __future__ import annotations

import pytest

from icepick.agent.controller import AgentController
from icepick.agent.registry import ALLOWED_ACTIONS, ToolNotAllowed, ToolRegistry
from icepick.llm_hosts.base import ManagerLLMHost, SubjectLLMHost


def _subject():
    return SubjectLLMHost("http://a:1/v1", "qwen3-8b", "s.log", "s.cache")


def _manager():
    return ManagerLLMHost("http://b:2/v1", "llama", "m.log", "m.cache")


def test_subject_host_rejected_by_controller():
    with pytest.raises(TypeError):
        AgentController(_subject())  # type: ignore[arg-type]


def test_manager_host_accepted():
    AgentController(_manager())


def test_registry_rejects_unknown_actions():
    reg = ToolRegistry()
    with pytest.raises(ToolNotAllowed):
        reg.register("shell.exec", lambda **kw: None)


def test_registry_dispatch_requires_allowlisted_action():
    reg = ToolRegistry()
    sample = next(iter(ALLOWED_ACTIONS))
    reg.register(sample, lambda **kw: "ok")
    assert reg.dispatch(sample, {}) == "ok"
    with pytest.raises(ToolNotAllowed):
        reg.dispatch("shell.exec", {})

"""Agent controller — validates and dispatches manager-model actions.

Owns session state, allowed actions, and execution flow. Receives a
``ManagerLLMHost`` only; refuses to accept a subject host.

The manager model proposes structured actions; the controller validates
each one against the allowlist and the per-action argument schema before
dispatching to a registered tool. Approval-gated operations still require
human approval through ``ApprovalGate``.
"""

from __future__ import annotations

from icepick.llm_hosts.base import ManagerLLMHost


class AgentController:
    def __init__(self, manager_host: ManagerLLMHost):
        if not isinstance(manager_host, ManagerLLMHost):
            raise TypeError(
                "AgentController requires a ManagerLLMHost; got "
                f"{type(manager_host).__name__}"
            )
        self.manager = manager_host

    def step(self, user_message: str) -> dict:
        raise NotImplementedError("AgentController.step is not yet implemented")

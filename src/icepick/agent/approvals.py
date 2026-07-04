"""Approval gate.

Blocks acquisition, generation, scraping, confirmation calls, threshold
changes, and writes that require approval. The controller may propose an
action; this gate decides whether human approval is needed before
``ToolRegistry.dispatch`` is allowed to run.
"""

from __future__ import annotations

APPROVAL_REQUIRED_ACTIONS = frozenset(
    {
        "gate.run",
        "confirm.run",
        "handoff.export",
    }
)


class ApprovalRequired(RuntimeError):
    pass


class ApprovalGate:
    def __init__(self):
        self._approved: set = set()

    def approve(self, action: str) -> None:
        self._approved.add(action)

    def check(self, action: str) -> None:
        if action in APPROVAL_REQUIRED_ACTIONS and action not in self._approved:
            raise ApprovalRequired(f"{action} requires explicit approval")

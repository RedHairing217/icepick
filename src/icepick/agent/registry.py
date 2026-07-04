"""Tool registry — maps allowlisted action names to safe Python functions.

A tool registered here must not accept or expose a ``SubjectLLMHost``. The
controller's type-checked wiring is the first defence; this registry is
the second.
"""

from __future__ import annotations

from icepick.contracts.actions import ALLOWED_ACTIONS


class ToolNotAllowed(ValueError):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}

    def register(self, action: str, fn) -> None:
        if action not in ALLOWED_ACTIONS:
            raise ToolNotAllowed(
                f"refusing to register {action!r}; not on the allowlist"
            )
        self._tools[action] = fn

    def dispatch(self, action: str, args: dict):
        if action not in ALLOWED_ACTIONS:
            raise ToolNotAllowed(f"action {action!r} is not allowed")
        if action not in self._tools:
            raise KeyError(f"no tool registered for {action!r}")
        return self._tools[action](**args)

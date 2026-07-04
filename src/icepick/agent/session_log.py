"""Session log.

Records user prompts, manager-model replies, proposed actions, approvals,
executed actions, results, errors, and output paths. One JSONL file per
session.
"""

from __future__ import annotations


def append_event(log_path, event: dict) -> None:
    raise NotImplementedError("agent.session_log.append_event is not yet implemented")

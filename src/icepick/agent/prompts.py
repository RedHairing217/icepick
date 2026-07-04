"""System prompts for the manager model.

The system prompt describes the pipeline, the current run state, and the
available actions. The model must reply with either plain assistant text
or a single JSON action object.
"""

from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """\
You are the manager assistant for an Icepick processing run.
You may either reply in plain assistant text, or emit one JSON object of the form
{{"action": "<allowed_action>", "args": {{...}}}}.
Allowed actions: {allowed_actions}
Run state: {run_state}
"""


def render(run_state: str, allowed_actions: list) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        allowed_actions=", ".join(allowed_actions),
        run_state=run_state,
    )

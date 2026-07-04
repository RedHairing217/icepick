"""Action requests — the only way the agent controller talks to the rest.

The manager model emits either plain text or a single JSON object matching
``AgentAction``. The controller validates the action name against the
allowlist and the args against the per-action argument schemas registered
in ``icepick.agent.registry``.

Initial allowlist:

- ``intake.plan``
- ``intake.validate_manifest``
- ``intake.show_plan``
- ``gate.run``
- ``routing.summarise``
- ``triage.summarise``
- ``confirm.dry_run``
- ``confirm.run``
- ``buckets.list``
- ``reports.show_summary``
- ``handoff.export``

Blocked, always: arbitrary shell, arbitrary writes, deleting outputs,
scrape/generation without an approved manifest, threshold mutation without
approval, network calls outside configured adapters, direct mutation of
verdicts or buckets.
"""

from __future__ import annotations

from dataclasses import dataclass

ALLOWED_ACTIONS = frozenset(
    {
        "intake.plan",
        "intake.validate_manifest",
        "intake.show_plan",
        "gate.run",
        "routing.summarise",
        "triage.summarise",
        "confirm.dry_run",
        "confirm.run",
        "buckets.list",
        "reports.show_summary",
        "handoff.export",
    }
)


@dataclass(frozen=True)
class AgentAction:
    action: str
    args: dict

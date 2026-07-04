"""Icepick — processing surface with in-house acquisition for ModelBreaker-style records.

Three subsystems live under this package and must remain independently
runnable and independently testable:

- ``icepick.processing``: ingest, checks, routing, triage, confirm,
  escalation, merge-back, reports. Runs without allocation or chat.
- ``icepick.allocation``: intake planning, manifests, adapters, manual
  mounts, handoff. Dry-runs without chat.
- ``icepick.agent``: optional manager-model chat control. Allowlisted
  action dispatch only. Built last.

Cross-subsystem talk happens through ``icepick.contracts`` only — never by
importing another subsystem's private modules.
"""

__version__ = "0.0.1"

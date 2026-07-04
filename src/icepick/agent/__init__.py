"""Agent subsystem (low priority, built last).

Manager-model chat over an allowlisted controller. Proposes structured
actions; ``AgentController`` validates and executes only allowed operations.
Approval-gated operations still require human approval.

Must NOT block processing or allocation work, and must never receive a
SubjectLLMHost reference.
"""

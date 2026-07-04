"""LM Studio host client.

OpenAI-compatible chat completions. Validates server reachability via
``/models`` and confirms the expected model id is loaded before
returning. Fails closed if the host is unavailable before approved
acquisition or confirmation starts.

Defaults (overridable via config):
- subject: http://127.0.0.1:1234/v1, qwen/qwen3-8b
- manager: http://127.0.0.1:1235/v1, a separate local model
"""

from __future__ import annotations

from icepick.llm_hosts.base import ManagerLLMHost, SubjectLLMHost


class LMStudioSubject(SubjectLLMHost):
    pass


class LMStudioManager(ManagerLLMHost):
    pass

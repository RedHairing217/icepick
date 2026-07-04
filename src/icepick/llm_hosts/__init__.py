"""Local LLM host adapters.

Two roles, enforced by separate classes — never by convention alone:

- ``SubjectLLMHost``: the model under test. Used by pass@k and confirmation
  sampling. Never reachable from the agent controller.
- ``ManagerLLMHost``: the chat-controller model. Used by the agent console.
  Never reachable from sampling or confirmation code.

A run fails closed if the two roles share a base URL, a model id, a log
path, or a cache path.
"""

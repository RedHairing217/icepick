"""Anthropic subject-model backend for pass@k rollouts.

The SDK is imported lazily (mirrors the groundtruth adapter) so
flow_testing mode and the qwen_http/openai paths run without the
``anthropic`` package installed. The client is built on first ``call``,
never in the constructor — ``build_backend`` can therefore hand out a
kill-switched instance (placeholder key) without touching the SDK.

``think`` maps to nothing here: the Claude models this stage targets run
without extended thinking, so the flag is accepted (protocol parity)
and deliberately ignored. If extended thinking is ever wanted, wire it
to the ``thinking`` request parameter — do not overload temperature.

Per the ``ModelBackend`` protocol: no retries (runner's job), raise on
transport errors, never return a partial list.
"""

from __future__ import annotations

from icepick.processing.pass_at_k.config import SYSTEM_PROMPT


class AnthropicBackend:
    """k sequential ``messages.create`` calls per question."""

    name = "anthropic"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key  # may be the kill-switch placeholder '[API key]'
        self._client = None
        self._input_tokens = 0
        self._output_tokens = 0

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "the 'anthropic' package is required for the anthropic backend; "
                "install with `pip install icepick[judge]` or add anthropic to your env"
            ) from exc
        self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def call(
        self,
        question: str,
        *,
        k: int,
        temperature: float,
        max_tokens: int,
        think: bool,  # accepted for protocol parity; see module docstring
        timeout: float,
    ) -> list:
        client = self._ensure_client()
        outputs = []
        for _ in range(k):
            response = client.messages.create(
                model=self.model,
                system=SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": question}],
                timeout=timeout,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self._output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            # Join text blocks; non-text blocks (none expected without
            # tools/thinking) are skipped rather than crashing the rollout.
            outputs.append(
                "".join(
                    getattr(block, "text", "")
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                )
            )
        return outputs

    def usage(self) -> dict:
        """Tokens accumulated across every call this instance has made."""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        }

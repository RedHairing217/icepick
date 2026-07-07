"""OpenAI subject-model backend for pass@k rollouts.

The SDK is imported lazily (mirrors the groundtruth adapter's pattern)
so flow_testing mode and the other backends run without the ``openai``
package installed. The client is built on first ``call``, never in the
constructor — ``build_backend`` can therefore hand out a kill-switched
instance (placeholder key) without touching the SDK.

``think`` maps to nothing here: reasoning models (gpt-5.x, o-series)
always reason, non-reasoning models never do, and this backend
deliberately does not wire a per-request effort knob — reasoning depth
stays a property of the model choice. The flag is accepted for
protocol parity and deliberately ignored.

Per the ``ModelBackend`` protocol: no retries (runner's job), raise on
transport errors, never return a partial list.
"""

from __future__ import annotations

from icepick.processing.pass_at_k.config import SYSTEM_PROMPT

# Reasoning models (gpt-5.x and the o1/o3/o4 family) outright reject the
# temperature parameter and the legacy max_tokens parameter — the API 400s
# on both (verified against gpt-5.5, 2026-07-06); they take
# max_completion_tokens instead. Gate by model-name prefix.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAIBackend:
    """k sequential ``chat.completions.create`` calls per question."""

    name = "openai"

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
            import openai  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "the 'openai' package is required for the openai backend; "
                "install with `pip install openai` or add openai to your env"
            ) from exc
        self._client = openai.OpenAI(api_key=self.api_key)
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
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "timeout": timeout,
        }
        # Reasoning models (gpt-5.x, o-series) reject temperature and the
        # legacy max_tokens parameter; non-reasoning models keep the
        # historical kwargs unchanged.
        if self.model.startswith(_REASONING_PREFIXES):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        outputs = []
        for _ in range(k):
            response = client.chat.completions.create(**kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self._output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            outputs.append(response.choices[0].message.content or "")
        return outputs

    def usage(self) -> dict:
        """Tokens accumulated across every call this instance has made."""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        }

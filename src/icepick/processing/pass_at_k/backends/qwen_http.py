"""Qwen over a local OpenAI-compatible HTTP endpoint (LM Studio et al.).

Port of ModelBreaker's ``realmath/harvest_realmath.py:call_qwen`` — the
payload shape, the ``/no_think`` suffix convention and the system prompt
are kept byte-identical so pass@k numbers stay comparable to MB's
70-record harvest. The one structural difference: MB made one call per
invocation and looped at the harvest layer; here the backend owns the
k-loop so the runner can treat every backend identically.

Per the ``ModelBackend`` protocol this backend does NOT retry (that is
the runner's job), raises on transport errors, and never returns a
partial list — an error on sample 3 of 8 aborts the whole ``call``.

Ctrl-C boundary: the checkpoint commits per completed record via
``ScrapeCheckpoint.commit_record`` (see ``pass_at_k/checkpoint.py``), so
data integrity is identical to MB's — every finished record stays on
disk, only the in-flight record is discarded. What differs is
wall-clock waste on interrupt: MB loses up to one rollout of Qwen work,
icepick loses up to k rollouts (one full record's batch). Local Qwen
costs no money either way, and a resume re-runs the interrupted record
from scratch.

Optional API key, no kill switch: the default endpoint is local (LM
Studio et al.) and costs nothing per call, so no key is required —
``api_key=None`` sends no auth header and the local path is unchanged.
When ``api_key`` is set (resolved from ``--qwen-key-file``), every
request instead carries an ``Authorization: Bearer <token>`` header,
which lets this backend also target a remote OpenAI-compatible gateway
sitting behind bearer auth (e.g. Admiral Tangerine fronting LM Studio).
There is still no kill switch here: the key is an auth requirement, not
a spend risk, so it is exempt from the paid-backend gating described in
the config module docstring.
"""

from __future__ import annotations

from icepick.processing.pass_at_k.config import SYSTEM_PROMPT


class QwenHttpBackend:
    """POSTs to an OpenAI-compatible ``/v1/chat/completions`` URL."""

    name = "qwen_http"

    def __init__(self, url: str, model: str, api_key: str | None = None):
        self.url = url
        self.model = model
        self.api_key = api_key  # None for local/keyless endpoints; see module docstring
        self._input_tokens = 0
        self._output_tokens = 0

    def call(
        self,
        question: str,
        *,
        k: int,
        temperature: float,
        max_tokens: int,
        think: bool,
        timeout: float,
    ) -> list:
        """Return exactly ``k`` raw outputs via k sequential requests.

        Sequential on purpose: per-sample cache granularity lives
        upstream, and a local single-GPU endpoint serialises anyway.
        """
        import requests  # lazy: only production scoring needs the network

        # Qwen3 convention (ported from MB): appending " /no_think"
        # disables the reasoning phase; think=True sends the bare question.
        user = question if think else question + " /no_think"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        outputs = []
        for _ in range(k):
            # Only add the Authorization header when a key is configured —
            # the no-key call below is byte-for-byte what this backend has
            # always sent, so the local/keyless path stays unchanged.
            post_kwargs = {"timeout": timeout}
            if self.api_key:
                post_kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}
            r = requests.post(self.url, json=payload, **post_kwargs)
            r.raise_for_status()
            data = r.json()
            # OpenAI-compatible servers report prompt_/completion_tokens;
            # some local builds omit usage entirely — tolerate both.
            usage = data.get("usage")
            if usage:
                self._input_tokens += int(usage.get("prompt_tokens") or 0)
                self._output_tokens += int(usage.get("completion_tokens") or 0)
            outputs.append(data["choices"][0]["message"]["content"])
        return outputs

    def usage(self) -> dict:
        """Tokens accumulated across every call this instance has made."""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        }

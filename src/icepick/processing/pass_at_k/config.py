"""Pass@k stage configuration.

The stage runs each record's problem ``k`` times against a target model,
scores the outputs against the truth answer, and stamps ``pass_at_k`` +
``label`` back onto the record. It is positionable like groundtruth: run
it AFTER the wellposed cascade so rollouts are not wasted on ill-posed
problems (recommended), or standalone on any handoff JSONL.

POLICY: pass@k runs against ``qwen_http`` — a local OpenAI-compatible
endpoint (LM Studio / vLLM / Ollama) — matching ModelBreaker's original
harvest architecture. Paid backends (``anthropic``, ``openai``) are
disabled by default and refuse to run even with ``allow_live_calls``
unless ``i_understand_paid_backend_is_off_policy`` is ALSO set. The
policy exists because k rollouts × N records × paid tokens dwarfs the
rest of the pipeline's cost — a slip here is the single biggest
accidental-overspend risk, so it fails closed.

Kill switch layering (top → bottom):
  1. Default backend is ``qwen_http`` (no paid backend selected by default).
  2. ``anthropic`` / ``openai`` require ``allow_live_calls=True`` AND
     ``i_understand_paid_backend_is_off_policy=True`` in production.
  3. Even then, the backend builder loads real keys only behind the same
     flags; without them the client uses a placeholder literal so any
     accidental invocation hits 401 without spending money.

Band constants live at ``contracts/records.py`` — imported, never
re-declared. NOTE: ModelBreaker's harvest used a band of (0.125, 0.875);
icepick's contract is (0.125, 0.75). Records with pass@k in (0.75, 0.875]
label ``solved`` here where MB labelled them ``band`` — expected when
comparing against MB's 70-record dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from icepick.config import ConfigError
from icepick.contracts.records import BAND_HI, BAND_LO  # noqa: F401  (re-exported for the stage)

BACKEND_QWEN_HTTP = "qwen_http"
BACKEND_ANTHROPIC = "anthropic"
BACKEND_OPENAI = "openai"
BACKEND_VALUES = (BACKEND_QWEN_HTTP, BACKEND_ANTHROPIC, BACKEND_OPENAI)

# Paid backends sit behind the allow_live_calls kill switch AND the
# off-policy acknowledgement flag. Local backends bypass both.
PAID_BACKENDS = (BACKEND_ANTHROPIC, BACKEND_OPENAI)

# Backend-specific default subject models, used when cfg.model is None.
# The paid-backend entries are intentionally left unset — an operator who
# opts into a paid backend must also state which model they want. That
# forecloses the accidental "default = Haiku on a 10k-record scrape"
# blast radius.
DEFAULT_MODELS = {
    BACKEND_QWEN_HTTP: "qwen/qwen3-8b",
    BACKEND_ANTHROPIC: None,
    BACKEND_OPENAI: None,
}

# The one system prompt every backend sends, ported verbatim from
# ModelBreaker's harvest so Qwen comparisons stay apples-to-apples.
SYSTEM_PROMPT = "Solve the problem. State only the final answer inside \\boxed{}."


@dataclass
class PassAtKConfig:
    """One field per decision; enums over free text; refuse ambiguous forms."""

    mode: str = "production"
    output_dir: Path = field(default_factory=lambda: Path("out/pass_at_k"))
    backend: str = BACKEND_QWEN_HTTP  # policy default; qwen_http is free + local
    model: Optional[str] = None  # backend-specific default when None
    k: int = 8
    temperature: float = 0.7
    max_tokens: int = 8192
    think: bool = False  # request reasoning; <think> tags are always stripped before scoring
    max_concurrent: int = 4  # concurrent record scoring; rollouts within a record stay sequential
    request_timeout_s: float = 180.0
    calibration_sheet: Optional[Path] = None
    anthropic_key_file: Optional[Path] = None
    openai_key_file: Optional[Path] = None
    backend_url: Optional[str] = None  # qwen_http only
    allow_live_calls: bool = False  # kill switch #1: paid backends refuse production without it
    # Kill switch #2: policy states pass@k runs against qwen_http only.
    # Selecting a paid backend requires BOTH flags — operators explicitly
    # acknowledge they are going off-policy. Single-flag opt-in was too
    # easy to trigger accidentally; two flags force a deliberate decision.
    i_understand_paid_backend_is_off_policy: bool = False
    keep_garbage: bool = False  # score records with junk truth instead of dropping them
    # Optional cost estimation. When either rate is set, the run manifest
    # carries a token_usage.estimated_cost block (marked is_estimate: true).
    cost_per_input_mtok: Optional[float] = None
    cost_per_output_mtok: Optional[float] = None
    # Per-record retry policy for backend errors (network, 429/5xx).
    max_retries: int = 3
    retry_base_delay: float = 2.0
    retry_max_delay: float = 30.0

    @property
    def resolved_model(self) -> str:
        default = DEFAULT_MODELS[self.backend]
        if self.model:
            return self.model
        if default is None:
            # Paid backends have no default model — the operator must
            # state one, so the "Haiku slipped through by default" blast
            # radius stays boxed in.
            raise ConfigError(
                f"pass_at_k.model is required for backend={self.backend!r}: "
                "policy is qwen_http; paid backends have no default model "
                "so operators must state one explicitly"
            )
        return default

    def validate(self) -> None:
        if self.mode not in ("production", "flow_testing"):
            raise ConfigError(
                f"pass_at_k.mode must be 'production' or 'flow_testing', got {self.mode!r}"
            )
        if self.mode == "flow_testing" and self.calibration_sheet is None:
            raise ConfigError("pass_at_k.mode=flow_testing requires calibration_sheet")
        if self.backend not in BACKEND_VALUES:
            raise ConfigError(
                f"pass_at_k.backend must be one of {BACKEND_VALUES}, got {self.backend!r}"
            )
        if self.k < 1:
            raise ConfigError(f"pass_at_k.k must be >= 1, got {self.k}")
        if self.temperature < 0:
            raise ConfigError(
                f"pass_at_k.temperature must be >= 0, got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ConfigError(
                f"pass_at_k.max_tokens must be >= 1, got {self.max_tokens}"
            )
        if self.max_concurrent < 1:
            raise ConfigError(
                f"pass_at_k.max_concurrent must be >= 1, got {self.max_concurrent}"
            )
        if self.max_retries < 0:
            raise ConfigError(
                f"pass_at_k.max_retries must be >= 0, got {self.max_retries}"
            )
        if self.mode != "production":
            return
        # Production-only checks below: flow_testing never touches a backend.
        if self.backend == BACKEND_QWEN_HTTP and not self.backend_url:
            raise ConfigError(
                "pass_at_k.backend_url is required for the qwen_http backend "
                "(e.g. http://127.0.0.1:1234/v1/chat/completions)"
            )
        if self.backend in PAID_BACKENDS:
            if not self.allow_live_calls:
                raise ConfigError(
                    f"pass_at_k.backend={self.backend!r} spends real API calls in "
                    "production; pass --allow-live-calls to opt in (kill switch, "
                    "see module docstring)"
                )
            if not self.i_understand_paid_backend_is_off_policy:
                raise ConfigError(
                    f"pass_at_k.backend={self.backend!r} is off-policy: pass@k "
                    "is meant to run against qwen_http (local, free). Selecting "
                    "a paid backend requires BOTH --allow-live-calls AND "
                    "--i-understand-paid-backend-is-off-policy so the choice is "
                    "deliberate. See the module docstring for the rationale."
                )
            # Explicit model required for paid backends — see resolved_model.
            if not self.model:
                raise ConfigError(
                    f"pass_at_k.model is required for backend={self.backend!r}: "
                    "paid backends have no default model so operators must "
                    "state one explicitly (e.g. 'claude-sonnet-4-6')"
                )
        if self.backend == BACKEND_ANTHROPIC and self.anthropic_key_file is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise ConfigError(
                "pass_at_k.anthropic_key_file is required in production mode "
                "(or set ANTHROPIC_API_KEY in the environment)"
            )
        if self.backend == BACKEND_OPENAI and self.openai_key_file is None and not os.environ.get("OPENAI_API_KEY"):
            raise ConfigError(
                "pass_at_k.openai_key_file is required in production mode "
                "(or set OPENAI_API_KEY in the environment)"
            )

    def echo(self) -> dict:
        # ``resolved_model`` raises ConfigError when a paid backend has no
        # model set. echo() is called from summary + manifest reporting
        # after validate() has run, so a raise here would be a code bug,
        # not an operator concern — but fall back to the raw ``self.model``
        # value so the manifest still writes cleanly if it ever fires.
        try:
            model = self.resolved_model
        except ConfigError:
            model = self.model
        return {
            "mode": self.mode,
            "output_dir": str(self.output_dir),
            "backend": self.backend,
            "model": model,
            "k": self.k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "think": self.think,
            "max_concurrent": self.max_concurrent,
            "request_timeout_s": self.request_timeout_s,
            "calibration_sheet": str(self.calibration_sheet) if self.calibration_sheet else None,
            "anthropic_key_file": str(self.anthropic_key_file) if self.anthropic_key_file else None,
            "openai_key_file": str(self.openai_key_file) if self.openai_key_file else None,
            "backend_url": self.backend_url,
            "allow_live_calls": self.allow_live_calls,
            "i_understand_paid_backend_is_off_policy": self.i_understand_paid_backend_is_off_policy,
            "keep_garbage": self.keep_garbage,
            "cost_per_input_mtok": self.cost_per_input_mtok,
            "cost_per_output_mtok": self.cost_per_output_mtok,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
            "retry_max_delay": self.retry_max_delay,
        }

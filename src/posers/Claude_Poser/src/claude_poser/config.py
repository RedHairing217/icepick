"""Configuration for the c01 well-posedness check.

Knobs are kept here so summaries can echo the exact policy used for a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


PROCESSOR_MODES = ("production", "flow_testing")
JUDGE_PROVIDERS = ("anthropic", "openai")
EXTRACTED_JUDGE_POLICIES = ("always", "on_scanner_hit")

_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
}


@dataclass
class WellposedConfig:
    # Tier control
    enable_judge: bool = False
    judge_samples: int = 3
    judge_uphold: int = 2

    # Provider selection — swap between API backends without code changes.
    judge_provider: str = "anthropic"
    judge_model: Optional[str] = None  # resolved via resolve_model() if None
    judge_timeout_s: float = 30.0

    # Provider credentials / endpoints (read at instantiation time so an
    # --env-file load before constructing the config is picked up).
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    )

    # Cache
    judge_cache_path: Optional[str] = None

    # Extracted-provenance policy — controls whether the judge is gated on
    # scanner hits. "always" (default) always defers to the judge for
    # extracted records; "on_scanner_hit" restores the old cost-gating
    # behavior (scanner false-negatives become full-pass verdicts).
    extracted_judge_policy: str = "always"

    # Mode + replay
    processor_mode: str = "production"
    calibration_sheet: Optional[str] = None

    # ----- derived helpers -----

    def resolve_model(self) -> str:
        """Return the model id to use: explicit override > provider env var > default."""
        if self.judge_model:
            return self.judge_model
        if self.judge_provider == "anthropic":
            return os.environ.get("ANTHROPIC_MODEL") or _DEFAULT_MODELS["anthropic"]
        if self.judge_provider == "openai":
            return os.environ.get("OPENAI_MODEL") or _DEFAULT_MODELS["openai"]
        raise ValueError(f"unknown judge_provider {self.judge_provider!r}")

    def active_api_key(self) -> Optional[str]:
        if self.judge_provider == "anthropic":
            return self.anthropic_api_key
        if self.judge_provider == "openai":
            return self.openai_api_key
        return None

    # ----- run-summary view -----

    def echo(self) -> dict:
        """Serialisable view for summary files. Strips secrets, keeps presence flags."""
        d = asdict(self)
        d.pop("anthropic_api_key", None)
        d.pop("openai_api_key", None)
        d["anthropic_api_key_present"] = bool(self.anthropic_api_key)
        d["openai_api_key_present"] = bool(self.openai_api_key)
        if self.judge_provider in JUDGE_PROVIDERS:
            d["resolved_model"] = self.resolve_model()
        return d

    def validate(self) -> None:
        if self.processor_mode not in PROCESSOR_MODES:
            raise ValueError(
                f"processor_mode must be one of {PROCESSOR_MODES}, got {self.processor_mode!r}"
            )
        if self.processor_mode == "flow_testing" and not self.calibration_sheet:
            raise ValueError("flow_testing mode requires --calibration-sheet")
        if self.judge_provider not in JUDGE_PROVIDERS:
            raise ValueError(
                f"judge_provider must be one of {JUDGE_PROVIDERS}, got {self.judge_provider!r}"
            )
        if self.extracted_judge_policy not in EXTRACTED_JUDGE_POLICIES:
            raise ValueError(
                f"extracted_judge_policy must be one of {EXTRACTED_JUDGE_POLICIES}, "
                f"got {self.extracted_judge_policy!r}"
            )
        if self.judge_samples < 1:
            raise ValueError("judge_samples must be >= 1")
        if not (1 <= self.judge_uphold <= self.judge_samples):
            raise ValueError("judge_uphold must satisfy 1 <= uphold <= samples")

"""Groundtruth-stage configuration.

The check sits between ingest and the gate. It is positionable: run it
BEFORE pass@k to discard records before paying sampling cost, or AFTER
pass@k to filter pass@k survivors before the gate. The config does not
encode position — it's the user's choice of which ``--input`` to feed.

The bar for ``published`` is deliberately strict: the paper must be
peer-reviewed AND indexed in a reputable bibliographic database
(Scopus, Web of Science, DBLP, MathSciNet, or equivalent). Predatory
journals, preprint-only postings, and unindexed workshop papers do not
pass. The bar is encoded in the system prompt the adapter sends to
Claude; ``custom_bar_instructions`` lets the user override or extend it
without forking the adapter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from icepick.config import ConfigError


@dataclass
class GroundtruthConfig:
    """One field per decision; enums over free text; refuse ambiguous forms."""

    mode: str = "production"
    output_dir: Path = field(default_factory=lambda: Path("out/groundtruth"))
    anthropic_key_file: Optional[Path] = None
    judge_model: str = "claude-opus-4-7"
    judge_samples: int = 3
    judge_uphold: int = 2
    max_concurrent: int = 8
    request_timeout_s: float = 60.0
    cache_path: Optional[Path] = None
    calibration_sheet: Optional[Path] = None
    discard_generated: bool = True
    custom_bar_instructions: Optional[str] = None
    # Optional cost estimation. When either rate is set, the run manifest
    # carries a token_usage.estimated_cost block (marked is_estimate: true).
    cost_per_input_mtok: Optional[float] = None
    cost_per_output_mtok: Optional[float] = None

    def validate(self) -> None:
        if self.mode not in ("production", "flow_testing"):
            raise ConfigError(
                f"groundtruth.mode must be 'production' or 'flow_testing', got {self.mode!r}"
            )
        if self.mode == "flow_testing" and self.calibration_sheet is None:
            raise ConfigError(
                "groundtruth.mode=flow_testing requires calibration_sheet"
            )
        if self.judge_samples < 1:
            raise ConfigError(
                f"groundtruth.judge_samples must be >= 1, got {self.judge_samples}"
            )
        if self.judge_uphold < 1 or self.judge_uphold > self.judge_samples:
            raise ConfigError(
                f"groundtruth.judge_uphold ({self.judge_uphold}) must be in "
                f"[1, judge_samples] (judge_samples={self.judge_samples})"
            )
        if self.max_concurrent < 1:
            raise ConfigError(
                f"groundtruth.max_concurrent must be >= 1, got {self.max_concurrent}"
            )
        if (
            self.mode == "production"
            and self.anthropic_key_file is None
            and not os.environ.get("ANTHROPIC_API_KEY")
        ):
            raise ConfigError(
                "groundtruth.anthropic_key_file is required in production mode "
                "(or set ANTHROPIC_API_KEY in the environment)"
            )

    def echo(self) -> dict:
        return {
            "mode": self.mode,
            "output_dir": str(self.output_dir),
            "anthropic_key_file": str(self.anthropic_key_file) if self.anthropic_key_file else None,
            "judge_model": self.judge_model,
            "judge_samples": self.judge_samples,
            "judge_uphold": self.judge_uphold,
            "max_concurrent": self.max_concurrent,
            "request_timeout_s": self.request_timeout_s,
            "cache_path": str(self.cache_path) if self.cache_path else None,
            "calibration_sheet": str(self.calibration_sheet) if self.calibration_sheet else None,
            "discard_generated": self.discard_generated,
            "custom_bar_instructions": self.custom_bar_instructions,
            "cost_per_input_mtok": self.cost_per_input_mtok,
            "cost_per_output_mtok": self.cost_per_output_mtok,
        }

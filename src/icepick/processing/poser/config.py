"""Wellposed-stage configuration.

The wellposed stage runs a *fleet* of ``(build, provider)`` combinations.
Each combination is one poser invocation:

    build      = which poser binary to drive       (``claude`` | ``codex``)
    provider   = which judge API backend it calls  (``anthropic`` | ``openai``)

There are four legal combinations:

    claude + anthropic     codex + anthropic
    claude + openai        codex + openai

Any subset can run in a single invocation; they execute in parallel.
The single human-in-the-loop decision is **which combinations to run**.
Everything else has sensible defaults.

Defaults intentionally favour conservatism: judge tier on, three samples
with two-of-three uphold, intersection policy when more than one combo
runs (a record passes only if every combo agrees it is well-posed).

Naming history: Claude_Poser was originally ``Anthro_Poser`` and
Codex_Poser was originally ``GPT_Poser``. CLI binaries followed the
rename to ``claude-poser`` and ``codex-poser``. The ``provider``
dimension was introduced when both posers gained an OpenAI judge
backend alongside their original Anthropic one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from icepick.config import ConfigError

# --- builds (which poser binary) ----------------------------------------
BUILD_CLAUDE = "claude"
BUILD_CODEX = "codex"
BUILD_CHOICES = (BUILD_CLAUDE, BUILD_CODEX)

# --- providers (which judge backend) ------------------------------------
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_CHOICES = (PROVIDER_ANTHROPIC, PROVIDER_OPENAI)

# --- combination policies (how the gate-input file is assembled) --------
POLICY_INTERSECT = "intersect"        # admit iff ALL combos admit
POLICY_UNION = "union"                # admit iff ANY combo admits
POLICY_MAJORITY = "majority"          # admit iff > half admit
POLICY_PREFER = "prefer:"             # prefix; followed by combo_key e.g. "prefer:claude:anthropic"
COMPARISON_POLICIES_BASE = (POLICY_INTERSECT, POLICY_UNION, POLICY_MAJORITY)

# --- extracted-provenance judge policy (claude-poser only) --------------
# 'always' (default): for extracted-provenance records, always defer to the
#   judge when the judge tier is enabled. The scanner provides supplementary
#   evidence but does not gate the judge call.
# 'on_scanner_hit': legacy cost-gating — only call the judge when the
#   scanner fires. Preserves the pre-fix behaviour where scanner
#   false-negatives become full-pass verdicts. Use only for replays of
#   trusted corpora when cost matters more than correctness.
EXTRACTED_JUDGE_POLICY_ALWAYS = "always"
EXTRACTED_JUDGE_POLICY_ON_HIT = "on_scanner_hit"
EXTRACTED_JUDGE_POLICIES = (EXTRACTED_JUDGE_POLICY_ALWAYS, EXTRACTED_JUDGE_POLICY_ON_HIT)


@dataclass(frozen=True)
class Combo:
    """One fleet entry: which poser, which provider."""

    build: str
    provider: str

    def key(self) -> str:
        """Stable identifier — used in file names, manifest keys, and policy strings."""
        return f"{self.build}:{self.provider}"

    def slug(self) -> str:
        """Filesystem-safe identifier — for output file basenames."""
        return f"{self.build}_{self.provider}"


def parse_combo(spec: str) -> Combo:
    """Parse a 'build:provider' string. Raises ConfigError on bad spec."""
    if ":" not in spec:
        raise ConfigError(
            f"combo {spec!r} must be 'build:provider' (e.g. 'claude:anthropic')"
        )
    build, _, provider = spec.partition(":")
    build, provider = build.strip().lower(), provider.strip().lower()
    if build not in BUILD_CHOICES:
        raise ConfigError(f"combo {spec!r}: build must be one of {BUILD_CHOICES}, got {build!r}")
    if provider not in PROVIDER_CHOICES:
        raise ConfigError(f"combo {spec!r}: provider must be one of {PROVIDER_CHOICES}, got {provider!r}")
    return Combo(build=build, provider=provider)


def all_combos() -> list:
    return [Combo(b, p) for b in BUILD_CHOICES for p in PROVIDER_CHOICES]


@dataclass
class PoserSettings:
    """Per-build knobs. Defaults match the discovery synthesis recommendations.

    ``judge_model`` overrides the provider-default model for THIS build,
    on whichever provider it runs against. If you want different models
    per provider, leave this None and rely on the env file's
    ``ANTHROPIC_MODEL`` / ``OPENAI_MODEL`` values.
    """

    cli_path: str = ""
    judge_model: Optional[str] = None
    extra_args: list = field(default_factory=list)


@dataclass
class WellposedConfig:
    """Wellposed-stage config.

    Fields are deliberately form-like: one clear decision per field,
    enums over free text, dangerous/costly choices grouped together.
    """

    combos: list = field(default_factory=list)
    mode: str = "production"
    output_dir: Path = field(default_factory=lambda: Path("out/wellposed"))
    anthropic_key_file: Optional[Path] = None
    openai_key_file: Optional[Path] = None
    enable_judge_tier: bool = True
    judge_samples: int = 3
    judge_uphold: int = 2
    calibration_sheet: Optional[Path] = None
    comparison_policy: str = POLICY_INTERSECT
    # Only claude-poser accepts this flag; codex-poser ignores it. The
    # setting is recorded in the manifest regardless so audit trails
    # capture the operator's intent even for codex-only runs.
    extracted_judge_policy: str = EXTRACTED_JUDGE_POLICY_ALWAYS
    serialize_fleet: bool = False
    # Optional cost estimation, mirroring GroundtruthConfig. When either rate
    # is set, the poser run_manifest carries a token_usage.estimated_cost
    # block (marked is_estimate: true).
    cost_per_input_mtok: Optional[float] = None
    cost_per_output_mtok: Optional[float] = None
    claude: PoserSettings = field(
        default_factory=lambda: PoserSettings(cli_path="claude-poser")
    )
    codex: PoserSettings = field(
        default_factory=lambda: PoserSettings(cli_path="codex-poser")
    )

    def validate(self) -> None:
        """Refuse ambiguous forms instead of guessing."""
        if not self.combos:
            raise ConfigError(
                "wellposed.combos must list at least one (build, provider) combination"
            )
        seen_keys: set = set()
        for combo in self.combos:
            if not isinstance(combo, Combo):
                raise ConfigError(f"wellposed.combos entries must be Combo, got {type(combo).__name__}")
            if combo.build not in BUILD_CHOICES:
                raise ConfigError(
                    f"combo {combo.key()!r}: build must be one of {BUILD_CHOICES}, got {combo.build!r}"
                )
            if combo.provider not in PROVIDER_CHOICES:
                raise ConfigError(
                    f"combo {combo.key()!r}: provider must be one of {PROVIDER_CHOICES}, got {combo.provider!r}"
                )
            if combo.key() in seen_keys:
                raise ConfigError(f"duplicate combo {combo.key()!r} in wellposed.combos")
            seen_keys.add(combo.key())

        if self.mode not in ("production", "flow_testing"):
            raise ConfigError(
                f"wellposed.mode must be 'production' or 'flow_testing', got {self.mode!r}"
            )
        self._validate_policy()
        if self.mode == "flow_testing" and self.calibration_sheet is None:
            raise ConfigError("wellposed.mode=flow_testing requires calibration_sheet")
        for combo in self.combos:
            if combo.build == BUILD_CODEX and self.enable_judge_tier and self.mode == "flow_testing":
                raise ConfigError(
                    f"combo {combo.key()!r}: codex build does not support --judge in flow_testing mode; "
                    "set enable_judge_tier=false or switch mode to production"
                )
        if self.enable_judge_tier and self.mode == "production":
            providers_used = {c.provider for c in self.combos}
            if (
                PROVIDER_ANTHROPIC in providers_used
                and self.anthropic_key_file is None
                and not os.environ.get("ANTHROPIC_API_KEY")
            ):
                raise ConfigError(
                    "wellposed.anthropic_key_file is required when any combo uses provider=anthropic "
                    "and enable_judge_tier=true in production mode (or set ANTHROPIC_API_KEY in env)"
                )
            if (
                PROVIDER_OPENAI in providers_used
                and self.openai_key_file is None
                and not os.environ.get("OPENAI_API_KEY")
            ):
                raise ConfigError(
                    "wellposed.openai_key_file is required when any combo uses provider=openai "
                    "and enable_judge_tier=true in production mode (or set OPENAI_API_KEY in env)"
                )
        if self.judge_uphold > self.judge_samples:
            raise ConfigError(
                f"wellposed.judge_uphold ({self.judge_uphold}) cannot exceed "
                f"judge_samples ({self.judge_samples})"
            )
        if self.extracted_judge_policy not in EXTRACTED_JUDGE_POLICIES:
            raise ConfigError(
                f"wellposed.extracted_judge_policy must be one of "
                f"{EXTRACTED_JUDGE_POLICIES}, got {self.extracted_judge_policy!r}"
            )

    def _validate_policy(self) -> None:
        p = self.comparison_policy
        if p in COMPARISON_POLICIES_BASE:
            return
        if p.startswith(POLICY_PREFER):
            target_key = p[len(POLICY_PREFER):]
            keys = {c.key() for c in self.combos}
            if target_key not in keys:
                raise ConfigError(
                    f"comparison_policy={p!r} refers to combo {target_key!r} which is not in the fleet"
                )
            return
        raise ConfigError(
            f"wellposed.comparison_policy must be one of {COMPARISON_POLICIES_BASE} "
            f"or 'prefer:<build>:<provider>', got {p!r}"
        )

    def echo(self) -> dict:
        """Serialisable representation for run manifests and summaries."""
        return {
            "combos": [c.key() for c in self.combos],
            "mode": self.mode,
            "output_dir": str(self.output_dir),
            "anthropic_key_file": str(self.anthropic_key_file) if self.anthropic_key_file else None,
            "openai_key_file": str(self.openai_key_file) if self.openai_key_file else None,
            "enable_judge_tier": self.enable_judge_tier,
            "judge_samples": self.judge_samples,
            "judge_uphold": self.judge_uphold,
            "calibration_sheet": str(self.calibration_sheet) if self.calibration_sheet else None,
            "comparison_policy": self.comparison_policy,
            "extracted_judge_policy": self.extracted_judge_policy,
            "serialize_fleet": self.serialize_fleet,
            "cost_per_input_mtok": self.cost_per_input_mtok,
            "cost_per_output_mtok": self.cost_per_output_mtok,
            "claude": {
                "cli_path": self.claude.cli_path,
                "judge_model": self.claude.judge_model,
                "extra_args": list(self.claude.extra_args),
            },
            "codex": {
                "cli_path": self.codex.cli_path,
                "judge_model": self.codex.judge_model,
                "extra_args": list(self.codex.extra_args),
            },
        }

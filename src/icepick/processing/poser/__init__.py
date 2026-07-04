"""Pre-gate well-posedness stage.

Well-posedness is **not** a gate check. It is decided before the gate
ever runs, by an external poser tool — currently ``Claude_Poser`` or
``Codex_Poser`` (CLI binaries ``claude-poser`` and ``codex-poser``).
Each poser can route to either the Anthropic or the OpenAI judge API,
giving four legal combinations:

    claude:anthropic   claude:openai   codex:anthropic   codex:openai

Any subset of those four can run in a single icepick invocation; they
execute in parallel. The single human-in-the-loop decision is the set
of ``(build, provider)`` combinations to run.

Naming history: Claude_Poser was originally ``Anthro_Poser`` and
Codex_Poser was originally ``GPT_Poser``. CLI binary names tracked the
rename. The provider dimension was introduced when both posers gained
an OpenAI judge backend alongside their original Anthropic one.

Design rules from the spec:

- Family and source are data, not code. Poser build and provider are
  data too — a new combo is wiring, not a new branch in routing logic.
- Each poser is invoked via subprocess. icepick never imports poser
  internals so the two tools can evolve in their own venvs.
- icepick **injects** ``uid`` into every input record before invoking a
  poser. All posers preserve uid if supplied. Injection is the single
  canonical join key across all combos.
- Adapters produce canonical ``PoserVerdict`` records; raw poser output
  is preserved verbatim in ``raw_payload`` so nothing is lost.
- Provider segregation is enforced at the poser layer: claude-poser
  refuses to load the other provider's key file when ``--provider X``
  is set, and codex-poser only reads the ``--key-env`` file for the
  selected ``--judge-provider``.
- Full automation: the only human decision is which combinations to run.
"""

from icepick.processing.poser.base import (
    CANONICAL_STATUSES,
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserAdapter,
    PoserRequest,
    PoserRunResult,
    PoserVerdict,
    compute_uid,
    inject_uid,
)
from icepick.processing.poser.cascade import (
    DEFAULT_STAGES,
    CascadeConfig,
    CascadeOutcome,
    CascadeStageOutcome,
    StageSpec,
    parse_stages,
    run_cascade,
)
from icepick.processing.poser.config import (
    BUILD_CHOICES,
    BUILD_CLAUDE,
    BUILD_CODEX,
    COMPARISON_POLICIES_BASE,
    POLICY_INTERSECT,
    POLICY_MAJORITY,
    POLICY_PREFER,
    POLICY_UNION,
    PROVIDER_ANTHROPIC,
    PROVIDER_CHOICES,
    PROVIDER_OPENAI,
    Combo,
    PoserSettings,
    WellposedConfig,
    all_combos,
    parse_combo,
)

__all__ = [
    "BUILD_CHOICES",
    "BUILD_CLAUDE",
    "BUILD_CODEX",
    "CANONICAL_STATUSES",
    "COMPARISON_POLICIES_BASE",
    "CascadeConfig",
    "CascadeOutcome",
    "CascadeStageOutcome",
    "Combo",
    "DEFAULT_STAGES",
    "POLICY_INTERSECT",
    "POLICY_MAJORITY",
    "POLICY_PREFER",
    "POLICY_UNION",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_CHOICES",
    "PROVIDER_OPENAI",
    "PoserAdapter",
    "PoserRequest",
    "PoserRunResult",
    "PoserSettings",
    "PoserVerdict",
    "STATUS_DEFER",
    "STATUS_ERROR",
    "STATUS_ILL_POSED",
    "STATUS_WELL_POSED",
    "StageSpec",
    "WellposedConfig",
    "all_combos",
    "compute_uid",
    "inject_uid",
    "parse_combo",
    "parse_stages",
    "run_cascade",
]

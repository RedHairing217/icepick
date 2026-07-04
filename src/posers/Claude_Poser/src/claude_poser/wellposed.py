"""c01 well-posedness check.

Tiered decision, in this order:

  1. Trusted provenance (computed, or manual+truth_policy=trusted)
     -> code-pass immediately.

  2. Extracted / external / unknown provenance:
       a. Run the code-tier scanner. Hits are recorded as evidence but do
          NOT gate the judge — a false-negative in the scanner used to
          become a full-pass verdict the judge never got to correct.

       b. If --judge is enabled AND extracted_judge_policy == "always"
          (the default): defer to the 3-sample judge. Scanner hits ride
          along as `code_hits` in the result so the judge's decision is
          auditable against the code-tier evidence.

       c. If --judge is enabled AND extracted_judge_policy == "on_scanner_hit"
          (old cost-gating behavior): only defer when the scanner fires;
          otherwise return code-pass. Preserved as an opt-in for
          cost-sensitive runs.

       d. If --judge is disabled: code-tier only. Hits -> flag, else pass.
          This is not recommended for extracted records — the scanner is
          empirically weak on arXiv text — but it stays available for
          fully-offline use.

  flow_testing mode replays every judge call from the calibration sheet
  regardless of provider, so no external API is touched.
"""

from __future__ import annotations

from typing import Iterable, Optional

from . import dangling
from .calibration_replay import load_sheet, make_replay_caller
from .config import WellposedConfig
from .judge import judge_wellposed
from .judge_cache import JudgeCache
from .schema import is_self_contained_provenance
from .scoring import score_from_check


def _build_result(
    record: dict,
    tier: str,
    status: str,
    *,
    code_hits: Optional[list] = None,
    judge: Optional[dict] = None,
) -> dict:
    result = {
        "uid": record["uid"],
        "rid": record["rid"],
        "source": record["source"],
        "provenance": record["provenance"],
        "tier": tier,
        "status": status,
        "code_hits": [h.to_dict() for h in (code_hits or [])],
        "judge": judge,
    }
    score, label = score_from_check(result)
    result["wellposed_score"] = round(score, 4)
    result["wellposed_status"] = label
    return result


def _call_judge(
    record: dict,
    cfg: WellposedConfig,
    cache: Optional[JudgeCache],
    replay_sheet: Optional[dict],
    hits: list,
) -> dict:
    caller = None
    if cfg.processor_mode == "flow_testing":
        if replay_sheet is None:
            raise RuntimeError("flow_testing mode requires a loaded calibration sheet")
        caller = make_replay_caller(replay_sheet, record["uid"])
    outcome = judge_wellposed(record["statement"], cfg, cache=cache, caller=caller)
    return _build_result(
        record,
        tier="judge",
        status=outcome.majority_verdict,
        code_hits=hits,
        judge=outcome.to_dict(),
    )


def check_record(
    record: dict,
    cfg: WellposedConfig,
    cache: Optional[JudgeCache] = None,
    replay_sheet: Optional[dict] = None,
) -> dict:
    # Tier 0: provenance trust
    if is_self_contained_provenance(record):
        return _build_result(record, tier="code", status="pass")

    # Tier 1: code-tier scan (always runs — its hits are evidence, not a gate)
    hits = dangling.scan(record["statement"])

    # Tier 2: judge (if enabled)
    if cfg.enable_judge:
        if cfg.extracted_judge_policy == "always":
            # New default: always defer for extracted records. Scanner hits
            # travel with the judge result as auditable evidence.
            return _call_judge(record, cfg, cache, replay_sheet, hits)

        if cfg.extracted_judge_policy == "on_scanner_hit":
            # Legacy cost-gating: judge only fires when scanner triggers.
            if hits:
                return _call_judge(record, cfg, cache, replay_sheet, hits)
            return _build_result(record, tier="code", status="pass")

        raise ValueError(
            f"unknown extracted_judge_policy {cfg.extracted_judge_policy!r}"
        )

    # Tier 3: code-only fallback (judge disabled)
    if hits:
        return _build_result(record, tier="code", status="flag", code_hits=hits)
    return _build_result(record, tier="code", status="pass")


def check_records(
    records: Iterable[dict],
    cfg: WellposedConfig,
    cache: Optional[JudgeCache] = None,
) -> list[dict]:
    cfg.validate()
    replay_sheet: Optional[dict] = None
    if cfg.processor_mode == "flow_testing":
        replay_sheet = load_sheet(cfg.calibration_sheet)
    return [check_record(r, cfg, cache=cache, replay_sheet=replay_sheet) for r in records]

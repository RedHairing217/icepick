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

from . import dangling, degeneracy
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
    degeneracy_hits: Optional[list] = None,
    review_flags: Optional[list] = None,
) -> dict:
    result = {
        "uid": record["uid"],
        "rid": record["rid"],
        "source": record["source"],
        "provenance": record["provenance"],
        "tier": tier,
        "status": status,
        "code_hits": [h.to_dict() for h in (code_hits or [])],
        "degeneracy_hits": [h.to_dict() for h in (degeneracy_hits or [])],
        "review_flags": sorted(set(review_flags or [])),
        "judge": judge,
    }
    score, label = score_from_check(result)
    result["wellposed_score"] = round(score, 4)
    result["wellposed_status"] = label
    return result


def _answer_consistency(judge_dict: dict, stored_answer: Optional[str]) -> str:
    """Audit pass-samples' derived answers against the record's stored answer.

    Returns "match" | "mismatch" | "unknown". Lexical only (normalise_math
    plus containment either way) — structurally different but equal
    expressions come back "mismatch", which costs a review glance, not the
    record. "unknown" when there is nothing to compare (no stored answer,
    or no pass sample volunteered a derived answer).
    """
    if not stored_answer:
        return "unknown"
    want = degeneracy.normalise_math(str(stored_answer))
    if not want:
        return "unknown"
    derived = [
        degeneracy.normalise_math(s.get("derived_answer") or "")
        for s in judge_dict.get("samples") or []
        if s.get("verdict") == "pass" and s.get("derived_answer")
    ]
    derived = [d for d in derived if d]
    if not derived:
        return "unknown"
    for d in derived:
        if d == want or d in want or want in d:
            return "match"
    return "mismatch"


def _call_judge(
    record: dict,
    cfg: WellposedConfig,
    cache: Optional[JudgeCache],
    replay_sheet: Optional[dict],
    hits: list,
    degeneracy_hits: list,
) -> dict:
    caller = None
    if cfg.processor_mode == "flow_testing":
        if replay_sheet is None:
            raise RuntimeError("flow_testing mode requires a loaded calibration sheet")
        caller = make_replay_caller(replay_sheet, record["uid"])
    outcome = judge_wellposed(record["statement"], cfg, cache=cache, caller=caller)
    judge_dict = outcome.to_dict()

    review_flags: list = []
    if degeneracy_hits:
        review_flags.append("degenerate_candidate")
    consistency = _answer_consistency(judge_dict, record.get("answer"))
    judge_dict["answer_consistency"] = consistency
    if consistency == "mismatch":
        review_flags.append("answer_mismatch")

    return _build_result(
        record,
        tier="judge",
        status=outcome.majority_verdict,
        code_hits=hits,
        judge=judge_dict,
        degeneracy_hits=degeneracy_hits,
        review_flags=review_flags,
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

    # Tier 1: code-tier scans (always run — hits are evidence, not a gate).
    # The degeneracy scan needs the stored answer; records without one
    # simply skip it.
    hits = dangling.scan(record["statement"])
    deg_hits = degeneracy.scan(record["statement"], record.get("answer") or "")

    # Tier 2: judge (if enabled)
    if cfg.enable_judge:
        if cfg.extracted_judge_policy == "always":
            # New default: always defer for extracted records. Scanner hits
            # travel with the judge result as auditable evidence.
            return _call_judge(record, cfg, cache, replay_sheet, hits, deg_hits)

        if cfg.extracted_judge_policy == "on_scanner_hit":
            # Legacy cost-gating: judge only fires when scanner triggers.
            if hits:
                return _call_judge(record, cfg, cache, replay_sheet, hits, deg_hits)
            return _build_result(
                record, tier="code", status="pass",
                degeneracy_hits=deg_hits,
                review_flags=["degenerate_candidate"] if deg_hits else None,
            )

        raise ValueError(
            f"unknown extracted_judge_policy {cfg.extracted_judge_policy!r}"
        )

    # Tier 3: code-only fallback (judge disabled)
    review = ["degenerate_candidate"] if deg_hits else None
    if hits:
        return _build_result(
            record, tier="code", status="flag", code_hits=hits,
            degeneracy_hits=deg_hits, review_flags=review,
        )
    return _build_result(
        record, tier="code", status="pass",
        degeneracy_hits=deg_hits, review_flags=review,
    )


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

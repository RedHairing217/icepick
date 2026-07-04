"""Map a c01 check result to a well-posedness score in [0.0, 1.0].

Score convention:
- 1.0  fully well-posed (code-pass on trusted provenance, or 3/3 judge pass)
- 0.0  ill-posed or confirmed defect (code flag without judge, judge majority flag,
       or insufficient_context majority)
- 0.5  defer (judge unreachable or split with no majority)
- fractional values reflect judge-sample agreement (votes_for_wellposed / samples)

Status is the categorical label downstream stages would route on:
  pass | flag | insufficient_context | defer
"""

from __future__ import annotations

from typing import Optional


def score_from_check(result: dict) -> tuple[float, str]:
    tier = result.get("tier")
    status = result.get("status")

    if tier == "code":
        if status == "pass":
            return 1.0, "pass"
        if status == "flag":
            return 0.0, "flag"

    if tier == "judge":
        judge = result.get("judge") or {}
        verdict = judge.get("majority_verdict")
        n = max(1, len(judge.get("samples") or []))
        wellposed_votes = int(judge.get("wellposed_votes", 0))
        score = wellposed_votes / n
        if verdict == "pass":
            return score, "pass"
        if verdict == "flag":
            return score, "flag"
        if verdict == "insufficient_context":
            return 0.0, "insufficient_context"
        return 0.5, "defer"

    # No tier reached a decision (e.g. judge disabled and we shouldn't have
    # been called): treat as defer.
    return 0.5, "defer"

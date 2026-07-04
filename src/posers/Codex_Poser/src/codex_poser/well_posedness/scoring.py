"""c01 well-posedness scoring with an optional judge tier."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Callable, Iterable

from .contracts import DEFER, ERROR, FLAG, PASS, PassKRecord, WellPosednessResult

CHECK_ID = "c01_wellposed"

_REFERENCE = re.compile(r"\\(?:eqref|ref|cref|Cref|autoref|vref|pageref|labelcref)\s*\{[^}]*\}?")
_CITATION = re.compile(r"\\(?:cite|citep|citet|citeauthor|citeyear)\s*\{[^}]*\}?")
_DANGLING_LABEL = re.compile(r"\\label\s*\{[^}]*\}?")
_CONTEXT_PHRASES = re.compile(
    r"\b("
    r"as above|defined above|defined below|as before|the previous theorem|"
    r"the following theorem|this theorem|this lemma|the lemma|the proposition|"
    r"the figure|the table|shown below|shown above|in the paper|in this paper|"
    r"in the article|from the text|from the passage|using the notation"
    r")\b",
    re.IGNORECASE,
)
_LATEX_EQUATION_NUMBER = re.compile(r"\((?:\d+(?:\.\d+)*|[A-Za-z])\)")

SCORE_BY_STATUS = {
    PASS: 1.0,
    FLAG: 0.0,
    DEFER: 0.5,
    ERROR: 0.0,
}


_PROMPT = (
    "You audit a mathematics problem for well-posedness. Dangling cross-references "
    "have already been checked separately, so do not consider those. Decide only "
    "whether the self-contained statement determines a single answer: flag it if "
    "notation is used but never defined, or the answer is otherwise underdetermined. "
    "A problem that is simply hard, or solvable by doing the stated task, is "
    "well-posed. Do not solve it.\n\n"
    "Statement:\n{statement}\n\nStored answer:\n{truth}\n\n"
    "Pass@k context:\n{passk_context}\n\n"
    "Soft context signals from code: {soft_context}\n\n"
    "Reply with a JSON object only, keys: determined (true or false), "
    "insufficient_context (true or false: true when the statement relies on context "
    "not included, a referenced lemma, an undefined paper-specific symbol, or a "
    "missing definition), reason (one sentence), confidence (0 to 1)."
)


def score_records(
    records: Iterable[PassKRecord],
    judge: Callable[[str], str] | None = None,
    judge_samples: int = 3,
    judge_uphold: int = 2,
) -> list[tuple[PassKRecord, WellPosednessResult]]:
    return [
        (
            record,
            score_record(
                record,
                judge=judge,
                judge_samples=judge_samples,
                judge_uphold=judge_uphold,
            ),
        )
        for record in records
    ]


def score_record(
    record: PassKRecord,
    judge: Callable[[str], str] | None = None,
    judge_samples: int = 3,
    judge_uphold: int = 2,
) -> WellPosednessResult:
    """Score one record under the isolated c01 contract."""

    statement = record.statement.strip()
    if not statement:
        return WellPosednessResult(
            check_id=CHECK_ID,
            status=ERROR,
            score=SCORE_BY_STATUS[ERROR],
            detail="missing statement",
            signals={"input_error": "missing_statement"},
        )

    structural = structural_hits(statement)
    soft_context = soft_context_signals(statement)
    passk_warnings = passk_context_warnings(record)
    signals = {
        "structural": structural,
        "soft_context": soft_context,
        "passk_context": passk_warnings,
    }

    if structural:
        kinds = ", ".join(f"{name} x{count}" for name, count in structural.items())
        return WellPosednessResult(
            check_id=CHECK_ID,
            status=FLAG,
            score=SCORE_BY_STATUS[FLAG],
            detail=f"statement cites unresolved external material: {kinds}",
            signals=signals,
        )

    if record.is_computed:
        detail = "well-posed by construction (computed provenance)"
        if soft_context:
            detail += "; soft context signals retained for review visibility"
        return WellPosednessResult(
            check_id=CHECK_ID,
            status=PASS,
            score=SCORE_BY_STATUS[PASS],
            detail=detail,
            signals=signals,
        )

    if judge is not None:
        return judge_residue(
            record=record,
            judge=judge,
            signals=signals,
            samples=judge_samples,
            uphold=judge_uphold,
        )

    detail = "semantic residue requires judge or review"
    if soft_context:
        detail += "; soft context signals found"
    return WellPosednessResult(
        check_id=CHECK_ID,
        status=DEFER,
        score=SCORE_BY_STATUS[DEFER],
        detail=detail,
        signals=signals,
    )


def judge_residue(
    record: PassKRecord,
    judge: Callable[[str], str],
    signals: dict,
    samples: int,
    uphold: int,
) -> WellPosednessResult:
    samples = max(1, samples)
    uphold = max(1, min(uphold, samples))
    prompt = _PROMPT.format(
        statement=record.statement,
        truth=record.truth_strings[0] if record.truth_strings else "",
        passk_context=json.dumps(
            {
                "label": record.label,
                "pass_at_k": record.pass_at_k,
                "n_correct": record.n_correct,
                "n_wrong": record.n_wrong,
                "n_degenerate": record.n_degenerate,
            },
            sort_keys=True,
        ),
        soft_context=json.dumps(signals.get("soft_context") or {}, sort_keys=True),
    )
    replies = []
    parsed = []
    usage_total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "samples_with_usage": 0,
    }
    for index in range(samples):
        sample_prompt = f"{prompt}\n\n<<corroboration sample {index + 1} of {samples}>>"
        result = judge(sample_prompt)
        # ``judge`` returns (text, usage) post-patch; the old contract was
        # str-only. Tolerate both shapes so a downstream caller that
        # wraps an older judge still works.
        if isinstance(result, tuple):
            reply, sample_usage = result
        else:
            reply, sample_usage = result, {}
        replies.append(reply)
        if sample_usage:
            usage_total["samples_with_usage"] += 1
            for field in ("input_tokens", "output_tokens",
                          "cache_read_input_tokens", "cache_creation_input_tokens"):
                usage_total[field] += int(sample_usage.get(field) or 0)
        parsed_reply = parse_judge_reply(reply)
        if parsed_reply is not None:
            parsed.append(parsed_reply)

    judge_signals = dict(signals)
    insufficient_context_votes = sum(1 for row in parsed if bool(row.get("insufficient_context")))
    ill_posed = [row for row in parsed if not bool(row.get("determined"))]
    judge_signals["judge"] = {
        "samples_requested": samples,
        "samples_parsed": len(parsed),
        "uphold": uphold,
        "ill_posed_votes": len(ill_posed),
        "insufficient_context_votes": insufficient_context_votes,
        "insufficient_context": insufficient_context_votes >= uphold,
        "votes": _safe_judge_votes(parsed),
        "usage": usage_total,
    }
    if not parsed:
        return WellPosednessResult(
            check_id=CHECK_ID,
            status=ERROR,
            score=SCORE_BY_STATUS[ERROR],
            detail="judge replies not parseable",
            signals=judge_signals,
        )
    if len(ill_posed) >= uphold:
        first = ill_posed[0]
        confidence = _float_or_none(first.get("confidence"))
        return WellPosednessResult(
            check_id=CHECK_ID,
            status=FLAG,
            score=SCORE_BY_STATUS[FLAG],
            detail=f"judge found ill-posed residue ({len(ill_posed)}/{len(parsed)} votes): {first.get('reason', '')}",
            signals={**judge_signals, "judge_confidence": confidence},
        )
    determined_votes = len(parsed) - len(ill_posed)
    return WellPosednessResult(
        check_id=CHECK_ID,
        status=PASS,
        score=SCORE_BY_STATUS[PASS],
        detail=f"judge found determined residue ({determined_votes}/{len(parsed)} votes)",
        signals=judge_signals,
    )


def structural_hits(statement: str) -> dict[str, int]:
    hits = {
        "reference": len(_REFERENCE.findall(statement)),
        "citation": len(_CITATION.findall(statement)),
        "label": len(_DANGLING_LABEL.findall(statement)),
    }
    return {key: value for key, value in hits.items() if value}


def soft_context_signals(statement: str) -> dict[str, int]:
    phrases = [match.group(0).lower() for match in _CONTEXT_PHRASES.finditer(statement)]
    equation_numbers = _LATEX_EQUATION_NUMBER.findall(statement)
    counts = Counter(phrases)
    if equation_numbers:
        counts["bare_equation_number"] += len(equation_numbers)
    return dict(counts)


def passk_context_warnings(record: PassKRecord) -> dict[str, float | int | str]:
    warnings: dict[str, float | int | str] = {
        "n_total": record.n_total,
        "n_decided": record.n_decided,
    }
    if record.n_total == 0:
        warnings["warning"] = "no pass@k trial counts supplied"
        return warnings
    degenerate_share = record.n_degenerate / record.n_total
    warnings["degenerate_share"] = round(degenerate_share, 6)
    if record.n_decided == 0:
        warnings["warning"] = "no committed answers in pass@k trials"
    elif degenerate_share >= 0.5:
        warnings["warning"] = "high degenerate share in pass@k trials"
    return warnings


def status_counts(results: Iterable[WellPosednessResult]) -> dict[str, int]:
    counts = Counter(result.status for result in results)
    return {status: counts.get(status, 0) for status in (PASS, FLAG, DEFER, ERROR)}


def parse_judge_reply(reply: str) -> dict | None:
    cleaned = (reply or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _safe_judge_votes(parsed: list[dict]) -> list[dict]:
    safe = []
    for row in parsed:
        safe.append(
            {
                "determined": bool(row.get("determined")),
                "insufficient_context": bool(row.get("insufficient_context")),
                "reason": str(row.get("reason", "")),
                "confidence": _float_or_none(row.get("confidence")),
            }
        )
    return safe


def _float_or_none(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

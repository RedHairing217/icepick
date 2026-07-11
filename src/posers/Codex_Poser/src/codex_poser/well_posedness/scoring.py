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

# --- context/degeneracy lint patterns (advisory-only; see context_lint_hits) ---
_MISSING_CONTEXT_PHRASES = re.compile(
    r"\b("
    r"a system|an ODE|a PDE|the equation|the stated assumptions|"
    r"as in Lemma|as in Proposition|as specified|"
    r"appropriate conditions|suitable conditions"
    r")\b",
    re.IGNORECASE,
)
_SOURCE_LOCAL_PHRASES = re.compile(
    r"\b(using the notation|defined above|as before)\b",
    re.IGNORECASE,
)
_SYMBOL_TOKEN = r"[A-Za-z\\][A-Za-z0-9_\\']*"
_SYMBOL_DEFINITION = re.compile(
    r"\blet\s+(?P<let_sym>" + _SYMBOL_TOKEN + r")\s+denote"
    r"|\bdefine\s+(?P<def_sym>" + _SYMBOL_TOKEN + r")\b"
    r"|(?P<eq_sym>" + _SYMBOL_TOKEN + r")\s*:=",
    re.IGNORECASE,
)
_SYMBOL_ASK = re.compile(
    r"\b(?:find|compute|determine|what\s+is|evaluate)\b[^.?!\n]{0,60}?"
    r"\b(?P<sym>" + _SYMBOL_TOKEN + r")\b",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")

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


# Rubric v2: opt in via --judge-rubric-version v2 (score_record rubric_version
# kwarg). v1 above is the billed cache-key default and must stay untouched.
_PROMPT_V2 = """You audit a mathematics problem for well-posedness.

Use only the final Statement below. Do not rely on the source paper, the title, or memory of a
canonical theorem. The stored answer is a claim to verify against the statement.

A problem is well-posed only when a strong mathematician, given the statement alone, can derive
the stored answer uniquely up to equivalent form.

Attempt the derivation. The stored answer is a claim to verify: flag unless the statement as
written forces it uniquely (up to equivalent form). Knowing the intended theorem is not
evidence — a recalled or canonical answer that the statement does not force is exactly the
failure mode you are auditing for. Do not supply missing context from memory: if the derivation
needs an equation, hypothesis, normalization, convention, or definition the statement does not
contain, that is a flag, not a gap for you to fill.

Flag the problem when any of these hold:
- an equation, object, hypothesis, convention, normalization, or symbol needed for the answer is
  missing or paper-local;
- more than one non-equivalent answer could satisfy the question;
- the question asks for a sharp, optimal, exact, largest, smallest, or canonical value but the
  statement only gives an existence claim or a bound;
- the stored answer is merely defined or displayed in the statement and the item is
  transcription/definition recall rather than a mathematical task;
- the stored answer contradicts the literal statement, even if it resembles a known theorem.

These are NOT flags: difficulty; advanced but field-standard terminology or notation (an object
a strong mathematician can define and use without the paper — e.g. a classical named operator —
is standard even if this statement does not define it); an answer given in one of several
equivalent forms.

Statement:
{statement}

Stored answer:
{truth}

Pass@k context:
{passk_context}

Soft context signals from code:
{soft_context}

Reply with a JSON object only, keys: determined (true or false), insufficient_context (true or
false), reason (one sentence), confidence (0 to 1)."""


# Rubric v3: opt in via --judge-rubric-version v3. Bounded-sketch redesign after
# v2 failed the sentinel hard gate 4/7 (execution_validation_20260711T060641Z):
# flag only on a NAMEABLE missing ingredient; standardness is an executable
# self-test; replies stay JSON-only by instruction (v2's derivation prose
# truncated at the judge token cap).
_PROMPT_V3 = """You audit a mathematics problem for well-posedness.

Use only the final Statement below. Do not rely on the source paper, the title, or memory of a
canonical theorem. The stored answer is a claim to verify against the statement.

A problem is well-posed only when the statement alone forces the stored answer uniquely (up to
equivalent form). Knowing the intended theorem is not evidence — a recalled or canonical answer
that the statement does not force is exactly the failure mode you are auditing for.

Sketch — do not write out — the derivation that would take a strong mathematician from the
statement to the stored answer: identify the ingredients that pin the answer down (equations,
hypotheses, conventions, normalizations, definitions). Difficulty is not the test; a long or
hard derivation whose ingredients are all present is well-posed.

An ingredient does not count as missing when it is standard: a strong mathematician could write
its definition from the name alone, without this paper. Classical named operators (e.g. the
fractional Laplacian, a Laplace-Beltrami operator), standard weak or variational formulations,
and field-standard conventions and scalings are standard even when this statement does not
define them. If you can supply the definition yourself, supply it and continue the sketch.

Flag the problem only when you can NAME a specific missing ingredient or a specific defect, in
one of these forms:
- an equation, object, hypothesis, convention, normalization, or definition that the derivation
  needs, that the statement does not contain, and that is paper-local rather than standard;
- more than one non-equivalent answer satisfies the question as posed (name the second reading);
- the question asks for a sharp, optimal, exact, largest, smallest, or canonical value but the
  statement only supports an existence claim or a one-sided bound;
- the stored answer is merely restated or displayed in the statement (transcription or
  definition recall, not a mathematical task);
- the stored answer contradicts the literal statement.
If you cannot name the missing ingredient or defect, do not flag.

These are NOT flags: difficulty or length of the derivation; advanced but standard terminology
or notation; an answer given in one of several equivalent forms.

Statement:
{statement}

Stored answer:
{truth}

Pass@k context:
{passk_context}

Soft context signals from code:
{soft_context}

Set insufficient_context to true only when the named missing ingredient is context the source
paper had and this statement dropped. Reply with a single JSON object only — no derivation text,
no markdown, keys: determined (true or false), insufficient_context (true or false), reason (one
sentence naming the missing ingredient or defect, or the ingredients that force the answer),
confidence (0 to 1)."""


def score_records(
    records: Iterable[PassKRecord],
    judge: Callable[[str], str] | None = None,
    judge_samples: int = 3,
    judge_uphold: int = 2,
    *,
    rubric_version: str = "v1",
    context_lint_mode: str = "off",
) -> list[tuple[PassKRecord, WellPosednessResult]]:
    return [
        (
            record,
            score_record(
                record,
                judge=judge,
                judge_samples=judge_samples,
                judge_uphold=judge_uphold,
                rubric_version=rubric_version,
                context_lint_mode=context_lint_mode,
            ),
        )
        for record in records
    ]


def score_record(
    record: PassKRecord,
    judge: Callable[[str], str] | None = None,
    judge_samples: int = 3,
    judge_uphold: int = 2,
    *,
    rubric_version: str = "v1",
    context_lint_mode: str = "off",
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
    if context_lint_mode == "advisory":
        # Computed once, attached to the shared ``signals`` dict so every
        # exit path below (including the judge path's copy) carries it.
        truth_text = record.truth_strings[0] if record.truth_strings else ""
        signals["context_lint"] = {
            "mode": "advisory",
            **context_lint_hits(statement, truth_text),
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
            rubric_version=rubric_version,
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
    *,
    rubric_version: str = "v1",
) -> WellPosednessResult:
    samples = max(1, samples)
    uphold = max(1, min(uphold, samples))
    if rubric_version == "v2":
        template = _PROMPT_V2
    elif rubric_version == "v3":
        template = _PROMPT_V3
    else:
        template = _PROMPT
    prompt = template.format(
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
        "reasoning_tokens": 0,
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
            for field in ("input_tokens", "output_tokens", "reasoning_tokens",
                          "cache_read_input_tokens", "cache_creation_input_tokens"):
                usage_total[field] += int(sample_usage.get(field) or 0)
        parsed_reply = parse_judge_reply(reply)
        if parsed_reply is not None:
            parsed.append(parsed_reply)

    judge_signals = dict(signals)
    insufficient_context_votes = sum(1 for row in parsed if bool(row.get("insufficient_context")))
    ill_posed = [row for row in parsed if not bool(row.get("determined"))]
    judge_signals["judge"] = {
        "rubric_version": rubric_version,
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


def context_lint_hits(statement: str, truth: str) -> dict:
    """Advisory-only context/degeneracy lint (see --context-lint-mode).

    Never influences status or score. Returns
    ``{"classes": {name: [snippets]}, "hit_count": n}`` with empty classes
    omitted.
    """
    classes: dict[str, list[str]] = {}

    missing_context = [match.group(0) for match in _MISSING_CONTEXT_PHRASES.finditer(statement)]
    if missing_context:
        classes["missing_context_placeholder"] = missing_context

    source_local = [match.group(0) for match in _SOURCE_LOCAL_PHRASES.finditer(statement)]
    if source_local:
        classes["source_local_language"] = source_local

    defines_then_asks = _defines_then_asks_symbols(statement)
    if defines_then_asks:
        classes["defines_then_asks"] = defines_then_asks

    # "Verbatim" is exact-case by design; short answers (< 6 chars) are
    # excluded since coincidental recall of e.g. "0" or "true" is common.
    normalized_truth = _WHITESPACE.sub(" ", truth).strip()
    if len(normalized_truth) >= 6:
        normalized_statement = _WHITESPACE.sub(" ", statement)
        if normalized_truth in normalized_statement:
            classes["verbatim_formula_recall"] = [normalized_truth]

    hit_count = sum(len(snippets) for snippets in classes.values())
    return {"classes": classes, "hit_count": hit_count}


def _defines_then_asks_symbols(statement: str) -> list[str]:
    """Conservative define-then-ask heuristic: a symbol introduced via
    ``<sym> :=``/``let <sym> denote``/``define <sym>`` that reappears inside
    a later find/compute/determine/what-is/evaluate clause.
    """
    defined_at: dict[str, int] = {}
    for match in _SYMBOL_DEFINITION.finditer(statement):
        symbol = match.group("let_sym") or match.group("def_sym") or match.group("eq_sym")
        if symbol and symbol not in defined_at:
            defined_at[symbol] = match.end()

    hits = []
    for match in _SYMBOL_ASK.finditer(statement):
        symbol = match.group("sym")
        define_end = defined_at.get(symbol)
        if define_end is not None and match.start() > define_end and symbol not in hits:
            hits.append(symbol)
    return hits


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

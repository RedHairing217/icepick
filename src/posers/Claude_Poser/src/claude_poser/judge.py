"""c01 judge with 3-sample, 2-of-3 majority corroboration.

Supports two API backends, selected by cfg.judge_provider:
- "anthropic" : Messages API via the anthropic SDK (optional install).
- "openai"    : Chat Completions API via stdlib urllib. Honours
                cfg.openai_base_url, so OpenAI-compatible local servers
                (LM Studio, Ollama, vLLM, Together, etc.) work by pointing
                that URL at the server's /v1 endpoint. Reasoning-family
                models (gpt-5.x, o-series) are sent the reasoning-model
                parameter surface (reasoning_effort via
                cfg.openai_reasoning_effort / OPENAI_REASONING_EFFORT,
                max_completion_tokens, no temperature); other models keep
                the historical wire format byte-for-byte.

Each judge sample returns:
    {"verdict": "pass" | "flag", "insufficient_context": bool, "reason": str,
     "derived_answer": str | null}

Corroboration rules:
- A 'flag' is upheld only on a majority of samples.
- 'insufficient_context' is upheld on a majority and short-circuits to a
  dedicated status downstream.
- If the active provider is unreachable, samples come back as 'error' and
  the corroborated result is 'defer'. ``defer_reason`` distinguishes a
  quorum broken by sample errors ("judge_errors") from a genuine split
  among substantive votes ("split") so downstream can retry the former
  instead of silently discarding the record.
- Live calls that come back 'error' (parse failure, transient API fault)
  are retried up to cfg.judge_error_retries times before the error vote
  stands. The stage-3 kill analysis (2026-07-04) found 8% of gpt-4.1-mini
  samples were bad-JSON errors that converted to kills.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from .config import WellposedConfig
from .judge_cache import JudgeCache


JUDGE_SYSTEM = (
    "You audit a mathematics problem statement for well-posedness. "
    "Decide whether the statement, on its own, determines a single answer "
    "for a competent reader with standard mathematical background. Do not "
    "solve the problem — only judge whether it is answerable.\n\n"
    "Flag the statement (verdict = \"flag\") in any of these cases:\n"
    "  - it references external material that is not present (a section, "
    "theorem, lemma, equation, figure, or prior problem by number or name);\n"
    "  - it uses notation or symbols that are never defined in the statement "
    "and are not standard mathematical convention (e.g. paper-specific "
    "functions like l(F D_{2^k}), operators, or named objects introduced "
    "only in the source paper);\n"
    "  - a definition needed to interpret the question is missing;\n"
    "  - the answer is otherwise underdetermined by the given text.\n\n"
    "Pass the statement (verdict = \"pass\") if a competent reader could, in "
    "principle, produce a single correct answer from the text alone — even if "
    "the problem is hard or requires substantial computation. Hard is not "
    "ill-posed.\n\n"
    "Set insufficient_context = true when the flag reason is that the "
    "problem relies on external material or undefined paper-specific notation "
    "the reader cannot recover. This is a stronger claim than a generic flag: "
    "it means the problem cannot be salvaged without the missing context. "
    "Set it to false when the statement is self-contained but ill-posed for "
    "other reasons (ambiguous, contradictory, underdetermined by choice of "
    "input rather than by missing definition).\n\n"
    "When your verdict is \"pass\" and you can state the final answer "
    "concisely, put it in derived_answer (a short expression or value, no "
    "working). If stating it would need extensive computation you have not "
    "done, or your verdict is \"flag\", set derived_answer to null. Do not "
    "guess: only fill derived_answer when you are confident. This field is "
    "audited against the record's stored answer — a pass whose derived "
    "answer contradicts the stored answer is routed to human review.\n\n"
    "Respond with one JSON object only, no prose, matching:\n"
    '{"verdict": "pass" | "flag", "insufficient_context": true | false, '
    '"reason": "<one short sentence>", "derived_answer": "<answer>" | null}'
)

# Bump when the reply schema or system prompt changes meaningfully. The
# marker rides in the user prompt because the judge cache keys on
# (provider, model, prompt, sample_id) — NOT the system prompt — so a schema
# change must roll the key or stale replies would be served under the new
# schema. v2: added derived_answer (answer-consistency audit).
PROMPT_VERSION = "v2"


def build_prompt(statement: str) -> str:
    return (
        "Problem statement:\n"
        "----\n"
        f"{statement}\n"
        "----\n"
        f"Reply with the JSON object only. (judge schema {PROMPT_VERSION})"
    )


@dataclass
class JudgeSample:
    sample_id: int
    verdict: str  # "pass" | "flag" | "error"
    insufficient_context: bool
    reason: str
    derived_answer: Optional[str] = None  # judge's own answer when passing (schema v2)
    usage: Optional[dict] = None  # raw {input_tokens, output_tokens, ...} from the API

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "verdict": self.verdict,
            "insufficient_context": self.insufficient_context,
            "reason": self.reason,
            "derived_answer": self.derived_answer,
            "usage": self.usage,
        }


@dataclass
class JudgeOutcome:
    samples: list[JudgeSample]
    majority_verdict: str  # "pass" | "flag" | "insufficient_context" | "defer"
    wellposed_votes: int
    flag_votes: int
    ic_votes: int
    error_votes: int
    provider: str
    model: str
    # Only set when majority_verdict == "defer":
    #   "judge_errors" — error votes broke the quorum (no side could reach
    #                    uphold even with unanimity among parsed samples);
    #   "split"        — substantive votes genuinely disagree.
    defer_reason: Optional[str] = None

    def usage_total(self) -> dict:
        """Sum every sample's usage. Cache hits and errors contribute zero."""
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "samples_with_usage": 0,
        }
        for s in self.samples:
            if not s.usage:
                continue
            totals["samples_with_usage"] += 1
            for field in ("input_tokens", "output_tokens", "reasoning_tokens",
                          "cache_read_input_tokens", "cache_creation_input_tokens"):
                totals[field] += int(s.usage.get(field) or 0)
        return totals

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "majority_verdict": self.majority_verdict,
            "wellposed_votes": self.wellposed_votes,
            "flag_votes": self.flag_votes,
            "insufficient_context_votes": self.ic_votes,
            "error_votes": self.error_votes,
            "defer_reason": self.defer_reason,
            "samples": [s.to_dict() for s in self.samples],
            "usage": self.usage_total(),
        }


# --------------------------------------------------------------------------- #
# Reply parsing
# --------------------------------------------------------------------------- #


# A backslash not opening a legal JSON escape — models judging LaTeX-heavy
# statements routinely emit raw TeX macros ("\alpha", "\mathcal{L}") inside
# JSON strings, which json.loads rejects as "Invalid \escape". Doubling the
# offending backslash preserves the intended text and is a no-op on valid
# JSON (the pattern cannot match inside a legal escape).
_BAD_JSON_ESCAPE = re.compile(r'\\(?![\\"/bfnrtu])')


def _parse_reply(text: str) -> dict:
    """Tolerant JSON extraction from the model's reply."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return {"verdict": "error", "insufficient_context": False, "reason": "no JSON"}
    candidate = text[a : b + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as first_err:
        try:
            obj = json.loads(_BAD_JSON_ESCAPE.sub(r"\\\\", candidate))
        except json.JSONDecodeError:
            return {"verdict": "error", "insufficient_context": False,
                    "reason": f"bad JSON: {first_err}"}
    verdict = str(obj.get("verdict", "")).lower()
    if verdict not in ("pass", "flag"):
        verdict = "error"
    derived = obj.get("derived_answer")
    return {
        "verdict": verdict,
        "insufficient_context": bool(obj.get("insufficient_context", False)),
        "reason": str(obj.get("reason", "")),
        "derived_answer": str(derived) if derived not in (None, "") else None,
    }


# --------------------------------------------------------------------------- #
# Provider callers
# --------------------------------------------------------------------------- #


def _err(reason: str) -> dict:
    return {"verdict": "error", "insufficient_context": False, "reason": reason}


# OpenAI's reasoning family (gpt-5.x, o-series) takes a different parameter
# surface on chat completions, verified live against gpt-5.5 on 2026-07-06:
# max_tokens is rejected (use max_completion_tokens), any non-default
# temperature is rejected, and thinking depth is set via reasoning_effort.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Reasoning tokens bill as completion tokens and count against the cap. The
# 400-token budget that fits the JSON verdict on non-reasoning models can be
# consumed entirely by thinking on a hard statement, returning an empty
# message — reasoning models get a much larger cap instead.
_REASONING_MAX_COMPLETION_TOKENS = 4000

# High-effort reasoning regularly exceeds the 30s default timeout on hard
# statements (4/25 timed out in the 2026-07-06 stage-3 revalidation).
# Timeouts become error votes, and systematic errors break quorum into
# defer — so the reasoning branch floors the timeout instead. An explicit
# cfg.judge_timeout_s above the floor is still honoured.
_REASONING_MIN_TIMEOUT_S = 120.0


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(_REASONING_MODEL_PREFIXES)


def _extract_anthropic_usage(msg) -> dict:
    """Pull a normalized usage dict from an Anthropic SDK Message."""
    usage = getattr(msg, "usage", None)
    if usage is None:
        return {}
    out = {}
    for attr in ("input_tokens", "output_tokens",
                 "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = getattr(usage, attr, None)
        if value is not None:
            out[attr] = value
    return out


def _extract_openai_usage(payload: dict) -> dict:
    """Pull a normalized usage dict from an OpenAI chat-completions payload.

    OpenAI uses ``prompt_tokens``/``completion_tokens``; we rename to the
    Anthropic-style ``input_tokens``/``output_tokens`` for uniform rollup.
    """
    usage = payload.get("usage") or {}
    out = {}
    if "prompt_tokens" in usage:
        out["input_tokens"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        out["output_tokens"] = usage["completion_tokens"]
    # Reasoning models report thinking spend under completion_tokens_details;
    # it is already included in completion_tokens but recorded separately so
    # cost audits can see where the output budget went.
    details = usage.get("completion_tokens_details") or {}
    if details.get("reasoning_tokens"):
        out["reasoning_tokens"] = details["reasoning_tokens"]
    return out


def _call_anthropic_once(cfg: WellposedConfig, prompt: str) -> dict:
    """Anthropic Messages API call. Returns a parsed reply dict; never raises."""
    if not cfg.anthropic_api_key:
        return _err("no anthropic api key")
    try:
        import anthropic  # type: ignore
    except ImportError:
        return _err("anthropic SDK not installed (pip install anthropic)")
    try:
        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key, timeout=cfg.judge_timeout_s)
        msg = client.messages.create(
            model=cfg.resolve_model(),
            max_tokens=400,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "")
        reply = _parse_reply(text)
        usage = _extract_anthropic_usage(msg)
        if usage:
            reply["usage"] = usage
        return reply
    except Exception as e:  # noqa: BLE001 — judge errors must not crash the gate
        return _err(f"anthropic api error: {e!r}")


def _call_openai_once(cfg: WellposedConfig, prompt: str) -> dict:
    """OpenAI Chat Completions API call via stdlib urllib.

    Works against api.openai.com or any OpenAI-compatible server (LM Studio,
    Ollama, vLLM, Together AI, etc.) — set OPENAI_BASE_URL or
    --openai-base-url to point elsewhere. Most local servers accept any
    bearer token; the real OpenAI API requires a valid key.
    """
    if not cfg.openai_api_key:
        return _err("no openai api key")
    base = cfg.openai_base_url.rstrip("/")
    url = f"{base}/chat/completions"
    model = cfg.resolve_model()
    payload_dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    timeout_s = cfg.judge_timeout_s
    if _is_reasoning_model(model):
        payload_dict["max_completion_tokens"] = _REASONING_MAX_COMPLETION_TOKENS
        payload_dict["reasoning_effort"] = cfg.openai_reasoning_effort
        timeout_s = max(timeout_s, _REASONING_MIN_TIMEOUT_S)
    else:
        # Wire format kept byte-identical for non-reasoning models so requests
        # (and the disk caches keyed on provider/model/prompt) are unchanged.
        payload_dict["max_tokens"] = 400
        payload_dict["temperature"] = 1.0
    body = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {cfg.openai_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:  # noqa: BLE001
            detail = ""
        return _err(f"openai http {e.code}: {detail}")
    except urllib.error.URLError as e:
        return _err(f"openai url error: {e.reason!r}")
    except Exception as e:  # noqa: BLE001
        return _err(f"openai api error: {e!r}")
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        return _err(f"openai unexpected payload: {e!r}")
    reply = _parse_reply(text)
    usage = _extract_openai_usage(payload)
    if usage:
        reply["usage"] = usage
    return reply


def get_provider_caller(provider: str) -> Callable[[WellposedConfig, str], dict]:
    """Return the per-provider caller. Resolved at call time, so monkeypatching
    `judge._call_anthropic_once` / `judge._call_openai_once` works in tests."""
    if provider == "anthropic":
        return _call_anthropic_once
    if provider == "openai":
        return _call_openai_once
    raise ValueError(f"unknown judge_provider {provider!r}")


# --------------------------------------------------------------------------- #
# Corroboration + entry point
# --------------------------------------------------------------------------- #


def _corroborate(
    samples: list[JudgeSample],
    cfg: WellposedConfig,
    *,
    provider: str,
    model: str,
) -> JudgeOutcome:
    n = len(samples)
    wellposed_votes = sum(1 for s in samples if s.verdict == "pass" and not s.insufficient_context)
    flag_votes = sum(1 for s in samples if s.verdict == "flag" and not s.insufficient_context)
    ic_votes = sum(1 for s in samples if s.insufficient_context)
    error_votes = sum(1 for s in samples if s.verdict == "error")

    uphold = cfg.judge_uphold
    if ic_votes >= uphold:
        majority = "insufficient_context"
    elif flag_votes >= uphold:
        majority = "flag"
    elif wellposed_votes >= uphold:
        majority = "pass"
    else:
        majority = "defer"

    # Quorum broken by errors: even unanimity among the parsed samples could
    # not reach uphold. Overrides any accidental majority among the rest.
    quorum_broken = error_votes > n - uphold
    if quorum_broken:
        majority = "defer"

    defer_reason: Optional[str] = None
    if majority == "defer":
        defer_reason = "judge_errors" if quorum_broken else "split"

    return JudgeOutcome(
        samples=samples,
        majority_verdict=majority,
        wellposed_votes=wellposed_votes,
        flag_votes=flag_votes,
        ic_votes=ic_votes,
        error_votes=error_votes,
        provider=provider,
        model=model,
        defer_reason=defer_reason,
    )


def judge_wellposed(
    statement: str,
    cfg: WellposedConfig,
    cache: Optional[JudgeCache] = None,
    caller: Optional[Callable[[WellposedConfig, str], dict]] = None,
) -> JudgeOutcome:
    """Run cfg.judge_samples independent calls and corroborate.

    Cache key includes the provider and the resolved model so an Anthropic
    reply cannot be served for an OpenAI call (or vice versa), and a model
    switch cannot reuse stale replies.
    """
    prompt = build_prompt(statement)
    provider = cfg.judge_provider
    model = cfg.resolve_model()
    dispatched = caller if caller is not None else get_provider_caller(provider)

    samples: list[JudgeSample] = []
    for sample_id in range(cfg.judge_samples):
        reply: Optional[dict] = None
        if cache is not None:
            cached = cache.get(provider, model, prompt, sample_id)
            if cached is not None:
                reply = cached["reply"]
        if reply is None:
            reply = dispatched(cfg, prompt)
            # Error replies (parse failure, transient API fault) get a bounded
            # number of fresh attempts before the error vote stands — a single
            # unlucky sample must not decide the record's fate.
            for _ in range(cfg.judge_error_retries):
                if reply.get("verdict") != "error":
                    break
                reply = dispatched(cfg, prompt)
            if cache is not None and reply.get("verdict") != "error":
                cache.put(provider, model, prompt, sample_id, reply)
        samples.append(
            JudgeSample(
                sample_id=sample_id,
                verdict=reply.get("verdict", "error"),
                insufficient_context=bool(reply.get("insufficient_context", False)),
                reason=str(reply.get("reason", "")),
                derived_answer=reply.get("derived_answer") or None,
                usage=reply.get("usage") or None,
            )
        )
    return _corroborate(samples, cfg, provider=provider, model=model)

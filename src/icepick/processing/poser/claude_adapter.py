"""Claude_Poser adapter.

Drives ``claude-poser score`` as a subprocess. icepick injects ``uid``
on every input record before invoking, then reads the poser's JSON
output and normalises it onto ``PoserVerdict``.

Provider routing:

- ``provider=anthropic`` -> ``--provider anthropic --anthropic-key-file <anthro_key.env>``
- ``provider=openai``    -> ``--provider openai    --openai-key-file <openai_key.env>``

claude-poser refuses to load the OTHER provider's key file when
``--provider X`` is set, so credentials stay segregated even when both
providers run in parallel.

Field renaming from the discovery synthesis:
    wellposed_status   -> verdict_status (after canonical mapping)
    wellposed_score    -> verdict_score
    wellposed_votes / judge_majority / code_hit_count -> verdict_signals
    insufficient_context -> verdict_status='defer', verdict_detail.original_status='insufficient_context'

Claude_Poser was originally distributed as ``Anthro_Poser`` (CLI
``anthro-poser``); the rename happened in late 2026.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from icepick.processing.poser.base import (
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserRequest,
    PoserRunResult,
    PoserVerdict,
    inject_uid,
)
from icepick.processing.poser.config import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    Combo,
)

_STATUS_MAP = {
    "pass": STATUS_WELL_POSED,
    "flag": STATUS_ILL_POSED,
    "defer": STATUS_DEFER,
    # insufficient_context is a CONFIRMED verdict (score 0.0 upstream) meaning
    # the statement relies on external material or undefined paper-specific
    # notation the reader cannot recover. Treating it as defer would let
    # downstream retry policies re-judge a decision we already made. Map to
    # ill-posed so it exits the pipeline.
    "insufficient_context": STATUS_ILL_POSED,
}


class ClaudePoserAdapter:
    build = "claude"

    def plan(self, records: list, cfg, combo: Combo, work_dir: Path) -> PoserRequest:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        slug = combo.slug()
        input_path = work_dir / f"{slug}_input.jsonl"
        output_path = work_dir / f"{slug}_verdicts.json"
        cache_path = work_dir / f"{slug}_judge_cache.jsonl"

        prepared = inject_uid(records)
        with input_path.open("w", encoding="utf-8") as fh:
            for record in prepared:
                fh.write(json.dumps(record) + "\n")

        argv = [
            cfg.claude.cli_path,
            "score",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--mode",
            cfg.mode,
            "--provider",
            combo.provider,
            "--judge-samples",
            str(cfg.judge_samples),
            "--judge-uphold",
            str(cfg.judge_uphold),
            "--judge-cache",
            str(cache_path),
        ]

        if cfg.claude.judge_model:
            argv.extend(["--judge-model", cfg.claude.judge_model])

        if cfg.enable_judge_tier:
            argv.append("--judge")

        # Extracted-provenance judge policy. Only claude-poser accepts this
        # flag; codex-poser ignores it. Passed unconditionally so operator
        # intent is honoured whether or not --judge is on (the flag becomes
        # a no-op if the judge tier is off).
        argv.extend(["--extracted-judge-policy", cfg.extracted_judge_policy])

        # Provider-segregated key file. claude-poser refuses to load the
        # other provider's file when --provider is set, by design.
        if combo.provider == PROVIDER_ANTHROPIC and cfg.anthropic_key_file:
            argv.extend(["--anthropic-key-file", str(cfg.anthropic_key_file)])
        elif combo.provider == PROVIDER_OPENAI and cfg.openai_key_file:
            argv.extend(["--openai-key-file", str(cfg.openai_key_file)])

        if cfg.mode == "flow_testing" and cfg.calibration_sheet:
            argv.extend(["--calibration-sheet", str(cfg.calibration_sheet)])

        argv.extend(cfg.claude.extra_args)

        return PoserRequest(
            argv=argv,
            env={},
            input_path=input_path,
            output_path=output_path,
            cache_path=cache_path,
            poser_name=combo.key(),
        )

    def run(self, request: PoserRequest) -> PoserRunResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                request.argv,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            return PoserRunResult(
                exit_code=127,
                stdout="",
                stderr=f"poser binary not found: {exc}",
                output_path=request.output_path,
                wall_clock_seconds=time.monotonic() - start,
            )
        return PoserRunResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            output_path=request.output_path,
            wall_clock_seconds=time.monotonic() - start,
        )

    def normalise(self, raw_output_path: Path, input_uids: list, *, combo: Combo) -> list:
        raw_output_path = Path(raw_output_path)
        if not raw_output_path.exists():
            return [
                _error_verdict(uid, combo, "claude output file missing")
                for uid in input_uids
            ]
        try:
            payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return [
                _error_verdict(uid, combo, f"claude output parse failed: {exc.msg}")
                for uid in input_uids
            ]

        model = payload.get("judge_model") or payload.get("model") or ""
        rows = payload.get("records") or payload.get("verdicts") or []

        verdicts: list = []
        seen_uids: set = set()
        for row in rows:
            uid = row.get("uid") or ""
            if not uid:
                continue
            seen_uids.add(uid)
            verdicts.append(_normalise_row(row, model, combo))

        for uid in input_uids:
            if uid not in seen_uids:
                verdicts.append(_error_verdict(uid, combo, "uid missing from claude output"))

        return verdicts


def _normalise_row(row: dict, model: str, combo: Combo) -> PoserVerdict:
    raw_status = (row.get("wellposed_status") or "").lower()
    status = _STATUS_MAP.get(raw_status, STATUS_DEFER)
    score = _coerce_float(row.get("wellposed_score"), default=0.5)
    detail: dict = {"original_status": raw_status, "provider": combo.provider}
    if "tier" in row:
        detail["tier"] = row["tier"]

    judge_block = row.get("judge") or {}
    if not isinstance(judge_block, dict):
        judge_block = {}

    # A defer whose quorum was broken by judge sample errors (bad JSON,
    # transient API fault) is an infrastructure failure, not a verdict.
    # Map it to STATUS_ERROR so the cascade's per-uid retry machinery
    # re-runs it instead of silently discarding the record. The stage-3
    # kill analysis (2026-07-04) found 4 of 40 kills were exactly this.
    if status == STATUS_DEFER and judge_block.get("defer_reason") == "judge_errors":
        status = STATUS_ERROR
        detail["error_reason"] = "judge quorum broken by sample errors"

    if judge_block.get("defer_reason"):
        detail["defer_reason"] = judge_block["defer_reason"]
    if judge_block.get("answer_consistency"):
        detail["answer_consistency"] = judge_block["answer_consistency"]
    if row.get("review_flags"):
        detail["review_flags"] = list(row["review_flags"])

    signals = {
        k: row[k]
        for k in ("wellposed_votes", "flag_votes", "insufficient_context_votes",
                  "error_votes", "judge_majority", "code_hit_count")
        if k in row
    }
    # Token usage lives under judge.usage (set by Claude_Poser per record).
    usage = judge_block.get("usage")
    if usage:
        signals["usage"] = usage
    return PoserVerdict(
        uid=row["uid"],
        source=row.get("source", ""),
        verdict_status=status,
        verdict_score=score,
        poser_name=combo.key(),
        poser_model=model,
        verdict_detail=detail,
        verdict_signals=signals,
        raw_payload=row,
    )


def _error_verdict(uid: str, combo: Combo, reason: str) -> PoserVerdict:
    return PoserVerdict(
        uid=uid,
        source="",
        verdict_status=STATUS_ERROR,
        verdict_score=0.0,
        poser_name=combo.key(),
        poser_model="",
        verdict_detail={"error_reason": reason, "provider": combo.provider},
    )


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

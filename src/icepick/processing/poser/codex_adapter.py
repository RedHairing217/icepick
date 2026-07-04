"""Codex_Poser adapter.

Drives ``codex-poser score`` as a subprocess. Same uid injection and
canonical normalisation as the Claude adapter, with these codex-poser
specifics handled here:

- provider flag is ``--judge-provider`` (not ``--provider``)
- secrets flag is ``--key-env`` (single path; codex picks the right
  defaults of ``../anthro_key.env`` or ``../openai_key.env``, but
  icepick always passes the path explicitly so the default never silently
  activates)
- no ``--calibration-sheet``; flow_testing disables judge entirely
- output JSON embeds metadata; no sidecar

Codex_Poser was originally distributed as ``GPT_Poser`` (CLI
``gpt-poser``); the rename happened in late 2026.
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
    "error": STATUS_ERROR,
}


class CodexPoserAdapter:
    build = "codex"

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
            cfg.codex.cli_path,
            "score",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--mode",
            cfg.mode,
            "--judge-provider",
            combo.provider,
            "--judge-samples",
            str(cfg.judge_samples),
            "--judge-uphold",
            str(cfg.judge_uphold),
            "--judge-cache",
            str(cache_path),
            "--format",
            "json",
        ]

        if cfg.codex.judge_model:
            argv.extend(["--judge-model", cfg.codex.judge_model])

        # codex-poser refuses --judge under flow_testing; config.validate()
        # has already rejected that combination for codex combos, but guard here too.
        if cfg.enable_judge_tier and cfg.mode == "production":
            argv.append("--judge")

        # codex-poser takes a single --key-env; icepick picks the right
        # provider's key file and passes it explicitly.
        key_file = None
        if combo.provider == PROVIDER_ANTHROPIC:
            key_file = cfg.anthropic_key_file
        elif combo.provider == PROVIDER_OPENAI:
            key_file = cfg.openai_key_file
        if key_file:
            argv.extend(["--key-env", str(key_file)])

        argv.extend(cfg.codex.extra_args)

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
                _error_verdict(uid, combo, "codex output file missing")
                for uid in input_uids
            ]
        try:
            payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return [
                _error_verdict(uid, combo, f"codex output parse failed: {exc.msg}")
                for uid in input_uids
            ]

        model = (
            payload.get("parameters", {}).get("judge_model")
            or payload.get("judge_model")
            or payload.get("model")
            or ""
        )
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
                verdicts.append(_error_verdict(uid, combo, "uid missing from codex output"))
        return verdicts


def _normalise_row(row: dict, model: str, combo: Combo) -> PoserVerdict:
    raw_status = (row.get("well_posedness_status") or row.get("verdict_status") or "").lower()
    status = _STATUS_MAP.get(raw_status, STATUS_DEFER)
    score = _coerce_float(
        row.get("well_posedness_score", row.get("verdict_score")),
        default=0.5,
    )
    detail: dict = {"original_status": raw_status, "provider": combo.provider}
    if "well_posedness_check" in row:
        detail["check"] = row["well_posedness_check"]
    if "well_posedness_detail" in row:
        detail["detail"] = row["well_posedness_detail"]
    signals = row.get("signals") or {}
    if isinstance(signals, list):
        signals = {"signals": signals}
    # Token usage lives under signals.judge.usage (set by Codex_Poser per record).
    if isinstance(signals, dict):
        judge_block = signals.get("judge")
        if isinstance(judge_block, dict) and judge_block.get("usage"):
            signals = dict(signals)
            signals["usage"] = judge_block["usage"]
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

"""Write well-posedness scores to JSON or CSV.

Two output shapes:
  - JSON: {"run": {...}, "records": [...]}
  - CSV : one row per record with score, status, tier, and key flags.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .config import WellposedConfig


CSV_COLUMNS = (
    "uid",
    "rid",
    "source",
    "provenance",
    "tier",
    "wellposed_status",
    "wellposed_score",
    "judge_majority",
    "wellposed_votes",
    "flag_votes",
    "insufficient_context_votes",
    "error_votes",
    "code_hit_count",
)


def _flatten(result: dict) -> dict:
    judge = result.get("judge") or {}
    return {
        "uid": result.get("uid"),
        "rid": result.get("rid"),
        "source": result.get("source"),
        "provenance": result.get("provenance"),
        "tier": result.get("tier"),
        "wellposed_status": result.get("wellposed_status"),
        "wellposed_score": result.get("wellposed_score"),
        "judge_majority": judge.get("majority_verdict"),
        "wellposed_votes": judge.get("wellposed_votes"),
        "flag_votes": judge.get("flag_votes"),
        "insufficient_context_votes": judge.get("insufficient_context_votes"),
        "error_votes": judge.get("error_votes"),
        "code_hit_count": len(result.get("code_hits") or []),
    }


def _run_meta(
    cfg: WellposedConfig,
    inputs: Sequence[str],
    results: Sequence[dict],
) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["wellposed_status"]] = counts.get(r["wellposed_status"], 0) + 1
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module": "claude_poser",
        "check": "c01_wellposed",
        "processor_mode": cfg.processor_mode,
        "calibration_sheet": cfg.calibration_sheet,
        "calibration_replay": cfg.processor_mode == "flow_testing",
        "inputs": list(inputs),
        "input_count": len(results),
        "counts": counts,
        "parameters": cfg.echo(),
    }


def write_json(
    path: str | Path,
    cfg: WellposedConfig,
    inputs: Sequence[str],
    results: Sequence[dict],
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run": _run_meta(cfg, inputs, results), "records": list(results)}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def write_csv(
    path: str | Path,
    cfg: WellposedConfig,
    inputs: Sequence[str],
    results: Iterable[dict],
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialised = list(results)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in materialised:
            writer.writerow(_flatten(r))
    # Pair the CSV with a sidecar summary so run metadata is not lost.
    sidecar = p.with_suffix(p.suffix + ".summary.json")
    sidecar.write_text(json.dumps(_run_meta(cfg, inputs, materialised), indent=2), encoding="utf-8")
    return p

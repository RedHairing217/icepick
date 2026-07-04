"""Calibration replay for flow_testing mode.

In flow_testing mode every judge call must be served from preprocessed
calibration data, never from the real Anthropic API. The calibration sheet
is a JSONL file keyed by (uid, sample_id) that returns the exact reply shape
the production judge would have returned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .config import WellposedConfig


def load_sheet(path: str | Path) -> dict[tuple[str, int], dict]:
    """Build a lookup keyed by (uid, sample_id)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"calibration sheet not found: {p}")
    table: dict[tuple[str, int], dict] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("section") != "judge":
                continue
            uid = entry.get("uid")
            sample_id = entry.get("sample_id", 0)
            if uid is None:
                continue
            table[(str(uid), int(sample_id))] = entry.get("reply") or {}
    if not table:
        raise ValueError(f"calibration sheet has no judge entries: {p}")
    return table


def make_replay_caller(sheet: dict[tuple[str, int], dict], uid: str) -> Callable:
    """Return a caller(cfg, prompt) that pulls from the sheet for this uid."""
    counter = {"i": 0}

    def caller(cfg: WellposedConfig, prompt: str) -> dict:
        sid = counter["i"]
        counter["i"] += 1
        reply = sheet.get((uid, sid))
        if reply is None:
            return {
                "verdict": "error",
                "insufficient_context": False,
                "reason": f"calibration miss uid={uid} sample={sid}",
            }
        return reply

    return caller

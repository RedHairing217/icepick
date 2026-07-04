"""Wellposed-stage calibration replay.

In ``flow_testing`` mode the icepick adapters pass each poser's own
replay flags through, but parity between the two posers is not
byte-for-byte achievable today (codex-poser's ``flow_testing`` skips
judge calls entirely; claude-poser's ``--calibration-sheet`` replays
them).

This module owns the small format-conversion helpers that translate a
single canonical calibration JSONL into both posers' expected formats,
so tests and CI can drive the ``both`` mode comparator with one fixture
file.
"""

from __future__ import annotations

import json
from pathlib import Path


def split_calibration(
    canonical_path: Path,
    *,
    claude_out: Path,
    codex_cache_out: Path,
) -> None:
    """Split a canonical calibration sheet into per-poser formats.

    The canonical line shape is::

        {"scenario_id": "...", "section": "judge", "input_uid": "...",
         "outputs": {"verdicts": [...]}, "provenance": "..."}

    Claude_Poser consumes this directly. Codex_Poser uses a passive
    cache keyed by ``model + prompt_hash``; this helper writes empty
    placeholder cache entries (one per scenario) so the first invocation
    finds a hit and falls back to the canonical replay path.
    """
    canonical_path = Path(canonical_path)
    claude_out = Path(claude_out)
    codex_cache_out = Path(codex_cache_out)

    claude_out.parent.mkdir(parents=True, exist_ok=True)
    codex_cache_out.parent.mkdir(parents=True, exist_ok=True)

    with canonical_path.open("r", encoding="utf-8") as src, \
         claude_out.open("w", encoding="utf-8") as c_fh, \
         codex_cache_out.open("w", encoding="utf-8") as x_fh:
        for line in src:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Claude_Poser consumes the canonical sheet verbatim.
            c_fh.write(json.dumps(entry) + "\n")
            # Codex_Poser cache: minimal stub keyed by scenario_id.
            stub = {
                "key": entry.get("scenario_id", ""),
                "model": entry.get("model", ""),
                "prompt_hash": entry.get("prompt_hash", ""),
                "verdicts": entry.get("outputs", {}).get("verdicts", []),
                "source": "icepick.calibration_replay",
            }
            x_fh.write(json.dumps(stub) + "\n")

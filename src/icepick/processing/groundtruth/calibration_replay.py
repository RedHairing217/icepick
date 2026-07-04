"""Groundtruth-stage calibration replay for flow_testing mode.

The cheat sheet is keyed by ``arxiv_id`` (one entry per paper, not per
record) and replaces every Anthropic call. Shape:

    {"arxiv_id": "2403.12345", "verdict_status": "published",
     "venue": "NeurIPS 2024", "publication_year": 2024,
     "indexed_in": ["DBLP"], "evidence_urls": [],
     "judge_votes": ["published", "published", "published"]}

Outputs from a flow_testing run are stamped ``calibration_replay: true``
in the manifest and must not enter accepted production downstreams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from icepick.processing.groundtruth.base import (
    STATUS_DEFER,
    GroundtruthVerdict,
)


class CalibrationSheetIncomplete(RuntimeError):
    """Raised when flow_testing needs an arxiv_id the sheet doesn't cover."""


class CalibrationReplay:
    def __init__(self, sheet_path):
        self.sheet_path = Path(sheet_path)
        self._by_arxiv: dict = {}
        self._load()

    def _load(self) -> None:
        if not self.sheet_path.exists():
            raise CalibrationSheetIncomplete(
                f"calibration sheet not found: {self.sheet_path}"
            )
        with self.sheet_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                aid = entry.get("arxiv_id")
                if aid:
                    self._by_arxiv[aid] = entry

    def lookup(
        self,
        *,
        arxiv_id: str,
        paper_title: Optional[str],
        uid_for_error_attribution: str,
        judge_model: str,
    ) -> GroundtruthVerdict:
        entry = self._by_arxiv.get(arxiv_id)
        if entry is None:
            # Missing scenarios route to defer rather than crashing the run.
            return GroundtruthVerdict(
                uid=uid_for_error_attribution,
                source="",
                verdict_status=STATUS_DEFER,
                arxiv_id=arxiv_id,
                judge_model=judge_model,
                judge_votes=[],
                judge_majority=STATUS_DEFER,
                reasoning=f"no calibration entry for arxiv_id={arxiv_id}",
                confidence="low",
                raw_payload={"calibration_replay": True, "missing": True},
            )
        return GroundtruthVerdict(
            uid=uid_for_error_attribution,
            source="",
            verdict_status=entry.get("verdict_status", STATUS_DEFER),
            arxiv_id=arxiv_id,
            venue=entry.get("venue"),
            publication_year=entry.get("publication_year"),
            indexed_in=entry.get("indexed_in") or [],
            evidence_urls=entry.get("evidence_urls") or [],
            judge_model=judge_model,
            judge_votes=entry.get("judge_votes") or [entry.get("verdict_status", STATUS_DEFER)],
            judge_majority=entry.get("verdict_status", STATUS_DEFER),
            reasoning=entry.get("reasoning", "calibration replay"),
            confidence=entry.get("confidence", "high"),
            raw_payload={"calibration_replay": True, "entry": entry},
        )

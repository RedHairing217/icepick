"""Calibration replay tests."""

from __future__ import annotations

import json

import pytest

from icepick.processing.groundtruth.base import STATUS_DEFER, STATUS_PUBLISHED
from icepick.processing.groundtruth.calibration_replay import (
    CalibrationReplay,
    CalibrationSheetIncomplete,
)


def _write_sheet(tmp_path, entries):
    path = tmp_path / "sheet.jsonl"
    with path.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return path


def test_lookup_hits_a_known_arxiv_id(tmp_path):
    sheet = _write_sheet(tmp_path, [
        {"arxiv_id": "2403.12345", "verdict_status": "published",
         "venue": "NeurIPS 2024", "indexed_in": ["DBLP"]},
    ])
    replay = CalibrationReplay(sheet)
    verdict = replay.lookup(
        arxiv_id="2403.12345", paper_title=None,
        uid_for_error_attribution="uid_x", judge_model="m",
    )
    assert verdict.verdict_status == STATUS_PUBLISHED
    assert verdict.venue == "NeurIPS 2024"
    assert verdict.raw_payload["calibration_replay"] is True


def test_missing_arxiv_id_returns_defer(tmp_path):
    sheet = _write_sheet(tmp_path, [])
    replay = CalibrationReplay(sheet)
    verdict = replay.lookup(
        arxiv_id="2403.99999", paper_title=None,
        uid_for_error_attribution="uid_x", judge_model="m",
    )
    assert verdict.verdict_status == STATUS_DEFER
    assert verdict.raw_payload["missing"] is True


def test_missing_sheet_raises(tmp_path):
    with pytest.raises(CalibrationSheetIncomplete):
        CalibrationReplay(tmp_path / "does-not-exist.jsonl")

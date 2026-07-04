"""Cross-subsystem check: realmath_scrape handoff feeds processing unchanged.

Allocation-side replay writes ``handoff/records.jsonl``; processing must
accept it through the same ``load_inputs`` path as any other JSONL, and
the groundtruth stage must apply its provenance rules to it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from icepick import cli
from icepick.allocation.adapters import realmath_scrape
from icepick.contracts.manifests import ApprovedManifest, SOURCE_REALMATH_SCRAPE
from icepick.processing.ingest import load_inputs

FIXTURE = "realmath/qa_candidates.jsonl"


def _replay(tmp_path, fixtures_dir):
    manifest = ApprovedManifest(
        run_id="20260701T120000Z",
        source_type=SOURCE_REALMATH_SCRAPE,
        processor_mode="flow_testing",
        requested_by="alice",
        requested_at="2026-07-01T00:00:00Z",
        approved_by="alice",
        approved_at="2026-07-01T00:00:00Z",
        source_name="realmath_2026Q2",
        target_count=5,
        call_budget=0,
        judge_enabled=False,
        confirmation_enabled=False,
        enable_leakage=False,
        enable_duplication=True,
        enable_robustness=False,
        output_dir=str(tmp_path / "intake"),
        calibration_sheet=str(fixtures_dir / FIXTURE),
    )
    return realmath_scrape.run(
        manifest, now=datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    )


def test_handoff_records_load_as_problem_records(tmp_path, fixtures_dir):
    outcome = _replay(tmp_path, fixtures_dir)
    records = list(load_inputs([(outcome.handoff_path, "realmath_2026Q2")]))
    assert len(records) == 7
    assert all(r.statement for r in records)
    assert all(len(r.uid) == 12 for r in records)
    assert {r.provenance for r in records} == {"extracted", "computed"}
    by_statement = {r.statement: r for r in records}
    hydra = by_statement[
        "Determine the number of rising-continuous functions satisfying "
        "the semi-basic p-Hydra functional equation fixing 0."
    ]
    assert hydra.answer_value == "1"
    assert "1" in hydra.truth_strings
    assert hydra.family == "realmath"


def test_groundtruth_stage_accepts_the_handoff(tmp_path, fixtures_dir, capsys):
    outcome = _replay(tmp_path, fixtures_dir)

    # Calibration sheet for the papers the handoff references.
    sheet = tmp_path / "groundtruth_sheet.jsonl"
    verdicts = [
        {"arxiv_id": "2412.19095", "verdict_status": "published", "venue": "Test Journal",
         "publication_year": 2025, "indexed_in": ["Scopus"],
         "judge_votes": ["published"] * 3, "reasoning": "fixture", "confidence": "high"},
        {"arxiv_id": "2412.02902", "verdict_status": "published", "venue": "Test Journal",
         "publication_year": 2025, "indexed_in": ["MathSciNet"],
         "judge_votes": ["published"] * 3, "reasoning": "fixture", "confidence": "high"},
        {"arxiv_id": "2501.00003", "verdict_status": "unpublished",
         "judge_votes": ["unpublished"] * 3, "reasoning": "fixture", "confidence": "high"},
        {"arxiv_id": "2502.11111", "verdict_status": "published", "venue": "Test Journal",
         "publication_year": 2026, "indexed_in": ["Scopus"],
         "judge_votes": ["published"] * 3, "reasoning": "fixture", "confidence": "high"},
        {"arxiv_id": "2502.22222", "verdict_status": "published", "venue": "Test Journal",
         "publication_year": 2026, "indexed_in": ["Scopus"],
         "judge_votes": ["published"] * 3, "reasoning": "fixture", "confidence": "high"},
    ]
    with sheet.open("w") as fh:
        for row in verdicts:
            fh.write(json.dumps(row) + "\n")

    rc = cli.main([
        "processing", "groundtruth",
        "--mode", "flow_testing",
        "--calibration-sheet", str(sheet),
        "--input", str(outcome.handoff_path),
        "--output-dir", str(tmp_path / "groundtruth"),
    ])
    assert rc == 0

    summary = json.loads(capsys.readouterr().out)
    # 4 published, 1 unpublished; the computed record and the record
    # without an arxiv_id are discarded pre-lookup.
    assert summary["counts"]["published"] == 4
    assert summary["counts"]["unpublished"] == 1
    assert summary["counts"]["discarded"] == 2

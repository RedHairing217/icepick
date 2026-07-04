import json
from pathlib import Path

from claude_poser.config import WellposedConfig
from claude_poser.schema import normalise_record
from claude_poser.wellposed import check_record, check_records


def _write_sheet(path: Path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_flow_testing_requires_sheet():
    cfg = WellposedConfig(processor_mode="flow_testing", calibration_sheet=None)
    try:
        cfg.validate()
    except ValueError:
        return
    raise AssertionError("flow_testing without sheet must fail validation")


def test_flow_testing_replays_from_sheet(tmp_path):
    sheet = tmp_path / "sheet.jsonl"
    rec = normalise_record({
        "source": "rm",
        "provenance": "extracted",
        "statement": "Using Theorem 3.2 from the previous section, deduce A.",
    }, rid=0)
    _write_sheet(sheet, [
        {"section": "judge", "uid": rec["uid"], "sample_id": 0,
         "reply": {"verdict": "pass", "insufficient_context": False, "reason": "ok"}},
        {"section": "judge", "uid": rec["uid"], "sample_id": 1,
         "reply": {"verdict": "pass", "insufficient_context": False, "reason": "ok"}},
        {"section": "judge", "uid": rec["uid"], "sample_id": 2,
         "reply": {"verdict": "flag", "insufficient_context": False, "reason": "?"}},
    ])
    cfg = WellposedConfig(
        enable_judge=True,
        processor_mode="flow_testing",
        calibration_sheet=str(sheet),
    )
    results = check_records([rec], cfg)
    assert results[0]["tier"] == "judge"
    assert results[0]["wellposed_status"] == "pass"
    assert abs(results[0]["wellposed_score"] - 2 / 3) < 1e-3


def test_flow_testing_calibration_miss_defers(tmp_path):
    sheet = tmp_path / "sheet.jsonl"
    rec = normalise_record({
        "source": "rm",
        "provenance": "extracted",
        "statement": "Using Theorem 3.2 from the previous section, deduce A.",
    }, rid=0)
    other_uid = "deadbeef" * 4
    _write_sheet(sheet, [
        {"section": "judge", "uid": other_uid, "sample_id": 0,
         "reply": {"verdict": "pass", "insufficient_context": False, "reason": "ok"}},
    ])
    cfg = WellposedConfig(
        enable_judge=True,
        processor_mode="flow_testing",
        calibration_sheet=str(sheet),
    )
    results = check_records([rec], cfg)
    assert results[0]["wellposed_status"] == "defer"

"""flow_testing mode: calibration replay — zero backends, zero checkpoints.

The injected backend is a booby trap (any attribute access raises), so
these tests prove the replay path never touches it. Configs validate
cleanly because flow_testing skips every production-only check.
"""

from __future__ import annotations

import json

import pytest

from icepick.processing.pass_at_k.base import (
    LABEL_BAND,
    LABEL_DROP,
    LABEL_MISDIRECTION,
    LABEL_SOLVED,
)
from icepick.processing.pass_at_k.calibration_replay import (
    DROP_NOT_IN_SHEET,
    CalibrationSheetIncomplete,
    load_calibration_sheet,
)
from icepick.processing.pass_at_k.config import PassAtKConfig
from icepick.processing.pass_at_k.runner import run


class _Boom:
    """Any attribute access is an error: flow_testing must never touch it."""

    def __getattr__(self, name):
        raise AssertionError(f"backend attribute {name!r} touched in flow_testing")


def _write_sheet(tmp_path, entries, name="sheet.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    return path


def _cfg(tmp_path, sheet, out_name="out"):
    return PassAtKConfig(
        mode="flow_testing",
        output_dir=tmp_path / out_name,
        calibration_sheet=sheet,
    )


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


_SHEET = [
    {"uid": "u1", "pass_at_k": 1.0, "n_correct": 4, "label": "solved"},
    {"uid": "u2", "pass_at_k": 0.0, "n_wrong": 4, "modal_wrong": "7",
     "top_wrong_share": 1.0},                       # no label -> derived misdirection
    {"uid": "u3", "pass_at_k": 0.5, "n_correct": 2, "n_wrong": 2},  # derived band
    {"uid": "u4", "pass_at_k": 0.0, "top_wrong_share": 0.25},       # derived collapse
]

_RECORDS = [
    {"uid": "u1", "source": "rm", "statement": "Q1", "truth": "4"},
    {"uid": "u2", "source": "rm", "statement": "Q2", "truth": "4"},
    {"uid": "u3", "source": "rm", "statement": "Q3", "truth": "4"},
    {"uid": "u4", "source": "rm", "statement": "Q4", "truth": "4"},
]


def test_replay_is_deterministic_and_never_builds_a_backend(tmp_path):
    sheet = _write_sheet(tmp_path, _SHEET)
    outcome1 = run(cfg=_cfg(tmp_path, sheet, "out1"), records=_RECORDS, backend=_Boom())
    outcome2 = run(cfg=_cfg(tmp_path, sheet, "out2"), records=_RECORDS, backend=_Boom())

    # Deterministic: two runs over the same sheet are byte-identical.
    assert outcome1.output_path.read_text() == outcome2.output_path.read_text()

    rows = _rows(outcome1.output_path)
    assert [r["label"] for r in rows] == [
        LABEL_SOLVED, LABEL_MISDIRECTION, LABEL_BAND, "collapse",
    ]
    assert rows[0]["pass_at_k"] == 1.0
    assert rows[1]["modal_wrong"] == "7"
    assert rows[2]["n_correct"] == 2

    assert outcome1.model_calls == 0
    assert outcome1.resumed_records == 0
    assert outcome1.interrupted is False
    # No checkpoint cache is ever created in flow_testing.
    assert not (tmp_path / "out1" / "_progress").exists()

    manifest = json.loads(outcome1.manifest_path.read_text())
    assert manifest["calibration_replay"] is True
    assert manifest["model_calls"] == 0
    assert manifest["config"]["mode"] == "flow_testing"


def test_uid_missing_from_sheet_becomes_a_drop(tmp_path):
    sheet = _write_sheet(tmp_path, _SHEET[:1])  # only u1 covered
    records = _RECORDS[:1] + [
        {"uid": "u9", "source": "rm", "statement": "Q9", "truth": "4"},
    ]
    outcome = run(cfg=_cfg(tmp_path, sheet), records=records, backend=_Boom())

    rows = _rows(outcome.output_path)
    missing = rows[1]
    assert missing["uid"] == "u9"
    assert missing["label"] == LABEL_DROP
    assert missing["drop_reason"] == DROP_NOT_IN_SHEET
    assert missing["pass_at_k"] is None
    assert outcome.counts["dropped"] == 1
    assert outcome.counts[LABEL_SOLVED] == 1


def test_sheet_label_wins_verbatim_even_when_it_disagrees_with_derivation(tmp_path):
    # pass_at_k=1.0 derives 'solved'; the sheet says 'band'. Sheet wins —
    # spec non-goal: replay does not audit the sheet's labels.
    sheet = _write_sheet(tmp_path, [{"uid": "u1", "pass_at_k": 1.0, "label": "band"}])
    outcome = run(cfg=_cfg(tmp_path, sheet), records=_RECORDS[:1], backend=_Boom())

    (row,) = _rows(outcome.output_path)
    assert row["label"] == LABEL_BAND
    assert row["pass_at_k"] == 1.0


def test_label_is_derived_only_when_sheet_omits_it(tmp_path):
    sheet = _write_sheet(tmp_path, [
        {"uid": "u1", "pass_at_k": 0.25},                          # band
        {"uid": "u2", "pass_at_k": 0.0, "top_wrong_share": 0.5},   # misdirection
    ])
    outcome = run(cfg=_cfg(tmp_path, sheet), records=_RECORDS[:2], backend=_Boom())

    rows = _rows(outcome.output_path)
    assert [r["label"] for r in rows] == [LABEL_BAND, LABEL_MISDIRECTION]


def test_missing_sheet_file_raises(tmp_path):
    with pytest.raises(CalibrationSheetIncomplete):
        load_calibration_sheet(tmp_path / "does-not-exist.jsonl")
    # And through the runner, after validate() (path is set, file absent).
    with pytest.raises(CalibrationSheetIncomplete):
        run(
            cfg=_cfg(tmp_path, tmp_path / "does-not-exist.jsonl"),
            records=_RECORDS[:1],
            backend=_Boom(),
        )

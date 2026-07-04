"""Pass@k calibration replay for flow_testing mode.

The cheat sheet is keyed by ``uid`` — pass@k is a per-record measurement,
unlike groundtruth's per-paper sheet — and replaces every backend call.
Shape (one JSONL row per record)::

    {"uid": "abc123...", "pass_at_k": 0.25,
     "n_correct": 2, "n_wrong": 5, "n_degenerate": 1,
     "modal_wrong": "7", "top_wrong_share": 0.5, "label": "band"}

Only ``uid`` and ``pass_at_k`` matter; everything else is optional. When
``label`` is present it is taken VERBATIM, even if it disagrees with what
``derive_label`` would compute — flow_testing replays the sheet, it does
not audit it (spec non-goal, same as the production passthrough for
records arriving with ``pass_at_k`` already set). When ``label`` is
absent it is derived from ``pass_at_k`` + ``top_wrong_share`` exactly as
production would.

Records whose uid is missing from the sheet become ``drop`` rows with
``drop_reason: not_in_calibration_sheet`` — deterministic and loud in the
counts, rather than crashing the flow test (mirrors groundtruth's
missing-scenario → defer routing).

Outputs from a flow_testing run are stamped ``calibration_replay: true``
in the manifest and must not enter accepted production downstreams.
"""

from __future__ import annotations

import json
from pathlib import Path

from icepick.processing.pass_at_k.base import LABEL_DROP, PassAtKRecord
from icepick.processing.pass_at_k.scoring import derive_label

DROP_NOT_IN_SHEET = "not_in_calibration_sheet"


class CalibrationSheetIncomplete(RuntimeError):
    """Raised when flow_testing's calibration sheet cannot be loaded."""


def load_calibration_sheet(path) -> dict:
    """Load the sheet as ``{uid: entry}``. Blank lines are skipped."""
    path = Path(path)
    if not path.exists():
        raise CalibrationSheetIncomplete(f"calibration sheet not found: {path}")
    out: dict = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            uid = entry.get("uid")
            if uid:
                out[uid] = entry
    return out


def replay(cfg, prepared_records: list) -> list:
    """Stamp every record deterministically from the sheet.

    Zero model calls, zero checkpoint — replay is pure function of
    (sheet, records) so two runs over the same inputs are byte-identical.
    Returns stamped rows in input order.
    """
    sheet = load_calibration_sheet(cfg.calibration_sheet)
    rows = []
    for record in prepared_records:
        uid = record["uid"]
        entry = sheet.get(uid)
        if entry is None:
            rec = PassAtKRecord(
                uid=uid,
                source=record.get("source", ""),
                pass_at_k=None,
                n_correct=0,
                n_wrong=0,
                n_degenerate=0,
                label=LABEL_DROP,
                modal_wrong=None,
                top_wrong_share=0.0,
                rollout_uids=[],
                drop_reason=DROP_NOT_IN_SHEET,
            )
        else:
            pass_at_k = entry.get("pass_at_k")
            top_wrong_share = entry.get("top_wrong_share") or 0.0
            label = entry.get("label") or derive_label(pass_at_k, top_wrong_share)
            rec = PassAtKRecord(
                uid=uid,
                source=record.get("source", ""),
                pass_at_k=pass_at_k,
                n_correct=int(entry.get("n_correct") or 0),
                n_wrong=int(entry.get("n_wrong") or 0),
                n_degenerate=int(entry.get("n_degenerate") or 0),
                label=label,
                modal_wrong=entry.get("modal_wrong"),
                top_wrong_share=top_wrong_share,
                rollout_uids=list(entry.get("rollout_uids") or []),
                drop_reason=entry.get("drop_reason"),
            )
        rows.append(rec.stamp(record))
    return rows

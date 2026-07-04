"""Schema and ingest tests — the only stage with working code in this build.

These tests double as the executable spec for the first implementation
step: every row normalises onto ``ProblemRecord``, ``uid`` is stable
across load order and file set, and provenance / truth-policy defaults
follow the design rules.
"""

from __future__ import annotations

import json
from pathlib import Path

from icepick.contracts.records import (
    PROVENANCE_COMPUTED,
    PROVENANCE_EXTRACTED,
    PROVENANCE_MANUAL,
    TRUTH_POLICY_EXTRACTED,
    TRUTH_POLICY_TRUSTED,
    TRUTH_POLICY_UNKNOWN,
)
from icepick.processing import ingest, schema


def _write(path: Path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_loads_generated_and_extracted_rows(mixed_jsonl):
    records = list(ingest.load_inputs([(mixed_jsonl, "mixed")]))
    assert len(records) == 2
    generated, extracted = records
    assert generated.family == "calculus"
    assert generated.provenance == PROVENANCE_COMPUTED
    assert generated.truth_policy == TRUTH_POLICY_TRUSTED
    assert generated.label == "band"
    assert extracted.family is None
    assert extracted.provenance == PROVENANCE_EXTRACTED
    assert extracted.truth_policy == TRUTH_POLICY_EXTRACTED
    assert extracted.label == "misdirection"


def test_rid_increments_across_inputs(tmp_path):
    a = _write(tmp_path / "a.jsonl", [{"question": "q1", "answer": "1"}])
    b = _write(tmp_path / "b.jsonl", [{"question": "q2", "answer": "2"}])
    records = list(ingest.load_inputs([(a, "s1"), (b, "s2")]))
    assert [r.rid for r in records] == [0, 1]


def test_uid_is_stable_across_input_order(tmp_path):
    rows = [
        {"question": "alpha", "answer": "1"},
        {"question": "beta", "answer": "2"},
    ]
    forward = _write(tmp_path / "f.jsonl", rows)
    backward = _write(tmp_path / "b.jsonl", list(reversed(rows)))
    uids_forward = {r.statement: r.uid for r in ingest.load_inputs([(forward, "S")])}
    uids_backward = {r.statement: r.uid for r in ingest.load_inputs([(backward, "S")])}
    assert uids_forward == uids_backward


def test_uid_differs_by_source(tmp_path):
    row = {"question": "the same statement", "answer": "1"}
    path = _write(tmp_path / "p.jsonl", [row])
    a = next(ingest.load_inputs([(path, "alpha")]))
    b = next(ingest.load_inputs([(path, "beta")]))
    assert a.uid != b.uid


def test_provenance_override_takes_precedence(tmp_path):
    row = {"family": "calculus", "question": "q", "answer": "1"}
    path = _write(tmp_path / "p.jsonl", [row])
    records = list(
        ingest.load_inputs(
            [(path, "drop")],
            provenance_overrides={"drop": PROVENANCE_MANUAL},
        )
    )
    assert records[0].provenance == PROVENANCE_MANUAL
    assert records[0].truth_policy == TRUTH_POLICY_UNKNOWN


def test_column_map_renames_csv_style_keys(tmp_path):
    row = {"prompt_text": "q", "gold": "42"}
    path = _write(tmp_path / "p.jsonl", [row])
    records = list(
        ingest.load_inputs(
            [(path, "csv_drop")],
            column_maps={"csv_drop": {"statement": "prompt_text", "answer": "gold"}},
            provenance_overrides={"csv_drop": PROVENANCE_MANUAL},
            truth_policy_overrides={"csv_drop": TRUTH_POLICY_UNKNOWN},
        )
    )
    assert records[0].statement == "q"
    assert records[0].truth_strings == ["42"]
    assert records[0].truth_policy == TRUTH_POLICY_UNKNOWN


def test_explicit_label_is_trusted_over_derivation():
    raw = {
        "question": "q",
        "answer": "1",
        "pass_at_k": 0.99,
        "label": "band",
    }
    rec = schema.from_raw(raw, source="s", rid=0)
    assert rec.label == "band"


def test_collapse_when_below_band_with_scattered_wrong():
    raw = {
        "question": "q",
        "answer": "1",
        "pass_at_k": 0.0,
        "top_wrong_share": 0.2,
    }
    rec = schema.from_raw(raw, source="s", rid=0)
    assert rec.label == "collapse"


def test_misdirection_when_below_band_with_dominant_wrong():
    raw = {
        "question": "q",
        "answer": "1",
        "pass_at_k": 0.0,
        "top_wrong_share": 0.8,
        "top_wrong_value": "wrong",
    }
    rec = schema.from_raw(raw, source="s", rid=0)
    assert rec.label == "misdirection"
    assert rec.modal_wrong == "wrong"


def test_n_total_excludes_degenerate():
    raw = {
        "question": "q",
        "answer": "1",
        "correct": 2,
        "wrong_complete": 1,
        "degenerate": 5,
    }
    rec = schema.from_raw(raw, source="s", rid=0)
    assert rec.n_correct == 2
    assert rec.n_wrong == 1
    assert rec.n_degenerate == 5
    assert rec.n_total == 3

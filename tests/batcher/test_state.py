"""Tests for src/icepick/batcher/state.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick.batcher.state import (
    STATES_LINEAR,
    load_all_states,
    load_state,
    transition,
)

NOW = "2026-07-07T00:00:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch_dir(tmp_path: Path, name: str = "batch10") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# Basic linear transitions
# ---------------------------------------------------------------------------


def test_initial_transition_to_sliced(tmp_path):
    bd = _make_batch_dir(tmp_path)
    st = transition(bd, "SLICED", now_iso=NOW)
    assert st["state"] == "SLICED"
    assert st["created_at"] == NOW
    assert len(st["history"]) == 1
    assert st["history"][0]["from"] is None
    assert st["history"][0]["to"] == "SLICED"


def test_full_linear_lifecycle(tmp_path):
    bd = _make_batch_dir(tmp_path)
    for state in STATES_LINEAR:
        st = transition(bd, state, now_iso=NOW)
        assert st["state"] == state

    # READY_TO_FOLD flag file must exist.
    flag = bd / "READY_TO_FOLD"
    assert flag.exists()
    flag_data = json.loads(flag.read_text())
    assert flag_data["batch"] == bd.name
    assert flag_data["at"] == NOW


def test_transition_no_op_when_already_in_state(tmp_path):
    bd = _make_batch_dir(tmp_path)
    transition(bd, "SLICED", now_iso=NOW)
    st2 = transition(bd, "SLICED", now_iso=NOW)
    assert st2["state"] == "SLICED"
    # history should not have grown.
    assert len(st2["history"]) == 1


def test_transition_illegal_skip_raises(tmp_path):
    bd = _make_batch_dir(tmp_path)
    transition(bd, "SLICED", now_iso=NOW)
    with pytest.raises(ValueError, match="Illegal state transition"):
        transition(bd, "CASCADE_DONE", now_iso=NOW)


def test_transition_backwards_raises(tmp_path):
    bd = _make_batch_dir(tmp_path)
    transition(bd, "SLICED", now_iso=NOW)
    transition(bd, "MOUNTED", now_iso=NOW)
    with pytest.raises(ValueError, match="Illegal state transition"):
        transition(bd, "SLICED", now_iso=NOW)


def test_first_transition_must_be_sliced(tmp_path):
    bd = _make_batch_dir(tmp_path)
    with pytest.raises(ValueError, match="First transition must be to 'SLICED'"):
        transition(bd, "MOUNTED", now_iso=NOW)


# ---------------------------------------------------------------------------
# FROZEN
# ---------------------------------------------------------------------------


def test_frozen_from_sliced(tmp_path):
    bd = _make_batch_dir(tmp_path)
    transition(bd, "SLICED", now_iso=NOW)
    st = transition(bd, "FROZEN", note="exec_failed: oops", now_iso=NOW)
    assert st["state"] == "FROZEN"
    assert st["frozen"]["reason"] == "exec_failed: oops"
    assert st["frozen"]["from_state"] == "SLICED"


def test_frozen_from_mounted(tmp_path):
    bd = _make_batch_dir(tmp_path)
    transition(bd, "SLICED", now_iso=NOW)
    transition(bd, "MOUNTED", now_iso=NOW)
    st = transition(bd, "FROZEN", note="cost_guard_tripped", now_iso=NOW)
    assert st["frozen"]["from_state"] == "MOUNTED"


def test_cannot_freeze_ready_batch(tmp_path):
    bd = _make_batch_dir(tmp_path)
    for s in STATES_LINEAR:
        transition(bd, s, now_iso=NOW)
    with pytest.raises(ValueError, match="Cannot FREEZE a READY batch"):
        transition(bd, "FROZEN", now_iso=NOW)


# ---------------------------------------------------------------------------
# extra fields
# ---------------------------------------------------------------------------


def test_extra_fields_stored_in_state(tmp_path):
    bd = _make_batch_dir(tmp_path)
    transition(bd, "SLICED", now_iso=NOW)
    st = transition(bd, "MOUNTED", now_iso=NOW, extra={"mount_run_dir": "/some/run"})
    assert st["mount_run_dir"] == "/some/run"


def test_cascade_data_stored(tmp_path):
    bd = _make_batch_dir(tmp_path)
    for s in ["SLICED", "MOUNTED"]:
        transition(bd, s, now_iso=NOW)
    st = transition(
        bd, "CASCADE_DONE", now_iso=NOW,
        extra={"cascade_data": {"cost_usd": 2.31}},
    )
    assert st["cascade_data"]["cost_usd"] == 2.31


# ---------------------------------------------------------------------------
# load_state / load_all_states
# ---------------------------------------------------------------------------


def test_load_state_missing_dir(tmp_path):
    bd = tmp_path / "batch99"
    # Does not exist → empty dict
    st = load_state(bd)
    assert st == {}


def test_load_all_states_empty(tmp_path):
    batches = tmp_path / "batches"
    result = load_all_states(batches)
    assert result == {}


def test_load_all_states_sorted_by_number(tmp_path):
    batches = tmp_path / "batches"
    for n in [12, 10, 11]:
        bd = batches / f"batch{n}"
        bd.mkdir(parents=True)
        transition(bd, "SLICED", now_iso=NOW)

    result = load_all_states(batches)
    keys = list(result.keys())
    assert keys == ["batch10", "batch11", "batch12"]


def test_load_all_states_excludes_non_batch_dirs(tmp_path):
    batches = tmp_path / "batches"
    batches.mkdir()
    (batches / "some_other_dir").mkdir()
    (batches / "batch10").mkdir()
    transition(batches / "batch10", "SLICED", now_iso=NOW)

    result = load_all_states(batches)
    assert "some_other_dir" not in result
    assert "batch10" in result


# ---------------------------------------------------------------------------
# Atomic write (state.json is valid JSON on disk)
# ---------------------------------------------------------------------------


def test_state_json_is_valid_on_disk(tmp_path):
    bd = _make_batch_dir(tmp_path)
    transition(bd, "SLICED", now_iso=NOW)
    raw = (bd / "state.json").read_text()
    data = json.loads(raw)
    assert data["state"] == "SLICED"

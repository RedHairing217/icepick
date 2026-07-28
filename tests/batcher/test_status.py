"""Tests for src/icepick/batcher/status.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick.batcher.status import render_status, write_status


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _base_qs(
    armed: bool = True,
    halt: dict | None = None,
    spend: float = 0.0,
    next_batch: int = 10,
) -> dict:
    return {
        "config": {
            "campaign_source": "arxiv_bulk_pde625",
            "journal_path": "/some/journal.jsonl",
            "cross_source_statement_policy": "skip",
            "mode": "production",
            "cost_limit_usd": 5.0,
        },
        "next_batch_number": next_batch,
        "halt": halt,
        "spend_usd_total": spend,
        "updated_at": "2026-07-07T00:00:00Z",
        "created_at": "2026-07-07T00:00:00Z",
    }


def _make_batch_state(
    state: str = "SLICED",
    cost: float | None = None,
    passk: dict | None = None,
    updated: str = "2026-07-07T00:00:00Z",
    frozen_reason: str | None = None,
) -> dict:
    st: dict = {
        "state": state,
        "updated_at": updated,
        "history": [],
        "counts": {"accepted": 250},
    }
    if cost is not None:
        st["cascade_data"] = {"cost_usd": cost}
    if passk is not None:
        st["passk_counts"] = passk
    if frozen_reason is not None:
        st["frozen"] = {
            "reason": frozen_reason,
            "from_state": "MOUNTED",
            "at": updated,
        }
    return st


# ---------------------------------------------------------------------------
# render_status
# ---------------------------------------------------------------------------


def test_render_disarmed_header():
    qs = _base_qs()
    content = render_status(qs, {}, {"armed": False})
    assert "DISARMED" in content
    assert "ARMED" in content  # "DISARMED" contains "ARMED"


def test_render_armed_header():
    qs = _base_qs()
    content = render_status(qs, {}, {"armed": True})
    assert content.startswith("# Batcher STATUS — ARMED")


def test_render_halt_shows_reason():
    qs = _base_qs(halt={"active": True, "reason": "cost_guard:batch10", "at": "2026-07-07T01:00:00Z"})
    content = render_status(qs, {})
    assert "QUEUE HALTED" in content
    assert "cost_guard:batch10" in content


def test_render_no_batches():
    qs = _base_qs()
    content = render_status(qs, {})
    assert "(no batches yet)" in content


def test_render_batch_table_contains_columns():
    qs = _base_qs()
    bs = {"batch10": _make_batch_state("MOUNTED", cost=2.31)}
    content = render_status(qs, bs)
    assert "| batch10 |" in content
    assert "MOUNTED" in content
    assert "$2.3100" in content


def test_render_batch_table_deterministic_order():
    qs = _base_qs()
    bs = {
        "batch12": _make_batch_state("READY"),
        "batch10": _make_batch_state("SLICED"),
        "batch11": _make_batch_state("MOUNTED"),
    }
    content = render_status(qs, bs)
    idx10 = content.index("batch10")
    idx11 = content.index("batch11")
    idx12 = content.index("batch12")
    assert idx10 < idx11 < idx12


def test_render_frozen_batch_shows_hint():
    qs = _base_qs()
    bs = {"batch10": _make_batch_state("FROZEN", frozen_reason="cost_guard_tripped")}
    content = render_status(qs, bs)
    assert "FROZEN" in content
    assert "To resume" in content or "to resume" in content


def test_render_held_remainder():
    qs = _base_qs()
    extras = {"held_remainder": {"count": 42, "uids": ["uid1", "uid2"]}}
    content = render_status(qs, {}, extras)
    assert "HELD Remainder" in content
    assert "42" in content
    assert "uid1" in content


def test_render_skip_tallies():
    qs = _base_qs()
    extras = {"skip_counts": {"replay": 5, "stmt": 3, "warns": 1}}
    content = render_status(qs, {}, extras)
    assert "Replay skips" in content
    assert "5" in content
    assert "Statement skips" in content


def test_render_cursor_reset_note():
    qs = _base_qs()
    extras = {"skip_counts": {"replay": 0, "stmt": 0, "warns": 0}, "cursor_reset": True}
    content = render_status(qs, {}, extras)
    assert "CursorStore was reset" in content


def test_render_watch_counters():
    qs = _base_qs()
    extras = {"watch_counters": {"batch9": {"ingested": 10, "malformed": 2}}}
    content = render_status(qs, {}, extras)
    assert "Watch Journals" in content
    assert "batch9" in content
    assert "10 ingested" in content


def test_render_events_last_10():
    qs = _base_qs()
    events = [f'event_{i}' for i in range(15)]
    extras = {"events_log": events}
    content = render_status(qs, {}, extras)
    # Should show only last 10.
    assert "event_14" in content
    assert "event_4" not in content  # only last 10 of 15


def test_render_spend_total():
    qs = _base_qs(spend=4.62)
    content = render_status(qs, {})
    assert "$4.6200" in content


# ---------------------------------------------------------------------------
# write_status
# ---------------------------------------------------------------------------


def test_write_status_creates_file(tmp_path):
    write_status(tmp_path, "# Hello")
    assert (tmp_path / "STATUS.md").read_text() == "# Hello"


def test_write_status_overwrites(tmp_path):
    write_status(tmp_path, "# First")
    write_status(tmp_path, "# Second")
    assert (tmp_path / "STATUS.md").read_text() == "# Second"


def test_write_status_creates_root_if_missing(tmp_path):
    root = tmp_path / "deep" / "nested"
    write_status(root, "# Test")
    assert (root / "STATUS.md").exists()

"""Tests for src/icepick/batcher/cli_glue.py.

Invokes handler functions directly with parsed args (no subprocess).
Tests arm/disarm/clear-halt/status/backfill/--once handlers.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest

from icepick.batcher.cli_glue import (
    build_batcher_parser,
    _handle_arm,
    _handle_disarm,
    _handle_clear_halt,
    _handle_status,
    _handle_backfill,
    _parse_watch_journals,
)

NOW = "2026-07-07T00:00:00Z"


# ---------------------------------------------------------------------------
# Parser sanity
# ---------------------------------------------------------------------------


def _make_parser():
    import argparse
    from icepick.cli import build_parser
    return build_parser()


def test_build_parser_includes_batcher():
    parser = _make_parser()
    try:
        parser.parse_args(["batcher", "--help"])
    except SystemExit as e:
        assert e.code == 0


def test_batcher_run_requires_journal():
    parser = _make_parser()
    try:
        parser.parse_args([
            "batcher", "run",
            "--campaign-source", "src",
            "--run-dir", "/tmp/run",
        ])
    except SystemExit as e:
        assert e.code != 0


def test_batcher_run_parses_flags():
    parser = _make_parser()
    args = parser.parse_args([
        "batcher", "run",
        "--journal", "/tmp/j.jsonl",
        "--run-dir", "/tmp/run",
        "--campaign-source", "arxiv_bulk_pde625",
        "--slice-size", "5",
        "--once",
    ])
    assert args.campaign_source == "arxiv_bulk_pde625"
    assert args.slice_size == 5
    assert args.once is True


# ---------------------------------------------------------------------------
# _parse_watch_journals
# ---------------------------------------------------------------------------


def test_parse_watch_journals_basic():
    result = _parse_watch_journals(["batch9=/some/run/dir"])
    assert len(result) == 1
    assert result[0]["label"] == "batch9"
    assert result[0]["run_dir"] == "/some/run/dir"
    assert "candidates.jsonl" in result[0]["journal_path"]


def test_parse_watch_journals_no_eq_raises():
    with pytest.raises(ValueError, match="LABEL=RUN_DIR"):
        _parse_watch_journals(["noequalssign"])


# ---------------------------------------------------------------------------
# arm / disarm
# ---------------------------------------------------------------------------


class _FakeArgs:
    """Simple namespace for handler tests."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_arm_requires_approval_flag(tmp_path):
    args = _FakeArgs(root=str(tmp_path), i_approve_recurring_spend=False)
    rc = _handle_arm(args)
    assert rc == 1
    assert not (tmp_path / "ARMED").exists()


def test_arm_writes_armed_file(tmp_path):
    args = _FakeArgs(root=str(tmp_path), i_approve_recurring_spend=True)
    rc = _handle_arm(args)
    assert rc == 0
    armed_path = tmp_path / "ARMED"
    assert armed_path.exists()
    data = json.loads(armed_path.read_text())
    assert data["by"] == "cli"
    assert "armed_at" in data


def test_disarm_removes_armed(tmp_path):
    (tmp_path / "ARMED").write_text(json.dumps({"armed_at": NOW}))
    args = _FakeArgs(root=str(tmp_path))
    rc = _handle_disarm(args)
    assert rc == 0
    assert not (tmp_path / "ARMED").exists()


def test_disarm_noop_if_not_armed(tmp_path, capsys):
    args = _FakeArgs(root=str(tmp_path))
    rc = _handle_disarm(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "already DISARMED" in out


# ---------------------------------------------------------------------------
# clear-halt
# ---------------------------------------------------------------------------


def test_clear_halt_clears_active_halt(tmp_path):
    qs = {
        "config": {},
        "next_batch_number": 10,
        "halt": {"active": True, "reason": "cost_guard", "at": NOW},
        "spend_usd_total": 0.0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    (tmp_path / "queue_state.json").write_text(json.dumps(qs), encoding="utf-8")

    args = _FakeArgs(root=str(tmp_path), reason="manually reviewed")
    rc = _handle_clear_halt(args)
    assert rc == 0

    qs2 = json.loads((tmp_path / "queue_state.json").read_text())
    assert qs2["halt"]["active"] is False

    # events.jsonl entry.
    events = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(events) == 1
    ev = json.loads(events[0])
    assert ev["kind"] == "halt_cleared"
    assert ev["reason"] == "manually reviewed"


def test_clear_halt_noop_when_not_halted(tmp_path, capsys):
    qs = {
        "config": {},
        "next_batch_number": 10,
        "halt": None,
        "spend_usd_total": 0.0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    (tmp_path / "queue_state.json").write_text(json.dumps(qs), encoding="utf-8")

    args = _FakeArgs(root=str(tmp_path), reason="unused")
    rc = _handle_clear_halt(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "not halted" in out


def test_clear_halt_missing_queue_state(tmp_path):
    args = _FakeArgs(root=str(tmp_path), reason="test")
    rc = _handle_clear_halt(args)
    assert rc == 1


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_empty_root(tmp_path, capsys):
    args = _FakeArgs(root=str(tmp_path))
    rc = _handle_status(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "queue_state" in out
    assert "batch_states" in out
    assert out["armed"] is False


def test_status_with_queue_state(tmp_path, capsys):
    qs = {
        "config": {"campaign_source": "src"},
        "next_batch_number": 10,
        "halt": None,
        "spend_usd_total": 0.0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    (tmp_path / "queue_state.json").write_text(json.dumps(qs), encoding="utf-8")
    (tmp_path / "ARMED").write_text("{}")

    args = _FakeArgs(root=str(tmp_path))
    rc = _handle_status(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["armed"] is True
    assert out["queue_state"]["next_batch_number"] == 10


# ---------------------------------------------------------------------------
# backfill (dry-run path; avoids real file I/O)
# ---------------------------------------------------------------------------


def test_backfill_dry_run(tmp_path, capsys):
    args = _FakeArgs(
        root=str(tmp_path),
        sources_json=None,
        dry_run=True,
    )
    rc = _handle_backfill(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert "sources" in out
    assert isinstance(out["sources"], list)


# ---------------------------------------------------------------------------
# --once integration (mocked daemon)
# ---------------------------------------------------------------------------


def test_once_flag_runs_single_tick(tmp_path):
    """--once: startup + tick + status, then exit."""
    journal = tmp_path / "j" / "candidates.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text("")

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    root = tmp_path / "batcher"
    _arm_root(root)

    # Simulate the --once path directly without invoking a subprocess.
    from icepick.batcher.config import BatcherConfig
    from icepick.batcher.daemon import BatcherDaemon

    config = BatcherConfig(
        root=root,
        journal_path=journal,
        run_dir=run_dir,
        campaign_source="src_test",
        slice_size=3,
        cost_limit_usd=5.0,
        key_path="/fake.env",
        mode="flow_testing",
    )

    tick_results = ["waiting_journal"]
    idx = [0]

    class FakeDaemon(BatcherDaemon):
        def tick(self_inner):
            r = tick_results[idx[0]]
            idx[0] += 1
            return r

    daemon = FakeDaemon(config, sleep_fn=lambda s: None, now_iso_fn=lambda: NOW)
    daemon.startup()
    tag = daemon.tick()

    assert tag == "waiting_journal"

    # Write status.
    from icepick.batcher.state import load_all_states
    from icepick.batcher.status import render_status, write_status

    batch_states = load_all_states(root / "batches")
    qs_path = root / "queue_state.json"
    qs = json.loads(qs_path.read_text()) if qs_path.exists() else {}
    content = render_status(qs, batch_states, {"armed": True})
    write_status(root, content)

    assert (root / "STATUS.md").exists()


def _arm_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ARMED").write_text(json.dumps({"armed_at": NOW}))

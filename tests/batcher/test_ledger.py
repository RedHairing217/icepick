"""Tests for src/icepick/batcher/ledger.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, call

import pytest

from icepick.batcher.ledger import Ledger, LedgerRow, LedgerCorruption, Verdict


NOW = "2026-07-07T00:00:00Z"


def _make_row(
    uid="uid1",
    stmt_key="sk1",
    content_hash="ch1",
    batch="batch1",
    source_journal="journal.jsonl",
    journal_line=0,
    sliced_at=NOW,
    warn_only=False,
) -> LedgerRow:
    return LedgerRow(
        uid=uid,
        stmt_key=stmt_key,
        content_hash=content_hash,
        batch=batch,
        source_journal=source_journal,
        journal_line=journal_line,
        sliced_at=sliced_at,
        warn_only=warn_only,
    )


# ---------------------------------------------------------------------------
# Load: empty / missing file
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_missing_file(self, tmp_path):
        ledger = Ledger.load(tmp_path / "ledger")
        assert ledger.torn_tail_skipped is False
        assert ledger._by_uid == {}

    def test_load_empty_file(self, tmp_path):
        d = tmp_path / "ledger"
        d.mkdir()
        (d / "consumed_uids.jsonl").write_text("", encoding="utf-8")
        ledger = Ledger.load(d)
        assert ledger._by_uid == {}

    def test_load_valid_rows(self, tmp_path):
        d = tmp_path / "ledger"
        d.mkdir()
        row = _make_row()
        (d / "consumed_uids.jsonl").write_text(
            json.dumps(
                {
                    "uid": row.uid,
                    "stmt_key": row.stmt_key,
                    "content_hash": row.content_hash,
                    "batch": row.batch,
                    "source_journal": row.source_journal,
                    "journal_line": row.journal_line,
                    "sliced_at": row.sliced_at,
                    "warn_only": row.warn_only,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ledger = Ledger.load(d)
        assert "uid1" in ledger._by_uid

    def test_torn_tail_tolerated(self, tmp_path):
        """A non-JSON last line is skipped with a warning; torn_tail_skipped is set."""
        d = tmp_path / "ledger"
        d.mkdir()
        row = _make_row()
        good_line = json.dumps(
            {
                "uid": row.uid,
                "stmt_key": row.stmt_key,
                "content_hash": row.content_hash,
                "batch": row.batch,
                "source_journal": row.source_journal,
                "journal_line": row.journal_line,
                "sliced_at": row.sliced_at,
                "warn_only": row.warn_only,
            }
        )
        (d / "consumed_uids.jsonl").write_text(
            good_line + "\n" + '{"uid": "partial',
            encoding="utf-8",
        )
        with pytest.warns(RuntimeWarning, match="Torn tail"):
            ledger = Ledger.load(d)
        assert ledger.torn_tail_skipped is True
        assert "uid1" in ledger._by_uid  # good row was loaded

    def test_mid_file_corruption_raises(self, tmp_path):
        """An unparseable non-final line raises LedgerCorruption."""
        d = tmp_path / "ledger"
        d.mkdir()
        good = json.dumps(
            {
                "uid": "u1",
                "stmt_key": "sk1",
                "content_hash": "ch1",
                "batch": "b1",
                "source_journal": "j",
                "journal_line": 0,
                "sliced_at": NOW,
                "warn_only": False,
            }
        )
        good2 = json.dumps(
            {
                "uid": "u2",
                "stmt_key": "sk2",
                "content_hash": "ch2",
                "batch": "b2",
                "source_journal": "j",
                "journal_line": 1,
                "sliced_at": NOW,
                "warn_only": False,
            }
        )
        (d / "consumed_uids.jsonl").write_text(
            good + "\n" + "{BAD JSON}\n" + good2 + "\n",
            encoding="utf-8",
        )
        with pytest.raises(LedgerCorruption):
            Ledger.load(d)

    def test_warn_only_rows_indexed_separately(self, tmp_path):
        d = tmp_path / "ledger"
        d.mkdir()
        row = _make_row(uid="wu", stmt_key="wsk", warn_only=True)
        (d / "consumed_uids.jsonl").write_text(
            json.dumps(
                {
                    "uid": row.uid,
                    "stmt_key": row.stmt_key,
                    "content_hash": row.content_hash,
                    "batch": row.batch,
                    "source_journal": row.source_journal,
                    "journal_line": row.journal_line,
                    "sliced_at": row.sliced_at,
                    "warn_only": row.warn_only,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ledger = Ledger.load(d)
        assert "wu" not in ledger._by_uid
        assert "wu" in ledger._warn_uids


# ---------------------------------------------------------------------------
# Check: verdict precedence
# ---------------------------------------------------------------------------


class TestCheck:
    def _ledger_with_row(self, tmp_path, row: LedgerRow) -> Ledger:
        ledger = Ledger.load(tmp_path / "ledger")
        ledger.append_all([row])
        return ledger

    def test_new(self, tmp_path):
        ledger = Ledger.load(tmp_path / "ledger")
        v = ledger.check("u_new", "sk_new", "ch_new")
        assert v.kind == "new"
        assert v.prior is None

    def test_replay(self, tmp_path):
        row = _make_row(uid="u1", content_hash="ch1")
        ledger = self._ledger_with_row(tmp_path, row)
        v = ledger.check("u1", "sk_other", "ch1")
        assert v.kind == "replay"
        assert v.prior is not None
        assert v.prior.uid == "u1"

    def test_uid_conflict(self, tmp_path):
        row = _make_row(uid="u1", content_hash="ch1")
        ledger = self._ledger_with_row(tmp_path, row)
        v = ledger.check("u1", "sk_other", "ch_DIFFERENT")
        assert v.kind == "uid_conflict"
        assert v.prior.uid == "u1"

    def test_stmt_conflict(self, tmp_path):
        row = _make_row(uid="u1", stmt_key="sk1")
        ledger = self._ledger_with_row(tmp_path, row)
        # Different uid, same stmt_key
        v = ledger.check("u_new", "sk1", "ch_new")
        assert v.kind == "stmt_conflict"
        assert v.prior.stmt_key == "sk1"

    def test_uid_takes_precedence_over_stmt(self, tmp_path):
        """uid conflict beats stmt_key conflict."""
        row = _make_row(uid="u1", stmt_key="sk1", content_hash="ch1")
        ledger = self._ledger_with_row(tmp_path, row)
        # Same uid, same stmt_key, different content_hash → uid_conflict wins
        v = ledger.check("u1", "sk1", "ch_different")
        assert v.kind == "uid_conflict"

    def test_warn_uid_hit(self, tmp_path):
        row = _make_row(uid="wu", stmt_key="wsk", warn_only=True)
        ledger = self._ledger_with_row(tmp_path, row)
        # uid matches a warn-only row → 'warn'
        v = ledger.check("wu", "sk_new", "ch_new")
        assert v.kind == "warn"
        assert v.prior.uid == "wu"

    def test_warn_stmt_hit(self, tmp_path):
        row = _make_row(uid="wu", stmt_key="wsk", warn_only=True)
        ledger = self._ledger_with_row(tmp_path, row)
        # stmt_key matches a warn-only row (uid is new) → 'warn'
        v = ledger.check("u_new", "wsk", "ch_new")
        assert v.kind == "warn"

    def test_blocking_beats_warn_uid(self, tmp_path):
        """A blocking uid collision takes precedence over a warn-only uid hit."""
        blocking = _make_row(uid="u1", stmt_key="sk1", content_hash="ch1", warn_only=False)
        warn = _make_row(uid="u1", stmt_key="sk_w", warn_only=True, batch="hist:warn")
        ledger = Ledger.load(tmp_path / "ledger")
        ledger.append_all([blocking, warn])
        v = ledger.check("u1", "sk_x", "ch_diff")
        # blocking index has u1 → uid_conflict (not warn)
        assert v.kind == "uid_conflict"


# ---------------------------------------------------------------------------
# append_all: fsync, idempotence, persistence
# ---------------------------------------------------------------------------


class TestAppendAll:
    def test_fsync_called(self, tmp_path):
        """os.fsync must be called once per append_all invocation."""
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        row = _make_row()
        with patch("icepick.batcher.ledger.os.fsync") as mock_fsync:
            ledger.append_all([row])
            assert mock_fsync.call_count == 1

    def test_fsync_not_called_when_nothing_to_write(self, tmp_path):
        """No fsync when all rows are already in the ledger."""
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        row = _make_row()
        ledger.append_all([row])  # first write
        with patch("icepick.batcher.ledger.os.fsync") as mock_fsync:
            ledger.append_all([row])  # idempotent — nothing written
            mock_fsync.assert_not_called()

    def test_idempotence_same_uid_batch(self, tmp_path):
        """Re-appending a (uid, batch) pair that's already in the ledger is a no-op."""
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        row = _make_row(uid="u1", batch="batch1")
        ledger.append_all([row])
        ledger.append_all([row])  # second time — no-op
        # The file should have exactly one line.
        lines = [
            ln for ln in (d / "consumed_uids.jsonl").read_text().splitlines() if ln.strip()
        ]
        assert len(lines) == 1

    def test_different_batch_same_uid_written(self, tmp_path):
        """Same uid in a different batch IS written (cross-label historical dup)."""
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        row1 = _make_row(uid="u1", batch="hist:batch1")
        row2 = _make_row(uid="u1", batch="hist:batch2")
        ledger.append_all([row1, row2])
        lines = [
            ln for ln in (d / "consumed_uids.jsonl").read_text().splitlines() if ln.strip()
        ]
        assert len(lines) == 2

    def test_in_memory_index_updated(self, tmp_path):
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        row = _make_row(uid="u1", stmt_key="sk1")
        ledger.append_all([row])
        assert "u1" in ledger._by_uid
        assert "sk1" in ledger._by_stmt

    def test_persists_to_disk(self, tmp_path):
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        row = _make_row()
        ledger.append_all([row])
        # Reload from disk
        ledger2 = Ledger.load(d)
        assert "uid1" in ledger2._by_uid

    def test_multiple_rows_single_fsync(self, tmp_path):
        """A batch of rows should trigger exactly one fsync."""
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        rows = [_make_row(uid=f"u{i}", stmt_key=f"sk{i}", batch=f"b{i}") for i in range(5)]
        with patch("icepick.batcher.ledger.os.fsync") as mock_fsync:
            ledger.append_all(rows)
            assert mock_fsync.call_count == 1


# ---------------------------------------------------------------------------
# log_skip
# ---------------------------------------------------------------------------


class TestLogSkip:
    def test_log_skip_writes_jsonl(self, tmp_path):
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        skip = {"uid": "u1", "reason": "stmt_conflict", "batch": "batch10"}
        ledger.log_skip(skip)
        skips_path = d / "cross_source_skips.jsonl"
        assert skips_path.exists()
        data = json.loads(skips_path.read_text().strip())
        assert data["uid"] == "u1"

    def test_log_skip_fsync_called(self, tmp_path):
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        with patch("icepick.batcher.ledger.os.fsync") as mock_fsync:
            ledger.log_skip({"uid": "u1"})
            assert mock_fsync.call_count == 1

    def test_log_skip_appends(self, tmp_path):
        d = tmp_path / "ledger"
        ledger = Ledger.load(d)
        ledger.log_skip({"uid": "u1"})
        ledger.log_skip({"uid": "u2"})
        lines = [
            ln
            for ln in (d / "cross_source_skips.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) == 2

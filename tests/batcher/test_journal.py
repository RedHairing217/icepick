"""Tests for src/icepick/batcher/journal.py.

All fixtures are synthetic (pytest tmp_path); no real journal files are read.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from icepick.batcher.journal import (
    CursorStore,
    JournalCorruption,
    JournalRow,
    JournalTailer,
    journal_quiet_seconds,
    read_manifest_source_name,
    run_concluded,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict], *, torn_tail: bytes | None = None) -> None:
    """Write rows as a JSONL file; optionally append torn_tail bytes (no newline)."""
    with path.open("wb") as fh:
        for row in rows:
            fh.write(json.dumps(row).encode() + b"\n")
        if torn_tail is not None:
            fh.write(torn_tail)
        fh.flush()


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("ab") as fh:
        for row in rows:
            fh.write(json.dumps(row).encode() + b"\n")
        fh.flush()


def _make_row(n: int) -> dict:
    return {"arxiv_id": f"2601.{n:05d}", "candidate": {"statement": f"Theorem {n}"}}


def _fresh_tailer(journal: Path, cursor: CursorStore) -> JournalTailer:
    return JournalTailer(journal, cursor)


def _fresh_cursor(store_path: Path) -> CursorStore:
    cs = CursorStore(store_path)
    cs.load()
    return cs


# ---------------------------------------------------------------------------
# JournalRow dataclass
# ---------------------------------------------------------------------------


class TestJournalRow:
    def test_fields(self):
        row = JournalRow(
            line_no=1,
            byte_start=0,
            byte_end=42,
            raw=b'{"x": 1}\n',
            row={"x": 1},
        )
        assert row.line_no == 1
        assert row.byte_start == 0
        assert row.byte_end == 42
        assert row.raw == b'{"x": 1}\n'
        assert row.row == {"x": 1}


# ---------------------------------------------------------------------------
# CursorStore — load/save/get/advance
# ---------------------------------------------------------------------------


class TestCursorStore:
    def test_load_missing_file_gives_empty_state(self, tmp_path):
        cs = _fresh_cursor(tmp_path / "cursor.json")
        assert cs.get(tmp_path / "journal.jsonl") == (0, 0)

    def test_save_creates_file(self, tmp_path):
        cursor_path = tmp_path / "cursor.json"
        cs = _fresh_cursor(cursor_path)
        cs.save()
        assert cursor_path.exists()

    def test_get_returns_defaults_for_unknown_journal(self, tmp_path):
        cs = _fresh_cursor(tmp_path / "cursor.json")
        assert cs.get(tmp_path / "nonexistent.jsonl") == (0, 0)

    def test_advance_updates_in_memory_only(self, tmp_path):
        cursor_path = tmp_path / "cursor.json"
        journal = tmp_path / "journal.jsonl"
        cs = _fresh_cursor(cursor_path)
        row = JournalRow(line_no=3, byte_start=0, byte_end=50, raw=b"x\n", row={})
        cs.advance(journal, row)
        # in-memory updated
        assert cs.get(journal) == (3, 50)
        # file not written yet
        assert not cursor_path.exists()

    def test_advance_then_save_persists(self, tmp_path):
        cursor_path = tmp_path / "cursor.json"
        journal = tmp_path / "journal.jsonl"
        cs = _fresh_cursor(cursor_path)
        row = JournalRow(line_no=5, byte_start=0, byte_end=99, raw=b"x\n", row={})
        cs.advance(journal, row)
        cs.save()
        # reload from disk
        cs2 = _fresh_cursor(cursor_path)
        assert cs2.get(journal) == (5, 99)

    def test_save_is_atomic_tmp_file_gone_after(self, tmp_path):
        cursor_path = tmp_path / "cursor.json"
        cs = _fresh_cursor(cursor_path)
        cs.save()
        tmp_file = cursor_path.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_simulated_crash_tmp_exists_original_intact(self, tmp_path):
        """If process dies between writing tmp and os.replace, original is intact."""
        cursor_path = tmp_path / "cursor.json"
        journal = tmp_path / "journal.jsonl"
        # Write an initial known-good cursor
        cs = _fresh_cursor(cursor_path)
        row = JournalRow(line_no=2, byte_start=0, byte_end=30, raw=b"x\n", row={})
        cs.advance(journal, row)
        cs.save()
        # Simulate crash: leave a half-written tmp file (won't be swapped in)
        tmp_file = cursor_path.with_suffix(".tmp")
        tmp_file.write_text('{"journals": {}}')  # corrupted/incomplete tmp
        # load should still return the original good cursor
        cs2 = _fresh_cursor(cursor_path)
        assert cs2.get(journal) == (2, 30)

    def test_corrupt_cursor_file_gives_empty_state(self, tmp_path):
        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text("not json at all")
        cs = _fresh_cursor(cursor_path)
        assert cs.get(tmp_path / "x.jsonl") == (0, 0)


# ---------------------------------------------------------------------------
# JournalTailer — basic reading
# ---------------------------------------------------------------------------


class TestJournalTailerBasic:
    def test_empty_journal_returns_empty(self, tmp_path):
        journal = tmp_path / "candidates.jsonl"
        journal.write_bytes(b"")
        cs = _fresh_cursor(tmp_path / "cursor.json")
        tailer = _fresh_tailer(journal, cs)
        rows = tailer.read_new()
        assert rows == []

    def test_missing_journal_returns_empty(self, tmp_path):
        journal = tmp_path / "candidates.jsonl"
        cs = _fresh_cursor(tmp_path / "cursor.json")
        tailer = _fresh_tailer(journal, cs)
        assert tailer.read_new() == []

    def test_reads_rows(self, tmp_path):
        journal = tmp_path / "candidates.jsonl"
        data = [_make_row(i) for i in range(5)]
        _write_jsonl(journal, data)
        cs = _fresh_cursor(tmp_path / "cursor.json")
        tailer = _fresh_tailer(journal, cs)
        rows = tailer.read_new()
        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert row.row == data[i]
            assert row.line_no == i + 1

    def test_line_no_is_1_based(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        _write_jsonl(journal, [_make_row(1)])
        cs = _fresh_cursor(tmp_path / "cursor.json")
        rows = _fresh_tailer(journal, cs).read_new()
        assert rows[0].line_no == 1

    def test_byte_end_is_exclusive_and_includes_newline(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        raw = json.dumps(_make_row(1)).encode() + b"\n"
        journal.write_bytes(raw)
        cs = _fresh_cursor(tmp_path / "cursor.json")
        rows = _fresh_tailer(journal, cs).read_new()
        assert rows[0].byte_start == 0
        assert rows[0].byte_end == len(raw)
        assert rows[0].raw == raw

    def test_row_raw_matches_file_bytes(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        raw = b'{"arxiv_id": "2601.00001", "candidate": {"statement": "T1"}}\n'
        journal.write_bytes(raw)
        cs = _fresh_cursor(tmp_path / "cursor.json")
        rows = _fresh_tailer(journal, cs).read_new()
        assert rows[0].raw == raw


# ---------------------------------------------------------------------------
# Growth between calls
# ---------------------------------------------------------------------------


class TestJournalTailerGrowth:
    def test_growth_between_calls(self, tmp_path):
        journal = tmp_path / "candidates.jsonl"
        cursor_path = tmp_path / "cursor.json"

        batch1 = [_make_row(i) for i in range(3)]
        _write_jsonl(journal, batch1)

        cs = _fresh_cursor(cursor_path)
        tailer = _fresh_tailer(journal, cs)
        rows1 = tailer.read_new()
        assert len(rows1) == 3
        cs.save()

        # Append more rows
        batch2 = [_make_row(i) for i in range(3, 6)]
        _append_jsonl(journal, batch2)

        # Reload cursor from disk (simulate process restart)
        cs2 = _fresh_cursor(cursor_path)
        tailer2 = _fresh_tailer(journal, cs2)
        rows2 = tailer2.read_new()
        assert len(rows2) == 3
        for i, row in enumerate(rows2):
            assert row.row == batch2[i]
        # Line numbers continue from where we left off
        assert rows2[0].line_no == 4

    def test_no_double_read_without_save(self, tmp_path):
        """In-memory cursor advance prevents re-reading in same tailer instance."""
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"
        _write_jsonl(journal, [_make_row(0), _make_row(1)])

        cs = _fresh_cursor(cursor_path)
        tailer = _fresh_tailer(journal, cs)

        rows1 = tailer.read_new()
        assert len(rows1) == 2

        # No new lines added — second call returns nothing
        rows2 = tailer.read_new()
        assert rows2 == []


# ---------------------------------------------------------------------------
# Torn tail handling
# ---------------------------------------------------------------------------


class TestTornTail:
    def test_torn_tail_no_newline_excluded(self, tmp_path):
        """A line without trailing newline is treated as in-progress write; excluded."""
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"

        complete = [_make_row(i) for i in range(2)]
        torn = json.dumps(_make_row(99)).encode()  # no \n
        _write_jsonl(journal, complete, torn_tail=torn)

        cs = _fresh_cursor(cursor_path)
        rows = _fresh_tailer(journal, cs).read_new()
        assert len(rows) == 2
        assert rows[-1].row == complete[-1]

    def test_torn_tail_included_after_newline_appended(self, tmp_path):
        """After the writer completes the line, a second call surfaces it."""
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"

        complete = [_make_row(0)]
        torn_bytes = json.dumps(_make_row(1)).encode()
        _write_jsonl(journal, complete, torn_tail=torn_bytes)

        cs = _fresh_cursor(cursor_path)
        tailer = _fresh_tailer(journal, cs)
        rows1 = tailer.read_new()
        assert len(rows1) == 1
        cs.save()

        # Writer finishes the line
        with journal.open("ab") as fh:
            fh.write(b"\n")
            fh.flush()

        cs2 = _fresh_cursor(cursor_path)
        tailer2 = _fresh_tailer(journal, cs2)
        rows2 = tailer2.read_new()
        assert len(rows2) == 1
        assert rows2[0].row == _make_row(1)
        assert rows2[0].line_no == 2

    def test_torn_tail_cursor_not_advanced_past_it(self, tmp_path):
        """Cursor byte_offset must not include the torn line after read."""
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"

        complete_raw = json.dumps(_make_row(0)).encode() + b"\n"
        torn_raw = json.dumps(_make_row(1)).encode()  # no newline
        journal.write_bytes(complete_raw + torn_raw)

        cs = _fresh_cursor(cursor_path)
        tailer = _fresh_tailer(journal, cs)
        tailer.read_new()
        cs.save()

        cs2 = _fresh_cursor(cursor_path)
        _, byte_offset = cs2.get(journal.resolve())
        # byte_offset should be exactly at the end of the complete line
        assert byte_offset == len(complete_raw)


# ---------------------------------------------------------------------------
# JournalCorruption on bad JSON in complete line
# ---------------------------------------------------------------------------


class TestJournalCorruption:
    def test_newline_terminated_bad_json_raises(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        cs = _fresh_cursor(tmp_path / "cursor.json")

        good_line = json.dumps(_make_row(0)).encode() + b"\n"
        bad_line = b"not valid json at all\n"
        journal.write_bytes(good_line + bad_line)

        tailer = _fresh_tailer(journal, cs)
        with pytest.raises(JournalCorruption) as exc_info:
            tailer.read_new()

        err = exc_info.value
        assert err.line_no == 2
        assert err.path == journal.resolve()
        assert len(err.snippet) <= 120

    def test_corruption_snippet_max_120_chars(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        cs = _fresh_cursor(tmp_path / "cursor.json")

        # A line with more than 120 bytes of garbage
        garbage = b"x" * 200 + b"\n"
        journal.write_bytes(garbage)

        tailer = _fresh_tailer(journal, cs)
        with pytest.raises(JournalCorruption) as exc_info:
            tailer.read_new()
        assert len(exc_info.value.snippet) <= 120


# ---------------------------------------------------------------------------
# Empty line handling
# ---------------------------------------------------------------------------


class TestEmptyLines:
    def test_empty_lines_skipped_but_counted(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"

        r0 = json.dumps(_make_row(0)).encode() + b"\n"
        empty = b"\n"
        r1 = json.dumps(_make_row(1)).encode() + b"\n"
        journal.write_bytes(r0 + empty + r1)

        cs = _fresh_cursor(cursor_path)
        rows = _fresh_tailer(journal, cs).read_new()

        # Only 2 rows returned (empty line skipped)
        assert len(rows) == 2
        assert rows[0].line_no == 1
        # Empty line counted as line 2, so next data line is 3
        assert rows[1].line_no == 3

    def test_empty_lines_do_not_corrupt(self, tmp_path):
        """Empty lines between valid records should not raise JournalCorruption."""
        journal = tmp_path / "j.jsonl"
        journal.write_bytes(b"\n" * 5 + json.dumps(_make_row(0)).encode() + b"\n")
        cs = _fresh_cursor(tmp_path / "cursor.json")
        rows = _fresh_tailer(journal, cs).read_new()
        assert len(rows) == 1
        assert rows[0].line_no == 6

    def test_cursor_advanced_past_empty_lines(self, tmp_path):
        """Cursor byte_offset skips over empty lines so they are not re-read."""
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"
        r0 = json.dumps(_make_row(0)).encode() + b"\n"
        empty = b"\n"
        journal.write_bytes(r0 + empty)

        cs = _fresh_cursor(cursor_path)
        tailer = _fresh_tailer(journal, cs)
        tailer.read_new()
        cs.save()

        cs2 = _fresh_cursor(cursor_path)
        _, byte_offset = cs2.get(journal.resolve())
        assert byte_offset == len(r0) + len(empty)


# ---------------------------------------------------------------------------
# Limit parameter
# ---------------------------------------------------------------------------


class TestLimit:
    def test_limit_caps_returned_rows(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"
        _write_jsonl(journal, [_make_row(i) for i in range(10)])

        cs = _fresh_cursor(cursor_path)
        rows = _fresh_tailer(journal, cs).read_new(limit=3)
        assert len(rows) == 3

    def test_limit_cursor_advanced_only_to_limit(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"
        data = [_make_row(i) for i in range(5)]
        _write_jsonl(journal, data)

        cs = _fresh_cursor(cursor_path)
        tailer = _fresh_tailer(journal, cs)
        rows = tailer.read_new(limit=2)
        cs.save()

        # Next call gets the remaining 3
        cs2 = _fresh_cursor(cursor_path)
        rows2 = _fresh_tailer(journal, cs2).read_new()
        assert len(rows2) == 3
        assert rows2[0].row == data[2]

    def test_limit_none_reads_all(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        _write_jsonl(journal, [_make_row(i) for i in range(7)])
        cs = _fresh_cursor(tmp_path / "cursor.json")
        rows = _fresh_tailer(journal, cs).read_new(limit=None)
        assert len(rows) == 7


# ---------------------------------------------------------------------------
# Byte-offset resume exactness across process restarts
# ---------------------------------------------------------------------------


class TestByteOffsetResume:
    def test_resume_exactness_across_restart(self, tmp_path):
        """New CursorStore instance resumes at exact byte boundary."""
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"

        batch1 = [_make_row(i) for i in range(4)]
        batch2 = [_make_row(i) for i in range(4, 8)]
        _write_jsonl(journal, batch1)

        # Session 1: read batch1, save cursor
        cs1 = _fresh_cursor(cursor_path)
        t1 = _fresh_tailer(journal, cs1)
        rows1 = t1.read_new()
        assert len(rows1) == 4
        cs1.save()

        # Append batch2 (simulating continued extraction)
        _append_jsonl(journal, batch2)

        # Session 2: fresh CursorStore — must pick up exactly from byte boundary
        cs2 = _fresh_cursor(cursor_path)
        t2 = _fresh_tailer(journal, cs2)
        rows2 = t2.read_new()
        assert len(rows2) == 4
        for i, row in enumerate(rows2):
            assert row.row == batch2[i], f"row {i} mismatch"
        # No overlap
        for r1 in rows1:
            for r2 in rows2:
                assert r1.byte_start != r2.byte_start

    def test_multiple_restarts_no_loss_no_duplication(self, tmp_path):
        """Simulate 3 restart cycles; all rows seen exactly once."""
        journal = tmp_path / "j.jsonl"
        cursor_path = tmp_path / "cursor.json"
        all_seen = []

        for cycle in range(3):
            batch = [_make_row(cycle * 5 + i) for i in range(5)]
            _append_jsonl(journal, batch)

            cs = _fresh_cursor(cursor_path)
            t = _fresh_tailer(journal, cs)
            rows = t.read_new()
            all_seen.extend(rows)
            cs.save()

        assert len(all_seen) == 15
        # No duplicates by byte_start
        byte_starts = [r.byte_start for r in all_seen]
        assert len(byte_starts) == len(set(byte_starts))


# ---------------------------------------------------------------------------
# read_manifest_source_name helper
# ---------------------------------------------------------------------------


class TestReadManifestSourceName:
    def _make_manifest(self, tmp_path: Path, extra: dict | None = None) -> Path:
        """Write a synthetic manifest matching the real ApprovedManifest schema."""
        run_dir = tmp_path / "runs" / "20260101T000000Z"
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": "20260101T000000Z",
            "source_type": "arxiv_bulk",
            "processor_mode": "production",
            "requested_by": "cli",
            "requested_at": "2026-01-01T00:00:00Z",
            "approved_by": "nicky",
            "approved_at": "2026-01-01T00:00:00Z",
            "source_name": "arxiv_bulk_pde625",
            "target_count": 500,
            "call_budget": 0,
            "judge_enabled": False,
            "confirmation_enabled": False,
            "enable_leakage": False,
            "enable_duplication": False,
            "enable_robustness": False,
            "output_dir": "out/intake",
        }
        if extra:
            manifest.update(extra)
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        return run_dir

    def test_returns_source_name(self, tmp_path):
        run_dir = self._make_manifest(tmp_path)
        assert read_manifest_source_name(run_dir) == "arxiv_bulk_pde625"

    def test_raises_keyerror_if_absent(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Manifest without source_name
        payload = {"run_id": "x", "source_type": "arxiv_bulk"}
        (run_dir / "manifest.json").write_text(json.dumps(payload))
        with pytest.raises(KeyError, match="source_name"):
            read_manifest_source_name(run_dir)

    def test_keyerror_message_lists_present_fields(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        payload = {"run_id": "x", "other_field": "y"}
        (run_dir / "manifest.json").write_text(json.dumps(payload))
        with pytest.raises(KeyError) as exc_info:
            read_manifest_source_name(run_dir)
        # The error message should mention source_name
        assert "source_name" in str(exc_info.value)

    def test_different_source_name_values(self, tmp_path):
        run_dir = self._make_manifest(tmp_path, {"source_name": "pde_diverse_qa_500"})
        assert read_manifest_source_name(run_dir) == "pde_diverse_qa_500"


# ---------------------------------------------------------------------------
# run_concluded
# ---------------------------------------------------------------------------


class TestRunConcluded:
    def test_concluded_when_no_incomplete_marker(self, tmp_path):
        run_dir = tmp_path / "run"
        progress = run_dir / "_progress"
        progress.mkdir(parents=True)
        assert run_concluded(run_dir) is True

    def test_not_concluded_when_incomplete_exists(self, tmp_path):
        run_dir = tmp_path / "run"
        progress = run_dir / "_progress"
        progress.mkdir(parents=True)
        (progress / "INCOMPLETE").write_text("run in progress\n")
        assert run_concluded(run_dir) is False

    def test_concluded_when_no_progress_dir(self, tmp_path):
        """No progress dir at all → marker can't exist → concluded."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert run_concluded(run_dir) is True

    def test_incomplete_removed_signals_concluded(self, tmp_path):
        run_dir = tmp_path / "run"
        progress = run_dir / "_progress"
        progress.mkdir(parents=True)
        marker = progress / "INCOMPLETE"
        marker.write_text("in progress\n")
        assert run_concluded(run_dir) is False
        marker.unlink()
        assert run_concluded(run_dir) is True


# ---------------------------------------------------------------------------
# journal_quiet_seconds
# ---------------------------------------------------------------------------


class TestJournalQuietSeconds:
    def test_returns_inf_for_missing_file(self, tmp_path):
        assert journal_quiet_seconds(tmp_path / "nonexistent.jsonl") == float("inf")

    def test_returns_nonnegative_float(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        journal.write_bytes(b"")
        secs = journal_quiet_seconds(journal)
        assert isinstance(secs, float)
        assert secs >= 0.0

    def test_recent_file_has_small_quiet_seconds(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        journal.write_bytes(b"")
        # Just created — should be very fresh
        secs = journal_quiet_seconds(journal)
        assert secs < 5.0  # generous bound


# ---------------------------------------------------------------------------
# Cursor advance/save ordering contract
# ---------------------------------------------------------------------------


class TestCursorOrderingContract:
    def test_save_must_be_called_explicitly(self, tmp_path):
        """advance() does not persist; only save() does."""
        cursor_path = tmp_path / "cursor.json"
        journal = tmp_path / "j.jsonl"

        cs = _fresh_cursor(cursor_path)
        row = JournalRow(line_no=1, byte_start=0, byte_end=20, raw=b"x\n", row={})
        cs.advance(journal, row)
        # Deliberately do NOT call cs.save()

        cs2 = _fresh_cursor(cursor_path)
        # Should see default (0, 0) since save was never called
        assert cs2.get(journal) == (0, 0)

    def test_save_after_advance_persists_correct_position(self, tmp_path):
        cursor_path = tmp_path / "cursor.json"
        journal = tmp_path / "j.jsonl"
        data = [_make_row(i) for i in range(3)]
        _write_jsonl(journal, data)

        cs = _fresh_cursor(cursor_path)
        t = _fresh_tailer(journal, cs)
        rows = t.read_new()
        last_row = rows[-1]
        cs.save()

        cs2 = _fresh_cursor(cursor_path)
        line_count, byte_offset = cs2.get(journal.resolve())
        assert line_count == last_row.line_no
        assert byte_offset == last_row.byte_end

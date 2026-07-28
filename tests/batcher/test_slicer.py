"""Tests for src/icepick/batcher/slicer.py.

All journals are synthetic (written to tmp_path); no real extraction processes
are touched.  All tests use stdlib only.

Journal row shape: {"arxiv_id": ..., "candidate": {"statement": ..., "answer": ..., "metadata": {...}}}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List
from unittest.mock import patch, call

import pytest

from icepick.batcher.identity import compute_uid, stmt_key as make_stmt_key, content_hash
from icepick.batcher.journal import JournalTailer, CursorStore, JournalRow
from icepick.batcher.ledger import Ledger, LedgerRow
from icepick.batcher.slicer import (
    SliceConfig,
    SliceOutcome,
    cut_slice,
    recover_pending_slice,
    _make_dummy_journal_row,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOW = "2026-07-07T00:00:00Z"
SOURCE = "arxiv_bulk_pde625"


# ---------------------------------------------------------------------------
# Journal-building helpers
# ---------------------------------------------------------------------------


def _make_row_dict(
    arxiv_id: str = "2401.00001",
    statement: str = "The integral converges.",
    answer: str = "True",
    metadata: dict | None = None,
) -> dict:
    """Produce a raw journal row dict (as stored in candidates.jsonl)."""
    return {
        "arxiv_id": arxiv_id,
        "candidate": {
            "statement": statement,
            "answer": answer,
            "metadata": metadata or {},
        },
    }


def _write_journal(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL to path (each line newline-terminated)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_n_rows(n: int, prefix: str = "stmt") -> list[dict]:
    """Generate n distinct journal rows."""
    return [
        _make_row_dict(
            arxiv_id=f"2401.{i:05d}",
            statement=f"{prefix}_{i}: The equation holds.",
            answer="True",
        )
        for i in range(n)
    ]


def _make_tailer(journal_path: Path, cursor: CursorStore) -> JournalTailer:
    return JournalTailer(journal_path, cursor)


def _fresh(tmp_path: Path, rows: list[dict], batch_no: int = 10):
    """Build a fresh (journal, cursor, ledger, batches_root, config) fixture."""
    journal_path = tmp_path / "journal" / "candidates.jsonl"
    _write_journal(journal_path, rows)

    cursor_path = tmp_path / "state" / "cursor.json"
    cursor = CursorStore(cursor_path)
    cursor.load()

    ledger_dir = tmp_path / "ledger"
    ledger = Ledger.load(ledger_dir)

    batches_root = tmp_path / "batches"
    config = SliceConfig(campaign_source=SOURCE, slice_size=5)

    tailer = _make_tailer(journal_path, cursor)

    return journal_path, cursor, ledger, batches_root, config, tailer


# ---------------------------------------------------------------------------
# Basic 'sliced' path
# ---------------------------------------------------------------------------


class TestCutSliceBasic:
    def test_sliced_returns_sliced_kind(self, tmp_path):
        rows = _make_n_rows(7)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert outcome.kind == "sliced"
        assert outcome.counts["accepted"] == 5

    def test_batch_dir_created(self, tmp_path):
        rows = _make_n_rows(6)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert outcome.batch_dir == batches_root / "batch10"
        assert (batches_root / "batch10").is_dir()

    def test_slice_records_written(self, tmp_path):
        rows = _make_n_rows(6)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        records_path = batches_root / "batch10" / "slice_records.jsonl"
        assert records_path.exists()
        lines = [l for l in records_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 5

    def test_slice_records_have_uid_and_source(self, tmp_path):
        rows = _make_n_rows(5)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        records_path = batches_root / "batch10" / "slice_records.jsonl"
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            assert "uid" in rec
            assert rec["source"] == SOURCE
            # original fields preserved
            assert "statement" in rec
            assert "answer" in rec

    def test_manifest_written(self, tmp_path):
        rows = _make_n_rows(5)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        manifest_path = batches_root / "batch10" / "slice_manifest.json"
        assert manifest_path.exists()
        m = json.loads(manifest_path.read_text())
        assert m["batch"] == "batch10"
        assert m["campaign_source"] == SOURCE
        assert m["slice_size"] == 5
        assert len(m["entries"]) == 5
        assert m["counts"]["accepted"] == 5

    def test_ledger_populated(self, tmp_path):
        rows = _make_n_rows(5)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        ledger2 = Ledger.load(tmp_path / "ledger")
        assert len(ledger2._by_uid) == 5

    def test_cursor_advanced_after_commit(self, tmp_path):
        rows = _make_n_rows(7)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        line_count, byte_offset = cursor.get(journal_path)
        # Cursor should be at exactly the 5th accepted row.
        assert line_count == 5
        assert byte_offset > 0

    def test_cursor_persisted_to_disk(self, tmp_path):
        rows = _make_n_rows(7)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        cursor2 = CursorStore(tmp_path / "state" / "cursor.json")
        cursor2.load()
        lc, bo = cursor2.get(journal_path)
        assert lc == 5
        assert bo > 0

    def test_second_slice_uses_fresh_cursor(self, tmp_path):
        rows = _make_n_rows(12)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        o1 = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert o1.kind == "sliced"
        # Re-create tailer (cursor already advanced in-memory)
        tailer2 = _make_tailer(journal_path, cursor)
        o2 = cut_slice(tailer2, cursor, ledger, batches_root, 11, config, NOW)
        assert o2.kind == "sliced"
        assert o2.counts["accepted"] == 5


# ---------------------------------------------------------------------------
# Exactness: 1003 rows → 4 slices of 250, remainder 3
# ---------------------------------------------------------------------------


class TestExactness:
    def test_1003_rows_four_slices_remainder(self, tmp_path):
        """1003 rows → exactly 4 batches of 250 + 'insufficient' with 3."""
        rows = _make_n_rows(1003, prefix="exact")
        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)

        cursor_path = tmp_path / "state" / "cursor.json"
        cursor = CursorStore(cursor_path)
        cursor.load()

        ledger_dir = tmp_path / "ledger"
        ledger = Ledger.load(ledger_dir)

        batches_root = tmp_path / "batches"
        config = SliceConfig(campaign_source=SOURCE, slice_size=250)

        for batch_no in range(10, 14):
            tailer = _make_tailer(journal_path, cursor)
            outcome = cut_slice(tailer, cursor, ledger, batches_root, batch_no, config, NOW)
            assert outcome.kind == "sliced", f"batch {batch_no} expected sliced, got {outcome.kind}: {outcome.detail}"
            assert outcome.counts["accepted"] == 250

        # Capture state before the insufficient call.
        ledger_before = (tmp_path / "ledger" / "consumed_uids.jsonl").read_bytes()
        cursor_before = (cursor_path).read_bytes()
        batches_before = set(p.name for p in batches_root.iterdir())

        tailer5 = _make_tailer(journal_path, cursor)
        insuff = cut_slice(tailer5, cursor, ledger, batches_root, 14, config, NOW)
        assert insuff.kind == "insufficient"
        assert insuff.counts["pending_size"] == 3

        # Assert zero side effects from insufficient call.
        assert (tmp_path / "ledger" / "consumed_uids.jsonl").read_bytes() == ledger_before
        assert cursor_path.read_bytes() == cursor_before
        assert set(p.name for p in batches_root.iterdir()) == batches_before

    def test_cursor_parked_precisely_after_250th_row(self, tmp_path):
        """Cursor line_count must equal exactly 250 after first slice."""
        rows = _make_n_rows(300, prefix="cursor_test")
        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)

        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        ledger = Ledger.load(tmp_path / "ledger")
        config = SliceConfig(campaign_source=SOURCE, slice_size=250)

        tailer = _make_tailer(journal_path, cursor)
        cut_slice(tailer, cursor, ledger, tmp_path / "batches", 10, config, NOW)

        line_count, byte_offset = cursor.get(journal_path)
        assert line_count == 250

        # The byte_offset must point exactly at the start of line 251.
        with journal_path.open("rb") as fh:
            fh.seek(byte_offset)
            next_raw = fh.readline()
        next_row = json.loads(next_raw)
        # Row 251 (0-indexed: 250) should be the 251st row.
        assert next_row["arxiv_id"] == f"2401.{250:05d}"


# ---------------------------------------------------------------------------
# Replay: dup row mid-stream → still exactly N, dup logged once at commit
# ---------------------------------------------------------------------------


class TestReplay:
    def test_dup_row_causes_refill(self, tmp_path):
        """Inserting a duplicate in position 3 still produces exactly 5 accepted."""
        base_rows = _make_n_rows(9)
        # Duplicate row[2] at position 3 (between index 2 and 3).
        rows = base_rows[:3] + [base_rows[2]] + base_rows[3:]
        assert len(rows) == 10

        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        assert outcome.kind == "sliced"
        assert outcome.counts["accepted"] == 5
        assert outcome.counts["replay_skips"] == 1

    def test_dup_logged_in_manifest(self, tmp_path):
        """Duplicate is logged in the manifest skips list."""
        base_rows = _make_n_rows(9)
        rows = base_rows[:3] + [base_rows[2]] + base_rows[3:]

        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        m = json.loads((batches_root / "batch10" / "slice_manifest.json").read_text())
        replay_skips = [s for s in m["skips"] if s["kind"] == "replay"]
        assert len(replay_skips) == 1

    def test_intra_slice_dup_same_row_twice_in_read_batch(self, tmp_path):
        """Two identical rows in the same read batch: second collapses to replay."""
        row = _make_row_dict(statement="Unique theorem for intra dup test.")
        other_rows = _make_n_rows(6)
        rows = [row, row] + other_rows  # row appears twice at positions 0, 1

        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        assert outcome.kind == "sliced"
        assert outcome.counts["accepted"] == 5
        assert outcome.counts["replay_skips"] == 1

    def test_replay_from_ledger_history(self, tmp_path):
        """A row whose uid+content_hash is already in the ledger (historical) → replay+refill."""
        base_rows = _make_n_rows(7)
        # Pre-seed the first row into the ledger as history.
        first_row_dict = base_rows[0]
        first_statement = first_row_dict["candidate"]["statement"]
        uid = compute_uid(SOURCE, first_statement)
        sk = make_stmt_key(first_statement)
        ch = content_hash(first_row_dict)

        ledger_dir = tmp_path / "ledger"
        ledger = Ledger.load(ledger_dir)
        ledger.append_all([LedgerRow(
            uid=uid, stmt_key=sk, content_hash=ch,
            batch="hist:batch1", source_journal="old.jsonl",
            journal_line=1, sliced_at=NOW,
        )])

        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, base_rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        batches_root = tmp_path / "batches"
        config = SliceConfig(campaign_source=SOURCE, slice_size=5)
        tailer = _make_tailer(journal_path, cursor)

        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert outcome.kind == "sliced"
        assert outcome.counts["accepted"] == 5
        assert outcome.counts["replay_skips"] == 1


# ---------------------------------------------------------------------------
# uid_conflict abort
# ---------------------------------------------------------------------------


class TestUidConflict:
    def test_uid_conflict_aborts(self, tmp_path):
        """Same uid (same source+statement) but different answer → different content_hash → abort."""
        statement = "The integral of e^x is e^x."
        row_v1 = _make_row_dict(statement=statement, answer="True", arxiv_id="2401.00001")
        row_v2 = _make_row_dict(statement=statement, answer="False", arxiv_id="2401.00001")

        uid = compute_uid(SOURCE, statement)
        sk = make_stmt_key(statement)
        ch_v1 = content_hash(row_v1)

        ledger_dir = tmp_path / "ledger"
        ledger = Ledger.load(ledger_dir)
        ledger.append_all([LedgerRow(
            uid=uid, stmt_key=sk, content_hash=ch_v1,
            batch="hist:batch1", source_journal="old.jsonl",
            journal_line=1, sliced_at=NOW,
        )])

        other_rows = _make_n_rows(6)
        rows = [row_v2] + other_rows

        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        tailer = _make_tailer(journal_path, cursor)
        config = SliceConfig(campaign_source=SOURCE, slice_size=5)

        ledger_before = Ledger.load(ledger_dir)
        outcome = cut_slice(tailer, cursor, ledger, tmp_path / "batches", 10, config, NOW)

        assert outcome.kind == "aborted"
        assert outcome.detail == "uid_conflict"
        assert outcome.abort_info is not None
        assert outcome.abort_info["uid"] == uid

    def test_uid_conflict_nothing_committed(self, tmp_path):
        """Nothing is written to disk on uid_conflict abort."""
        statement = "Cauchy's theorem states..."
        row_v1 = _make_row_dict(statement=statement, answer="True")
        row_v2 = _make_row_dict(statement=statement, answer="Different")

        uid = compute_uid(SOURCE, statement)
        sk = make_stmt_key(statement)
        ch_v1 = content_hash(row_v1)

        ledger_dir = tmp_path / "ledger"
        ledger = Ledger.load(ledger_dir)
        ledger.append_all([LedgerRow(
            uid=uid, stmt_key=sk, content_hash=ch_v1,
            batch="hist:b1", source_journal="j.jsonl",
            journal_line=1, sliced_at=NOW,
        )])

        # Record state before.
        ledger_before_bytes = (ledger_dir / "consumed_uids.jsonl").read_bytes()

        other_rows = _make_n_rows(6)
        rows = [row_v2] + other_rows

        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        tailer = _make_tailer(journal_path, cursor)
        batches_root = tmp_path / "batches"

        cut_slice(tailer, cursor, ledger, batches_root, 10,
                  SliceConfig(campaign_source=SOURCE, slice_size=5), NOW)

        # Ledger unchanged.
        assert (ledger_dir / "consumed_uids.jsonl").read_bytes() == ledger_before_bytes
        # No batch dir created.
        assert not batches_root.exists()

    def test_intra_slice_uid_conflict_aborts(self, tmp_path):
        """Two rows with same statement but different answers within one read batch → abort."""
        statement = "The sum of angles is 180°."
        row_a = _make_row_dict(statement=statement, answer="True")
        row_b = _make_row_dict(statement=statement, answer="False")
        other = _make_n_rows(6)
        rows = [row_a] + other[:2] + [row_b] + other[2:]

        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        assert outcome.kind == "aborted"
        assert outcome.detail == "uid_conflict"

    def test_abort_info_fields_complete(self, tmp_path):
        """abort_info contains all required fields."""
        statement = "Complete abort_info test statement."
        row_v1 = _make_row_dict(statement=statement, answer="A")
        row_v2 = _make_row_dict(statement=statement, answer="B")

        uid = compute_uid(SOURCE, statement)
        sk = make_stmt_key(statement)
        ch_v1 = content_hash(row_v1)

        ledger_dir = tmp_path / "ledger"
        ledger = Ledger.load(ledger_dir)
        ledger.append_all([LedgerRow(
            uid=uid, stmt_key=sk, content_hash=ch_v1,
            batch="hist:b1", source_journal="j.jsonl",
            journal_line=5, sliced_at=NOW,
        )])

        rows = [row_v2] + _make_n_rows(6)
        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        tailer = _make_tailer(journal_path, cursor)

        outcome = cut_slice(tailer, cursor, ledger, tmp_path / "batches", 10,
                            SliceConfig(campaign_source=SOURCE, slice_size=5), NOW)

        ai = outcome.abort_info
        for field in ("uid", "stmt_key", "journal_line", "prior_batch",
                      "prior_journal_line", "prior_content_hash", "new_content_hash"):
            assert field in ai, f"abort_info missing '{field}'"


# ---------------------------------------------------------------------------
# Statement policy: skip / abort / allow
# ---------------------------------------------------------------------------


def _make_stmt_variant_rows(base_stmt: str, n_other: int = 6):
    """Make a whitespace/case variant of base_stmt (same stmt_key, different uid)."""
    # Different case + whitespace → same normalised form → same stmt_key.
    variant_stmt = "  " + base_stmt.upper() + "  "
    # Confirm they differ as uids but match as stmt_keys.
    assert compute_uid(SOURCE, base_stmt) != compute_uid(SOURCE, variant_stmt)
    assert make_stmt_key(base_stmt) == make_stmt_key(variant_stmt)

    row_original = _make_row_dict(statement=base_stmt, answer="True", arxiv_id="2401.99901")
    row_variant = _make_row_dict(statement=variant_stmt, answer="True", arxiv_id="2401.99902")
    other_rows = _make_n_rows(n_other)
    return row_original, row_variant, other_rows


class TestStmtPolicies:
    def _seed_ledger(self, ledger_dir, row_original):
        """Pre-seed the ledger with row_original under the original statement."""
        stmt = row_original["candidate"]["statement"]
        uid = compute_uid(SOURCE, stmt)
        sk = make_stmt_key(stmt)
        ch = content_hash(row_original)
        ledger = Ledger.load(ledger_dir)
        ledger.append_all([LedgerRow(
            uid=uid, stmt_key=sk, content_hash=ch,
            batch="hist:b1", source_journal="old.jsonl",
            journal_line=1, sliced_at=NOW,
        )])
        return ledger

    def test_skip_policy_skips_and_refills(self, tmp_path):
        base_stmt = "The fourier transform of a Gaussian is Gaussian."
        row_orig, row_variant, other = _make_stmt_variant_rows(base_stmt)
        ledger_dir = tmp_path / "ledger"
        ledger = self._seed_ledger(ledger_dir, row_orig)

        # Journal: variant row at position 0, then 8 others.
        rows = [row_variant] + other
        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        tailer = _make_tailer(journal_path, cursor)
        config = SliceConfig(campaign_source=SOURCE, slice_size=5,
                             cross_source_statement_policy="skip")
        batches_root = tmp_path / "batches"

        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        assert outcome.kind == "sliced"
        assert outcome.counts["accepted"] == 5
        assert outcome.counts["stmt_skips"] == 1

    def test_skip_policy_logs_skip(self, tmp_path):
        base_stmt = "Parseval's theorem holds for L2 functions."
        row_orig, row_variant, other = _make_stmt_variant_rows(base_stmt)
        ledger_dir = tmp_path / "ledger"
        ledger = self._seed_ledger(ledger_dir, row_orig)

        rows = [row_variant] + other
        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        tailer = _make_tailer(journal_path, cursor)
        config = SliceConfig(campaign_source=SOURCE, slice_size=5,
                             cross_source_statement_policy="skip")

        cut_slice(tailer, cursor, ledger, tmp_path / "batches", 10, config, NOW)

        skips_path = ledger_dir / "cross_source_skips.jsonl"
        assert skips_path.exists()
        lines = [l for l in skips_path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1
        skip_entry = json.loads(lines[0])
        assert skip_entry["kind"] == "stmt_skip"

    def test_abort_policy_aborts_on_stmt_conflict(self, tmp_path):
        base_stmt = "Green's theorem relates line integrals."
        row_orig, row_variant, other = _make_stmt_variant_rows(base_stmt)
        ledger_dir = tmp_path / "ledger"
        ledger = self._seed_ledger(ledger_dir, row_orig)

        rows = [row_variant] + other
        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        tailer = _make_tailer(journal_path, cursor)
        config = SliceConfig(campaign_source=SOURCE, slice_size=5,
                             cross_source_statement_policy="abort")

        outcome = cut_slice(tailer, cursor, ledger, tmp_path / "batches", 10, config, NOW)

        assert outcome.kind == "aborted"
        assert outcome.detail == "stmt_conflict"
        assert outcome.abort_info is not None

    def test_allow_policy_accepts_and_warns(self, tmp_path):
        base_stmt = "Stokes theorem generalises Green's theorem."
        row_orig, row_variant, other = _make_stmt_variant_rows(base_stmt)
        ledger_dir = tmp_path / "ledger"
        ledger = self._seed_ledger(ledger_dir, row_orig)

        rows = [row_variant] + other
        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        tailer = _make_tailer(journal_path, cursor)
        config = SliceConfig(campaign_source=SOURCE, slice_size=5,
                             cross_source_statement_policy="allow")

        outcome = cut_slice(tailer, cursor, ledger, tmp_path / "batches", 10, config, NOW)

        assert outcome.kind == "sliced"
        assert outcome.counts["accepted"] == 5
        assert outcome.counts["warns"] >= 1


# ---------------------------------------------------------------------------
# Warn-set acceptance
# ---------------------------------------------------------------------------


class TestWarnSet:
    def test_warn_set_hit_accepted_with_warn_count(self, tmp_path):
        """A uid that is in the warn-only index → accepted + warn counted."""
        statement = "The residue theorem applies here."
        uid = compute_uid(SOURCE, statement)
        sk = make_stmt_key(statement)
        row_dict = _make_row_dict(statement=statement, answer="Yes")
        ch = content_hash(row_dict)

        ledger_dir = tmp_path / "ledger"
        ledger = Ledger.load(ledger_dir)
        ledger.append_all([LedgerRow(
            uid=uid, stmt_key=sk, content_hash=ch,
            batch="hist:warn_orphan", source_journal="orphan.jsonl",
            journal_line=1, sliced_at=NOW, warn_only=True,
        )])

        other_rows = _make_n_rows(6)
        rows = [row_dict] + other_rows

        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        tailer = _make_tailer(journal_path, cursor)
        config = SliceConfig(campaign_source=SOURCE, slice_size=5)

        outcome = cut_slice(tailer, cursor, ledger, tmp_path / "batches", 10, config, NOW)

        assert outcome.kind == "sliced"
        assert outcome.counts["accepted"] == 5
        assert outcome.counts["warns"] >= 1


# ---------------------------------------------------------------------------
# Insufficient (not enough rows)
# ---------------------------------------------------------------------------


class TestInsufficient:
    def test_insufficient_when_too_few_rows(self, tmp_path):
        rows = _make_n_rows(3)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert outcome.kind == "insufficient"
        assert outcome.counts["pending_size"] == 3

    def test_insufficient_zero_side_effects_ledger(self, tmp_path):
        rows = _make_n_rows(4)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)

        ledger_path = tmp_path / "ledger" / "consumed_uids.jsonl"
        before = ledger_path.read_bytes() if ledger_path.exists() else b""
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        after = ledger_path.read_bytes() if ledger_path.exists() else b""
        assert before == after

    def test_insufficient_zero_side_effects_no_batch_dir(self, tmp_path):
        rows = _make_n_rows(2)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert not batches_root.exists()

    def test_insufficient_zero_side_effects_cursor_unchanged(self, tmp_path):
        rows = _make_n_rows(3)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cursor_file = tmp_path / "state" / "cursor.json"
        before = cursor_file.read_bytes() if cursor_file.exists() else b""
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        after = cursor_file.read_bytes() if cursor_file.exists() else b""
        assert before == after


# ---------------------------------------------------------------------------
# Journal row validation
# ---------------------------------------------------------------------------


class TestJournalRowValidation:
    def test_missing_candidate_aborts(self, tmp_path):
        rows = [{"arxiv_id": "x", "no_candidate": "oops"}, *_make_n_rows(6)]
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert outcome.kind == "aborted"
        assert outcome.detail == "journal_row_invalid"
        assert outcome.abort_info["journal_line"] == 1

    def test_empty_statement_aborts(self, tmp_path):
        rows = [{"arxiv_id": "x", "candidate": {"statement": "", "answer": "y"}}, *_make_n_rows(6)]
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert outcome.kind == "aborted"
        assert outcome.detail == "journal_row_invalid"

    def test_missing_statement_aborts(self, tmp_path):
        rows = [{"arxiv_id": "x", "candidate": {"answer": "y"}}, *_make_n_rows(6)]
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert outcome.kind == "aborted"
        assert outcome.detail == "journal_row_invalid"

    def test_invalid_row_after_good_rows_still_aborts(self, tmp_path):
        good = _make_n_rows(3)
        bad = [{"arxiv_id": "x", "candidate": {"statement": ""}}]
        more_good = _make_n_rows(4, prefix="more")
        rows = good + bad + more_good
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        outcome = cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        assert outcome.kind == "aborted"
        assert outcome.detail == "journal_row_invalid"


# ---------------------------------------------------------------------------
# batch_dir_conflict
# ---------------------------------------------------------------------------


class TestBatchDirConflict:
    def test_batch_dir_conflict_when_manifest_exists(self, tmp_path):
        rows = _make_n_rows(6)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)

        # First slice succeeds.
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        # Reset cursor and ledger to simulate re-try with same batch_no.
        new_cursor = CursorStore(tmp_path / "state2" / "cursor.json")
        new_cursor.load()
        rows2 = _make_n_rows(6, prefix="other")
        journal_path2 = tmp_path / "journal2" / "candidates.jsonl"
        _write_journal(journal_path2, rows2)
        new_ledger = Ledger.load(tmp_path / "ledger2")
        tailer2 = _make_tailer(journal_path2, new_cursor)

        outcome = cut_slice(tailer2, new_cursor, new_ledger, batches_root, 10, config, NOW)

        assert outcome.kind == "aborted"
        assert outcome.detail == "batch_dir_conflict"


# ---------------------------------------------------------------------------
# Crash injection at every commit boundary
# ---------------------------------------------------------------------------


def _run_cut_slice_clean(tmp_path_base: Path, rows: list[dict], batch_no: int = 10):
    """Run cut_slice to completion, return (outcome, final_ledger, final_cursor)."""
    journal_path = tmp_path_base / "journal" / "candidates.jsonl"
    _write_journal(journal_path, rows)
    cursor = CursorStore(tmp_path_base / "state" / "cursor.json")
    cursor.load()
    ledger = Ledger.load(tmp_path_base / "ledger")
    batches_root = tmp_path_base / "batches"
    config = SliceConfig(campaign_source=SOURCE, slice_size=5)
    tailer = _make_tailer(journal_path, cursor)
    outcome = cut_slice(tailer, cursor, ledger, batches_root, batch_no, config, NOW)
    return outcome, ledger, cursor


class TestCrashRecovery:
    """Test crash injection at every step boundary b→c→d→e→f."""

    def _reference_state(self, tmp_path):
        """Run a clean cut_slice and capture the reference final state."""
        rows = _make_n_rows(7)
        journal_path = tmp_path / "ref" / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "ref" / "state" / "cursor.json")
        cursor.load()
        ledger = Ledger.load(tmp_path / "ref" / "ledger")
        batches_root = tmp_path / "ref" / "batches"
        config = SliceConfig(campaign_source=SOURCE, slice_size=5)
        tailer = _make_tailer(journal_path, cursor)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        ref_uids = set(ledger._by_uid.keys())
        ref_manifest = json.loads((batches_root / "batch10" / "slice_manifest.json").read_text())
        ref_cursor_line, ref_cursor_byte = cursor.get(journal_path)
        return ref_uids, ref_manifest, ref_cursor_line, ref_cursor_byte

    def _setup_crash(self, tmp_path, rows):
        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        ledger = Ledger.load(tmp_path / "ledger")
        batches_root = tmp_path / "batches"
        config = SliceConfig(campaign_source=SOURCE, slice_size=5)
        tailer = _make_tailer(journal_path, cursor)
        return journal_path, cursor, ledger, batches_root, config, tailer

    def test_crash_between_b_and_c_tmp_file_cleaned(self, tmp_path):
        """Crash after writing slice_records.jsonl.tmp but before os.replace → tmp removed."""
        rows = _make_n_rows(7)
        journal_path, cursor, ledger, batches_root, config, tailer = self._setup_crash(
            tmp_path, rows)

        call_count = {"n": 0}
        original_replace = os.replace

        def crash_on_first_replace(src, dst):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Simulate crash: leave the tmp file in place, don't replace.
                raise OSError("simulated crash between b and c")
            return original_replace(src, dst)

        with patch("icepick.batcher.slicer.os.replace", side_effect=crash_on_first_replace):
            with pytest.raises(OSError, match="simulated crash between b and c"):
                cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        # recover_pending_slice should see no manifest and clean up tmp files.
        def tailer_factory(path):
            return _make_tailer(path, cursor)

        result = recover_pending_slice(batches_root, ledger, cursor, tailer_factory)
        assert "recomputable" in result

        # No .tmp files should remain.
        batch_dir = batches_root / "batch10"
        assert not list(batch_dir.glob("*.tmp"))

    def test_crash_between_c_and_d_manifest_exists_recovery(self, tmp_path):
        """Crash after manifest written but before ledger.append_all → recovery re-runs d-f."""
        rows = _make_n_rows(7)
        journal_path, cursor, ledger, batches_root, config, tailer = self._setup_crash(
            tmp_path, rows)

        original_append = ledger.append_all

        def crash_on_append(rows_arg):
            raise OSError("simulated crash at d")

        with patch.object(ledger, "append_all", side_effect=crash_on_append):
            with pytest.raises(OSError, match="simulated crash at d"):
                cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        # Manifest should exist (written in step c before step d).
        manifest_path = batches_root / "batch10" / "slice_manifest.json"
        assert manifest_path.exists()

        # Recovery: reload ledger (simulates fresh process).
        ledger2 = Ledger.load(tmp_path / "ledger")
        cursor2 = CursorStore(tmp_path / "state" / "cursor.json")
        cursor2.load()

        def tailer_factory(path):
            return _make_tailer(path, cursor2)

        result = recover_pending_slice(batches_root, ledger2, cursor2, tailer_factory)
        assert result.get("recovered") == "batch10"
        assert "ledger_append_all" in result.get("actions", [])

        # Ledger should now have exactly 5 uids.
        assert len(ledger2._by_uid) == 5

    def test_crash_between_d_and_e_idempotent_recovery(self, tmp_path):
        """Crash after ledger append but before log_skip → recovery re-runs e-f."""
        rows = _make_n_rows(7)
        # Add a stmt_skip scenario to exercise step e.
        base_stmt = "The divergence theorem holds in R3."
        row_orig = _make_row_dict(statement=base_stmt, answer="Yes")
        ch = content_hash(row_orig)
        uid = compute_uid(SOURCE, base_stmt)
        sk = make_stmt_key(base_stmt)

        ledger_dir = tmp_path / "ledger"
        ledger = Ledger.load(ledger_dir)
        ledger.append_all([LedgerRow(
            uid=uid, stmt_key=sk, content_hash=ch,
            batch="hist:b1", source_journal="old.jsonl",
            journal_line=1, sliced_at=NOW,
        )])

        # Variant of the row (same stmt_key, different uid) → stmt_conflict/skip.
        variant_stmt = "  " + base_stmt.upper() + "  "
        assert make_stmt_key(variant_stmt) == sk
        row_variant = _make_row_dict(statement=variant_stmt, answer="Yes")
        other_rows = _make_n_rows(7, prefix="crash_e")
        rows = [row_variant] + other_rows

        journal_path = tmp_path / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()
        tailer = _make_tailer(journal_path, cursor)
        config = SliceConfig(campaign_source=SOURCE, slice_size=5, cross_source_statement_policy="skip")

        log_call_count = {"n": 0}
        original_log_skip = ledger.log_skip

        def crash_on_first_log_skip(entry):
            log_call_count["n"] += 1
            if log_call_count["n"] == 1:
                raise OSError("simulated crash at e")
            return original_log_skip(entry)

        with patch.object(ledger, "log_skip", side_effect=crash_on_first_log_skip):
            with pytest.raises(OSError, match="simulated crash at e"):
                cut_slice(tailer, cursor, ledger, batches_root := tmp_path / "batches",
                          10, config, NOW)

        # Ledger has been written (step d completed).
        ledger2 = Ledger.load(ledger_dir)
        # The 5 new uids should be in ledger (step d ran).
        new_uids = set(ledger2._by_uid.keys()) - {uid}
        assert len(new_uids) == 5

        # Recovery: re-run e-f.
        cursor2 = CursorStore(tmp_path / "state" / "cursor.json")
        cursor2.load()

        def tailer_factory(path):
            return _make_tailer(path, cursor2)

        result = recover_pending_slice(tmp_path / "batches", ledger2, cursor2, tailer_factory)
        assert result.get("recovered") == "batch10"

    def test_crash_between_e_and_f_cursor_recovery(self, tmp_path):
        """Crash after log_skip but before cursor.save → cursor advanced in recovery."""
        rows = _make_n_rows(7)
        journal_path, cursor, ledger, batches_root, config, tailer = self._setup_crash(
            tmp_path, rows)

        original_save = cursor.save

        def crash_on_save():
            raise OSError("simulated crash at f")

        with patch.object(cursor, "save", side_effect=crash_on_save):
            with pytest.raises(OSError, match="simulated crash at f"):
                cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        # Ledger is written, manifest is written, cursor is NOT saved.
        journal_path_resolved = (tmp_path / "journal" / "candidates.jsonl").resolve()
        cursor2 = CursorStore(tmp_path / "state" / "cursor.json")
        cursor2.load()
        lc, bo = cursor2.get(journal_path_resolved)
        # Cursor should be at 0 (not saved) OR behind the commit point.
        # Either way, recovery should advance it.
        ledger2 = Ledger.load(tmp_path / "ledger")

        def tailer_factory(path):
            return _make_tailer(path, cursor2)

        result = recover_pending_slice(batches_root, ledger2, cursor2, tailer_factory)
        assert result.get("recovered") == "batch10"

        # Cursor should now be correct.
        cursor3 = CursorStore(tmp_path / "state" / "cursor.json")
        cursor3.load()
        lc3, bo3 = cursor3.get(journal_path_resolved)
        assert lc3 == 5
        assert bo3 > 0

    def test_recovery_then_fresh_cut_slice_matches_no_crash_state(self, tmp_path):
        """After recovery, a fresh cut_slice produces identical final state as no-crash."""
        rows = _make_n_rows(7)

        # --- Crash scenario ---
        crash_dir = tmp_path / "crash"
        journal_path = crash_dir / "journal" / "candidates.jsonl"
        _write_journal(journal_path, rows)
        cursor = CursorStore(crash_dir / "state" / "cursor.json")
        cursor.load()
        ledger = Ledger.load(crash_dir / "ledger")
        batches_root = crash_dir / "batches"
        config = SliceConfig(campaign_source=SOURCE, slice_size=5)
        tailer = _make_tailer(journal_path, cursor)

        # Crash between c and d (after manifest written, before ledger).
        def crash_on_append(rows_arg):
            raise OSError("crash")

        with patch.object(ledger, "append_all", side_effect=crash_on_append):
            with pytest.raises(OSError):
                cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        # Recover.
        ledger_r = Ledger.load(crash_dir / "ledger")
        cursor_r = CursorStore(crash_dir / "state" / "cursor.json")
        cursor_r.load()

        def tailer_factory(path):
            return _make_tailer(path, cursor_r)

        recover_pending_slice(batches_root, ledger_r, cursor_r, tailer_factory)

        # --- Clean scenario ---
        clean_dir = tmp_path / "clean"
        journal_path_c = clean_dir / "journal" / "candidates.jsonl"
        _write_journal(journal_path_c, rows)
        cursor_c = CursorStore(clean_dir / "state" / "cursor.json")
        cursor_c.load()
        ledger_c = Ledger.load(clean_dir / "ledger")
        tailer_c = _make_tailer(journal_path_c, cursor_c)
        cut_slice(tailer_c, cursor_c, ledger_c, clean_dir / "batches", 10, config, NOW)

        # Compare uid sets — must be identical.
        crash_uids = set(ledger_r._by_uid.keys())
        clean_uids = set(ledger_c._by_uid.keys())
        assert crash_uids == clean_uids

        # Compare manifests (entries should be identical in uid/stmt_key/content_hash).
        crash_m = json.loads((batches_root / "batch10" / "slice_manifest.json").read_text())
        clean_m = json.loads((clean_dir / "batches" / "batch10" / "slice_manifest.json").read_text())

        def _entry_key(e):
            return (e["uid"], e["stmt_key"], e["content_hash"])

        assert sorted(_entry_key(e) for e in crash_m["entries"]) == \
               sorted(_entry_key(e) for e in clean_m["entries"])


# ---------------------------------------------------------------------------
# Determinism: two identical setups → byte-identical outputs
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_setup_byte_identical_records(self, tmp_path):
        rows = _make_n_rows(6)

        def run(sub: str):
            d = tmp_path / sub
            journal_path = d / "journal" / "candidates.jsonl"
            _write_journal(journal_path, rows)
            cursor = CursorStore(d / "state" / "cursor.json")
            cursor.load()
            ledger = Ledger.load(d / "ledger")
            batches_root = d / "batches"
            config = SliceConfig(campaign_source=SOURCE, slice_size=5)
            tailer = _make_tailer(journal_path, cursor)
            cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
            return (d / "batches" / "batch10" / "slice_records.jsonl").read_bytes()

        r1 = run("run1")
        r2 = run("run2")
        assert r1 == r2

    def test_identical_setup_byte_identical_manifest_entries(self, tmp_path):
        rows = _make_n_rows(6)

        def run(sub: str):
            d = tmp_path / sub
            journal_path = d / "journal" / "candidates.jsonl"
            _write_journal(journal_path, rows)
            cursor = CursorStore(d / "state" / "cursor.json")
            cursor.load()
            ledger = Ledger.load(d / "ledger")
            batches_root = d / "batches"
            config = SliceConfig(campaign_source=SOURCE, slice_size=5)
            tailer = _make_tailer(journal_path, cursor)
            cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
            m = json.loads((d / "batches" / "batch10" / "slice_manifest.json").read_text())
            return [(e["uid"], e["stmt_key"], e["content_hash"]) for e in m["entries"]]

        e1 = run("run1")
        e2 = run("run2")
        assert e1 == e2


# ---------------------------------------------------------------------------
# recover_pending_slice
# ---------------------------------------------------------------------------


class TestRecoverPendingSlice:
    def test_no_batches_returns_empty(self, tmp_path):
        batches_root = tmp_path / "batches"
        ledger = Ledger.load(tmp_path / "ledger")
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()

        def tailer_factory(path):
            return _make_tailer(path, cursor)

        result = recover_pending_slice(batches_root, ledger, cursor, tailer_factory)
        assert result == {}

    def test_empty_batches_dir_returns_empty(self, tmp_path):
        batches_root = tmp_path / "batches"
        batches_root.mkdir(parents=True)
        ledger = Ledger.load(tmp_path / "ledger")
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()

        def tailer_factory(path):
            return _make_tailer(path, cursor)

        result = recover_pending_slice(batches_root, ledger, cursor, tailer_factory)
        assert result == {}

    def test_complete_batch_no_recovery_needed(self, tmp_path):
        """A complete batch (manifest present, ledger populated, cursor advanced) → still recovered (idempotent)."""
        rows = _make_n_rows(6)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        # Re-load all state (fresh process).
        ledger2 = Ledger.load(tmp_path / "ledger")
        cursor2 = CursorStore(tmp_path / "state" / "cursor.json")
        cursor2.load()

        def tailer_factory(path):
            return _make_tailer(path, cursor2)

        result = recover_pending_slice(batches_root, ledger2, cursor2, tailer_factory)
        # Should return recovered (idempotent re-run of d-f).
        assert "recovered" in result
        # Ledger still has exactly 5 uids (idempotent — no duplicates added).
        assert len(ledger2._by_uid) == 5

    def test_pre_commit_crash_tmp_cleaned(self, tmp_path):
        """Pre-commit crash (no manifest): .tmp files removed, reports 'recomputable'."""
        batches_root = tmp_path / "batches"
        batch_dir = batches_root / "batch10"
        batch_dir.mkdir(parents=True)
        # Simulate a leftover tmp file.
        tmp_file = batch_dir / "slice_records.jsonl.tmp"
        tmp_file.write_text("partial data", encoding="utf-8")

        ledger = Ledger.load(tmp_path / "ledger")
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()

        def tailer_factory(path):
            return _make_tailer(path, cursor)

        result = recover_pending_slice(batches_root, ledger, cursor, tailer_factory)
        assert result.get("recomputable") == "batch10"
        assert not tmp_file.exists()

    def test_pre_commit_crash_leaves_records_jsonl(self, tmp_path):
        """Pre-commit crash: slice_records.jsonl (without .tmp) is left untouched."""
        batches_root = tmp_path / "batches"
        batch_dir = batches_root / "batch10"
        batch_dir.mkdir(parents=True)
        records_file = batch_dir / "slice_records.jsonl"
        records_file.write_text("some partial records", encoding="utf-8")

        ledger = Ledger.load(tmp_path / "ledger")
        cursor = CursorStore(tmp_path / "state" / "cursor.json")
        cursor.load()

        def tailer_factory(path):
            return _make_tailer(path, cursor)

        recover_pending_slice(batches_root, ledger, cursor, tailer_factory)
        # slice_records.jsonl must still exist (not removed).
        assert records_file.exists()
        assert records_file.read_text() == "some partial records"

    def test_manifest_recovery_appends_ledger(self, tmp_path):
        """Recovery from manifest (step d crash) re-populates ledger."""
        rows = _make_n_rows(7)

        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)

        # Simulate crash at step d: manifest written, ledger NOT written.
        def crash_append(rows_arg):
            raise OSError("crash at d")

        with patch.object(ledger, "append_all", side_effect=crash_append):
            with pytest.raises(OSError):
                cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        ledger2 = Ledger.load(tmp_path / "ledger")
        cursor2 = CursorStore(tmp_path / "state" / "cursor.json")
        cursor2.load()

        def tailer_factory(path):
            return _make_tailer(path, cursor2)

        recover_pending_slice(batches_root, ledger2, cursor2, tailer_factory)
        assert len(ledger2._by_uid) == 5


# ---------------------------------------------------------------------------
# Manifest: journal_span correctness
# ---------------------------------------------------------------------------


class TestManifestSpan:
    def test_journal_span_from_line(self, tmp_path):
        rows = _make_n_rows(6)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        m = json.loads((batches_root / "batch10" / "slice_manifest.json").read_text())
        assert m["journal_span"]["from_line"] == 1

    def test_journal_span_through_line_equals_slice_size(self, tmp_path):
        rows = _make_n_rows(7)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        m = json.loads((batches_root / "batch10" / "slice_manifest.json").read_text())
        # All 5 rows accepted, no skips → through_line = 5.
        assert m["journal_span"]["through_line"] == 5
        assert m["journal_span"]["through_byte"] > 0

    def test_journal_span_accounts_for_skips(self, tmp_path):
        """If rows 1-3 accepted, row 4 is replay, rows 5-7 accepted → through_line = 7."""
        base = _make_n_rows(9)
        # Insert duplicate of row[3] at position 3 (before row 3 by 0-index).
        rows = base[:3] + [base[3]] + base[3:]
        assert len(rows) == 10

        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(tmp_path, rows)
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)
        m = json.loads((batches_root / "batch10" / "slice_manifest.json").read_text())
        # With 1 skip at position 4 (line 4), the 5th accepted is at line 6.
        assert m["journal_span"]["through_line"] == 6


# ---------------------------------------------------------------------------
# uid pre-injection matches funnel's compute_uid
# ---------------------------------------------------------------------------


class TestUidPreInjection:
    def test_uid_matches_compute_uid(self, tmp_path):
        statement = "The Laplacian of a harmonic function is zero."
        row = _make_row_dict(statement=statement)
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(
            tmp_path, [row] + _make_n_rows(5))
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        records_path = batches_root / "batch10" / "slice_records.jsonl"
        first_record = json.loads(records_path.read_text().splitlines()[0])
        expected_uid = compute_uid(SOURCE, statement)
        assert first_record["uid"] == expected_uid

    def test_original_candidate_fields_preserved(self, tmp_path):
        """source and uid are added but all original fields are preserved."""
        row = _make_row_dict(statement="Theorem X holds.", answer="Yes",
                             metadata={"paper": "2401.00001"})
        journal_path, cursor, ledger, batches_root, config, tailer = _fresh(
            tmp_path, [row] + _make_n_rows(5))
        cut_slice(tailer, cursor, ledger, batches_root, 10, config, NOW)

        records_path = batches_root / "batch10" / "slice_records.jsonl"
        first_record = json.loads(records_path.read_text().splitlines()[0])
        assert first_record["statement"] == "Theorem X holds."
        assert first_record["answer"] == "Yes"
        assert first_record["metadata"] == {"paper": "2401.00001"}
        assert first_record["source"] == SOURCE
        assert "uid" in first_record

"""Tests for src/icepick/batcher/backfill.py.

All fixtures are synthetic (tmp_path only); no out/** files are read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick.batcher.backfill import backfill, load_sources
from icepick.batcher.identity import compute_uid, stmt_key as make_stmt_key
from icepick.batcher.ledger import Ledger


NOW = "2026-07-07T00:00:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _simple_source(label: str, rel_path: str, warn_only: bool = False) -> dict:
    return {
        "label": label,
        "path": rel_path,
        "warn_only": warn_only,
        "expect_has_uid_field": False,
    }


def _run_backfill(tmp_path, rows_by_label: dict[str, list[dict]], warn_only_labels: set = None):
    """Build a synthetic repo structure and run backfill; return (ledger, summary)."""
    warn_only_labels = warn_only_labels or set()
    repo_root = tmp_path / "repo"
    sources = []
    for label, rows in rows_by_label.items():
        rel = f"out/{label}/records.jsonl"
        _write_jsonl(repo_root / rel, rows)
        sources.append(_simple_source(label, rel, warn_only=label in warn_only_labels))

    ledger = Ledger.load(tmp_path / "ledger")
    summary = backfill(ledger, sources, repo_root, NOW)
    return ledger, summary


# ---------------------------------------------------------------------------
# load_sources
# ---------------------------------------------------------------------------


class TestLoadSources:
    def test_loads_default(self):
        sources = load_sources()
        labels = [s["label"] for s in sources]
        assert "batch0" in labels
        assert "stage1rescue" in labels
        assert "orphan_045533" in labels

    def test_loads_custom(self, tmp_path):
        custom = [
            {"label": "x", "path": "out/x/r.jsonl", "warn_only": False, "expect_has_uid_field": False}
        ]
        p = tmp_path / "sources.json"
        p.write_text(json.dumps(custom), encoding="utf-8")
        result = load_sources(p)
        assert result[0]["label"] == "x"

    def test_orphan_is_warn_only(self):
        sources = load_sources()
        orphan = next(s for s in sources if s["label"] == "orphan_045533")
        assert orphan["warn_only"] is True

    def test_stage1rescue_expects_uid(self):
        sources = load_sources()
        s1r = next(s for s in sources if s["label"] == "stage1rescue")
        assert s1r["expect_has_uid_field"] is True


# ---------------------------------------------------------------------------
# Basic backfill behaviour
# ---------------------------------------------------------------------------


class TestBackfillBasic:
    def test_rows_appended(self, tmp_path):
        rows = [{"source": "s", "statement": "stmt A", "answer": "1"}]
        ledger, summary = _run_backfill(tmp_path, {"lbl": rows})
        assert summary["lbl"]["rows"] == 1
        assert summary["lbl"]["appended"] == 1
        assert summary["lbl"]["distinct_uids"] == 1

    def test_uid_computed_from_source_and_statement(self, tmp_path):
        row = {"source": "src1", "statement": "Prove X."}
        ledger, summary = _run_backfill(tmp_path, {"lbl": [row]})
        expected_uid = compute_uid("src1", "Prove X.")
        assert expected_uid in ledger._by_uid

    def test_batch_tag_is_hist_label(self, tmp_path):
        row = {"source": "s", "statement": "s1"}
        ledger, summary = _run_backfill(tmp_path, {"mybatch": [row]})
        uid = compute_uid("s", "s1")
        assert ledger._by_uid[uid].batch == "hist:mybatch"

    def test_missing_statement_skipped(self, tmp_path):
        rows = [
            {"source": "s", "statement": "valid"},
            {"source": "s", "other_field": "no statement here"},
        ]
        ledger, summary = _run_backfill(tmp_path, {"lbl": rows})
        assert summary["lbl"]["rows"] == 2
        assert summary["lbl"]["appended"] == 1
        assert summary["lbl"]["missing_statement"] == 1

    def test_missing_file_marked(self, tmp_path):
        repo_root = tmp_path / "repo"
        sources = [_simple_source("missing_label", "out/nonexistent/records.jsonl")]
        ledger = Ledger.load(tmp_path / "ledger")
        summary = backfill(ledger, sources, repo_root, NOW)
        assert summary["missing_label"]["missing_file"] is True
        assert summary["missing_label"]["appended"] == 0


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


class TestBackfillIdempotence:
    def test_idempotent_re_run(self, tmp_path):
        rows = [{"source": "s", "statement": "once"}]
        ledger, summary1 = _run_backfill(tmp_path, {"lbl": rows})
        # Re-run with same data against the SAME ledger
        repo_root = tmp_path / "repo"
        sources = [_simple_source("lbl", "out/lbl/records.jsonl")]
        summary2 = backfill(ledger, sources, repo_root, NOW)
        assert summary2["lbl"]["appended"] == 0  # already in ledger


# ---------------------------------------------------------------------------
# uid mismatch counting
# ---------------------------------------------------------------------------


class TestUidMismatch:
    def test_uid_mismatch_counted_and_row_uid_used(self, tmp_path):
        """When a row has a uid that differs from computed, count it + use row's uid."""
        row = {"source": "s", "statement": "stmt", "uid": "OVERRIDE_UID"}
        ledger, summary = _run_backfill(tmp_path, {"lbl": [row]})
        assert summary["lbl"]["uid_mismatches"] == 1
        # The row's own uid must be in the ledger (not computed uid)
        assert "OVERRIDE_UID" in ledger._by_uid

    def test_uid_match_not_counted(self, tmp_path):
        """When row uid matches computed uid, no mismatch."""
        source = "s"
        statement = "stmt"
        computed = compute_uid(source, statement)
        row = {"source": source, "statement": statement, "uid": computed}
        ledger, summary = _run_backfill(tmp_path, {"lbl": [row]})
        assert summary["lbl"]["uid_mismatches"] == 0


# ---------------------------------------------------------------------------
# Cross-label duplicate uids (expected, no error)
# ---------------------------------------------------------------------------


class TestCrossLabelDups:
    def test_cross_label_dup_no_error(self, tmp_path):
        """Same uid in two labels writes one blocking row + one hist: row."""
        row = {"source": "s", "statement": "stmt_shared"}
        labels = {"label_a": [row], "label_b": [row]}
        ledger, summary = _run_backfill(tmp_path, labels)
        # Both labels should report appended (different (uid, batch) pairs)
        assert summary["label_a"]["appended"] == 1
        assert summary["label_b"]["appended"] == 1

    def test_cross_label_dup_second_run_idempotent(self, tmp_path):
        """After initial backfill, re-running either label is a no-op."""
        row = {"source": "s", "statement": "stmt_shared"}
        ledger, _ = _run_backfill(tmp_path, {"label_a": [row], "label_b": [row]})
        # Re-run
        repo_root = tmp_path / "repo"
        sources = [_simple_source("label_a", "out/label_a/records.jsonl")]
        summary2 = backfill(ledger, sources, repo_root, NOW)
        assert summary2["label_a"]["appended"] == 0


# ---------------------------------------------------------------------------
# warn_only rows
# ---------------------------------------------------------------------------


class TestWarnOnly:
    def test_warn_only_rows_dont_block(self, tmp_path):
        """A record in a warn_only source should produce a 'warn' verdict, not a block."""
        row = {"source": "s", "statement": "orphan_stmt"}
        ledger, _ = _run_backfill(tmp_path, {"orphan": [row]}, warn_only_labels={"orphan"})
        uid = compute_uid("s", "orphan_stmt")
        assert uid in ledger._warn_uids
        assert uid not in ledger._by_uid
        # Check verdict is warn, not replay/conflict
        sk = make_stmt_key("orphan_stmt")
        import hashlib, json
        ch = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        v = ledger.check(uid, sk, ch)
        assert v.kind == "warn"

    def test_warn_only_distinct_uids_counted(self, tmp_path):
        rows = [
            {"source": "s", "statement": "s1"},
            {"source": "s", "statement": "s2"},
        ]
        ledger, summary = _run_backfill(tmp_path, {"orphan": rows}, warn_only_labels={"orphan"})
        assert summary["orphan"]["distinct_uids"] == 2


# ---------------------------------------------------------------------------
# Torn tail in source file
# ---------------------------------------------------------------------------


class TestTornTailInSourceFile:
    def test_torn_tail_in_source_skipped(self, tmp_path):
        """A partial last line in a source file is tolerated; good lines are processed."""
        repo_root = tmp_path / "repo"
        path = repo_root / "out" / "lbl" / "records.jsonl"
        good_row = {"source": "s", "statement": "good"}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(good_row) + "\n")
            fh.write('{"source": "s", "statement": "partial')  # torn
        sources = [_simple_source("lbl", "out/lbl/records.jsonl")]
        ledger = Ledger.load(tmp_path / "ledger")
        summary = backfill(ledger, sources, repo_root, NOW)
        assert summary["lbl"]["appended"] == 1  # only the good row


# ---------------------------------------------------------------------------
# distinct_uids counting
# ---------------------------------------------------------------------------


class TestDistinctUids:
    def test_duplicate_uid_within_label_counted_once(self, tmp_path):
        """Two rows with the same source+statement produce one distinct_uid."""
        rows = [
            {"source": "s", "statement": "same"},
            {"source": "s", "statement": "same"},
        ]
        ledger, summary = _run_backfill(tmp_path, {"lbl": rows})
        assert summary["lbl"]["distinct_uids"] == 1
        # Both rows are appended (same uid, same batch → idempotence guard fires on 2nd)
        assert summary["lbl"]["appended"] == 1

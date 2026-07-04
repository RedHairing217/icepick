"""Manual mount adapter unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick.allocation.adapters import manual_mount


def _read_records(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --- scan ---------------------------------------------------------------------

def test_scan_single_jsonl_file(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"statement": "q"}) + "\n")
    result = manual_mount.scan(p)
    assert len(result) == 1
    assert result[0].file_type == "jsonl"
    assert result[0].path == p


def test_scan_directory_finds_supported_files_only(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"statement": "q1"}) + "\n")
    (tmp_path / "b.csv").write_text("statement\nq2\n")
    (tmp_path / "c.txt").write_text("not supported")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.jsonl").write_text(json.dumps({"statement": "deep"}) + "\n")
    result = manual_mount.scan(tmp_path)
    file_names = {r.path.name for r in result}
    assert file_names == {"a.jsonl", "b.csv"}  # nested ignored, .txt skipped


def test_scan_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        manual_mount.scan(tmp_path / "does_not_exist")


def test_scan_unsupported_single_file_returns_empty(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"binary")
    assert manual_mount.scan(p) == []


# --- mount per file type ------------------------------------------------------

def test_mount_jsonl_writes_canonical_handoff(tmp_path):
    src = tmp_path / "src.jsonl"
    rows = [
        {"statement": "what is 2+2", "answer": "4", "arxiv_id": "2403.11111"},
        {"statement": "what is 3*3", "answer": "9", "arxiv_id": "2403.22222"},
    ]
    with src.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    result = manual_mount.mount(
        path=src, source="batch_001", provenance="manual",
        truth_policy="unknown", output_dir=tmp_path / "out",
    )

    assert result.record_count == 2
    out = _read_records(result.records_path)
    assert all(r["source"] == "batch_001" for r in out)
    assert all(r["provenance"] == "manual" for r in out)
    assert all(r["truth_policy"] == "unknown" for r in out)
    assert {r["arxiv_id"] for r in out} == {"2403.11111", "2403.22222"}


def test_mount_json_array_handoff(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps([
        {"statement": "q1", "answer": "a1"},
        {"statement": "q2", "answer": "a2"},
    ]))
    result = manual_mount.mount(
        path=src, source="s", provenance="external",
        output_dir=tmp_path / "out",
    )
    assert result.record_count == 2


def test_mount_json_object_with_records_key(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"records": [{"statement": "q"}], "meta": "x"}))
    result = manual_mount.mount(
        path=src, source="s", provenance="external",
        output_dir=tmp_path / "out",
    )
    assert result.record_count == 1


def test_mount_csv_requires_column_map(tmp_path):
    src = tmp_path / "src.csv"
    src.write_text("question,gold\nq1,a1\n")
    result = manual_mount.mount(
        path=src, source="s", provenance="external",
        output_dir=tmp_path / "out",
    )
    # CSV without column_map → file skipped with reason
    assert result.record_count == 0
    assert len(result.files_skipped) == 1
    assert "column_map" in result.files_skipped[0][1].lower() or "column-map" in result.files_skipped[0][1].lower()


def test_mount_csv_with_column_map_projects_columns(tmp_path):
    src = tmp_path / "src.csv"
    src.write_text("question,gold,note\nq1,a1,n1\nq2,a2,n2\n")
    result = manual_mount.mount(
        path=src, source="customer", provenance="external",
        column_map={"statement": "question", "answer": "gold"},
        output_dir=tmp_path / "out",
    )
    out = _read_records(result.records_path)
    assert len(out) == 2
    assert out[0]["statement"] == "q1"
    assert out[0]["answer"] == "a1"
    # Unmapped column preserved under raw_columns
    assert out[0]["raw_columns"] == {"note": "n1"}


def test_mount_tsv_with_column_map(tmp_path):
    src = tmp_path / "src.tsv"
    src.write_text("question\tgold\nq1\ta1\n")
    result = manual_mount.mount(
        path=src, source="s", provenance="external",
        column_map={"statement": "question", "answer": "gold"},
        output_dir=tmp_path / "out",
    )
    out = _read_records(result.records_path)
    assert out[0]["statement"] == "q1"


# --- semantic guarantees ------------------------------------------------------

def test_mount_directory_combines_supported_files(tmp_path):
    src_dir = tmp_path / "drop"
    src_dir.mkdir()
    (src_dir / "a.jsonl").write_text(json.dumps({"statement": "a"}) + "\n")
    (src_dir / "b.jsonl").write_text(json.dumps({"statement": "b"}) + "\n")
    (src_dir / "c.txt").write_text("ignored")

    result = manual_mount.mount(
        path=src_dir, source="dir_drop", provenance="manual",
        output_dir=tmp_path / "out",
    )
    out = _read_records(result.records_path)
    statements = {r["statement"] for r in out}
    assert statements == {"a", "b"}


def test_mount_does_not_modify_source(tmp_path):
    """The mount path must remain byte-identical after mount() runs."""
    src = tmp_path / "src.jsonl"
    original = json.dumps({"statement": "q1"}) + "\n"
    src.write_text(original)
    src_mtime = src.stat().st_mtime
    src_bytes = src.read_bytes()

    manual_mount.mount(
        path=src, source="s", provenance="manual",
        output_dir=tmp_path / "out",
    )

    assert src.read_bytes() == src_bytes  # contents unchanged
    assert src.stat().st_mtime == src_mtime  # mtime unchanged


def test_mount_preserves_record_level_overrides(tmp_path):
    """If a record already declares provenance, it wins over the mount-level stamp."""
    src = tmp_path / "src.jsonl"
    src.write_text(json.dumps({
        "statement": "q",
        "provenance": "computed",  # explicit override
        "source": "upstream",
    }) + "\n")

    result = manual_mount.mount(
        path=src, source="batch", provenance="manual",
        output_dir=tmp_path / "out",
    )
    out = _read_records(result.records_path)
    assert out[0]["provenance"] == "computed"  # record-level wins
    assert out[0]["source"] == "upstream"


def test_mount_skips_records_without_statement(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"statement": "q1"}) + "\n"
        + json.dumps({"answer": "no statement here"}) + "\n"
    )
    result = manual_mount.mount(
        path=src, source="s", provenance="manual",
        output_dir=tmp_path / "out",
    )
    assert result.record_count == 1
    assert any("missing 'statement'" in reason for _, reason in result.files_skipped)


def test_mount_handles_malformed_jsonl_with_file_skip(tmp_path):
    """Bad lines abort that file but don't crash the whole mount."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"statement": "ok"}) + "\n" + "{not json\n")
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps({"statement": "fine"}) + "\n")
    src_dir = tmp_path / "drop"
    src_dir.mkdir()
    (src_dir / "bad.jsonl").write_text(bad.read_text())
    (src_dir / "good.jsonl").write_text(good.read_text())

    result = manual_mount.mount(
        path=src_dir, source="s", provenance="manual",
        output_dir=tmp_path / "out",
    )
    # bad.jsonl: first line wrote (ok), then the second line aborted file iteration.
    # The "ok" record from bad.jsonl gets written before the abort, plus the good.jsonl record.
    assert result.record_count == 2
    assert len(result.files_skipped) == 1
    assert "bad.jsonl" in str(result.files_skipped[0][0])

"""Bulk production runs pause and resume; they do not die or double-bill.

Three ways a run pauses — Ctrl-C, an exhausted call_budget, a chunk fetch
death — and each leaves enough on disk that re-running the same command
finishes without redoing paid work: checkpointed papers are not re-mined,
cached QA is not re-called, and journaled chunks are not re-downloaded. No
network — index/store are fakes, QA is a fixture generator.
"""

from __future__ import annotations

import gzip
import json
import socket
from datetime import datetime, timezone

import pytest

from icepick.allocation.adapters import arxiv_bulk
from icepick.contracts.manifests import ApprovedManifest, SOURCE_ARXIV_BULK

NOW = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

# OWN manifest: TWO 2025-01 chunks so chunk-level journaling is observable.
_MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<arXivSRC>
  <file>
    <filename>src/arXiv_src_2501_001.tar</filename>
    <yymm>2501</yymm>
    <seq_num>1</seq_num>
    <first_item>2501.00001</first_item>
    <last_item>2501.00150</last_item>
    <num_items>150</num_items>
    <size>1000000000</size>
    <md5sum>aaaa1111bbbb2222cccc3333dddd4444</md5sum>
    <content_md5sum>1111aaaa2222bbbb3333cccc4444dddd</content_md5sum>
    <timestamp>2025-02-04 09:22:11</timestamp>
  </file>
  <file>
    <filename>src/arXiv_src_2501_002.tar</filename>
    <yymm>2501</yymm>
    <seq_num>2</seq_num>
    <first_item>2501.00200</first_item>
    <last_item>2501.00400</last_item>
    <num_items>200</num_items>
    <size>2000000000</size>
    <md5sum>bbbb2222cccc3333dddd4444eeee5555</md5sum>
    <content_md5sum>2222bbbb3333cccc4444dddd5555eeee</content_md5sum>
    <timestamp>2025-02-04 09:45:00</timestamp>
  </file>
</arXivSRC>
"""

# One paper per chunk (ids fall in each chunk's [first,last] range).
_PAPERS = {
    "2501.00101": {  # chunk 001
        "title": "Paper One", "primary_category": "math.AP", "categories": ("math.AP",),
        "tex": r"\begin{theorem}Solution count for one is one.\end{theorem}",
    },
    "2501.00305": {  # chunk 002
        "title": "Paper Two", "primary_category": "math.AP", "categories": ("math.AP",),
        "tex": r"\begin{theorem}Solution count for two is two.\end{theorem}",
    },
}


class _FakeMeta:
    def __init__(self, arxiv_id, info):
        self.arxiv_id = arxiv_id
        self.primary_category = info["primary_category"]
        self.categories = info["categories"]
        self.title = info["title"]


class _FakeIndex:
    def __init__(self, papers):
        self._papers = papers
        self.oai_requests = 1

    def lookup(self, arxiv_id):
        info = self._papers.get(arxiv_id)
        return _FakeMeta(arxiv_id, info) if info else None

    def ids_for(self, *, category, yymm, primary_only):
        return [pid for pid in self._papers if pid.startswith(yymm + ".")]


class _FakeStore:
    def __init__(self, papers, work_dir, *, downloads_log, fail_on=None):
        self._papers = papers
        self._work_dir = work_dir
        self._downloads_log = downloads_log
        self._fail_on = fail_on
        self.chunk_downloads = 0
        self.chunk_bytes = 0
        self.corrupt_downloads = 0
        self.corrupt_bytes = 0
        self.released = []

    def fetch(self, entry):
        if self._fail_on is not None and entry.filename == self._fail_on:
            raise KeyboardInterrupt  # Ctrl-C while fetching the second chunk
        self._downloads_log.append(entry.filename)
        path = self._work_dir / entry.filename.replace("/", "_")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"chunk")
        # Mirror the real store: a completed download bumps the lifetime
        # counters (the adapter counts by DELTA per W3 M3, so a fake that
        # never increments would read as all-adoptions and never bill).
        self.chunk_downloads += 1
        self.chunk_bytes += int(getattr(entry, "size_bytes", 0) or 0)
        return path

    def extract_matching(self, chunk_path, wanted_ids):
        for arxiv_id in self._papers:
            if arxiv_id in wanted_ids:
                yield arxiv_id, gzip.compress(self._papers[arxiv_id]["tex"].encode())

    def release(self, entry_or_path):
        self.released.append(getattr(entry_or_path, "filename", entry_or_path))


class _AdoptingStore(_FakeStore):
    """Every fetch adopts a resident verified file: no download happened,
    so no counter moves and nothing is logged (the real store's resume
    adoption path, W3 M3)."""

    def fetch(self, entry):
        path = self._work_dir / entry.filename.replace("/", "_")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"chunk")
        return path


def _make_qa_generator(calls_log):
    def generate(statement, **kwargs):
        calls_log.append(statement)
        if kwargs.get("model_callback"):
            kwargs["model_callback"]("fake-sonnet-test")
        return {"question": f"Q: {statement}", "answer": "42"}

    return generate


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during a bulk resume test")

    monkeypatch.setattr(socket, "socket", _blocked)


def _manifest(tmp_path, **overrides):
    manifest_path = tmp_path / "src_manifest.xml"
    if not manifest_path.exists():
        manifest_path.write_text(_MANIFEST_XML, encoding="utf-8")
    window = {
        "year": 2025, "month": 1, "category": "math.AP",
        "extraction": "qa", "manifest_path": str(manifest_path),
    }
    window.update(overrides.pop("scrape_window", {}))
    base = dict(
        run_id="20260706T120000Z",
        source_type=SOURCE_ARXIV_BULK,
        processor_mode="production",
        requested_by="alice",
        requested_at="2026-07-06T00:00:00Z",
        approved_by="bob",
        approved_at="2026-07-06T01:00:00Z",
        source_name="arxiv_bulk_2025Q1",
        target_count=5,
        call_budget=1000,
        judge_enabled=False,
        confirmation_enabled=False,
        enable_leakage=False,
        enable_duplication=True,
        enable_robustness=False,
        families=["pde"],
        scrape_window=window,
        truth_policy="extracted",
        output_dir=str(tmp_path / "intake"),
    )
    base.update(overrides)
    return ApprovedManifest(**base)


def _run_dir(tmp_path):
    return tmp_path / "intake" / "runs" / "20260706T120000Z"


def _wire(monkeypatch, downloads_log, qa_calls_log, *, fail_on=None):
    def fake_build_index(manifest, run_dir, window, counts, charge):
        idx = _FakeIndex(_PAPERS)
        charge("oai_requests")
        counts["oai_requests"] = idx.oai_requests
        return idx

    def fake_open_store(manifest, run_dir, window):
        (run_dir / "_chunks").mkdir(parents=True, exist_ok=True)
        return _FakeStore(
            _PAPERS, run_dir / "_chunks",
            downloads_log=downloads_log, fail_on=fail_on,
        )

    monkeypatch.setattr(arxiv_bulk, "_build_category_index", fake_build_index)
    monkeypatch.setattr(arxiv_bulk, "_open_chunk_store", fake_open_store)
    monkeypatch.setattr(
        arxiv_bulk, "_default_qa_generator",
        lambda s: _make_qa_generator(qa_calls_log),
    )


def test_ctrl_c_mid_run_is_resumable_without_redoing_work(tmp_path, monkeypatch):
    downloads: list = []
    qa_calls: list = []

    # First invocation: chunk 001 completes and commits paper one; fetching
    # chunk 002 raises Ctrl-C, so the run pauses with paper one checkpointed.
    _wire(monkeypatch, downloads, qa_calls, fail_on="src/arXiv_src_2501_002.tar")
    first = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    assert first.interrupted is True
    assert first.record_count == 1
    assert (_run_dir(tmp_path) / "_progress" / "INCOMPLETE").exists()
    # Journal sits DIRECTLY in the checkpoint's progress dir (§4, amended).
    assert (_run_dir(tmp_path) / "_progress" / "chunks_done.jsonl").exists()
    report = first.report_path.read_text()
    assert "INTERRUPTED" in report and "resumable" in report

    # chunk 001 journaled after committing its paper; chunk 002 never fetched.
    assert downloads == ["src/arXiv_src_2501_001.tar"]
    assert qa_calls == ["Solution count for one is one."]

    # Second invocation of the SAME manifest: completes without re-downloading
    # chunk 001, re-mining paper one, or re-calling its cached QA.
    downloads.clear()
    qa_calls.clear()
    _wire(monkeypatch, downloads, qa_calls)  # no failure this time
    second = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    assert second.interrupted is False
    assert second.record_count == 2
    assert second.acquisition["resumed_papers"] == 1
    assert not (_run_dir(tmp_path) / "_progress" / "INCOMPLETE").exists()

    # Chunk 001 NOT re-downloaded (journaled); only chunk 002 fetched now.
    assert downloads == ["src/arXiv_src_2501_002.tar"]
    # Paper one's QA served from the cache — only paper two calls the generator.
    assert qa_calls == ["Solution count for two is two."]
    assert "Resumed: 1 papers served from the checkpoint" in second.report_path.read_text()


def test_exhausted_budget_pauses_and_resumes(tmp_path, monkeypatch):
    downloads: list = []
    qa_calls: list = []
    _wire(monkeypatch, downloads, qa_calls)

    # Budget of 3 = 1 oai + 1 chunk_download + 1 qa call: exactly enough for
    # the first chunk's one paper, then exhausted before the second chunk's
    # download. Bypass the pre-run estimate gate (that guards the operator;
    # here we test mid-run exhaustion) by patching the estimate for this run.
    monkeypatch.setattr(arxiv_bulk, "_estimated_calls", lambda tc, extraction="latex": 3)
    first = arxiv_bulk.run(_manifest(tmp_path, call_budget=3), now=NOW)
    assert first.interrupted is True
    assert first.record_count == 1
    assert any("call budget 3 exhausted" in w for w in first.warnings)
    assert first.acquisition["total_calls"] == 3
    assert downloads == ["src/arXiv_src_2501_001.tar"]

    # Resume with a fresh (ample) budget: cached work costs nothing, run finishes.
    downloads.clear()
    qa_calls.clear()
    monkeypatch.setattr(arxiv_bulk, "_estimated_calls", lambda tc, extraction="latex": 10)
    second = arxiv_bulk.run(_manifest(tmp_path, call_budget=1000), now=NOW)
    assert second.interrupted is False
    assert second.record_count == 2
    assert second.acquisition["resumed_papers"] == 1
    assert downloads == ["src/arXiv_src_2501_002.tar"]  # chunk one not re-downloaded
    assert qa_calls == ["Solution count for two is two."]  # paper one's QA cached


def test_budget_exhausted_during_index_build_pauses_cleanly(tmp_path, monkeypatch):
    # W3 H1: exhaustion while the OAI index is still building must be the same
    # checkpointed pause as mid-loop exhaustion — never an escaping exception,
    # and the INCOMPLETE marker must exist (begin() precedes the index build).
    downloads: list = []
    qa_calls: list = []
    _wire(monkeypatch, downloads, qa_calls)

    def exhausting_index(manifest, run_dir, window, counts, charge):
        while True:  # pages "forever": the budget is what stops it
            charge("oai_requests")

    monkeypatch.setattr(arxiv_bulk, "_build_category_index", exhausting_index)
    monkeypatch.setattr(arxiv_bulk, "_estimated_calls", lambda *a, **k: 2)
    result = arxiv_bulk.run(_manifest(tmp_path, call_budget=2), now=NOW)

    assert result.interrupted is True  # pause, not crash
    assert (_run_dir(tmp_path) / "_progress" / "INCOMPLETE").exists()
    acq = result.acquisition
    assert acq["oai_requests"] == 2 and acq["chunk_downloads"] == 0
    assert downloads == []
    assert any("call budget 2 exhausted" in w for w in result.warnings)


def test_adopted_resident_chunk_is_not_rebilled(tmp_path, monkeypatch):
    # W3 M3: a fetch the store satisfies by adopting a resident verified file
    # is free — no chunk_downloads/chunk_bytes count, no budget consumption.
    downloads: list = []
    qa_calls: list = []
    _wire(monkeypatch, downloads, qa_calls)

    def adopting_open_store(manifest, run_dir, window):
        (run_dir / "_chunks").mkdir(parents=True, exist_ok=True)
        return _AdoptingStore(_PAPERS, run_dir / "_chunks", downloads_log=downloads)

    monkeypatch.setattr(arxiv_bulk, "_open_chunk_store", adopting_open_store)
    result = arxiv_bulk.run(_manifest(tmp_path), now=NOW)

    assert result.interrupted is False
    assert result.record_count == 2
    acq = result.acquisition
    assert acq["chunk_downloads"] == 0 and acq["chunk_bytes"] == 0
    assert acq["total_calls"] == acq["oai_requests"] + acq["qa_calls"]
    assert downloads == []


def test_pause_mid_chunk_warns_about_retained_file(tmp_path, monkeypatch):
    # W3 M2: a pause between a chunk's fetch and its release deliberately
    # retains the file (resume adopts it free) and says so in the warnings.
    # Budget 4 = oai(1) + chunk1(1) + qa1(1) + chunk2(1); paper two's QA
    # charge then exhausts it with chunk 002 fetched but not yet released.
    downloads: list = []
    qa_calls: list = []
    _wire(monkeypatch, downloads, qa_calls)
    monkeypatch.setattr(arxiv_bulk, "_estimated_calls", lambda *a, **k: 4)
    result = arxiv_bulk.run(_manifest(tmp_path, call_budget=4), now=NOW)

    assert result.interrupted is True
    retained = [w for w in result.warnings if "retained for resume" in w]
    assert retained and "arXiv_src_2501_002" in retained[0]


def test_index_build_is_bounded_by_window_from_date(tmp_path, monkeypatch):
    # W3 H2: the real index build must pass the window-derived from_date
    # superset bound to CategoryIndex.build (initial request only; §2).
    captured = {}

    def capture_build(self, *, oai_set, fetcher, from_date=None, **kwargs):
        captured["oai_set"] = oai_set
        captured["from_date"] = from_date

    monkeypatch.setattr(
        "icepick.allocation.bulk.category_index.CategoryIndex.build", capture_build
    )
    counts = {"oai_requests": 0, "chunk_downloads": 0, "qa_calls": 0, "chunk_bytes": 0}

    arxiv_bulk._build_category_index(
        None, tmp_path, {"year": 2025, "month": 3, "category": "math.AP"},
        counts, lambda kind: None,
    )
    assert captured["from_date"] == "2025-03-01"
    assert captured["oai_set"] == "math"

    arxiv_bulk._build_category_index(
        None, tmp_path, {"category": "math.AP"}, counts, lambda kind: None,
    )
    assert captured["from_date"] is None  # no year -> unbounded walk


def test_completed_rerun_is_idempotent_and_free(tmp_path, monkeypatch):
    downloads: list = []
    qa_calls: list = []
    _wire(monkeypatch, downloads, qa_calls)

    first = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    handoff = first.handoff_path.read_bytes()
    assert sorted(downloads) == ["src/arXiv_src_2501_001.tar", "src/arXiv_src_2501_002.tar"]

    downloads.clear()
    qa_calls.clear()
    second = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    assert second.handoff_path.read_bytes() == handoff
    assert second.acquisition["resumed_papers"] == 2
    assert downloads == []  # both chunks journaled, nothing re-downloaded
    assert qa_calls == []   # both papers' QA cached


def test_chunk_journal_lands_at_the_contract_path(tmp_path, monkeypatch):
    downloads: list = []
    qa_calls: list = []
    _wire(monkeypatch, downloads, qa_calls)
    arxiv_bulk.run(_manifest(tmp_path), now=NOW)

    # DIRECTLY in the checkpoint's progress dir, alongside its ledger (§4, amended).
    journal = _run_dir(tmp_path) / "_progress" / "chunks_done.jsonl"
    assert journal.exists()
    rows = [json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
    filenames = {r["filename"] for r in rows}
    assert filenames == {"src/arXiv_src_2501_001.tar", "src/arXiv_src_2501_002.tar"}
    for row in rows:
        assert "papers" in row  # {"filename": ..., "papers": n}

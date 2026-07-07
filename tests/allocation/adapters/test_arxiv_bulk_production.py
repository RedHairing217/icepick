"""Production run: arxiv_bulk.run drives the S3 bulk pipeline in-house.

Everything paid is injected — a fake CategoryIndex-shaped object (no OAI), a
fake ChunkStore-shaped object (no S3), an OWN src manifest on tmp_path, and a
fixture QA generator (no Anthropic). The tests exercise THIS adapter's
orchestration — chunk selection, id scoping, per-chunk extract+mine, budget
accounting, run layout, spend_rows — not the sibling modules' internals. No
network anywhere.
"""

from __future__ import annotations

import gzip
import io
import json
import socket
import tarfile
from datetime import datetime, timezone

import pytest

from icepick.allocation.adapters import arxiv_bulk
from icepick.allocation.scrape import realmath as source
from icepick.contracts.manifests import ApprovedManifest, SOURCE_ARXIV_BULK

NOW = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

# OWN src manifest: one 2025-01 chunk covering 2501.00001..2501.00500. Never
# the sibling-owned fixture. size chosen so chunk_gb rounds to a clean value.
_MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<arXivSRC>
  <file>
    <filename>src/arXiv_src_2501_001.tar</filename>
    <yymm>2501</yymm>
    <seq_num>1</seq_num>
    <first_item>2501.00001</first_item>
    <last_item>2501.00500</last_item>
    <num_items>500</num_items>
    <size>1500000000</size>
    <md5sum>aaaa1111bbbb2222cccc3333dddd4444</md5sum>
    <content_md5sum>1111aaaa2222bbbb3333cccc4444dddd</content_md5sum>
    <timestamp>2025-02-04 09:22:11</timestamp>
  </file>
</arXivSRC>
"""

# Two papers in the chunk, each with a distinct single-theorem LaTeX source.
_PAPERS = {
    "2501.00101": {
        "title": "On a nonlinear PDE",
        "primary_category": "math.AP",
        "categories": ("math.AP", "math.FA"),
        "tex": r"\begin{theorem}The unique solution count for problem A is three.\end{theorem}",
    },
    "2501.00202": {
        "title": "Banach space methods",
        "primary_category": "math.AP",
        "categories": ("math.AP",),
        "tex": r"\begin{theorem}The unique solution count for problem B is seven.\end{theorem}",
    },
}


# --- fakes mirroring the frozen §2/§3 shapes ---------------------------------


class _FakeMeta:
    def __init__(self, arxiv_id, info):
        self.arxiv_id = arxiv_id
        self.primary_category = info["primary_category"]
        self.categories = info["categories"]
        self.title = info["title"]


class _FakeIndex:
    """CategoryIndex-shaped: lookup / ids_for + realmath-shaped telemetry attrs.

    Mirrors the amended §2 surface: oai_requests plus lifetime
    rate_limit_events / rate_limit_backoff_seconds / rate_limit_statuses.
    """

    def __init__(self, papers, *, primary_only_drops=(), throttle=None):
        self._papers = papers
        self._primary_only_drops = set(primary_only_drops)
        self.oai_requests = 2  # pretend two OAI pages were paged
        # OAI throttle telemetry (str(status) keys), lifetime per instance.
        self.rate_limit_events = 0
        self.rate_limit_backoff_seconds = 0.0
        self.rate_limit_statuses = {}
        if throttle:
            status, backoff = throttle
            self.rate_limit_events = 1
            self.rate_limit_backoff_seconds = float(backoff)
            self.rate_limit_statuses = {str(status): 1}

    def lookup(self, arxiv_id):
        info = self._papers.get(arxiv_id)
        return _FakeMeta(arxiv_id, info) if info else None

    def ids_for(self, *, category, yymm, primary_only):
        ids = [pid for pid in self._papers if pid.startswith(yymm + ".")]
        if primary_only:
            ids = [pid for pid in ids if pid not in self._primary_only_drops]
        return ids


class _FakeChecksumError(RuntimeError):
    """Stand-in for chunk_store.ChecksumError (a RuntimeError subclass, §3)."""


class _FakeStore:
    """ChunkStore-shaped (amended §3): fetch / extract_matching / release,
    with chunk_downloads / chunk_bytes and corrupt_downloads / corrupt_bytes
    counters. ``release`` is purge-only; residency overflow RAISES RuntimeError.
    """

    def __init__(self, papers, work_dir, *, missing=(), corrupt=()):
        self._papers = papers
        self._work_dir = work_dir
        self._missing = set(missing)
        self._corrupt = set(corrupt)  # filenames whose transfer checksum-fails
        self.chunk_downloads = 0
        self.chunk_bytes = 0
        self.corrupt_downloads = 0
        self.corrupt_bytes = 0
        self.fetched = []
        self.released = []
        self.resident = set()

    def fetch(self, entry):
        if entry.filename in self._corrupt:
            # Egress billed even though the checksum failed and we kept nothing.
            # ChecksumError is a RuntimeError subclass (§3); a local stand-in
            # keeps this fake independent of the sibling chunk_store landing.
            self.corrupt_downloads += 1
            self.corrupt_bytes += entry.size_bytes
            raise _FakeChecksumError(f"md5 mismatch for {entry.filename}")
        self.chunk_downloads += 1
        self.chunk_bytes += entry.size_bytes
        self.fetched.append(entry.filename)
        self.resident.add(entry.filename)
        if len(self.resident) > 2:
            raise RuntimeError("more than 2 chunks resident at once")
        path = self._work_dir / entry.filename.replace("/", "_")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"chunk")
        return path

    def extract_matching(self, chunk_path, wanted_ids):
        for arxiv_id in self._papers:
            if arxiv_id in wanted_ids and arxiv_id not in self._missing:
                yield arxiv_id, gzip.compress(self._papers[arxiv_id]["tex"].encode())

    def release(self, entry_or_path):
        # purge-only (no extraction — extraction is the adapter's job)
        name = getattr(entry_or_path, "filename", entry_or_path)
        self.released.append(name)
        self.resident.discard(name)


def _fixture_qa_generator(statement, **kwargs):
    """Deterministic offline QA generator; never touches Anthropic.

    Turns any theorem into a Q+A with a numeric answer so classify_answer
    keeps it. Reports usage + model through the same callbacks the real one
    uses, so token_usage + qa_model render in the report.
    """
    if kwargs.get("model_callback"):
        kwargs["model_callback"]("fake-sonnet-test")
    if kwargs.get("usage_callback"):
        kwargs["usage_callback"]({"input_tokens": 10, "output_tokens": 4})
    return {"question": f"Q: {statement}", "answer": "42"}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during production bulk test")

    monkeypatch.setattr(socket, "socket", _blocked)


@pytest.fixture(autouse=True)
def _inject_seams(monkeypatch):
    """Wire the fake index, fake store, and fixture QA generator."""
    index = _FakeIndex(_PAPERS)
    store_holder = {}

    def fake_build_index(manifest, run_dir, window, counts, charge):
        # honour the budget/counter contract for OAI just like the real path
        for _ in range(index.oai_requests):
            charge("oai_requests")
        counts["oai_requests"] = max(counts["oai_requests"], index.oai_requests)
        return index

    def fake_open_store(manifest, run_dir, window):
        store = _FakeStore(_PAPERS, run_dir / "_chunks")
        (run_dir / "_chunks").mkdir(parents=True, exist_ok=True)
        store_holder["store"] = store
        return store

    monkeypatch.setattr(arxiv_bulk, "_build_category_index", fake_build_index)
    monkeypatch.setattr(arxiv_bulk, "_open_chunk_store", fake_open_store)
    monkeypatch.setattr(arxiv_bulk, "_default_qa_generator", lambda s: _fixture_qa_generator)
    return {"index": index, "store_holder": store_holder}


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


def test_production_run_extracts_mines_and_writes_the_handoff(tmp_path):
    outcome = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    assert outcome.calibration_replay is False
    assert outcome.processor_mode == "production"
    assert outcome.paper_count == 2
    assert outcome.record_count == 2
    assert outcome.handoff_path.exists()

    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert {r["arxiv_id"] for r in records} == {"2501.00101", "2501.00202"}
    for record in records:
        assert record["source"] == "arxiv_bulk_2025Q1"
        assert record["provenance"] == "extracted"
        assert record["family"] == "pde"
        assert record["answer"] == "42"
        assert "calibration_replay" not in record.get("metadata", {})


def test_production_run_reports_bulk_spend_rows(tmp_path):
    outcome = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    acq = outcome.acquisition
    assert acq is not None
    assert acq["oai_requests"] == 2
    assert acq["chunk_downloads"] == 1
    assert acq["chunk_bytes"] == 1_500_000_000
    assert acq["qa_calls"] == 2
    assert acq["qa_model"] == "fake-sonnet-test"
    # total_calls sums the three call kinds; chunk_bytes is NOT a call.
    assert acq["total_calls"] == 2 + 1 + 2
    assert acq["call_budget"] == 1000
    assert acq["spend_rows"] == [
        ["oai_request", 2],
        ["chunk_download", 1],
        ["chunk_gb", 1.5],
        ["qa_generation (fake-sonnet-test)", 2],
    ]


def test_production_report_renders_the_bulk_spend_table(tmp_path):
    outcome = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    report = outcome.report_path.read_text()
    assert report.startswith("# arXiv bulk source report")  # honest, source-stamped title
    assert "## Spend (acquisition calls)" in report
    assert "| oai_request | 2 |" in report
    assert "| chunk_download | 1 |" in report
    assert "| chunk_gb | 1.5 |" in report
    assert "| qa_generation (fake-sonnet-test) | 2 |" in report
    assert "| total | 5 / 1000 budgeted |" in report
    assert "calibration_replay: false" in report
    # token usage from the fixture generator surfaces too.
    assert "## LLM token usage" in report


def test_production_surfaces_oai_throttle_telemetry_from_the_index(tmp_path, monkeypatch):
    """§2 amendment: the CategoryIndex's own lifetime 429/503 telemetry (from
    build()) must reach the acquisition dict and the report, since the
    checkpoint never sees OAI throttling in the bulk path."""
    def fake_build_index(manifest, run_dir, window, counts, charge):
        idx = _FakeIndex(_PAPERS, throttle=(503, 6.0))
        for _ in range(idx.oai_requests):
            charge("oai_requests")
        counts["oai_requests"] = idx.oai_requests
        return idx

    monkeypatch.setattr(arxiv_bulk, "_build_category_index", fake_build_index)
    outcome = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    acq = outcome.acquisition
    assert acq["rate_limit_events"] == 1
    assert acq["rate_limit_backoff_seconds"] == pytest.approx(6.0)
    assert acq["rate_limit_statuses"] == {"503": 1}
    report = outcome.report_path.read_text()
    assert "## arXiv throttle telemetry" in report
    assert "| 429/503 encounters | 1 |" in report
    assert "| total backoff seconds | 6.0 |" in report


def test_production_surfaces_corrupt_download_egress(tmp_path, monkeypatch):
    """§3 amendment: a checksum-failed transfer billed egress but yielded no
    chunk. Its papers are skipped (run continues), and the paid unit is visible
    in acquisition + a ``chunk_download_corrupt`` spend_row (invariant 2)."""
    two_chunk_xml = """<?xml version="1.0" encoding="UTF-8"?>
<arXivSRC>
  <file>
    <filename>src/arXiv_src_2501_001.tar</filename>
    <yymm>2501</yymm><seq_num>1</seq_num>
    <first_item>2501.00100</first_item><last_item>2501.00150</last_item>
    <num_items>50</num_items><size>1000000000</size>
    <md5sum>aaaa1111bbbb2222cccc3333dddd4444</md5sum>
    <content_md5sum>1111aaaa2222bbbb3333cccc4444dddd</content_md5sum>
    <timestamp>2025-02-04 09:22:11</timestamp>
  </file>
  <file>
    <filename>src/arXiv_src_2501_002.tar</filename>
    <yymm>2501</yymm><seq_num>2</seq_num>
    <first_item>2501.00200</first_item><last_item>2501.00250</last_item>
    <num_items>50</num_items><size>2000000000</size>
    <md5sum>bbbb2222cccc3333dddd4444eeee5555</md5sum>
    <content_md5sum>2222bbbb3333cccc4444dddd5555eeee</content_md5sum>
    <timestamp>2025-02-04 09:45:00</timestamp>
  </file>
</arXivSRC>
"""
    papers = {
        "2501.00105": {  # chunk 001 (will be corrupt)
            "title": "Corrupt chunk paper", "primary_category": "math.AP",
            "categories": ("math.AP",),
            "tex": r"\begin{theorem}Solution count for one is one.\end{theorem}",
        },
        "2501.00205": {  # chunk 002 (clean)
            "title": "Clean chunk paper", "primary_category": "math.AP",
            "categories": ("math.AP",),
            "tex": r"\begin{theorem}Solution count for two is two.\end{theorem}",
        },
    }
    manifest_path = tmp_path / "two_chunk_manifest.xml"
    manifest_path.write_text(two_chunk_xml, encoding="utf-8")

    def fake_build_index(manifest, run_dir, window, counts, charge):
        idx = _FakeIndex(papers)
        counts["oai_requests"] = idx.oai_requests
        return idx

    def fake_open_store(manifest, run_dir, window):
        (run_dir / "_chunks").mkdir(parents=True, exist_ok=True)
        return _FakeStore(
            papers, run_dir / "_chunks",
            corrupt=("src/arXiv_src_2501_001.tar",),
        )

    monkeypatch.setattr(arxiv_bulk, "_build_category_index", fake_build_index)
    monkeypatch.setattr(arxiv_bulk, "_open_chunk_store", fake_open_store)

    manifest = _manifest(
        tmp_path,
        scrape_window={"extraction": "latex", "manifest_path": str(manifest_path)},
    )
    outcome = arxiv_bulk.run(manifest, now=NOW)

    acq = outcome.acquisition
    assert acq["corrupt_downloads"] == 1
    assert acq["corrupt_bytes"] == 1_000_000_000
    assert acq["chunk_downloads"] == 1  # only the clean chunk counts as a good download
    assert ["chunk_download_corrupt", 1] in acq["spend_rows"]
    # Corrupt chunk's paper skipped; only the clean chunk's paper mined.
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert [r["arxiv_id"] for r in records] == ["2501.00205"]
    assert any("failed checksum" in w for w in outcome.warnings)
    report = outcome.report_path.read_text()
    assert "| chunk_download_corrupt | 1 |" in report


def test_production_clean_run_omits_the_corrupt_spend_row(tmp_path):
    """A clean run keeps the report clean: no chunk_download_corrupt row, but
    the acquisition keys are still present and zero."""
    outcome = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    acq = outcome.acquisition
    assert acq["corrupt_downloads"] == 0
    assert acq["corrupt_bytes"] == 0
    assert all(row[0] != "chunk_download_corrupt" for row in acq["spend_rows"])
    assert "chunk_download_corrupt" not in outcome.report_path.read_text()


def test_production_latex_mode_makes_no_qa_calls(tmp_path):
    outcome = arxiv_bulk.run(
        _manifest(tmp_path, scrape_window={"extraction": "latex"}), now=NOW
    )
    assert outcome.acquisition["qa_calls"] == 0
    assert outcome.record_count == 2
    # latex-mode records carry the raw theorem statement, no LLM answer field.
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert all("solution count" in r["statement"] for r in records)


def test_production_run_downloads_each_chunk_once_and_releases_it(tmp_path, _inject_seams):
    arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    store = _inject_seams["store_holder"]["store"]
    assert store.fetched == ["src/arXiv_src_2501_001.tar"]
    assert store.released == ["src/arXiv_src_2501_001.tar"]
    assert store.resident == set()  # freed after extraction, disk discipline


def test_production_missing_id_in_chunk_is_skipped_with_a_warning(tmp_path, monkeypatch):
    def fake_open_store(manifest, run_dir, window):
        (run_dir / "_chunks").mkdir(parents=True, exist_ok=True)
        return _FakeStore(_PAPERS, run_dir / "_chunks", missing=("2501.00202",))

    monkeypatch.setattr(arxiv_bulk, "_open_chunk_store", fake_open_store)
    outcome = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    # Only the present paper is mined; the missing one is skipped, not fetched.
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert [r["arxiv_id"] for r in records] == ["2501.00101"]
    assert any("2501.00202" in w and "not found in chunk" in w for w in outcome.warnings)


def test_production_local_fetcher_never_hits_the_network(tmp_path, monkeypatch):
    """The e-print network fetcher must never be called in the bulk path."""
    def boom(*args, **kwargs):
        raise AssertionError("bulk run reached the network e-print fetcher")

    monkeypatch.setattr(source, "default_latex_source_fetcher", boom)
    outcome = arxiv_bulk.run(_manifest(tmp_path), now=NOW)
    assert outcome.record_count == 2


def test_production_max_papers_caps_the_id_pool(tmp_path):
    outcome = arxiv_bulk.run(
        _manifest(tmp_path, scrape_window={"max_papers": 1}), now=NOW
    )
    assert outcome.paper_count == 1
    assert outcome.record_count == 1


def test_production_primary_only_drops_cross_listed(tmp_path, monkeypatch):
    def fake_build_index(manifest, run_dir, window, counts, charge):
        idx = _FakeIndex(_PAPERS, primary_only_drops=("2501.00202",))
        for _ in range(idx.oai_requests):
            charge("oai_requests")
        counts["oai_requests"] = idx.oai_requests
        return idx

    monkeypatch.setattr(arxiv_bulk, "_build_category_index", fake_build_index)
    outcome = arxiv_bulk.run(
        _manifest(tmp_path, scrape_window={"primary_only": True}), now=NOW
    )
    records = [json.loads(l) for l in outcome.handoff_path.read_text().splitlines() if l.strip()]
    assert [r["arxiv_id"] for r in records] == ["2501.00101"]


def test_production_surplus_preserved_past_max_per_paper(tmp_path, monkeypatch):
    """A theorem-dense paper past max_per_paper flows to surplus, never dropped."""
    dense = {
        "2501.00101": {
            "title": "Dense", "primary_category": "math.AP", "categories": ("math.AP",),
            "tex": (
                r"\begin{theorem}Solution count one is one.\end{theorem}"
                r"\begin{theorem}Solution count two is two.\end{theorem}"
                r"\begin{theorem}Solution count three is three.\end{theorem}"
            ),
        },
    }

    def fake_build_index(manifest, run_dir, window, counts, charge):
        idx = _FakeIndex(dense)
        counts["oai_requests"] = idx.oai_requests
        return idx

    def fake_open_store(manifest, run_dir, window):
        (run_dir / "_chunks").mkdir(parents=True, exist_ok=True)
        return _FakeStore(dense, run_dir / "_chunks")

    monkeypatch.setattr(arxiv_bulk, "_build_category_index", fake_build_index)
    monkeypatch.setattr(arxiv_bulk, "_open_chunk_store", fake_open_store)
    outcome = arxiv_bulk.run(
        _manifest(tmp_path, scrape_window={"extraction": "latex", "max_per_paper": 1}),
        now=NOW,
    )
    assert outcome.record_count == 1
    assert outcome.surplus_count == 2  # two extra theorems preserved, not rejected
    assert outcome.surplus_path is not None and outcome.surplus_path.exists()


def test_production_refuses_missing_manifest_path(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.scrape_window = {"extraction": "qa", "category": "math.AP"}
    with pytest.raises(ValueError, match="manifest_path"):
        arxiv_bulk.run(manifest, now=NOW)


def test_production_refuses_abstract_extraction(tmp_path):
    manifest_path = tmp_path / "src_manifest.xml"
    manifest_path.write_text(_MANIFEST_XML, encoding="utf-8")
    manifest = _manifest(tmp_path)
    manifest.scrape_window = {
        "extraction": "abstract", "category": "math.AP",
        "manifest_path": str(manifest_path),
    }
    with pytest.raises(ValueError, match="extraction"):
        arxiv_bulk.run(manifest, now=NOW)


def test_production_refuses_a_budget_below_the_estimate(tmp_path):
    manifest = _manifest(tmp_path, call_budget=2)  # far below the qa estimate
    with pytest.raises(ValueError, match="call_budget"):
        arxiv_bulk.run(manifest, now=NOW)


def test_production_refuses_a_foreign_source_type(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.source_type = "realmath_scrape"
    with pytest.raises(ValueError, match="source_type"):
        arxiv_bulk.run(manifest, now=NOW)


def test_production_unused_tar_helpers_are_importable():
    """Guard: the tar/io imports exist for building chunk members if needed."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz"):
        pass
    assert buf.getvalue()


def test_default_oai_fetcher_returns_oairesponse_on_200(monkeypatch):
    # The live-path wiring gap that crashed the first production run: the
    # default fetcher must be REAL. Offline: urlopen is monkeypatched.
    import urllib.request

    from icepick.allocation.bulk.category_index import OAIResponse

    class _Resp:
        status = 200

        def read(self):
            return b"<xml>ok</xml>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["ua"] = request.get_header("User-agent")
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = arxiv_bulk._default_oai_fetcher()("https://oaipmh.example/oai?verb=X")
    assert isinstance(out, OAIResponse)
    assert (out.status, out.retry_after, out.text) == (200, None, "<xml>ok</xml>")
    assert seen["ua"].startswith("icepick-arxiv-bulk/") and seen["timeout"] == 60


def test_default_oai_fetcher_returns_503_with_retry_after(monkeypatch):
    # Throttles come back as data (CategoryIndex owns backoff), never raise.
    import email.message
    import io as _io
    import urllib.error
    import urllib.request

    headers = email.message.Message()
    headers["Retry-After"] = "30"

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 503, "throttled", headers, _io.BytesIO(b"slow down")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = arxiv_bulk._default_oai_fetcher()("https://oaipmh.example/oai?verb=X")
    assert (out.status, out.retry_after, out.text) == (503, 30.0, "slow down")

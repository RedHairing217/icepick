"""Production runs pause and resume; they do not die.

Adapter- and CLI-level restartability: an interrupted production run writes
a loudly-marked partial handoff plus a checkpoint, exits non-zero from the
CLI, and re-running the very same command completes without refetching what
was already acquired. No network — fetchers are canned.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone

import pytest

from icepick import cli
from icepick.allocation.adapters import realmath_scrape
from icepick.allocation.manifests import write_manifest
from icepick.allocation.scrape import realmath as source
from icepick.contracts.manifests import ApprovedManifest, SOURCE_REALMATH_SCRAPE

NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2604.00001v1</id>
    <title>Paper One</title>
    <summary>Abstract one.</summary>
    <arxiv:primary_category term="math.AP"/>
    <category term="math.AP"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2604.00002v1</id>
    <title>Paper Two</title>
    <summary>Abstract two.</summary>
    <arxiv:primary_category term="math.AP"/>
    <category term="math.AP"/>
  </entry>
</feed>"""

_EMPTY = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

def _tex_for(arxiv_id: str) -> bytes:
    # Distinct statement per paper — otherwise normalise's statement dedup
    # (correctly) collapses the two papers into one handoff record.
    return rf"\begin{{theorem}}Interesting result in {arxiv_id}.\end{{theorem}}".encode()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during a resume test")

    monkeypatch.setattr(socket, "socket", _blocked)


@pytest.fixture(autouse=True)
def _canned_arxiv(monkeypatch):
    monkeypatch.setattr(
        source, "default_arxiv_fetcher",
        lambda query, *, start, max_results: _FEED if start == 0 else _EMPTY,
    )


def _manifest(tmp_path, **overrides):
    base = dict(
        run_id="20260703T120000Z",
        source_type=SOURCE_REALMATH_SCRAPE,
        processor_mode="production",
        requested_by="alice",
        requested_at="2026-07-03T00:00:00Z",
        approved_by="bob",
        approved_at="2026-07-03T01:00:00Z",
        source_name="pde_2026Q2",
        target_count=5,
        call_budget=1000,
        judge_enabled=False,
        confirmation_enabled=False,
        enable_leakage=False,
        enable_duplication=True,
        enable_robustness=False,
        families=["pde"],
        scrape_window={"category": "math.AP", "extraction": "latex"},
        truth_policy="extracted",
        output_dir=str(tmp_path / "intake"),
    )
    base.update(overrides)
    return ApprovedManifest(**base)


def _run_dir(tmp_path):
    return tmp_path / "intake" / "runs" / "20260703T120000Z"


def test_interrupted_production_run_is_resumable(tmp_path, monkeypatch):
    import gzip

    latex_fetches: list = []
    interrupt_armed = {"on": True}

    def latex_fetcher(arxiv_id, **kwargs):
        if arxiv_id == "2604.00002" and interrupt_armed["on"]:
            raise KeyboardInterrupt  # Ctrl-C mid-run, after paper one committed
        latex_fetches.append(arxiv_id)
        return gzip.compress(_tex_for(arxiv_id))

    monkeypatch.setattr(source, "default_latex_source_fetcher", latex_fetcher)

    # First invocation: pauses, keeps paper one, marks everything resumable.
    first = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    assert first.interrupted is True
    assert first.record_count == 1
    assert (_run_dir(tmp_path) / "_progress" / "INCOMPLETE").exists()
    report = first.report_path.read_text()
    assert "INTERRUPTED" in report and "resumable" in report
    assert any("rerun the same" in w for w in first.warnings)

    # Second invocation of the SAME manifest: completes without refetching.
    interrupt_armed["on"] = False
    second = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    assert second.interrupted is False
    assert second.record_count == 2
    assert second.acquisition["resumed_papers"] == 1
    assert not (_run_dir(tmp_path) / "_progress" / "INCOMPLETE").exists()
    # Paper one's e-print was fetched exactly once across both invocations.
    assert latex_fetches == ["2604.00001", "2604.00002"]
    assert "Resumed: 1 papers served from the checkpoint" in second.report_path.read_text()


def test_completed_production_rerun_is_idempotent_and_free(tmp_path, monkeypatch):
    import gzip

    latex_fetches: list = []

    def latex_fetcher(arxiv_id, **kwargs):
        latex_fetches.append(arxiv_id)
        return gzip.compress(_tex_for(arxiv_id))

    monkeypatch.setattr(source, "default_latex_source_fetcher", latex_fetcher)

    first = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    handoff = first.handoff_path.read_bytes()
    second = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    assert second.handoff_path.read_bytes() == handoff
    assert second.acquisition["resumed_papers"] == 2
    assert len(latex_fetches) == 2  # both fetched in run one, zero on the rerun


def test_report_carries_throttle_telemetry_from_a_429_killed_first_invocation(tmp_path, monkeypatch):
    """Run-lifetime telemetry, end to end: invocation one dies to arXiv's
    limiter before committing a single paper (so it writes no report at
    all); the cooled-down resume completes, and its source_report.md must
    still show the throttling that killed invocation one."""
    import gzip

    def throttled(query, *, start, max_results):
        # What a final-retry 429 does inside _http_get: notify the scrape's
        # observers (durable telemetry + cooldown stamp), then die.
        source._notify_rate_limit(429, 6.0)
        raise OSError("429 Client Error: rate limited")

    monkeypatch.setattr(source, "default_arxiv_fetcher", throttled)
    with pytest.raises(OSError):
        realmath_scrape.run(_manifest(tmp_path), now=NOW)
    progress = _run_dir(tmp_path) / "_progress"
    assert (progress / "rate_limit_events.jsonl").exists()
    assert not (_run_dir(tmp_path) / "reports" / "source_report.md").exists()

    # Resume after the cooldown: canned feed and e-prints, run completes.
    monkeypatch.setenv("ICEPICK_ARXIV_COOLDOWN_SECONDS", "0")
    monkeypatch.setattr(
        source, "default_arxiv_fetcher",
        lambda query, *, start, max_results: _FEED if start == 0 else _EMPTY,
    )
    monkeypatch.setattr(
        source, "default_latex_source_fetcher",
        lambda arxiv_id, **kw: gzip.compress(_tex_for(arxiv_id)),
    )
    outcome = realmath_scrape.run(_manifest(tmp_path), now=NOW)
    assert outcome.interrupted is False
    assert outcome.acquisition["rate_limit_events"] == 1
    assert outcome.acquisition["rate_limit_backoff_seconds"] == pytest.approx(6.0)
    assert outcome.acquisition["rate_limit_statuses"] == {"429": 1}
    report = outcome.report_path.read_text()
    assert "## arXiv throttle telemetry" in report
    assert "| 429/503 encounters | 1 |" in report
    assert "| total backoff seconds | 6.0 |" in report


def test_flow_testing_replay_creates_no_progress_dir(tmp_path, fixtures_dir):
    manifest = _manifest(
        tmp_path, processor_mode="flow_testing",
        calibration_sheet=str(fixtures_dir / "realmath" / "qa_candidates.jsonl"),
        scrape_window=None,
    )
    outcome = realmath_scrape.run(manifest, now=NOW)
    assert outcome.interrupted is False
    assert not (_run_dir(tmp_path) / "_progress").exists()


def test_cli_reports_a_network_death_as_a_clean_resumable_failure(tmp_path, monkeypatch, capsys):
    """A throttled/dead network is a coded failure with resume advice, not a traceback."""
    def throttled(query, *, start, max_results):
        raise OSError("429 Client Error: rate limited")

    monkeypatch.setattr(source, "default_arxiv_fetcher", throttled)
    manifest = _manifest(tmp_path)
    manifest_path = write_manifest(manifest, manifest.output_dir)

    rc = cli.main(["allocation", "run", "--manifest", str(manifest_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "E_NETWORK" in err
    assert "rerun the same command" in err


def test_cli_run_exits_nonzero_on_an_interrupted_run(tmp_path, monkeypatch, capsys):
    import gzip

    def interrupting_fetcher(arxiv_id, **kwargs):
        if arxiv_id == "2604.00002":
            raise KeyboardInterrupt
        return gzip.compress(_tex_for(arxiv_id))

    monkeypatch.setattr(source, "default_latex_source_fetcher", interrupting_fetcher)
    manifest = _manifest(tmp_path)
    manifest_path = write_manifest(manifest, manifest.output_dir)

    rc = cli.main(["allocation", "run", "--manifest", str(manifest_path)])
    assert rc == 1  # a chained pipeline must not consume the partial handoff
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "interrupted_resumable"
    assert "resume" in summary["next"]

    # The printed "next" command really does finish the job.
    monkeypatch.setattr(
        source, "default_latex_source_fetcher",
        lambda arxiv_id, **kw: gzip.compress(_tex_for(arxiv_id)),
    )
    rc = cli.main(["allocation", "run", "--manifest", str(manifest_path)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "complete"
    assert summary["counts"]["handoff_records"] == 2

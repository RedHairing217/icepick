"""Restartability: pause/restart acceptable, full kill unacceptable.

Checkpoint store unit tests plus scrape-level resume semantics: every
finished paper is committed to disk, an interrupt (Ctrl-C) pauses cleanly,
re-running resumes without refetching, and cached QA answers never re-bill.
No network anywhere — feeds and fetchers are canned.
"""

from __future__ import annotations

import socket
from datetime import datetime, timedelta, timezone

import pytest

from icepick.allocation.scrape import realmath as source
from icepick.allocation.scrape.checkpoint import RateLimitCooldownError, ScrapeCheckpoint

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
  <entry>
    <id>http://arxiv.org/abs/2604.00003v1</id>
    <title>Paper Three</title>
    <summary>Abstract three.</summary>
    <arxiv:primary_category term="math.AP"/>
    <category term="math.AP"/>
  </entry>
</feed>"""

_EMPTY = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _feed_fetcher(query, *, start, max_results):
    return _FEED if start == 0 else _EMPTY


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a checkpoint test")

    monkeypatch.setattr(socket, "socket", _blocked)


# --- store unit tests -----------------------------------------------------------


def test_commits_survive_a_new_instance(tmp_path):
    first = ScrapeCheckpoint(tmp_path / "_progress")
    first.commit("2604.00001", [{"statement": "s1"}, {"statement": "s2"}])

    reloaded = ScrapeCheckpoint(tmp_path / "_progress")
    assert reloaded.stored_candidates("2604.00001") == [{"statement": "s1"}, {"statement": "s2"}]
    assert reloaded.stored_candidates("2604.00002") is None
    assert reloaded.resumed_papers == 1


def test_incomplete_marker_lifecycle(tmp_path):
    checkpoint = ScrapeCheckpoint(tmp_path / "_progress")
    assert checkpoint.resuming is False
    checkpoint.begin()
    assert (tmp_path / "_progress" / "INCOMPLETE").exists()
    assert ScrapeCheckpoint(tmp_path / "_progress").resuming is True
    checkpoint.mark_complete()
    assert not (tmp_path / "_progress" / "INCOMPLETE").exists()


def test_caching_generator_never_bills_twice(tmp_path):
    calls = []

    def generator(statement, **kwargs):
        calls.append(statement)
        return {"question": f"Q:{statement}", "answer": "4"}

    first = ScrapeCheckpoint(tmp_path / "_progress")
    cached = first.caching_generator(generator)
    assert cached("theorem A")["answer"] == "4"
    assert cached("theorem A")["answer"] == "4"  # hit, no second call
    # A fresh instance (new process) reads the same disk cache.
    second = ScrapeCheckpoint(tmp_path / "_progress")
    assert second.caching_generator(generator)("theorem A")["answer"] == "4"
    assert calls == ["theorem A"]


def test_caching_generator_caches_no_answer_results(tmp_path):
    calls = []

    def generator(statement, **kwargs):
        calls.append(statement)
        return None  # theorem states no closed-form answer

    checkpoint = ScrapeCheckpoint(tmp_path / "_progress")
    cached = checkpoint.caching_generator(generator)
    assert cached("theorem B") is None
    assert ScrapeCheckpoint(tmp_path / "_progress").caching_generator(generator)("theorem B") is None
    assert calls == ["theorem B"]


def test_caching_gate_never_bills_twice(tmp_path):
    calls = []

    def gate(statement, **kwargs):
        calls.append(statement)
        return "accept" in statement

    first = ScrapeCheckpoint(tmp_path / "_progress")
    cached = first.caching_gate(gate)
    assert cached("accept theorem") is True
    assert cached("reject theorem") is False
    assert cached("accept theorem") is True

    second = ScrapeCheckpoint(tmp_path / "_progress")
    assert second.caching_gate(gate)("reject theorem") is False
    assert calls == ["accept theorem", "reject theorem"]


def test_rate_limit_marker_blocks_only_during_cooldown(tmp_path, monkeypatch):
    monkeypatch.setenv("ICEPICK_ARXIV_COOLDOWN_SECONDS", "1200")
    now = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)
    checkpoint = ScrapeCheckpoint(tmp_path / "_progress")
    checkpoint.stamp_rate_limited(now=now)

    with pytest.raises(RateLimitCooldownError, match="retry after 12:20 UTC"):
        checkpoint.enforce_rate_limit_cooldown(now=now)

    checkpoint.enforce_rate_limit_cooldown(now=now + timedelta(minutes=21))
    checkpoint.clear_rate_limit()
    checkpoint.enforce_rate_limit_cooldown(now=now)


def test_rate_limit_telemetry_accumulates_across_instances(tmp_path):
    first = ScrapeCheckpoint(tmp_path / "_progress")
    first.record_rate_limit(429, 3.0)
    first.record_rate_limit(429, 6.0)

    # A new instance (a resumed invocation) merges the prior events and
    # keeps counting — the totals cover the run's whole lifetime.
    second = ScrapeCheckpoint(tmp_path / "_progress")
    second.record_rate_limit(503, 1.5)
    assert second.rate_limit_telemetry() == {
        "events": 3,
        "backoff_seconds": pytest.approx(10.5),
        "statuses": {"429": 2, "503": 1},
    }


def test_clearing_the_cooldown_marker_keeps_the_telemetry_log(tmp_path):
    checkpoint = ScrapeCheckpoint(tmp_path / "_progress")
    checkpoint.record_rate_limit(429, 3.0)
    checkpoint.stamp_rate_limited()
    checkpoint.clear_rate_limit()  # the next successful request drops the marker...
    assert not (tmp_path / "_progress" / "rate_limited_at").exists()
    # ...but the event log is history, not transient cooldown state.
    assert ScrapeCheckpoint(tmp_path / "_progress").rate_limit_telemetry()["events"] == 1


def test_torn_rate_limit_event_line_is_skipped_not_fatal(tmp_path):
    checkpoint = ScrapeCheckpoint(tmp_path / "_progress")
    checkpoint.record_rate_limit(429, 3.0)
    with (tmp_path / "_progress" / "rate_limit_events.jsonl").open("a") as fh:
        fh.write('{"at": "2026-07-04T07:12:51Z", "status": 503, "backoff_')
    assert ScrapeCheckpoint(tmp_path / "_progress").rate_limit_telemetry() == {
        "events": 1,
        "backoff_seconds": pytest.approx(3.0),
        "statuses": {"429": 1},
    }


def test_torn_final_line_is_skipped_not_fatal(tmp_path):
    checkpoint = ScrapeCheckpoint(tmp_path / "_progress")
    checkpoint.commit("2604.00001", [{"statement": "s1"}])
    # Simulate a kill mid-write: a truncated trailing line.
    with (tmp_path / "_progress" / "candidates.jsonl").open("a") as fh:
        fh.write('{"arxiv_id": "2604.00002", "candi')
    reloaded = ScrapeCheckpoint(tmp_path / "_progress")
    assert reloaded.stored_candidates("2604.00001") == [{"statement": "s1"}]
    assert reloaded.stored_candidates("2604.00002") is None


# --- scrape-level resume semantics ------------------------------------------------


def test_interrupt_pauses_cleanly_and_resume_completes_without_redoing_work(tmp_path):
    extracted: list = []

    def flaky_extractor(paper, *, family=None):
        if paper.arxiv_id == "2604.00002" and not (tmp_path / "resumed").exists():
            raise KeyboardInterrupt  # operator hits Ctrl-C mid-paper
        extracted.append(paper.arxiv_id)
        return [{"arxiv_id": paper.arxiv_id, "statement": f"S {paper.arxiv_id}"}]

    progress = tmp_path / "_progress"

    first = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="s", target_count=10,
        fetcher=_feed_fetcher, extractor=flaky_extractor,
        checkpoint=ScrapeCheckpoint(progress),
    )
    assert first.interrupted is True
    assert [c["arxiv_id"] for c in first.candidates] == ["2604.00001"]  # committed work kept
    assert extracted == ["2604.00001"]

    (tmp_path / "resumed").touch()  # let the extractor succeed on the rerun
    second = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="s", target_count=10,
        fetcher=_feed_fetcher, extractor=flaky_extractor,
        checkpoint=ScrapeCheckpoint(progress),
    )
    assert second.interrupted is False
    assert second.resumed_papers == 1  # paper one came from the store
    assert [c["arxiv_id"] for c in second.candidates] == ["2604.00001", "2604.00002", "2604.00003"]
    # Paper one was extracted exactly once across both invocations.
    assert extracted == ["2604.00001", "2604.00002", "2604.00003"]


def test_resumed_run_equals_an_uninterrupted_control_run(tmp_path):
    def extractor(paper, *, family=None):
        return [{"arxiv_id": paper.arxiv_id, "statement": f"S {paper.arxiv_id}"}]

    control = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="s", target_count=10,
        fetcher=_feed_fetcher, extractor=extractor,
    )
    checkpointed = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="s", target_count=10,
        fetcher=_feed_fetcher, extractor=extractor,
        checkpoint=ScrapeCheckpoint(tmp_path / "_progress"),
    )
    rerun = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="s", target_count=10,
        fetcher=_feed_fetcher, extractor=extractor,
        checkpoint=ScrapeCheckpoint(tmp_path / "_progress"),
    )
    assert checkpointed.candidates == control.candidates
    assert rerun.candidates == control.candidates  # idempotent re-run, all from store
    assert rerun.resumed_papers == 3


def test_qa_resume_rebills_only_the_lost_in_flight_theorem(tmp_path, monkeypatch):
    """Kill mid-paper: at most the in-flight item is redone; cached QA is free."""
    import gzip

    tex = (
        r"\begin{theorem}The first count is one.\end{theorem}"
        r"\begin{lemma}The second count is two.\end{lemma}"
    )
    monkeypatch.setattr(
        source, "default_latex_source_fetcher",
        lambda arxiv_id, **kw: gzip.compress(tex.encode()),
    )
    generator_calls: list = []
    interrupt_armed = {"on": True}

    def generator(statement, **kwargs):
        if "second" in statement and interrupt_armed["on"]:
            raise KeyboardInterrupt  # killed mid-paper, after theorem one was paid for
        generator_calls.append(statement)
        return {"question": statement, "answer": "1" if "first" in statement else "2"}

    monkeypatch.setattr(source, "default_qa_quality_gate", lambda s, **kw: True)
    monkeypatch.setattr(source, "default_qa_generator", lambda s, **kw: generator(s, **kw))

    one_paper_feed = _FEED.split("<entry>")[0] + "<entry>" + _FEED.split("<entry>")[1] + "</feed>"
    fetcher = lambda q, *, start, max_results: one_paper_feed if start == 0 else _EMPTY  # noqa: E731
    progress = tmp_path / "_progress"

    first = source.scrape(
        scrape_window={"category": "math.AP", "extraction": "qa"},
        source_name="s", target_count=10, fetcher=fetcher,
        checkpoint=ScrapeCheckpoint(progress),
    )
    assert first.interrupted is True
    assert first.candidates == []  # paper never finished, nothing committed

    interrupt_armed["on"] = False
    second = source.scrape(
        scrape_window={"category": "math.AP", "extraction": "qa"},
        source_name="s", target_count=10, fetcher=fetcher,
        checkpoint=ScrapeCheckpoint(progress),
    )
    assert second.interrupted is False
    assert len(second.candidates) == 2
    # Theorem one was billed once (first run, cached); only theorem two on resume.
    assert [("first" in s) for s in generator_calls] == [True, False]
    assert second.qa_calls == 1  # the resumed invocation spent one LLM call, not two


def test_budget_is_never_exceeded_mid_paper_and_resume_finishes_the_job(tmp_path, monkeypatch):
    """The live-pilot bug: one many-theorem paper must not spend past the cap."""
    import gzip

    tex = "".join(
        rf"\begin{{theorem}}Count number {i} equals {i}.\end{{theorem}}" for i in range(10)
    )
    monkeypatch.setattr(
        source, "default_latex_source_fetcher",
        lambda arxiv_id, **kw: gzip.compress(tex.encode()),
    )
    generator_calls: list = []

    def generator(statement, **kwargs):
        generator_calls.append(statement)
        return {"question": statement, "answer": str(len(generator_calls))}

    monkeypatch.setattr(source, "default_qa_quality_gate", lambda s, **kw: True)
    monkeypatch.setattr(source, "default_qa_generator", lambda s, **kw: generator(s, **kw))

    one_paper_feed = _FEED.split("<entry>")[0] + "<entry>" + _FEED.split("<entry>")[1] + "</feed>"
    fetcher = lambda q, *, start, max_results: one_paper_feed if start == 0 else _EMPTY  # noqa: E731
    progress = tmp_path / "_progress"

    first = source.scrape(
        scrape_window={"category": "math.AP", "extraction": "qa"},
        source_name="s", target_count=20, fetcher=fetcher, call_budget=5,
        checkpoint=ScrapeCheckpoint(progress),
    )
    # Exactly 5 paid calls: 1 arXiv query + 1 e-print fetch + 2 gate calls
    # (Haiku pre-filter) + 1 Sonnet generator call. Charge order goes
    # gate→qa per theorem, so the budget clips between theorem 2's gate
    # and its generator call.
    assert first.queries + first.latex_fetches + first.gate_calls + first.qa_calls == 5
    assert first.gate_calls == 2
    assert first.qa_calls == 1
    assert first.interrupted is True
    assert any("call budget 5 exhausted" in w for w in first.warnings)

    # Rerun without a cap: the one paid generator answer and the two prior
    # gate verdicts come from cache for free. Only uncached gates/generator
    # calls are billed on resume.
    second = source.scrape(
        scrape_window={"category": "math.AP", "extraction": "qa"},
        source_name="s", target_count=20, fetcher=fetcher,
        checkpoint=ScrapeCheckpoint(progress),
    )
    assert second.interrupted is False
    assert len(second.candidates) == 10
    assert len(generator_calls) == 10  # each theorem's generator billed exactly once, ever
    assert second.qa_calls == 9  # only the unpaid nine on the rerun (theorem 1 hit cache)
    assert second.gate_calls == 8  # first two gate verdicts hit the new disk cache

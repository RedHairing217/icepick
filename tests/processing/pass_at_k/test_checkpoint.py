"""Restartability: pause/restart acceptable, full kill unacceptable.

Checkpoint store unit tests for the pass@k stage, mirroring the scraper's
(:mod:`tests.allocation.scrape.test_checkpoint`): every finished record and
rollout is committed to disk as it happens, a resumed run re-bills nothing
it already paid for, a torn final line (kill mid-write) is skipped rather
than fatal, and concurrent commits never interleave. No network anywhere.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

from icepick.processing.pass_at_k.checkpoint import PassAtKCheckpoint, rollout_key


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a checkpoint test")

    monkeypatch.setattr(socket, "socket", _blocked)


# --- store unit tests -----------------------------------------------------------


def test_commits_survive_a_new_instance(tmp_path):
    first = PassAtKCheckpoint(tmp_path / "_progress")
    row = {"uid": "u1", "statement": "s", "pass_at_k": 0.5, "label": "band"}
    first.commit_record("u1", row)

    reloaded = PassAtKCheckpoint(tmp_path / "_progress")
    assert reloaded.stored_record("u1") == row
    assert reloaded.stored_record("u2") is None
    assert reloaded.resumed_records == 1


def test_incomplete_marker_lifecycle(tmp_path):
    checkpoint = PassAtKCheckpoint(tmp_path / "_progress")
    assert checkpoint.resuming is False
    checkpoint.begin()
    assert (tmp_path / "_progress" / "INCOMPLETE").exists()
    assert PassAtKCheckpoint(tmp_path / "_progress").resuming is True
    checkpoint.mark_complete()
    assert not (tmp_path / "_progress" / "INCOMPLETE").exists()
    assert PassAtKCheckpoint(tmp_path / "_progress").resuming is False


def test_cache_never_bills_twice(tmp_path):
    first = PassAtKCheckpoint(tmp_path / "_progress")
    key = rollout_key("qwen/qwen3-8b", "What is 2+2?", 0.7, False, 0)
    assert first.cached_output(key) is None
    first.store_output(key, "\\boxed{4}")
    assert first.cached_output(key) == "\\boxed{4}"
    # A fresh instance (new process) reads the same disk cache.
    second = PassAtKCheckpoint(tmp_path / "_progress")
    assert second.cached_output(key) == "\\boxed{4}"


def test_empty_string_output_is_cached(tmp_path):
    first = PassAtKCheckpoint(tmp_path / "_progress")
    key = rollout_key("qwen/qwen3-8b", "What is 2+2?", 0.7, False, 1)
    first.store_output(key, "")  # a real (if useless) response, paid for once
    second = PassAtKCheckpoint(tmp_path / "_progress")
    assert second.cached_output(key) == ""
    assert second.cached_output(key) is not None


def test_rollouts_append_to_the_audit_trail(tmp_path):
    checkpoint = PassAtKCheckpoint(tmp_path / "_progress")
    row = {
        "uid": "u1",
        "rollout_uid": "u1-r00",
        "sample_idx": 0,
        "output": "\\boxed{4}",
        "candidate": "4",
        "verdict": "correct",
        "from_cache": False,
    }
    checkpoint.append_rollout(row)
    lines = (tmp_path / "_progress" / "rollouts.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [row]
    # The audit data stays forever and a reload with it present is fine.
    PassAtKCheckpoint(tmp_path / "_progress")


def test_torn_final_line_is_skipped_not_fatal(tmp_path):
    checkpoint = PassAtKCheckpoint(tmp_path / "_progress")
    checkpoint.commit_record("u1", {"uid": "u1", "label": "band"})
    checkpoint.store_output("k1", "out")
    # Simulate a kill mid-write: truncated trailing lines.
    with (tmp_path / "_progress" / "records_done.jsonl").open("a") as fh:
        fh.write('{"uid": "u2", "lab')
    with (tmp_path / "_progress" / "llm_cache.jsonl").open("a") as fh:
        fh.write('{"key": "k2", "outp')
    reloaded = PassAtKCheckpoint(tmp_path / "_progress")
    assert reloaded.stored_record("u1") == {"uid": "u1", "label": "band"}
    assert reloaded.stored_record("u2") is None
    assert reloaded.cached_output("k1") == "out"
    assert reloaded.cached_output("k2") is None


# --- rollout_key ------------------------------------------------------------------


def test_rollout_key_stable_and_distinct_per_field():
    base = rollout_key("qwen/qwen3-8b", "What is 2+2?", 0.7, False, 0)
    assert base == rollout_key("qwen/qwen3-8b", "What is 2+2?", 0.7, False, 0)  # stable
    assert len(base) == 16 and all(c in "0123456789abcdef" for c in base)
    assert base != rollout_key("qwen/qwen3-8b", "What is 2+2?", 0.7, False, 1)  # sample_idx
    assert base != rollout_key("claude-haiku-4-5", "What is 2+2?", 0.7, False, 0)  # model
    assert base != rollout_key("qwen/qwen3-8b", "What is 2+2?", 0.0, False, 0)  # temperature
    assert base != rollout_key("qwen/qwen3-8b", "What is 2+2?", 0.7, True, 0)  # think


# --- concurrency ------------------------------------------------------------------


def test_concurrent_commits_produce_intact_lines(tmp_path):
    checkpoint = PassAtKCheckpoint(tmp_path / "_progress")

    def worker(thread_idx):
        for i in range(25):
            uid = f"t{thread_idx}-{i:02d}"
            checkpoint.commit_record(uid, {"uid": uid, "pass_at_k": 0.5, "label": "band"})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = (tmp_path / "_progress" / "records_done.jsonl").read_text().splitlines()
    assert len(lines) == 100
    rows = [json.loads(line) for line in lines]  # every line parses: no interleaving
    assert {r["uid"] for r in rows} == {f"t{t}-{i:02d}" for t in range(4) for i in range(25)}
    assert PassAtKCheckpoint(tmp_path / "_progress").resumed_records == 100

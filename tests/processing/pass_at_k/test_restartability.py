"""Restartability at the runner level: pause/restart acceptable, full kill not.

Mirrors the scrape-level resume tests in
``tests/allocation/scrape/test_checkpoint.py``: an interrupt (Ctrl-C from
inside a backend call) pauses cleanly, finished work is kept, and the
rerun resumes at the exact rollout the first run died on without
re-billing anything already paid for.

Config note (same as test_runner_with_fake_backend.py): backends are
always injected, so ``backend='qwen_http'`` + a dummy ``backend_url``
keeps ``cfg.validate()`` green with no kill-switch flag and no network.
"""

from __future__ import annotations

import json

from icepick.processing.pass_at_k.base import LABEL_BAND, LABEL_SOLVED
from icepick.processing.pass_at_k.config import PassAtKConfig
from icepick.processing.pass_at_k.runner import run

_Q1, _Q2 = "What is 2+2?", "What is 5+5?"

_RECORDS = [
    {"uid": "u1", "source": "rm", "statement": _Q1, "truth": "4"},
    {"uid": "u2", "source": "rm", "statement": _Q2, "truth": "10"},
]

# Full rollout script: u1 -> solved (4/4), u2 -> band (2/4, modal wrong "3").
_OUTPUTS = {
    _Q1: ["\\boxed{4}"] * 4,
    _Q2: ["\\boxed{10}", "\\boxed{10}", "\\boxed{3}", "\\boxed{3}"],
}


class _CountingBackend:
    """Pops scripted outputs per question; every call is a paid call."""

    name = "fake"

    def __init__(self, outputs_by_question):
        self._outputs = {q: list(v) for q, v in outputs_by_question.items()}
        self.calls = []

    def call(self, question, *, k, temperature, max_tokens, think, timeout):
        self.calls.append(question)
        return [self._outputs[question].pop(0) for _ in range(k)]


class _FlakyBackend(_CountingBackend):
    """Raises KeyboardInterrupt (operator Ctrl-C) on paid call N."""

    def __init__(self, outputs_by_question, interrupt_on_call):
        super().__init__(outputs_by_question)
        self._interrupt_on = interrupt_on_call
        self._call_no = 0

    def call(self, question, **kwargs):
        self._call_no += 1
        if self._call_no == self._interrupt_on:
            raise KeyboardInterrupt
        return super().call(question, **kwargs)


def _cfg(tmp_path, out_name="out", **overrides):
    base = dict(
        mode="production",
        output_dir=tmp_path / out_name,
        backend="qwen_http",
        backend_url="http://127.0.0.1:9/never-called",
        model="fake-model",
        k=4,
        temperature=0.0,
        max_concurrent=1,   # deterministic sequential order for the interrupt
        max_retries=0,      # keep paid-call accounting exact
    )
    base.update(overrides)
    return PassAtKConfig(**base)


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_interrupt_pauses_cleanly_and_resume_rebills_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    progress = cfg.output_dir / "_progress"
    sleeps = []

    # Control: an uninterrupted run in its own dir, same scripted outputs.
    control = run(
        cfg=_cfg(tmp_path, out_name="control"),
        records=_RECORDS,
        backend=_CountingBackend(_OUTPUTS),
    )
    control_rows = _rows(control.output_path)
    assert [r["label"] for r in control_rows] == [LABEL_SOLVED, LABEL_BAND]

    # Run 1: Ctrl-C lands on paid call 6 — mid-record-2, after its first rollout.
    flaky = _FlakyBackend(_OUTPUTS, interrupt_on_call=6)
    outcome1 = run(cfg=cfg, records=_RECORDS, backend=flaky, sleep_fn=sleeps.append)

    assert outcome1.interrupted is True
    assert outcome1.model_calls == 5           # u1's four + u2's first rollout
    assert len(flaky.calls) == 5
    rows1 = _rows(outcome1.output_path)
    assert [r["uid"] for r in rows1] == ["u1"]  # committed work kept, u2 absent
    assert (progress / "INCOMPLETE").exists()   # not marked complete
    assert len(_rows(progress / "records_done.jsonl")) == 1
    manifest1 = json.loads(outcome1.manifest_path.read_text())
    assert manifest1["interrupted"] is True
    assert manifest1["warnings"]                # warns the operator to re-run

    # Run 2: healthy backend, same command. Record 1 comes from the store;
    # record 2 resumes at the exact rollout it died on — so the run-2 script
    # carries only the outputs run 1 did not already pay for (u2 sample 0
    # replays from the cache, never from the backend).
    counting = _CountingBackend({_Q1: [], _Q2: _OUTPUTS[_Q2][1:]})
    outcome2 = run(cfg=cfg, records=_RECORDS, backend=counting, sleep_fn=sleeps.append)

    assert outcome2.interrupted is False
    assert outcome2.resumed_records == 1        # u1 served from the store
    assert counting.calls == [_Q2] * 3          # only u2's unpaid rollouts
    assert outcome2.model_calls == 3
    assert not (progress / "INCOMPLETE").exists()

    rows2 = _rows(outcome2.output_path)
    assert [r["uid"] for r in rows2] == ["u1", "u2"]
    assert len({r["uid"] for r in rows2}) == 2  # regenerated clean, no duplicates
    assert rows2 == control_rows                # identical to the uninterrupted run

    # Cached rollouts were never re-billed: exactly k * n_records paid, ever.
    total_paid = len(flaky.calls) + len(counting.calls)
    assert total_paid == cfg.k * len(_RECORDS)
    assert sleeps == []  # interrupts never burn retries

    # The audit trail shows u2's first rollout replayed from cache on run 2.
    audit = _rows(progress / "rollouts.jsonl")
    u2_first = [r for r in audit if r["uid"] == "u2" and r["sample_idx"] == 0]
    assert len(u2_first) == 2                   # run 1 paid it, run 2 replayed it
    assert [r["from_cache"] for r in u2_first] == [False, True]


def test_rerun_after_completion_is_free_and_idempotent(tmp_path):
    cfg = _cfg(tmp_path)

    outcome1 = run(cfg=cfg, records=_RECORDS, backend=_CountingBackend(_OUTPUTS))
    assert outcome1.interrupted is False
    assert outcome1.model_calls == 8

    class _ExplodingBackend:
        name = "fake"

        def call(self, question, **kwargs):
            raise AssertionError("a completed run must not re-bill anything")

    outcome2 = run(cfg=cfg, records=_RECORDS, backend=_ExplodingBackend())
    assert outcome2.model_calls == 0
    assert outcome2.resumed_records == 2
    assert _rows(outcome2.output_path) == _rows(outcome1.output_path)
    assert outcome2.counts == outcome1.counts


def test_interrupt_on_the_very_first_record_keeps_nothing_but_resumes_fully(tmp_path):
    cfg = _cfg(tmp_path)
    progress = cfg.output_dir / "_progress"

    # Ctrl-C on the very first paid call: nothing committed, nothing cached...
    flaky = _FlakyBackend(_OUTPUTS, interrupt_on_call=1)
    outcome1 = run(cfg=cfg, records=_RECORDS, backend=flaky)
    assert outcome1.interrupted is True
    assert outcome1.model_calls == 0
    assert _rows(outcome1.output_path) == []
    assert (progress / "INCOMPLETE").exists()

    # ...so the rerun pays for everything, exactly once.
    counting = _CountingBackend(_OUTPUTS)
    outcome2 = run(cfg=cfg, records=_RECORDS, backend=counting)
    assert outcome2.interrupted is False
    assert outcome2.model_calls == cfg.k * len(_RECORDS)
    assert [r["label"] for r in _rows(outcome2.output_path)] == [LABEL_SOLVED, LABEL_BAND]
    assert not (progress / "INCOMPLETE").exists()

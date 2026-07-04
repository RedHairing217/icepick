"""Runner orchestration with a fake backend — no SDKs, no network.

Config choice, documented once for all three runner test files: the
tests always inject ``backend=<fake>``, so ``build_backend`` never runs
and no kill-switch flag is needed. To keep ``cfg.validate()`` green in
production mode we use ``backend='qwen_http'`` (exempt from the paid
kill switch) with a dummy ``backend_url`` that is never contacted.
"""

from __future__ import annotations

import json

from icepick.processing.pass_at_k.base import (
    DROP_DEGENERATE,
    DROP_GARBAGE_TRUTH,
    DROP_UNVERIFIABLE,
    LABEL_BAND,
    LABEL_DROP,
    LABEL_MISDIRECTION,
    LABEL_SOLVED,
)
from icepick.processing.pass_at_k.config import PassAtKConfig
from icepick.processing.pass_at_k.runner import run


class _FakeBackend:
    """Scripted outputs keyed by question; records calls; reports usage."""

    name = "fake"

    def __init__(self, outputs_by_question: dict):
        self._outputs = {q: list(v) for q, v in outputs_by_question.items()}
        self.calls = []
        self._input_tokens = 0
        self._output_tokens = 0

    def call(self, question, *, k, temperature, max_tokens, think, timeout):
        self.calls.append(question)
        out = [self._outputs[question].pop(0) for _ in range(k)]
        self._input_tokens += 10 * k
        self._output_tokens += 5 * k
        return out

    def usage(self):
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        }


class _ExplodingBackend:
    """A backend that must never be called (drops / passthroughs)."""

    name = "fake"

    def call(self, question, **kwargs):
        raise AssertionError(f"backend must not be called (got {question!r})")


def _cfg(tmp_path, **overrides):
    base = dict(
        mode="production",
        output_dir=tmp_path / "out",
        backend="qwen_http",                       # kill-switch-exempt; see module docstring
        backend_url="http://127.0.0.1:9/never-called",
        model="fake-model",
        k=4,
        temperature=0.0,
        max_concurrent=1,
    )
    base.update(overrides)
    return PassAtKConfig(**base)


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


_THREE_RECORDS = [
    {"uid": "uid-solved", "source": "rm", "statement": "What is 2+2?",
     "truth": "4", "extra": "keep-me"},
    {"uid": "uid-band", "source": "rm", "statement": "What is 1+3?",
     "truth": "4"},
    {"uid": "uid-misdir", "source": "rm", "statement": "What is 8-4?",
     "truth": "4"},
]

_THREE_OUTPUTS = {
    "What is 2+2?": ["\\boxed{4}"] * 4,
    "What is 1+3?": ["\\boxed{4}", "\\boxed{4}", "\\boxed{7}", "\\boxed{9}"],
    "What is 8-4?": ["\\boxed{7}"] * 4,
}


def test_three_labels_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    backend = _FakeBackend(_THREE_OUTPUTS)

    outcome = run(cfg=cfg, records=_THREE_RECORDS, backend=backend)

    rows = _rows(outcome.output_path)
    assert [r["uid"] for r in rows] == ["uid-solved", "uid-band", "uid-misdir"]
    assert len({r["uid"] for r in rows}) == 3  # no duplicate uids

    solved, band, misdir = rows
    assert solved["pass_at_k"] == 1.0
    assert solved["label"] == LABEL_SOLVED
    assert solved["n_correct"] == 4 and solved["n_wrong"] == 0 and solved["n_degenerate"] == 0
    assert solved["rollout_uids"] == [f"uid-solved-r{i:02d}" for i in range(4)]
    # Original fields preserved verbatim.
    assert solved["statement"] == "What is 2+2?"
    assert solved["truth"] == "4"
    assert solved["extra"] == "keep-me"

    assert band["pass_at_k"] == 0.5
    assert band["label"] == LABEL_BAND
    assert band["top_wrong_share"] == 0.25
    assert band["modal_wrong"] == "7"  # tie 7-vs-9 resolves first-seen

    assert misdir["pass_at_k"] == 0.0
    assert misdir["label"] == LABEL_MISDIRECTION
    assert misdir["top_wrong_share"] == 1.0
    assert misdir["modal_wrong"] == "7"

    assert outcome.counts[LABEL_SOLVED] == 1
    assert outcome.counts[LABEL_BAND] == 1
    assert outcome.counts[LABEL_MISDIRECTION] == 1
    assert outcome.counts["dropped"] == 0
    assert outcome.counts["pre_labeled"] == 0
    assert outcome.model_calls == 12
    assert outcome.resumed_records == 0
    assert outcome.interrupted is False

    # Input echo written with uids.
    input_rows = _rows(cfg.output_dir / "pass_at_k_input.jsonl")
    assert [r["uid"] for r in input_rows] == [r["uid"] for r in _THREE_RECORDS]

    # Manifest exists and echoes the config.
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["stage"] == "pass_at_k"
    assert manifest["config"] == cfg.echo()
    assert manifest["config"]["k"] == 4
    assert manifest["counts"] == outcome.counts
    assert manifest["calibration_replay"] is False


def test_threadpool_path_matches_sequential_labels(tmp_path):
    cfg = _cfg(tmp_path, max_concurrent=2)
    backend = _FakeBackend(_THREE_OUTPUTS)

    outcome = run(cfg=cfg, records=_THREE_RECORDS, backend=backend)

    rows = _rows(outcome.output_path)
    # Output stays in input order even when scoring is concurrent.
    assert [r["uid"] for r in rows] == ["uid-solved", "uid-band", "uid-misdir"]
    assert [r["label"] for r in rows] == [LABEL_SOLVED, LABEL_BAND, LABEL_MISDIRECTION]
    assert outcome.model_calls == 12


def test_manifest_token_usage_and_estimated_cost(tmp_path):
    cfg = _cfg(tmp_path, cost_per_input_mtok=15.0, cost_per_output_mtok=75.0)
    backend = _FakeBackend({"What is 2+2?": ["\\boxed{4}"] * 4})
    records = [{"uid": "u1", "source": "rm", "statement": "What is 2+2?", "truth": "4"}]

    outcome = run(cfg=cfg, records=records, backend=backend)

    usage = json.loads(outcome.manifest_path.read_text())["token_usage"]
    assert usage["input_tokens"] == 40   # 4 calls x 10
    assert usage["output_tokens"] == 20  # 4 calls x 5
    cost = usage["estimated_cost"]
    assert cost["is_estimate"] is True
    assert cost["input_usd"] == round(40 / 1_000_000 * 15.0, 6)
    assert cost["output_usd"] == round(20 / 1_000_000 * 75.0, 6)
    assert cost["total_usd"] == round(cost["input_usd"] + cost["output_usd"], 6)
    assert cost["rates_per_mtok"] == {"input_usd": 15.0, "output_usd": 75.0}
    assert outcome.token_usage == usage


def test_retry_backoff_then_success(tmp_path):
    class _FlakyThenOk:
        name = "fake"

        def __init__(self):
            self.calls = 0

        def call(self, question, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("transient")
            return ["\\boxed{4}"]

    cfg = _cfg(tmp_path, k=1)  # defaults: max_retries=3, base 2.0, cap 30.0
    backend = _FlakyThenOk()
    delays = []

    outcome = run(
        cfg=cfg,
        records=[{"uid": "u1", "source": "rm", "statement": "Q", "truth": "4"}],
        backend=backend,
        sleep_fn=delays.append,
    )

    assert delays == [2.0, 4.0]     # base * 2**attempt, capped
    assert backend.calls == 3        # two failures + one success
    assert outcome.model_calls == 1  # only paid successes count
    (row,) = _rows(outcome.output_path)
    assert row["label"] == LABEL_SOLVED


def test_exhausted_retries_degenerate_uncached_and_rebilled_next_run(tmp_path):
    class _ScriptedBackend:
        """Each script element is a string (success) or an Exception (raise)."""

        name = "fake"

        def __init__(self, script):
            self._script = list(script)
            self.paid = 0

        def call(self, question, **kwargs):
            action = self._script.pop(0)
            if isinstance(action, Exception):
                raise action
            self.paid += 1
            return [action]

    cfg = _cfg(tmp_path, max_retries=0)  # single attempt per sample
    records = [{"uid": "u1", "source": "rm", "statement": "Q", "truth": "4"}]

    # Run 1: sample 2 fails permanently -> degenerate, run still completes.
    backend1 = _ScriptedBackend(
        ["\\boxed{4}", "\\boxed{4}", RuntimeError("backend down"), "\\boxed{4}"]
    )
    outcome1 = run(cfg=cfg, records=records, backend=backend1)
    assert outcome1.interrupted is False
    (row1,) = _rows(outcome1.output_path)
    assert row1["n_degenerate"] == 1 and row1["n_correct"] == 3
    assert row1["pass_at_k"] == 0.75
    assert row1["label"] == LABEL_BAND  # 0.75 == BAND_HI, still in band

    progress = cfg.output_dir / "_progress"
    # The failed sample was NOT cached; the error text is in the audit trail.
    cache_rows = _rows(progress / "llm_cache.jsonl")
    assert len(cache_rows) == 3
    audit = _rows(progress / "rollouts.jsonl")
    assert any(r["output"].startswith("[backend_error]") for r in audit)
    # The record was NOT committed: a backend error must not freeze a bad
    # score (the file does not even exist — nothing was ever committed).
    records_done = progress / "records_done.jsonl"
    assert not records_done.exists() or not _rows(records_done)

    # Run 2 (healthy backend, same output dir): re-bills exactly the failed sample.
    backend2 = _ScriptedBackend(["\\boxed{4}"])
    outcome2 = run(cfg=cfg, records=records, backend=backend2)
    assert backend2.paid == 1
    assert outcome2.model_calls == 1
    assert outcome2.resumed_records == 0  # re-scored, not served from the store
    (row2,) = _rows(outcome2.output_path)
    assert row2["pass_at_k"] == 1.0 and row2["label"] == LABEL_SOLVED
    assert backend1.paid + backend2.paid == cfg.k  # each sample billed once, ever


def test_pre_labeled_record_passes_through_verbatim(tmp_path):
    cfg = _cfg(tmp_path)
    record = {
        "uid": "u-mb", "source": "modelbreaker", "statement": "Q", "truth": "4",
        "pass_at_k": 0.5, "label": "band", "n_correct": 4, "n_wrong": 4,
        "n_degenerate": 0, "modal_wrong": "7", "top_wrong_share": 0.5,
    }

    outcome = run(cfg=cfg, records=[record], backend=_ExplodingBackend())

    (row,) = _rows(outcome.output_path)
    assert row == record  # identical row out, label kept as-is
    assert outcome.counts["pre_labeled"] == 1
    assert outcome.model_calls == 0


def test_pre_labeled_without_label_gets_one_derived(tmp_path):
    cfg = _cfg(tmp_path)
    record = {"uid": "u-mb2", "source": "mb", "statement": "Q", "truth": "4",
              "pass_at_k": 1.0}

    outcome = run(cfg=cfg, records=[record], backend=_ExplodingBackend())

    (row,) = _rows(outcome.output_path)
    assert row["label"] == LABEL_SOLVED  # derived only because it was missing
    assert row["pass_at_k"] == 1.0
    assert outcome.counts["pre_labeled"] == 1
    assert outcome.model_calls == 0


def test_garbage_and_unverifiable_truths_drop_without_backend(tmp_path):
    cfg = _cfg(tmp_path)
    records = [
        {"uid": "u-junk", "source": "rm", "statement": "Q1",
         "truth": "\\mathbb{R}^n"},                       # junk macro
        {"uid": "u-prose", "source": "rm", "statement": "Q2",
         "truth": "the function is continuous"},          # prose tier
        {"uid": "u-empty", "source": "rm", "statement": "Q3", "truth": "   "},
    ]

    outcome = run(cfg=cfg, records=records, backend=_ExplodingBackend())

    rows = _rows(outcome.output_path)
    by_uid = {r["uid"]: r for r in rows}
    assert by_uid["u-junk"]["label"] == LABEL_DROP
    assert by_uid["u-junk"]["drop_reason"] == DROP_GARBAGE_TRUTH
    assert by_uid["u-prose"]["label"] == LABEL_DROP
    assert by_uid["u-prose"]["drop_reason"] == DROP_UNVERIFIABLE
    assert by_uid["u-empty"]["drop_reason"] == DROP_GARBAGE_TRUTH
    assert all(r["pass_at_k"] is None for r in rows)
    assert all(r["rollout_uids"] == [] for r in rows)
    assert outcome.counts["dropped"] == 3
    assert outcome.counts[LABEL_DROP] == 3
    assert outcome.model_calls == 0


def test_degenerate_dominated_record_is_dropped(tmp_path):
    cfg = _cfg(tmp_path)
    backend = _FakeBackend({
        "Q": ["\\boxed{4}", "I cannot solve this", "no idea", "\\boxed{4}"],
    })
    records = [{"uid": "u-deg", "source": "rm", "statement": "Q", "truth": "4"}]

    outcome = run(cfg=cfg, records=records, backend=backend)

    (row,) = _rows(outcome.output_path)
    assert row["n_degenerate"] == 2  # 2/4 >= DEGENERATE_DROP_FRACTION
    assert row["label"] == LABEL_DROP
    assert row["drop_reason"] == DROP_DEGENERATE
    assert row["pass_at_k"] == 0.5   # still recorded for the audit trail
    assert outcome.counts["dropped"] == 1

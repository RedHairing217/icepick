"""Pass@k stage runner — top-level orchestration.

Flow (mirrors the groundtruth runner):

  1. Validate config, inject uids, write ``pass_at_k_input.jsonl``.
  2. flow_testing → replay the calibration sheet (zero model calls, zero
     checkpoint), still writing the same outputs + manifest.
  3. production → build the backend (or take the injected fake), open a
     :class:`PassAtKCheckpoint` under ``<output_dir>/_progress/`` and
     score records — up to ``max_concurrent`` records in parallel; the k
     rollouts *within* a record stay sequential so per-sample cache keys
     replay deterministically.
  4. Write ``pass_at_k.jsonl`` (input order, finished records only) and
     ``pass_at_k_manifest.json``.

Restartability contract, same spirit as the scraper: pause/restart
acceptable, full kill unacceptable. Every paid model output is cached
per sample and every finished record is committed as it happens, so a
Ctrl-C loses at most the in-flight call; re-running the same command
resumes, serving finished records from the store and already-paid
rollouts from the cache. Outputs are regenerated wholesale from
in-memory results on every run — never appended across runs — so a
resumed run produces a clean ``pass_at_k.jsonl`` with zero duplicate
uids.

Two deliberate wrinkles worth knowing about:

  * A backend failure that survives the retry loop becomes a
    *degenerate* rollout (``raw_output`` = ``[backend_error] ...``). The
    run still completes, but nothing is cached for the failed sample and
    the record is NOT committed to ``records_done`` — transient outages
    should not freeze a bad score into the store. The next run re-scores
    that record, re-billing exactly the samples that failed (the
    successful ones replay from ``llm_cache`` for free).
  * Records arriving with ``pass_at_k`` already set pass through
    verbatim and never touch the backend (spec non-goal: no relabeling
    of ModelBreaker's records). Their label is derived only when it is
    missing entirely.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from icepick.processing.pass_at_k.backends import build_backend
from icepick.processing.pass_at_k.base import (
    DEGENERATE_DROP_FRACTION,
    DROP_DEGENERATE,
    DROP_GARBAGE_TRUTH,
    DROP_UNVERIFIABLE,
    LABEL_DROP,
    LABEL_VALUES,
    ROLLOUT_CORRECT,
    ROLLOUT_DEGENERATE,
    ROLLOUT_WRONG,
    PassAtKRecord,
    RolloutResult,
    rollout_uid,
)
from icepick.processing.pass_at_k.calibration_replay import replay
from icepick.processing.pass_at_k.checkpoint import PassAtKCheckpoint, rollout_key
from icepick.processing.pass_at_k.config import PassAtKConfig
from icepick.processing.pass_at_k.scoring import (
    derive_label,
    extract_candidate,
    strip_think,
    truth_garbage,
    tally_rollouts,
)
from icepick.processing.pass_at_k.verifier import classify, verify
from icepick.processing.poser.base import inject_uid

# Tiers verify() can actually score; everything else is dropped up front.
VERIFIABLE_TIERS = ("number", "tuple", "expr")

# Field fallback chains for heterogeneous handoff shapes (MB harvest rows
# say question/answer; icepick rows say statement/truth).
STATEMENT_FIELDS = ("statement", "question", "prompt", "problem")
TRUTH_FIELDS = ("truth", "answer")

# How a record was produced, for counts/resume bookkeeping.
_KIND_SCORED = "scored"
_KIND_PRE_LABELED = "pre_labeled"
_KIND_RESUMED = "resumed"


@dataclass
class PassAtKOutcome:
    manifest_path: Path
    output_path: Path
    counts: dict
    interrupted: bool = False
    resumed_records: int = 0
    model_calls: int = 0
    token_usage: dict = field(default_factory=dict)


def run(
    *,
    cfg: PassAtKConfig,
    records: Iterable[dict],
    backend=None,
    sleep_fn=time.sleep,
) -> PassAtKOutcome:
    """End-to-end pass@k scoring. Returns paths and counts.

    ``backend`` injection lets tests substitute a fake; leave it None and
    the runner builds the backend for ``cfg`` (kill switch included).
    ``sleep_fn`` injection makes retry backoff instant in tests.
    """
    cfg.validate()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = inject_uid(list(records))
    input_path = output_dir / "pass_at_k_input.jsonl"
    with input_path.open("w", encoding="utf-8") as fh:
        for record in prepared:
            fh.write(json.dumps(record) + "\n")

    if cfg.mode == "flow_testing":
        return _run_replay(cfg, prepared, output_dir=output_dir, input_path=input_path)

    backend = backend or build_backend(cfg)
    checkpoint = PassAtKCheckpoint(output_dir / "_progress")
    checkpoint.begin()

    tally = _SharedTally()
    results: dict = {}  # input index -> (stamped_row, kind)
    interrupted = False

    if cfg.max_concurrent == 1:
        # Plain loop: deterministic call order for tests and local runs.
        try:
            for idx, record in enumerate(prepared):
                results[idx] = _score_record(
                    record, cfg=cfg, backend=backend, checkpoint=checkpoint,
                    sleep_fn=sleep_fn, tally=tally,
                )
        except KeyboardInterrupt:
            interrupted = True
    else:
        executor = ThreadPoolExecutor(max_workers=cfg.max_concurrent)
        future_to_idx = {
            executor.submit(
                _score_record, record, cfg=cfg, backend=backend,
                checkpoint=checkpoint, sleep_fn=sleep_fn, tally=tally,
            ): idx
            for idx, record in enumerate(prepared)
        }
        try:
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
        except KeyboardInterrupt:
            interrupted = True
            for future in future_to_idx:
                future.cancel()  # pending work is abandoned; running work drains
        finally:
            executor.shutdown(wait=True)
        if interrupted:
            # Keep anything that finished while the pool was draining —
            # those records are committed on disk, so the output should
            # agree with the checkpoint.
            for future, idx in future_to_idx.items():
                if idx in results or future.cancelled() or not future.done():
                    continue
                try:
                    if future.exception() is None:
                        results[idx] = future.result()
                except BaseException:
                    continue

    if not interrupted:
        checkpoint.mark_complete()

    counts = _new_counts()
    resumed = 0
    rows = []
    for idx in sorted(results):
        row, kind = results[idx]
        label = row.get("label")
        counts[label] = counts.get(label, 0) + 1
        if kind == _KIND_PRE_LABELED:
            counts["pre_labeled"] += 1
        if kind == _KIND_RESUMED:
            resumed += 1
        if label == LABEL_DROP:
            counts["dropped"] += 1
        rows.append(row)

    warnings = []
    if interrupted:
        warnings.append(
            "run interrupted; finished records were kept — re-run the same "
            "command to resume without re-billing"
        )

    return _emit(
        cfg,
        output_dir=output_dir,
        input_path=input_path,
        record_count=len(prepared),
        rows=rows,
        counts=counts,
        model_calls=tally.model_calls,
        resumed_records=resumed,
        interrupted=interrupted,
        warnings=warnings,
        token_usage=_token_usage(backend, cfg),
        calibration_replay=False,
    )


# --- flow_testing ------------------------------------------------------------


def _run_replay(cfg: PassAtKConfig, prepared: list, *, output_dir: Path, input_path: Path) -> PassAtKOutcome:
    """Replay the calibration sheet: no backend, no checkpoint, no retries."""
    rows = replay(cfg, prepared)
    counts = _new_counts()
    for row in rows:
        label = row.get("label")
        counts[label] = counts.get(label, 0) + 1
        if label == LABEL_DROP:
            counts["dropped"] += 1
    return _emit(
        cfg,
        output_dir=output_dir,
        input_path=input_path,
        record_count=len(prepared),
        rows=rows,
        counts=counts,
        model_calls=0,
        resumed_records=0,
        interrupted=False,
        warnings=[],
        token_usage={},
        calibration_replay=True,
    )


# --- per-record scoring ------------------------------------------------------


def _score_record(record: dict, *, cfg: PassAtKConfig, backend, checkpoint, sleep_fn, tally):
    """Score one record. Returns ``(stamped_row, kind)``.

    Thread-safe: the checkpoint locks internally, the tally locks its
    counter, and everything else is local to the record.
    """
    uid = record["uid"]

    # (a) Already measured upstream (e.g. ModelBreaker's 70-record set):
    # passthrough verbatim; derive the label only when missing entirely.
    if record.get("pass_at_k") is not None:
        row = dict(record)
        if row.get("label") is None:
            row["label"] = derive_label(
                row["pass_at_k"], row.get("top_wrong_share") or 0.0
            )
        return row, _KIND_PRE_LABELED

    # (b) Committed by a previous (interrupted) run: serve from the store.
    stored = checkpoint.stored_record(uid)
    if stored is not None:
        return stored, _KIND_RESUMED

    # (c) Pre-filters — no backend call for records we cannot score.
    statement = _statement_of(record)
    truth = _truth_of(record)
    if truth is None or (truth_garbage(truth) and not cfg.keep_garbage):
        return _drop(record, DROP_GARBAGE_TRUTH, checkpoint), _KIND_SCORED
    tier, truth_obj = classify(truth)
    if tier not in VERIFIABLE_TIERS:
        return _drop(record, DROP_UNVERIFIABLE, checkpoint), _KIND_SCORED

    # (d)+(e) k sequential rollouts, each cached per sample.
    verdicts: list = []
    candidates: list = []
    backend_errors = 0
    for idx in range(cfg.k):
        key = rollout_key(
            cfg.resolved_model, statement, cfg.temperature, cfg.think, idx
        )
        cached = checkpoint.cached_output(key)  # "" is a hit; None is a miss
        from_cache = cached is not None
        failed = False
        if from_cache:
            raw = cached
        else:
            raw, failed = _call_with_retries(
                backend, statement, cfg=cfg, sleep_fn=sleep_fn
            )
            if failed:
                backend_errors += 1  # nothing cached: the next run retries it
            else:
                checkpoint.store_output(key, raw)
                tally.paid()
        if failed:
            candidate, verdict = None, ROLLOUT_DEGENERATE
        else:
            candidate = extract_candidate(strip_think(raw))
            if candidate is None:
                verdict = ROLLOUT_DEGENERATE
            elif verify(candidate, truth_obj, tier):
                verdict = ROLLOUT_CORRECT
            else:
                verdict = ROLLOUT_WRONG
        result = RolloutResult(
            rollout_uid=rollout_uid(uid, idx),
            sample_idx=idx,
            raw_output=raw,
            candidate=candidate,
            verdict=verdict,
            from_cache=from_cache,
        )
        checkpoint.append_rollout(result.to_jsonl_row(uid))
        verdicts.append(verdict)
        candidates.append(candidate)

    # (f) Tally and label.
    tallied = tally_rollouts(verdicts, candidates)
    pass_at_k = tallied["n_correct"] / cfg.k
    if tallied["n_degenerate"] / cfg.k >= DEGENERATE_DROP_FRACTION:
        # pass_at_k stays set for the audit trail even though we drop.
        label, drop_reason = LABEL_DROP, DROP_DEGENERATE
    else:
        label = derive_label(pass_at_k, tallied["top_wrong_share"])
        drop_reason = None
    rec = PassAtKRecord(
        uid=uid,
        source=record.get("source", ""),
        pass_at_k=pass_at_k,
        n_correct=tallied["n_correct"],
        n_wrong=tallied["n_wrong"],
        n_degenerate=tallied["n_degenerate"],
        label=label,
        modal_wrong=tallied["modal_wrong"],
        top_wrong_share=tallied["top_wrong_share"],
        rollout_uids=[rollout_uid(uid, i) for i in range(cfg.k)],
        drop_reason=drop_reason,
    )
    stamped = rec.stamp(record)
    if backend_errors == 0:
        checkpoint.commit_record(uid, stamped)
    return stamped, _KIND_SCORED


def _call_with_retries(backend, statement: str, *, cfg: PassAtKConfig, sleep_fn):
    """One rollout's paid call. Returns ``(raw_output, failed)``.

    Retries transient ``Exception``s with capped exponential backoff.
    ``KeyboardInterrupt`` is deliberately NOT caught here — a Ctrl-C
    mid-call must pause the run, not burn a retry.
    """
    last_exc = None
    for attempt in range(cfg.max_retries + 1):
        try:
            outputs = backend.call(
                statement,
                k=1,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                think=cfg.think,
                timeout=cfg.request_timeout_s,
            )
            return outputs[0], False
        except Exception as exc:
            last_exc = exc
            if attempt < cfg.max_retries:
                sleep_fn(
                    min(cfg.retry_max_delay, cfg.retry_base_delay * (2 ** attempt))
                )
    return f"[backend_error] {last_exc}", True


def _drop(record: dict, reason: str, checkpoint) -> dict:
    """A pre-filter drop: no rollouts, zero tallies, committed like any record."""
    rec = PassAtKRecord(
        uid=record["uid"],
        source=record.get("source", ""),
        pass_at_k=None,
        n_correct=0,
        n_wrong=0,
        n_degenerate=0,
        label=LABEL_DROP,
        modal_wrong=None,
        top_wrong_share=0.0,
        rollout_uids=[],
        drop_reason=reason,
    )
    stamped = rec.stamp(record)
    checkpoint.commit_record(record["uid"], stamped)
    return stamped


# --- helpers -----------------------------------------------------------------


class _SharedTally:
    """Thread-safe paid-call counter; accurate even across an interrupt."""

    def __init__(self):
        self._lock = threading.Lock()
        self.model_calls = 0

    def paid(self) -> None:
        with self._lock:
            self.model_calls += 1


def _statement_of(record: dict) -> str:
    for field_name in STATEMENT_FIELDS:
        value = record.get(field_name)
        if value:
            return str(value)
    return ""


def _truth_of(record: dict):
    for field_name in TRUTH_FIELDS:
        value = record.get(field_name)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _new_counts() -> dict:
    counts = {label: 0 for label in LABEL_VALUES}
    counts["pre_labeled"] = 0
    counts["dropped"] = 0
    return counts


def _token_usage(backend, cfg: PassAtKConfig) -> dict:
    """Backend-reported tokens, plus a clearly-marked cost estimate.

    Backends expose ``usage()`` voluntarily (the protocol doesn't demand
    it), so we probe with ``getattr``. Shape mirrors the groundtruth
    manifest: estimates carry ``is_estimate: true`` and echo the rates
    so dashboards can flag them as approximate rather than measured.
    """
    usage_fn = getattr(backend, "usage", None)
    usage = dict(usage_fn()) if callable(usage_fn) else {}
    if cfg.cost_per_input_mtok is None and cfg.cost_per_output_mtok is None:
        return usage
    in_rate = cfg.cost_per_input_mtok or 0.0
    out_rate = cfg.cost_per_output_mtok or 0.0
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    usage["estimated_cost"] = {
        "input_usd": round(input_tokens / 1_000_000 * in_rate, 6),
        "output_usd": round(output_tokens / 1_000_000 * out_rate, 6),
        "total_usd": round(
            (input_tokens / 1_000_000 * in_rate)
            + (output_tokens / 1_000_000 * out_rate),
            6,
        ),
        "is_estimate": True,
        "rates_per_mtok": {"input_usd": in_rate, "output_usd": out_rate},
    }
    return usage


def _emit(
    cfg: PassAtKConfig,
    *,
    output_dir: Path,
    input_path: Path,
    record_count: int,
    rows: list,
    counts: dict,
    model_calls: int,
    resumed_records: int,
    interrupted: bool,
    warnings: list,
    token_usage: dict,
    calibration_replay: bool,
) -> PassAtKOutcome:
    """Write pass_at_k.jsonl + manifest from in-memory rows (never append)."""
    output_path = output_dir / "pass_at_k.jsonl"
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    manifest = {
        "stage": "pass_at_k",
        "config": cfg.echo(),
        "calibration_replay": calibration_replay,
        "inputs": {
            "pass_at_k_input": str(input_path),
            "record_count": record_count,
        },
        "outputs": {"pass_at_k": str(output_path)},
        "counts": counts,
        "model_calls": model_calls,
        "resumed_records": resumed_records,
        "interrupted": interrupted,
        "warnings": warnings,
        "token_usage": token_usage,
    }
    manifest_path = output_dir / "pass_at_k_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return PassAtKOutcome(
        manifest_path=manifest_path,
        output_path=output_path,
        counts=counts,
        interrupted=interrupted,
        resumed_records=resumed_records,
        model_calls=model_calls,
        token_usage=token_usage,
    )

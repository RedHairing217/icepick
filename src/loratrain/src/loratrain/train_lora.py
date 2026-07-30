"""Thin client to the operator's remote training server (W3).

The server address comes EXCLUSIVELY from ``config.TRAIN_SERVER_URL`` --
this module never spells out a host, port, or scheme itself; it only
concatenates ``config.TRAIN_SERVER_URL`` with a path fragment (see
``ENDPOINT_TRAIN`` / ``ENDPOINT_STATUS`` below). Since the SSH-tunnel-only
decision (RUNBOOK D-R1, revised 2026-07-25) that URL is tunnel-local: the
box binds its status server to loopback and the operator reaches it via
the section 6 SSH local-forward (``tunnel.py``), so requests here only
work while that tunnel is up. To retarget training at a different box,
the operator edits ``config.TRAIN_SERVER_IP`` (the ssh/scp target) --
nothing in this file changes.

W3 stub: ``assert_baseline_captured`` and ``build_job_payload`` are REAL
and independently tested. ``submit_job`` and ``main`` validate their
guards and then refuse -- no HTTP client is imported or wired up yet;
that lands when W3 is unblocked by operator approval (remote source-
weight fetch + dataset upload to the operator's own box, README D1/D2).
"""

from __future__ import annotations

from pathlib import Path

from loratrain import config
from loratrain.build_dataset import sha256_file

# Path fragments only -- no hosts, no scheme. Concatenated onto
# config.TRAIN_SERVER_URL at call time (see submit_job).
ENDPOINT_TRAIN = "/train"
ENDPOINT_STATUS = "/status"


class OrderingError(RuntimeError):
    """Training was attempted out of order (e.g. before the baseline exists).

    Enforces README invariant #1, "Baseline before training": training
    must never run before a non-empty ``baseline_greedy.jsonl`` exists.
    """


def assert_baseline_captured(baseline_path: Path) -> str:
    """Hard-fail unless the greedy baseline has already been captured.

    ``baseline_path`` is expected to be ``config.BASELINE_GREEDY_PATH``,
    i.e. the exact ``baseline_greedy.jsonl`` filename
    ``evalharness/run_eval.py`` writes. Returns the file's sha256 (via
    ``build_dataset.sha256_file``) so callers can embed it in the job
    payload / run manifest.
    """
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        raise OrderingError(
            f"baseline not found at {baseline_path}. Capture the greedy "
            "baseline FIRST via evalharness-run (README 'Exact train->serve "
            "recipe', step 1) -- training refuses to run before a baseline "
            "exists."
        )
    if not baseline_path.read_text(encoding="utf-8").strip():
        raise OrderingError(
            f"baseline at {baseline_path} is empty (0 bytes or whitespace "
            "only). Capture the greedy baseline FIRST via evalharness-run "
            "(README 'Exact train->serve recipe', step 1) -- training "
            "refuses to run before a real baseline exists."
        )
    return sha256_file(baseline_path)


def build_job_payload(dataset_path: Path, baseline_sha256: str) -> dict:
    """Pure builder for the remote-training job payload. No I/O, no network."""
    return {
        "base_model": config.BASE_MODEL_HF_ID,
        "adapter_format": config.ADAPTER_FORMAT,
        "dataset_file": str(dataset_path),
        "baseline_greedy_sha256": baseline_sha256,
        "seed": config.SEED,
        "hyperparams": {
            "rank": config.LORA_RANK,
            "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT,
            "lr": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "micro_batch_size": config.MICRO_BATCH_SIZE,
            "max_seq_len": config.MAX_SEQ_LEN,
            # v2 pins (2026-07-29) -- see upload_guard.write_run_config.
            "grad_accum_steps": config.GRAD_ACCUM_STEPS,
            "lr_scheduler_type": config.LR_SCHEDULER_TYPE,
            "warmup_ratio": config.WARMUP_RATIO,
            "weight_decay": config.WEIGHT_DECAY,
        },
        "weight_policy": config.WEIGHT_POLICY,
        "weight_policy_label": config.weight_policy_label(),
        "dataset_schema": "prompt_completion.v2",
        "completion_only_loss": True,
    }


def submit_job(payload: dict):
    """Would POST ``payload`` to the operator's training server -- not wired yet.

    Deliberately does not import an HTTP client at module scope: this
    stub only assembles the URL (proving it derives from
    ``config.TRAIN_SERVER_URL``, the single source of truth) and then
    refuses. Live-call machinery lands with W3, gated on operator
    approval.
    """
    url = config.TRAIN_SERVER_URL + ENDPOINT_TRAIN
    raise NotImplementedError(
        f"W3 — remote training is gated on operator approval. Would POST to {url}."
    )


def main(argv=None) -> int:
    config.validate_config()
    assert_baseline_captured(config.BASELINE_GREEDY_PATH)
    raise NotImplementedError(
        "W3 — gated: requires operator approval (remote source-weight fetch "
        "+ dataset upload to the operator's training server)."
    )


if __name__ == "__main__":
    raise SystemExit(main())

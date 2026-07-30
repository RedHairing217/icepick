"""The RUNBOOK section 5 guarded uploader -- the ONE permitted data upload.

There is no arbitrary-file argument here on purpose: the payload is fixed by
``config.py`` (the dataset it built in W2, plus two receipts this module
generates itself), never something a caller can point at an arbitrary path.
Before any bytes move, this module independently re-runs the same guards
``build_dataset.py`` enforces at harvest time (leakage, verified-correctness,
dedupe) plus three upload-specific checks: the identity preflight receipt
(RUNBOOK section 0.3) must already be a PASS, the target must not still be the
loopback placeholder, and no blocklisted filename may ride along.

Every host/port value comes from ``config`` (the SSH port from
``config.TRAIN_SERVER_SSH_PORT`` -- RUNBOOK Appendix A, applied 2026-07-25 --
with the ``TRAIN_SSH_PORT`` env var as fallback); this module never spells
out an address itself (see ``config.py`` for the single-source-of-truth rule
this package-wide scan enforces).
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from loratrain import build_dataset, config, verify_base_identity

BLOCKLIST_PATTERNS = (
    "eval_set",
    "holdout",
    "eval_paper_split",
    "band_corpus",
    "baseline_greedy",
)


class UploadRefused(RuntimeError):
    """The upload was refused by one of this module's guards.

    Deliberately NOT raised for leakage: ``build_dataset.LeakageError``
    propagates as itself (see ``validate_dataset``) so a leakage hit is
    never mistaken for an ordinary configuration refusal.
    """




def _check_manifest_corpus_sha(manifest_path) -> None:
    """W2-manifest corpus-sha check (extracted 2026-07-26, schema-drift fix).

    W2's ``build_manifest`` nests the corpus record ({"corpus": {"sha256":
    ...}}); the historical flat ``corpus_sha256`` key is accepted as a
    fallback. Missing entirely and mismatching are distinct refusals.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise UploadRefused(f"{manifest_path} not found -- W2 manifest missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UploadRefused(f"{manifest_path}: invalid JSON ({exc})") from exc
    sha = manifest.get("corpus_sha256") or (manifest.get("corpus") or {}).get("sha256")
    if sha is None:
        raise UploadRefused(
            f"{manifest_path}: no corpus sha recorded (neither 'corpus_sha256' nor 'corpus.sha256') -- W2 manifest malformed."
        )
    if sha != config.EXPECTED_CORPUS_SHA256:
        raise UploadRefused(
            f"{manifest_path}: manifest corpus sha {sha[:16]} != pinned {config.EXPECTED_CORPUS_SHA256[:16]} -- corpus moved since W2 build."
        )

def resolve_ssh_port() -> int:
    """Resolve the pod's external SSH port from config or the environment.

    ``config.TRAIN_SERVER_SSH_PORT`` is the RUNBOOK Appendix A field
    (applied 2026-07-25; per-pod value set at section 1.3);
    ``TRAIN_SSH_PORT`` remains the env fallback for states where the
    attribute is absent. Either source is accepted; config wins if both
    are set.
    """
    raw = getattr(config, "TRAIN_SERVER_SSH_PORT", None)
    if raw is None:
        raw = os.environ.get("TRAIN_SSH_PORT")
    if raw is None:
        raise UploadRefused(
            "no SSH port configured -- set config.TRAIN_SERVER_SSH_PORT (RUNBOOK "
            "Appendix A) or export TRAIN_SSH_PORT (RUNBOOK section 1.3) before uploading."
        )
    try:
        port = int(raw)
    except (TypeError, ValueError):
        raise UploadRefused(
            f"SSH port {raw!r} is not an integer (RUNBOOK section 1.3 / Appendix A)."
        ) from None
    if not (1 <= port <= 65535):
        raise UploadRefused(
            f"SSH port {port} is out of range [1, 65535] (RUNBOOK section 1.3 / Appendix A)."
        )
    return port


def check_target() -> None:
    """Refuse while ``config.TRAIN_SERVER_IP`` is still the loopback placeholder.

    Shared by this uploader and ``tunnel.py`` (both drive ssh at the pod).
    A hostname that ``ipaddress.ip_address`` cannot parse is treated as
    acceptable (it is presumably a real pod hostname, not the shipped
    placeholder) -- only a literal loopback address is refused.
    """
    try:
        addr = ipaddress.ip_address(config.TRAIN_SERVER_IP)
    except ValueError:
        return  # a hostname, not an IP literal -- acceptable
    if addr.is_loopback:
        raise UploadRefused(
            "config.TRAIN_SERVER_IP is still the loopback placeholder -- set it to "
            "the rented pod's real address per RUNBOOK section 1.3 first."
        )


def check_blocklist(paths) -> None:
    """Refuse if any path's basename matches a blocklisted (case-insensitive) pattern.

    These are the filenames RUNBOOK section 5 names as radioactive to the
    remote box: eval set / holdout / eval-paper-split / band corpus /
    baseline-greedy artifacts. This check runs over the actual upload
    payload, independent of ``validate_dataset``'s content checks.
    """
    for path in paths:
        name = Path(path).name.lower()
        for pattern in BLOCKLIST_PATTERNS:
            if pattern in name:
                raise UploadRefused(
                    f"refusing to upload {path}: basename matches blocklisted pattern "
                    f"{pattern!r} (RUNBOOK section 5)."
                )


def validate_dataset() -> dict:
    """Re-run every W2 guard against the built dataset, independently, at upload time.

    Returns ``{"sha256": ..., "rows": ...}`` on success. Every failure mode
    raises ``UploadRefused`` EXCEPT leakage, which raises
    ``build_dataset.LeakageError`` directly (never caught/wrapped here) so it
    is never mistaken for an ordinary configuration problem.
    """
    dataset_path = Path(config.SFT_DATASET_PATH)
    if not dataset_path.exists():
        raise UploadRefused(f"{dataset_path} does not exist -- run W2 first (loratrain-build-dataset).")

    rows = []
    with dataset_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise UploadRefused(f"{dataset_path}:{lineno}: invalid JSON ({exc})") from exc

    for rownum, row in enumerate(rows, start=1):
        try:
            build_dataset.assert_verified_correct(row)
        except build_dataset.TraceIntegrityError as exc:
            raise UploadRefused(f"{dataset_path} row {rownum}: {exc}") from exc

    deduped = build_dataset.dedupe_examples(rows)
    if len(deduped) != len(rows):
        raise UploadRefused(
            f"{dataset_path} contains (uid, rollout_uid) duplicates "
            f"({len(rows)} rows, {len(deduped)} unique) -- dedupe before upload."
        )

    manifest_path = Path(config.DATASET_MANIFEST_PATH)
    _check_manifest_corpus_sha(manifest_path)

    eval_papers = build_dataset.load_eval_papers(config.EVAL_PAPER_SPLIT_PATH, config.EXPECTED_SPLIT_SHA256_16)

    eval_set_path = Path(config.EVAL_SET_PATH)
    if not eval_set_path.exists():
        raise UploadRefused("eval_set.jsonl not found -- cannot prove non-leakage; run evalharness-build-set first.")
    eval_uids = set()
    with eval_set_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            uid = json.loads(line).get("uid")
            if uid is not None:
                eval_uids.add(uid)

    build_dataset.assert_no_leakage(rows, eval_papers, eval_uids)  # LeakageError propagates as itself

    return {"sha256": build_dataset.sha256_file(dataset_path), "rows": len(rows)}


def write_run_config(path) -> None:
    """Write ``run_config.json``: every hyperparameter/seed/pin read live from config.

    The box never hardcodes a parameter -- this is the one file that
    carries them across the wire.
    """
    payload = {
        # Explicit list from config (v2, 2026-07-30). Was
        # [SEED, SEED+1, SEED+2] -- capped the campaign at 3 seeds and broke
        # across month boundaries; see config.SEEDS for the pinned control set.
        "seeds": list(config.SEEDS),
        "hyperparams": {
            "rank": config.LORA_RANK,
            "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT,
            "lr": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "micro_batch_size": config.MICRO_BATCH_SIZE,
            "max_seq_len": config.MAX_SEQ_LEN,
            # v2 pins (2026-07-29): explicit and recorded. In v1 these were a
            # hardcoded literal (grad-accum) + silent SFTConfig defaults
            # (scheduler/warmup/weight-decay), invisible to every manifest.
            "grad_accum_steps": config.GRAD_ACCUM_STEPS,
            "lr_scheduler_type": config.LR_SCHEDULER_TYPE,
            "warmup_ratio": config.WARMUP_RATIO,
            "weight_decay": config.WEIGHT_DECAY,
        },
        # Dataset v2 contract markers (2026-07-29): which weight policy this
        # run's dataset was built under, and that the box-side trainer must
        # run its prompt/completion completion-only-loss path.
        "weight_policy": config.WEIGHT_POLICY,
        "weight_policy_label": config.weight_policy_label(),
        "dataset_schema": "prompt_completion.v2",
        "completion_only_loss": True,
        "base_model": config.BASE_MODEL_HF_ID,
        "base_model_revision": verify_base_identity.FP16_REVISION,
        "adapter_format": config.ADAPTER_FORMAT,
        "llamacpp_tag": verify_base_identity.LLAMACPP_TAG,
        "serve_quant": config.SERVE_QUANT,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_receipt(path, dataset_info: dict, payload_shas: dict) -> None:
    """Write ``upload_receipt.json``: the sha the box re-checks before training."""
    payload = {
        "dataset_sha256": dataset_info["sha256"],
        "dataset_rows": dataset_info["rows"],
        "files": dict(payload_shas),
        "generated_unix": time.time(),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_scp_command(files, ssh_port) -> list:
    """Pure argv builder for the guarded scp -- no subprocess started here."""
    return ["scp", "-P", str(ssh_port)] + [str(f) for f in files] + [f"root@{config.TRAIN_SERVER_IP}:/workspace/run/"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="upload_guard")
    parser.add_argument("--execute", action="store_true", help="actually scp; default is dry-run (print the command only)")
    parser.add_argument(
        "--identity-receipt",
        default=None,
        help="path to the section 0.3 identity receipt (default: config.DATA_DIR/identity_receipt.json)",
    )
    args = parser.parse_args(argv)

    identity_receipt_path = (
        Path(args.identity_receipt) if args.identity_receipt else (config.DATA_DIR / "identity_receipt.json")
    )

    try:
        config.validate_config()
        check_target()
        ssh_port = resolve_ssh_port()

        if not identity_receipt_path.exists():
            raise UploadRefused(
                f"{identity_receipt_path} not found -- run verify_base_identity (RUNBOOK "
                "section 0.3) first; no upload without a passing identity preflight."
            )
        try:
            identity_receipt = json.loads(identity_receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UploadRefused(f"{identity_receipt_path}: invalid JSON ({exc})") from exc
        if identity_receipt.get("verdict") != "PASS":
            raise UploadRefused(
                f"{identity_receipt_path} verdict={identity_receipt.get('verdict')!r} -- "
                "identity preflight (RUNBOOK section 0.3) has not passed."
            )

        dataset_info = validate_dataset()

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        run_config_path = config.DATA_DIR / "run_config.json"
        upload_receipt_path = config.DATA_DIR / "upload_receipt.json"
        write_run_config(run_config_path)

        dataset_path = Path(config.SFT_DATASET_PATH)
        payload_shas = {
            dataset_path.name: dataset_info["sha256"],
            run_config_path.name: build_dataset.sha256_file(run_config_path),
        }
        write_receipt(upload_receipt_path, dataset_info, payload_shas)

        payload = [dataset_path, run_config_path, upload_receipt_path]
        check_blocklist(payload)

        cmd = build_scp_command(payload, ssh_port)

        if not args.execute:
            print(cmd)
            print("DRY RUN — nothing uploaded")
            return 0

        subprocess.run(cmd, check=True)
        print(f"uploaded {len(payload)} files; dataset sha256={dataset_info['sha256']} rows={dataset_info['rows']}")
        return 0

    except UploadRefused as exc:
        print(f"UPLOAD REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

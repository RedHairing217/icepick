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

Per-invocation overrides (additive, 2026-08-01, for running several boxes
concurrently without editing ``config.py``'s operator block once per box):
``resolve_server_ip``/``resolve_ssh_port`` accept an explicit override
(highest precedence), then a dedicated ``*_OVERRIDE`` environment variable,
then fall back to the existing config/``TRAIN_SSH_PORT`` chain unchanged.
``write_run_config`` similarly accepts a validated seeds SUBSET in place of
the full ``config.SEEDS`` cohort. None of this changes default behavior when
no override is supplied, and none of it hardcodes an address -- overrides
are always caller- or environment-supplied strings, never literals here.
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

def resolve_ssh_port(override: "int | str | None" = None) -> int:
    """Resolve the pod's external SSH port: override > config > env.

    ``config.TRAIN_SERVER_SSH_PORT`` is the RUNBOOK Appendix A field
    (applied 2026-07-25; per-pod value set at section 1.3);
    ``TRAIN_SSH_PORT`` remains the env fallback for states where the
    attribute is absent. Either source is accepted; config wins if both
    are set. This is the pre-existing, byte-unchanged behavior when
    ``override`` is omitted and ``TRAIN_SERVER_SSH_PORT_OVERRIDE`` is unset.

    ``override`` (e.g. a ``--server-ssh-port`` CLI flag) takes precedence
    over everything else when given; the ``TRAIN_SERVER_SSH_PORT_OVERRIDE``
    env var is the same override expressed without a flag. Both are
    additive (2026-08-01): they let one invocation target a specific box
    without touching ``config.py``'s operator block, for running several
    boxes concurrently. Neither is a literal address -- both are always
    caller- or environment-supplied at call time.
    """
    raw = override
    if raw is None:
        raw = os.environ.get("TRAIN_SERVER_SSH_PORT_OVERRIDE")
    if raw is None:
        raw = getattr(config, "TRAIN_SERVER_SSH_PORT", None)
    if raw is None:
        raw = os.environ.get("TRAIN_SSH_PORT")
    if raw is None:
        raise UploadRefused(
            "no SSH port configured -- pass --server-ssh-port, export "
            "TRAIN_SERVER_SSH_PORT_OVERRIDE, set config.TRAIN_SERVER_SSH_PORT (RUNBOOK "
            "Appendix A), or export TRAIN_SSH_PORT (RUNBOOK section 1.3) before uploading."
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


def resolve_server_ip(override: "str | None" = None) -> str:
    """Resolve the pod's public IP/hostname: override > env > config.

    ``override`` (e.g. a ``--server-ip`` CLI flag) takes precedence when
    given; then the ``TRAIN_SERVER_ADDRESS_OVERRIDE`` environment variable;
    then ``config.TRAIN_SERVER_IP`` unchanged -- the pre-existing,
    byte-unchanged behavior when neither override channel is used.

    Additive (2026-08-01), mirroring ``resolve_ssh_port``: lets one
    invocation target a specific box's address without editing
    ``config.py``'s operator block, for running several boxes concurrently.
    This function never spells out an address itself -- ``override`` is
    always caller- or environment-supplied (RUNBOOK single-source-of-truth
    rule; ``tests/test_config.py::test_single_source_of_truth_for_server_address``
    scans this file for IP/URL literals same as every other module here).
    """
    if override is not None:
        return override
    env_override = os.environ.get("TRAIN_SERVER_ADDRESS_OVERRIDE")
    if env_override:
        return env_override
    return config.TRAIN_SERVER_IP


def check_target(ip: "str | None" = None) -> None:
    """Refuse while the effective target is still the loopback placeholder.

    Shared by this uploader and ``tunnel.py`` (both drive ssh at the pod).
    A hostname that ``ipaddress.ip_address`` cannot parse is treated as
    acceptable (it is presumably a real pod hostname, not the shipped
    placeholder) -- only a literal loopback address is refused.

    ``ip``, if given, is the already-resolved effective address (e.g. from
    ``resolve_server_ip``) and is checked instead of
    ``config.TRAIN_SERVER_IP`` -- additive (2026-08-01): a per-invocation
    ``--server-ip`` override must be judged on ITS OWN value, not config's,
    so a stale loopback placeholder left in config.py never blocks a
    correctly-overridden run. Omitting ``ip`` reproduces the pre-existing
    behavior exactly (checks ``config.TRAIN_SERVER_IP``).
    """
    if ip is None:
        ip = config.TRAIN_SERVER_IP
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return  # a hostname, not an IP literal -- acceptable
    if addr.is_loopback:
        raise UploadRefused(
            "the effective target address is still the loopback placeholder -- set "
            "config.TRAIN_SERVER_IP per RUNBOOK section 1.3, or pass --server-ip / "
            "export TRAIN_SERVER_ADDRESS_OVERRIDE, to the pod's real address first."
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


def _row_is_v3_shaped(row) -> bool:
    """v3 rows are discriminated by ``provenance.verify_receipt`` -- a field
    the v1/v2 harvest never wrote (fail-closed: absence routes to the strict
    legacy check). See docs/SESSION_HANDOFF.md "AUTHORIZATION -- upload_guard
    accepts v3-shaped dataset provenance" (2026-08-01, commit 560c7ff)."""
    prov = row.get("provenance")
    return isinstance(prov, dict) and "verify_receipt" in prov


_HEX64 = frozenset("0123456789abcdef")


def _assert_v3_row(row, rownum: int, dataset_path) -> str:
    """Equal-strictness per-row checks for a v3-shaped row; returns its uid.
    Inline by design -- importing ``loratrain.v3`` here would break the
    isolation rule that no existing module imports v3.

    Fail-clean (2026-08-01; training-ops review of 72cfc39, see
    out/v3_full_run_20260801/opslog_train4x.md "fail-safe gap"): every
    access below is shape-guarded so a malformed row raises UploadRefused
    naming the row -- never a bare KeyError/IndexError/AttributeError/
    TypeError -- even when this function runs without
    ``_validate_v3_dataset``'s wellformedness pre-pass in front of it.
    The prompt guard checks only what this function itself reads
    (``prompt[1]["content"]``); full schema strictness (pinned system
    prompt, no-think suffix, completion shape) stays with
    ``build_dataset.assert_prompt_completion_wellformed``."""
    prov = row.get("provenance")
    if not isinstance(prov, dict):
        raise UploadRefused(
            f"{dataset_path} row {rownum}: v3 row provenance is not a dict "
            f"(got {type(prov).__name__})"
        )
    uid = prov.get("uid")
    if not uid or not isinstance(uid, str):
        raise UploadRefused(f"{dataset_path} row {rownum}: v3 row missing provenance.uid")
    receipt = prov.get("verify_receipt")
    if not isinstance(receipt, dict) or receipt.get("verified") is not True:
        raise UploadRefused(
            f"{dataset_path} row {rownum} (uid {uid}): verify_receipt.verified is not True -- "
            "only endpoint-verified regen traces may upload."
        )
    idx = prov.get("regen_sample_idx")
    if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
        raise UploadRefused(f"{dataset_path} row {rownum} (uid {uid}): bad regen_sample_idx {idx!r}")
    sha = prov.get("proof_raw_sha")
    if not (isinstance(sha, str) and len(sha) == 64 and set(sha.lower()) <= _HEX64):
        raise UploadRefused(f"{dataset_path} row {rownum} (uid {uid}): proof_raw_sha is not 64-hex")
    if prov.get("source_tier") not in ("band", "collapse", "anchor_solved"):
        # "anchor_solved" added 2026-08-02 (ledger authorization commit
        # 2abe292; Nicky's v3b proof-injection anchors): side-excluded
        # solved-tier records repurposed as retention anchors, hint-
        # regenerated like every other v3 row and censused separately.
        raise UploadRefused(
            f"{dataset_path} row {rownum} (uid {uid}): source_tier {prov.get('source_tier')!r} "
            "not in ('band', 'collapse', 'anchor_solved')"
        )
    prompt = row.get("prompt")
    if (
        not isinstance(prompt, list)
        or len(prompt) != 2
        or not isinstance(prompt[1], dict)
        or not isinstance(prompt[1].get("content"), str)
    ):
        raise UploadRefused(
            f"{dataset_path} row {rownum} (uid {uid}): malformed prompt shape -- "
            "expected [system, user] message dicts with string user content."
        )
    user_content = prompt[1]["content"]
    if config.V3_HINT_MARKER in user_content:
        raise UploadRefused(
            f"{dataset_path} row {rownum} (uid {uid}): training prompt contains the hint marker -- "
            "the hint must never reach a stored training prompt."
        )
    return uid


def _validate_v3_dataset(rows, dataset_path) -> dict:
    """The v3 branch of validate_dataset (authorization: commit 560c7ff).

    Equal strictness, different mechanics: per-row inline checks +
    wellformedness; uid-level dedupe (cap1 -- v3 rows have no rollout_uid);
    sha-chain to the build manifest (the build ran the statement-leakage
    screen against the CURRENT eval set; a byte-identical file inherits that
    screen); membership re-asserted against the pinned v3 proof-split's
    train_side_uids (the old 200/100 eval frame is VOID for v3 data --
    checking it would wrongly refuse former-holdout train rows).

    Membership is tier-dispatched (2026-08-02, v3b): band/collapse rows keep
    the train_side_uids rule verbatim; ``anchor_solved`` rows -- solved-tier,
    disjoint from the split's 921-record universe by construction -- are
    instead checked NOT-in-eval at uid AND paper level. See the tier-dispatch
    comment below and the ledger authorization for why that is the stricter
    check, not a relaxation."""
    # Wellformedness runs per-row (fail-clean fix 2026-08-01; training-ops
    # review of 72cfc39, opslog_train4x.md "fail-safe gap"): a malformed
    # prompt/completion shape must refuse as UploadRefused naming the row,
    # not escape main()'s `except UploadRefused` as TraceIntegrityError --
    # nor as a bare AttributeError when a message entry is not a dict (the
    # check's own ``prompt[0].get`` access). Per-row calls are behavior-
    # identical to the previous one-shot call: the check is a pure
    # per-example loop that raises on its first bad example (build_dataset
    # itself also invokes it one example at a time). LeakageError -- and a
    # nested UploadRefused -- pass through unwrapped: the catch-all must
    # never re-class those two (module contract, validate_dataset docstring).
    for rownum, row in enumerate(rows, start=1):
        try:
            build_dataset.assert_prompt_completion_wellformed([row])
        except build_dataset.TraceIntegrityError as exc:
            raise UploadRefused(f"{dataset_path} row {rownum}: {exc}") from exc
        except (UploadRefused, build_dataset.LeakageError):
            raise
        except Exception as exc:
            raise UploadRefused(
                f"{dataset_path} row {rownum}: malformed prompt/completion shape "
                f"broke the wellformedness check itself ({type(exc).__name__}: {exc})"
            ) from exc
    uids = [
        _assert_v3_row(row, rownum, dataset_path)
        for rownum, row in enumerate(rows, start=1)
    ]
    if len(set(uids)) != len(uids):
        raise UploadRefused(f"{dataset_path}: duplicate uids in v3 dataset (cap1 violated)")
    # Tier is re-read here (rather than returned by _assert_v3_row) to keep that
    # function's single-value contract and its existing callers unchanged; the
    # row shape is already fully validated by the time we get here.
    tiers = [(row.get("provenance") or {}).get("source_tier") for row in rows]

    recomputed = build_dataset.sha256_file(dataset_path)
    manifest_path = Path(config.DATASET_MANIFEST_PATH)
    if not manifest_path.exists():
        raise UploadRefused(f"{manifest_path} does not exist -- v3 dataset must carry its build manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = (manifest.get("dataset") or {}).get("sha256")
    if recorded != recomputed:
        raise UploadRefused(
            f"{dataset_path}: sha256 {recomputed[:16]} != build manifest's {str(recorded)[:16]} -- "
            "the file drifted since build-dataset ran its guards; rebuild before upload."
        )
    corpus_sha = ((manifest.get("inputs") or {}).get("corpus") or {}).get("sha256")
    if corpus_sha != config.EXPECTED_CORPUS_SHA256:
        raise UploadRefused(
            f"{manifest_path}: corpus sha {str(corpus_sha)[:16]} != pinned "
            f"{config.EXPECTED_CORPUS_SHA256[:16]}"
        )

    split_path = Path(config.V3_SPLIT_PATH)
    split_bytes = split_path.read_bytes()
    import hashlib as _hashlib
    if _hashlib.sha256(split_bytes).hexdigest() != config.V3_EXPECTED_SPLIT_SHA256:
        raise UploadRefused(f"{split_path}: sha256 != pinned V3_EXPECTED_SPLIT_SHA256")
    split_obj = json.loads(split_bytes.decode("utf-8"))
    train_side = set(split_obj["train_side_uids"])

    # Membership is TIER-DISPATCHED (authorization: ledger entry "v3b
    # anchor_solved membership exemption", 2026-08-02; prereg Amendment 6 =
    # commit 0139327). band/collapse rows are unchanged -- they must be in the
    # split's train side. "anchor_solved" rows are solved-tier and therefore
    # disjoint from the split's 921-record band/collapse/misdirection universe
    # BY CONSTRUCTION (censused 0/95 in train_side), so the train-side rule
    # would refuse 100% of them. Their rule instead checks non-eval-ness
    # DIRECTLY, at both uid and paper level -- stricter in the dimension this
    # guard exists to protect, since train-side membership is only a
    # positive-list proxy for "not eval".
    anchor_tier = "anchor_solved"
    outside = [
        u for u, t in zip(uids, tiers)
        if t != anchor_tier and u not in train_side
    ]
    if outside:
        raise build_dataset.LeakageError(
            f"{dataset_path}: {len(outside)} row uid(s) outside the v3 split's train side "
            f"(first: {outside[:3]}) -- refusing."
        )

    anchor_idx = [i for i, t in enumerate(tiers) if t == anchor_tier]
    if anchor_idx:
        # Both split fields are required for the anchor rule. Missing/empty is a
        # CLEAN refusal, never a bare KeyError and never a silent skip: the whole
        # point of the exemption is that these two checks replace the train-side
        # one, so a split that cannot support them must stop the upload.
        raw_eval_uids = split_obj.get("eval_set_uids")
        if not raw_eval_uids:
            raise UploadRefused(
                f"{split_path}: eval_set_uids is missing or empty -- cannot run the "
                f"{anchor_tier} uid-level eval-disjointness check; refusing rather than skipping it."
            )
        eval_uids = set(raw_eval_uids)
        eval_papers = set((split_obj.get("papers") or {}).get("eval_papers") or [])
        if not eval_papers:
            raise UploadRefused(
                f"{split_path}: papers.eval_papers is missing or empty -- cannot run the "
                f"{anchor_tier} paper-level disjointness check; refusing rather than skipping it."
            )
        in_eval = [uids[i] for i in anchor_idx if uids[i] in eval_uids]
        if in_eval:
            raise build_dataset.LeakageError(
                f"{dataset_path}: {len(in_eval)} {anchor_tier} row uid(s) are in the v3 split's "
                f"eval set (first: {in_eval[:3]}) -- refusing."
            )
        no_paper, bad_paper = [], []
        for i in anchor_idx:
            paper = ((rows[i].get("provenance") or {}).get("arxiv_id") or "")
            paper = paper.strip() if isinstance(paper, str) else ""
            if not paper:
                no_paper.append(uids[i])
            elif paper in eval_papers:
                bad_paper.append((uids[i], paper))
        if no_paper:
            raise UploadRefused(
                f"{dataset_path}: {len(no_paper)} {anchor_tier} row(s) carry no provenance.arxiv_id "
                f"(first: {no_paper[:3]}) -- the paper-level eval-disjointness check cannot run on "
                "them; refusing rather than passing them silently."
            )
        if bad_paper:
            raise build_dataset.LeakageError(
                f"{dataset_path}: {len(bad_paper)} {anchor_tier} row(s) come from papers in the v3 "
                f"split's eval set (first: {bad_paper[:3]}) -- refusing."
            )
    return {"sha256": recomputed, "rows": len(rows)}


def validate_dataset() -> dict:
    """Re-run every W2 guard against the built dataset, independently, at upload time.

    Returns ``{"sha256": ..., "rows": ...}`` on success. Every failure mode
    raises ``UploadRefused`` EXCEPT leakage, which raises
    ``build_dataset.LeakageError`` directly (never caught/wrapped here) so it
    is never mistaken for an ordinary configuration problem.

    Provenance-shape dispatch (2026-08-01, authorization commit 560c7ff):
    a dataset whose rows ALL carry v3-shaped provenance routes to
    ``_validate_v3_dataset``; all-legacy routes to the original checks
    byte-unchanged; a MIXED file refuses outright.
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

    v3_flags = [_row_is_v3_shaped(row) for row in rows]
    if any(v3_flags):
        if not all(v3_flags):
            raise UploadRefused(
                f"{dataset_path}: MIXED provenance shapes ({sum(v3_flags)} v3-shaped of "
                f"{len(rows)} rows) -- a dataset must be all-legacy or all-v3; refusing."
            )
        return _validate_v3_dataset(rows, dataset_path)

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


def _validate_seeds_subset(seeds) -> list:
    """Validate a seeds-subset override: non-empty, subset of config.SEEDS, no dups.

    Additive (2026-08-01): lets one box's ``run_config.json`` ship fewer
    than the full ``config.SEEDS`` cohort (e.g. 3 seeds for a per-box
    split, or a single staged seed) without editing ``config.py`` itself.
    Returns the validated list, order preserved as given -- callers may
    list a subset in whatever order suits their box assignment; only
    membership/uniqueness is enforced. Raises ``UploadRefused`` (not
    ``ValueError``) naming the specific problem, consistent with every
    other refusal this module raises.
    """
    if not seeds:
        raise UploadRefused("seeds override must be a non-empty list -- got an empty/falsy value.")
    seeds = list(seeds)
    if len(set(seeds)) != len(seeds):
        raise UploadRefused(f"seeds override contains duplicates: {seeds!r}")
    allowed = set(config.SEEDS)
    unknown = [s for s in seeds if s not in allowed]
    if unknown:
        raise UploadRefused(
            f"seeds override {unknown!r} not present in config.SEEDS {list(config.SEEDS)!r} -- "
            "the override must be a subset of the pinned control cohort, never a seed "
            "outside it."
        )
    return seeds


def write_run_config(path, identity_receipt: dict = None, seeds=None) -> None:
    """Write ``run_config.json``: every hyperparameter/seed/pin read live from config.

    The box never hardcodes a parameter -- this is the one file that
    carries them across the wire.

    ``identity_receipt`` is the ALREADY-PARSED section 0.3 identity receipt
    (``main`` has read it by the time this is called, to check its
    verdict) -- passed through here so a ``dequant`` scheme run can carry
    its ``base_manifest_sha256`` chain link without a second file read.
    ``None`` is accepted (and is the right value under the fp16 scheme,
    which never has a manifest-sha to echo) -- every existing call
    site/test that only exercises the fp16 path keeps working unchanged.

    ``seeds``, if given, overrides ``config.SEEDS`` with a validated
    SUBSET (see ``_validate_seeds_subset``) -- the additive per-box
    seed-scoping knob (2026-08-01) for shipping fewer than the full
    cohort to one box. ``None`` (the default) reproduces
    ``list(config.SEEDS)`` exactly -- byte-unchanged for every existing
    call site/test.

    NOTE on ``base_source_sha256``'s shape (adjudicated report-only, review
    2026-07-30): it holds a 40-hex git commit sha under the fp16 scheme
    (``verify_base_identity.FP16_REVISION``) but a 64-hex sha256 under the
    dequant scheme (``verify_base_identity.EXPECTED_BASE_GGUF_SHA256``) --
    same field name, two different hash algorithms/lengths depending on
    ``config.BASE_SCHEME``. Documented here rather than renamed: the field
    means "the pin identifying this run's base source," and which kind of
    pin that is follows directly from ``base_scheme`` sitting right next to
    it in this same payload.
    """
    effective_seeds = _validate_seeds_subset(seeds) if seeds is not None else list(config.SEEDS)
    payload = {
        # Explicit list from config (v2, 2026-07-30). Was
        # [SEED, SEED+1, SEED+2] -- capped the campaign at 3 seeds and broke
        # across month boundaries; see config.SEEDS for the pinned control set.
        # Per-box subset override (2026-08-01): "seeds" ships EFFECTIVE_SEEDS,
        # not necessarily the full cohort -- see the "seeds" parameter above.
        "seeds": effective_seeds,
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
        # Base-scheme provenance (T4, 2026-07-30): which of the two ways
        # this run's LoRA base was obtained, plus the pin identifying that
        # source, so runs from the two schemes are never silently
        # comparable (see verify_base_identity.check_same_base_scheme /
        # --compare-runs and this module's own preflight chain check,
        # check_base_scheme, below). ADDITIVE ONLY under the shipped fp16
        # default -- see test_write_run_config_additive_only_under_default_scheme.
        "base_scheme": config.BASE_SCHEME,
        "base_source_sha256": (
            verify_base_identity.EXPECTED_BASE_GGUF_SHA256
            if config.BASE_SCHEME == config.BASE_SCHEME_DEQUANT
            else verify_base_identity.FP16_REVISION
        ),
    }
    if config.BASE_SCHEME == config.BASE_SCHEME_DEQUANT:
        chain = (identity_receipt or {}).get("chain") or {}
        base_manifest_sha256 = chain.get("dequant_manifest_sha256")
        if not base_manifest_sha256:
            # Review fix #8 (fail-open null): a dequant-scheme run_config.json
            # with no manifest-sha chain link is not a degraded-but-shippable
            # artifact -- it means the identity receipt this call was handed
            # either predates the chain (T3) or was hand-edited, and there is
            # nothing on the other end of check_base_scheme's chain-of-custody
            # to verify against. Refuse rather than silently emitting a null.
            raise UploadRefused(
                "config.BASE_SCHEME is dequant_q4km but the identity receipt carries no "
                "chain.dequant_manifest_sha256 -- refusing to ship a run_config.json with no "
                "manifest-sha chain link for a dequant-scheme run. Re-run verify_base_identity "
                "--dequant-dir first."
            )
        payload["base_manifest_sha256"] = base_manifest_sha256
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def check_base_scheme(identity_receipt: dict) -> None:
    """Refuse if the identity receipt's base scheme disagrees with ``config.BASE_SCHEME``.

    A receipt written before this feature existed carries no ``scheme`` key
    at all -- treated as ``config.BASE_SCHEME_FP16`` for backward
    compatibility (every receipt before T3/T4 necessarily came from the
    fp16 structural comparator). This is the upload-time half of the
    "never silently compare runs from different base schemes" rule (see
    ``verify_base_identity.check_same_base_scheme`` / ``--compare-runs``
    for the run-comparison half, applied after the fact to two already-
    trained runs).
    """
    receipt_scheme = identity_receipt.get("scheme", config.BASE_SCHEME_FP16)
    if receipt_scheme != config.BASE_SCHEME:
        raise UploadRefused(
            f"identity receipt scheme {receipt_scheme!r} != config.BASE_SCHEME "
            f"{config.BASE_SCHEME!r} -- refusing to upload: the identity preflight "
            "(RUNBOOK section 0.3) was verified against a different base scheme than "
            "this config is currently set to train against. Runs from the two base "
            "schemes must never be silently comparable -- re-run verify_base_identity "
            "for the scheme config.BASE_SCHEME actually names, or flip config.BASE_SCHEME "
            "to match the receipt you have."
        )


def check_source_verified(identity_receipt: dict) -> None:
    """Refuse a dequant-scheme upload unless the identity receipt PROVES a
    verified source-GGUF match (review fix #1, round 4 -- rewritten
    fail-closed; supersedes review fix #2, round 3, which was a fail-open
    blocklist).

    The round-3 version enumerated shapes to REFUSE (``source_verified is
    False``, or absent with a null ``gguf_sha256``) -- a blocklist, which
    is fail-OPEN by construction: every shape not explicitly listed
    quietly PASSES. The attack-replay found three that did:
    ``{"source_verified": True, "gguf_sha256": null}``, ``{"source_verified":
    "false"}`` (a STRING, not the boolean marker -- `"false" is False` is
    ``False`` in Python, so the old identity check missed it), and
    ``{"source_verified": True, "gguf_sha256": "00"*32}`` (a well-formed
    but WRONG hash -- the round-3 code never compared the receipt's sha
    against the pin at upload time AT ALL). None of these are
    cryptographic proof the local GGUF file was hashed and matched --
    yet none tripped the specific conditions the blocklist checked for.

    Now a WHITELIST: the upload proceeds ONLY when
    ``chain.source_verified is True`` (identity check -- a truthy
    non-``bool`` value such as the string ``"true"`` does NOT count) AND
    ``chain.gguf_sha256 == verify_base_identity.EXPECTED_BASE_GGUF_SHA256``
    (exact equality against the pin -- not "looks like a sha256", not
    "matches the manifest's own unverified claim"). Anything else --
    absent, wrong type, wrong value -- refuses.

    LEGACY-SHAPE HANDLING: there is no backward-compatibility carve-out
    here, deliberately. Every other pinned-value check in this codebase
    that tolerates an old shape (e.g. ``check_base_scheme`` treating a
    receipt with no ``scheme`` key as fp16) does so because that old shape
    was ACTUALLY SHIPPED and has real artifacts to stay compatible with.
    The dequant scheme, its receipts, and this chain field have never
    shipped an upload -- there is nothing legacy to grandfather in, so
    every non-conforming shape (including one that predates this field
    entirely) refuses the same way. Only meaningful under
    ``config.BASE_SCHEME_DEQUANT`` -- callers must gate on that themselves
    (the fp16 receipt shape has no ``chain.source_verified`` concept at
    all; see ``main()``).
    """
    chain = identity_receipt.get("chain") or {}
    source_verified = chain.get("source_verified")
    gguf_sha256 = chain.get("gguf_sha256")
    if source_verified is True and gguf_sha256 == verify_base_identity.EXPECTED_BASE_GGUF_SHA256:
        return
    raise UploadRefused(
        "identity receipt chain does not prove a verified source-GGUF match -- refusing to "
        f"upload: chain.source_verified={source_verified!r} (must be exactly True) and "
        f"chain.gguf_sha256={gguf_sha256!r} (must equal the pinned "
        f"EXPECTED_BASE_GGUF_SHA256={verify_base_identity.EXPECTED_BASE_GGUF_SHA256!r}). "
        "Re-run verify_base_identity --dequant-dir WITHOUT --skip-file-sha before uploading."
    )


def write_receipt(path, dataset_info: dict, payload_shas: dict) -> None:
    """Write ``upload_receipt.json``: the sha the box re-checks before training."""
    payload = {
        "dataset_sha256": dataset_info["sha256"],
        "dataset_rows": dataset_info["rows"],
        "files": dict(payload_shas),
        "generated_unix": time.time(),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_scp_command(files, ssh_port, server_ip: "str | None" = None) -> list:
    """Pure argv builder for the guarded scp -- no subprocess started here.

    ``server_ip``, if given, is used in place of ``config.TRAIN_SERVER_IP``
    (additive, 2026-08-01 -- e.g. the already-resolved value from
    ``resolve_server_ip``). Omitting it reproduces the pre-existing
    behavior exactly.
    """
    ip = server_ip if server_ip is not None else config.TRAIN_SERVER_IP
    return ["scp", "-P", str(ssh_port)] + [str(f) for f in files] + [f"root@{ip}:/workspace/run/"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="upload_guard")
    parser.add_argument("--execute", action="store_true", help="actually scp; default is dry-run (print the command only)")
    parser.add_argument(
        "--identity-receipt",
        default=None,
        help="path to the section 0.3 identity receipt (default: config.DATA_DIR/identity_receipt.json)",
    )
    parser.add_argument(
        "--server-ip",
        default=None,
        help=(
            "override config.TRAIN_SERVER_IP for THIS invocation only (additive, "
            "2026-08-01 -- see resolve_server_ip). Lets one call target a specific "
            "box without editing config.py's operator block; default (omitted) "
            "reproduces config.TRAIN_SERVER_IP exactly. Never hardcode an address "
            "here in code -- always pass it at invocation (RUNBOOK single-source-"
            "of-truth rule)."
        ),
    )
    parser.add_argument(
        "--server-ssh-port",
        default=None,
        type=int,
        help=(
            "override the resolved SSH port for THIS invocation only (additive, "
            "2026-08-01 -- see resolve_ssh_port). Default (omitted) reproduces the "
            "existing config.TRAIN_SERVER_SSH_PORT / TRAIN_SSH_PORT resolution."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "comma-separated subset of config.SEEDS to ship in run_config.json "
            "(additive, 2026-08-01 -- see write_run_config/_validate_seeds_subset). "
            "Default (omitted) ships the full config.SEEDS cohort, unchanged."
        ),
    )
    args = parser.parse_args(argv)

    identity_receipt_path = (
        Path(args.identity_receipt) if args.identity_receipt else (config.DATA_DIR / "identity_receipt.json")
    )

    try:
        config.validate_config()
        server_ip = resolve_server_ip(args.server_ip)
        check_target(server_ip)
        ssh_port = resolve_ssh_port(args.server_ssh_port)

        seeds_override = None
        if args.seeds is not None:
            try:
                seeds_override = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            except ValueError:
                raise UploadRefused(
                    f"--seeds must be a comma-separated list of ints, got {args.seeds!r}"
                ) from None

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
        check_base_scheme(identity_receipt)
        if config.BASE_SCHEME == config.BASE_SCHEME_DEQUANT:
            check_source_verified(identity_receipt)

        dataset_info = validate_dataset()

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        run_config_path = config.DATA_DIR / "run_config.json"
        upload_receipt_path = config.DATA_DIR / "upload_receipt.json"
        write_run_config(run_config_path, identity_receipt, seeds=seeds_override)

        dataset_path = Path(config.SFT_DATASET_PATH)
        payload_shas = {
            dataset_path.name: dataset_info["sha256"],
            run_config_path.name: build_dataset.sha256_file(run_config_path),
        }
        write_receipt(upload_receipt_path, dataset_info, payload_shas)

        payload = [dataset_path, run_config_path, upload_receipt_path]
        check_blocklist(payload)

        cmd = build_scp_command(payload, ssh_port, server_ip)

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

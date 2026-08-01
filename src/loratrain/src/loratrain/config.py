"""Single source of truth for loratrain configuration.

The remote training server's IP is operator-editable HERE and NOWHERE
ELSE (see the banner below). Every other module in this package builds
its requests from ``TRAIN_SERVER_URL``; ``tests/test_config.py::
test_single_source_of_truth_for_server_address`` scans every other
``*.py`` file in this package and fails the suite if an IP or URL
literal shows up outside this file.

Address semantics since the SSH-tunnel-only decision (RUNBOOK D-R1,
revised 2026-07-25): the box binds its status server to the container's
loopback interface, so ``TRAIN_SERVER_IP`` (the pod's public IP) is the
ssh/scp target and NOTHING else, ``TRAIN_SERVER_PORT`` is the M4-LOCAL
end of the RUNBOOK section 6 status tunnel, and ``TRAIN_SERVER_URL`` is
the tunnel-local URL the operator curls -- it never carries the pod's
address. ``validate_config()`` enforces that form. The RUNBOOK
Appendix A operator-block rewrite (the ``TRAIN_SERVER_SSH_PORT`` field
plus the tunnel-local URL derivation line) was APPLIED 2026-07-25 on
Nicky's go-ahead, so the block below is the tunnel-era contract;
``TRAIN_SERVER_SSH_PORT`` is preferred over the ``TRAIN_SSH_PORT`` env
fallback wherever ssh is driven (see ``upload_guard.resolve_ssh_port``).

This module also holds the pinned corpus/split identity, hyperparameter
defaults, derived paths, and ``validate_config()``, which collects every
configuration problem it finds and raises them together (rather than
failing fast on the first one) so an operator editing this file sees the
whole list in one pass.
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path

# ============================================================================
# OPERATOR-EDITABLE SECTION -- READ THIS FIRST
# ============================================================================
# The remote training box's address lives in EXACTLY ONE place: the operator
# variables immediately below (TRAIN_SERVER_IP, TRAIN_SERVER_PORT,
# TRAIN_SERVER_SSH_PORT). TRAIN_SERVER_URL is derived -- tunnel-local, from
# TRAIN_SERVER_PORT only (RUNBOOK D-R1) -- and must never be hand-edited.
# tests/test_config.py::test_single_source_of_truth_for_server_address scans
# every *.py file in this package (except this one) for IP/URL literals and
# fails the suite if it finds one -- so there is nowhere else to put it.
# ============================================================================

TRAIN_SERVER_IP = "127.0.0.1"   # <-- EDIT HERE: the pod's public IP -- the ssh/scp target and NOTHING else (single source of truth). RESET to placeholder 2026-07-30 at section 9 teardown: the v2 box (pod b3njpwlrpbzh26, 69.30.85.67) was TERMINATED -- RunPod reassigns IPs to other tenants, so a dead address must never linger here. NOTE: HEAD carried run-1's dead 69.30.85.138 from 2026-07-27 to 2026-07-30 because that round's reset was made in the working tree but never committed -- commit the reset, not just the edit.
TRAIN_SERVER_PORT = 8000        # M4-LOCAL end of the section 6 status tunnel; edit only if local 8000 is occupied
TRAIN_SERVER_SSH_PORT = 22       # <-- EDIT HERE when provisioning: the pod's external TCP port mapped to container 22 (reset to shipped default 2026-07-30, same reason as the IP). NOTE: RunPod reassigns this mapping on every container recreate -- observed 22117 -> 22145 -> 22174 across a single restart on 2026-07-30 -- so re-read it from the pod and update here after any stop/start/restart, or ssh/scp/tunnel all fail with "connection refused".
TRAIN_SERVER_URL = f"http://127.0.0.1:{TRAIN_SERVER_PORT}"  # derived, tunnel-local -- what the operator curls while the section 6 tunnel is up; never carries the pod IP; never edit this line
TRAIN_SERVER_KEY_FILE = None  # optional path proxy to a bearer-key file (raw token or KEY=VALUE); None = keyless; contents never printed/logged

# ============================================================================
# Status-tunnel constant (SSH-tunnel-only, RUNBOOK D-R1 revised 2026-07-25).
# NOT an operator knob: this is the box-side half of the section 6 tunnel.
# ============================================================================

# Container-loopback port the box's status server binds (run_remote_train.sh
# starts it with --bind on loopback). MUST equal that script's STATUS_PORT
# default -- tests/test_remote_scripts.py fails the suite on drift. The
# operator-side half of the tunnel is TRAIN_SERVER_PORT above.
TRAIN_STATUS_BOX_PORT = 8000

# ============================================================================
# Pinned corpus / split identity (corpus dated 2026-07-22 -- the sha the
# 2026-07-16 repair-lane fold produced when the band went 309 -> 293 rows;
# split repointed 2026-07-26 on Nicky's ruling, see below)
# ============================================================================

EXPECTED_CORPUS_SHA256 = "e0975e112f05d03e599c9fac25fd27e523fd8b4b24664281f8a596f9d8646554"  # pinned 2026-07-22
EXPECTED_CORPUS_ROWS = 293  # pinned 2026-07-22

# Split pin -- REPOINTED 2026-07-26 (Nicky's ruling: "keep 200/100; backfill
# the 7-record shortfall from the GGUF 7/8 pool"). corpus_split_200_100.json
# is authoritative again; the prior derived-view split (frozen
# eval_paper_split.json, sha16 110a4bf27320f2b1) is RETIRED to
# evalharness/data/retired_20260726/ (non-destructive, ruling recorded
# there). Full-file sha256 over exact bytes -- same pin-check style as
# EXPECTED_CORPUS_SHA256 above (see build_dataset.assert_split_pinned).
EXPECTED_SPLIT_SHA256 = "768436f4e55e2a46eb5abafbd1d12eebe16e764f95361d5506ba6ea29ea9bc00"  # pinned 2026-07-26 (evalharness/data/corpus_split_200_100.json: 109 eval_papers, 200 train_uids incl. 7 GGUF-7/8 backfill, 100 holdout_uids)
EXPECTED_SPLIT_SHA256_16 = EXPECTED_SPLIT_SHA256[:16]  # DERIVED -- not an independent pin. Kept only so load_eval_papers' own sha16 defense-in-depth check (and upload_guard.py's call of it) keep working unchanged; EXPECTED_SPLIT_SHA256 above is the single source of truth.

BASE_MODEL_HF_ID = "Qwen/Qwen3-8B"  # pinned 2026-07-22 (README D2)
SERVE_QUANT = "GGUF-Q4_K_M"  # pinned 2026-07-22 (README D3: baseline scored format, path A serving quant)
ADAPTER_FORMAT = "peft"  # pinned 2026-07-22 (README D1: binding contract is on the artifact, not the remote stack)

# ============================================================================
# Base-scheme provenance (T4, 2026-07-30). The LoRA base can be obtained two
# ways: the pinned upstream fp16 HF revision (verify_base_identity.FP16_REVISION,
# the original W1/W2 path), or by dequantizing the pinned deployment GGUF
# (gguf_to_hf.py's output, identity-checked by
# verify_base_identity.check_dequant_manifest / --dequant-dir). Both name the
# SAME underlying weights; they are not interchangeable in a training run's
# provenance because the dequant path recovers them through Q4_K_M rounding
# rather than reading the fp16 tensors directly.
#
# BASE_SCHEME is OPERATOR-EDITABLE, mirroring the WEIGHT_POLICY knob above:
# BASE_SCHEME_FP16 is the shipped default (current behavior, unchanged by
# this revision) -- flipping it to BASE_SCHEME_DEQUANT is the operator's
# decision, gated on gguf_to_hf.py's output passing its own identity
# preflight, not something this revision does unilaterally.
#
# Runs built under the two schemes must NEVER be silently compared (their
# base weights are recovered through different paths): upload_guard's
# preflight refuses when the identity receipt's scheme disagrees with
# BASE_SCHEME (upload_guard.check_base_scheme), and
# verify_base_identity.check_same_base_scheme / --compare-runs is the
# post-hoc tripwire for comparing two already-trained runs.
# ============================================================================

BASE_SCHEME_FP16 = "fp16_hf_revision"
BASE_SCHEME_DEQUANT = "dequant_q4km"
BASE_SCHEME = BASE_SCHEME_FP16  # <-- EDIT HERE to flip: operator decision (see block comment above), not a default this revision changes. Flipped back 2026-08-01: dq campaign CLOSED (k=8 verdict dq ≈ v1 — quantization mismatch is not the limiter), per this comment's own flip-back rule and v3_full_run_skeleton P4 (fp16 stands). Prior state for the record: FLIPPED 2026-07-30 for the dequant arm per Nicky's Rule-A release (out/analysis/dequant_arm/GATE_FAILURE_FINDING.md).
VALID_BASE_SCHEMES = (BASE_SCHEME_FP16, BASE_SCHEME_DEQUANT)

# Expected `tensor_census.total` in gguf_to_hf.py's dequant_manifest.json for
# the pinned Qwen3-8B base -- checked by
# verify_base_identity.check_dequant_manifest (T3). Pinned per the tool's own
# manifest contract, not re-derived here.
EXPECTED_DEQUANT_TENSOR_TOTAL = 399

# ============================================================================
# GGUF 7/8 backfill roster (Nicky's ruling 2026-07-26). The split keeps a
# 200-uid train set by backfilling the 7-record shortfall the 2026-07-16
# repair-lane fold left (band_corpus 309 -> 293) from the GGUF 7/8 rescore
# pool. These 7 uids are NOT in band_corpus.jsonl by construction -- that
# absence is exactly why they need backfilling -- so build_dataset.py's
# membership guard exempts precisely this pinned set (and only this set)
# from its normal "train uid must be in the corpus" refusal.
#
# Each value names the FIRST-PASS rescore pass_at_k.jsonl the record's row
# (and its n_correct==7 / label=="solved" guarantee) is drawn from.
# Deliberately NOT out/remote_rescore/rerun_7of8_local/pass_at_k.jsonl:
# that dir is a SELECTION-BIASED re-roll of these same 7 uids (re-sampled
# specifically because they were near-misses on the first pass), not an
# unbiased draw -- training on it would upweight a re-rolled-until-better
# trace.
#
# Resolved 2026-07-26 from the split's train_backfill_7of8_uids
# (evalharness/data/corpus_split_200_100.json, full uids recovered from
# their truncated form in the ruling). build_dataset.build() independently
# re-verifies (defense in depth, same idiom as the corpus/split sha pins)
# that this dict's keys are EXACTLY that split field's uid set -- see
# assert_backfill_mapping_complete. validate_config() below only sanity-
# checks this dict's own shape (dict, non-empty str keys/values); it
# does not touch disk.
# ============================================================================

BACKFILL_TRACE_SOURCES = {
    "11df559573d69311d33e069dd5d05f27": "out/remote_rescore/tier2_7of8/pass_at_k.jsonl",
    "1e386900aa80145e2692a4baebcb548c": "out/remote_rescore/tier2_7of8/pass_at_k.jsonl",
    "5dc9a792f07167e373cf0741439d0955": "out/remote_rescore/tier2_7of8/pass_at_k.jsonl",
    "b59d35fbb31942e16f8008afa94f4b9f": "out/remote_rescore/tier2_7of8/pass_at_k.jsonl",
    "b9b2d07629ce44d22dd4ed09beeda1ab": "out/remote_rescore/tier2_7of8/pass_at_k.jsonl",
    "47a8c1fe5fc49a328a32f23b347763d8": "out/remote_rescore/tier1_band/pass_at_k.jsonl",
    "e5564bbf09e3f21bc179d41ee7344104": "out/remote_rescore/tier1_band/pass_at_k.jsonl",
}

# --- Pass@k wire-format pins (README D4: byte-identity with the scoring run) --
# Frozen copies of src/icepick/processing/pass_at_k/config.py::SYSTEM_PROMPT and
# the qwen_http " /no_think" suffix (think=False, echoed by every rescore run
# manifest). loratrain is stdlib-only with zero icepick imports (README top),
# so these are deliberate duplicates, pinned here and tripwired by
# tests/test_build_dataset.py -- if pass@k's wire format ever changes, BOTH
# copies move in one deliberate edit, or train/serve distributions drift apart.
PASS_AT_K_SYSTEM_PROMPT = "Solve the problem. State only the final answer inside \\boxed{}."  # pinned 2026-07-25
PASS_AT_K_NO_THINK_SUFFIX = " /no_think"  # pinned 2026-07-25 (leading space is load-bearing)

SEED = 20260722

# --- Training seed set (v2 campaign, 2026-07-30) -------------------------------
# SEED (above) is the DATASET seed: build_dataset.apply_weight_policy ranks each
# uid's traces by sha256("{seed}:{uid}:{rollout_uid}"), so changing it reshuffles
# which traces cap1 keeps. It stays pinned to the value
# data/v2/*/dataset_manifest.json was already built under -- never retune it to
# add training seeds.
#
# SEEDS is the TRAINING seed list run_config.json ships to the box, one adapter
# per entry. v1's upload_guard derived this as [SEED, SEED+1, SEED+2], which both
# capped a campaign at 3 seeds and breaks across month boundaries (20260731 + 1 =
# 20260732, not 20260801) -- v1's 12 control seeds were in fact assembled across
# three separate runs, never from that expression. Pinned here as the explicit v1
# control set so v2 is seed-PAIRED with v1 and the measured per-seed sd of 3.45pp
# works for the comparison instead of against it. 20260728 is deliberately
# ABSENT: it was the stage-A HP screen's shared seed on a 160-record subset, not
# a control seed (docs/lora_campaign_results.md).
SEEDS = [
    # v3-full-run cohort (R5: 12 FRESH seeds; Nicky release 2026-08-01, pinned
    # here BEFORE training per PREREGISTRATION_V3 Amendment 1 §2). Deliberately
    # disjoint from every prior campaign seed: the v1/v2 cohort was
    # 20260722-27/29-31 + 20260801-03 (20260728 excluded as the HP-screen
    # seed), so date-like values continue at 202609xx to stay collision-free
    # and visually distinguishable as a new cohort.
    20260901, 20260902, 20260903, 20260904, 20260905, 20260906,
    20260907, 20260908, 20260909, 20260910, 20260911, 20260912,
]
# Prior cohort, retired from the ACTIVE list 2026-08-01 (v1/v2/dq arms are
# closed; their manifests pin their own seeds -- this constant only feeds NEW
# run_configs): 20260722-24 (run-1), 20260725-27/29-30 (stage R),
# 20260731/20260801-03 (D2 extension).

# --- Hyperparameters ---------------------------------------------------------
# Conventional starting points, not tuned -- W3 review (README "Open items").
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
EPOCHS = 3
MICRO_BATCH_SIZE = 4
MAX_SEQ_LEN = 4096

# --- Trainer schedule/accumulation pins (v2, 2026-07-29) ----------------------
# In v1 these four were INVISIBLE: grad-accum was a hardcoded literal at
# remote/train_qwen3_lora.py:131 and the other three were silent SFTConfig
# defaults, none recorded in any run manifest (found by the 2026-07-28
# unbriefed external reviews -- docs/lora_params_rationale.md section 1).
# The VALUES are deliberately unchanged from what v1 actually ran (effective
# batch 16 = 4 micro x 4 accum; linear decay to 0; no warmup; no weight
# decay) so v1<->v2 comparisons isolate the dataset fix -- these pins make
# them visible and manifest-recorded, not different.
GRAD_ACCUM_STEPS = 4
LR_SCHEDULER_TYPE = "linear"
WARMUP_RATIO = 0.0
WEIGHT_DECAY = 0.0

# --- Dataset v2 weight policy (defect 2: gradient weight == n_correct) --------
# v1 emitted one SFT row per verified-correct trace, so a record's gradient
# mass equaled how often the BASE model already solved it (rows-per-uid
# histogram {1:38, 2:37, 3:29, 4:24, 5:34, 6:31, 7:7} over 200 uids / 700
# rows) -- anti-difficulty weighting: the hardest band records got 1/7 the
# weight of near-ceiling ones. WEIGHT_POLICY caps/reweights that:
#   "cap1"    -- one trace per uid (default; 200 rows at N=200)
#   "capk"    -- at most WEIGHT_POLICY_CAP_K traces per uid
#   "inverse" -- keep every trace, stamp each row with weight = 1/n_traces
#                so each uid's total gradient mass is equal (requires the
#                trainer's weighted-loss path; rows gain a "weight" field)
# Which policy SHIPS is Nicky's decision (work order 2026-07-29) -- all three
# are implemented; this default is the build default, not the ruling.
# Trace selection under the cap policies is deterministic by seed --
# build_dataset.apply_weight_policy ranks each uid's traces by
# sha256("{seed}:{uid}:{rollout_uid}") and keeps the first cap -- never a
# silent "first N by file order"; the rule is echoed into the manifest.
WEIGHT_POLICY = "cap1"
WEIGHT_POLICY_CAP_K = 3  # only read when WEIGHT_POLICY == "capk"
VALID_WEIGHT_POLICIES = ("cap1", "capk", "inverse")


def weight_policy_label(policy=None, cap_k=None) -> str:
    """Directory/report label for a weight policy: cap1 | cap<k> | inverse.

    Defaults to the module-level knobs. Pure string derivation -- no
    validation here (``validate_config()`` owns that), so importing this
    module with a bad knob still lets validate_config report the full
    problem list instead of crashing at import time.
    """
    policy = WEIGHT_POLICY if policy is None else policy
    cap_k = WEIGHT_POLICY_CAP_K if cap_k is None else cap_k
    if policy == "capk":
        return f"cap{cap_k}"
    return str(policy)

# --- Paths --------------------------------------------------------------------
# config.py sits at icepick/src/loratrain/src/loratrain/config.py, so
# parents[4] from this file is the icepick repo root:
#   parents[0] = .../icepick/src/loratrain/src/loratrain
#   parents[1] = .../icepick/src/loratrain/src
#   parents[2] = .../icepick/src/loratrain
#   parents[3] = .../icepick/src
#   parents[4] = .../icepick
# Verified manually against (REPO_ROOT / "AGENTS.md").exists() -- see the
# sanity check run outside the test suite; fix this file, not the check, if
# that index is ever wrong.
REPO_ROOT = Path(__file__).resolve().parents[4]
SUBREPO_ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = REPO_ROOT / "out/corpus_pde625/band_corpus.jsonl"
EVAL_PAPER_SPLIT_PATH = REPO_ROOT / "evalharness/data/corpus_split_200_100.json"  # repointed 2026-07-26 (Nicky's ruling) -- old eval_paper_split.json RETIRED to evalharness/data/retired_20260726/
TRAIN_UIDS_PATH = REPO_ROOT / "evalharness/data/train_uids.txt"      # output of evalharness-build-set; operator may repoint to the build's --output-dir
EVAL_SET_PATH = REPO_ROOT / "evalharness/data/eval_set.jsonl"        # same
BASELINE_GREEDY_PATH = REPO_ROOT / "out/evalharness/run1/baseline_greedy.jsonl"  # the eval run dir the operator captures the baseline into

DATA_DIR = SUBREPO_ROOT / "data"
# Dataset v2 (2026-07-29): builds land under data/v2/<policy-label>/ so the
# v1 artifacts (data/sft_train.jsonl and data/run1_final/**, the run-1
# comparison baseline) are never overwritten. SFT_DATASET_PATH /
# DATASET_MANIFEST_PATH -- what upload_guard validates and ships when W3
# reopens -- now track the CURRENT (v2, policy-labeled) build; leaving them
# on the v1 file would silently upload the defective-recipe dataset, the
# exact silent-default failure mode this revision removes.
DATA_V2_DIR = DATA_DIR / "v2"
SFT_DATASET_PATH = DATA_V2_DIR / weight_policy_label() / "sft_train.jsonl"
DATASET_MANIFEST_PATH = DATA_V2_DIR / weight_policy_label() / "dataset_manifest.json"
ADAPTER_DIR = DATA_DIR / "adapter"
RUN_MANIFEST_PATH = DATA_DIR / "run_manifest.json"

_HOSTNAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA16_RE = re.compile(r"^[0-9a-f]{16}$")


class ConfigError(ValueError):
    """Raised by ``validate_config()`` when one or more settings are invalid.

    Collects every problem found rather than raising on the first one,
    so an operator editing this file sees the whole list in one pass.
    """


def _is_valid_ip_or_hostname(value) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        ipaddress.IPv4Address(value)
        return True
    except ValueError:
        pass
    return bool(_HOSTNAME_RE.match(value))


def validate_config() -> None:
    """Validate every setting in this module; raise ``ConfigError`` listing ALL problems.

    Never fails fast -- every check below runs regardless of earlier
    failures, and a single ``ConfigError`` is raised at the end
    enumerating everything wrong, so an operator sees the complete list
    in one pass rather than fixing issues one at a time.
    """
    problems = []

    # --- train server address --------------------------------------------------
    if not _is_valid_ip_or_hostname(TRAIN_SERVER_IP):
        problems.append(
            f"TRAIN_SERVER_IP={TRAIN_SERVER_IP!r} must be a valid IPv4 address or a "
            "hostname matching ^[a-zA-Z][a-zA-Z0-9.-]*$ (empty/whitespace is rejected)"
        )

    if (
        not isinstance(TRAIN_SERVER_PORT, int)
        or isinstance(TRAIN_SERVER_PORT, bool)
        or not (1 <= TRAIN_SERVER_PORT <= 65535)
    ):
        problems.append(
            f"TRAIN_SERVER_PORT must be an int in [1, 65535] (got {TRAIN_SERVER_PORT!r})"
        )

    if (
        not isinstance(TRAIN_STATUS_BOX_PORT, int)
        or isinstance(TRAIN_STATUS_BOX_PORT, bool)
        or not (1 <= TRAIN_STATUS_BOX_PORT <= 65535)
    ):
        problems.append(
            f"TRAIN_STATUS_BOX_PORT must be an int in [1, 65535] (got {TRAIN_STATUS_BOX_PORT!r})"
        )

    # TRAIN_SERVER_SSH_PORT (Appendix A, applied 2026-07-25) is looked up
    # tolerantly: upload_guard/tunnel fall back to the TRAIN_SSH_PORT env var
    # when the attribute is absent (tests exercise that path by deleting it,
    # and resolve_ssh_port owns the missing-everywhere refusal) -- so absence
    # is not a config problem here, but a present-and-invalid value is.
    ssh_port = globals().get("TRAIN_SERVER_SSH_PORT")
    if ssh_port is not None and (
        not isinstance(ssh_port, int)
        or isinstance(ssh_port, bool)
        or not (1 <= ssh_port <= 65535)
    ):
        problems.append(
            f"TRAIN_SERVER_SSH_PORT must be an int in [1, 65535] (got {ssh_port!r})"
        )

    # SSH-tunnel-only (RUNBOOK D-R1, 2026-07-25): the URL the operator curls
    # is tunnel-local -- loopback host + TRAIN_SERVER_PORT -- and NEVER carries
    # the pod's address (the box binds its status server to loopback, so the
    # pod IP is unreachable on that port by design; it is the ssh/scp target
    # only). The Appendix A derivation line spelling this form was applied
    # 2026-07-25, so any mismatch now means the derived line was hand-edited.
    expected_url = f"http://127.0.0.1:{TRAIN_SERVER_PORT}"
    if TRAIN_SERVER_URL != expected_url:
        problems.append(
            f"TRAIN_SERVER_URL must be the tunnel-local status URL {expected_url!r} "
            f"(got {TRAIN_SERVER_URL!r}). Since the SSH-tunnel-only decision (RUNBOOK "
            "D-R1, 2026-07-25) the box's status endpoint binds loopback and is reached "
            "through the RUNBOOK section 6 SSH local-forward, so the URL never carries "
            "the pod's IP. Restore the derived line (Appendix A, applied 2026-07-25): "
            "edit TRAIN_SERVER_PORT if the local port must change; never hand-edit "
            "the URL itself."
        )

    if TRAIN_SERVER_KEY_FILE is not None and not isinstance(TRAIN_SERVER_KEY_FILE, (str, Path)):
        problems.append(
            "TRAIN_SERVER_KEY_FILE must be None, a str, or a Path (existence is not "
            f"required at validate time) -- got {type(TRAIN_SERVER_KEY_FILE).__name__}"
        )

    # --- hyperparameters ---------------------------------------------------------
    for name, value in (
        ("LORA_RANK", LORA_RANK),
        ("LORA_ALPHA", LORA_ALPHA),
        ("EPOCHS", EPOCHS),
        ("MICRO_BATCH_SIZE", MICRO_BATCH_SIZE),
        ("MAX_SEQ_LEN", MAX_SEQ_LEN),
        ("GRAD_ACCUM_STEPS", GRAD_ACCUM_STEPS),
        ("WEIGHT_POLICY_CAP_K", WEIGHT_POLICY_CAP_K),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            problems.append(f"{name} must be a positive int (got {value!r})")

    if not isinstance(LR_SCHEDULER_TYPE, str) or not LR_SCHEDULER_TYPE:
        problems.append(
            f"LR_SCHEDULER_TYPE must be a non-empty str (got {LR_SCHEDULER_TYPE!r})"
        )

    if (
        not isinstance(WARMUP_RATIO, (int, float))
        or isinstance(WARMUP_RATIO, bool)
        or not (0 <= WARMUP_RATIO < 1)
    ):
        problems.append(f"WARMUP_RATIO must satisfy 0 <= WARMUP_RATIO < 1 (got {WARMUP_RATIO!r})")

    if (
        not isinstance(WEIGHT_DECAY, (int, float))
        or isinstance(WEIGHT_DECAY, bool)
        or WEIGHT_DECAY < 0
    ):
        problems.append(f"WEIGHT_DECAY must be a non-negative number (got {WEIGHT_DECAY!r})")

    if WEIGHT_POLICY not in VALID_WEIGHT_POLICIES:
        problems.append(
            f"WEIGHT_POLICY must be one of {VALID_WEIGHT_POLICIES} (got {WEIGHT_POLICY!r})"
        )

    if BASE_SCHEME not in VALID_BASE_SCHEMES:
        problems.append(
            f"BASE_SCHEME must be one of {VALID_BASE_SCHEMES} (got {BASE_SCHEME!r})"
        )

    if (
        not isinstance(LORA_DROPOUT, (int, float))
        or isinstance(LORA_DROPOUT, bool)
        or not (0 < LORA_DROPOUT < 1)
    ):
        problems.append(f"LORA_DROPOUT must satisfy 0 < LORA_DROPOUT < 1 (got {LORA_DROPOUT!r})")

    if (
        not isinstance(LEARNING_RATE, (int, float))
        or isinstance(LEARNING_RATE, bool)
        or not (0 < LEARNING_RATE < 1)
    ):
        problems.append(f"LEARNING_RATE must satisfy 0 < LEARNING_RATE < 1 (got {LEARNING_RATE!r})")

    if not isinstance(SEED, int) or isinstance(SEED, bool) or SEED <= 0:
        problems.append(f"SEED must be a positive int (got {SEED!r})")

    # SEEDS ships to the box as run_config.json's seed loop -- a malformed or
    # duplicate-bearing list would silently train fewer adapters than the
    # campaign claims, so it fails here rather than on the box.
    if not isinstance(SEEDS, (list, tuple)) or not SEEDS:
        problems.append(f"SEEDS must be a non-empty list of ints (got {SEEDS!r})")
    else:
        bad = [s for s in SEEDS if not isinstance(s, int) or isinstance(s, bool) or s <= 0]
        if bad:
            problems.append(f"every SEEDS entry must be a positive int (got {bad!r})")
        if len(set(SEEDS)) != len(SEEDS):
            problems.append(f"SEEDS contains duplicates (got {SEEDS!r})")

    # --- pins ----------------------------------------------------------------------
    if not isinstance(EXPECTED_CORPUS_SHA256, str) or not _SHA256_RE.match(EXPECTED_CORPUS_SHA256):
        problems.append(
            f"EXPECTED_CORPUS_SHA256 must be 64 lowercase hex chars (got {EXPECTED_CORPUS_SHA256!r})"
        )

    if not isinstance(EXPECTED_SPLIT_SHA256, str) or not _SHA256_RE.match(EXPECTED_SPLIT_SHA256):
        problems.append(
            f"EXPECTED_SPLIT_SHA256 must be 64 lowercase hex chars (got {EXPECTED_SPLIT_SHA256!r})"
        )

    if not isinstance(EXPECTED_SPLIT_SHA256_16, str) or not _SHA16_RE.match(EXPECTED_SPLIT_SHA256_16):
        problems.append(
            f"EXPECTED_SPLIT_SHA256_16 must be 16 hex chars (got {EXPECTED_SPLIT_SHA256_16!r})"
        )

    # Shape/type only -- NOT a hardcoded "must be 7": that would be a magic
    # number this function can't verify is still correct, and would fight
    # tests that legitimately monkeypatch a smaller/empty roster. The real
    # "is this exactly the split's declared backfill set" check is disk-
    # driven, in build_dataset.assert_backfill_mapping_complete (runs at
    # build() time, called from main() after this validation passes).
    if not isinstance(BACKFILL_TRACE_SOURCES, dict):
        problems.append(
            f"BACKFILL_TRACE_SOURCES must be a dict (got {type(BACKFILL_TRACE_SOURCES).__name__})"
        )
    else:
        bad_keys = [k for k in BACKFILL_TRACE_SOURCES if not isinstance(k, str) or not k]
        bad_values = [v for v in BACKFILL_TRACE_SOURCES.values() if not isinstance(v, str) or not v]
        if bad_keys or bad_values:
            problems.append(
                "BACKFILL_TRACE_SOURCES keys and values must all be non-empty str "
                f"(bad keys={bad_keys!r}, bad values={bad_values!r})"
            )

    if not isinstance(PASS_AT_K_SYSTEM_PROMPT, str) or not PASS_AT_K_SYSTEM_PROMPT:
        problems.append(
            f"PASS_AT_K_SYSTEM_PROMPT must be a non-empty str (got {PASS_AT_K_SYSTEM_PROMPT!r})"
        )

    if not isinstance(PASS_AT_K_NO_THINK_SUFFIX, str) or not PASS_AT_K_NO_THINK_SUFFIX:
        problems.append(
            f"PASS_AT_K_NO_THINK_SUFFIX must be a non-empty str (got {PASS_AT_K_NO_THINK_SUFFIX!r})"
        )
    elif not PASS_AT_K_NO_THINK_SUFFIX.startswith(" "):
        problems.append(
            "PASS_AT_K_NO_THINK_SUFFIX must start with a space -- the leading "
            f"space is load-bearing (README D4 wire-format byte-identity), got "
            f"{PASS_AT_K_NO_THINK_SUFFIX!r}"
        )

    if problems:
        raise ConfigError(
            f"{len(problems)} configuration problem(s) in loratrain/config.py:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


# ============================================================================
# V3 (proof-hint arm) constants -- ADDITIVE ONLY (docs/lora_v3_proofhint_
# execution_skeleton.md), appended 2026-07-31. Consumed exclusively by
# loratrain.v3 (src/loratrain/src/loratrain/v3.py); nothing above this
# banner is touched, reverted, or reformatted by this addition, and no
# existing module imports v3 (isolation constraint, skeleton section 0 --
# see also v3.py's own module docstring). validate_config() above is NOT
# extended to cover these knobs -- see v3.validate_v3_config() in the new
# module, kept separate rather than edited into this file's existing
# function body.
# ============================================================================

# Skeleton P1: "sample the base model ... up to k_regen=6 tries; keep the
# FIRST candidate whose endpoint verifies". Also echoed into the regen
# bundle's manifest so a box-side P2 consumer knows the budget it is
# generating against.
V3_K_REGEN = 6

# Frozen copies of the pass@k generation params the regen bundle's hint
# prompts are meant to be sampled under at P2 (box regeneration, out of
# this module's scope -- P1 only builds prompts and records these in the
# bundle manifest; it makes no network/model call itself). Same
# duplication rationale as PASS_AT_K_SYSTEM_PROMPT / PASS_AT_K_NO_THINK_
# SUFFIX above: loratrain is stdlib-only with zero icepick imports, so
# these are pinned copies -- of icepick.processing.pass_at_k.config.
# PassAtKConfig's temperature default (0.7) and the production pass@k
# completion budget (RUNBOOK section 0.4 / verify_dequant_parity.
# DEFAULT_MAX_NEW_TOKENS = 2048) -- never a live read of that config.
V3_REGEN_TEMPERATURE = 0.7
V3_REGEN_MAX_TOKENS = 2048

# The hint-block delimiter make-regen-bundle inserts between the bare
# statement and the paper-derived solution text (v3.py's
# build_regen_prompt). Load-bearing as an exact literal: v3.py's hint-
# never-in-training-prompt guard greps this exact string against every
# FINAL training prompt and hard-fails on any hit -- the guard standing
# between "hint at generation time only" (v3's whole premise) and a
# silent leak into the serve-time prompt shape. Placement note
# (orchestrator ruling, 2026-07-31, surfaced to Nicky in the window
# report): the pinned
# PASS_AT_K_NO_THINK_SUFFIX stays TERMINAL on the regen user turn -- i.e.
# AFTER this marker and the solution text, not sandwiched between the
# statement and this marker -- see v3.py's build_regen_prompt docstring
# for the full rationale (a literal reading of the skeleton's prose would
# have stranded the suffix mid-prompt, which the Qwen soft-switch
# convention does not tolerate).
V3_HINT_MARKER = "\n\nReference solution (from the source paper):\n"

# R3 (release checkbox, skeleton section 1): curriculum mix. As written
# ("60/40 collapse/band hinted rows + 25% unhinted anchor rows") the
# arithmetic is ambiguous -- v3.py implements and documents the reading
# "final dataset = 75% hinted (60/40 collapse/band split by source-record
# tier) + 25% anchor", the orchestrator's 2026-07-31 arithmetic ruling
# (surfaced to Nicky in the window report). Three independent knobs (not folded into one
# expression) so a future re-ruling is a value edit here, not a code
# change; the two "derived" ones are kept as their own named constants
# purely for readability at call sites, never independently edited.
V3_ANCHOR_FRACTION = 0.0  # RE-RULED 0.25 -> 0.0 (Nicky, 2026-08-01, v3-full-run window): the 08-01 overbuild composition (468 hinted rows, skeleton section 2) supersedes the 07-31 blend design ENTIRELY -- no unhinted v2-cap1 anchor. Decisive extra reason: v2-cap1 rows are old-train records, some of which are EVAL-side under the new proof-split, so anchoring with them would train on eval statements. anchor_count computes to 0; the draw is skipped.
V3_HINTED_FRACTION = 1.0 - V3_ANCHOR_FRACTION  # 1.0 -- derived
V3_HINTED_COLLAPSE_FRACTION = 0.60
V3_HINTED_BAND_FRACTION = 1.0 - V3_HINTED_COLLAPSE_FRACTION  # 0.40 -- derived

# Deterministic anchor draw (R3): src/loratrain/data/v2/cap1/sft_train.jsonl
# rows are ranked by sha256(f"{V3_ANCHOR_SEED_STRING}:{uid}") ascending --
# same idiom as build_dataset._selection_rank / config.SEED -- so a
# rebuild against unchanged v2/cap1 data reproduces byte-identical anchor
# selection. Pinned like SEED; not a per-run knob.
V3_ANCHOR_SEED_STRING = "lora-v3-proofhint:anchor-draw:v1"

# R5 (release checkbox): hint-insufficient fallback (a record that
# verifies 0/k even WITH the proof in context). "drop_and_census" is the
# default (keeps everything on-policy); the skeleton's other listed
# option ("include solution_text verbatim" -- off-policy contamination)
# is deliberately NOT implemented by this builder -- out of scope until
# Nicky rules it in.
V3_HINT_INSUFFICIENT_POLICY = "drop_and_census"
VALID_V3_HINT_INSUFFICIENT_POLICIES = ("drop_and_census",)

# Tier-resolution fallback pool for the R3 60/40 collapse/band split
# (orchestrator ruling, 2026-07-31, from a completed inventory
# cross-check, surfaced to Nicky in the window report): a hinted row's tier = "band" iff its uid is in
# band_corpus.jsonl (authoritative, takes precedence unconditionally);
# ONLY for uids absent from band_corpus does v3.py fall back to this pool,
# reading the NESTED `pass_at_k_results.label` field (NOT a flat
# top-level `label` -- every one of this file's ~2021 rows has that flat
# key absent; a flat lookup silently returns None for all of them, which
# is exactly the trap v3.py's tests pin against). 34 known records carry
# stale collapse/misdirection labels here after the 2026-07-15
# gguf_rescore fold rebanded them in band_corpus without refreshing this
# pool -- band_corpus-first precedence is what makes those resolve
# correctly. No sha/row-count pin here (none was given, unlike the
# corpus/split pins above) -- v3.py records this file's live sha in its
# manifest rather than pre-pinning an unverified number.
V3_WELLPOSED_POOL_PATH = REPO_ROOT / "out/corpus_pde625/wellposed_all_with_passk.json"

# --- v3-fullrun split rebuild (2026-08-01, docs/v3_full_run_skeleton.md P2)
# -----------------------------------------------------------------------
# RETIRE-NOTE: EXPECTED_SPLIT_SHA256 / EXPECTED_SPLIT_SHA256_16 /
# EVAL_PAPER_SPLIT_PATH above (lines ~81-82, ~269) are now VOID for v3 --
# Nicky's 2026-08-01 ruling retired the old 200-train/100-holdout
# band-vs-band-holdout carve-out entirely, in favor of a proof-bearing
# (train) / proofless (eval) rule over the full 921-record
# band+collapse+misdirection universe. There is no holdout concept in the
# new split at all (see v3.py's load_split_uid_sets / assert_train_split_
# only, whose holdout branch is retired accordingly -- the unknown-uid
# hard-fail is kept and strengthened: it is now the ONLY offender class).
# The old constants above are left byte-identical, per instruction --
# historical (pre-v3) arms, build_dataset.py, and upload_guard.py are
# still pinned to them and must keep working unchanged.
V3_SPLIT_PATH = REPO_ROOT / "evalharness/data/corpus_split_v3_proofsplit_20260801.json"

# Frozen 2026-08-01 (P2 split-build session). 921-record 3-tier universe
# (band 317 / collapse 405 / misdirection 199) classified proof-bearing
# (train_side, 187/217/87) vs proofless (eval_pool); train_allocated_uids
# take all 187 band + 87 misdirection + a ranked 194-of-217 collapse
# (sha256("v3-fullrun:collapse-alloc:v1:"+uid) ascending); eval_set_uids
# take all eligible proofless band, a ranked 97 collapse, a ranked 96
# misdirection, each AFTER excluding (a) live ungradeable-by-name records
# and (b) any proofless record whose paper also hosts any train-side
# record (paper-level disjointness -- the load-bearing guard; uid-level
# alone is not sufficient). See the split artifact's own "composition" and
# "notes" blocks for the full reconciliation receipts, including why the
# achieved eval total (286) fell short of the original 322 target (~100%
# paper-conflict, not defect) and the band_corpus-precedence / verifier-fix
# judgment calls made while building it.
V3_EXPECTED_SPLIT_SHA256 = "69735899efe9270e175b54cb39c11f6aed0f245524dd321a930fcdee8893761d"
V3_EXPECTED_SPLIT_SHA256_16 = V3_EXPECTED_SPLIT_SHA256[:16]  # DERIVED -- same convention as EXPECTED_SPLIT_SHA256_16 above; V3_EXPECTED_SPLIT_SHA256 is the single source of truth.

# Manifest-provenance tolerance (2026-08-01, orchestrator decision -- see
# v3.py's assert_manifest_split_pin docstring for the full reasoning): a
# solutions_v3.jsonl manifest published by proof-import BEFORE this split
# existed recorded the OLD split's sha16 (EXPECTED_SPLIT_SHA256_16) under
# input_shas.split -- those rows are still valid, already-verified
# proof-bearing records, and forcing a re-publish under a corrected
# manifest is unneeded churn. Consulted ONLY at the manifest-pin
# provenance step, in ADDITION to whatever sha16 the caller is currently
# pinned to; it never weakens the uid-membership guard
# (assert_train_split_only), which always checks the LIVE split's
# train_side_uids regardless of which sha16 a manifest recorded here.
V3_ACCEPTED_MANIFEST_SPLIT_SHA16S = (EXPECTED_SPLIT_SHA256_16, V3_EXPECTED_SPLIT_SHA256_16)

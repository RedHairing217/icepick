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

TRAIN_SERVER_IP = "69.30.85.138"   # <-- EDIT HERE: the pod's public IP -- the ssh/scp target and NOTHING else (single source of truth)
TRAIN_SERVER_PORT = 8000        # M4-LOCAL end of the section 6 status tunnel; edit only if local 8000 is occupied
TRAIN_SERVER_SSH_PORT = 22092   # <-- EDIT HERE when provisioning: the pod's external TCP port mapped to container 22
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

# --- Hyperparameters ---------------------------------------------------------
# Conventional starting points, not tuned -- W3 review (README "Open items").
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
EPOCHS = 3
MICRO_BATCH_SIZE = 4
MAX_SEQ_LEN = 4096

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
SFT_DATASET_PATH = DATA_DIR / "sft_train.jsonl"
DATASET_MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"
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
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            problems.append(f"{name} must be a positive int (got {value!r})")

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

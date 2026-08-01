"""v3 -- proof-as-hint self-regeneration dataset builder (skeleton P1 only).

Mission slug ``lora-v3-proofhint`` (docs/lora_v3_proofhint_execution_skeleton.
md). Hypothesis: v1/v2 saturated because own-rollout training carries no
information the model lacks; v3 imports the missing information (paper
proofs, mined by the separate ``proof-import`` mission into
``solutions_v3.jsonl``) but trains on the model's OWN re-derivation of each
proof -- hint at data-generation time, on-policy tokens at training time.
**The hint must never appear in the training prompt** (serve time has no
hint, so train/serve prompts must match byte-for-byte).

ISOLATION (skeleton section 0, binding): this is the ONLY new module. It
imports FROM ``loratrain.build_dataset`` / ``loratrain.config`` (guards,
pins, wire-format constants) but no existing loratrain module imports this
one, and no pre-existing file is edited except an additive, clearly-marked
constants section appended to the bottom of ``config.py``. See
``tests/test_v3.py``'s isolation tests for the enforced half of this
contract.

CLI (run from ``src/loratrain`` with ``PYTHONPATH=src``)::

    python -m loratrain.v3 make-regen-bundle --solutions <path> \\
        --bundle-dir <path>
    python -m loratrain.v3 build-dataset --bundle-dir <path> \\
        --solutions <path> --rollouts <path> --output-dir <path>

Upstream input contract (frozen 2026-07-31,
``out/proof_import_20260731T185338Z/CONTRACTS.md``; this module does NOT
build it, does not glob for it, and hard-refuses when it is absent)::

    solutions_v3.jsonl   one row per verified, train-split-only record:
        {uid, question, proof_raw_sha, solution_text, answer,
         provenance: {arxiv_id, match_method, match_confidence,
                       sonnet_cache_key, verified: true}}
        `question` == the record's wire statement. 100% endpoint-verified.
    manifest.json         sibling of solutions_v3.jsonl: INPUT shas,
                           censuses, spend. PINNED against the first real
                           publish (out/proof_import_20260731T185338Z,
                           2026-07-31): the real manifest records input
                           shas only -- ``input_shas.split`` etc., as
                           16-hex PREFIXES -- and does NOT record the
                           published file's own sha; the lane records
                           output shas via stem-named sidecars instead
                           (its ``bundle.sha256`` idiom). This module
                           looks up the solutions sha under
                           ``solutions_v3.sha256`` / ``solutions_sha256``
                           / ``sha256`` / ``input_shas.solutions_v3``,
                           falling back to a ``solutions_v3.sha256`` or
                           ``solutions_v3.jsonl.sha256`` sidecar beside
                           the file; the split sha under ``split.sha256``
                           / ``split_sha256`` / ``split_sha16`` /
                           ``input_shas.split``. Recorded values may be
                           >=16-hex prefixes of the full sha256; anything
                           shorter, non-hex, or mismatched hard-refuses.

Two subcommands
================

``make-regen-bundle`` (local, trivial-CPU, skeleton P1 first half): reads
solutions_v3.jsonl + its manifest, re-verifies the sha-chain, re-asserts
every uid is a TRAIN uid of the pinned split (config.V3_EXPECTED_SPLIT_
SHA256) -- any uid absent from the split's ``train_side_uids`` hard-fails
naming it (split-rebuild-2026-08-01.md: no holdout concept exists in this
split -- see ``load_split_uid_sets`` / ``assert_train_split_only``) -- and
emits a box-shippable bundle: one ``{uid, regen_prompt}`` row per record,
where ``regen_prompt`` is the pass@k wire prompt WITH a paper-derived hint
appended (see ``build_regen_prompt``). No answer-key field, ever, in the
bundle.

**Interpreted deviation (orchestrator ruling, 2026-07-31, surfaced to
Nicky in the window report; recorded here AND in every bundle manifest).** The skeleton's
prose ("wire-format question + hint") reads as
``statement + suffix + hint``, which stakes the pinned
``config.PASS_AT_K_NO_THINK_SUFFIX`` (" /no_think") in the MIDDLE of the
user turn. The suffix is a Qwen soft-switch token and must stay TERMINAL on
the user turn to reliably take effect, so the actual construction is::

    statement + V3_HINT_MARKER + solution_text + PASS_AT_K_NO_THINK_SUFFIX

The hint-never-in-training-prompt guard (``assert_hint_never_in_training_
prompt``) keys on the exact marker string ``config.V3_HINT_MARKER``
("\\n\\nReference solution (from the source paper):\\n") appearing in ZERO
published training prompts.

``build-dataset`` (local, skeleton P1 second half): takes the bundle + a
``--rollouts`` file (box P2 output, out of this module's scope: jsonl of
``{uid, sample_idx, output}`` -- generation order, up to ``k_regen``
samples per uid). For each uid, tries samples in ascending ``sample_idx``
order (capped to ``< config.V3_K_REGEN``) and keeps the FIRST whose
extracted endpoint verifies against the uid's PINNED answer (read from
solutions_v3.jsonl, never the bundle) -- mirroring ``out/passk8_sweep/
grade.py``'s exact verify chain::

    tier, truth = verifier.classify(answer)
    candidate = scoring.extract_candidate(scoring.strip_think(output or ""))
    verified = candidate is not None and verifier.verify(candidate, truth, tier)

``strip_think`` is for VERIFICATION ONLY -- the STORED completion is the
model's ``output`` VERBATIM (never normalized/trimmed/reflowed; v2
precedent: stored completions already begin ``"\\n\\n"`` with no think
block, because the generation path omits it before the rollout's
``output`` field is written). cap1: at most one kept trace per uid, ever,
across the WHOLE final dataset (hinted rows AND anchor rows draw from
disjoint uid sets -- see ``draw_anchor_rows``). A uid with zero verified
samples is "hint_insufficient" -> R5 default (``config.
V3_HINT_INSUFFICIENT_POLICY == "drop_and_census"``): dropped, counted, uids
named in the manifest, never silent. A uid absent from the rollouts file
entirely is counted separately (``missing_from_rollouts``) -- a different
failure class (P2 never produced output at all) from "produced output that
never verified".

The verifier chain is injected as ``verify_fn(output, answer) -> bool``
(default: ``default_verify_fn``, which lazily imports
``icepick.processing.pass_at_k`` -- so importing this module, or running
its test suite, never requires icepick's sympy/antlr4 stack to be
importable). Tests inject a fake ``verify_fn`` and need no icepick import
at all.

Blend (R3, skeleton section 1): ambiguous as written ("60/40 collapse/band
hinted rows + 25% unhinted v2-cap1 band rows"). Implemented reading
(orchestrator arithmetic ruling, 2026-07-31): **final dataset = 75% hinted
(config.V3_HINTED_FRACTION) + 25% anchor (config.V3_ANCHOR_FRACTION)**,
where the hinted 75% further splits 60/40 collapse/band
(config.V3_HINTED_COLLAPSE_FRACTION / V3_HINTED_BAND_FRACTION) BY
SOURCE-RECORD TIER -- observed and recorded, never enforced by dropping a
verified hinted row: today's only exercised proof-import target set is
"train-split BAND records" (R2 default), which can legitimately yield zero
collapse-tier records, and force-capping the majority tier to hit a fixed
ratio would mean silently discarding hard-won verified traces to preserve
an input composition this module does not control. ``anchor_count =
round(hinted_count * V3_ANCHOR_FRACTION / V3_HINTED_FRACTION)`` (i.e.
``hinted_count / 3``) unhinted rows are drawn DETERMINISTICALLY from
``src/loratrain/data/v2/cap1/sft_train.jsonl`` (ranked by
``sha256(f"{config.V3_ANCHOR_SEED_STRING}:{uid}")`` ascending -- same idiom
as ``build_dataset._selection_rank``), EXCLUDING any uid already used by a
hinted row (global cap1). Every number here is a named ``config.V3_*``
constant, never a magic literal, and every one of them is echoed into the
dataset manifest's ``blend`` block alongside the ACHIEVED (not just
nominal) composition.

Source-tier resolution for the R3 60/40 split (orchestrator ruling,
2026-07-31, from a completed inventory cross-check -- ``resolve_
hinted_tier``): a hinted uid's tier = "band" IFF the uid is in
``band_corpus.jsonl`` -- membership there is authoritative and takes
UNCONDITIONAL precedence. Only for a uid absent from ``band_corpus.jsonl``
does resolution fall back to ``config.V3_WELLPOSED_POOL_PATH``
(``wellposed_all_with_passk.json``, ~2021 records), reading the label at
the NESTED path ``pass_at_k_results.label`` -- deliberately NOT a flat
top-level ``label`` key, which is absent on every row in that file and
would silently resolve to ``None`` for all of them (the exact trap
``tests/test_v3.py`` pins a regression test against). ``"collapse"`` and
``"misdirection"`` pool labels both map to the collapse bucket (the
campaign treats them as one tier); ``"band"`` maps to band. Any other
resolved value (uid in neither source, pool label absent, or a label like
``"solved"`` -- e.g. 5 of the 7 GGUF-7/8 backfill uids, which exist ONLY in
the pool) belongs to NEITHER 60/40 bucket: excluded from the blend
entirely (never added to ``hinted_rows``, never crashes), censused in the
dataset manifest's ``censuses.excluded_offtier`` block (count, uids,
resolved label per uid). band_corpus precedence exists because 34 known
records carry STALE collapse/misdirection labels in the pool -- rebanded
into band_corpus.jsonl by the 2026-07-15 ``gguf_rescore`` fold without the
pool itself ever being refreshed.

Restartability: both subcommands always run their full guard + build
pipeline (cheap, CPU-only, small N -- no reason to special-case a resume
path at the cost of extra control-flow complexity); only the FINAL publish
step is conditional. If the target directory already holds a COMPLETE
prior publish (its own manifest parses, names the right stage, and every
file it lists on disk matches the sha it recorded) whose recorded
``input_signature`` (a sha256 over every input sha + the relevant
``config.V3_*``/wire-format pins) matches this run's, the existing
manifest is returned unchanged (idempotent no-op -- true restart-and-
resume). If it exists but the signature differs, this run refuses
(``PublishConflictError``, naming the directory) UNLESS ``--force-new-dir``
is given, in which case it publishes under the first available
``<dir>__N`` sibling instead -- NEVER overwriting or touching the original
directory. This is not a guard bypass (no sha pin, uid-membership check, or
leakage guard is ever skippable by any flag in this module); it only
resolves an output-location collision.

Surfaced assumptions / deviations (see also the final report this module's
author filed alongside it):
  1. no_think-suffix placement in the regen prompt (see above) --
     orchestrator-ruled (2026-07-31), not this module's own call.
  2. R3 arithmetic reading (75/25 split of hinted/anchor, 60/40 nominal
     within hinted) -- orchestrator-ruled.
  3. Source-tier resolution (band_corpus-first, wellposed-pool-nested-
     label fallback, off-tier exclusion) -- orchestrator-ruled (2026-07-31,
     from a completed inventory cross-check), not this module's own call; see
     ``resolve_hinted_tier``'s docstring.
  4. The manifest.json field-name tolerance list (above) -- originally an
     inference from CONTRACTS.md's prose (the lane had not published yet);
     VERIFIED AND PINNED 2026-07-31 against the first real publish:
     ``input_shas.*`` 16-hex prefixes, no published-file sha in the
     manifest, stem-named sha sidecar for outputs instead.
  5. Restartability's exact ``--force-new-dir`` semantics -- this module's
     own design (the brief named the flag but not its precise behavior);
     documented above and in ``_resolve_publish_dir``.
  6. Split rebuild (2026-08-01, docs/v3_full_run_skeleton.md P2,
     split-rebuild-2026-08-01.md, Nicky's ruling): the old 200-train/
     100-holdout split is VOID. The new split (config.V3_SPLIT_PATH /
     V3_EXPECTED_SPLIT_SHA256) has NO holdout -- proof-bearing records are
     train_side, proofless records are eval_pool. ``load_split_uid_sets``
     now returns only a ``train_uids`` set (read from the split's
     ``train_side_uids`` key); ``assert_train_split_only``'s holdout branch
     (``LeakageError``) is retired, and its unknown-uid branch
     (``UnknownUidError``) is now the ONLY offender class -- strengthened,
     not weakened: a former-holdout uid, a proofless/eval uid, and a flat
     typo all hit the exact same named refusal. ``assert_manifest_split_
     pin`` additionally tolerates the OLD split's sha16
     (config.EXPECTED_SPLIT_SHA256_16) at the manifest-provenance step ONLY
     (config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S) -- solutions rows published
     before the rebuild recorded the old sha and are still valid, verified,
     proof-bearing records; the uid-membership guard is never relaxed by
     this tolerance, it always checks the LIVE split's train_side_uids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from loratrain import build_dataset, config

# ============================================================================
# Constants
# ============================================================================

BUNDLE_FILENAME = "regen_bundle.jsonl"
BUNDLE_MANIFEST_FILENAME = "regen_bundle_manifest.json"
DATASET_FILENAME = "sft_train.jsonl"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"

STAGE_REGEN_BUNDLE = "loratrain_v3_regen_bundle"
STAGE_BUILD_DATASET = "loratrain_v3_build_dataset"

REQUIRED_SOLUTION_KEYS = ("uid", "question", "proof_raw_sha", "solution_text", "answer")
REQUIRED_BUNDLE_ROW_KEYS = ("uid", "regen_prompt")
REQUIRED_ROLLOUT_KEYS = ("uid", "sample_idx", "output")

REGEN_BUNDLE_GUARD_STEPS = (
    "assert_solutions_and_manifest_exist",
    "assert_solutions_sha_chain",
    "assert_manifest_split_pin",
    "assert_split_pinned[disk,reused=build_dataset.assert_split_pinned]",
    "load_solutions_rows",
    "assert_train_split_only[unknown_hard_fail]",
    "assert_no_statement_leakage_exact[vs_eval_set.jsonl]",
    "build_regen_prompts[hint_marker+solution_text,suffix_terminal]",
    "assert_bundle_rows_well_formed[uid+regen_prompt_only,no_answer_key]",
    "resolve_publish_dir[restartability]",
    "atomic_publish[tmp->reverify->rename]",
)

BUILD_DATASET_GUARD_STEPS = (
    "assert_solutions_and_manifest_exist",
    "assert_solutions_sha_chain",
    "assert_manifest_split_pin",
    "assert_split_pinned[disk,reused=build_dataset.assert_split_pinned]",
    "load_solutions_rows",
    "assert_train_split_only[unknown_hard_fail]",
    "load_bundle[sha_verified]",
    "assert_bundle_solutions_sha_chain_link",
    "assert_bundle_uid_set_equals_solutions_train_uids",
    "assert_corpus_pinned[reused=build_dataset.assert_corpus_pinned]",
    "load_rollouts",
    "select_first_verified_per_uid[ascending_sample_idx,cap_k_regen]",
    "census_hint_insufficient_and_missing_from_rollouts",
    "resolve_hinted_tier[band_corpus_first,wellposed_pool_nested_label_fallback]",
    "census_excluded_offtier",
    "draw_anchor_rows[deterministic_seeded_rank,excludes_hinted_uids]",
    "assert_prompt_completion_wellformed[reused=build_dataset]",
    "assert_hint_never_in_training_prompt",
    "assert_no_statement_leakage_in_prompts[substring,vs_eval_set.jsonl]",
    "loss_mass_census",
    "resolve_publish_dir[restartability]",
    "atomic_publish[tmp->reverify->rename]",
)


# ============================================================================
# Exceptions
# ============================================================================


class SolutionsIntegrityError(RuntimeError):
    """solutions_v3.jsonl or its manifest.json is missing, malformed, or the
    sha-chain between them (or against the pinned split) does not check out.
    """


class UnknownUidError(RuntimeError):
    """A solutions-file uid is not in the pinned split's ``train_side_uids``
    -- solutions_v3.jsonl and the split have desynced.

    RETIRED 2026-08-01 (split rebuild): the split has no holdout concept
    any more, so this is now the ONLY offender class ``assert_train_split_
    only`` raises -- a former-holdout uid, a proofless/eval_pool uid, and a
    flat typo all hit this exact refusal, named. Kept as its own exception
    type (rather than folded into ``SolutionsIntegrityError``) because the
    failure mode is still distinct: an unknown uid is not proven to be
    eval-radioactive, it is proven to be UNTRACKED by the split this module
    is pinned to.
    """


class BundleIntegrityError(RuntimeError):
    """The regen bundle (regen_bundle.jsonl + its manifest) is missing,
    malformed, or its sha-chain back to the solutions file is broken.
    """


class RolloutIntegrityError(RuntimeError):
    """The box regen rollouts file is missing or malformed."""


class BlendError(RuntimeError):
    """The R3 blend cannot be assembled as specified (e.g. not enough
    v2/cap1 anchor rows remain after excluding hinted uids, or the final
    dataset would be empty).
    """


class PublishConflictError(RuntimeError):
    """The requested output directory already holds a completed publish
    built from DIFFERENT inputs, and ``--force-new-dir`` was not given.
    """


# ============================================================================
# Small helpers
# ============================================================================


def _lookup_first(d: dict, *dotted_paths: str):
    """First present value among dotted-key lookup paths in ``d``, else None."""
    for path in dotted_paths:
        cur = d
        ok = True
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return None


def _row_uid(row: dict):
    return (row.get("provenance") or {}).get("uid")


def _shown(items, limit=5) -> str:
    items = list(items)
    shown = ", ".join(str(i) for i in items[:limit])
    more = "" if len(items) <= limit else f" (+{len(items) - limit} more)"
    return shown + more


def _anchor_rank(uid: str) -> str:
    return hashlib.sha256(f"{config.V3_ANCHOR_SEED_STRING}:{uid}".encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl_bytes(rows) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


# ============================================================================
# Solutions / manifest / split loaders and guards
# ============================================================================


def load_solutions_manifest(manifest_path: Path) -> dict:
    """Parse the proof-import manifest.json sibling of solutions_v3.jsonl."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise SolutionsIntegrityError(
            f"{manifest_path} not found -- proof-import's manifest.json must "
            "sit beside solutions_v3.jsonl (docs/proof_import_execution_"
            "skeleton.md P5); refusing to trust an unmanifested solutions file."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SolutionsIntegrityError(f"{manifest_path}: invalid JSON ({exc})") from exc
    if not isinstance(manifest, dict):
        raise SolutionsIntegrityError(f"{manifest_path}: top-level JSON is not an object")
    return manifest


# Manifest key paths / sidecar names for the solutions sha -- PINNED against
# the first real proof-import publish (out/proof_import_20260731T185338Z,
# 2026-07-31): its manifest.json records INPUT shas only (16-hex prefixes,
# e.g. input_shas.split) and never the published file's own sha, which the
# lane instead records via a stem-named sha sidecar (its bundle.sha256
# idiom: "<hex>  <filename>").
MANIFEST_SOLUTIONS_SHA_PATHS = (
    "solutions_v3.sha256", "solutions_sha256", "sha256", "input_shas.solutions_v3",
)
SOLUTIONS_SHA_SIDECAR_NAMES = ("solutions_v3.sha256", "solutions_v3.jsonl.sha256")


def _recorded_sha_matches(recorded, recomputed: str) -> bool:
    """True iff ``recorded`` is a >=16-char hex PREFIX of ``recomputed``.
    The real publish records 16-hex prefixes; fixtures record full 64-hex,
    which is trivially its own prefix. Anything shorter than 16 hex chars
    is too weak to pin and is rejected.
    """
    rec = str(recorded).strip().lower()
    return (
        len(rec) >= 16
        and all(c in "0123456789abcdef" for c in rec)
        and recomputed.startswith(rec)
    )


def assert_solutions_sha_chain(solutions_path: Path, manifest_path: Path, manifest: dict) -> str:
    """Hard-refuse unless ``solutions_path`` exists and its recomputed
    sha256 matches what the manifest -- or, failing that, a sha sidecar
    beside the solutions file -- records for it (>=16-hex-prefix
    semantics, ``_recorded_sha_matches``). Returns the recomputed full
    sha256 on success.
    """
    solutions_path = Path(solutions_path)
    if not solutions_path.exists():
        raise SolutionsIntegrityError(
            f"{solutions_path} not found -- proof-import has not published "
            "solutions_v3.jsonl yet (docs/proof_import_execution_skeleton.md: "
            "'Hard dependency'). This builder refuses to run without it."
        )
    recomputed = build_dataset.sha256_file(solutions_path)
    recorded = _lookup_first(manifest, *MANIFEST_SOLUTIONS_SHA_PATHS)
    source = manifest_path
    if recorded is None:
        for name in SOLUTIONS_SHA_SIDECAR_NAMES:
            cand = solutions_path.parent / name
            if cand.exists():
                tokens = cand.read_text(encoding="utf-8").split()
                if tokens:
                    recorded = tokens[0]
                    source = cand
                    break
    if recorded is None:
        raise SolutionsIntegrityError(
            f"{manifest_path}: no solutions sha recorded (looked for "
            + " / ".join(MANIFEST_SOLUTIONS_SHA_PATHS)
            + ") and no sha sidecar ("
            + " / ".join(SOLUTIONS_SHA_SIDECAR_NAMES)
            + ") beside the solutions file -- refusing to trust an "
            "unverifiable solutions file."
        )
    if not _recorded_sha_matches(recorded, recomputed):
        raise SolutionsIntegrityError(
            f"{solutions_path} sha256={recomputed[:16]} != recorded "
            f"{str(recorded)[:16]} ({source}) -- recorded value must be a "
            ">=16-hex prefix of the recomputed sha; the solutions file "
            "moved, the record is stale, or the record is too short to pin. "
            "Sha-chain broken, refusing."
        )
    return recomputed


def assert_manifest_split_pin(manifest: dict, manifest_path: Path, expected_split_sha16: str) -> None:
    """Hard-refuse unless the manifest's recorded split sha (16-hex,
    tolerant lookup) matches the pinned split's sha16 -- OR is a listed
    provenance-era pin (see below).

    Provenance-era tolerance (2026-08-01, split rebuild, orchestrator
    decision): solutions_v3.jsonl rows published by proof-import BEFORE
    the new split existed recorded the OLD split's sha16
    (config.EXPECTED_SPLIT_SHA256_16, the now-VOID 200-train/100-holdout
    split) under ``input_shas.split``. Those rows are still valid,
    already-verified, 100%-endpoint-verified proof-bearing records --
    forcing a re-publish under a corrected manifest just to update this
    one field would be unneeded churn. This check therefore ALSO accepts
    any sha16 explicitly listed in ``config.V3_ACCEPTED_MANIFEST_SPLIT_
    SHA16S`` (today: the old split's sha16 and the current one), in
    ADDITION to whatever ``expected_split_sha16`` the caller passed.

    This tolerance is scoped to THIS provenance check ONLY. The actual
    membership guard (``assert_train_split_only``) is never relaxed by it
    -- it always checks the LIVE, pinned split's ``train_side_uids``,
    regardless of which sha16 a manifest recorded here. A solutions row
    from the old split era still hard-fails at that later guard if its uid
    is not ALSO a member of the new split's train_side_uids.
    """
    recorded = _lookup_first(manifest, "split.sha256", "split_sha256", "split_sha16", "input_shas.split")
    if recorded is None:
        raise SolutionsIntegrityError(
            f"{manifest_path}: no split sha recorded (looked for split.sha256 "
            "/ split_sha256 / split_sha16 / input_shas.split) -- manifest malformed."
        )
    recorded16 = str(recorded)[:16]
    accepted = {expected_split_sha16} | set(config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S)
    if recorded16 not in accepted:
        raise SolutionsIntegrityError(
            f"{manifest_path}: recorded split sha16={recorded16} not in the "
            f"accepted set {sorted(accepted)!r} (pinned config split sha16="
            f"{expected_split_sha16}, plus any listed provenance-era pins in "
            "config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S) -- proof-import ran "
            "against a split v3 does not recognize at all; refusing."
        )


def load_split_uid_sets(split_path: Path, expected_split_sha256: str) -> set:
    """Pin-check the split (reuses ``build_dataset.assert_split_pinned``)
    then return ``train_uids: set`` read directly off the split JSON's own
    ``train_side_uids`` key.

    RETIRED 2026-08-01 (split rebuild, split-rebuild-2026-08-01.md): this
    used to return ``(train_uids, holdout_uids)``. The new split
    (proof-bearing/train_side vs proofless/eval_pool) has no holdout
    concept at all, so there is nothing to return a second set for.
    Callers that need to prove a solutions row is not eval-radioactive
    rely entirely on ``assert_train_split_only``'s strengthened unknown-uid
    hard-fail: any uid absent from ``train_side_uids`` refuses, named --
    a former-holdout uid, a proofless/eval uid, or a typo all hit it alike.
    """
    split_path = Path(split_path)
    build_dataset.assert_split_pinned(split_path, expected_split_sha256)
    data = json.loads(split_path.read_text(encoding="utf-8"))
    train_uids = set(data.get("train_side_uids") or [])
    if not train_uids:
        raise ValueError(f"{split_path}: 'train_side_uids' is empty -- not a usable split")
    return train_uids


def load_solutions_rows(solutions_path: Path) -> list:
    """Parse solutions_v3.jsonl; hard-refuse on any row missing a required
    non-empty key (see ``REQUIRED_SOLUTION_KEYS``).
    """
    solutions_path = Path(solutions_path)
    rows = []
    with solutions_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SolutionsIntegrityError(f"{solutions_path}:{lineno}: invalid JSON ({exc})") from exc
            missing = [k for k in REQUIRED_SOLUTION_KEYS if not row.get(k)]
            if missing:
                raise SolutionsIntegrityError(
                    f"{solutions_path}:{lineno}: row missing required non-empty "
                    f"key(s) {missing} (uid={row.get('uid')!r})"
                )
            rows.append(row)
    if not rows:
        raise SolutionsIntegrityError(f"{solutions_path} contains no rows -- not a usable solutions file")
    return rows


def assert_train_split_only(rows, train_uids: set, *, uid_key: str = "uid") -> None:
    """Hard-refuse if any row's uid is not in ``train_uids`` (the frozen
    split's ``train_side_uids``), naming every offender. Collects every
    offender before raising (never fail-fast on the first one).

    RETIRED 2026-08-01 (split rebuild, split-rebuild-2026-08-01.md): the
    old holdout-uid branch (``build_dataset.LeakageError``, "HOLDOUT
    uids") is gone -- the new split has no holdout concept, only
    train_side_uids (proof-bearing) vs eval_set_uids (proofless). This
    unknown-uid check is now the ONLY offender class, and it is
    STRENGTHENED, not weakened: a former-holdout uid, a proofless/
    eval-pool uid, and a flat typo all hit this exact same named refusal
    -- there is no longer a softer way to be wrong.
    """
    unknown_offenders = sorted({r[uid_key] for r in rows if r.get(uid_key) not in train_uids})
    if unknown_offenders:
        raise UnknownUidError(
            f"{len(unknown_offenders)} solutions-file uid(s) are not in the "
            f"pinned split's train_side_uids: {_shown(unknown_offenders)} -- "
            "solutions_v3.jsonl and the pinned split (config.V3_EXPECTED_"
            "SPLIT_SHA256) have desynced, or this is a proofless/eval-pool "
            "uid that must never appear in a training solutions file."
        )


# ============================================================================
# Wire prompt builders
# ============================================================================


def build_question_only_prompt(statement: str) -> str:
    """The exact serve-time / training-time user-turn text: statement + the
    pinned no-think suffix, nothing else -- byte-identical in construction
    to ``build_dataset.build_sft_example``'s user content
    (``record["statement"] + config.PASS_AT_K_NO_THINK_SUFFIX``). v3's
    whole premise is that train and serve prompts match; this function is
    deliberately the same expression, not a v3-flavored reimplementation.
    """
    return statement + config.PASS_AT_K_NO_THINK_SUFFIX


def build_regen_prompt(statement: str, solution_text: str) -> str:
    """The generation-time-ONLY hint-augmented user-turn text (P2 input).

    See the module docstring's "Interpreted deviation" note: the no-think
    suffix stays TERMINAL on the user turn (after the hint block), so the
    exact construction is::

        statement + V3_HINT_MARKER + solution_text + PASS_AT_K_NO_THINK_SUFFIX

    Never stored as a training prompt (see ``build_question_only_prompt``);
    only ever shipped as bundle content for the (out-of-scope-here) P2 box
    generation call.
    """
    return statement + config.V3_HINT_MARKER + solution_text + config.PASS_AT_K_NO_THINK_SUFFIX


# ============================================================================
# Statement-leakage guards (replicate build_dataset's idiom -- solutions_v3
# rows carry the full statement under "question", not "statement", and
# final training rows carry it embedded in a prompt string, not as a bare
# field, so neither shape can directly reuse build_dataset.assert_no_cross_
# split_statement_dups / assert_no_cross_uid_statement_dups.)
# ============================================================================


def assert_no_statement_leakage_exact(rows, eval_statements: set, *, statement_key: str = "question") -> None:
    """Bundle-time guard: no solutions-file row's full statement equals a
    full eval_set.jsonl statement (record-level analog of ``build_dataset.
    assert_no_cross_split_statement_dups``).
    """
    offenders = sorted({r.get("uid") for r in rows if r.get(statement_key) in eval_statements})
    if offenders:
        raise build_dataset.LeakageError(
            f"{len(offenders)} solutions-file record(s) share a full statement "
            f"with an eval_set.jsonl record: {_shown(offenders)} -- an eval "
            "problem's text inside the train pipeline is leakage even under a "
            "different uid; refusing to proceed."
        )


def assert_no_statement_leakage_in_prompts(rows, eval_statements: set) -> None:
    """Dataset-time guard: no FINAL training prompt textually CONTAINS a
    full eval_set.jsonl statement as a substring (prompts are
    statement+suffix, not a bare statement, so containment -- not equality
    -- is the correct test here).
    """
    offenders = []
    for row in rows:
        user_content = row["prompt"][1]["content"]
        for statement in eval_statements:
            if statement and statement in user_content:
                offenders.append(_row_uid(row))
                break
    if offenders:
        raise build_dataset.LeakageError(
            f"{len(offenders)} training prompt(s) textually contain a holdout/"
            f"eval_set.jsonl statement: {_shown(sorted(set(offenders)))} -- "
            "refusing to proceed."
        )


def assert_hint_never_in_training_prompt(rows, hint_marker: str) -> None:
    """Hard-refuse if ``hint_marker`` appears in ANY final training
    prompt's user content -- the guard between "hint at generation time
    only" (v3's entire premise) and a silent leak into the serve-time
    prompt shape.
    """
    offenders = [_row_uid(row) for row in rows if hint_marker in row["prompt"][1]["content"]]
    if offenders:
        raise build_dataset.LeakageError(
            f"{len(offenders)} training prompt(s) contain the hint marker "
            f"{hint_marker!r} -- the hint must NEVER reach a stored training "
            "prompt (skeleton P1: 'the hint appears only at generation time "
            f"-- never in the training prompt'). Offending uids: {_shown(offenders)}"
        )


# ============================================================================
# make-regen-bundle
# ============================================================================


def _bundle_input_signature(*, solutions_sha256, split_sha256, eval_set_sha256, k_regen) -> str:
    payload = {
        "solutions_sha256": solutions_sha256,
        "split_sha256": split_sha256,
        "eval_set_sha256": eval_set_sha256,
        "k_regen": k_regen,
        "hint_marker": config.V3_HINT_MARKER,
        "system_prompt": config.PASS_AT_K_SYSTEM_PROMPT,
        "no_think_suffix": config.PASS_AT_K_NO_THINK_SUFFIX,
        "temperature": config.V3_REGEN_TEMPERATURE,
        "max_tokens": config.V3_REGEN_MAX_TOKENS,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def make_regen_bundle(
    *,
    solutions_path: Path,
    manifest_path: Path,
    split_path: Path,
    expected_split_sha256: str,
    eval_set_path: Path,
    bundle_dir: Path,
    k_regen: int,
    force_new_dir: bool = False,
) -> dict:
    """Build the hint-augmented regen prompt bundle (skeleton P1, first half).

    Nothing is written until every guard has passed (build_dataset.py's
    ordering convention). See the module docstring for the full contract.
    """
    solutions_path = Path(solutions_path)
    manifest_path = Path(manifest_path)
    split_path = Path(split_path)
    eval_set_path = Path(eval_set_path)
    bundle_dir = Path(bundle_dir)

    manifest = load_solutions_manifest(manifest_path)
    solutions_sha256 = assert_solutions_sha_chain(solutions_path, manifest_path, manifest)
    assert_manifest_split_pin(manifest, manifest_path, expected_split_sha256[:16])

    train_uids = load_split_uid_sets(split_path, expected_split_sha256)

    solutions_rows = load_solutions_rows(solutions_path)
    assert_train_split_only(solutions_rows, train_uids)

    if not eval_set_path.exists():
        raise SolutionsIntegrityError(f"{eval_set_path} not found -- cannot prove non-leakage")
    eval_statements = build_dataset.load_eval_statements(eval_set_path)
    assert_no_statement_leakage_exact(solutions_rows, eval_statements)
    eval_set_sha256 = build_dataset.sha256_file(eval_set_path)

    bundle_rows = []
    for row in solutions_rows:
        regen_prompt = build_regen_prompt(row["question"], row["solution_text"])
        bundle_rows.append({"uid": row["uid"], "regen_prompt": regen_prompt})

    # Defense in depth: re-check the freshly-built bundle rows themselves
    # (never trust construction alone -- same idiom as build_dataset's
    # "assert on examples, not just on the records that produced them").
    for brow in bundle_rows:
        missing = [k for k in REQUIRED_BUNDLE_ROW_KEYS if not brow.get(k)]
        if missing:
            raise BundleIntegrityError(f"built bundle row missing {missing}: {brow}")
        if "answer" in brow or "solution_text" in brow:
            raise BundleIntegrityError(f"built bundle row carries an answer-key-shaped field: {brow}")
        if config.V3_HINT_MARKER not in brow["regen_prompt"]:
            raise BundleIntegrityError(f"built bundle row for uid={brow['uid']} is missing the hint marker")

    content_bytes = _jsonl_bytes(bundle_rows)
    bundle_sha256 = _sha256_bytes(content_bytes)

    input_signature = _bundle_input_signature(
        solutions_sha256=solutions_sha256,
        split_sha256=expected_split_sha256,
        eval_set_sha256=eval_set_sha256,
        k_regen=k_regen,
    )

    publish_dir, existing = _resolve_publish_dir(
        bundle_dir,
        force_new_dir=force_new_dir,
        manifest_name=BUNDLE_MANIFEST_FILENAME,
        expected_stage=STAGE_REGEN_BUNDLE,
        input_signature=input_signature,
    )
    if existing is not None:
        return existing

    publish_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = publish_dir / BUNDLE_FILENAME
    bundle_tmp_path = publish_dir / (BUNDLE_FILENAME + ".tmp")
    manifest_out_path = publish_dir / BUNDLE_MANIFEST_FILENAME

    bundle_tmp_path.write_bytes(content_bytes)
    # Re-read the WRITTEN bytes and re-verify before publishing under the
    # final name (atomic tmp -> verify -> rename, build_dataset.py's idiom).
    reread_rows = []
    with bundle_tmp_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            reread_rows.append(json.loads(line))
    if len(reread_rows) != len(bundle_rows):
        raise BundleIntegrityError(
            f"post-write verification read {len(reread_rows)} row(s), expected "
            f"{len(bundle_rows)} -- bundle NOT published (tmp left for forensics)."
        )
    for brow in reread_rows:
        if any(k not in brow for k in REQUIRED_BUNDLE_ROW_KEYS) or "answer" in brow or "solution_text" in brow:
            raise BundleIntegrityError(
                f"post-write verification found a malformed row: {brow} -- "
                "bundle NOT published (tmp left for forensics)."
            )
    bundle_tmp_path.replace(bundle_path)

    out_manifest = {
        "stage": STAGE_REGEN_BUNDLE,
        "created": datetime.now(timezone.utc).isoformat(),
        "solutions": {"path": str(solutions_path), "sha256": solutions_sha256, "rows": len(solutions_rows)},
        "solutions_manifest_path": str(manifest_path),
        "split": {
            "path": str(split_path),
            "sha256": expected_split_sha256,
            "n_train_side_uids": len(train_uids),
            "paper_disjointness_note": (
                "train_papers ∩ eval_papers == ∅, asserted as a hard build-time invariant "
                "of the split artifact itself (evalharness/data/corpus_split_v3_"
                "proofsplit_20260801.json) -- paper-level disjointness is the load-bearing "
                "guard there; this module's own runtime guard remains uid-level "
                "(assert_train_split_only). No holdout concept exists in this split "
                "(retired 2026-08-01, split-rebuild-2026-08-01.md)."
            ),
        },
        "eval_set": {"path": str(eval_set_path), "sha256": eval_set_sha256},
        "k_regen": k_regen,
        "generation_params": {
            "temperature": config.V3_REGEN_TEMPERATURE,
            "max_tokens": config.V3_REGEN_MAX_TOKENS,
            "system_prompt": config.PASS_AT_K_SYSTEM_PROMPT,
            "no_think_suffix": config.PASS_AT_K_NO_THINK_SUFFIX,
            "backend_note": "qwen_http on a box (P2) -- out of scope for this module; this manifest only records the params P2 must use.",
        },
        "hint_marker": config.V3_HINT_MARKER,
        "no_think_placement_note": (
            "Interpreted deviation (orchestrator ruling, 2026-07-31): the pinned "
            "no-think suffix is TERMINAL on the regen user turn (after the "
            "hint + solution text), not immediately after the bare statement "
            "-- a literal skeleton-prose reading would strand it mid-prompt. "
            "See v3.py module docstring / build_regen_prompt."
        ),
        "bundle": {"path": str(bundle_path), "sha256": bundle_sha256, "rows": len(bundle_rows)},
        "published_files": {BUNDLE_FILENAME: bundle_sha256},
        "input_signature": input_signature,
        "guards": list(REGEN_BUNDLE_GUARD_STEPS),
        "resumed_from_existing_publish": False,
    }
    manifest_out_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    return out_manifest


# ============================================================================
# build-dataset
# ============================================================================


def load_bundle(bundle_dir: Path) -> tuple:
    """Load + sha-verify a make-regen-bundle publish. Returns
    ``(rows: list[{uid, regen_prompt}], manifest: dict)``.
    """
    bundle_dir = Path(bundle_dir)
    bundle_path = bundle_dir / BUNDLE_FILENAME
    manifest_path = bundle_dir / BUNDLE_MANIFEST_FILENAME
    if not bundle_path.exists() or not manifest_path.exists():
        raise BundleIntegrityError(
            f"{bundle_dir} does not hold a complete regen bundle -- expected "
            f"both {BUNDLE_FILENAME} and {BUNDLE_MANIFEST_FILENAME}. Run "
            "make-regen-bundle first."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleIntegrityError(f"{manifest_path}: invalid JSON ({exc})") from exc

    recomputed = build_dataset.sha256_file(bundle_path)
    recorded = _lookup_first(manifest, "bundle.sha256")
    if recorded is None or recorded != recomputed:
        raise BundleIntegrityError(
            f"{bundle_path} sha256={recomputed[:16]} does not match the bundle "
            f"manifest's recorded {(str(recorded)[:16] if recorded else None)!r} "
            "-- bundle sha-chain broken, refusing to trust it."
        )

    rows = []
    with bundle_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_BUNDLE_ROW_KEYS if not row.get(k)]
            if missing:
                raise BundleIntegrityError(f"{bundle_path}:{lineno}: row missing required key(s) {missing}")
            rows.append(row)
    if not rows:
        raise BundleIntegrityError(f"{bundle_path} contains no rows")
    return rows, manifest


def load_rollouts(rollouts_path: Path) -> tuple:
    """Parse the box regen rollouts jsonl into ``{uid: {sample_idx: output}}``.

    Returns ``(by_uid, census)``. Duplicate ``(uid, sample_idx)`` entries
    are not an error (LAST occurrence wins, same "later pass wins"
    convention as ``build_dataset``'s rollouts.jsonl handling) but are
    counted.
    """
    rollouts_path = Path(rollouts_path)
    if not rollouts_path.exists():
        raise RolloutIntegrityError(
            f"{rollouts_path} not found -- run P2 (box regeneration) first; "
            "build-dataset refuses without the rollouts file."
        )
    by_uid: dict = {}
    n_rows = 0
    n_duplicate = 0
    with rollouts_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RolloutIntegrityError(f"{rollouts_path}:{lineno}: invalid JSON ({exc})") from exc
            missing = [k for k in REQUIRED_ROLLOUT_KEYS if k not in row]
            if missing:
                raise RolloutIntegrityError(f"{rollouts_path}:{lineno}: row missing required key(s) {missing}")
            uid = row["uid"]
            try:
                sample_idx = int(row["sample_idx"])
            except (TypeError, ValueError):
                raise RolloutIntegrityError(
                    f"{rollouts_path}:{lineno}: sample_idx={row['sample_idx']!r} is not an int"
                )
            samples = by_uid.setdefault(uid, {})
            if sample_idx in samples:
                n_duplicate += 1
            samples[sample_idx] = row["output"]  # last occurrence wins
    if n_rows == 0:
        raise RolloutIntegrityError(f"{rollouts_path} contains no rows -- not a usable rollouts file")
    census = {"rows": n_rows, "unique_uids": len(by_uid), "duplicate_uid_sample_idx_entries": n_duplicate}
    return by_uid, census


def default_verify_fn(output: str, answer: str) -> bool:
    """The audited icepick verify chain, mirroring ``out/passk8_sweep/
    grade.py`` exactly. Lazily imported so that importing this module (or
    running its tests, which inject a fake ``verify_fn``) never requires
    icepick's sympy/antlr4 stack.
    """
    from icepick.processing.pass_at_k.scoring import extract_candidate, strip_think
    from icepick.processing.pass_at_k.verifier import classify, verify as verifier_verify

    tier, truth = classify(answer)
    candidate = extract_candidate(strip_think(output or ""))
    return candidate is not None and bool(verifier_verify(candidate, truth, tier))


def select_first_verified(samples: dict, answer: str, k_regen: int, verify_fn) -> dict:
    """Try ``samples`` (``{sample_idx: output}``) in ascending sample_idx
    order, capped to ``< k_regen``; return ``{"sample_idx", "output",
    "k_tried"}`` for the FIRST verified one, else ``None``. ``k_tried`` is
    ``sample_idx + 1`` (traceable directly from the kept provenance, and
    visibly signals a gap if earlier tries are missing from the rollouts
    file rather than merely unverified).
    """
    in_budget = sorted(idx for idx in samples if isinstance(idx, int) and 0 <= idx < k_regen)
    for sample_idx in in_budget:
        output = samples[sample_idx]
        if verify_fn(output, answer):
            return {"sample_idx": sample_idx, "output": output, "k_tried": sample_idx + 1}
    return None


WELLPOSED_LABEL_TO_BUCKET = {
    "band": "band",
    "collapse": "collapse",
    "misdirection": "collapse",  # campaign treats collapse/misdirection as one tier
}


def load_wellposed_pool(pool_path: Path) -> dict:
    """Parse ``wellposed_all_with_passk.json`` (a JSON LIST of records) into
    ``{uid: record}``. No sha/row-count pin (none was given -- see
    ``config.V3_WELLPOSED_POOL_PATH``'s comment); callers record this
    file's live sha in the manifest instead.
    """
    pool_path = Path(pool_path)
    if not pool_path.exists():
        raise SolutionsIntegrityError(f"{pool_path} not found -- required for source-tier resolution fallback")
    try:
        data = json.loads(pool_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SolutionsIntegrityError(f"{pool_path}: invalid JSON ({exc})") from exc
    if not isinstance(data, list):
        raise SolutionsIntegrityError(f"{pool_path}: expected a JSON list of records, got {type(data).__name__}")
    by_uid = {}
    for record in data:
        uid = record.get("uid")
        if uid is not None:
            by_uid[uid] = record
    return by_uid


def resolve_hinted_tier(uid: str, band_corpus_uids: set, wellposed_pool_by_uid: dict) -> tuple:
    """Resolve one verified-hinted uid's R3 blend bucket (orchestrator
    ruling, 2026-07-31 -- see module docstring "Source-tier resolution"). Returns
    ``(bucket, resolved_label)`` where ``bucket`` is ``"band"``,
    ``"collapse"``, or ``None`` (neither 60/40 bucket -- caller must
    exclude the row from the blend and census it, never crash).

    band_corpus.jsonl membership is checked FIRST and is authoritative
    (unconditional precedence over the pool, per the ruling: 34 records
    carry pool labels stale since the 2026-07-15 gguf_rescore fold).
    ``resolved_label`` is band_corpus membership's own implicit label
    ("band", since band_corpus.jsonl is band-only by construction) in
    that branch, else the pool's ``pass_at_k_results.label`` (NESTED --
    read this path deliberately, never a flat top-level ``label``, which
    is absent on every pool row and would silently read as ``None`` for
    all of them), else a synthetic marker when the uid is not in the pool
    either or the pool entry has no ``pass_at_k_results`` block at all.
    """
    if uid in band_corpus_uids:
        return "band", "band"
    entry = wellposed_pool_by_uid.get(uid)
    if entry is None:
        return None, "not_in_wellposed_pool"
    pass_at_k_results = entry.get("pass_at_k_results")
    if not isinstance(pass_at_k_results, dict):
        return None, "missing_pass_at_k_results"
    label = pass_at_k_results.get("label")
    return WELLPOSED_LABEL_TO_BUCKET.get(label), label


def build_v3_sft_row(
    *, uid, question, output, proof_raw_sha, sample_idx, k_tried, source_tier, arxiv_id=None
) -> dict:
    """One final SFT row: schema-identical to v2 cap1's
    ({prompt, completion, provenance}) so ``train_qwen3_lora.py --dataset``
    consumes it unchanged. ``prompt`` NEVER carries the hint (see
    ``build_question_only_prompt``); ``completion`` is the model's own
    ``output`` copied VERBATIM (never stripped/normalized).
    """
    return {
        "prompt": [
            {"role": "system", "content": config.PASS_AT_K_SYSTEM_PROMPT},
            {"role": "user", "content": build_question_only_prompt(question)},
        ],
        "completion": [
            {"role": "assistant", "content": output},
        ],
        "provenance": {
            "uid": uid,
            "proof_raw_sha": proof_raw_sha,
            "regen_sample_idx": sample_idx,
            "verify_receipt": {"k_tried": k_tried, "verified": True},
            "source_tier": source_tier,
            "arxiv_id": arxiv_id,
        },
    }


def draw_anchor_rows(v2_cap1_rows: list, exclude_uids: set, n_needed: int) -> list:
    """Deterministically select ``n_needed`` v2/cap1 rows whose uid is NOT
    in ``exclude_uids`` (global cap1: every uid appears at most once across
    the WHOLE final dataset). Ranked by
    ``sha256(f"{config.V3_ANCHOR_SEED_STRING}:{uid}")`` ascending (same
    idiom as ``build_dataset._selection_rank``) -- deterministic given
    fixed v2/cap1 content + a fixed exclude set. Each returned row is a
    shallow copy with ``provenance["source_tier"] = "anchor"`` ADDED
    (every original v2/cap1 provenance key is preserved for audit).
    """
    pool = [r for r in v2_cap1_rows if (r.get("provenance") or {}).get("uid") not in exclude_uids]
    if len(pool) < n_needed:
        raise BlendError(
            f"need {n_needed} anchor row(s) (R3 blend, {config.V3_ANCHOR_FRACTION:.0%} "
            f"of the final dataset) but only {len(pool)} v2/cap1 row(s) remain "
            f"after excluding {len(exclude_uids)} uid(s) already used as hinted "
            "rows -- refusing to under-fill the anchor quota silently."
        )
    ranked = sorted(pool, key=lambda r: _anchor_rank((r.get("provenance") or {}).get("uid")))
    selected = ranked[:n_needed]
    out = []
    for row in selected:
        stamped = dict(row)
        prov = dict(row.get("provenance") or {})
        prov["source_tier"] = "anchor"
        stamped["provenance"] = prov
        out.append(stamped)
    return out


def verify_written_v3_dataset(dataset_path: Path, expected_rows: int, eval_statements: set) -> int:
    """Post-write audit: re-read ``dataset_path`` from disk and re-verify
    every row. The v3-shaped counterpart of ``build_dataset.
    verify_written_dataset`` -- REPLICATES the idiom rather than reusing
    that function, because its per-row check (``assert_verified_correct``)
    requires ``provenance.verdict``/``verbatim_output``/``rollout_uid``
    keys v3's own-regeneration provenance does not carry; the schema-shape
    check IS reused directly (``assert_prompt_completion_wellformed``),
    since that part of the contract (prompt/completion shape, pinned
    system prompt, pinned no-think suffix) is identical by design.
    """
    dataset_path = Path(dataset_path)
    rows = []
    with dataset_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise build_dataset.TraceIntegrityError(f"{dataset_path}:{lineno}: invalid JSON ({exc})") from exc
    build_dataset.assert_prompt_completion_wellformed(rows)
    assert_hint_never_in_training_prompt(rows, config.V3_HINT_MARKER)
    assert_no_statement_leakage_in_prompts(rows, eval_statements)
    if len(rows) != expected_rows:
        raise build_dataset.TraceIntegrityError(
            f"{dataset_path}: re-read {len(rows)} row(s) from disk, expected {expected_rows}"
        )
    return len(rows)


def _dataset_input_signature(
    *,
    solutions_sha256,
    bundle_sha256,
    rollouts_sha256,
    split_sha256,
    eval_set_sha256,
    corpus_sha256,
    wellposed_pool_sha256,
    v2_cap1_sha256,
    k_regen,
) -> str:
    payload = {
        "solutions_sha256": solutions_sha256,
        "bundle_sha256": bundle_sha256,
        "rollouts_sha256": rollouts_sha256,
        "split_sha256": split_sha256,
        "eval_set_sha256": eval_set_sha256,
        "corpus_sha256": corpus_sha256,
        "wellposed_pool_sha256": wellposed_pool_sha256,
        "v2_cap1_dataset_sha256": v2_cap1_sha256,
        "k_regen": k_regen,
        "hint_marker": config.V3_HINT_MARKER,
        "anchor_fraction": config.V3_ANCHOR_FRACTION,
        "hinted_collapse_fraction": config.V3_HINTED_COLLAPSE_FRACTION,
        "anchor_seed_string": config.V3_ANCHOR_SEED_STRING,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def build_dataset_cmd(
    *,
    bundle_dir: Path,
    solutions_path: Path,
    manifest_path: Path,
    rollouts_path: Path,
    split_path: Path,
    expected_split_sha256: str,
    eval_set_path: Path,
    corpus_path: Path,
    expected_corpus_sha256: str,
    expected_corpus_rows: int,
    wellposed_pool_path: Path,
    v2_cap1_dataset_path: Path,
    output_dir: Path,
    k_regen: int,
    verify_fn=None,
    force_new_dir: bool = False,
) -> dict:
    """Verify regen rollouts, blend with the v2/cap1 anchor, publish the
    final SFT dataset (skeleton P1, second half). See the module docstring
    for the full contract. ``verify_fn`` defaults to
    ``default_verify_fn`` (lazy icepick import) when ``None``.
    """
    bundle_dir = Path(bundle_dir)
    solutions_path = Path(solutions_path)
    manifest_path = Path(manifest_path)
    rollouts_path = Path(rollouts_path)
    split_path = Path(split_path)
    eval_set_path = Path(eval_set_path)
    corpus_path = Path(corpus_path)
    wellposed_pool_path = Path(wellposed_pool_path)
    v2_cap1_dataset_path = Path(v2_cap1_dataset_path)
    output_dir = Path(output_dir)
    if verify_fn is None:
        verify_fn = default_verify_fn

    # --- guards: solutions / split / bundle chain (defense in depth: full
    # re-verification, independent of whatever make-regen-bundle already
    # checked when the bundle was built) -----------------------------------
    manifest = load_solutions_manifest(manifest_path)
    solutions_sha256 = assert_solutions_sha_chain(solutions_path, manifest_path, manifest)
    assert_manifest_split_pin(manifest, manifest_path, expected_split_sha256[:16])

    train_uids = load_split_uid_sets(split_path, expected_split_sha256)

    solutions_rows = load_solutions_rows(solutions_path)
    assert_train_split_only(solutions_rows, train_uids)
    solutions_by_uid = {r["uid"]: r for r in solutions_rows}

    bundle_rows, bundle_manifest = load_bundle(bundle_dir)
    bundle_solutions_sha = _lookup_first(bundle_manifest, "solutions.sha256")
    if bundle_solutions_sha != solutions_sha256:
        raise BundleIntegrityError(
            f"bundle manifest's recorded solutions sha ({str(bundle_solutions_sha)[:16]!r}) "
            f"!= this run's solutions sha ({solutions_sha256[:16]!r}) -- the bundle was "
            "built from a DIFFERENT solutions_v3.jsonl than this run is reading; refusing."
        )
    bundle_uids = {r["uid"] for r in bundle_rows}
    solutions_train_uids = set(solutions_by_uid)
    if bundle_uids != solutions_train_uids:
        only_bundle = sorted(bundle_uids - solutions_train_uids)
        only_solutions = sorted(solutions_train_uids - bundle_uids)
        raise BundleIntegrityError(
            "bundle uid set != solutions-file train uid set -- "
            f"{len(only_bundle)} uid(s) only in the bundle: {_shown(only_bundle)}; "
            f"{len(only_solutions)} uid(s) only in solutions: {_shown(only_solutions)}"
        )
    bundle_k_regen = bundle_manifest.get("k_regen")
    if bundle_k_regen != k_regen:
        raise BundleIntegrityError(
            f"bundle was built under k_regen={bundle_k_regen!r}, this run is pinned to "
            f"k_regen={k_regen!r} -- refusing to mix regeneration budgets."
        )

    if not eval_set_path.exists():
        raise SolutionsIntegrityError(f"{eval_set_path} not found -- cannot prove non-leakage")
    eval_statements = build_dataset.load_eval_statements(eval_set_path)
    eval_set_sha256 = build_dataset.sha256_file(eval_set_path)

    build_dataset.assert_corpus_pinned(corpus_path, expected_corpus_sha256, expected_corpus_rows)
    corpus_rows = build_dataset.load_corpus(corpus_path)
    band_corpus_uids = {r.get("uid") for r in corpus_rows}
    corpus_sha256 = expected_corpus_sha256

    wellposed_pool_by_uid = load_wellposed_pool(wellposed_pool_path)
    wellposed_pool_sha256 = build_dataset.sha256_file(wellposed_pool_path)

    rollouts_by_uid, rollouts_census = load_rollouts(rollouts_path)
    rollouts_sha256 = build_dataset.sha256_file(rollouts_path)
    rollouts_ignored = sorted(set(rollouts_by_uid) - bundle_uids)

    # --- per-uid first-verified-wins harvest, then R3 tier resolution -----
    hinted_rows = []
    try_n_histogram: dict = {}
    hint_insufficient_uids = []
    missing_from_rollouts_uids = []
    excluded_offtier = []  # [{"uid": ..., "resolved_label": ...}, ...]
    tier_counts = {"collapse": 0, "band": 0}

    for brow in bundle_rows:  # bundle order: deterministic (solutions-file order)
        uid = brow["uid"]
        srow = solutions_by_uid[uid]
        samples = rollouts_by_uid.get(uid)
        if not samples:
            missing_from_rollouts_uids.append(uid)
            continue
        found = select_first_verified(samples, srow["answer"], k_regen, verify_fn)
        if found is None:
            hint_insufficient_uids.append(uid)
            continue
        bucket, resolved_label = resolve_hinted_tier(uid, band_corpus_uids, wellposed_pool_by_uid)
        if bucket is None:
            # Neither 60/40 bucket (e.g. a GGUF-7/8 backfill uid whose only
            # pool label is "solved") -- excluded from the blend entirely,
            # never silently dropped or crashed (orchestrator ruling).
            excluded_offtier.append({"uid": uid, "resolved_label": resolved_label})
            continue
        tier_counts[bucket] = tier_counts.get(bucket, 0) + 1
        arxiv_id = (srow.get("provenance") or {}).get("arxiv_id")
        hinted_rows.append(
            build_v3_sft_row(
                uid=uid,
                question=srow["question"],
                output=found["output"],
                proof_raw_sha=srow["proof_raw_sha"],
                sample_idx=found["sample_idx"],
                k_tried=found["k_tried"],
                source_tier=bucket,
                arxiv_id=arxiv_id,
            )
        )
        try_n_histogram[found["k_tried"]] = try_n_histogram.get(found["k_tried"], 0) + 1

    # --- R3 blend: 75% hinted (60/40 collapse/band, observed not forced)
    # + 25% deterministic v2/cap1 anchor, excluding hinted uids -----------
    hinted_count = len(hinted_rows)
    anchor_count = round(hinted_count * config.V3_ANCHOR_FRACTION / config.V3_HINTED_FRACTION)

    if not v2_cap1_dataset_path.exists():
        raise BlendError(f"{v2_cap1_dataset_path} not found -- cannot draw the R3 anchor")
    v2_cap1_rows = []
    with v2_cap1_dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                v2_cap1_rows.append(json.loads(line))
    v2_cap1_sha256 = build_dataset.sha256_file(v2_cap1_dataset_path)

    hinted_uids = {_row_uid(r) for r in hinted_rows}
    anchor_pool_size = sum(
        1 for r in v2_cap1_rows if (r.get("provenance") or {}).get("uid") not in hinted_uids
    )
    anchor_rows = draw_anchor_rows(v2_cap1_rows, hinted_uids, anchor_count) if anchor_count > 0 else []

    final_rows = hinted_rows + anchor_rows
    if not final_rows:
        raise BlendError(
            "the R3 blend produced ZERO final rows (0 hinted, 0 anchor) -- "
            "refusing to publish an empty dataset."
        )

    # --- final guards on the published shape (defense in depth) ----------
    build_dataset.assert_prompt_completion_wellformed(final_rows)
    assert_hint_never_in_training_prompt(final_rows, config.V3_HINT_MARKER)
    assert_no_statement_leakage_in_prompts(final_rows, eval_statements)

    loss_mass_census = {
        "rows": len(final_rows),
        "prompts_hint_free": sum(
            1 for r in final_rows if config.V3_HINT_MARKER not in r["prompt"][1]["content"]
        ),
        "completions_nonempty": sum(
            1 for r in final_rows if r["completion"][0].get("content")
        ),
        "schema": "prompt_completion, version 2 (completion-only loss) -- matches v2 cap1's trainer contract byte-for-byte",
    }

    content_bytes = _jsonl_bytes(final_rows)
    dataset_sha256 = _sha256_bytes(content_bytes)

    input_signature = _dataset_input_signature(
        solutions_sha256=solutions_sha256,
        bundle_sha256=_lookup_first(bundle_manifest, "bundle.sha256"),
        rollouts_sha256=rollouts_sha256,
        split_sha256=expected_split_sha256,
        eval_set_sha256=eval_set_sha256,
        corpus_sha256=corpus_sha256,
        wellposed_pool_sha256=wellposed_pool_sha256,
        v2_cap1_sha256=v2_cap1_sha256,
        k_regen=k_regen,
    )

    publish_dir, existing = _resolve_publish_dir(
        output_dir,
        force_new_dir=force_new_dir,
        manifest_name=DATASET_MANIFEST_FILENAME,
        expected_stage=STAGE_BUILD_DATASET,
        input_signature=input_signature,
    )
    if existing is not None:
        return existing

    publish_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = publish_dir / DATASET_FILENAME
    dataset_tmp_path = publish_dir / (DATASET_FILENAME + ".tmp")
    manifest_out_path = publish_dir / DATASET_MANIFEST_FILENAME

    dataset_tmp_path.write_bytes(content_bytes)
    n = verify_written_v3_dataset(dataset_tmp_path, len(final_rows), eval_statements)
    if n != len(final_rows):
        raise build_dataset.TraceIntegrityError(
            f"post-write verification read {n} row(s), expected {len(final_rows)} -- "
            "dataset NOT published (tmp left for forensics)."
        )
    dataset_tmp_path.replace(dataset_path)

    out_manifest = {
        "stage": STAGE_BUILD_DATASET,
        "created": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "solutions": {"path": str(solutions_path), "sha256": solutions_sha256, "rows": len(solutions_rows)},
            "solutions_manifest_path": str(manifest_path),
            "bundle": {
                "path": str(bundle_dir / BUNDLE_FILENAME),
                "sha256": _lookup_first(bundle_manifest, "bundle.sha256"),
                "rows": len(bundle_rows),
            },
            "rollouts": {"path": str(rollouts_path), "sha256": rollouts_sha256, **rollouts_census},
            "split": {
                "path": str(split_path),
                "sha256": expected_split_sha256,
                "n_train_side_uids": len(train_uids),
                "paper_disjointness_note": (
                    "train_papers ∩ eval_papers == ∅, asserted as a hard build-time invariant "
                    "of the split artifact itself (evalharness/data/corpus_split_v3_"
                    "proofsplit_20260801.json) -- paper-level disjointness is the load-bearing "
                    "guard there; this module's own runtime guard remains uid-level "
                    "(assert_train_split_only). No holdout concept exists in this split "
                    "(retired 2026-08-01, split-rebuild-2026-08-01.md)."
                ),
            },
            "eval_set": {"path": str(eval_set_path), "sha256": eval_set_sha256},
            "corpus": {"path": str(corpus_path), "sha256": corpus_sha256, "rows": expected_corpus_rows},
            "wellposed_pool": {"path": str(wellposed_pool_path), "sha256": wellposed_pool_sha256, "n_records": len(wellposed_pool_by_uid)},
            "v2_cap1_dataset": {"path": str(v2_cap1_dataset_path), "sha256": v2_cap1_sha256, "rows": len(v2_cap1_rows)},
        },
        "verify_fn": "default_icepick_chain" if verify_fn is default_verify_fn else "injected",
        "k_regen": k_regen,
        "censuses": {
            "solutions_rows": len(solutions_rows),
            "bundle_rows": len(bundle_rows),
            "rollouts_uids_ignored_not_in_bundle": len(rollouts_ignored),
            "verified_on_try_n_histogram": {str(k): v for k, v in sorted(try_n_histogram.items())},
            "hint_insufficient": {"count": len(hint_insufficient_uids), "uids": sorted(hint_insufficient_uids)},
            "missing_from_rollouts": {"count": len(missing_from_rollouts_uids), "uids": sorted(missing_from_rollouts_uids)},
            "excluded_offtier": {
                "count": len(excluded_offtier),
                "uids": sorted(e["uid"] for e in excluded_offtier),
                "resolved_labels": {e["uid"]: e["resolved_label"] for e in excluded_offtier},
            },
            "kept_hinted": hinted_count,
        },
        "blend": {
            "ratios_config": {
                "anchor_fraction": config.V3_ANCHOR_FRACTION,
                "hinted_fraction": config.V3_HINTED_FRACTION,
                "hinted_collapse_fraction": config.V3_HINTED_COLLAPSE_FRACTION,
                "hinted_band_fraction": config.V3_HINTED_BAND_FRACTION,
            },
            "arithmetic_interpretation": (
                "R3 as written ('60/40 collapse/band hinted rows + 25% unhinted "
                "anchor rows') is ambiguous. Implemented reading (orchestrator "
                "ruling, 2026-07-31): final dataset = 75% hinted (60/40 "
                "collapse/band split BY SOURCE-RECORD TIER, observed not "
                "forced) + 25% anchor. anchor_count = round(hinted_count * "
                "V3_ANCHOR_FRACTION / V3_HINTED_FRACTION)."
            ),
            "hinted_count": hinted_count,
            "hinted_collapse_count": tier_counts.get("collapse", 0),
            "hinted_band_count": tier_counts.get("band", 0),
            "hinted_collapse_achieved_fraction": (tier_counts.get("collapse", 0) / hinted_count) if hinted_count else None,
            "hinted_band_achieved_fraction": (tier_counts.get("band", 0) / hinted_count) if hinted_count else None,
            "nominal_vs_achieved_note": (
                "the 60/40 collapse/band split is a NOMINAL target this builder "
                "records but never enforces by dropping verified hinted rows -- "
                "today's only exercised proof-import target set (R2 default) is "
                "100% band, which makes a forced 60/40 split impossible without "
                "discarding hard-won verified traces; see module docstring "
                "'Blend'."
            ),
            "anchor_count_formula": "round(hinted_count * V3_ANCHOR_FRACTION / V3_HINTED_FRACTION)",
            "anchor_count": len(anchor_rows),
            "anchor_pool_size_available": anchor_pool_size,
            "anchor_pool_excludes_hinted_uids": True,
            "anchor_seed_string": config.V3_ANCHOR_SEED_STRING,
            "final_rows": len(final_rows),
        },
        "source_tier_resolution": {
            "method": (
                "band_corpus.jsonl uid membership FIRST, unconditional precedence "
                "(band_corpus is band-label-only by construction); ONLY for a uid "
                "absent from band_corpus does resolution fall back to "
                "wellposed_all_with_passk.json's NESTED pass_at_k_results.label "
                "(never the absent flat top-level 'label' key). 'collapse' and "
                "'misdirection' pool labels both map to the collapse bucket. Any "
                "other resolved value (not in either source, or a label such as "
                "'solved') is excluded from the blend -- see censuses."
                "excluded_offtier -- never crashes."
            ),
            "ruling": (
                "Orchestrator ruling, 2026-07-31, from a "
                "completed inventory cross-check: 34 records carry pool labels "
                "stale since the 2026-07-15 gguf_rescore fold rebanded them into "
                "band_corpus without the pool being refreshed, hence band_corpus "
                "precedence; the 7 GGUF-7/8 backfill uids exist only in the pool "
                "(5 labelled solved -> excluded_offtier, 2 labelled band -> band)."
            ),
        },
        "loss_mass_census": loss_mass_census,
        "sft_schema": {
            "format": "prompt_completion",
            "version": 2,
            "completion_only_loss": True,
            "note": "identical wire contract to loratrain/data/v2/cap1 -- train_qwen3_lora.py --dataset consumes this file unchanged.",
        },
        "hint_marker": config.V3_HINT_MARKER,
        "dataset": {"path": str(dataset_path), "sha256": dataset_sha256, "rows": len(final_rows)},
        "published_files": {DATASET_FILENAME: dataset_sha256},
        "input_signature": input_signature,
        "guards": list(BUILD_DATASET_GUARD_STEPS),
        "resumed_from_existing_publish": False,
    }
    manifest_out_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    return out_manifest


# ============================================================================
# Restartability (shared by both subcommands)
# ============================================================================


def _existing_manifest(output_dir: Path, manifest_name: str, expected_stage: str):
    """Return the parsed manifest dict if ``output_dir`` holds a COMPLETE
    prior publish for ``expected_stage``, else ``None``. "Complete" =
    manifest.json parses, its stage matches, and every file it lists under
    ``published_files`` exists with a matching sha256. Any other state
    (missing manifest, unparseable JSON, stage mismatch, missing/mismatched
    referenced file) is treated as NOT complete -- the caller proceeds to
    (re)build fresh into this directory (a partial/crashed prior attempt is
    not preserved as sacred; only a verified-complete publish is).
    """
    manifest_path = output_dir / manifest_name
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(manifest, dict) or manifest.get("stage") != expected_stage:
        return None
    for rel_path, sha in (manifest.get("published_files") or {}).items():
        p = output_dir / rel_path
        if not p.exists() or build_dataset.sha256_file(p) != sha:
            return None
    return manifest


def _resolve_publish_dir(
    requested_dir: Path, *, force_new_dir: bool, manifest_name: str, expected_stage: str, input_signature: str
):
    """Restartability gate shared by both subcommands (module docstring
    "Restartability"). Returns ``(publish_dir, existing_manifest_or_None)``.
    """
    existing = _existing_manifest(requested_dir, manifest_name, expected_stage)
    if existing is None:
        return requested_dir, None
    if existing.get("input_signature") == input_signature:
        resumed = dict(existing)
        resumed["resumed_from_existing_publish"] = True
        return requested_dir, resumed
    if not force_new_dir:
        raise PublishConflictError(
            f"{requested_dir} already holds a completed {expected_stage} publish "
            f"built from DIFFERENT inputs (recorded input_signature="
            f"{existing.get('input_signature')!r} != this run's {input_signature!r}). "
            "Refusing to overwrite a completed publish -- pass --force-new-dir to "
            "publish under a fresh sibling directory (the existing one is left "
            "untouched), or point at a different --bundle-dir/--output-dir."
        )
    n = 2
    while True:
        candidate = requested_dir.parent / f"{requested_dir.name}__{n}"
        if not candidate.exists():
            return candidate, None
        n += 1


# ============================================================================
# V3 config validation (companion to config.validate_config -- kept
# separate because that existing function may not be edited; see this
# module's docstring and config.py's V3 constants section)
# ============================================================================


def validate_v3_config() -> None:
    """Validate the V3 (proof-hint arm) constants appended to config.py.
    Never fails fast; collects every problem, same convention as
    ``config.validate_config``.
    """
    problems = []
    if not isinstance(config.V3_K_REGEN, int) or isinstance(config.V3_K_REGEN, bool) or config.V3_K_REGEN < 1:
        problems.append(f"V3_K_REGEN must be a positive int (got {config.V3_K_REGEN!r})")
    if not (0 < config.V3_ANCHOR_FRACTION < 1):
        problems.append(f"V3_ANCHOR_FRACTION must be in (0, 1) (got {config.V3_ANCHOR_FRACTION!r})")
    if not (0 < config.V3_HINTED_COLLAPSE_FRACTION < 1):
        problems.append(
            f"V3_HINTED_COLLAPSE_FRACTION must be in (0, 1) (got {config.V3_HINTED_COLLAPSE_FRACTION!r})"
        )
    if not config.V3_HINT_MARKER or not isinstance(config.V3_HINT_MARKER, str):
        problems.append("V3_HINT_MARKER must be a non-empty str")
    if not config.V3_ANCHOR_SEED_STRING or not isinstance(config.V3_ANCHOR_SEED_STRING, str):
        problems.append("V3_ANCHOR_SEED_STRING must be a non-empty str")
    if config.V3_HINT_INSUFFICIENT_POLICY not in config.VALID_V3_HINT_INSUFFICIENT_POLICIES:
        problems.append(
            "V3_HINT_INSUFFICIENT_POLICY must be one of "
            f"{config.VALID_V3_HINT_INSUFFICIENT_POLICIES} (got {config.V3_HINT_INSUFFICIENT_POLICY!r})"
        )
    _hex64 = lambda s: isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
    _hex16 = lambda s: isinstance(s, str) and len(s) == 16 and all(c in "0123456789abcdef" for c in s)
    if not _hex64(config.V3_EXPECTED_SPLIT_SHA256):
        problems.append(
            f"V3_EXPECTED_SPLIT_SHA256 must be 64 lowercase hex chars (got {config.V3_EXPECTED_SPLIT_SHA256!r})"
        )
    if not _hex16(config.V3_EXPECTED_SPLIT_SHA256_16):
        problems.append(
            f"V3_EXPECTED_SPLIT_SHA256_16 must be 16 hex chars (got {config.V3_EXPECTED_SPLIT_SHA256_16!r})"
        )
    if (
        not isinstance(config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S, tuple)
        or not config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S
        or not all(_hex16(s) for s in config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S)
    ):
        problems.append(
            "V3_ACCEPTED_MANIFEST_SPLIT_SHA16S must be a non-empty tuple of 16-hex-char str "
            f"(got {config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S!r})"
        )
    if problems:
        raise config.ConfigError(
            f"{len(problems)} configuration problem(s) in loratrain/v3.py's V3 constants:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


# ============================================================================
# CLI
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m loratrain.v3",
        description=(
            "v3 proof-as-hint regeneration dataset builder (skeleton P1). "
            "Two subcommands: make-regen-bundle, build-dataset. No flag here "
            "can override a sha pin or skip/weaken a guard; --force-new-dir "
            "only resolves an output-directory collision (see module docstring)."
        ),
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    b = sub.add_parser("make-regen-bundle", help="Build the hint-augmented regen prompt bundle for P2.")
    b.add_argument(
        "--solutions", type=Path, required=True,
        help="Path to solutions_v3.jsonl (proof-import P5 publish). Required, explicit -- no default, no auto-discovery of out/proof_import_*.",
    )
    b.add_argument(
        "--manifest", type=Path, default=None,
        help="Path to the solutions file's manifest.json (default: its sibling manifest.json).",
    )
    b.add_argument("--split", type=Path, default=config.V3_SPLIT_PATH, help="Path to the pinned split file.")
    b.add_argument("--eval-set", type=Path, default=config.EVAL_SET_PATH, help="Path to eval_set.jsonl.")
    b.add_argument(
        "--bundle-dir", type=Path, required=True,
        help="Output dir for regen_bundle.jsonl + its manifest. Required -- no default.",
    )
    b.add_argument(
        "--force-new-dir", action="store_true",
        help="If --bundle-dir already holds a completed publish from DIFFERENT inputs, publish under a fresh sibling dir instead of refusing (the existing dir is left untouched).",
    )

    d = sub.add_parser("build-dataset", help="Verify regen rollouts, blend with the v2/cap1 anchor, publish the final SFT dataset.")
    d.add_argument("--bundle-dir", type=Path, required=True, help="Dir holding regen_bundle.jsonl + its manifest (make-regen-bundle output).")
    d.add_argument(
        "--solutions", type=Path, required=True,
        help="Path to solutions_v3.jsonl -- SAME file used for make-regen-bundle. Required, explicit -- no auto-discovery.",
    )
    d.add_argument("--manifest", type=Path, default=None, help="Path to the solutions file's manifest.json (default: its sibling manifest.json).")
    d.add_argument("--rollouts", type=Path, required=True, help="Box regen outputs: jsonl of {uid, sample_idx, output}.")
    d.add_argument("--split", type=Path, default=config.V3_SPLIT_PATH, help="Path to the pinned split file.")
    d.add_argument("--eval-set", type=Path, default=config.EVAL_SET_PATH, help="Path to eval_set.jsonl.")
    d.add_argument("--corpus", type=Path, default=config.CORPUS_PATH, help="Path to band_corpus.jsonl (source-tier resolution, authoritative for 'band').")
    d.add_argument(
        "--wellposed-pool", type=Path, default=config.V3_WELLPOSED_POOL_PATH,
        help="Path to wellposed_all_with_passk.json (source-tier resolution fallback for uids absent from --corpus).",
    )
    d.add_argument(
        "--v2-cap1-dataset", type=Path, default=(config.DATA_V2_DIR / "cap1" / DATASET_FILENAME),
        help="Path to the v2 cap1 sft_train.jsonl this build draws its R3 anchor rows from.",
    )
    d.add_argument("--output-dir", type=Path, required=True, help="Output dir for sft_train.jsonl + dataset_manifest.json. Required -- no default.")
    d.add_argument(
        "--force-new-dir", action="store_true",
        help="If --output-dir already holds a completed publish from DIFFERENT inputs, publish under a fresh sibling dir instead of refusing (the existing dir is left untouched).",
    )

    return p


def main(argv=None) -> int:
    """CLI entrypoint: validate config, run the requested subcommand, print
    a short JSON summary. Guard failures propagate uncaught (a refusal is
    always loud, never a silently-swallowed nonzero exit) -- same
    convention as ``build_dataset.main``.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config.validate_config()
    validate_v3_config()

    if args.subcommand == "make-regen-bundle":
        manifest_path = args.manifest or (args.solutions.parent / "manifest.json")
        manifest = make_regen_bundle(
            solutions_path=args.solutions,
            manifest_path=manifest_path,
            split_path=args.split,
            expected_split_sha256=config.V3_EXPECTED_SPLIT_SHA256,
            eval_set_path=args.eval_set,
            bundle_dir=args.bundle_dir,
            k_regen=config.V3_K_REGEN,
            force_new_dir=args.force_new_dir,
        )
        summary = {
            "stage": manifest["stage"],
            "bundle_path": manifest["bundle"]["path"],
            "rows": manifest["bundle"]["rows"],
            "sha256": manifest["bundle"]["sha256"],
            "resumed_from_existing_publish": manifest.get("resumed_from_existing_publish", False),
        }
    else:
        manifest = build_dataset_cmd(
            bundle_dir=args.bundle_dir,
            solutions_path=args.solutions,
            manifest_path=args.manifest or (args.solutions.parent / "manifest.json"),
            rollouts_path=args.rollouts,
            split_path=args.split,
            expected_split_sha256=config.V3_EXPECTED_SPLIT_SHA256,
            eval_set_path=args.eval_set,
            corpus_path=args.corpus,
            expected_corpus_sha256=config.EXPECTED_CORPUS_SHA256,
            expected_corpus_rows=config.EXPECTED_CORPUS_ROWS,
            wellposed_pool_path=args.wellposed_pool,
            v2_cap1_dataset_path=args.v2_cap1_dataset,
            output_dir=args.output_dir,
            k_regen=config.V3_K_REGEN,
            verify_fn=None,
            force_new_dir=args.force_new_dir,
        )
        summary = {
            "stage": manifest["stage"],
            "dataset_path": manifest["dataset"]["path"],
            "rows": manifest["dataset"]["rows"],
            "sha256": manifest["dataset"]["sha256"],
            "censuses": manifest["censuses"],
            "blend": {k: manifest["blend"][k] for k in ("hinted_count", "anchor_count", "final_rows")},
            "resumed_from_existing_publish": manifest.get("resumed_from_existing_publish", False),
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

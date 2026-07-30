"""Harvest verified-correct rollout traces for train uids into an SFT jsonl.

W2: fully implemented. Every guard function below is REAL and
independently tested; ``build()`` runs them all, in a fixed order,
before a single byte is written to disk, then writes
``data/sft_train.jsonl`` + ``data/dataset_manifest.json`` and re-verifies
the WRITTEN file from disk before returning. ``main()`` is the CLI
entrypoint (``loratrain-build-dataset``): validate config, run
``build()`` with the config pins, print a short JSON summary.

SFT example schema v2 (one JSON object per line of the built
``sft_train.jsonl``; datasets land under ``data/v2/<policy-label>/``,
never over the v1 artifacts)::

    {
      "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
      ],
      "completion": [
        {"role": "assistant", "content": "..."}
      ],
      "weight": 0.5,          # ONLY under the "inverse" weight policy
      "provenance": {
        "uid": "...",
        "rollout_uid": "...",
        "sample_idx": 0,
        "arxiv_id": "...",
        "source_file": "...",
        "trace_file": "...",
        "reconciled_via": "routed",
        "verdict": "correct",
        "verbatim_output": true,
        "corpus_sha256": "...",
        "backfill_7of8": false
      }
    }

``source_file`` is the corpus row's provenance CLAIM
(``corpus_provenance.source_file``); ``trace_file`` is the rollouts file
the trace was ACTUALLY harvested from, and ``reconciled_via`` says how
the two relate (``"routed"``: the claimed run's rollouts file
reconciles; ``"unique_alternative"``: it does not, and exactly one
other registry file does -- see "Rollout reconciliation" below).
``backfill_7of8`` is ``true`` iff this example's record came from the
pinned GGUF 7/8 backfill roster (``config.BACKFILL_TRACE_SOURCES``,
Nicky's ruling 2026-07-26) rather than ``band_corpus.jsonl`` -- see
"GGUF 7/8 backfill" below.

Why prompt/completion columns (defect-1 fix, 2026-07-29): v1 emitted a
single ``messages`` list that the box-side trainer pre-templated into
one string -- a language-modeling dataset, so SFTTrainer computed loss
over the FULL sequence, prompt tokens included (measured: 21.6% of
trained characters were system+user text, the identical system prompt
repeated in all 700 rows). In the pinned trl 0.29.1, a
prompt/completion dataset gets completion-only loss by default: the
trainer renders ``prompt`` with ``add_generation_prompt=True``, renders
``prompt + completion`` through the same chat template v1 used (total
rendered text byte-identical to v1's -- the loss mask is the only
train-time delta), and masks every prompt-token label to -100.
``assistant_only_loss`` is NOT usable here -- it requires a
``{% generation %}``-tagged chat template, which Qwen3's lacks.

Weight policy (defect-2 fix, same date): v1's one-row-per-correct-trace
harvest made a record's gradient mass equal ``n_correct`` -- how often
the BASE model already solved it (anti-difficulty). ``apply_weight_policy``
(config knob ``WEIGHT_POLICY``, default ``cap1``; CLI ``--weight-policy``)
caps rows per uid (``cap1``/``capk``, seeded deterministic selection --
``SELECTION_RULE``) or stamps per-row ``weight = 1/n`` (``inverse``).
Which policy ships is Nicky's decision; all three build.

The assistant message's ``content`` is the rollout's ``output`` field
copied VERBATIM -- never the corpus ``answer`` field. This is the
grader-equivalence defense (README D4): the grader accepts
algebraically-equivalent answers in forms the canonical ``answer``
string does not cover, so training on ``answer`` would teach a canonical
surface form the base model doesn't need and would fabricate a gain
that isn't there. Targets must be the base model's own accepted output,
not the corpus's idea of the "right" string. ``assert_verified_correct``
and the ``verbatim_output`` provenance flag exist to make this
checkable, not just promised.

Rollout reconciliation (measured reality, 2026-07-25 -- two independent
sessions confirmed the same numbers on the pinned corpus): the
``_progress/rollouts.jsonl`` files are APPEND-ACROSS-PASSES logs, not
one-row-per-rollout tables. A rescore pass re-samples under the SAME
``rollout_uid``, so a file can hold several lines per key with
different ``output``/``verdict`` (tier1_band: 651 duplicate entries,
104 keys whose content differs across occurrences); and for 15 of the
293 corpus rows the row's routed file (its
``corpus_provenance.source_file`` sibling) does not reconcile at all --
each of those 15 reconciles in EXACTLY ONE other rollouts file on disk
(a later local rerun the corpus row's counts were actually taken from).
The protocol here (credit: the parallel W2 session, verified
independently before adoption): index every file by LAST occurrence per
``(uid, rollout_uid)`` (later pass wins); a row's authoritative file is
its routed file if that reconciles (all rollout_uids present, uid
match, verdict tally == the row's ``n_correct``/``n_wrong``/
``n_degenerate``), else the UNIQUE registry file that reconciles;
zero or two-plus reconciling candidates -> ``TraceIntegrityError``.
Each rollouts line's ``output``+``verdict`` pair was written together
by the scoring run, so a harvested ``verdict=="correct"`` line is a
verified-correct trace, and the tally match ties the harvested SET to
the label-producing pass at MULTISET level. Documented residual
(cross-review, 2026-07-25, accepted by both sessions): a later pass
that rewrote a subset of a row's rollout_uids with same-verdict lines
-- or compensating verdict flips that preserve the multiset -- yields a
mixed-pass harvest set that still reconciles; per-line pass identity is
not guaranteed. That is acceptable for D4 because each harvested line
remains independently verifier-accepted, which is the load-bearing
property; the manifest and every example's provenance keep the audit
trail honest by recording which file actually supplied the trace
(``trace_file``), how it reconciled (``reconciled_via``), and per-file
duplicate counts.

GGUF 7/8 backfill (Nicky's ruling 2026-07-26, added on top of the W2
design above without touching any of it): the split
(``evalharness/data/corpus_split_200_100.json``, ``config.
EVAL_PAPER_SPLIT_PATH``) keeps a 200-uid train set by backfilling the
7-record shortfall the 2026-07-16 repair-lane fold left (band_corpus
309 -> 293) from the GGUF 7/8 rescore pool -- 7 pinned uids
(``config.BACKFILL_TRACE_SOURCES``) that are NOT in ``band_corpus.jsonl``
by construction. ``select_train_records`` exempts exactly this pinned
set from its "train uid must be in the corpus" refusal;
``load_backfill_records`` loads each one's row from its pinned
FIRST-PASS ``pass_at_k.jsonl`` (asserting ``n_correct==7`` and
``label=="solved"``), synthesizes a ``corpus_provenance.source_file``
onto it, and merges it into the same record list as every corpus-
resident record -- from that point on EVERY guard below (leakage,
statement-dup, rollout reconciliation, harvest, dedupe, verbatim,
write/verify) runs unmodified over both kinds of record identically.
``assert_backfill_mapping_complete`` independently re-verifies (defense
in depth) that the split's declared ``train_backfill_7of8_uids`` and
``config.BACKFILL_TRACE_SOURCES`` name the exact same uids. Each
harvested backfill example's ``provenance.backfill_7of8`` is ``true``;
the manifest's ``backfill_7of8`` block records the roster, pinned
source files + shas, and per-uid harvested trace counts.

Build flow (``build()``'s guard order -- see also ``BUILD_GUARD_STEPS``,
echoed into the written manifest as an audit trail): reject retired
split paths -> pin-check the corpus -> pin-check the split (full
sha256) -> load eval_papers (sha16, redundant defense-in-depth) -> load
+ cross-check the backfill roster against config.BACKFILL_TRACE_SOURCES
-> require the derived split (train_uids.txt / eval_set.jsonl) to
already exist -> load it -> assert train/eval uid disjointness ->
select this build's train records from the pinned corpus (backfill
uids exempted) -> load + validate the pinned backfill records and merge
them in -> paper/uid leakage -> full-statement duplicate checks (within
train, and train-vs-eval) -> resolve every train record's routed
rollouts file + discover the registry -> load each registry file
(last-occurrence index) -> reconcile every record to its authoritative
file (routed, else unique alternative, else hard fail) -> harvest
verified-correct traces from the authoritative index (per-record
verdict-tally re-check against the corpus row; backfill examples
stamped) -> dedupe -> apply the weight policy (defect 2) -> re-verify +
assert prompt/completion well-formedness + weight policy honored +
verbatim targets + re-check leakage (nested shape, defense in depth) ->
write the dataset + manifest (with its backfill + weight-policy audit
blocks) -> re-read the WRITTEN file and re-verify every row from disk
(schema + policy included). Nothing is written until every guard above
the "write" step has passed.

Guard + build functions in this module (all real, all independently tested):

  sha256_file                          -- streamed file hash, 1 MiB chunks
  load_uid_list                        -- read a newline uid list
  assert_corpus_pinned                 -- corpus sha256 + row-count pin
  assert_split_pinned                  -- split full sha256 pin (authoritative)
  load_eval_papers                     -- split sha16 pin (redundant) -> arxiv_id set
  load_backfill_uids                   -- split -> train_backfill_7of8_uids list
  assert_backfill_mapping_complete     -- split roster == config.BACKFILL_TRACE_SOURCES
  assert_no_leakage                    -- paper-level + uid-level hard fail
  dedupe_examples                      -- (uid, rollout_uid) de-dup
  assert_no_cross_uid_statement_dups   -- full-statement cross-uid collision
  assert_verified_correct              -- verdict/verbatim_output/rollout_uid
  assert_not_retired_path              -- refuse evalharness/data/retired_*
  load_corpus                          -- parse band_corpus.jsonl
  load_eval_uids                       -- eval_set.jsonl -> uid set
  load_eval_statements                 -- eval_set.jsonl -> full-statement set
  assert_train_eval_disjoint           -- train/eval uid-level hard fail
  assert_no_cross_split_statement_dups -- full-statement train-vs-eval collision
  select_train_records                 -- corpus rows for this build's train uids (backfill exempted)
  load_backfill_records                -- pinned GGUF 7/8 rows -> corpus-row-shaped records
  rollouts_path_for                    -- source_file -> its _progress/rollouts.jsonl
  load_rollout_file                    -- ONE file -> last-occurrence index + dup count
  discover_registry                    -- routed files + REGISTRY_GLOBS -> candidate files
  reconcile_record                     -- routed-or-unique-alternative resolution
  harvest_correct_traces               -- verdict-tally re-check + SFT harvest
  build_sft_example                    -- one verbatim SFT example (never reads "answer")
  apply_weight_policy                  -- defect-2 fix: cap1/capk/inverse row policy
  assert_weight_policy_honored         -- per-uid caps / exact 1/n weights re-check
  assert_prompt_completion_wellformed  -- defect-1 fix: schema + wire-format pins
  assert_verbatim_targets              -- byte-identity of assistant content vs source
  verify_written_dataset               -- post-write, re-read-from-disk audit
  build_manifest / write_dataset       -- pure manifest assembly / jsonl writer
  build                                -- the full guarded orchestrator
  main                                 -- CLI entrypoint (loratrain-build-dataset)

See README "Split & corpus" and "Non-negotiable ordering & invariants"
for the invariants these enforce, and D4 for the grader-equivalence
defense in full.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from loratrain import config


class LeakageError(RuntimeError):
    """A harvested example resolves to an eval paper or eval uid.

    Eval-paper records are radioactive to training (README "Split &
    corpus") -- this is a hard fail, never a warning.
    """


class PinMismatchError(RuntimeError):
    """A pinned input (corpus or eval-paper split) no longer matches its pin."""


class TraceIntegrityError(RuntimeError):
    """An example is not a verified-correct, verbatim rollout trace."""


class DuplicateRecordError(RuntimeError):
    """Two different uids share a byte-identical full statement (finding F4)."""


class SplitNotBuiltError(RuntimeError):
    """The derived split inputs (train_uids.txt / eval_set.jsonl) do not exist yet.

    ``train_uids.txt`` / ``eval_set.jsonl`` must already have been
    produced (README "Split & corpus") before this module runs -- it
    never builds them itself. Any ``evalharness/data/retired_*`` path
    (e.g. ``retired_20260716/``, ``retired_20260726/``) is retired and
    must never be pointed at; see ``assert_not_retired_path``.
    """


class WeightPolicyError(RuntimeError):
    """The weight policy (defect-2 fix) is unknown, misconfigured, or violated.

    Raised both at application time (``apply_weight_policy`` refusing an
    unknown policy / non-positive cap) and at re-verification time
    (``assert_weight_policy_honored`` finding a per-uid row count above
    the cap, or an ``inverse`` weight that is not exactly ``1/n``).
    """


class BackfillMappingError(RuntimeError):
    """The split's ``train_backfill_7of8_uids`` and config's
    ``BACKFILL_TRACE_SOURCES`` disagree.

    Both must name the EXACT SAME set of uids (Nicky's ruling
    2026-07-26: the GGUF 7/8 backfill is a pinned universe extension, not
    an open-ended one) -- a mismatch means the split's backfill roster
    moved without config.py's pinned source mapping being updated in
    lockstep, or vice versa. See ``assert_backfill_mapping_complete``.
    """


def sha256_file(path: Path) -> str:
    """Stream ``path`` in 1 MiB chunks and return its full sha256 hex digest."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_uid_list(path: Path) -> list:
    """Read a newline-delimited uid list (e.g. ``train_uids.txt``).

    Blank lines are dropped; surrounding whitespace is stripped.
    ``FileNotFoundError`` propagates naturally from the failed open if
    ``path`` does not exist. Raises ``ValueError`` if no uids remain
    after filtering (an empty or all-blank file is not a usable uid
    list).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh]
    uids = [line for line in lines if line]
    if not uids:
        raise ValueError(f"{path} contains no uids (empty or all-blank file)")
    return uids


def assert_corpus_pinned(corpus_path: Path, expected_sha256: str, expected_rows: int) -> None:
    """Hard-fail unless ``corpus_path`` matches the pinned sha256 AND row count.

    Streams the file exactly once, hashing bytes and counting rows
    (newline-terminated, plus a final unterminated line if present) in
    the same pass. This is the guard between build_dataset and a corpus
    that moved out from under the pin -- reband, repair-lane fold,
    in-place edit, wrong file entirely.
    """
    corpus_path = Path(corpus_path)
    h = hashlib.sha256()
    rows = 0
    size = 0
    with corpus_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            rows += chunk.count(b"\n")
            size += len(chunk)
    if size > 0:
        with corpus_path.open("rb") as fh:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                rows += 1  # trailing line with no final newline still counts
    actual_sha256 = h.hexdigest()

    if actual_sha256 != expected_sha256 or rows != expected_rows:
        raise PinMismatchError(
            f"{corpus_path} no longer matches the pinned corpus identity in "
            f"config.py: sha256[:16] expected={expected_sha256[:16]} "
            f"actual={actual_sha256[:16]}, rows expected={expected_rows} "
            f"actual={rows}. This means the corpus moved (reband, repair-lane "
            "fold, in-place edit) since the pin was set -- if that move was "
            "deliberate, update EXPECTED_CORPUS_SHA256 / EXPECTED_CORPUS_ROWS "
            "in config.py as an explicit rebase; do not silently proceed "
            "against a moved corpus."
        )


def assert_split_pinned(split_path: Path, expected_sha256: str) -> None:
    """Hard-fail unless ``split_path`` matches the pinned FULL sha256.

    Same pin-check style as ``assert_corpus_pinned`` -- full-file sha256
    over exact bytes, compared in full, no truncation -- promoted to the
    split file's PRIMARY identity check 2026-07-26 (Nicky's ruling:
    corpus_split_200_100.json is authoritative again). Runs BEFORE
    ``load_eval_papers``, whose own sha16 pin check keeps running
    unchanged afterward as a redundant defense-in-depth layer (same
    idiom this module uses throughout -- e.g. ``build()``'s nested-shape
    ``assert_no_leakage`` re-check).
    """
    split_path = Path(split_path)
    actual_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PinMismatchError(
            f"{split_path} sha256 mismatch: expected {expected_sha256}, got "
            f"{actual_sha256}. This is the authoritative split file "
            "(evalharness/data/corpus_split_200_100.json, Nicky's ruling "
            "2026-07-26) -- if it moved deliberately, update "
            "EXPECTED_SPLIT_SHA256 in config.py as an explicit rebase; do "
            "not proceed against an unpinned split."
        )


def load_eval_papers(split_path: Path, expected_sha16: str) -> set:
    """Load + integrity-check the split; return its ``eval_papers`` arxiv_id set.

    Independently re-checks a sha16 pin (defense in depth, paper-level --
    README "Split & corpus"): historically the ONLY integrity check on
    the frozen ``eval_paper_split.json``, and kept unchanged (same 16-hex
    truncated comparison, same signature) after the 2026-07-26 repoint so
    ``upload_guard.py`` and this module's other existing callers/tests
    keep working byte-for-byte as before. ``build()`` now runs
    ``assert_split_pinned`` (full sha256) immediately before this as the
    PRIMARY pin -- this check is the secondary, redundant one. Reads
    ``eval_papers`` generically off whatever JSON object ``split_path``
    points at, so it works unchanged against corpus_split_200_100.json's
    richer schema (which carries ``eval_papers`` alongside
    ``train_uids``/``holdout_uids``/``train_backfill_7of8_uids``, all
    ignored here) exactly as it did against the old 2-key frozen file.
    """
    split_path = Path(split_path)
    actual_sha16 = hashlib.sha256(split_path.read_bytes()).hexdigest()[:16]
    if actual_sha16 != expected_sha16:
        raise PinMismatchError(
            f"{split_path} sha256[:16] mismatch: expected {expected_sha16}, "
            f"got {actual_sha16}. This split file is a frozen/pinned artifact "
            "-- if it moved deliberately, update EXPECTED_SPLIT_SHA256_16 in "
            "config.py as an explicit rebase; do not proceed against an "
            "unpinned split."
        )
    data = json.loads(split_path.read_text(encoding="utf-8"))
    eval_papers = set(data["eval_papers"])
    if not eval_papers:
        raise ValueError(f"{split_path}: 'eval_papers' is empty -- not a usable frozen split")
    return eval_papers


def load_backfill_uids(split_path: Path) -> list:
    """Read ``train_backfill_7of8_uids`` off the (already pin-verified) split file.

    Called after ``assert_split_pinned``/``load_eval_papers`` have
    already verified ``split_path``'s bytes, so this does not re-hash --
    it only parses. Raises ``ValueError`` if the key is entirely absent
    (a split file predating the 2026-07-26 schema) or is not a list of
    non-empty strings; an empty list (``[]``) is valid and means this
    split needs no backfill.
    """
    split_path = Path(split_path)
    data = json.loads(split_path.read_text(encoding="utf-8"))
    if "train_backfill_7of8_uids" not in data:
        raise ValueError(
            f"{split_path}: missing 'train_backfill_7of8_uids' -- not a "
            "2026-07-26-schema split file"
        )
    backfill_uids = data["train_backfill_7of8_uids"]
    if not isinstance(backfill_uids, list) or any(
        not isinstance(u, str) or not u for u in backfill_uids
    ):
        raise ValueError(
            f"{split_path}: 'train_backfill_7of8_uids' must be a list of "
            f"non-empty str uids (got {backfill_uids!r})"
        )
    return backfill_uids


def assert_backfill_mapping_complete(split_backfill_uids, trace_sources: dict) -> None:
    """Hard-fail unless ``trace_sources`` names EXACTLY the split's backfill uids.

    Set equality, both directions: a uid the split declares but
    ``config.BACKFILL_TRACE_SOURCES`` does not pin means this module
    would silently refuse that uid later (or worse, silently drop it);
    a uid ``BACKFILL_TRACE_SOURCES`` pins but the split no longer
    declares means the pin is stale. Either is a desync between the two
    independently-maintained sources of truth and must hard-fail loudly
    (``BackfillMappingError``), never resolve itself by picking a side.
    """
    split_set = set(split_backfill_uids)
    trace_set = set(trace_sources)
    if split_set == trace_set:
        return
    missing_from_sources = sorted(split_set - trace_set)
    missing_from_split = sorted(trace_set - split_set)
    parts = []
    if missing_from_sources:
        parts.append(
            f"{len(missing_from_sources)} uid(s) in the split's "
            f"train_backfill_7of8_uids but NOT in config.BACKFILL_TRACE_SOURCES: "
            f"{missing_from_sources}"
        )
    if missing_from_split:
        parts.append(
            f"{len(missing_from_split)} uid(s) in config.BACKFILL_TRACE_SOURCES "
            f"but NOT in the split's train_backfill_7of8_uids: {missing_from_split}"
        )
    raise BackfillMappingError(
        "BACKFILL MAPPING DESYNC (" + "; ".join(parts) + "). The split and "
        "config.py's pinned backfill source mapping must name the exact same "
        "uid set -- refusing to guess which side is stale."
    )


def _normalize_provenance(example: dict) -> dict:
    """Return a flat ``{"uid": ..., "arxiv_id": ..., ...}`` view of an example.

    Accepts both the flat shape (``{"uid", "arxiv_id", ...}``) and the
    nested SFT-example shape (``{"provenance": {"uid", "arxiv_id", ...}}``
    -- see module docstring); nested wins when both are present.
    """
    provenance = example.get("provenance")
    if isinstance(provenance, dict):
        return provenance
    return example


def assert_no_leakage(examples, eval_papers, eval_uids) -> None:
    """Hard-fail if any example resolves to an eval paper or an eval uid.

    ``examples`` is an iterable of dicts in either the flat or nested
    provenance shape (see ``_normalize_provenance``). Paper-level
    leakage (``arxiv_id in eval_papers``) and uid-level leakage (``uid
    in eval_uids``) are checked independently; either alone is
    sufficient to hard-fail (README: "eval-paper records are radioactive
    to training").
    """
    eval_papers = eval_papers if isinstance(eval_papers, set) else set(eval_papers)
    eval_uids = eval_uids if isinstance(eval_uids, set) else set(eval_uids)

    paper_offenders = []
    uid_offenders = []
    for example in examples:
        prov = _normalize_provenance(example)
        uid = prov.get("uid")
        arxiv_id = prov.get("arxiv_id")
        if arxiv_id is not None and arxiv_id in eval_papers:
            paper_offenders.append((uid, arxiv_id))
        if uid is not None and uid in eval_uids:
            uid_offenders.append((uid, arxiv_id))

    if not paper_offenders and not uid_offenders:
        return

    parts = []
    if paper_offenders:
        shown = ", ".join(f"{u}(arxiv_id={a})" for u, a in paper_offenders[:5])
        more = "" if len(paper_offenders) <= 5 else f" (+{len(paper_offenders) - 5} more)"
        parts.append(
            f"PAPER-level: {len(paper_offenders)} example(s) whose arxiv_id is "
            f"an eval paper: {shown}{more}"
        )
    if uid_offenders:
        shown = ", ".join(f"{u}(arxiv_id={a})" for u, a in uid_offenders[:5])
        more = "" if len(uid_offenders) <= 5 else f" (+{len(uid_offenders) - 5} more)"
        parts.append(
            f"UID-level: {len(uid_offenders)} example(s) whose uid is in the "
            f"eval set: {shown}{more}"
        )

    raise LeakageError(
        "LEAKAGE GUARD TRIPPED (" + "; ".join(parts) + "). Eval-paper records "
        "are radioactive to training (README 'Split & corpus') -- refusing "
        "to write anything."
    )


def dedupe_examples(examples: list) -> list:
    """Drop exact ``(uid, rollout_uid)`` duplicates, keeping the first occurrence.

    Order-preserving and deterministic: a single pass over ``examples``
    in input order, keeping the first example seen for each key.
    """
    seen = set()
    result = []
    for example in examples:
        prov = _normalize_provenance(example)
        key = (prov.get("uid"), prov.get("rollout_uid"))
        if key in seen:
            continue
        seen.add(key)
        result.append(example)
    return result


def assert_no_cross_uid_statement_dups(records) -> None:
    """Hard-fail if two DIFFERENT uids share a byte-identical FULL statement.

    Keys on the full statement string, never a truncated prefix or hash
    thereof -- finding F4: truncated keys manufacture ghost duplicates
    that don't actually collide. Repeats of the same uid (e.g. several
    correct rollouts of one record) are expected and fine; only a
    statement shared ACROSS uids is a problem.
    """
    statement_to_uids: dict = {}
    for record in records:
        statement = record.get("statement")
        if statement is None:
            continue
        statement_to_uids.setdefault(statement, set()).add(record.get("uid"))

    collisions = {s: uids for s, uids in statement_to_uids.items() if len(uids) > 1}
    if not collisions:
        return

    items = list(collisions.items())[:5]
    shown = "; ".join(f"uids={sorted(u for u in uids if u is not None)}" for _, uids in items)
    more = "" if len(collisions) <= 5 else f" (+{len(collisions) - 5} more)"
    raise DuplicateRecordError(
        f"{len(collisions)} statement(s) shared across DIFFERENT uids (full-"
        f"statement match, finding F4 -- truncated keys manufacture ghosts): "
        f"{shown}{more}"
    )


def assert_verified_correct(example: dict) -> None:
    """Hard-fail unless ``example`` is a verified-correct, verbatim rollout.

    Checks ``provenance.verdict == "correct"``, ``provenance.
    verbatim_output is True``, and a non-empty ``provenance.rollout_uid``
    -- the three guarantees this module's SFT targets rely on (README
    D4: verdict gates correctness, verbatim_output gates the grader-
    equivalence defense, rollout_uid ties the example to its source for
    audit).
    """
    prov = _normalize_provenance(example)
    uid = prov.get("uid", "<unknown>")
    verdict = prov.get("verdict")
    verbatim_output = prov.get("verbatim_output")
    rollout_uid = prov.get("rollout_uid")

    problems = []
    if verdict != "correct":
        problems.append(f"verdict={verdict!r} (must be 'correct')")
    if verbatim_output is not True:
        problems.append(f"verbatim_output={verbatim_output!r} (must be True)")
    if not rollout_uid:
        problems.append(f"rollout_uid={rollout_uid!r} (must be non-empty)")

    if problems:
        raise TraceIntegrityError(
            f"uid={uid}: not a verified-correct verbatim trace: {'; '.join(problems)}"
        )


# ============================================================================
# W2 build: split-derivation guards, rollout harvesting, and the orchestrator
# ============================================================================


def assert_not_retired_path(*paths) -> None:
    """Hard-fail if any given path has a path component starting with "retired".

    Two retired split generations live on disk today:
    ``evalharness/data/retired_20260716/`` (the original 200/100 split,
    retired 2026-07-16) and ``evalharness/data/retired_20260726/`` (the
    frozen ``eval_paper_split.json`` derived-view split, retired on
    Nicky's 2026-07-26 ruling in favor of ``corpus_split_200_100.json``)
    -- silently consuming either would resolve a past ruling by
    accident. The check is a structural path-component scan
    (case-insensitive prefix match), not a hardcoded literal match on
    either directory name, so any ``retired_*`` directory -- past,
    present, or future -- trips it, including the current canonical
    split path itself if it is ever moved under one.
    """
    offenders = []
    for raw_path in paths:
        resolved = Path(raw_path).resolve()
        if any(part.lower().startswith("retired") for part in resolved.parts):
            offenders.append(str(raw_path))
    if offenders:
        raise SplitNotBuiltError(
            "refusing to read from a retired split path (a path component "
            f"starts with 'retired'): {offenders}. Every evalharness/data/"
            "retired_*/ directory holds a superseded split -- do NOT point "
            "this builder at any of them. The current canonical split is "
            "config.EVAL_PAPER_SPLIT_PATH (evalharness/data/"
            "corpus_split_200_100.json as of Nicky's 2026-07-26 ruling)."
        )


def load_corpus(corpus_path: Path) -> list:
    """Parse ``corpus_path`` (band_corpus.jsonl) into a list of row dicts.

    Pure jsonl parsing -- no pin check here; ``assert_corpus_pinned`` is
    a separate, earlier guard (see ``build``'s guard order) so the
    sha256 + row-count identity check always runs before a single row
    from this file is trusted.
    """
    corpus_path = Path(corpus_path)
    rows = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_eval_uids(eval_set_path: Path) -> set:
    """Collect every ``uid`` from ``eval_set.jsonl``.

    Raises ``ValueError`` if the file has zero rows, or if any row lacks
    a non-empty ``uid`` -- a malformed eval set must never silently
    yield a too-small (or empty) uid set, since ``assert_train_eval_
    disjoint`` and the leakage guards rely on this set to catch every
    train/eval collision.
    """
    eval_set_path = Path(eval_set_path)
    uids = set()
    n_rows = 0
    with eval_set_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            row = json.loads(line)
            uid = row.get("uid")
            if not uid:
                raise ValueError(
                    f"{eval_set_path}: row {n_rows} has no non-empty 'uid' (row={row!r})"
                )
            uids.add(uid)
    if n_rows == 0:
        raise ValueError(f"{eval_set_path} contains no rows -- not a usable eval set")
    return uids


def load_eval_statements(eval_set_path: Path) -> set:
    """Collect every ``statement`` from ``eval_set.jsonl``; hard-fail on a blank one.

    Full statement strings only, never a truncated prefix or hash
    thereof (finding F4, same rationale as
    ``assert_no_cross_uid_statement_dups``) -- this set feeds
    ``assert_no_cross_split_statement_dups``. A row with no non-empty
    ``statement`` raises ``ValueError`` rather than being skipped:
    ``evalharness.build_eval_set`` already hard-requires a non-blank
    statement on every eval row, so a blank one here means a malformed
    eval set -- and silently skipping it would quietly shrink the
    statement set the cross-split leakage check relies on (no warn
    mode, same posture as ``load_eval_uids``).
    """
    eval_set_path = Path(eval_set_path)
    statements = set()
    n_rows = 0
    with eval_set_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            row = json.loads(line)
            statement = row.get("statement")
            if not statement:
                raise ValueError(
                    f"{eval_set_path}: row {n_rows} has no non-empty 'statement' "
                    f"(uid={row.get('uid')!r}) -- a malformed eval set must not "
                    "silently weaken the cross-split statement-leakage check."
                )
            statements.add(statement)
    return statements


def assert_train_eval_disjoint(train_uids, eval_uids) -> None:
    """Hard-fail if any uid appears in both the train and eval uid sets.

    Runs before the corpus is even read (see ``build``'s guard order) --
    a train uid list that already collides with ``eval_set.jsonl`` is a
    split-construction bug, not something worth discovering only after
    traces have been harvested.
    """
    train_uids_set = train_uids if isinstance(train_uids, set) else set(train_uids)
    eval_uids_set = eval_uids if isinstance(eval_uids, set) else set(eval_uids)
    offenders = sorted(train_uids_set & eval_uids_set)
    if not offenders:
        return
    shown = ", ".join(offenders[:5])
    more = "" if len(offenders) <= 5 else f" (+{len(offenders) - 5} more)"
    raise LeakageError(
        f"{len(offenders)} uid(s) appear in BOTH train_uids and eval_set: "
        f"{shown}{more}. This is a split-construction bug -- train and eval "
        "must be disjoint at the uid level before anything else runs."
    )


def assert_no_cross_split_statement_dups(train_records, eval_statements) -> None:
    """Hard-fail if any train record's full statement is also an eval statement.

    An eval problem's text showing up in the train set is leakage even
    when it arrives under a different uid or a different (non-eval)
    paper -- the model would still be trained on the literal problem it
    is later scored against. Full-statement match only (finding F4).
    """
    eval_statements_set = (
        eval_statements if isinstance(eval_statements, set) else set(eval_statements)
    )
    offenders = []
    for record in train_records:
        statement = record.get("statement")
        if statement is not None and statement in eval_statements_set:
            offenders.append(record.get("uid"))
    if not offenders:
        return
    shown = ", ".join(str(u) for u in offenders[:5])
    more = "" if len(offenders) <= 5 else f" (+{len(offenders) - 5} more)"
    raise LeakageError(
        f"{len(offenders)} train record(s) share a full statement with an "
        f"eval_set record: {shown}{more}. An eval problem's text inside the "
        "train set is leakage even under a different uid/paper -- refusing "
        "to proceed."
    )


def select_train_records(corpus_rows, train_uids, backfill_uids=()) -> list:
    """Select corpus rows whose uid is a train uid, preserving corpus order.

    ``backfill_uids`` (Nicky's ruling 2026-07-26: the GGUF 7/8 backfill,
    ``config.BACKFILL_TRACE_SOURCES``) is the ONE pinned exception to
    "every train uid must be in the corpus" -- these uids are exempted
    from the missing-from-corpus refusal below because they are
    harvested separately by ``load_backfill_records`` and merged in by
    the caller, never selected from ``corpus_rows`` here. Anything else
    out-of-corpus still hard-fails exactly as before.

    Hard-fails ``ValueError`` on any of five desync symptoms: a
    duplicate uid within ``train_uids`` itself (corrupt uid list --
    ``load_uid_list`` does not dedupe), a duplicate uid across CORPUS
    rows (a dup-uid corpus would select several rows per train uid,
    and ``dedupe_examples`` downstream would silently mask the
    collision instead of surfacing it), a train uid absent from the
    corpus AND not in ``backfill_uids`` (split/corpus desync -- the
    split was built against a different corpus revision), a
    ``backfill_uids`` entry that IS present in the corpus (backfill/
    corpus desync -- a uid pinned as "not in the corpus, harvest from
    the rescore pool instead" turning up in the corpus means the two
    rosters disagree about where its trace lives), or a selected row
    with no non-empty ``statement`` (a corpus row too malformed to ever
    become a training example).
    """
    train_uids_list = list(train_uids)
    backfill_uids_set = set(backfill_uids)
    seen = set()
    duplicates = set()
    for uid in train_uids_list:
        if uid in seen:
            duplicates.add(uid)
        seen.add(uid)
    if duplicates:
        shown_dupes = sorted(duplicates)
        shown = ", ".join(shown_dupes[:5])
        more = "" if len(shown_dupes) <= 5 else f" (+{len(shown_dupes) - 5} more)"
        raise ValueError(
            f"train_uids contains {len(shown_dupes)} duplicate uid(s): "
            f"{shown}{more} -- corrupt uid list."
        )

    by_uid = {}
    corpus_dup_uids = set()
    for row in corpus_rows:
        uid = row.get("uid")
        if uid is None:
            continue
        if uid in by_uid:
            corpus_dup_uids.add(uid)
        else:
            by_uid[uid] = row
    if corpus_dup_uids:
        shown_dupes = sorted(corpus_dup_uids)
        shown = ", ".join(shown_dupes[:5])
        more = "" if len(shown_dupes) <= 5 else f" (+{len(shown_dupes) - 5} more)"
        raise ValueError(
            f"corpus contains {len(shown_dupes)} duplicate uid(s): {shown}{more} "
            "-- corpus integrity violation (dedupe_examples downstream would "
            "silently mask this instead of surfacing it)."
        )

    backfill_in_corpus = sorted(backfill_uids_set & set(by_uid))
    if backfill_in_corpus:
        shown = ", ".join(backfill_in_corpus[:5])
        more = "" if len(backfill_in_corpus) <= 5 else f" (+{len(backfill_in_corpus) - 5} more)"
        raise ValueError(
            f"{len(backfill_in_corpus)} backfill uid(s) unexpectedly found IN "
            f"the corpus: {shown}{more} -- backfill/corpus desync (a uid pinned "
            "as 'harvest from the rescore pool, not the corpus' turned up in "
            "band_corpus.jsonl; the backfill roster or the corpus moved out "
            "from under the other)."
        )

    missing = [
        uid for uid in train_uids_list if uid not in by_uid and uid not in backfill_uids_set
    ]
    if missing:
        shown = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ValueError(
            f"{len(missing)} train uid(s) not found in the corpus: {shown}{more} "
            "-- split/corpus desync (the split was likely built against a "
            "different corpus revision than the pinned one), and not covered "
            "by the pinned backfill roster either."
        )

    train_uids_set = seen - backfill_uids_set
    records = [row for row in corpus_rows if row.get("uid") in train_uids_set]

    blank = [r.get("uid") for r in records if not r.get("statement")]
    if blank:
        shown = ", ".join(str(u) for u in blank[:5])
        more = "" if len(blank) <= 5 else f" (+{len(blank) - 5} more)"
        raise ValueError(
            f"{len(blank)} selected train record(s) have no non-empty 'statement': "
            f"{shown}{more}."
        )

    return records


def load_backfill_records(trace_sources: dict, repo_root: Path) -> list:
    """Load + validate the pinned GGUF 7/8 backfill rows, corpus-row-shaped.

    For each ``uid -> source_file`` pin in ``trace_sources``
    (``config.BACKFILL_TRACE_SOURCES``): read ``repo_root /
    source_file`` (a ``pass_at_k.jsonl`` scoring-run output, NOT a
    corpus file) and find its row for ``uid``. Hard-fails
    ``TraceIntegrityError`` if the uid is absent from that file, appears
    more than once (ambiguous -- unlike ``_progress/rollouts.jsonl``,
    a ``pass_at_k.jsonl`` is a one-row-per-record scoring output, not a
    documented append-across-passes log, so a duplicate here is an
    integrity problem, not append-log noise), ``n_correct != 7``, or
    ``label != "solved"`` (the "GGUF 7/8" guarantee the backfill ruling
    depends on).

    Each returned record is the source row AS-IS plus one synthesized
    key, ``corpus_provenance: {"source_file": source_file}`` -- these
    rows never carry ``corpus_provenance`` themselves (they predate
    corpus assembly), so this synthesizes just enough shape for every
    downstream guard/harvest function (``rollouts_path_for``,
    ``reconcile_record``, ``harvest_correct_traces``,
    ``build_sft_example``, the leakage/statement-dup guards) to treat a
    backfill record identically to a real corpus record, with zero
    special-casing in any of them. ``record["answer"]`` is copied over
    unchanged (present in the source row) but -- as with every other
    record -- ``build_sft_example`` never reads it (D4).

    Returns records sorted by uid (deterministic; ``trace_sources``
    iteration order is not guaranteed across dict literals/JSON).
    """
    repo_root = Path(repo_root)
    records = []
    for uid, source_file in sorted(trace_sources.items()):
        source_path = repo_root / source_file
        if not source_path.exists():
            raise TraceIntegrityError(
                f"backfill uid={uid}: pinned source file {source_file} does not "
                "exist on disk -- BACKFILL_TRACE_SOURCES in config.py points at "
                "a file that moved or was never there."
            )
        matches = []
        with source_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("uid") == uid:
                    matches.append(row)
        if not matches:
            raise TraceIntegrityError(
                f"backfill uid={uid}: no row found in pinned source {source_file} "
                "-- the file does not contain this uid."
            )
        if len(matches) > 1:
            raise TraceIntegrityError(
                f"backfill uid={uid}: {len(matches)} rows found in pinned source "
                f"{source_file} -- ambiguous (pass_at_k.jsonl is a one-row-per-"
                "record scoring output; a duplicate uid here is a data-integrity "
                "problem, not append-log noise like rollouts.jsonl)."
            )
        row = matches[0]

        if row.get("n_correct") != 7:
            raise TraceIntegrityError(
                f"backfill uid={uid}: n_correct={row.get('n_correct')!r} (must be "
                f"7 -- the 'GGUF 7/8' guarantee) at {source_file}."
            )
        if row.get("label") != "solved":
            raise TraceIntegrityError(
                f"backfill uid={uid}: label={row.get('label')!r} (must be "
                f"'solved') at {source_file}."
            )
        if not row.get("statement"):
            raise ValueError(
                f"backfill uid={uid}: no non-empty 'statement' at {source_file}."
            )

        record = dict(row)
        record["corpus_provenance"] = {"source_file": source_file}
        records.append(record)

    return records


def rollouts_path_for(source_file: str, repo_root: Path) -> Path:
    """Resolve a corpus row's ``corpus_provenance.source_file`` to its rollouts path.

    The rollout file for a scoring run's ``pass_at_k.jsonl`` lives at its
    sibling ``_progress/rollouts.jsonl``. Existence is deliberately NOT
    checked here (cross-review gate, 2026-07-25): a missing routed file
    is treated as simply non-reconciling, so resolution falls through to
    the registry scan in ``reconcile_record`` -- the same protocol that
    already covers the 15 real rows whose routed file exists but does
    not account for their counts. If nothing on disk reconciles, the
    zero-candidate hard fail reports the routed file's absence
    explicitly; a missing file never surfaces as a bare
    ``FileNotFoundError`` mid-resolution.
    """
    repo_root = Path(repo_root)
    return repo_root / Path(source_file).parent / "_progress" / "rollouts.jsonl"


# Where rollouts files may legitimately live, relative to repo_root. The
# registry a record may reconcile against = its routed file + every routed
# file of every other selected record + every existing match of these
# globs. Two globs, not one: 2026-07-25 measurement found corpus rows
# whose source_file lies OUTSIDE out/remote_rescore/ (the 2026-07-16
# repair-lane audit dir), and 15 rows whose traces live in rerun dirs no
# source_file references -- glob discovery alone misses the former,
# source_file routing alone misses the latter, so both feed the registry.
REGISTRY_GLOBS = (
    "out/remote_rescore/*/_progress/rollouts.jsonl",
    "out/audits/extraction_defect_check_*/repair_lane/passk_rescore/_progress/rollouts.jsonl",
)

RECONCILED_VIA_ROUTED = "routed"
RECONCILED_VIA_UNIQUE_ALTERNATIVE = "unique_alternative"


def load_rollout_file(path: Path) -> tuple:
    """Load ONE rollouts.jsonl into a last-occurrence ``(uid, rollout_uid) -> row`` map.

    Returns ``(index, duplicate_entries)`` where ``duplicate_entries`` is
    the number of lines that re-keyed an already-seen ``(uid,
    rollout_uid)``. Duplicates are NOT an error: rollouts.jsonl files are
    append-across-passes logs (module docstring "Rollout reconciliation"
    -- tier1_band alone carries 651 duplicate entries), and a rescore
    pass re-samples under the same rollout_uid. LAST occurrence wins
    because the later pass is the one whose tallies the corpus row was
    stamped from -- measured: last-occurrence reconciles 278/293 rows at
    their routed file vs 246 for first-occurrence. Per-key content
    disagreement across occurrences is resolved by ``reconcile_record``'s
    tally check, never by trusting file order alone.
    """
    path = Path(path)
    index: dict = {}
    duplicate_entries = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row.get("uid"), row.get("rollout_uid"))
            if key in index:
                duplicate_entries += 1
            index[key] = row  # last occurrence wins
    return index, duplicate_entries


def discover_registry(routed_paths, repo_root: Path) -> list:
    """Return every rollouts file a record may reconcile against, sorted.

    Union of the records' own routed files and every match of
    ``REGISTRY_GLOBS`` under ``repo_root``, filtered to files that exist
    -- a routed path that is not on disk simply contributes nothing to
    the registry, so its records fall through to the alternative scan in
    ``reconcile_record`` (cross-review gate: a missing routed file is
    non-reconciling, never a load-time file error). Sorted by path
    string so the registry -- and therefore ambiguity detection and the
    manifest's registry echo -- is deterministic for a given disk state.
    A file appearing on disk later can only ADD reconciliation
    candidates, which either changes nothing (the routed file already
    reconciles) or trips the >=2-candidates ambiguity hard-fail in
    ``reconcile_record`` -- growth of the registry can never silently
    reroute a row.
    """
    repo_root = Path(repo_root)
    registry = {Path(p) for p in routed_paths}
    for pattern in REGISTRY_GLOBS:
        registry.update(repo_root.glob(pattern))
    return sorted((p for p in registry if p.exists()), key=str)


def _tally_against_index(record: dict, index: dict):
    """Verdict tally of ``record``'s rollout_uids against one file's index.

    Returns ``{"correct": n, "wrong": n, "degenerate": n}`` if every
    listed rollout_uid resolves in ``index`` with a matching ``uid``
    field, else ``None`` (this file cannot account for the record).
    Shared by ``reconcile_record`` (where tally == the corpus row's
    counts defines "reconciles") and kept deliberately tiny so the
    reconciliation criterion exists in exactly one place.
    """
    uid = record.get("uid")
    tallies = {"correct": 0, "wrong": 0, "degenerate": 0}
    for rollout_uid in record.get("rollout_uids") or []:
        row = index.get((uid, rollout_uid))
        if row is None or row.get("uid") != uid:
            return None
        verdict = row.get("verdict")
        if verdict in tallies:
            tallies[verdict] += 1
    return tallies


def _record_reconciles(record: dict, index: dict) -> bool:
    tallies = _tally_against_index(record, index)
    if tallies is None:
        return False
    return tallies == {
        "correct": record.get("n_correct"),
        "wrong": record.get("n_wrong"),
        "degenerate": record.get("n_degenerate"),
    }


def _expected_counts(record: dict) -> dict:
    return {
        "correct": record.get("n_correct"),
        "wrong": record.get("n_wrong"),
        "degenerate": record.get("n_degenerate"),
    }


def _diagnose_candidate(record: dict, index: dict) -> str:
    """One-line diagnosis of WHY a file does (not) account for ``record``.

    Feeds the refusal messages in ``reconcile_record`` (cross-review
    gate: refusals must name expected counts AND per-candidate observed
    state, so an operator can see at a glance which file failed how).
    """
    uid = record.get("uid")
    rollout_uids = record.get("rollout_uids") or []
    missing = sum(
        1
        for rollout_uid in rollout_uids
        if (row := index.get((uid, rollout_uid))) is None or row.get("uid") != uid
    )
    if missing:
        return f"missing-or-uid-mismatched {missing}/{len(rollout_uids)} rollout_uids"
    return f"observed tally {_tally_against_index(record, index)}"


def reconcile_record(record: dict, indexes_by_path: dict, routed_path: Path) -> tuple:
    """Resolve ``record`` to its authoritative rollouts file.

    Returns ``(index, path, via)`` where ``via`` is ``"routed"`` when the
    record's own routed file reconciles (the common case: 278/293 on the
    pinned corpus), or ``"unique_alternative"`` when it does not and
    EXACTLY ONE other registry file does (measured: 15/293, each with
    exactly one candidate). A routed file that is MISSING from
    ``indexes_by_path`` (not on disk) is treated as non-reconciling and
    falls through to the same registry scan -- never a bare file error
    mid-resolution. Zero reconciling files, or two or more, is a
    ``TraceIntegrityError`` naming the expected counts and each
    candidate's observed state: zero means no pass on disk accounts for
    the corpus row's counts; two-plus means the trace's origin is
    ambiguous and picking one silently would un-anchor the label<->trace
    tie D4 depends on.
    """
    routed_path = Path(routed_path)
    routed_index = indexes_by_path.get(routed_path)
    if routed_index is not None and _record_reconciles(record, routed_index):
        return routed_index, routed_path, RECONCILED_VIA_ROUTED

    candidates = [
        path
        for path, index in sorted(indexes_by_path.items(), key=lambda kv: str(kv[0]))
        if path != routed_path and _record_reconciles(record, index)
    ]
    if len(candidates) == 1:
        path = candidates[0]
        return indexes_by_path[path], path, RECONCILED_VIA_UNIQUE_ALTERNATIVE

    routed_state = (
        "MISSING from disk"
        if routed_index is None
        else _diagnose_candidate(record, routed_index)
    )
    if not candidates:
        per_candidate = "; ".join(
            f"{path}: {_diagnose_candidate(record, index)}"
            for path, index in sorted(indexes_by_path.items(), key=lambda kv: str(kv[0]))
            if path != routed_path
        )
        raise TraceIntegrityError(
            f"uid={record.get('uid')}: reconciles in NO registry file. "
            f"Expected counts {_expected_counts(record)}; routed {routed_path}: "
            f"{routed_state}; alternatives: [{per_candidate or 'none'}] -- no "
            "pass on disk accounts for this corpus row's "
            "n_correct/n_wrong/n_degenerate."
        )
    raise TraceIntegrityError(
        f"uid={record.get('uid')}: ambiguous trace origin. Expected counts "
        f"{_expected_counts(record)}; routed {routed_path}: {routed_state}; "
        f"{len(candidates)} alternative registry files ALL reconcile: "
        f"{[str(p) for p in candidates]} -- refusing to guess which pass "
        "produced the corpus label."
    )


def build_sft_example(
    record: dict,
    rollout: dict,
    corpus_sha256: str,
    trace_file: str,
    reconciled_via: str,
    backfill_7of8: bool = False,
) -> dict:
    """Build one SFT example from a train record + its verified-correct rollout.

    NEVER reads ``record["answer"]`` -- the explicit key access below is
    the proof (D4 grader-equivalence defense): the assistant target is
    ``rollout["output"]`` copied verbatim, never the corpus's canonical
    answer string. Must work even when ``record`` has no ``"answer"``
    key at all (exercised directly by tests). ``trace_file`` /
    ``reconciled_via`` record which rollouts file ACTUALLY supplied this
    trace and how it reconciled (module docstring "Rollout
    reconciliation") -- ``source_file`` stays the corpus row's original
    claim so the two are separately auditable. ``backfill_7of8`` stamps
    whether this example came from the pinned GGUF 7/8 backfill roster
    (Nicky's ruling 2026-07-26) rather than ``band_corpus.jsonl`` proper
    -- always present and explicit (``True``/``False``), never inferred
    from absence, so a manifest/dataset consumer never has to guess.

    Schema v2 (defect-1 fix, 2026-07-29): ``prompt`` / ``completion``
    message-list columns replace v1's single ``messages`` list. v1
    pre-templated all three messages into one string box-side, which fed
    SFTTrainer a language-modeling dataset -- full-sequence loss, no
    completion masking, 21.6% of trained characters on system+user text.
    In the pinned trl 0.29.1 a prompt/completion dataset gets
    completion-only loss BY DEFAULT: the trainer renders ``prompt`` with
    ``apply_chat_template(..., add_generation_prompt=True)``, renders
    ``prompt + completion`` through the same template v1 used (so the
    TOTAL rendered training text is byte-identical to v1's -- the loss
    mask is the only train-time delta), and sets every prompt token's
    label to -100. The message contents themselves are unchanged from
    v1: same wire-format pins, same verbatim ``rollout["output"]``
    target. (``assistant_only_loss`` is NOT usable here: it requires a
    ``{% generation %}``-tagged chat template, which Qwen3's lacks.)
    """
    return {
        "prompt": [
            {"role": "system", "content": config.PASS_AT_K_SYSTEM_PROMPT},
            {"role": "user", "content": record["statement"] + config.PASS_AT_K_NO_THINK_SUFFIX},
        ],
        "completion": [
            {"role": "assistant", "content": rollout["output"]},
        ],
        "provenance": {
            "uid": record["uid"],
            "rollout_uid": rollout["rollout_uid"],
            "sample_idx": rollout["sample_idx"],
            "arxiv_id": record["arxiv_id"],
            "source_file": record["corpus_provenance"]["source_file"],
            "trace_file": trace_file,
            "reconciled_via": reconciled_via,
            "verdict": "correct",
            "verbatim_output": True,
            "corpus_sha256": corpus_sha256,
            "backfill_7of8": backfill_7of8,
        },
    }


def harvest_correct_traces(
    records, authoritative, corpus_sha256: str, repo_root: Path, backfill_uids=frozenset()
) -> list:
    """Harvest one SFT example per verified-correct rollout of each record.

    ``authoritative`` maps each record's uid to its
    ``(index, path, via)`` triple from ``reconcile_record``. For every
    record: every listed ``rollout_uids`` entry must resolve in its
    authoritative index and each resolved rollout's own ``uid`` field
    must equal the record's uid; the per-verdict tally over exactly
    those rollouts must equal the record's own ``n_correct``/
    ``n_wrong``/``n_degenerate``; and ``n_correct`` must be >= 1 (a band
    record with zero correct traces contradicts band membership by
    construction). All three are structurally guaranteed by a
    successful ``reconcile_record`` (reconciliation IS the
    presence+uid+tally check) -- they are re-asserted here as defense
    in depth, same idiom as ``build``'s nested-shape
    ``assert_no_leakage`` re-check, and directly exercisable by unit
    tests that hand-build an ``authoritative`` map. ``backfill_uids``
    (default empty, so existing direct callers are unaffected) marks
    which uids' examples get ``provenance.backfill_7of8 = True`` --
    everything else (including any uid not in this set) gets ``False``.

    Deterministic output order: records in the given order, and within a
    record its correct rollouts ordered by ``sample_idx``.
    """
    repo_root = Path(repo_root)
    backfill_uids = set(backfill_uids)
    examples = []
    for record in records:
        uid = record.get("uid")
        rollout_uids = record.get("rollout_uids") or []
        if uid not in authoritative:
            raise TraceIntegrityError(
                f"uid={uid}: no authoritative rollouts file recorded -- "
                "reconcile_record must run before harvest."
            )
        index, trace_path, via = authoritative[uid]
        trace_path = Path(trace_path)
        try:
            trace_file = trace_path.relative_to(repo_root).as_posix()
        except ValueError:
            trace_file = str(trace_path)

        resolved = []
        for rollout_uid in rollout_uids:
            key = (uid, rollout_uid)
            rollout = index.get(key)
            if rollout is None:
                raise TraceIntegrityError(
                    f"uid={uid}: rollout_uid={rollout_uid!r} not found in its "
                    f"authoritative rollouts file {trace_file} -- corpus row "
                    "lists a rollout that file does not contain."
                )
            if rollout.get("uid") != uid:
                raise TraceIntegrityError(
                    f"uid={uid}: rollout_uid={rollout_uid!r} resolved to a "
                    f"rollout row whose own uid is {rollout.get('uid')!r} -- "
                    "forged or misrouted rollout_uid pointing at another "
                    "record's rollout."
                )
            resolved.append(rollout)

        tallies = {"correct": 0, "wrong": 0, "degenerate": 0}
        for rollout in resolved:
            verdict = rollout.get("verdict")
            if verdict in tallies:
                tallies[verdict] += 1

        expected = {
            "correct": record.get("n_correct"),
            "wrong": record.get("n_wrong"),
            "degenerate": record.get("n_degenerate"),
        }
        if tallies != expected:
            raise TraceIntegrityError(
                f"uid={uid}: rollout verdict tallies {tallies} at {trace_file} "
                f"do not match the corpus row's n_correct/n_wrong/n_degenerate "
                f"{expected} -- corpus row and rollouts have desynced."
            )

        if tallies["correct"] < 1:
            raise TraceIntegrityError(
                f"uid={uid}: n_correct=0 -- a band record with zero correct "
                "rollout traces contradicts band membership."
            )

        correct_rollouts = sorted(
            (r for r in resolved if r.get("verdict") == "correct"),
            key=lambda r: r.get("sample_idx"),
        )
        for rollout in correct_rollouts:
            output = rollout.get("output")
            if not output:
                raise TraceIntegrityError(
                    f"uid={uid}: rollout_uid={rollout.get('rollout_uid')!r} is "
                    "verdict=correct but has an empty 'output' -- cannot "
                    "harvest an empty target."
                )
            examples.append(
                build_sft_example(
                    record, rollout, corpus_sha256, trace_file, via,
                    backfill_7of8=uid in backfill_uids,
                )
            )

    return examples


def assert_verbatim_targets(examples, rollout_index) -> None:
    """Hard-fail unless every example's assistant content is byte-identical to its source.

    This is D4's "the builder asserts targets are verbatim rollout
    outputs" made concrete: the ``verbatim_output`` provenance flag is a
    promise, and this function is what makes the promise checkable
    rather than merely stamped. Schema v2: the assistant message lives
    in the example's ``completion`` column (its single message).
    """
    for example in examples:
        prov = example["provenance"]
        key = (prov["uid"], prov["rollout_uid"])
        rollout = rollout_index.get(key)
        if rollout is None:
            raise TraceIntegrityError(
                f"assert_verbatim_targets: no rollout found in the index for "
                f"{key} -- cannot verify verbatim-ness."
            )
        completion = example.get("completion") or []
        assistant_content = next(
            (m["content"] for m in completion if m.get("role") == "assistant"), None
        )
        if assistant_content is None:
            raise TraceIntegrityError(
                f"uid={prov['uid']} rollout_uid={prov['rollout_uid']}: example "
                "has no assistant message in its 'completion' column -- cannot "
                "verify verbatim-ness."
            )
        source_output = rollout["output"]
        if assistant_content != source_output:
            raise TraceIntegrityError(
                f"uid={prov['uid']} rollout_uid={prov['rollout_uid']}: assistant "
                "message content is not byte-identical to its source rollout "
                f"output (len {len(assistant_content)} vs {len(source_output)}) "
                "-- verbatim-target guarantee broken."
            )


def assert_prompt_completion_wellformed(examples) -> None:
    """Hard-fail unless every example's prompt/completion columns are well-formed.

    The defect-1 invariant made checkable: ``prompt`` must be exactly
    ``[system, user]`` with the system content equal to the pinned
    ``config.PASS_AT_K_SYSTEM_PROMPT`` and the user content a non-empty
    string ending in the pinned ``config.PASS_AT_K_NO_THINK_SUFFIX``
    (wire-format byte-identity, AGENTS.md invariant 2 / README D4);
    ``completion`` must be exactly ``[assistant]`` with non-empty string
    content. An empty prompt or completion would silently train on
    nothing (or on everything, if a consumer fell back to concatenation)
    -- this is the guard whose absence let defect 1 ship.
    """
    for example in examples:
        prov = _normalize_provenance(example)
        uid = prov.get("uid", "<unknown>")
        problems = []

        prompt = example.get("prompt")
        if (
            not isinstance(prompt, list)
            or len(prompt) != 2
            or prompt[0].get("role") != "system"
            or prompt[1].get("role") != "user"
        ):
            problems.append(
                f"prompt must be exactly [system, user] message dicts (got {prompt!r})"
            )
        else:
            system_content = prompt[0].get("content")
            user_content = prompt[1].get("content")
            if system_content != config.PASS_AT_K_SYSTEM_PROMPT:
                problems.append(
                    f"system content {system_content!r} != the pinned "
                    "PASS_AT_K_SYSTEM_PROMPT (wire-format drift)"
                )
            if not isinstance(user_content, str) or not user_content.strip():
                problems.append(f"user content must be a non-empty str (got {user_content!r})")
            elif not user_content.endswith(config.PASS_AT_K_NO_THINK_SUFFIX):
                problems.append(
                    f"user content does not end with the pinned no-think suffix "
                    f"{config.PASS_AT_K_NO_THINK_SUFFIX!r} (wire-format drift)"
                )

        completion = example.get("completion")
        if (
            not isinstance(completion, list)
            or len(completion) != 1
            or completion[0].get("role") != "assistant"
        ):
            problems.append(
                f"completion must be exactly [assistant] message dict (got {completion!r})"
            )
        elif not isinstance(completion[0].get("content"), str) or not completion[0]["content"]:
            problems.append(
                f"assistant content must be a non-empty str (got {completion[0].get('content')!r})"
            )
        elif "<think>" in completion[0]["content"] or "</think>" in completion[0]["content"]:
            # Latent wire-format hazard found during the v2 masking proof
            # (2026-07-29): Qwen3's chat template SPLITS assistant content on
            # '</think>' and re-normalizes it (strip/lstrip of newlines)
            # before rendering, so a think tag inside a stored-verbatim
            # target would be silently rewritten between the dataset bytes
            # and the trained bytes -- breaking the D4 verbatim<->trained
            # correspondence without any guard noticing. Measured 0/700
            # harvested traces carry these tags (LM Studio strips the think
            # block before the rollout 'output' field), so this refusal
            # changes nothing today; it exists to make the hazard loud if a
            # future harvest ever ships one.
            problems.append(
                "assistant content contains a '<think>'/'</think>' tag -- the "
                "Qwen3 chat template would split and re-normalize it at "
                "render time, silently changing the trained bytes vs the "
                "stored verbatim target"
            )

        if problems:
            raise TraceIntegrityError(
                f"uid={uid}: malformed prompt/completion example: {'; '.join(problems)}"
            )


def _selection_rank(seed: int, uid, rollout_uid) -> str:
    """Deterministic per-trace rank key for the cap policies.

    sha256 over ``"{seed}:{uid}:{rollout_uid}"`` -- a pure function of
    the seed and the trace's identity, so selection is reproducible from
    the manifest alone, independent of file order, dict order, or RNG
    state ("do not take 'first N' silently" -- work order 2026-07-29).
    """
    return hashlib.sha256(f"{seed}:{uid}:{rollout_uid}".encode("utf-8")).hexdigest()


SELECTION_RULE = (
    "per uid: rank traces by sha256('{seed}:{uid}:{rollout_uid}') ascending, "
    "keep the first min(cap, n_traces); kept rows stay in harvest order "
    "(sample_idx ascending within uid)"
)


def apply_weight_policy(examples: list, *, policy: str, cap_k: int, seed: int) -> tuple:
    """Apply the defect-2 gradient-weight policy; return ``(kept, policy_block)``.

    v1 emitted one row per verified-correct trace, so a record's gradient
    mass equaled ``n_correct`` -- how often the BASE model already solved
    it (anti-difficulty weighting; see config.WEIGHT_POLICY). Policies:

    - ``cap1``: keep exactly one trace per uid.
    - ``capk``: keep at most ``cap_k`` traces per uid.
    - ``inverse``: keep every trace; stamp each kept example with a
      top-level ``weight = 1/n`` (n = that uid's trace count post-dedupe,
      == the corpus row's ``n_correct``) so every uid carries equal total
      gradient mass. The weight is exactly representable/round-trippable
      as a JSON double; ``assert_weight_policy_honored`` re-checks
      ``weight == 1.0/n`` on exact equality.

    Selection under the cap policies follows ``SELECTION_RULE`` (seeded
    sha256 rank -- deterministic, manifest-recorded, never file order).
    Kept examples preserve their harvest order; examples are never
    mutated in place (kept rows are shallow-copied when a weight is
    stamped). ``policy_block`` is the manifest's ``weight_policy`` audit
    block: policy, label, cap, seed, selection rule, rows before/after,
    and the per-uid rows-per-uid histograms before/after.
    """
    if policy not in config.VALID_WEIGHT_POLICIES:
        raise WeightPolicyError(
            f"unknown weight policy {policy!r} -- must be one of "
            f"{config.VALID_WEIGHT_POLICIES}"
        )
    if policy == "capk" and (
        not isinstance(cap_k, int) or isinstance(cap_k, bool) or cap_k < 1
    ):
        raise WeightPolicyError(f"capk requires a positive int cap_k (got {cap_k!r})")

    by_uid: dict = {}
    for example in examples:
        prov = _normalize_provenance(example)
        by_uid.setdefault(prov.get("uid"), []).append(example)

    cap = {"cap1": 1, "capk": cap_k}.get(policy)  # None for inverse
    kept = []
    for uid, uid_examples in by_uid.items():
        if cap is None:  # inverse: keep all, equalize per-uid gradient mass
            weight = 1.0 / len(uid_examples)
            for example in uid_examples:
                stamped = dict(example)
                stamped["weight"] = weight
                kept.append(stamped)
            continue
        ranked = sorted(
            uid_examples,
            key=lambda ex: _selection_rank(
                seed, uid, _normalize_provenance(ex).get("rollout_uid")
            ),
        )
        selected_keys = {
            _normalize_provenance(ex).get("rollout_uid") for ex in ranked[:cap]
        }
        kept.extend(
            ex
            for ex in uid_examples
            if _normalize_provenance(ex).get("rollout_uid") in selected_keys
        )

    # Restore global harvest order (records order, sample_idx within record):
    # the per-uid grouping above preserved it within uids, and dict insertion
    # order preserved it across uids, so `kept` is already in harvest order
    # for the cap policies; the inverse branch appended in place too.

    def _histogram(rows_per_uid_counts) -> dict:
        hist: dict = {}
        for count in rows_per_uid_counts:
            hist[count] = hist.get(count, 0) + 1
        return {str(k): hist[k] for k in sorted(hist)}

    policy_block = {
        "policy": policy,
        "label": config.weight_policy_label(policy, cap_k),
        "cap_k": cap_k if policy == "capk" else None,
        "seed": seed,
        "selection_rule": SELECTION_RULE if cap is not None else
            "keep all traces; per-row weight = 1/n_traces(uid)",
        "rows_before": len(examples),
        "rows_after": len(kept),
        "n_uids": len(by_uid),
        "rows_per_uid_before": _histogram(len(v) for v in by_uid.values()),
        "rows_per_uid_after": _histogram(
            sum(1 for ex in kept if _normalize_provenance(ex).get("uid") == uid)
            for uid in by_uid
        ),
        "weighted": cap is None,
    }
    return kept, policy_block


def assert_weight_policy_honored(examples, *, policy: str, cap_k: int) -> None:
    """Hard-fail unless ``examples`` satisfy the declared weight policy.

    Re-verification counterpart of ``apply_weight_policy`` (same idiom as
    the verbatim/leakage re-checks -- the property is asserted on the
    final example list AND re-asserted from disk by
    ``verify_written_dataset``): per-uid row counts within the cap for
    the cap policies (and no stray ``weight`` field), and for
    ``inverse`` every row of a uid carrying exactly ``weight == 1.0/n``
    where n is that uid's row count.
    """
    if policy not in config.VALID_WEIGHT_POLICIES:
        raise WeightPolicyError(
            f"unknown weight policy {policy!r} -- must be one of "
            f"{config.VALID_WEIGHT_POLICIES}"
        )

    by_uid: dict = {}
    for example in examples:
        prov = _normalize_provenance(example)
        by_uid.setdefault(prov.get("uid"), []).append(example)

    problems = []
    if policy in ("cap1", "capk"):
        cap = 1 if policy == "cap1" else cap_k
        offenders = {uid: len(v) for uid, v in by_uid.items() if len(v) > cap}
        if offenders:
            shown = dict(list(sorted(offenders.items()))[:5])
            problems.append(
                f"{len(offenders)} uid(s) exceed the {policy} cap of {cap} "
                f"rows/uid: {shown}"
            )
        weighted = [
            _normalize_provenance(ex).get("uid")
            for ex in examples
            if "weight" in ex
        ]
        if weighted:
            problems.append(
                f"{len(weighted)} example(s) carry a 'weight' field under the "
                f"unweighted {policy} policy (first: {weighted[:3]})"
            )
    else:  # inverse
        for uid, uid_examples in sorted(by_uid.items(), key=lambda kv: str(kv[0])):
            expected = 1.0 / len(uid_examples)
            bad = [ex.get("weight") for ex in uid_examples if ex.get("weight") != expected]
            if bad:
                problems.append(
                    f"uid={uid}: {len(bad)} row(s) with weight != 1/{len(uid_examples)} "
                    f"(expected {expected!r}, got {bad[:3]!r})"
                )
                if len(problems) >= 5:
                    break

    if problems:
        raise WeightPolicyError(
            "WEIGHT POLICY VIOLATED (" + "; ".join(problems) + ") -- the "
            "dataset does not satisfy its declared policy; refusing."
        )


def verify_written_dataset(
    dataset_path: Path, rollout_index: dict, *, policy: str = None, cap_k: int = None
) -> int:
    """Post-write audit: re-read ``dataset_path`` and re-verify every row from disk.

    The final check in the D4 chain: ``verbatim_output`` is checked
    here, not just stamped by ``build_sft_example`` and asserted against
    in-memory examples by ``assert_verbatim_targets`` -- reading the
    bytes actually written to disk closes the loop against any bug
    between assembling ``examples`` and serializing them (encoding,
    truncation, a stray transform). Every row is also re-checked for
    prompt/completion well-formedness (defect-1 invariant), and -- when
    ``policy`` is given -- the whole file is re-checked against the
    declared weight policy (defect-2 invariant). Returns the verified
    row count.
    """
    dataset_path = Path(dataset_path)
    rows = []
    with dataset_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceIntegrityError(f"{dataset_path}:{lineno}: invalid JSON ({exc})") from exc
            assert_verified_correct(example)
            assert_prompt_completion_wellformed([example])
            assert_verbatim_targets([example], rollout_index)
            rows.append(example)
    if policy is not None:
        assert_weight_policy_honored(rows, policy=policy, cap_k=cap_k)
    return len(rows)


def write_dataset(examples, dataset_path: Path) -> None:
    """Write ``examples`` to ``dataset_path``, one ``json.dumps`` object per line.

    Plain ``json.dumps(example)`` with default (ascii-escaping) output --
    matches ``evalharness.build_eval_set``'s jsonl-writing convention.
    Verbatim-ness is a property of the DECODED string, not the on-disk
    escaping, so ascii-escaping the bytes on disk does not conflict with
    D4 (tests assert on the decoded content, per module contract).
    """
    dataset_path = Path(dataset_path)
    with dataset_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example) + "\n")


def _tally_verdicts(records) -> dict:
    """Sum n_correct/n_wrong/n_degenerate over a set of corpus records.

    Feeds the manifest's ``verdict_totals`` -- tallied over every train
    record's OWN rollouts (all of them, not just the harvested correct
    ones), so the manifest states the full correct/wrong/degenerate
    composition of the training band, not just what got emitted.
    """
    totals = {"correct": 0, "wrong": 0, "degenerate": 0}
    for record in records:
        totals["correct"] += record.get("n_correct") or 0
        totals["wrong"] += record.get("n_wrong") or 0
        totals["degenerate"] += record.get("n_degenerate") or 0
    return totals


def build_manifest(
    *,
    seed: int,
    corpus_path: Path,
    corpus_sha256: str,
    corpus_rows: int,
    split_path: Path,
    split_sha256: str,
    n_eval_papers: int,
    train_uids_path: Path,
    train_uids: list,
    eval_set_path: Path,
    n_eval_rows: int,
    dataset_path: Path,
    dataset_sha256: str,
    examples: list,
    verdict_totals: dict,
    duplicates_dropped: int,
    reconciliation: dict,
    backfill: dict,
    weight_policy: dict,
    guards: list,
) -> dict:
    """Assemble the ``dataset_manifest.json`` dict.

    Hashes ``train_uids_path`` and ``eval_set_path`` itself (via
    ``sha256_file``) so every input this build consumed is captured by a
    sha, not just the pinned corpus/split. ``per_uid_trace_counts``
    counts from ``train_uids`` (every train uid, not just uids that
    happen to appear in ``examples``), so a future change that loosens
    the ``n_correct >= 1`` guard in ``harvest_correct_traces`` cannot
    silently produce a uid with zero examples that this manifest hides.
    ``reconciliation`` (assembled by ``build``) records how every uid's
    traces were resolved -- per-uid trace_file + reconciled_via, the
    routed/unique_alternative totals, and the full registry (path,
    sha256, duplicate_entries per file) -- so a rebuild against a
    changed disk is diffable down to the exact rollouts bytes consumed.
    ``backfill`` (assembled by ``build``, Nicky's ruling 2026-07-26) is
    the GGUF 7/8 backfill audit block: uids, pinned source files + their
    shas, and per-uid harvested trace counts -- empty-shaped
    (``{"uids": [], "sources": [], "per_uid_trace_counts": {}}``) when
    this build's split has no backfill roster. ``weight_policy`` is
    ``apply_weight_policy``'s audit block (defect-2 fix: policy, cap,
    seed, selection rule, rows/histograms before and after).
    ``trainer_hyperparams`` echoes every training hyperparameter --
    including the four that were silent in v1 (grad-accum, scheduler,
    warmup, weight decay; config.py "Trainer schedule/accumulation
    pins") and the completion-only-loss contract -- so the dataset
    manifest alone states how its rows are meant to be consumed.
    """
    per_uid_trace_counts = {uid: 0 for uid in train_uids}
    for example in examples:
        uid = example["provenance"]["uid"]
        per_uid_trace_counts[uid] = per_uid_trace_counts.get(uid, 0) + 1

    return {
        "stage": "loratrain_build_dataset",
        "created": datetime.now(timezone.utc).isoformat(),
        "sft_schema": {
            "format": "prompt_completion",
            "version": 2,
            "completion_only_loss": True,
            "note": (
                "defect-1 fix (2026-07-29): prompt/completion message columns; "
                "trl 0.29.1 masks prompt-token labels to -100 by default for "
                "this dataset type. v1 was a pre-templated 'messages' LM "
                "dataset trained with full-sequence loss."
            ),
        },
        "seed": seed,
        "corpus": {
            "path": str(corpus_path),
            "sha256": corpus_sha256,
            "rows": corpus_rows,
        },
        "eval_paper_split": {
            "path": str(split_path),
            "sha256": split_sha256,
            "n_eval_papers": n_eval_papers,
        },
        "train_uids": {
            "path": str(train_uids_path),
            "sha256": sha256_file(train_uids_path),
            "count": len(train_uids),
        },
        "eval_set": {
            "path": str(eval_set_path),
            "sha256": sha256_file(eval_set_path),
            "n_rows": n_eval_rows,
        },
        "wire_format": {
            "system_prompt": config.PASS_AT_K_SYSTEM_PROMPT,
            "no_think_suffix": config.PASS_AT_K_NO_THINK_SUFFIX,
            "note": "byte-identical to pass@k qwen_http (AGENTS.md invariant 2)",
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "rows": len(examples),
            "n_train_uids": len(train_uids),
            "per_uid_trace_counts": per_uid_trace_counts,
        },
        "verdict_totals": dict(verdict_totals),
        "duplicates_dropped": duplicates_dropped,
        "reconciliation": reconciliation,
        "backfill_7of8": backfill,
        "weight_policy": dict(weight_policy),
        "trainer_hyperparams": {
            "rank": config.LORA_RANK,
            "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT,
            "lr": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "micro_batch_size": config.MICRO_BATCH_SIZE,
            "max_seq_len": config.MAX_SEQ_LEN,
            "grad_accum_steps": config.GRAD_ACCUM_STEPS,
            "lr_scheduler_type": config.LR_SCHEDULER_TYPE,
            "warmup_ratio": config.WARMUP_RATIO,
            "weight_decay": config.WEIGHT_DECAY,
            "completion_only_loss": True,
            "note": (
                "grad_accum_steps/lr_scheduler_type/warmup_ratio/weight_decay "
                "were silent in v1 (hardcoded literal + inherited SFTConfig "
                "defaults, unrecorded in manifests); values unchanged, now "
                "pinned and echoed (2026-07-29)."
            ),
        },
        "guards": list(guards),
    }


# Ordered guard-step names echoed into the manifest as an audit trail --
# see build()'s docstring and the module docstring's "Build flow"
# paragraph for what each step does. The original 17 (W2) steps are
# unchanged, in the same relative order; the 4 new entries (GGUF 7/8
# backfill, Nicky's ruling 2026-07-26) are inserted at the points they
# actually run.
BUILD_GUARD_STEPS = (
    "assert_not_retired_path",
    "assert_corpus_pinned",
    "assert_split_pinned",
    "load_eval_papers",
    "load_backfill_uids",
    "assert_backfill_mapping_complete",
    "split_exists",
    "assert_train_eval_disjoint",
    "select_train_records",
    "load_backfill_records[n_correct=7,label=solved]",
    "assert_no_leakage[records]",
    "assert_no_cross_uid_statement_dups",
    "assert_no_cross_split_statement_dups",
    "load_rollout_files[last_occurrence]",
    "reconcile_records[routed_or_unique_alternative]",
    "harvest_correct_traces",
    "dedupe_examples",
    "apply_weight_policy",
    "assert_verified_correct[examples]",
    "assert_prompt_completion_wellformed",
    "assert_weight_policy_honored",
    "assert_verbatim_targets",
    "assert_no_leakage[examples]",
    "verify_written_dataset[schema+policy]",
)


def build(
    *,
    corpus_path: Path,
    split_path: Path,
    train_uids_path: Path,
    eval_set_path: Path,
    output_dir: Path,
    expected_corpus_sha256: str,
    expected_corpus_rows: int,
    expected_split_sha256: str,
    backfill_trace_sources: dict,
    seed: int,
    weight_policy: str,
    weight_policy_cap_k: int,
    repo_root=None,
) -> dict:
    """Build ``sft_train.jsonl`` + ``dataset_manifest.json`` -- the W2 orchestrator.

    Every argument is a keyword; none of the pins/paths default here on
    purpose -- ``main()`` supplies the config pins explicitly and tests
    supply fixture values explicitly, so there is no code path where a
    stale or accidental default pin can slip in unnoticed. ``repo_root``
    defaults to ``config.REPO_ROOT`` (the real icepick repo root every
    corpus row's ``corpus_provenance.source_file`` resolves against);
    tests pass their own ``tmp_path`` fixture root instead.
    ``backfill_trace_sources`` is ``config.BACKFILL_TRACE_SOURCES`` (the
    GGUF 7/8 backfill roster, Nicky's ruling 2026-07-26) -- pass ``{}``
    for a split with no backfill uids. ``weight_policy`` /
    ``weight_policy_cap_k`` select the defect-2 gradient-weight policy
    (``apply_weight_policy``; config knobs WEIGHT_POLICY /
    WEIGHT_POLICY_CAP_K, CLI-overridable) -- required keywords like the
    pins, so no code path silently builds under an accidental policy.

    Guard order (see ``BUILD_GUARD_STEPS``, echoed into the written
    manifest as an audit trail, and the module docstring's "Build flow"
    paragraph) -- NOTHING is written to disk until every guard below the
    "write" step has passed. The original 17 W2 steps are unchanged; the
    4 new backfill steps (marked NEW) are inserted at the points they
    actually run:

      1. reject any input path under a retired split directory
      2. corpus sha256 + row-count pin
      3. NEW: split full sha256 pin (authoritative, 2026-07-26)
      4. eval-paper split sha16 pin (redundant defense-in-depth, unchanged)
      5. NEW: load the split's train_backfill_7of8_uids
      6. NEW: assert it is EXACTLY backfill_trace_sources' key set
      7. the derived split (train_uids.txt / eval_set.jsonl) must exist
      8. load train uids / eval uids / eval statements
      9. train/eval uid-level disjointness
      10. select this build's train records from the pinned corpus
          (backfill uids exempted from the in-corpus requirement)
      11. NEW: load + validate the pinned backfill records
          (n_correct==7, label=="solved") and merge into the record set
      12. paper- and uid-level leakage (flat shape) -- covers backfill too
      13. cross-uid and cross-split full-statement duplicate checks --
          covers backfill too
      14. resolve every record's routed rollouts file, discover the
          registry (REGISTRY_GLOBS), load each file as a
          last-occurrence index -- backfill records' synthesized
          corpus_provenance.source_file routes them identically
      15. reconcile every record to its authoritative file (routed,
          else unique alternative, else hard fail -- module docstring
          "Rollout reconciliation")
      16. harvest verified-correct traces from the authoritative
          indexes (per-record tally re-check, n_correct >= 1,
          non-empty output); backfill examples get
          provenance.backfill_7of8 = True
      17. dedupe on (uid, rollout_uid), then apply the weight policy
          (defect 2: cap1/capk seeded deterministic selection, or
          inverse per-row 1/n weights)
      18. re-verify every example: verified-correct, prompt/completion
          well-formed (defect 1), weight policy honored (defect 2),
          byte-identical targets, leakage re-check (nested shape,
          defense in depth)
      19. write the dataset to a .tmp sibling
      20. re-read the WRITTEN .tmp from disk and re-verify every row
          (schema and policy included); only then atomically publish it
          to its final name and write the manifest (with its backfill
          and weight-policy audit blocks) -- a verification failure
          leaves NO final artifacts (the .tmp stays behind for
          forensics)

    See README "Split & corpus", "Non-negotiable ordering & invariants",
    and D4 for the invariants each step enforces.
    """
    corpus_path = Path(corpus_path)
    split_path = Path(split_path)
    train_uids_path = Path(train_uids_path)
    eval_set_path = Path(eval_set_path)
    output_dir = Path(output_dir)
    repo_root = Path(repo_root) if repo_root is not None else config.REPO_ROOT

    # 1. Never consume a retired split, no matter what else is true.
    assert_not_retired_path(corpus_path, split_path, train_uids_path, eval_set_path)

    # 2. Corpus identity pin.
    assert_corpus_pinned(corpus_path, expected_corpus_sha256, expected_corpus_rows)

    # 3. Split identity pin -- authoritative full sha256 (2026-07-26).
    assert_split_pinned(split_path, expected_split_sha256)

    # 4. eval_papers, via the (now redundant, defense-in-depth) sha16 check.
    eval_papers = load_eval_papers(split_path, expected_split_sha256[:16])

    # 5-6. GGUF 7/8 backfill roster: load the split's declared uids and
    #      assert they are EXACTLY the pinned trace-source mapping's keys
    #      -- a desync here means the split and config.py's pinned source
    #      mapping have drifted apart (see assert_backfill_mapping_complete).
    split_backfill_uids = load_backfill_uids(split_path)
    assert_backfill_mapping_complete(split_backfill_uids, backfill_trace_sources)
    backfill_uids_set = set(backfill_trace_sources)

    # 7. The derived split must already exist -- this module never builds it.
    missing = [p for p in (train_uids_path, eval_set_path) if not p.exists()]
    if missing:
        raise SplitNotBuiltError(
            "the canonical split has not been built yet: "
            f"{[str(p) for p in missing]} not found. The split is a derived "
            "view produced by evalharness-build-set (README 'Split & corpus', "
            "recipe step 0) -- run it first, then retry. Never point this "
            "builder at an evalharness/data/retired_*/ directory; any split "
            "under one is retired."
        )

    # 8. Load the split's contents.
    train_uids = load_uid_list(train_uids_path)
    eval_uids = load_eval_uids(eval_set_path)
    eval_statements = load_eval_statements(eval_set_path)

    # 9. Train/eval must be disjoint at the uid level before the corpus is
    #    even read.
    assert_train_eval_disjoint(train_uids, eval_uids)

    # Scope the pinned backfill roster down to THIS build's train_uids.txt
    # -- symmetric with how a corpus-resident uid not listed in
    # train_uids.txt is simply not selected (no error): a pinned backfill
    # uid absent from this particular train_uids.txt is likewise just not
    # part of this build, never silently pulled in anyway. The full
    # backfill_uids_set (config-vs-split) stays as-is above; this narrower
    # set drives every remaining backfill step.
    train_uids_set_all = set(train_uids)
    active_backfill_uids = backfill_uids_set & train_uids_set_all
    active_backfill_trace_sources = {
        uid: backfill_trace_sources[uid] for uid in active_backfill_uids
    }

    # 10. Select this build's train records from the pinned corpus (the
    #     pinned backfill uids are the one exempted-from-corpus case).
    corpus_rows = load_corpus(corpus_path)
    records = select_train_records(corpus_rows, train_uids, backfill_uids=active_backfill_uids)

    # 11. Load + validate the pinned backfill records (n_correct==7,
    #     label=="solved") and merge them in -- from here on every guard
    #     below treats backfill and corpus-resident records identically.
    backfill_records = load_backfill_records(active_backfill_trace_sources, repo_root)
    records = records + backfill_records

    # 12. Paper- and uid-level leakage (flat shape -- a corpus row's own
    #    top-level "provenance" field, when present, is a status string
    #    like "extracted", never a dict, so _normalize_provenance falls
    #    through to the flat record correctly).
    assert_no_leakage(records, eval_papers, eval_uids)

    # 13. Full-statement duplicate checks, both within train and across the
    #     train/eval boundary.
    assert_no_cross_uid_statement_dups(records)
    assert_no_cross_split_statement_dups(records, eval_statements)

    # 14. Resolve every record's routed rollouts file (distinct
    #     source_files only -- several records commonly share one file),
    #     discover the registry, and load each file as a last-occurrence
    #     index (rollouts.jsonl files are append-across-passes logs --
    #     module docstring "Rollout reconciliation"). Backfill records'
    #     synthesized corpus_provenance.source_file (load_backfill_records)
    #     routes them through this exact same resolution, unmodified.
    routed_path_by_source = {}
    for record in records:
        source_file = record["corpus_provenance"]["source_file"]
        if source_file not in routed_path_by_source:
            routed_path_by_source[source_file] = rollouts_path_for(source_file, repo_root)
    registry_paths = discover_registry(routed_path_by_source.values(), repo_root)
    indexes_by_path = {}
    duplicate_entries_by_path = {}
    for path in registry_paths:
        indexes_by_path[path], duplicate_entries_by_path[path] = load_rollout_file(path)

    # 15. Reconcile every record to its authoritative rollouts file.
    authoritative = {}
    via_counts = {RECONCILED_VIA_ROUTED: 0, RECONCILED_VIA_UNIQUE_ALTERNATIVE: 0}
    for record in records:
        routed_path = routed_path_by_source[record["corpus_provenance"]["source_file"]]
        index, path, via = reconcile_record(record, indexes_by_path, routed_path)
        authoritative[record["uid"]] = (index, path, via)
        via_counts[via] += 1

    # 16. Harvest verified-correct traces from the authoritative indexes.
    #     active_backfill_uids stamps provenance.backfill_7of8 on the
    #     examples it covers.
    examples = harvest_correct_traces(
        records, authoritative, expected_corpus_sha256, repo_root,
        backfill_uids=active_backfill_uids,
    )

    # 17. Dedupe on (uid, rollout_uid).
    n_before_dedupe = len(examples)
    examples = dedupe_examples(examples)
    duplicates_dropped = n_before_dedupe - len(examples)

    # 17b. Apply the defect-2 gradient-weight policy (cap1/capk: seeded
    #      deterministic per-uid trace selection; inverse: per-row 1/n
    #      weights). Runs AFTER dedupe so caps count unique traces.
    examples, weight_policy_block = apply_weight_policy(
        examples, policy=weight_policy, cap_k=weight_policy_cap_k, seed=seed
    )

    # 18. Re-verify every example; assert prompt/completion well-formedness
    #     (defect-1 invariant) and the declared weight policy (defect-2
    #     invariant); assert byte-identical targets against a verification
    #     map drawn from each record's AUTHORITATIVE index; re-check
    #     leakage on the nested provenance shape (defense in depth).
    verification_index = {}
    for record in records:
        uid = record["uid"]
        index, _path, _via = authoritative[uid]
        for rollout_uid in record.get("rollout_uids") or []:
            verification_index[(uid, rollout_uid)] = index[(uid, rollout_uid)]
    for example in examples:
        assert_verified_correct(example)
    assert_prompt_completion_wellformed(examples)
    assert_weight_policy_honored(examples, policy=weight_policy, cap_k=weight_policy_cap_k)
    assert_verbatim_targets(examples, verification_index)
    assert_no_leakage(examples, eval_papers, eval_uids)

    # 19. Only now write anything to disk -- and atomically: the dataset
    #     goes to a .tmp sibling first, is verified FROM DISK there, and
    #     only then is published to its final name (Path.replace, atomic
    #     on POSIX). The manifest is written after the verified publish.
    #     On any verification failure the final filenames never exist;
    #     the .tmp is left behind for forensics (cross-review suggestion,
    #     2026-07-25).
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "sft_train.jsonl"
    dataset_tmp_path = output_dir / "sft_train.jsonl.tmp"
    manifest_path = output_dir / "dataset_manifest.json"

    write_dataset(examples, dataset_tmp_path)

    # 20. Re-read the WRITTEN bytes and re-verify every row from disk,
    #     BEFORE the file exists under its final name.
    n = verify_written_dataset(
        dataset_tmp_path, verification_index,
        policy=weight_policy, cap_k=weight_policy_cap_k,
    )
    if n != len(examples):
        raise TraceIntegrityError(
            f"post-write verification read {n} row(s) from {dataset_tmp_path}, "
            f"expected {len(examples)} -- write/verify count mismatch; the "
            "dataset was NOT published (tmp file left for forensics)."
        )
    dataset_tmp_path.replace(dataset_path)

    with eval_set_path.open("r", encoding="utf-8") as fh:
        n_eval_rows = sum(1 for line in fh if line.strip())

    def _rel(path):
        try:
            return Path(path).relative_to(repo_root).as_posix()
        except ValueError:
            return str(path)

    reconciliation = {
        "routed": via_counts[RECONCILED_VIA_ROUTED],
        "unique_alternative": via_counts[RECONCILED_VIA_UNIQUE_ALTERNATIVE],
        "per_uid": {
            uid: {"trace_file": _rel(path), "reconciled_via": via}
            for uid, (_index, path, via) in sorted(authoritative.items())
        },
        "registry": [
            {
                "path": _rel(path),
                "sha256": sha256_file(path),
                "duplicate_entries": duplicate_entries_by_path[path],
            }
            for path in registry_paths
        ],
    }

    backfill_per_uid_trace_counts = {uid: 0 for uid in active_backfill_uids}
    for example in examples:
        uid = example["provenance"]["uid"]
        if uid in backfill_per_uid_trace_counts:
            backfill_per_uid_trace_counts[uid] += 1
    backfill = {
        "uids": sorted(active_backfill_uids),
        "sources": [
            {"path": source_file, "sha256": sha256_file(repo_root / source_file)}
            for source_file in sorted(set(active_backfill_trace_sources.values()))
        ],
        "per_uid_trace_counts": backfill_per_uid_trace_counts,
    }

    verdict_totals = _tally_verdicts(records)
    manifest = build_manifest(
        seed=seed,
        corpus_path=corpus_path,
        corpus_sha256=expected_corpus_sha256,
        corpus_rows=expected_corpus_rows,
        split_path=split_path,
        split_sha256=expected_split_sha256,
        n_eval_papers=len(eval_papers),
        train_uids_path=train_uids_path,
        train_uids=train_uids,
        eval_set_path=eval_set_path,
        n_eval_rows=n_eval_rows,
        dataset_path=dataset_path,
        dataset_sha256=sha256_file(dataset_path),
        examples=examples,
        verdict_totals=verdict_totals,
        duplicates_dropped=duplicates_dropped,
        reconciliation=reconciliation,
        backfill=backfill,
        weight_policy=weight_policy_block,
        guards=list(BUILD_GUARD_STEPS),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loratrain-build-dataset",
        description=(
            "Harvest verified-correct rollout traces for this build's train "
            "uids into data/sft_train.jsonl (README D4: verbatim SFT "
            "targets, RFT-style)."
        ),
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=config.CORPUS_PATH,
        help=f"Path to band_corpus.jsonl (default: config.CORPUS_PATH = {config.CORPUS_PATH}).",
    )
    p.add_argument(
        "--split",
        type=Path,
        default=config.EVAL_PAPER_SPLIT_PATH,
        help=(
            "Path to the split file (evalharness/data/corpus_split_200_100.json "
            "as of Nicky's 2026-07-26 ruling; carries eval_papers, train_uids, "
            "holdout_uids, train_backfill_7of8_uids) (default: "
            f"config.EVAL_PAPER_SPLIT_PATH = {config.EVAL_PAPER_SPLIT_PATH})."
        ),
    )
    p.add_argument(
        "--train-uids",
        type=Path,
        default=config.TRAIN_UIDS_PATH,
        help=(
            "Path to train_uids.txt, the evalharness-build-set output "
            f"(default: config.TRAIN_UIDS_PATH = {config.TRAIN_UIDS_PATH})."
        ),
    )
    p.add_argument(
        "--eval-set",
        type=Path,
        default=config.EVAL_SET_PATH,
        help=(
            "Path to eval_set.jsonl, the evalharness-build-set output "
            f"(default: config.EVAL_SET_PATH = {config.EVAL_SET_PATH})."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where sft_train.jsonl and dataset_manifest.json are written "
            "(default: config.DATA_V2_DIR/<policy-label>, e.g. "
            f"{config.DATA_V2_DIR / config.weight_policy_label()}; v1 "
            "artifacts under data/ and data/run1_final/ are never the "
            "default target and must not be overwritten)."
        ),
    )
    p.add_argument(
        "--weight-policy",
        choices=list(config.VALID_WEIGHT_POLICIES),
        default=config.WEIGHT_POLICY,
        help=(
            "Defect-2 gradient-weight policy: cap1 (one trace per uid, "
            "default), capk (at most --cap-k traces per uid), or inverse "
            "(all traces, per-row weight = 1/n_traces). Which policy SHIPS "
            f"is Nicky's decision (default: config.WEIGHT_POLICY = "
            f"{config.WEIGHT_POLICY!r})."
        ),
    )
    p.add_argument(
        "--cap-k",
        type=int,
        default=config.WEIGHT_POLICY_CAP_K,
        help=(
            "Rows-per-uid cap for --weight-policy capk (default: "
            f"config.WEIGHT_POLICY_CAP_K = {config.WEIGHT_POLICY_CAP_K})."
        ),
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Root against which corpus_provenance.source_file paths resolve; "
            "default: the icepick repo root (config.REPO_ROOT). Not a guard "
            "bypass -- purely a path base, so tests can point it at a "
            "synthetic tmp_path tree."
        ),
    )
    return p


def main(argv=None) -> int:
    """CLI entrypoint: validate config, run the full guarded build, print a summary.

    No flag here can override a sha pin or skip/weaken a guard (no
    --force, no --allow-*) -- the only knobs are input/output PATHS and
    the path base rollouts resolve against. The sha pins and seed always
    come from config.py. Guard failures (``SplitNotBuiltError``,
    ``PinMismatchError``, ``LeakageError``, ``TraceIntegrityError``, ...)
    are never caught here -- they propagate to the caller so a refusal is
    always loud, never a silent nonzero exit swallowed by a try/except.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config.validate_config()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = config.DATA_V2_DIR / config.weight_policy_label(
            args.weight_policy, args.cap_k
        )

    manifest = build(
        corpus_path=args.corpus,
        split_path=args.split,
        train_uids_path=args.train_uids,
        eval_set_path=args.eval_set,
        output_dir=output_dir,
        expected_corpus_sha256=config.EXPECTED_CORPUS_SHA256,
        expected_corpus_rows=config.EXPECTED_CORPUS_ROWS,
        expected_split_sha256=config.EXPECTED_SPLIT_SHA256,
        backfill_trace_sources=config.BACKFILL_TRACE_SOURCES,
        seed=config.SEED,
        weight_policy=args.weight_policy,
        weight_policy_cap_k=args.cap_k,
        repo_root=args.repo_root,
    )

    summary = {
        "stage": manifest["stage"],
        "dataset_path": manifest["dataset"]["path"],
        "rows": manifest["dataset"]["rows"],
        "sha256": manifest["dataset"]["sha256"],
        "weight_policy": {
            "policy": manifest["weight_policy"]["policy"],
            "label": manifest["weight_policy"]["label"],
            "rows_before": manifest["weight_policy"]["rows_before"],
            "rows_after": manifest["weight_policy"]["rows_after"],
        },
        "reconciliation": {
            "routed": manifest["reconciliation"]["routed"],
            "unique_alternative": manifest["reconciliation"]["unique_alternative"],
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

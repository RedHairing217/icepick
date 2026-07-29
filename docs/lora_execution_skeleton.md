# LoRA Eval Execution — window skeleton (corpus as single source of truth)

> **CLOSED 2026-07-29 — executed in modified form; retained for the design arguments.**
> Nicky's 2026-07-26 ruling reinstated a stored 200/100 split
> (`evalharness/data/corpus_split_200_100.json`, sha `768436f4`, 7-record GGUF-7/8
> training-only backfill declared in-file) — superseding this skeleton's
> "derived view only, never a stored artifact" position — while the C1 corpus-sha-pin
> principle below WAS adopted on the consumer side (loratrain `config.py` pins corpus
> `e0975e11` + the full split sha; `build_dataset` runs a 21-step guard chain over
> both). The campaign then ran to n=12 on that split; final verdict:
> `docs/lora_consistency_verdict.md`.

Prepared 2026-07-16 by the reintegration session (Fable-5) on Nicky's ruling: **"undo split /
only skeleton for LoRA, no split yet / single source of truth corpus."** Mission slug:
**lora-eval-execution**. No split exists and none should be created until this window runs —
the split is a DERIVED VIEW, computed at run time, never a stored artifact.

## Why there is no split file

The 2026-07-15 `corpus_split_200_100.json` (200 train / 100 holdout, frozen band split) is
**retired** to `evalharness/data/retired_20260716/`. It failed as an artifact class, not by
accident:

- It **duplicated corpus state**, so it rotted the moment the corpus moved. The 07-16 repair fold
  (309→293) left **26 of its uids dangling**; patching it took three passes in one evening
  (alias-recover → band-pure → paper-list recompute) and each pass raised a new question.
- It **conflicted with the live split**: 21 of its train papers are declared eval-only by
  `eval_paper_split.json`. Moot in practice — see below — but it is exactly the contradiction two
  sources of truth guarantee.
- It **had no consumer**. `grep -rn corpus_split_200_100 --include=*.py` → nothing. It was
  referenced only in the ledger. Nothing was ever contaminated by it; it was dead on arrival.

`holdout_uids.txt` / `train_uids.txt` were retired alongside it: stale copies of a
`build_eval_set.py` **output**, mistaken for inputs.

## What is already correct (do not rebuild)

`evalharness/src/evalharness/build_eval_set.py` **already implements the derived-view pattern**:
it loads the sha-pinned `eval_paper_split.json` (the frozen paper-level eval holdout, 108 papers;
`EXPECTED_SPLIT_SHA256_16 = 110a4bf27320f2b1`, verified intact 07-16), then DERIVES the train set
from the corpus as "band records whose arxiv_id is not an eval paper", writing a fresh
`train_uids.txt` to its output dir every run, with `assert_no_leakage` enforcing the rule. Extend
this; do not replace it.

## The one real gap (this window's core)

**`build_eval_set.py` pins the SPLIT file's sha but NOT the corpus's.** Verified: no corpus sha
appears anywhere in `evalharness/src` or its README. So a corpus fold silently changes the derived
train/eval sets with no tripwire — the exact failure the retired split suffered, one level up.
The corpus has moved four times in six days (`01609862` → `1b9d5d62` → `13164e3f` → `e0975e11`).

**C1 — pin the corpus.** `build_eval_set` takes `--corpus` + `--expected-corpus-sha16` (current:
`e0975e112f05d03e`, band_corpus.jsonl, 293 rows) and hard-fails on mismatch exactly as it already
does for the split file. Every derived artifact records the corpus sha it was derived from.

**C2 — derived split, recorded by parameters not by uid lists.** A split is
`f(corpus_sha, seed, rule)`. Emit a `split_manifest.json` capturing those three plus resulting
counts — reproducible on demand, never a second source of truth. If the corpus moves, you
re-derive and the manifest tells you what changed; you never patch a uid list again.

**C3 — band-purity + selection rules, as code with asserts** (these were prose in the retired
file and drifted): holdout is band-labelled records **in the current corpus** (a record that
exited band on re-score is not eligible — see F3); double-scored/disagreeing-label records
excluded from holdout (the retired split's `ambiguous_excluded_from_holdout: 5`); train and
holdout **paper-disjoint**; anchors drawn from papers in neither. Assert each; do not trust prose.

**C4 — reconcile or retire `eval_paper_split.json`'s authority.** It is currently the one frozen
artifact left. Either (a) keep it frozen and pin it (status quo, works — it is the eval protocol's
definition and freezing an eval holdout is legitimate), or (b) derive it too from
`(corpus_sha, seed)`. **Recommend (a)**: an eval holdout SHOULD be frozen — that is the point of
a holdout. But record its corpus sha of origin so drift is visible. Do not let a second *train*
split exist alongside it, ever.

## Findings that bind this window (measured, 07-16 and prior)

- **F1 — repaired statements grade harder.** All 9 band-exits from the repair fold scored 0/8;
  self-contained statements remove recall shortcuts. Expect band shrinkage after any repair fold,
  and expect the derived holdout to shrink with it. Do not "fix" this by backfilling.
- **F2 — fabricated sharpness is a real extraction class** (5 records): the QA generator recasts
  proved bounds as "what is the sharp constant?". The wellposedness cascade cannot catch it
  without the source. Records in the eval set are not immune.
- **F3 — backend label instability enriches for defects**: defective statements grade noisily, so
  a record that enters band via re-score deserves a source check before it is eval-eligible.
- **F4 — the corpus is clean of duplicates** (verified 07-16: 0 uid dups, 0 full-statement dups in
  band + wellposed_all). Dup-detect on FULL statements — a truncated prefix key manufactures
  ghosts (two false positives investigated and cleared).
- **F5 — anchors are not band members by design** (solved/fail). Any check that tests anchors for
  band membership will falsely report them destroyed. They live in `wellposed_all`.

## Preflight / stop conditions

Pin `band_corpus.jsonl` sha at start and re-check at close (parallel sessions fold aggressively;
the corpus moved under a session once on 07-16). Verify disk + `ps` over task notifications.
Qwen: max one concurrent call machine-wide. STOP if: the corpus sha moves mid-window · any derived
set fails a leakage assert · a split file is about to be written as a durable artifact (that is
the anti-goal) · `eval_paper_split.json` sha ≠ `110a4bf27320f2b1` without a recorded decision.

## Deliverables

Extended `build_eval_set.py` (corpus pin + split_manifest emission + band-purity asserts) with
tests; a derived eval set + train uids under an output dir, never `data/`; `split_manifest.json`
recording `(corpus_sha, seed, rule, counts)`; a SESSION_HANDOFF addendum; no new frozen split.

Token/pause budget: [NICKY: set]. Arm checkboxes:
[ ] C1 corpus sha pin  [ ] C2 derived split + manifest  [ ] C3 band-purity asserts
[ ] C4 keep eval_paper_split frozen (recommended) — or derive it too
[ ] delete `retired_20260716/` after a grace period (default: keep)

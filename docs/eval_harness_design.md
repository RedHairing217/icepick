# LoRA Eval Harness — Design (final implementation → `evalharness/` sub-repo)

Status: DESIGN + frozen split artifact (2026-07-14). Owner: Nicky. Purpose: prove a
**measurable** performance improvement from LoRA-training qwen3-8b on the pde625 band
corpus — with a number that survives scrutiny.

Implemented: `evalharness/` scaffold (build_eval_set.py, run_eval.py, report.py, tests, README) landed 2026-07-15, uncommitted — see `evalharness/README.md`. 42/42 evalharness tests green; root suite unaffected (975 passed, 3 skipped, before and after).

**EXECUTED + CLOSED 2026-07-26 → 07-29.** Baseline captured (eval-band 43/100, greedy,
llama-server `b1-c0bc859`), then **12 control-config seeds** trained and evaluated under
this protocol. **Final verdict: `docs/lora_consistency_verdict.md`** (the campaign
deliverable) — at N=200 the dataset produces a **small, directionally consistent
positive effect** (9/12 seeds ≥0, mean +1.67pp, net +20pp, magnitudes +24 vs −4,
worst run −2) that does **not** reach conventional significance at n=12 (one-sided
sign p=.17, t p=.061). Direction supported; significance not established; practical
effect minimal. An n=8 read had shown significance; its own pre-registered extension
corrected that — the protocol's second self-correction. Supporting docs: `docs/lora_campaign_results.md`
(results + adjudicated config corrections), `docs/lora_params_rationale.md` (why these
hyperparameters), `docs/lora_decisions_2026-07-28.md` (decisions D1–D8).
Two dated corrections below (split authority; serving engine) supersede the original
text where marked. **This harness performed as designed: it was capable of returning a
null, and did.**

## Why this exists (the two failure modes it forecloses)

1. **Cross-quant/hardware confound.** Measured on the 25-record calibration set:
   MLX-4bit(M4) vs GGUF-Q4_K_M(remote) differ by mean |Δ|=1.32/8 per problem — the
   size of a plausible LoRA gain. Same-weights GGUF local-vs-remote differ by 0.75
   ≈ sampling noise (0.70) → **same quant on both boxes is mandatory; baseline and
   post-train eval must run on the same quant** (box then doesn't matter).
2. **Regression-to-the-mean from selection.** The training-set top-up re-rolls
   near-band 7/8 records; records selected on a lucky/unlucky draw will drift back
   on re-measurement, manufacturing fake "improvement." → **Selection values are
   never eval baselines.** The eval set is frozen before final training selection,
   split at the PAPER level, and scored fresh.

## Frozen artifacts (already produced)

- **[CORRECTED 2026-07-26 — Nicky's ruling]** the split below was RETIRED to
  `evalharness/data/retired_20260726/` and replaced by
  `evalharness/data/corpus_split_200_100.json` (sha256[:16] `768436f4`, seed 20260726):
  **100 pure-band holdout / 200 train (193 band + 7 GGUF-7/8 backfill,
  training-only)**, paper-disjoint incl. anchors, pinned against corpus `e0975e11`
  (293 rows). Historical text (pre-ruling): `eval_paper_split.json` — sha256[:16]
  `110a4bf27320f2b1`, seed 20260714. 723 universe papers → 108 eval papers (15%) /
  157 eval records.
- **The rule:** any record whose `arxiv_id` is in `eval_papers` is EXCLUDED from
  training, regardless of its band outcome. Training set = final cascade band MINUS
  eval-paper records. (Paper-level, because lemmas within a paper leak.)
- Known consequence: the cascade's band total funds BOTH sets. ~30–35 of the
  cascade's band records will fall on eval papers → training band ≈ cascade band
  − ~35. Plan the cascade/top-up target accordingly (i.e. "300 training band"
  needs ≈335 cascade band).

## Eval set composition (built from the 157 eval records once cascade completes)

| slice | source | role |
|---|---|---|
| **eval-band** (primary, expect ~30–40) | eval-paper records that scored band in the remote re-score | the improvement metric lives here |
| **anchor-solved** (~10) | eval-paper records at 8/8 | must STAY solved → catastrophic-forgetting detector |
| **anchor-fail** (~10) | eval-paper collapse records at 0/8 | must STAY failed → memorization/contamination detector |

Anchor drift in either direction is a red flag reported alongside the headline.

## Measurement protocol

- **Primary metric: greedy pass@1** (`--k 1 --temperature 0`, think off, max_tokens
  2048) on eval-band. Deterministic → zero sampling noise → maximum sensitivity.
  Baseline = base qwen3-8b (GGUF-Q4_K_M). Post = same + LoRA. Same box, same quant,
  same settings, same day if possible (server-side changes are a confound).
- **Secondary (distributional): pass@k k=8 temp 0.7** on eval-band, 3 repeats,
  compare n_correct distributions — catches gains greedy misses (probability mass
  shifts below the argmax). Report separately; never blend into the headline.
- **Significance: exact McNemar on paired greedy outcomes** (solved↔unsolved per
  record). With ~35 eval-band records, detecting a true +15pp needs discordant-pair
  counts reported honestly — the harness prints b (base-only correct), c (LoRA-only
  correct), and the exact binomial p. Under ~25 records, say "underpowered" out loud.
- Scoring: the existing sympy verifier via the pass@k runner — identical extraction
  and equivalence rules for base and LoRA. No manual regrades.

## Sub-repo layout (`evalharness/`, mirrors the poser sub-repo pattern)

```
evalharness/
  README.md            # quickstart: build-set → baseline → post → report
  data/
    eval_paper_split.json     # FROZEN (committed; never regenerate)
    eval_set.jsonl            # built once cascade completes (build_eval_set)
  src/evalharness/
    build_eval_set.py  # cascade outputs + split -> eval-band + anchors (asserts:
                       #   zero train-paper leakage, records have statement+answer)
    run_eval.py        # drives icepick pass_at_k (greedy k1 + optional k8x3);
                       #   backend/model/URL from a config block, key via file path
    report.py          # paired diff, McNemar exact, anchor drift, markdown report
  tests/               # split-immutability (sha pin), leakage guard, McNemar cases,
                       #   report golden-file
```

Implementation notes for the coder:
- Reuse `icepick processing pass_at_k` as the execution engine (subprocess or import);
  do NOT reimplement rollout/scoring. `--k 1 --temperature 0` is supported today;
  the bearer-auth `--qwen-key-file` path (commit `21092f4`) covers remote gateways.
- `eval_paper_split.json` is immutable: `build_eval_set.py` must verify its sha
  against the pinned value and hard-fail on mismatch.
- Never print key material; key files are path-proxies (`tangerine_api.env` format:
  raw token, no `KEY=`).
- **[CORRECTED 2026-07-25, loratrain D3]** LM Studio (0.4.15) CANNOT load a LoRA
  adapter — the original serving assumption below is refuted. Serving is **llama.cpp
  `llama-server --lora` at pinned build `b10107`** over the bit-identical Q4_K_M base;
  **engine parity is a first-class invariant**: baseline and post use the same build +
  serve flags (fingerprint `b1-c0bc859` logged on every response). llama-server fills
  omitted sampler params (top_k 40 / top_p 0.95 / min_p 0.05) — harmless at greedy
  temp-0 (primary), NOT comparable to other engines' k=8 draws (secondary). Historical
  text: the tuned model is exposed as a distinct model id on an OpenAI-compatible
  endpoint (LM Studio loads GGUF+LoRA or a merged export). The harness takes
  `--model-base` / `--model-tuned` ids and refuses to run both against different
  endpoints unless `--allow-cross-endpoint` (prints the quant warning).

## Protocol checklist (the README's contract)

1. Cascade completes → `build_eval_set.py` → `eval_set.jsonl` (+ counts printed).
2. Training set = final band MINUS eval papers (builder emits `train_uids.txt`;
   the LoRA pipeline consumes ONLY that).
3. **Before training:** `run_eval.py --model-base ...` → `baseline_greedy.jsonl`.
4. Train LoRA (outside this harness).
5. `run_eval.py --model-tuned ...` → `post_greedy.jsonl` (same box/quant/settings).
6. `report.py` → headline (Δ solved on eval-band, McNemar p, CI), anchor drift,
   secondary k8 distribution shift.
7. Any 7/8-rerun top-ups, band relabels, or selection games touch ONLY the training
   side — the eval set and its baselines are never re-selected or re-rolled.

## Open items

- Anchor counts depend on how many eval-paper records land 8/8 / stay collapse in
  the finished cascade — build step reports actuals.
- If eval-band < 25 after the cascade, widen eval-band to include eval-paper 7/8
  records (still paper-clean) and re-state power.
- Multi-seed LoRA runs (train 2–3 seeds, report spread) — recommended if compute
  allows; the harness supports repeated post-eval files.

# LoRA parameters & search design — rationale (written 2026-07-28)

Why the initial hyperparameters were what they were, and why the stage-A search was
shaped the way it was. Companion to `docs/lora_campaign_results.md` and
`src/loratrain/README.md` (D1–D4).

## 1. Initial parameters (run 1) — chosen deliberately, not tuned

| setting | value | rationale |
|---|---|---|
| rank / alpha | 16 / 32 | The community-standard small-SFT operating point; α=2r convention. Enough capacity for a few-hundred-example set without obvious overfit risk. |
| dropout | 0.05 | Standard light regularization; not a knob anyone expected to be load-bearing. |
| learning rate | 1e-4 | The canonical adapter-training LR for this model scale; the single most pre-tuned default in the LoRA literature. |
| epochs | 3 | Textbook for a few-hundred-example SFT set — enough passes to fit, short of memorization territory. |
| precision | **bf16 LoRA, QLoRA explicitly rejected** | Design decision (RUNBOOK §2): QLoRA's nf4 base would introduce a second quantization regime vs the Q4_K_M serving quant — a train/serve mismatch in the same family as the measured 1.32/8 MLX-vs-GGUF confound. bf16 keeps training on the true weights the GGUF was quantized from. |
| target modules | all-linear (`q/k/v/o + gate/up/down_proj`) | Default full coverage; maximizes what a given rank can express and avoids an unforced module-selection decision with no data to justify it. |
| max seq len | 4096 | Covers the harvested trace lengths with margin; well under both ctx pins. |
| micro-batch | 4 (× **hardcoded grad-accum 4 = effective batch 16**) | Fits bf16 8B + optimizer on the A40 with headroom. [CORRECTED 2026-07-28: the original text claimed "no gradient-accumulation complexity" — wrong; `train_qwen3_lora.py:131` hardcodes `gradient_accumulation_steps=4`, unrecorded in the run manifests. Flagged independently by both reviews.] |
| seeds | fixed, enumerated (20260722/3/4) | Reproducibility + the multi-seed protocol was a first-class design element from the start. |
| **inherited TRL defaults** (added 2026-07-28) | linear decay→0, no warmup, no weight decay, full-sequence loss (no completion masking) | **Not chosen — silently inherited from `SFTConfig` and unrecorded in manifests until the 07-28 reviews surfaced them.** Two are material: full-sequence loss puts ~21.6% of trained characters on system/user text, and trace-per-record harvesting makes gradient weight equal `n_correct` (anti-difficulty: hardest band records get 1/7 the weight of near-ceiling backfill). Both structural — they inherit unchanged at any N. |

**The honest framing:** these were picked as *robust community defaults*, documented in
`config.py` at scaffold time (07-22), with zero dataset-specific tuning. That was
deliberate: the campaign goal was proof-of-concept of a **dataset effect**, not maximal
performance — and tuning before having any baseline effect measurement would have spent
budget and (worse) risked holdout-discipline violations before the harness had proven
itself. Defaults-first, measure, then tune only what the evidence says might matter.

## 2. Stage-A search design (post-run-1) — why this grid, this shape

Run 1's findings drove every choice: mean +5.0pp, seed spread 10pp (variance-dominated),
all seeds fitting the training set equally well (~0.43 loss).

- **One knob at a time vs control (C1–C4), one interaction probe (C5).** With single-seed
  screens, attribution beats coverage: a full factorial would cost more than the
  information it returns at this noise level. C5 (lr↑ + epochs↑) was the one affordable
  look at an interaction — the "aggressive corner."
- **LR got two probes (2e-4, 5e-5) — the only two-sided axis.** LR is historically the
  highest-leverage LoRA knob, and run-1's high discordant churn (~30–37 pairs/seed) was
  compatible with *either* too-hot (scrambling) or too-cold (under-committing).
- **Epochs and rank probed UP only (6ep, r32/α64).** Reasoning at the time: train loss
  0.43 left room to fit more, so "undertrained/undercapacity" was the hypothesis worth
  one run each. **Acknowledged limitation:** the down-directions (2 epochs, r8) were not
  probed, and at N≈160 examples the overfit-risk argument says down was at least as
  interesting. If a stage A′ ever runs, it should look down, plus attention-only targets.
- **α/r held at 2.0 in the rank probe** so C4 tests *capacity in isolation* — doubling
  rank without retuning alpha would have changed the effective adapter LR simultaneously.
- **Same seed (20260728) across all six configs.** Run 1's core lesson was that seed
  variance dominates config-scale effects; sharing the seed removes seed luck from
  between-config comparisons entirely. The cost — results are conditional on one seed —
  was accepted because stage A is a screen, not a verdict.
- **Validation carved from the TRAIN side (40 records / 34 papers), holdout untouched.**
  Selection pressure may never touch the exam: any config comparison happens on val only;
  the 100-record holdout is reserved for replication reads of the final config. This is
  the same selection-vs-measurement separation the whole harness is built on.
- **1 seed × 40 records per config, ties resolve to control — pre-declared.** The screen
  can only catch gross effects (±3 is noise); the decision rule was written down before
  scores existed so the search couldn't quietly become cherry-picking.
- **Stage B (top-2 selection × 3 seeds) was DROPPED at goal reorientation.** Once the
  goal became "demonstrate consistent improvement" rather than "maximize improvement,"
  selecting configs by score became a non-goal — and selection is exactly what burns
  holdout validity. Its budget was redirected to stage R: 5 replication seeds of the
  control config, growing the holdout delta distribution to n=8 for a sign/t-test
  consistency claim.

## 3. Deliberately untested (and why)

Dropout, target-module subsets, α/r ratio, warmup/schedule, batch size, packing,
lower rank, fewer epochs — and, surfaced by the 07-28 reviews: **loss masking
(prompt vs completion), per-record gradient weighting, and eval sample count
(greedy pass@1 vs avg@8)** — the three axes now known to matter more than any knob
the grid actually probed. Budget triage under a ≤$7 envelope, plus the working
hypothesis — supported by run 1 and *directionally consistent with* stage A's screen
(control the only non-negative; every deviation −3 to −5 — though at ±9–10pp SE the
40-record single-seed screen lacks the power to CONFIRM flatness; softened 2026-07-28) — that **the loss surface w.r.t.
hyperparameters is flat-to-negative around the defaults because data volume, not
configuration, is the binding constraint.** Every config fits 160 examples; none can
extract information the examples don't contain.

**[SUPERSEDED 2026-07-29 — n=12 outcome.]** The "next lever is N≈1000" line originally
here is retired. The 12-seed consistency run closed at a small, directionally
consistent **+1.67pp** (9/12 seeds ≥0; magnitudes +24 vs −4) that does not reach
conventional significance (one-sided sign p=.17, t p=.061) — direction supported,
significance not established — on a recipe carrying two unfixed dataset defects
(full-sequence loss; gradient weight = `n_correct`). Per decision **D8**
(`docs/lora_decisions_2026-07-28.md`), scaling N addresses *magnitude*, not
*consistency*, and at N≈1000 would land on the same defective recipe — both defects are
structural and inherit unchanged. **The indicated next experiment is dataset build v2**
(prompt/completion masking + per-record weight cap/1-per-uid + explicit backfill-weight
ruling), with N≈1000 sequenced after it. Final campaign status:
`docs/lora_consistency_verdict.md`.

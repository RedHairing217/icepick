# LoRA dataset-v2 arm — verdict (2026-07-31)

**Result: fixing both dataset defects did not improve the holdout.** v2 is
statistically indistinguishable from v1, and both are indistinguishable from zero.

Arm under test: v2 = **completion-only loss masking** (defect 1) + **`cap1` weighting**,
one trace per record, 700 rows → 200 (defect 2). Spec: `docs/lora_v2_work_order.md`
(`b58eb8c`). Everything else held frozen: same corpus `e0975e11`, same split
`768436f4`, same 100-record holdout, same control hyperparameters, same 12 seeds,
same engine `b1-c0bc859` at **`-fa off`**, same frozen baseline **43/100**.

## Per-seed results (greedy pass@1, eval-band, n=12)

| seed | v2 solved | Δ pp | b/c | McNemar p | anchors kept / fail-solved | v1 Δ | paired (v2−v1) |
|---|---|---|---|---|---|---|---|
| 20260722 | 42/100 | −1 | 17/16 | 1.000 | 10 / 0 | +11 | −12 |
| 20260723 | 46/100 | +3 | 14/17 | 0.720 | 10 / 0 | +1 | +2 |
| 20260724 | 50/100 | **+7** | 15/22 | 0.324 | 9 / 0 | +3 | +4 |
| 20260725 | 46/100 | +3 | 16/19 | 0.736 | 10 / 0 | 0 | +3 |
| 20260726 | 47/100 | +4 | 18/22 | 0.636 | 9 / 0 | +3 | +1 |
| 20260727 | 42/100 | −1 | 17/16 | 1.000 | 10 / 1 | +1 | −2 |
| 20260729 | 44/100 | +1 | 13/14 | 1.000 | 8 / 0 | +4 | −3 |
| 20260730 | 44/100 | +1 | 16/17 | 1.000 | 9 / 0 | +1 | 0 |
| 20260731 | 40/100 | −3 | 16/13 | 0.711 | 10 / 0 | 0 | −3 |
| 20260801 | 44/100 | +1 | 18/19 | 1.000 | 9 / 0 | −1 | +2 |
| 20260802 | 39/100 | −4 | 16/12 | 0.572 | 10 / 0 | −1 | −3 |
| 20260803 | 39/100 | −4 | 19/15 | 0.608 | 10 / 0 | −2 | −2 |

## Statistics

**p-value convention (declared, resolving the inconsistency in the prior record):**
the hypothesis is directional (improvement), so the **one-sided** figure is primary;
the two-sided figure is reported alongside. Every conclusion below holds under either
convention — nothing here depends on the choice.

**v2 vs baseline (n=12):** mean **+0.58pp**, sd 3.37. Sign test **7+ / 5− / 0 ties →
p₁ = .387** (p₂ = .774). t(11) = 0.600, p₁ = .280 (p₂ = .561). 95% CI **[−1.56, +2.72]**
— spans zero.

**v2 vs v1, paired at matched seeds (n=12):** mean **−1.08pp**. Sign **5+ / 6− / 1 tie
→ p₂ = 1.000**. t(11) = −0.882, p₂ = .396. 95% CI [−3.79, +1.62]. Excluding seed
20260722 (v1's outlier): mean **−0.09pp** — the two recipes are, to measurement
precision, the same.

**v1 reference:** mean +1.67pp, sign 7+ / 3− / 2, p₂ = .344.

Anchors held everywhere: 8–10/10 solved-anchors kept, at most 1/10 fail-anchor solved.

## What this means

The work order pre-registered the decision rule: *"If v2 at N = 200 moves the mean
materially — roughly +3pp or better — the recipe was the bottleneck and scaling N
becomes worth funding. If v2 stays near +1.67pp, the ceiling is elsewhere."*

v2 delivered **+0.58pp**, below v1's +1.67 and far below the +3 threshold, with a
paired difference of essentially zero. **The recipe was not the bottleneck.**

Both defects were real and both are genuinely fixed — the masking was proven by
token-level decode, and `cap1` verifiably cut 700 rows to 200. They simply were not
what was limiting the result. That is a clean negative, and a useful one: it retires
the leading explanation for the v1 null.

**The pre-registered alternative now leads: self-distillation saturation.** The
training targets are the base model's own verified-correct rollouts, and training loss
floors within the first few steps (0.71 → ~0.45 by epoch 1, final ~0.43). A model
being taught its own successful outputs has little left to extract, regardless of how
cleanly the loss is masked or how the records are weighted.

**Consequence for scaling: N≈1000 is not indicated by this evidence.** Both structural
defects scaled with the training set, which was the main argument for fixing them
before scaling. With them fixed and the effect still absent, more data buys more of
the same saturated signal. Funding a 3–4× data run should now require a different
rationale than "the recipe was misdirected."

## What may and may not be claimed

**May claim:** two structural dataset defects were identified, fixed, and their effect
measured at 12 paired seeds; the fix produced no detectable holdout improvement
(+0.58pp, CI spanning zero) and no improvement over the unfixed recipe (paired −0.09pp
excluding one outlier seed). The measurement instrument is stable and the anchors held.

**May NOT claim:** that v2 is *worse* than v1 — the paired difference is not
significant either. That the defects were harmless in general — this measures one
model, one corpus, N=200, greedy pass@1. That the campaign has excluded all recipe
hypotheses — only these two were tested.

## Method notes

Twelve adapters trained 2026-07-30 on the v2 `cap1` dataset (fp16 HF base,
`completion_only_loss: True`), evaluated 2026-07-30/31 at explicit `-fa off`.

An earlier evaluation pass of these same adapters ran with `-fa on` and is **void** for
comparison purposes: the baseline and all v1 seeds were measured on the
auto-resolved-off path, and mixing attention kernels across arms is a confound. A
tripwire confirmed 3/3 byte-identity between explicit `-fa off` and the established
path before this series ran. The void `-fa` readings survive in
`out/evalharness/v2cap1_s*` as flash-attention sensitivity data; one of them
(seed 20260727) was contaminated by a cross-session port collision and is quarantined
under `out/evalharness/QUARANTINE_20260730_contaminated/`.

**Interim reads were misleading, again.** At 5 seeds this arm read +2.80pp; at 7 seeds
+2.29pp; at 12 seeds +0.58pp. The five seeds that finished last were the same block
that sank v1. This is the third time in this campaign that a partial read pointed
somewhere the complete read did not — after run-1 seed 1 (+11) and the n=8 sign test
(p=.016). The pre-registered stopping point is the only number worth quoting.

Artifacts: `out/evalharness/v2cap1_faoff_s<seed>/` (12 dirs, 120 rows each, every eval
preceded by a passing server-identity check).

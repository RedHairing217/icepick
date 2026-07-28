# LoRA campaign — analysis & decisions, 2026-07-28

**Lane:** analysis/decision (read-only). **Author:** analysis lane. **Consumer:** execution lane ("Sharp Pick") + Nicky.
**Artifacts:** `out/analysis/lora_consistency_analysis.json`, `out/analysis/lora_sensitivity_analysis.json`.

**Ground truth re-verified on disk this window:**
corpus `out/corpus_pde625/band_corpus.jsonl` 293 rows sha `e0975e112f05d03e` ✓ ·
split `evalharness/data/corpus_split_200_100.json` sha `768436f4e55e2a46` ✓ ·
baseline eval_band **43/100**, anchors 9/10 solved, 0/10 fail ✓ ·
`rc_R.json` hyperparams byte-match `rc_C0.json` (r16/α32/drop.05/lr1e-4/3ep/mb4/4096) ⇒ stage-R is a true control replication ✓.

---

## 0. STATUS CORRECTION — the brief overstates what exists

The brief says stage-R is "trained + GGUF-converted; holdout evals in flight … these + run-1's 3 give **n=8**".
**On disk there are 4 completed seed evals, not 8.**

| dir | state |
|---|---|
| `out/evalharness/run1/post_greedy.jsonl` | complete (seed 20260722) |
| `out/evalharness/run1_s20260723/post_greedy.jsonl` | complete |
| `out/evalharness/run1_s20260724/post_greedy.jsonl` | complete |
| `out/evalharness/run1_sR20260725/post_greedy.jsonl` | complete |
| `out/evalharness/run1_sR20260726/` | **in flight** — only `tuned_greedy/_progress` + `pass_at_k_input.jsonl` |
| seeds 20260727 / 20260729 / 20260730 | **absent — no directory at all** |

Every number below is **n=4**. The consistency verdict is *not* deliverable yet. Nothing in this doc should be
read as the final verdict; §1 is an interim read whose main function is to size the design in §5–6.

---

## 1. FINDINGS

### F1 — The first stage-R seed came in at exactly zero. Mean drops to +3.75pp.
Receipt: `out/analysis/lora_consistency_analysis.json` → `per_seed`.

| seed | stage | tuned | Δ pp | b (base-only) | c (tuned-only) | discordant | McNemar p |
|---|---|---|---|---|---|---|---|
| 20260722 | run1 | 54/100 | **+11** | 9 | 20 | 29 | .061 |
| 20260723 | run1 | 44/100 | +1 | 18 | 19 | 37 | 1.00 |
| 20260724 | run1 | 46/100 | +3 | 16 | 19 | 35 | .736 |
| 20260725 | stageR | 43/100 | **0** | 19 | 19 | 38 | 1.00 |

Mean **+3.75pp**, SD 4.99, SE 2.50, t(3)=1.50 (p>.15), 95% CI **[−4.19, +11.69]**.
Sign test 3 positive / 0 negative / **1 exact tie** → p=.25. The brief's "3/3 positive, mean +5.0" is stale.

### F2 — The entire effect is carried by one seed. (highest-salience finding)
Receipt: `lora_sensitivity_analysis.json` → `loo`.

| dropped seed | mean of remaining | SD |
|---|---|---|
| — (all 4) | +3.75 | 4.99 |
| **20260722** | **+1.33** | **1.53** |
| 20260723 | +4.67 | 5.69 |
| 20260724 | +4.00 | 6.08 |
| 20260725 | +5.00 | 5.29 |

Remove the +11 seed and the campaign's effect is **+1.33pp**. Note what this does *for* the mission, though:
the remaining three seeds (+1, +3, 0) have SD 1.53 — far more *consistent* than the full set. Under a
"small but consistent" success criterion, `{+1, +3, 0}` is a more honest picture than `+3.75 ± 4.99`, and it is
also a much weaker claim. **The +11 is the outlier, not the signal.** Do not build the campaign narrative on it.

### F3 — Record-level outcomes are strongly reproducible across independently trained adapters.
Receipt: `lora_sensitivity_analysis.json` → `structure`.

Per-record solve count across the 4 adapters vs. the iid-noise null (Bernoulli p=.4675 per slot):

| solved by | observed | expected if pure noise |
|---|---|---|
| 0/4 | **30** | 8.0 |
| 1/4 | 12 | 28.2 |
| 2/4 | 23 | 37.2 |
| 3/4 | 11 | 21.8 |
| 4/4 | **24** | 4.8 |

χ²=**157.4** (df=4; .001 crit 18.47). Variance ratio 2.38 ⇒ ICC ≈ **0.46**.
**54/100 records are frozen** (unanimous across all four adapters); all seed-level variance lives in the other 46.
This is the campaign's best structural result: the eval instrument is not thrashing, and the ~35 discordant
pairs/seed are not evidence of a broken measurement. But note the flip side — because decoding is greedy
(deterministic), *all* of that 46-record churn is genuine adapter-to-adapter difference, not decode sampling.

### F4 — Cannot yet distinguish "adapters truly differ in quality" from "per-record flip noise".
Receipt: variance decomposition, this window.
Observed var(T) across seeds = 24.92 (SD 4.99pp). Predicted var if every record flips independently
with its own rate = 13.42 (SD 3.66pp). Ratio **1.86** — point estimate says there *is* a seed-level
correlated quality component. But the variance-ratio test gives χ²=5.57, df=3, **p≈0.13**: at k=4 this is
**not resolvable**. It resolves at k=8–12. This is the single question the remaining stage-R seeds answer,
and it determines whether more *decoding* or more *seeds* is the right investment (see D3).

### F5 — The brief's analysis #2 (per-uid weight × eval flip) is **impossible as specified**.
Receipt: `holdout_uids ∩ train_uids = 0` (computed from the split file; the split is paper-disjoint by design).
No holdout record has a training trace count, so the join has no keys. **This analysis cannot be run, ever,
on this split.** I substituted the valid transfer-side proxy: corpus `n_correct` is *simultaneously* the
difficulty proxy and the defective training-weight variable, so the gradient is testable on the holdout.

### F6 — No difficulty gradient in the direction defect #2 predicts. If anything, the opposite.
Receipt: `lora_sensitivity_analysis.json` → `gradient`, `collapsed`.

| corpus n_correct | records | base | tuned (mean of 4) | Δ pp | SE pp | % of headroom |
|---|---|---|---|---|---|---|
| 1 (hardest, weight 1) | 27 | .222 | .250 | +2.78 | 8.33 | +3.6 |
| 2 | 20 | .250 | .325 | +7.50 | 10.47 | +10.0 |
| 3 | 13 | .462 | .519 | +5.77 | 13.86 | +10.7 |
| 4 | 13 | .615 | .635 | +1.92 | 13.36 | +5.0 |
| 5 | 10 | .500 | .550 | +5.00 | 15.73 | +10.0 |
| 6 (weight 6) | 17 | .765 | .765 | **0.00** | 10.29 | 0.0 |

Collapsed: **hard (n_correct 1–3): +5.00pp** · **easy (4–6): +1.87pp**. Spearman(tier, Δ) = **−0.54** (n=6 bins;
|ρ|≥0.83 needed for p<.05 — **not significant**).

Defect #2 predicts gains concentrate where training weight is high (easy records). The data show the
opposite sign, and the easy-end zero is partly a ceiling artifact (only .235 headroom at n_correct=6).
**Honest verdict: no detectable anti-difficulty gradient, at bin SEs of ±8–16pp.** This is underpowered,
but it is the only evidence available, and it does *not* support reweighting as a priority. See D5.

### F7 — Stage A reproduces exactly from disk; conclusions unchanged.
Receipt: `lora_consistency_analysis.json` → `stage_a_val`. BASE 20/40; C0 21 (+1, p=1.0), C4 18 (−2, p=.79),
C2 17 (−3, p=.63), C3 15 (−5, p=.30), C1 15 (−5, p=.27), C5 12 (−8, p=.077). Not one arm is significant,
including the "worst". At SE ±9.3pp the grid supports exactly one statement: **no gross win exists**; it
cannot rank the arms or establish flatness. Treat C1–C5 as *unrefuted*, not as *refuted*.

### F8 — Anchor drift is small and mostly benign; one item needs a ruling.
Receipt: `lora_sensitivity_analysis.json` → `anchors`. Base 9/10 solved, 0/10 fail. Seeds: 9/8/9/8 solved, 1/1/0/0 fail-solved.
- `b4a60d33bce8` sits in the **anchor_solved** slice but the **base model fails it** (base=0), and **all four
  adapters solve it**. That is either a mislabeled anchor or a genuine consistent recovery — worth knowing which.
- `4ea3bb00c528` is an **anchor_fail** solved by 2 of 4 adapters. A fail-anchor being solved is the guard
  firing; at 2/4 it is weak, but it should not be ignored.
- Remaining movement (4 anchors flipping in 1–2 seeds each) is within the F3 churn envelope. **No trend.**

---

## 2. POWER STATEMENT (read before any decision below)

Measurement SE, greedy pass@1, from the observed discordant rate .3475:

| n records | k=1 | k=3 | k=5 | k=8 | k=12 |
|---|---|---|---|---|---|
| 40 | 9.32 | 5.38 | 4.17 | 3.30 | 2.69 |
| 100 | 5.89 | 3.40 | 2.64 | 2.08 | 1.70 |
| 200 | 4.17 | 2.41 | 1.86 | 1.47 | 1.20 |

**t-test route** (magnitude), using the observed seed SD of 4.99pp — MDE at 80% power, α=.05:
k=5 → 6.25pp · **k=8 → 4.94pp** · k=10 → 4.42pp · k=16 → 3.49pp · k=20 → 3.12pp.
Seeds needed to reach p<.05 *if the true effect equals the observed value*: +5.0pp→7, **+3.75pp→10**, +3.0pp→14, +2.0pp→27.

> **The pre-committed n=8 design has MDE 4.94pp against a best-estimate effect of +3.75pp (or +1.33pp
> excluding the outlier). It is underpowered on the t-test route and will very likely return "n.s.".**

**Sign-test route** (consistency — the mission-aligned statistic). Needed positives for p<.05, and power:

| k seeds | need | power @ true P(improve)=0.8 | @0.9 | @0.95 |
|---|---|---|---|---|
| 8 | **8/8** | 0.17 | 0.43 | 0.66 |
| 10 | 9/10 | 0.38 | 0.74 | 0.91 |
| **12** | **10/12** | 0.56 | **0.89** | 0.98 |
| 16 | 13/16 | 0.60 | 0.93 | 0.99 |
| 20 | 15/20 | 0.80 | 0.99 | 1.00 |

Two consequences drive every recommendation below:
1. **n=8 demands a perfect 8/8 sweep** to clear p<.05 — power 0.43 even if the dataset genuinely helps 90%
   of the time. **n=12 needs only 10/12** and reaches power 0.89 at the same truth. n=12 is the cheapest
   design that can actually deliver the mission's verdict.
2. **Ties are fatal to the sign test and greedy manufactures them.** Greedy pass@1 at n=100 yields integer
   deltas; seed 20260725 landed on exactly 0 and is dropped, silently cutting n. 1-in-4 so far.

---

## 3. DECISIONS

**D1 [EXEC] — Finish stage-R exactly as specified. Change nothing mid-flight.**
Seeds 20260726/27/29/30, control config, greedy, 100-record holdout. Rationale: a protocol change mid-stage
makes the seeds non-poolable and destroys the only replication series that exists. The design flaws in §2 are
real but they are *insufficiency*, not *invalidity*.

**D2 [NICKY] — Extend the control replication from 8 to 12 seeds.**
Rationale: §2. n=8 requires an unbroken 8/8 to say anything; n=12 requires 10/12 and triples the power at the
effect size actually on the table. This is 4 more trainings + 4 more holdout evals of the *already frozen*
control config — it is not a new arm, not selection, and not a protocol change. Needs Nicky because it is box
time and spend. **This is the highest-value marginal spend in the campaign.** If only one thing is approved,
approve this.

**D3 [EXEC, after D1 completes] — Decide greedy-vs-avg@8 from the k=8 variance ratio, not now.**
F4's ratio (1.86, p≈.13 at k=4) is the discriminator:
- ratio ≈ 1 at k=8 ⇒ seed spread is measurement ⇒ **avg@8 @ temp 0.7 is worth it** (cuts the per-record
  sampling component from .249 to .031; SE(n=100) 4.99pp → 1.76pp) and, critically, **eliminates ties** so
  the sign test keeps its full n.
- ratio ≫ 1 at k=8 ⇒ genuine adapter-quality spread ⇒ avg@8 buys much less; **spend on seeds instead**.
Recompute the ratio the moment seed 8 lands; the rule above is pre-committed so this is not a selection decision.

**D4 [EXEC] — Adopt the sign test on seed-level deltas as the primary statistic; t-test/CI secondary.**
Rationale: Nicky's mission is consistency, not magnitude. The sign test tests exactly that, is distribution-free,
and is not wrecked by the +11 outlier that dominates the mean (F2). Pre-commit now, before the remaining
seeds land, so it cannot be accused of post-hoc selection. Report both; lead with the sign test.

**D5 [NICKY, recommend DEFER] — Do not build dataset v2 yet.**
Split the three defects:
- **#1 full-sequence loss (21.6% of trained characters are prompt).** A real defect, worth fixing, and the fix
  is well-understood (prompt/completion columns; completion-only masking is trl 0.29.1 default). But fixing it
  **changes the recipe and resets the seed count to zero.**
- **#2 weight = n_correct.** F6 finds **no gradient in the predicted direction** (hard records gained *more*:
  +5.00 vs +1.87pp). The motivating prediction is unsupported. Reweighting is not evidence-backed.
- **#3 undocumented knobs.** Documentation defect, not a training defect. Fix in the manifest at zero cost.

Rationale for deferring: the campaign currently has **no established effect to improve on**. Re-rolling the
recipe before the consistency verdict lands spends the replication series and leaves us with two half-powered
arms instead of one adequately-powered one. **Finish the verdict on the frozen recipe first; then, if the
verdict is positive, v2 becomes a clean "does fixing #1 raise it?" follow-up with a real baseline to beat.**
If the verdict is null, v2's masking fix becomes the leading hypothesis for *why* — also a better position.

**D6 [EXEC, zero cost, do now] — Write the hidden knobs into the manifests.**
`gradient_accumulation_steps=4` (`train_qwen3_lora.py:131`, ⇒ effective batch 16), scheduler linear→0,
no warmup, no weight decay. These are inherited TRL defaults absent from every run manifest. A reader
today cannot reproduce the run from the manifest. Documentation only — **do not touch trainer code.**

**D7 [decided, no further action] — No further hyperparameter search.**
Rationale: at SE ±9.3pp (F7) a 40-record single-seed screen cannot resolve anything smaller than a ~19pp
swing. Every untested axis on record (rank/epochs down, attention-only targets, dropout, α/r ratio, warmup)
targets effects far below that. A screen that could detect a 5pp difference between two arms needs ~n=200 ×
k=8 *per arm* — more compute than the entire replication series, spent on magnitude, which the mission
deprioritizes. **Rejected on power, not on interest.**

**D8 [NICKY, recommend AFTER v2] — Do not go to N≈1000 yet.**
N≈1000 addresses magnitude, not consistency, and would land on the same unfixed recipe. Sequence:
consistency verdict (D1/D2) → v2 recipe fix if warranted (D5) → then N-scaling. Going first would burn the
largest single expense in the campaign on a recipe with a known, unfixed loss-masking defect.

**D9 [EXEC] — Adjudicate the two anchor items in F8.**
Cheap, and one of them touches guard integrity. Details in §4-WO4.

---

## 4. EXEC-LANE WORK ORDERS

### WO1 — finish stage-R (D1)
- **Do:** complete holdout greedy evals for seeds 20260726, 20260727, 20260729, 20260730, control config,
  engine `b1-c0bc859`, greedy k1/temp0, 120-row eval set.
- **Artifacts:** `out/evalharness/run1_sR<seed>/post_greedy.jsonl`, 120 rows each.
- **Accept:** 4 files exist, 120 rows each, `eval_slice` counts 100/10/10, engine fingerprint `b1-c0bc859`
  in each log. **Change no hyperparameter, engine, or split.**

### WO2 — rerun the consistency analysis at n=8 (D1, D3, D4)
- **Do:** re-run both analysis scripts unchanged; they auto-discover the new seed dirs.
- **Report back, verbatim:** per-seed Δ table; sign test (positives/negatives/**ties**); mean, SD, t, 95% CI;
  **the F4 variance ratio and its χ² p-value**; the F3 frozen-record count.
- **Accept:** `seeds_present` lists 8; `seeds_missing` empty.
- **Then:** apply D3's pre-committed rule and report which branch fired. Do not act on it without Nicky (D2 gate).

### WO3 — manifest knob documentation (D6)
- **Do:** add to `src/loratrain/data/run1_final/dataset_manifest.json` and
  `data/hp_tuning/dataset_manifest_160.json` a `trainer_defaults` block recording:
  `gradient_accumulation_steps: 4`, `effective_batch_size: 16`, `lr_scheduler: "linear_to_zero"`,
  `warmup_steps: 0`, `weight_decay: 0.0`, `packing: false`, `save_strategy: "no"`,
  plus `source: "TRL SFTConfig defaults + train_qwen3_lora.py:131 hardcode"`.
- **Accept:** both manifests parse; no other key changes; **trainer source untouched**; sha of
  `sft_train.jsonl` unchanged.

### WO4 — anchor adjudication (D9)
- **Do:** for `b4a60d33bce8` — confirm from the corpus + baseline row whether it belongs in `anchor_solved`
  given base `n_correct=0`. Report the corpus row's `n_correct`/`pass_at_k` and the baseline generation.
  For `4ea3bb00c528` (anchor_fail, solved by seeds 20260722 & 20260723) — report both generations and the
  corpus `answer`, so Nicky can rule "genuine solve" vs "guard breach" vs "mislabeled anchor".
- **Accept:** a short note under `out/analysis/`; **no change to the split or the anchor slices** (out of scope).

### WO5 — box logs (blocked on this lane, needed for F4 follow-up)
- **Do:** fetch `/workspace/hp_stage_r.log` to `out/analysis/box_logs/`.
- **Why:** per-seed final train loss + token accuracy across the 8 control seeds tests whether the +11
  outlier's adapter is distinguishable *at training time*. If it is, F4's ratio>1 branch gains independent
  support and D3 resolves toward "more seeds".
- **Accept:** log present; per-seed final-loss table reported.

---

## 5. EXPLICITLY REJECTED (do not re-litigate)

1. **Per-uid training-weight × eval-flip join** — impossible; train ∩ holdout = 0 uids (F5). The
   difficulty-tier proxy (F6) is the only available substitute. Retiring this from the analysis backlog.
2. **Any further HP grid** (D7) — rejected on power, not interest. Reopen only with n≥200 and k≥8 *per arm*.
3. **Treating stage A as evidence that C1–C5 are worse than C0** (F7) — no arm is significant; ±9.3pp SE.
   "C5 is worst" is not a finding. C0's +1 is equally not a finding.
4. **Building the campaign claim on seed 20260722's +11pp** (F2) — leave-one-out drops the mean to +1.33.
5. **Switching to avg@8 right now** — deferred to D3's pre-committed rule at k=8. Switching mid-flight
   would fragment the replication series (D1).
6. **N≈1000 before the consistency verdict** (D8).
7. **Selection on the holdout** — no configuration comparison in this doc uses holdout data; the holdout is
   used only for replication reads of the frozen control config, per the brief's standing constraint.

---

## 6. WHAT THE CAMPAIGN MAY AND MAY NOT CLAIM

**May claim (supported today, n=4):**
- Four independently trained adapters on pde625 produce holdout deltas of +11, +3, +1, 0 pp: **no seed
  regressed below baseline.**
- Per-record outcomes are **strongly reproducible across independently trained adapters** (χ²=157.4 vs the
  noise null; ICC≈0.46; 54/100 records frozen). The evaluation is measuring something stable.
- Anchor guards held: 8–9/10 solved-anchors retained, 0–1/10 fail-anchors solved, no trend across seeds.
- Hyperparameter deviations from the control produced **no gross win** on a 40-record val screen.

**May NOT claim:**
- ❌ "pde625 improves the model" — **not established.** Mean +3.75pp, 95% CI **[−4.19, +11.69]**, spans zero.
- ❌ Any per-seed McNemar result as significant — the best is p=.061 and it is the outlier seed.
- ❌ "+5.0pp mean" (superseded by +3.75) or "3/3 positive" (it is 3 positive + 1 tie of 4).
- ❌ That C0 is the best configuration, or that C1–C5 are harmful (F7).
- ❌ That the effect is consistent — **1 of 4 seeds delivered exactly zero**, and removing one seed moves
  the mean from +3.75 to +1.33 (F2). Consistency is precisely what is not yet demonstrated.
- ❌ Any statement implying n=8 evals exist. Four do.

**The honest one-line status:** *four control replications on pde625 give a small non-negative holdout shift
(mean +3.75pp, CI spanning zero) carried substantially by one seed; the measurement itself is demonstrably
stable, and the consistency question the campaign exists to answer needs 12 seeds — not the 8 currently planned.*

---

## 7. WHAT I NEED FROM NICKY

1. **D2** — approve 4 additional control seeds (8→12). Highest-value marginal spend; without it the sign
   test needs a perfect 8/8 and will most likely land "inconclusive".
2. **D5** — confirm deferring dataset v2 until the consistency verdict lands.
3. **D8** — confirm N≈1000 comes after v2, not before.
4. **WO4** — rule on the two anchor items once the exec lane reports them.

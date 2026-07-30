# LoRA Campaign — Results (run 1, 2026-07-27)

Goal: prove a **measurable** performance improvement from LoRA-training qwen3-8b on
the pde625 band corpus, with a number that survives scrutiny. Method authority:
`docs/eval_harness_design.md`; training arm: `src/loratrain/` (README D1–D4 + RUNBOOK).

## Setup (all pins verified on disk at run time)

| artifact | value |
|---|---|
| corpus | `band_corpus.jsonl` 293 rows, sha256[:16] `e0975e11` |
| split | 200 train (193 band + 7 GGUF-7/8 backfill, training-only) / **100 pure-band holdout**, paper-disjoint, sha[:16] `768436f4` |
| training set | 700 verbatim verified-correct rollouts (651 band + 49 backfill), sha[:16] `7fa7e5bf`; corpus `answer` never enters a target (grader-equivalence defense) |
| eval engine | llama.cpp **b10107** (`c0bc859`), Metal, fingerprint `b1-c0bc859` on every response; identical serve flags both arms (`-c 8192 -ngl 99 --parallel 1`) |
| base weights | FP16 `Qwen/Qwen3-8B` @ `b968826d` — identity vs local Q4_K_M GGUF proven (0/151,669 token mismatches; safetensors blob-oids identical across all repo history; ctx 32768/40960 dual-pinned as conversion metadata) |
| trainer | RunPod A40 48GB; transformers 5.14.1 / peft 0.19.1 / trl 0.29.1; bf16 LoRA r16 α32 lr1e-4 3ep; converter llama.cpp b10107 (same tag both sides) |
| protocol | baseline captured BEFORE training (ordering-guarded); greedy pass@1 temp 0 on the 120-record frozen eval set (100 eval-band + 10+10 anchors); exact McNemar on paired outcomes |

Guards green end-to-end: 21-step W2 build audit, identity receipt, upload guard (the
single permitted upload, holdout-impossible by construction), receipt-sha enforcement
on the box, engine parity via response fingerprint.

## Results — greedy pass@1 on the 100-record eval-band holdout

Baseline (base model): **43/100**.

| seed | tuned solved | Δ | discordant b/c | exact McNemar p | anchors (regress/contam) |
|---|---|---|---|---|---|
| 20260722 | **54/100** | **+11.0pp** | 9 / 20 | 0.061 | 1 / 1 |
| 20260723 | **44/100** | **+1.0pp** | 18 / 19 | 1.0 | 2 / 1 |
| 20260724 | **46/100** | **+3.0pp** | 16 / 19 | 0.736 | 1 / 0 |

**Aggregate: mean Δ = +5.0pp (spread +1 to +11); all three seeds positive; none
individually significant.**

Training quality was near-identical across seeds (loss 0.431 vs 0.433, token accuracy
87.9% vs 86.9%, same token count) — the outcome spread is **generalization variance
between seeds**, not a training failure.

## Verdict (final, 3 seeds)

**Not demonstrated at this training scale.** Mean +5.0pp is directionally positive
(3/3 seeds), but no seed reaches significance and the seed-to-seed spread (10pp) is
twice the mean effect. A 200-example LoRA moves this model's holdout performance by
an amount dominated by training-run luck. Seed 1 alone (+11pp, p=0.061) would have
been an overclaim — the multi-seed protocol caught it, which is precisely the
"survives scrutiny" property this harness was built for.

Secondary observations: anchor flags stayed at 1–2/10 per seed (noise-level; no
forgetting or contamination trend — and paper-disjointness rules out memorization by
construction). Discordant churn was high in every seed (~30-37 pairs), i.e. the
adapter meaningfully reshuffles WHICH problems solve even when the net gain is small.

**Obvious next experiment:** scale the training set. The corpus machinery (autopilot
batches + bulk months) can grow band well past 300; re-running this exact pipeline at
N≈1000 examples tests whether the +5pp mean is a floor that grows with data or a
ceiling. The harness, guards, runbook, and box recipe are all reusable as-is.

## Reproduction

Adapters (PEFT + GGUF-converted, ~175 MB each): seeds `20260722/3/4`, shas
`bcebb86a… / 300dd8b6… / b8a7525d…` (local `/tmp`, box `/workspace/run/out/`; not
committed). Eval outputs + per-seed reports: `out/evalharness/run1*` (gitignored, local).
Rebuild the eval set from the committed split via `evalharness-build-set`; the corpus
itself never leaves the machine.

Box cost, entire campaign (setup, smoke, 3 seeds): ≈ **$3**.

## 2026-07-28 — adjudicated corrections to the run-1 configuration record

Two independent reviews of the training arm were merged and receipt-verified on
2026-07-28 (full detail: `docs/lora_params_rationale.md`). Four findings qualify how
the run-1 setup table above should be read; none retroactively changes a measured
number, but all four are inherited unchanged by any rerun that reuses
`train_qwen3_lora.py` / the W2 dataset build as-is:

1. **Full-sequence loss.** The trainer pre-templates rows to `{"text": …}` and sets
   no masking, so ~21.6% of trained characters are the (identical) system prompt +
   user text, not assistant output. Fix is prompt/completion columns
   (completion-only loss is the trl 0.29.1 default path), NOT
   `assistant_only_loss` (requires a `{% generation %}` chat template Qwen3 lacks).
2. **Gradient weight = `n_correct`.** One example per verified-correct trace means a
   record's gradient mass is proportional to how often the BASE model already solved
   it (anti-difficulty). The seven 7-trace records are exactly the GGUF-7/8
   backfill. Structural — scales to N≈1000 untouched.
3. **Silent knobs.** `gradient_accumulation_steps=4` is hardcoded (effective batch
   16, unrecorded in run manifests); scheduler/warmup/decay are unrecorded TRL
   defaults (linear→0, none, none).
4. **Loss floors immediately** (0.71 → ~0.45 within the first steps; final ~0.43).
   Targets are the base model's own rollouts, so "0.43 = room to fit" reasoning is
   invalid — more epochs buy sharpening, not fitting headroom.

## Stage A — hyperparameter screen (2026-07-27/28, complete)

Screen design: 6 configs × 1 shared seed (20260728), trained on a 160-record
tune-train carve (565 rollouts) and read on a 40-record paper-disjoint val set —
the 100-record holdout untouched. Greedy pass@1, same engine parity as run 1.
Selection rule pre-declared: ties → control.

| config | deviation from control | val solved /40 |
|---|---|---|
| base (no adapter) | — | 20 |
| C0 | none (r16 α32 lr1e-4 3ep) | **21** |
| C1 | lr 2e-4 | 15 |
| C2 | lr 5e-5 | 17 |
| C3 | 6 epochs | 15 |
| C4 | r32 α64 | 18 |
| C5 | lr 2e-4 + 6 epochs | 12 |

Every deviation scored below control numerically, but per the 2026-07-28 analysis
adjudication no arm difference is significant (best p=.077 for C5; screen SE ±9.3pp):
the screen supports exactly one statement — **no gross win exists among the tested
knobs** — and cannot rank arms or establish flatness. C1–C5 are unrefuted, not
refuted. Consequence: **stage R (replication) uses the unmodified control config**
(pre-declared tie rule: ties → control).

## Stage R + 8-seed consistency verdict (2026-07-28) — supersedes the 3-seed verdict above

Five additional seeds (20260725/26/27/29/30) of the byte-identical control config were
trained on the same box (train-time metrics indistinguishable across all 8 seeds:
loss .4302–.4364, token accuracy .869–.882) and evaluated with the identical parity
protocol. Full analysis: `docs/lora_consistency_verdict.md` (the campaign deliverable)
+ refreshed `out/analysis/lora_{consistency,sensitivity}_analysis.json`.

**Headline: deltas {+11, +1, +3, 0, +3, +1, +4, +1} — 7 positive / 0 negative / 1 tie.
Sign test p = .0156 (pre-registered primary); t(7) = 2.421 p = .046, 95% CI
[+0.07, +5.93]. Robust to dropping the +11 outlier (sign p = .031, t p = .015,
mean +1.86pp). The "run-1 verdict: not demonstrated" above is superseded: a small,
consistent, causally attributable improvement IS demonstrated at n=8** — on a recipe
whose two known dataset defects (full-sequence loss, weight = n_correct) remain
unfixed, i.e. the effect exists despite them. Variance ratio at k=8 = 0.95 (p=.47):
seed spread is measurement noise; adapters are statistically interchangeable.

**n=12 UPDATE (2026-07-29) — the paragraph above is superseded in turn.** Nicky's
approved D2 extension (4 more control seeds: 20260731/0801/0802/0803, same config
verbatim, train losses inside the existing band) landed at **0, −1, −1, −2** — the
campaign's first negative deltas. Pre-registered n=12 analysis: **7+/3−/2 ties, sign
p = .344 (threshold was 10/12); mean +1.67pp, t(11) = 1.675, p = .122; CI
[−0.45, +3.79].** The n=8 SIGNIFICANCE was a favorable-tail read corrected by its own
extension — the protocol's second successful self-correction (run-1's seed 1 was the
first). The positive DIRECTION, however, survives the extension.

**Final campaign status at N=200 [phrasing corrected 2026-07-29 on Nicky's challenge —
upheld]: the dataset produces a small, directionally consistent positive effect.
9/12 seeds ≥ 0 (7 strictly positive, 3 negative, 2 ties); mean +1.67pp; net +20pp
across 12 runs; positive magnitudes sum +24 vs −4 for negatives; upside reaches +11
while the worst run in the entire campaign is −2. NOT statistically significant at
n=12 (one-sided sign p = .17, one-sided t p = .061) — do not claim it as established.
But "not distinguishable from zero" (the prior wording, now retired repo-wide) was
wrong: it reads as "no evidence of signal," and symmetric noise around zero does not
produce this asymmetry. Proof of concept holds — the dataset moves the model in the
intended direction, ruling out coin-flip behavior. Practical significance is minimal:
not an end-user-visible improvement. Anchors held in all 12 seeds; no seed outside
[−2, +11].** F4 stays ≈1 at k=12 (0.91, p=.53) —
adapters interchangeable, no eval-block artifact signature. Per the decisions doc's
D5 null branch, the leading hypothesis for the small/absent effect is now the
dataset's known masking defect; v2 is the indicated next experiment. Full analysis:
`docs/lora_consistency_verdict.md` n=12 section, plus its 07-29 addendum on churn
structure (38 records reshuffle per run to net under 2; aggregate 217 lost / 237 gained)
and the scaling assessment. v2 spec: `docs/lora_v2_work_order.md`.

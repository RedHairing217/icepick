# LoRA campaign — 8-seed consistency verdict (2026-07-28)

> **SUPERSEDED 2026-07-29 — the verdict does not survive the D2 extension to n=12.**
> The four extension seeds landed at 0, −1, −1, −2 (the campaign's first negatives):
> sign test 7+/3−/2 ties p=.344; mean +1.67pp, t(11)=1.675 p=.122; 95% CI
> [−0.45, +3.79] spans zero. See "n=12 update" at the end of this file. The n=8
> analysis below is retained as the historical record of what a favorable-tail
> read looks like — and of the replication protocol correcting itself.

**Mission (Nicky, 2026-07-28):** demonstrate that the pde625 dataset CAUSES improvement.
Small is fine; consistency is the claim. Magnitude search is a non-goal; selection on
the holdout is forbidden.

**Instrument (pre-registered before any stage-R holdout read; D4 of
`docs/lora_decisions_2026-07-28.md` pre-commits the sign test as primary):** per-seed
holdout-delta distribution of the frozen control config vs the shared frozen baseline;
sign test primary, one-sample t + CI secondary.

## Result

**The consistency claim is demonstrated at n=8.** Both pre-committed statistics clear
α=.05, and the result survives removal of the one large seed.

Eight independently seeded trainings (identical config r16/α32/drop.05/lr1e-4/3ep,
identical 700-row dataset `7fa7e5bf`, RunPod A40, trl 0.29.1) evaluated greedy pass@1
on the 100-record pure-band holdout vs baseline 43/100 (llama.cpp b10107 both arms,
fingerprint `b1-c0bc859` verified live on both servers at serve time):

| seed | stage | tuned solved | Δ pp | b / c (discordant) | exact McNemar p | anchors kept /10 | fail-anchors solved /10 |
|---|---|---|---|---|---|---|---|
| 20260722 | run1 | 54/100 | **+11** | 9 / 20 | .061 | 9 | 1 |
| 20260723 | run1 | 44/100 | +1 | 18 / 19 | 1.0 | 8 | 1 |
| 20260724 | run1 | 46/100 | +3 | 16 / 19 | .736 | 9 | 0 |
| 20260725 | stageR | 43/100 | **0** | 19 / 19 | 1.0 | 8 | 0 |
| 20260726 | stageR | 46/100 | +3 | 18 / 21 | .749 | 10 | 1 |
| 20260727 | stageR | 44/100 | +1 | 18 / 19 | 1.0 | 8 | 0 |
| 20260729 | stageR | 47/100 | +4 | 18 / 22 | .636 | 10 | 0 |
| 20260730 | stageR | 44/100 | +1 | 22 / 23 | 1.0 | 8 | 1 |

**Primary (sign test on deltas, two-sided, ties dropped): 7 positive / 0 negative /
1 tie → p = .0156.** No seed regressed below baseline.

**Secondary (t on deltas): mean +3.0pp, SD 3.51, t(7) = 2.421, p = .046 two-sided;
95% CI [+0.07, +5.93]** — excludes zero.

### Robustness — the +11 outlier is not load-bearing

Leave out seed 20260722 (the +11, flagged by F2 of the decisions doc as the salience
risk): remaining deltas {+1, +3, 0, +3, +1, +4, +1} — mean +1.86, SD 1.46,
**sign test 6/0 p = .031, t(6) = 3.357, p = .015**. The outlier carried magnitude,
not consistency; the conservative effect estimate is **≈ +2pp**.

Train-time receipts (`out/analysis/box_logs/train_time_summary_20260728.md`): all 8
seeds are indistinguishable in final loss (.4302–.4364) and token accuracy
(.8685–.8824); the +11 seed sits mid-pack on both → its holdout excess was draw luck,
not a detectably better adapter.

### D3's pre-committed branch (F4 resolved)

Variance ratio of per-seed totals vs the independent-flip prediction at k=8:
**0.95 ≈ 1** (χ²(7) = 6.63, p = .47). The k=4 point estimate (1.86) was noise, as F4
anticipated. **Branch fired: seed-to-seed spread is measurement noise, not adapter
quality spread ⇒ avg@8 @ temp 0.7 is the value buy** (it also eliminates greedy's
integer ties — one of which materialized in seed 20260725 and cost the sign test an
observation). Per WO2, this branch is REPORTED, not acted on; instrument changes and
any seed extension are Nicky's call (D2 gate).

### Structure (F3 at k=8)

Per-record outcomes remain strongly reproducible across the 8 independent adapters:
χ² = 1491 vs the iid null, ICC ≈ 0.40; 35/100 records are frozen across all eight
(22 never solved, 13 always solved). The instrument is stable; the churn is real
per-record flip probability, uniform in aggregate across seeds (variance ratio ≈ 1).

### Anchors

Solved-anchors kept: 8–10/10 every seed; fail-anchors solved: 0–1/10 every seed; no
trend across seeds. Two boundary items are documented with full generations and label
history in `out/analysis/anchor_adjudication_20260728.md` (WO4): both flagged anchors
are records whose k=8 rescore verdicts flipped relative to the original processing era
(8/8-vs-7/8 and 0/8-vs-5/8 respectively) — i.e. measurement-boundary records, pending
Nicky's ruling; no change made to any slice.

## What may now be claimed (supersedes §6 of the decisions doc where counts differ)

- Eight independently trained adapters on pde625 produced holdout deltas of
  {+11, +4, +3, +3, +1, +1, +1, 0}: **no seed regressed; 7/8 improved; both
  pre-registered tests significant (p = .016 sign, p = .046 t).**
- **"pde625 causes a small, consistent holdout improvement" is now supported** at
  mean +3.0pp (95% CI [+0.07, +5.93]), conservative LOO estimate ≈ +2pp — under a
  recipe with two known, unfixed dataset defects (full-sequence loss, weight =
  n_correct). The effect exists despite them.
- The measurement is stable (F3), and adapters are statistically interchangeable
  (F4 ratio ≈ 1; train-time metrics indistinguishable).

Still may NOT be claimed: any per-seed significance (best individual McNemar p = .061);
any stage-A ranking (screen SE ±9.3pp; C1–C5 are unrefuted, not refuted); any
magnitude beyond the CI; anything about avg@8 numbers (not yet run).

## Open for Nicky (in decision-doc terms)

1. **D2 (8→12 seeds):** its premise ("n=8 will very likely return n.s.") did not
   materialize — n=8 delivered. Extension now buys CI tightening (MDE 4.94→3.49pp at
   n=16-class power), not rescue. Approve only if the tighter interval is worth ~$2.
2. **D3 follow-on:** the pre-committed branch says avg@8 is the right instrument
   upgrade for any future holdout reads. Needs your protocol sign-off (it changes the
   instrument; greedy k=1 remains the one already-measured series).
3. **D5 (dataset v2):** the doc's own sequencing logic now points at v2 — the verdict
   is positive, so "does fixing the masking raise it?" has a real baseline to beat
   (~$2-3 box time).
4. **D8:** N≈1000 after v2, per the doc.
5. **WO4 ruling** on the two anchors (evidence note ready).
6. **Box teardown:** every unique artifact is now local (5 R adapters + all four box
   logs archived under `out/analysis/box_logs/`). Box is idle at ≈$0.33/day.
7. **Commit release** for the uncommitted docs/manifest edits (this file,
   `lora_campaign_results.md`, `lora_params_rationale.md`, `lora_decisions_2026-07-28.md`,
   two dataset-manifest `trainer_defaults` blocks, `out/analysis/*`).

## Provenance

Baseline `out/evalharness/run1/baseline_greedy.jsonl` (43/100, captured pre-training);
seed evals `out/evalharness/run1*/post_greedy.jsonl` (120 rows each; stage-R runs
produced by the 2026-07-28 serial driver, resume-safe, one :8082 server at a time).
Analysis: `out/analysis/analyze.py` (n=8 auto-discovery, unchanged) +
`out/analysis/analyze2_n8.py` (byte-copy of `analyze2.py` + the four stage-R paths
its hardcoded n=4 dict lacked — the only deviation from "rerun unchanged", disclosed);
refreshed artifacts `lora_consistency_analysis.json` / `lora_sensitivity_analysis.json`
with pre-refresh n=4 snapshots preserved alongside (`*_n4_snapshot_*.json`).
Exact p-values recomputed with scipy (analyze.py brackets agree). Engine fingerprint
`b1-c0bc859` verified live on both arms this session; NOTE it is response-level only —
persisting it into eval artifacts is a v2 harness item.

## n=12 update (2026-07-29) — D2 extension REVERSES the verdict

Nicky approved D2 (2026-07-28); four more control seeds trained on the box
(20260731/20260801/20260802/20260803, same `rc_R.json` hyperparams verbatim, dataset
sha pre-flight `7fa7e5bf` PASS, train losses .4309–.4335 — inside the existing band)
and evaluated with the byte-identical greedy protocol:

| seed | stage | tuned | Δ pp | b / c | McNemar p | anchors kept | fail-solved |
|---|---|---|---|---|---|---|---|
| 20260731 | stageR2 | 43/100 | 0 | 20/20 | 1.0 | 9/10 | 0/10 |
| 20260801 | stageR2 | 42/100 | **−1** | 18/17 | 1.0 | 10/10 | 1/10 |
| 20260802 | stageR2 | 42/100 | **−1** | 22/21 | 1.0 | 9/10 | 1/10 |
| 20260803 | stageR2 | 41/100 | **−2** | 19/17 | .868 | 10/10 | 1/10 |

**Pre-registered n=12 analysis (the D2 design's own read):** deltas
{+11, +1, +3, 0, +3, +1, +4, +1, 0, −1, −1, −2} — 7 positive / 3 negative / 2 ties.
**Sign test p = .344** (10/12 was the significance threshold; 7 is nowhere near).
Mean **+1.67pp**, SD 3.45, t(11) = 1.675, **p = .122**; 95% CI **[−0.45, +3.79]**.
Leave-one-out (drop +11): mean +0.82, sign p = .51, t p = .18.

**The n=8 significance was a favorable-tail read that its own pre-registered
extension corrected.** Exactly as run-1's seed 1 alone would have been an overclaim
that the 3-seed protocol caught, the 8-seed sweep was an overclaim that the 12-seed
protocol caught. P(improve) point estimate: 0.70 (7/10 non-ties), Wilson 95%
[0.40, 0.89].

**Honest one-line status: at N=200 training examples, the pde625 dataset's holdout
effect is small-positive at best (point estimate ≈ +1.7pp) and not distinguishable
from zero at n=12 seeds; no seed moved outside [−2, +11]; anchors held everywhere.**

Still-standing structural results: F4 seed-level variance ratio at k=12 = **0.91**
(χ²(11) = 9.96, p = .53) — adapters remain statistically interchangeable, seed totals
vary exactly as independent per-record flips predict (this also argues against an
eval-side shift between the R and R2 blocks); F3 record-level ICC ≈ 0.42, 32/100
records frozen across all 12 adapters (21 never / 11 always solved). The
instrument-integrity repro audit **PASSED**: seed 20260726 re-evaluated 07-29
reproduced its 07-28 result exactly — 46/100, **0 per-record mismatches across all
120 records** (`out/analysis/repro_audit_20260729.md`) — so the R2 deltas are genuine
adapter behavior, not eval-environment drift.

Consequence per the decisions doc's own D5 branch: *"If the verdict is null, v2's
masking fix becomes the leading hypothesis for why"* — that branch is now live.
Analysis artifacts: `analyze_n12.py` / `analyze2_n12.py` (disclosed path-extensions),
refreshed JSONs, n=8 snapshots preserved (`*_n8_snapshot_20260728.json`).

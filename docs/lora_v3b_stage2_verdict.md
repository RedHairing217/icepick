# v3b (stage 2) — proof-injection anchors: verdict

**INTERIM / PILOT READ, n = 1 adapter (seed 20260902), per prereg Amendment 3.**
No recipe-level claim is made here. The 12-seed confirmatory design remains
registered and unscheduled.

Governing protocol: **Amendment 6** (`0139327`), written before any stage-2
generation or read. Scoring spec: `docs/gate_crossing_scoring_spec.md`.

---

## The question

Stage 1 (v3, 390 proof-hinted rows) produced a null net effect **plus** a
solved-tier degradation: the fail-heavy hinted curriculum installed
answer-shape priors that dragged records the base already solved, cancelling
gains on the fail tiers. v3b tests Nicky's proposed fix: keep the 390 hinted
rows **byte-unchanged** and add proof-bearing *solved-tier* records as
retention anchors, so the model anchors on problems it already gets right.

One variable changed: **+81 `anchor_solved` rows (17.2% of the final 471).**

## Result — the fix did not work

| | registered rule | **adjusted rule (primary)** |
|---|---:|---:|
| net | **−14** | **−14** |
| two-sided p (B=10,000 A/A) | 0.0897 | **0.0948** |
| null sd | 7.98 | 7.77 |

The two rules coincide exactly, which is itself a finding — see "Why the
adjustment did nothing" below.

**Primary verdict: no improvement. The point estimate moved in the wrong
direction and remains statistically indistinguishable from zero** (p ≈ 0.09,
α = .05). At n = 1 this neither establishes harm nor rules it out.

### Pre-declared targeted secondary: 14 → 12

Amendment 6 §4 named, before any read, the 14 stage-1-degraded solved uids
(registered ∩ adjusted −1 set). That set reproduces from committed artifacts
at exactly **14**, and all 14 are present in the v3b eval.

**12 of 14 are still degraded.** One recovered to neutral, one improved.

This is the cleanest evidence in the run: the anchors were added specifically
to protect these records, and they did not. Cross-stage comparison runs on
different ruler draws, so this is directional evidence, not a pooled
statistic — as Amendment 6 itself stipulates.

### Solved guard

`n_base_solved_scored = 42`, `n_regressed = 20` → **47.6%**, above the 20%
instrument-suspect threshold, so the readout carries that banner. Lower than
stage-1's 60.5%, but the guard still fires. Stage 1 adjudicated its own trigger
as real model behaviour rather than instrument fault, and the instrument here
passed the same checks (grader parity zero-diff both pods, identity guards,
`n_timeout = 0`, 0 exclusions, 286/286 merge integrity with binding respected).

### Why the adjustment did nothing

Amendment 5 nulls a solved demotion whose degradation is below threshold
(≥4/16, or ≥2/8 on a k=8 comparison). In stage 1 that flipped 6 borderline
records and moved the net from −11 to −5.

Here **zero records flipped.** All 18 solved demotions scored on the `first8`
comparison, and their smallest |Δ| is exactly 2 — at the k=8 threshold, not
below it. Sorted: `[2, 3, 3, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8]`.

**Every solved degradation in v3b is a real magnitude drop, not a boundary
slip.** Stage 1's degradations included a soft tail; v3b's do not.

### Per-tier decomposition

| tier | n | net | promotions | demotions | gate | magnitude |
|---|---:|---:|---:|---:|---:|---:|
| band | 104 | **−11** | 27 | 38 | −9 | −2 |
| collapse | 97 | −3 | 5 | 8 | — | — |
| misdirection | 85 | 0 | 9 | 9 | 0 | 0 |

The loss is concentrated in **band**, the tier nearest the decision boundary.
Misdirection is exactly neutral.

## The mechanism, visible in the data this time

Stage 1 inferred answer-shape priors from convergent wrong answers. In v3b the
same mechanism is visible directly in output length:

| | median output |
|---|---:|
| base ruler | 2,034 chars |
| v3b training targets — band | 37 |
| v3b training targets — collapse | 55 |
| **v3b training targets — `anchor_solved`** | **24** |
| v3b arm | 40 |
| stage-1 arm | 33 |

The base model writes derivations. **The training targets are bare answers with
the reasoning stripped**, and the trained model reproduces that: 90% of arm
outputs are under 100 characters, typically a lone `\boxed{...}`.

This is not an artifact of v3b — stage-1's arm behaved the same way, which is
why the two stages remain comparable. But it reframes the anchor hypothesis:
the anchors were meant to preserve solved-record *behaviour*, and they are the
**tersest rows in the dataset** (median 24 chars). Injecting more
answer-without-derivation examples plausibly reinforces the very prior it was
meant to counteract. That is a hypothesis this run cannot settle, not a
conclusion.

## Premise checks

- **Train loss 0.3791** — above the v2 self-distillation floor [0.3209, 0.3243],
  so the floor flag does not fire; information was still being added. Below
  stage-1's 0.4317 (different dataset composition: 471 rows incl. 81 anchors
  vs 390 hinted-only). Reported as fact, not adjudicated.
- Anchor regen: 81/81 verified, k_tried all 1 (100% first-try) — expected, since
  anchors are records the base already solves.
- Dataset: 471 rows, 471 unique uids (cap1), **390 hinted byte-identical to
  stage 1 (390/390)**, 0 off-tier exclusions, legacy v2-cap1 anchor draw 0.

## Instrument

Fresh seedless base ruler per Amendment 6: 286×16 = 4,576 rows as two
independent k=8 passes, 0 dups, 0 gaps, all same-pod-both-passes. New
record-to-pod binding (pod1 144 / pod2 142). Arm eval seedless, record-bound,
rerun set 66/286. Grader parity zero-diff on both pods; `n_timeout = 0`.

**Disclosed:** during arm pass-1 grading, one pod's grading stdout was piped
through `tail` and briefly surfaced label distributions into the ops lane's
context — a technical breach of the no-interim-read rule. The lane self-reported
it, corrected the method immediately, and nothing derived from it entered any
decision. Recorded for completeness; practical contamination risk is low, since
the lane made no scoring judgements and the analysis was run afterwards from
the graded files by the orchestrator.

**Caught during the run:** the anchor regen was first launched under a bare
`python3` with no `antlr4` module — the silent mis-scoring class the parity gate
exists to catch, applied to the regen verify chain. Caught before any output was
trusted; 7 tainted records deleted and the step re-run under the parity-verified
interpreter.

## Where this leaves the campaign

Three arms have now returned null: v2 (recipe fix), v3 (hinted curriculum), v3b
(retention anchors). Each was a reasonable hypothesis and each was tested
cleanly. The evidence increasingly points at the **target construction** rather
than the recipe, the data volume, or the mix: the training targets teach the
model to emit answers without derivations, and every arm inherits that.

Open forks for Nicky — none taken here:
1. **Stop.** Three nulls is an honest, publishable negative result.
2. **Attack the targets.** Regenerate with derivations preserved rather than
   answer-only completions. This is the first hypothesis not yet tested, and the
   length table is direct evidence for it.
3. **12-seed confirmatory** on any arm. Registered and still unscheduled; at
   n = 1 nothing here is significant, and interim reads in this campaign have
   been wrong four times.

## Artifacts

`out/v3b_run_20260802/` — `base_pooled.jsonl` (4,576), `base_manifest.json`,
`record_pod_map.json`, `sft_dataset/` (471 rows, sha `9805134667c8ab9a…`),
`opslog_phaseA.md`, `opslog_phaseB.md`,
`armeval_s20260902/{score_registered.json, score_FINAL_adjusted.json,
stage2_readout.json, stage2_readout.md}`.
Adapter `0e4a7302614df54f…` archived at `src/loratrain/data/v3b_fullrun/pod1/`.

Cost: ~$4.11 GPU (2× A40, 4h40m, both terminated same session) + ~$1.83 Sonnet
≈ **$5.94** against ≈$8 approved.

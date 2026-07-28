# Judge Comparison + Codex:Anthropic Funnel Analysis Skeleton

Paste this into a fresh context window after the 2026-07-10 band miss audit. The window has
two jobs: (1) compare the audit's findings against another judge's verdicts, (2) analyze the
codex:anthropic judge funnel to determine what structural adjustment would catch these miss
classes in the future.

**This is an analysis scaffold, not a launch authorization — and this window makes NO code
changes.** Its final deliverable is a report plus a SECONDARY skeleton that a third context
window will execute. Analysis only: no edits to `src/`, `tests/`, or any existing file.

## Fresh Window Prompt

You are the analyst for an Icepick judge-quality investigation. Use Claude Fable 5 as the
orchestrator model if available.

Repository: `/Users/redhairing/Desktop/helloworld/icepick`

Read first, in order:

1. `AGENTS.md` (binding invariants — especially: out/ append-only, launches hold-gated,
   $5 HITL spend line, judge models come from env files NEVER `--*-judge-model` flags,
   verify notifications against disk, parallel sessions common)
2. `docs/SESSION_HANDOFF.md` (current holds and corpus state; read at least the 2026-07-10
   audit addendum)
3. `docs/wellposed_miss_audit_summary_20260710.md` (WHAT was flagged and WHY — the 43
   candidates, 3 tiers, mechanism clusters 1-7, lane rates)
4. `out/audits/wellposed_band_miss_audit_20260710T010302Z/audit_report.md` (tiered full uid
   lists) and `miss_candidates.jsonl` (row-level evidence)
5. `docs/pipeline_controller.md` §"Stage 3: processing wellposed-cascade"
6. `src/posers/AGENTS.md` (judge cache-key semantics, model config traps)

Re-verify on disk before anything: `band_corpus.jsonl` row count + whether Nicky has folded
out any audited rows since 2026-07-10 (the audit assumed 309 rows, sha `01609862e21fde14…`).
Record git branch/dirty state and corpus counts in your analysis manifest.

Hard boundaries:

- **No code changes anywhere.** No edits to existing files. New files only under
  `out/audits/…` (a new analysis dir) and the two deliverable docs under `docs/`.
- Do not mutate, fold, relabel, or remove corpus rows.
- Launches are hold-gated: NO live judge runs, scrapes, pass@k, Qwen, or paid API calls
  unless Nicky explicitly releases them **in your session**. Disk-first analysis is the
  default and is sufficient to complete both parts.
- If a live judge run IS released: judge models come from `ANTHROPIC_MODEL`/`OPENAI_MODEL`
  in the key env files (`/Users/redhairing/Desktop/helloworld/{anthro,openai}_key.env`);
  never steer per-provider via `--*-judge-model` (per-BUILD, poisons cross-provider combos).
  Cost reference: gpt-5.5 judge ≈ $0.02/sample → 90 records x 3 samples ≈ **$5.4, OVER the
  $5 HITL line**; Sonnet judge ≈ $0.002-0.004/sample → ≈ $0.6-1.1. Account for every call.
- Treat the audit's tiers as reference labels with stated confidence, not ground truth. Do
  not re-litigate individual audit verdicts except where a judge disagreement forces a look.
- The `needs_human` rows (`e5ed37d5…`, `a7b98a81…`) stay excluded from recall metrics;
  report them separately. Never resolve them by assumption.

Suggested output directory: `out/audits/judge_comparison_YYYYMMDDTHHMMSSZ/`

## Part 1 — Compare the audit results with another judge

Goal: measure whether an independent judge discriminates the audit's misses from its keeps —
i.e., would a different judge have caught what codex:anthropic passed?

Study population (fixed before looking at any judge output):
- the 43 miss candidates (T1 34 / T2 5 / T3 4),
- ~45 negative controls sampled from confirmed keeps, stratified to match the candidates'
  source_batch and tier profile (sampling rule + seed list recorded in the manifest),
- the 2 needs_human rows (reported separately).

Comparison judges, in priority order:

A. **Disk-first (free, default).** Mine verdicts already on disk for these uids:
   - stage-3 advisory flags: `wellposed_all_with_passk.json` rows carry
     `stage3_advisory_flag` for batches 3-8 era records — did the advisory claude:openai
     stage flag any of the 11 misses from 2stage+advisory lanes?
   - the parallel-fleet run `out/wellposed_pde625_claude_anthropic/verdicts/` (claude build,
     anthropic provider — a different BUILD than codex:anthropic): which of the audited uids
     appear, and how were they judged?
   - cascade stage verdict files under `out/processing_*/cascade/stage_*/` for each
     candidate's source batch: pull the codex:anthropic verdict/rationale text for the 43
     (needed for Part 2 regardless).
   - rescue-panel rulings (`out/stage1_kill_census/…/sweep_rulings.jsonl` etc.) for the
     rescue-lane rows: the panels judged well-posedness adjacent questions — what did they
     say about `f5416819…` and `76ac6e18…`?
   Coverage will be partial (not every uid saw every judge) — report coverage explicitly;
   no silent gaps.

B. **Live judge run (ONLY if Nicky releases it in-session).** Blind single-combo run of a
   structurally different judge over the ~90-row set, in a NEW output dir, statements only
   (no tier labels attached): preferred `claude:openai` (different build AND provider;
   gpt-5.5-strength judge; ≈$5.4 → needs explicit approval), fallback `claude:anthropic`
   (different build, same provider, ≈$1, still hold-gated). Use the poser CLI / cascade
   single-stage mode exactly as documented in `docs/pipeline_controller.md`; restartability
   contract applies.

Metrics (per judge source):
- Confusion vs audit labels: recall on T1 / T1+T2 / all-43, false-flag rate on controls.
- Recall by miss_type and by mechanism cluster (1-7 from the summary doc) — the actionable
  question is WHICH classes another judge catches, not the overall rate.
- Per-lane breakdown (does the second judge help most exactly where the Sonnet-only lanes
  leak?).
- Qualitative: for every candidate the other judge ALSO passed, note whether its rationale
  shows it saw the defect and excused it, or never saw it (feeds Part 2 hypotheses).

## Part 2 — Codex:anthropic funnel structural analysis

Goal: determine why the codex:anthropic gate passed these 43 and what structural adjustment
(prompt surface, inputs, sampling/uphold policy, added stage, pre-filter) would catch them —
WITHOUT killing good theorems (remember the 82.5% stage-3 false-kill history and Nicky's
never-reject-good-theorems rule; over-tightening is the known failure mode).

Questions to answer from code + artifacts (read, don't modify):
- `src/icepick/processing/poser/cascade.py` + `src/posers/Codex_Poser/` (+ its README and
  the judge prompt/rubric text): what EXACTLY does the codex:anthropic judge see per record?
  Statement only? Statement+answer? Does it ever see `metadata.source_statement`? (If not,
  mechanism cluster 4 — extraction distortion — is structurally invisible to it.)
- Sampling policy: `--judge-samples 3 --judge-uphold 2` — what were the actual per-sample
  verdicts for the 43 (pull from stage verdict/cache files)? Were any 2-1 squeakers?
- What does the judge's rubric ask? Does any prompt line ask "is every symbol defined /
  does the statement stand alone / could multiple answers be true"? Map each of the 7 miss
  mechanism clusters to the rubric line that should have caught it (or note its absence).
- Rate the five seeded hypotheses against transcript evidence, per cluster:
  H1 judge never sees the source → distortion invisible; H2 judge scores
  expert-answerability rather than statement-determinacy (accepts "the expert would know
  8*pi"); H3 single-gate lanes lack the redundancy that catches judgment variance;
  H4 no stand-alone/symbol-closure check → dangling-referent cluster leaks;
  H5 no blind-solve probe → monotone-membership/existence-constant inversions leak.

Candidate structural adjustments to EVALUATE (expected catch-rate per cluster on the 43,
false-kill risk argued from control rows and the FK history, cost delta per 250-record
batch; recommend, don't implement):
- S1 `$0` pre-judge lint: reject/flag statements matching dangling-referent patterns
  ("a system", "an ODE", "the stated assumptions", "as in Lemma", "suitable/appropriate
  conditions", undefined single-use symbols). Estimate from the summary doc's cluster 1-2
  membership how many of the 43 die here and spot-check false-positive rate on ~50 keeps.
- S2 judge rubric addition: explicit stand-alone checklist (all symbols defined? equations
  present? qualifier (sharp/optimal/the constant) justified by the statement?).
- S3 give the judge `source_statement` alongside `statement` + ask "does the statement
  preserve everything load-bearing?" (kills cluster 4; watch prompt-size/cost and the
  judge-cache-key roll it causes — see `src/posers/AGENTS.md`).
- S4 blind-solve probe: one extra sample answers the question WITHOUT seeing the key;
  non-equivalent answer ⇒ flag (targets clusters 3/5/6). Cost it.
- S5 restore a second independent gate for Sonnet-only lanes (2nd build or advisory stage;
  the lane table's 17.0% vs 11.3% is the motivating signal).
- S6 uphold-policy tightening (e.g., 3-of-3 to pass) — quantify against the per-sample
  verdict data from the 43 and argue false-kill impact before recommending.

## Deliverables (this window)

1. `out/audits/judge_comparison_…/comparison_report.md` — Part 1 metrics + Part 2 findings,
   with per-cluster evidence tables and an explicit recommendation ranking of S1-S6 (or
   better ideas found in the data), each with expected catch/false-kill/cost.
2. **The secondary skeleton**: `docs/funnel_adjustment_execution_skeleton.md` — paste-ready
   for a THIRD context window that will implement the chosen adjustments. It must contain:
   - a Fresh Window Prompt naming the chosen adjustment(s) and the files to touch
     (cascade.py / poser prompt files / new lint module path), with the poser cache-key and
     `--*-judge-model` traps restated;
   - the validation experiment: re-judge the 43 candidates + the recorded 45 controls with
     the adjusted funnel; acceptance criteria stated NUMERICALLY up front (recommend: T1
     recall ≥ 60% at control false-kill ≤ 5%, adjust from your Part 1/2 data), plus the
     full three-suite test baseline gate (`AGENTS.md` Quick facts numbers current at that
     time);
   - cost table + spend/hold gates for any live judging the experiment needs;
   - stop conditions: no corpus mutation; no silent judge-prompt edits without noting the
     cache re-bill; suite regression = stop; results to Nicky before any production
     default flips.
3. One-paragraph addendum appended to `docs/SESSION_HANDOFF.md` at session end.

## Stop conditions

Stop and ask Nicky before:
- any live/paid judge run (regardless of amount — launches are hold-gated; over $5
  additionally needs the explicit spend sign-off);
- writing anything outside the new analysis dir + the two deliverable docs;
- any conclusion that requires re-labeling corpus rows or contradicting the audit's
  needs_human handling;
- recommending an adjustment whose false-kill projection cannot be bounded from data.

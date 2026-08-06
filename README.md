# IcePick

A data-centric experiment testing whether a curated math corpus can improve a small model through LoRA fine-tuning. The corpus is the deliverable, the training arms are the test, and four independent arms have now returned null.

---

## What This Project Is

IcePick builds its own training data end to end. It scrapes arXiv, extracts theorems into answerable problems, filters them through a judge cascade, scores every survivor against the target model, and assembles what survives into a pinned corpus. That corpus is then used to fine-tune the model it was scored against, under a pre-registered analysis.

The research goal was to demonstrate that the dataset causes improvement. Consistency is the claim, magnitude is not.

This was the final project of a mentored research engagement, scoped from the outset to public arXiv data only so the work would be presentable outside the confidentiality agreement covering the rest of it. Claude wrote essentially all of the code. What I own is the objective, the gates, the adjudication when parallel lanes disagreed, and the decision to commission audit against results I would rather have kept. The git history attributes all 71 commits to me and does not distinguish the two, which is why I am stating it here instead of leaving it to the repo.

Current as of 8/6/26. Every figure below traces to a file in this repo, and the corpus is pinned at sha16 `e0975e11`.

---

## Pipeline

Seven stages, built in sequence.

1. **Acquisition.** In-house arXiv scraper under a plan → approve → run contract, so no paid run starts without an explicit release. 26 intake runs produced 5,860 candidate records.
2. **Extraction.** A generator converts theorems from those papers into answerable problems, ~54,000 calls.
3. **Filtering.** A multi-stage judge cascade gates every record for well-posedness, run across two model builds and two provider backends so verdicts can be compared rather than trusted.
4. **Scoring.** Each surviving problem gets 8 rollouts against the target model, verified symbolically.
5. **Corpus assembly.** Audits, repair lanes, and rescue passes over the band.
6. **Eval harness.** A paper-disjoint split, a baseline captured before any training, and an engine parity requirement so tuned and untuned models are measured identically.
7. **Training and verdict.** LoRA fine-tuning on a rented A40, seeded replication, then a pre-registered analysis.

---

## Corpus

2,021 records scored across 1,180 distinct arXiv papers. The distribution is sharply bimodal.

| n_correct of 8 | records |
|-----|--------|
| 0 | 1,298 |
| 1-6 (band) | 317 |
| 7 | 84 |
| 8 | 321 |

Problems solved 1 to 6 times out of 8 form the training band, hard enough to teach and not impossible. That band is 15.7% of what was scored, which is the number that matters for yield planning: ~6 records have to be scraped, extracted, judged, and scored to produce 1 usable problem.

Full label distribution across the 2,021: solved 406, drop 694, collapse 405, band 317, misdirection 199.

The assembled band corpus is **293 records, sha16 `e0975e11`**. The GGUF re-score of 7/15/26 assembled 309, and the extraction-defect repair lane removed defectives the following day. Both numbers appear in older documents and neither is wrong, they are consecutive states. Fold history is in `out/corpus_pde625/corpus_manifest.json`.

---

## Results

| Arm | What changed | Result |
|-----|--------------|--------|
| v1 (n=8) | baseline campaign | mean +3.0pp, sign test p = .0156 |
| **v1 (n=12)** | **pre-registered extension** | **mean +1.67pp, 9 of 12 at or above zero, sign p = .344** |
| v2 (n=12) | loss masking + `cap1` weighting | mean +0.58pp, 95% CI [-1.56, +2.72] |
| v3 (n=1) | 390 proof-hinted rows | null, plus solved-tier degradation |
| v3b (n=1) | +81 retention anchors | net -14, two-sided p ≈ 0.095 |

Baseline was 43/100 on the v1 holdout. Every arm was pre-registered before its read.

**The headline is v1 at n=12, and it is a null.** At 8 seeds the campaign was significant at p = .0156. The extension to 12 was already registered and outstanding, so I approved and paid for it. The four added seeds came back at 0, -1, -1, -2 and eliminated the finding. There is no methodological line between run 8 and run 9, so stopping at 8 would have meant reporting the favorable half of one experiment.

The direction survives and the magnitude does not. Nine of twelve runs at or above zero, positive magnitudes summing +24 against -4, worst run -2 against a best of +11. Each run also reshuffles 35 to 45 problems to net under two, so the adapter changes which problems solve far more than how many.

---

## What I Learned

**A found defect is not the defect until you measure it.** Outside audit found two real problems in how the data taught the model: no loss masking, so ~22% of training signal went to prompt text instead of answers, and a weighting scheme that gave the easiest problems the most gradient influence. Both were genuinely fixed in v2, the masking proven by token-level decode. The work order pre-registered +3pp as the threshold for "the recipe was the bottleneck." v2 delivered +0.58pp, and paired against v1 at matched seeds it came in at -0.09pp excluding one outlier seed. The defects were real and fixing them changed nothing measurable.

**A discarded record leaves no artifact to inspect.** Every quality effort in this project was initially pointed at records the pipeline kept. I commissioned a random-sample census of the 886 records stage 1 destroyed instead, and 25.4% of them were false kills, 95% CI [18.4, 33.5] exact binomial at n=138, which is ~225 [162, 296] paid-for records wrongly thrown away. Nothing in the system reported the loss because there was nothing left to report on.

**Adverse selection inflates any rate conditioned on a second opinion.** Two earlier proxies read 82.5% and 58%, both measured on kills where another model had passed the record. The random-sample census read 25.4%. The gradient 82.5% → 58% → 25.4% is exactly what adverse selection predicts, and neither prior should be quoted for population-level claims again. I had also predicted error-mode kills would be near 100% false. Measured at 30%. My own table says REFUTED.

**Measure both error directions or you have measured neither.** The census covers what the filter destroyed. A separate audit of 309 accepted records found the opposite error, 43 miss candidates at 13.9%, using 16 independent first-pass reviewers with blind second review of every flag. Single-gate lanes ran 17.0% contamination against 11.3% for two-gate, which is the empirical case for the second independent judge and is worth more than any argument about which model is stricter.

**Split votes predict false kills.** 2-of-3 split kills were 50% false against 21.7% for unanimous 3-of-3, confirmed at 47.6% and 28.6% on a follow-on sweep. False kills were 83% standard-terminology pedantry, the judge killing what it failed to recognize as standard. Genuine misses were 90% underspecified-external, so the kills that were right were right for the right reason.

**Interim reads in this campaign have been wrong four times.** Run 1 seed 1 came in at +11. The n=8 sign test came in at p = .016. v2 read +2.80pp at five seeds and +2.29pp at seven before landing at +0.58pp at twelve, and the five seeds that finished last were the same block that sank v1. v3b's partial read misled as well. The pre-registered stopping point is the only number worth quoting, and this is the same finding arriving four separate times rather than a lesson I drew once.

**The evidence now points at target construction, not the recipe.** Three arms after v1 have returned null. The training targets are the base model's own verified-correct rollouts stripped to bare answers, a median of 24 to 55 characters against the base model's 2,034-character derivations, and ~90% of tuned outputs come back under 100 characters as a lone boxed answer. Every arm inherits that. Regenerating with derivations preserved is the first hypothesis not yet tested, and the length table is direct evidence for it.

**Fixing the recipe removed the argument for scaling.** Both structural defects scaled with the training set rather than diluting in it, which was the whole case for fixing them before spending on more data. With them fixed and the effect still absent, more data buys more of the same saturated signal. N≈1000 needs a different rationale now.

---

## The Split Rebuild

On 8/1/26 I rebuilt the holdout on a measured partition rule instead of the arbitrary one it started with. The old 200-train/100-holdout split is void and its records were reallocated, but held-out evaluation is unchanged as a practice and the leakage guard came out stronger than it went in.

The new rule is proof-bearing → train, proofless → eval, over the full 921-record three-tier universe of band, collapse, and misdirection. Solved records are excluded as useless and dropped records as having failed posedness testing.

| | band | collapse | misdirection | total |
|---|---|---|---|---|
| universe | 317 | 405 | 199 | **921** |
| train allocated | 187 | 194 | 87 | **468** |
| eval achieved | 104 | 97 | 85 | **286** |

Two things make the rule defensible. Proof availability was measured against difficulty rather than assumed, mean n_correct 3.19 proof-bearing against 3.23 proofless, Mann-Whitney p = 0.918, so the rule introduces no difficulty confound. And paper-level disjointness is the load-bearing leakage guard, 386 train-side papers against 238 eval papers with intersection zero, independently re-verified at freeze.

That guard is why the eval target of 322 came in at 286. The entire shortfall is paper conflict rather than defect, and I accepted the smaller eval instead of weakening the guard.

The rebuild is enforced in code rather than convention, and the enforcement got stricter. The old two-set contract carried a holdout-specific leakage check. It now carries one hard refusal that fires on any uid not on the train side, whether that is a former holdout record, an eval-side record, or a plain typo, so there are fewer ways for a record to reach training unnoticed than there were before.

I also renamed the held-out side from holdout to eval deliberately. Every record in the old holdout was reallocated, so nothing measured on the new split is comparable to the old 43/100 baseline, and a shared name would have invited exactly that comparison. Once a former holdout record is trained on it can never serve as eval again for any model trained on it, which is the reason the old artifact had to be voided rather than amended.

---

## Methodology

Each arm follows a consistent structure:

1. Identify what the previous arm's result rules out.
2. Form a falsifiable hypothesis about what change will move the holdout.
3. Change one variable at a time, holding corpus, split, seeds, baseline, and engine frozen.
4. Pre-register the decision rule and the primary statistic before any read.
5. Run to the registered stopping point, then read once.
6. Document the result, what it rules out, and what changes for the next arm.

Audit is structured to be independent of the work it reviews. Review windows get no project context and work directly from the code and data rather than from any summary of it, because briefing a reviewer transmits the briefer's assumptions. Running them across two model families addresses the same problem one level up, since two fresh windows of the same model can share a failure of imagination. The unbriefed reviewers found the largest defects in this project.

Spend is gated. No scrape, funnel, batch, or paid run starts without my release in the current session even when an approved manifest exists, and unapproved spend above $5 requires sign-off. The v1 campaign ran on ~$6 of GPU time and v3b came in at $5.94 against $8 approved.

---

## What Required Outside Audit

These are the defects my own process did not surface, and they define the limit I plan around.

**A wrong defect diagnosis, held for weeks.** I assumed dropped hypotheses dominated extraction defects at ~45%. Measured, it was 4%, and reference holes dominated at 10%. Root cause was a TeX cleaner stripping `\ref` destructively instead of resolving it, plus an extractor discarding a `has_external_refs` flag that had been set since the initial commit and consumed by nothing. All 320 audited rows carried flag=False. I got there by reasoning instead of measuring, and no absent tool caused that.

**A defect class no judge could catch.** The extractor turned a proved bound into a question asking for the sharp constant, with an answer the source never claimed. Detecting it requires the source document, which no downstream stage had.

**The dirtier slice was the one every number came from.** A companion audit found the holdout slice 20.8% defective against 3.8% for training.

**Two silent defects in how the data taught the model.** No loss masking and an inverted weighting scheme, both shipped in the first dataset build and both surviving a complete campaign. The specification pinned the input format precisely and never said how the loss should be computed.

---

## Technical Stack

- **Subject model:** Qwen3-8B, evaluated at Q4_K_M GGUF, k=8 rollouts, temp 0.7, thinking disabled
- **Eval engine:** llama.cpp b10107 (`c0bc859`), response fingerprint verified on every call, identical serve flags both arms
- **Training:** RunPod A40 48GB, transformers 5.14.1 / peft 0.19.1 / trl 0.29.1, bf16 LoRA r16 α32 lr1e-4 3ep
- **Judges:** Claude and Codex builds across Anthropic and OpenAI backends, four legal combinations, policies intersect / union / majority / prefer
- **Ground truth:** symbolic verification, LaTeX answers parsed through antlr
- **Acquisition:** in-house arXiv scraper, `plan → approve → run`, manifest-approved and budget-gated
- **Isolation:** through v1 only the training set was uploaded to the rented box, enforced by a guard refusing any other payload, with evaluation run locally on Metal. From v3 evaluation moved onto rented pods with grader parity verified zero-diff across both, record-to-pod binding, and identity checks at serve time.

---

## Where This Leaves It

Four arms, one supported direction with no magnitude, and three clean nulls. Each hypothesis was reasonable and each was tested cleanly, which makes this an honest negative result rather than a failed project.

I am not funding a fourth arm on the current target construction. The open fork is whether to regenerate the training targets with derivations preserved, which is the one hypothesis the evidence actually points at and the one thing no arm has tested. Stopping here and publishing three nulls is the other defensible option.

What I would not do is scale the data. That argument died with v2.

---

## Repository

The operating documentation this README replaced is at [`docs/README_legacy.md`](docs/README_legacy.md), covering pipeline stages, subsystems, processor modes, CLI, secrets convention, and non-goals. It is current, and it was moved because the front page of this repo is now the project's account of itself rather than its runbook.

> **Agents** (Claude Code, Codex, or any other harness): read [`AGENTS.md`](AGENTS.md) before touching anything. It is the canonical operating brief covering safety invariants, budget gates, holds, and test baselines. Nothing in this README supersedes it.

Where the numbers come from:

- [`docs/README_legacy.md`](docs/README_legacy.md): pipeline, CLI, and runbook
- [`docs/operator.md`](docs/operator.md): stage-by-stage runbook and cost estimation
- [`docs/scraper_runbook.md`](docs/scraper_runbook.md): in-house arXiv acquisition
- `out/corpus_pde625/corpus_manifest.json`: corpus provenance, fold history, k12 rechecks, deliberate exclusions
- `out/stage1_kill_census/stage1_kill_census.md`: the 886-kill false-kill census
- [`docs/wellposed_miss_audit_summary_20260710.md`](docs/wellposed_miss_audit_summary_20260710.md): the accepted-corpus contamination audit
- [`docs/lora_consistency_verdict.md`](docs/lora_consistency_verdict.md): v1, carrying its own SUPERSEDED banner
- [`docs/lora_v2_verdict.md`](docs/lora_v2_verdict.md) and [`docs/lora_v3b_stage2_verdict.md`](docs/lora_v3b_stage2_verdict.md): the follow-on arms
- [`docs/split-rebuild-2026-08-01.md`](docs/split-rebuild-2026-08-01.md): the split ruling

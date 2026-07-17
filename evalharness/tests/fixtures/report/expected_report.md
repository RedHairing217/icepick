# LoRA Eval Harness Report

Generated: 2099-01-01T00:00:00Z

- eval set: `eval_set.jsonl`
- baseline (base, greedy): `baseline_greedy.jsonl`
- post (tuned, greedy): `post_greedy.jsonl`

## Headline -- eval-band greedy pass@1

> **UNDERPOWERED**: eval-band has 6 scored record(s), below the design doc's 25-record power floor. Treat any delta below as a signal to investigate, not a claim.

- n (eval-band, paired) = 6
- base solved: 2 / 6 (33.3%)
- tuned solved: 4 / 6 (66.7%)
- delta solved: +2 (+33.3pp)
- discordant pairs: b (base-only correct) = 1, c (tuned-only correct) = 3
- exact McNemar p = 0.625
- 95% CI (normal approximation, paired) on delta: [-0.263, +0.930]

## Anchor drift

Anchors are eval-paper records at the extremes (8/8 solved, 0/8 failed) in the original remote rescore. They are sanity checks, not the headline: anchor-solved must STAY solved (else catastrophic forgetting); anchor-fail must STAY failed (else memorization/contamination).

### anchor-solved (must stay solved)

- n = 2
- base solved: 2 / 2 (100.0%)
- tuned solved: 1 / 2 (50.0%)
- stayed solved: 1 / 2
- regressed (base solved, tuned did not): 1 **(RED FLAG)**

### anchor-fail (must stay failed)

- n = 2
- base solved: 0 / 2 (0.0%)
- tuned solved: 1 / 2 (50.0%)
- stayed failed: 1 / 2
- contaminated (base failed, tuned solved): 1 **(RED FLAG)**

## Secondary -- k=8, temperature=0.7, x3 (distributional, informational only)

Never blended into the headline above -- catches probability-mass shifts greedy decoding misses.

| repeat | base mean n_correct | tuned mean n_correct |
|---|---|---|
| 0 | 3.5 | 5.5 |
| 1 | 3.5 | 5.5 |
| 2 | 3.5 | 5.5 |

- overall mean n_correct: base=3.5, tuned=5.5

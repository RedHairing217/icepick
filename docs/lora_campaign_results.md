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
| 20260724 | *(eval in flight at time of writing — appended on completion)* | | | | |

Training quality was near-identical across seeds (loss 0.431 vs 0.433, token accuracy
87.9% vs 86.9%, same token count) — the outcome spread is **generalization variance
between seeds**, not a training failure.

## Honest read (pending seed 3)

- Seed 1 alone (+11pp, p=0.061) would have been an overclaim; the multi-seed design
  caught it. Seed-to-seed spread (+11 vs +1) is as large as the candidate effect.
- Anchor flags are 1–2/10 per arm — within noise at n=10, watched across seeds.
  Paper-level train/eval disjointness makes memorization of eval items impossible by
  construction, so anchor-fail flips are generalization-or-noise, not leakage.
- Seed 3 arbitrates: consistent +8–14pp across seeds would be strong evidence; a
  +1-like repeat means the true mean effect is small and run-dependent at 200 examples.

## Reproduction

Adapters (PEFT + GGUF-converted, ~175 MB each): seeds `20260722/3/4`, shas
`bcebb86a… / 300dd8b6… / b8a7525d…` (local `/tmp`, box `/workspace/run/out/`; not
committed). Eval outputs + per-seed reports: `out/evalharness/run1*` (gitignored, local).
Rebuild the eval set from the committed split via `evalharness-build-set`; the corpus
itself never leaves the machine.

Box cost, entire campaign (setup, smoke, 3 seeds): ≈ **$3**.

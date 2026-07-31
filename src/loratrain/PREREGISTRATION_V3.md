# PREREGISTRATION — v3 proof-as-hint arm (`lora-v3-proofhint`)

Written 2026-07-31 ~19:2xZ by the v3 construction window, per
`docs/lora_v3_steering_prompt.md` §PRE-REGISTRATION. Frozen **before**: any v3
training, any v3 holdout read, availability of the dq-vs-v2 k=8 verdict, and
before `solutions_v3.jsonl` exists (the proof-import lane is mid-flight).
This file is append-only: changes land as dated amendments (the dq arm's
`PREREGISTRATION.md` Amendment-1 discipline), and only before the read they
affect.

## 1. Instrument (pinned)

- k=8 protocol, campaign standard: box generation (RunPod A40, llama.cpp CUDA,
  engine b10107 `c0bc859`, `-fa off` explicit, temp 0.7, max 2048, `/no_think`
  wire idiom, 8 rollouts/record), **grading local on the M4** via
  `out/passk8_sweep/grade.py`'s chain (`scoring.extract_candidate` →
  verifier → `tally_rollouts` → `derive_label`). Grading never runs where
  `antlr4-python3-runtime` is unverified (measured 2026-07-31: its absence
  silently corrupts ~70/120 records per config).
- Eval set: the pinned 120-record holdout eval set (split sha[:16]
  `768436f4`); corpus pin `e0975e11`/293; base GGUF pin `a7676d25…`.
- **Base ruler**: the 2026-07-31 sweep's base k=8 run, reused iff the serving
  engine build and GGUF are identical. If v3's eval pod cannot reproduce that
  engine identically, the base ruler is re-run on the v3 pod FIRST and **v2's
  transition counts are recomputed against that same new ruler** — the primary
  comparison is only ever made on one common ruler. No cross-backend mixing
  (INSTRUMENT_BACKEND_FINDING: greedy/rollout text is a property of
  weights × backend).
- Identity guard (`/v1/models` alias == intended config) before every config's
  generation; slot guard for any local serving.

## 2. Series

- **v3**: 12 adapters, 12 fresh seeds (R2 default), seeds pinned in
  `run_config` before training starts.
- **v2 comparator**: the existing 12 v2-cap1 adapters (seeds 20260722–0803
  cohort) already generated/graded by the 07-31 sweep.
- v3 seeds are fresh ⇒ the v2 comparison is **two-sample, unpaired** (n=12 vs
  n=12). Basis: F4 cohort-independence evidence (ratio ≈0.9–0.95, no
  eval-block artifact signature); declared here, not decided post hoc.

## 3. Primary endpoint

For seed s, let `T(s) = #{holdout records r : base-ruler label(r) = band AND
label under s = solved}` (band→solved transitions), computed at k=8 on the
common ruler. Primary statistic: `Δ = mean_s T_v3(s) − mean_s T_v2(s)`.

- Test: two-sided Welch's t across seeds (12 vs 12), α = 0.05; two-sided
  Mann–Whitney U reported as robustness check. No result is computed until
  **both series are 12/12 × 120 records** (the analysis script must refuse
  earlier — same contract as `analyze_dequant_arm.py`).
- The transition denominator is the base-ruler band set as measured (known
  ~70/100 under k=8, i.e. 30% label drift from the frozen "100 band"
  designation) — the denominator is whatever the common ruler says, recorded
  in the analysis output.

## 4. p-convention (binding, resolves the recorded inconsistency)

**All inferential tests in this arm are two-sided at α = 0.05.** (The campaign
record mixed conventions: p=.344 two-sided vs p=.17 one-sided on the same v1
data.) Historical one-sided numbers may be quoted only as clearly-labeled
parentheticals. No multiple-comparison correction on the single primary; if
any additional confirmatory comparison is added by amendment, Bonferroni over
the confirmatory set, stated in the amendment.

## 5. Secondary endpoints (reported, never gating)

1. Mean Δn_correct per seed vs the common base ruler (all 120 records).
2. band→collapse transitions per seed (the v1 k=8 regression signature).
3. Anchor instrument gates — **hard validity gates, not endpoints**: per
   config, fail-anchors 0/10 solved; solved-anchors ≥9/10 solved (07-31 base
   ruler reference: 10/10 at 7.80/8, fail 0.10/8, boundary anchor 4ea3bb00
   noted). A config failing anchor gates is an instrument fault → stop and
   surface, don't pool.
4. **Loss-floor covariate** (premise tripwire): per-seed final-epoch train
   loss under completion-only masking. Reference points, stated with their
   denominators because the skeleton's "≤0.45" is denominator-ambiguous:
   v1 floored at 0.431–0.433 (prompt+completion denominator); v2 floored at
   0.3209–0.3243 (completion-only). v3 trains completion-only, so the
   comparable reference is v2's band. Pre-registered flag: if v3's per-seed
   final losses land at-or-below the v2 floor band, report **"premise
   failed — hints added no fitting difficulty"** prominently in the verdict
   regardless of eval outcome. The exact numeric threshold, if Nicky wants
   one, is Nicky's ruling — both candidate readings surfaced here in advance.
5. Hint-copy census: n-gram overlap distribution between each record's hint
   (`solution_text`) and the kept trace (dataset-build fact, computed
   pre-eval). Reported; threshold is Nicky's.
6. Dataset-build censuses (pre-eval facts, not holdout reads):
   verified-on-try-n histogram, hint-insufficient rate (R5 drop census),
   blend composition vs the R3 60/40+25 targets.

## 6. Discipline

- **No interim reads.** No per-seed eval number is reported, quoted, or
  reasoned from until both series are complete (campaign fact: interim reads
  were wrong all three times — run-1 seed-1, n=8 verdict, v2-at-5-seeds).
  Training losses and build censuses are exempt (not holdout reads).
- Holdout answer key never leaves the M4; regen bundles carry train rows only.
- BASE_SCHEME per R4 (default `fp16_hf_revision`). **The dq-vs-v2 k=8 verdict
  is unread at freeze time.** If it triggers an R4 re-rule to `dequant_q4km`,
  a dated amendment must land here BEFORE v3 training, and the verdict must
  note that scheme-vs-recipe is then partially confounded against the fp16 v2
  comparator (the dq arm's own paired result is the scheme measurement).

## 7. Declared limitations

- Single shared base-ruler draw: ruler noise is common mode; it shifts
  absolute T(s) for both arms but cancels in the v3−v2 contrast (the 07-31
  sweep's ~1.3σ caveat).
- Unpaired seeds (v3 fresh vs v2 cohort) costs power vs the v1↔v2 paired
  design; accepted per R2 default.
- Treatment bundles two things by design: imported proof information AND the
  R3 curriculum shift (collapse-heavy mix). A positive result does not
  attribute between them; attribution would need a v3b ablation (out of
  scope, surface-only).

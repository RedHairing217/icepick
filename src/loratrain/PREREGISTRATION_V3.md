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

---

# AMENDMENT 1 — v3-full-run instrument (2026-08-01 ~10:1xZ)

Written BEFORE: any base-ruler generation, any arm generation, any read of any
eval-set measurement. Supersedes the 2026-07-31 sections above wherever they
conflict — the instrument they pinned (200/100 split sha `768436f4`, 120-record
holdout, local Metal grading, greedy-era comparators) was voided by Nicky's
2026-08-01 split-rebuild ruling. The sections above remain as the historical
record of the pre-rebuild design.

## 1. Instrument

- **Scoring authority:** `docs/gate_crossing_scoring_spec.md` (revised 2026-08-01)
  in full — k=16 code-gate labels (fail 0–1 / band 2–12 / solved 13–16), ±1 per
  problem, |Δ|≥4/16 magnitude criterion, solved-regression −1, missing data
  excluded pairwise and counted.
- **Eval set:** `evalharness/data/corpus_split_v3_proofsplit_20260801.json`
  (sha16 `69735899efe9270e`), `eval_set_uids` = **286 records (104 band / 97
  collapse / 85 misdirection)**. Nicky (2026-08-01) accepted 286 over the
  originally-ruled 322: the paper-disjointness guard (train ∩ eval papers = ∅,
  the load-bearing guard) excludes 91 candidates and is kept intact.
- **Base ruler:** k=16 as two independent k=8 passes (different explicit seeds,
  recorded; same serving config; both passes of a record on the SAME pod),
  engine b10107 `c0bc859`, `-fa off` explicit, temp 0.7, max_tokens 2048,
  `--parallel 8` pinned. Fresh ruler labels only — corpus tiers are never
  "before" labels.
- **Grading:** pod-side, R4-fixed verifier (commit `0aae56e`, landed pre-ruler;
  ungradeable-by-name class EMPTY — pre-fix 21-list preserved in provenance),
  zero-diff parity gate per pod against a known-good fixture config, plus the
  temp-0 cross-pod probe before pooling any sharded measurement.

## 2. Treatment

12 fresh seeds (R5), fp16 base (dq verdict: dq ≈ v1), frozen control
hyperparameters, trained on the **468-allocation hinted-only dataset** (no
anchor — Nicky 2026-08-01; `V3_ANCHOR_FRACTION = 0.0`), cap1, hint-insufficient
→ drop + census. The hint never appears in a training prompt (guard-enforced).

## 3. Primary endpoint — DECLARED

Per arm (seed) s: `net(s)` = Σ over eval records of the spec's ±1 gate-crossing
score, arm-vs-base on the common ruler, pooled-16 where rerun rules fire,
first-8-vs-first-8 elsewhere.

**Primary test:** two-sided Wilcoxon signed-rank of the 12 per-seed `net(s)`
values against the **A/A empirical null net** (the base ruler's two k=8 halves
scored against each other under the exact spec rules), α = 0.05. Paired-t
reported as sensitivity. The 12 seeds are replications of ONE treatment — one
primary, no multiplicity correction. Effect size = median per-seed net and the
promotion/demotion decomposition.

## 4. Secondaries (reported, never gating)

Promotions and demotions separately; gate-crossings vs |Δ|≥4/16 magnitude
moves as SEPARATE lines (a verdict resting mainly on intra-band fluctuation is
distrusted per spec); per-tier transition tables; per-seed training-loss floor
(v2 completion-only reference band 0.3209–0.3243; at-or-below ⇒ "premise
failed — hints added no fitting difficulty" reported prominently regardless of
eval outcome); hint-insufficiency rate and verified-on-try-n histogram; hint
n-gram copy census (threshold Nicky's); solved-guard: >20% of base-solved
regressing in any arm ⇒ run flagged instrument-suspect, investigated before
any reporting.

## 5. Explicitly out of scope

v3-vs-v2 head-to-head: NOT computable — v2 has no k-sampled measurements on
any current instrument (0/12 in the halted sweep); the ~$32 option to measure
the 12 v2 adapters on this ruler remains open and unexercised. Any such
comparison, if later funded, gets its own dated amendment BEFORE its first read.

## 6. Discipline

No interim reads: no per-seed eval number is reported, quoted, or reasoned
from until all 12 arms are complete AND the A/A null is computed (campaign
scar: interim reads were wrong all three times they were taken). Both passes
differ only in sampling seed. Pod identity recorded per config; anomalous arm
⇒ pod identity is the first suspect. Deviations land as dated amendments
BEFORE the affected read.

---

# AMENDMENT 2 — pod-effect cancellation replaces cross-pod byte-parity (2026-08-01 ~16:5xZ)

Written BEFORE any base-ruler generation completed and BEFORE any arm
generation. Trigger: the temp-0 cross-pod probe FAILED as a byte-equality
gate — 2/5 records diverged across cold pods and every pod exhibited a
warm-server call-order drift (full diagnostics in
`out/v3_full_run_20260801/opslog_phaseA.md`). The skeleton's premise
("identical A40s with identical builds must agree exactly") is empirically
false for greedy near-tie tokens under per-host float noise — consistent with
the campaign's 2026-07-30 finding that byte-exact greedy equality across
compute paths is structurally unachievable.

**Change:** the validity mechanism for pooling sharded measurements is now
**record-to-pod binding**, not cross-pod byte-parity:

1. Every eval record is bound to exactly one pod
   (`out/v3_full_run_20260801/baseline/record_pod_map.json`). Its base-ruler
   passes AND all 12 arm measurements run on that pod. Per-pod instrument
   deltas therefore cancel inside every record-level transition; pooling only
   aggregates per-record outcomes, never mixes cross-pod numerics within a
   record.
2. Phase C is re-sharded from config-per-pod to record-shard-per-pod (each
   pod serves all 12 adapters over its own shard). Generation count
   unchanged; ~36 extra server restarts ≈ +$1–2.
3. The temp-0 probe is retained as an advisory diagnostic (recorded, not
   gating). The A/A calibration is unchanged and remains same-pod by
   construction.
4. Warm-vs-cold server state between a record's two passes is
   sampling-noise-equivalent; it is part of what the A/A null measures.

No endpoint, test, α, or exclusion rule changes. This amendment is
variance-reducing by construction (it removes a cross-pod noise term from
every arm-vs-base comparison that the A/A null could not have modeled).

---

# AMENDMENT 3 — staged execution: one seed at a time (Nicky, 2026-08-01 ~18:2xZ)

Written before any arm generation or arm read. Nicky's instruction ("Let's
focus on one seed for now / One at a time") stages EXECUTION; this amendment
records what that does and does not change:

1. Stage 1 = seed 20260901 only: trained on the unchanged 396-row hinted
   dataset, evaluated on the unchanged 286-record eval set at k=8 (rerun-to-16
   per the spec's boundary rules), records pod-bound per Amendment 2.
2. The registered 12-seed primary (Amendment 1 §3) remains THE confirmatory
   analysis and is computable only if/when all 12 seeds complete. Nothing
   about the endpoint, test, α, or exclusion rules changes.
3. Any number read from stage 1 before 12/12 is an INTERIM/PILOT read, labeled
   as such wherever quoted. A single seed carries no significance machinery
   (historical per-seed sd ≈ 3.45pp on the old instrument; the A/A null is a
   distribution, not a threshold, at n=1). The campaign's record — interim
   reads wrong all three times they were taken — stands as the warning.
4. If the campaign extends to further seeds after any stage-1 read, the
   continuation is DATA-DEPENDENT: the final analysis must disclose it, and
   the clean options at that point are (a) exclude seed 20260901 from the
   confirmatory set as the pilot, or (b) include it with the optional-stopping
   caveat stated in the verdict. That choice is Nicky's, made then, recorded
   as its own amendment.

---

# AMENDMENT 4 — seedless arm sampling + stage-1 single-adapter analysis (2026-08-01 ~18:4xZ)

Written before any arm generation. Two changes, both Nicky-directed today:

1. **Seedless pass@k for arm evals.** Arm generation requests omit the seed
   field (server default RNG; rows record seed=null; `box_generate.py
   --unseeded`). The base ruler, already generated, used explicit seeds —
   both regimes draw from the IDENTICAL pinned serving distribution (engine
   `c0bc859`, `-fa off`, temp 0.7, max 2048, parallel 8), so estimates are
   unbiased and mixing is valid; sample independence is unaffected;
   per-rollout byte-reproducibility is explicitly waived (already unreliable
   under warm-server call-order drift, opslog_phaseA probe finding).
   Regeneration (training data) is untouched — it was already running seeded
   and is data generation, not measurement.

2. **Stage-1 primary read (n = 1 adapter, seed 20260901), declared before its
   eval:** observed statistic = the adapter's net gate-crossing score vs the
   base ruler over the 286 records (spec table, pooled-16 where rerun rules
   fire). Null = **bootstrap A/A distribution**: per record, resample which 8
   of the base's 16 same-pod samples form each half (B = 10,000), score
   half-vs-half under the exact spec rules, collect the net each time.
   Two-sided p = fraction of bootstrap |net| ≥ observed |net|. Report
   promotions and demotions separately, and gate-crossings vs |Δ|≥4/16
   magnitude moves as separate lines. This tests "this adapter differs from
   instrument noise" — it does NOT test recipe replication (Amendment 3 §3
   stands; per-seed draw-luck is invisible at n=1). If the campaign extends,
   Amendment 1's 12-seed primary resumes per Amendment 3 §4.

---

# AMENDMENT 5 — solved-demotion magnitude parameter (Nicky, 2026-08-02 ~00:3xZ)

Nicky's parameter adjustment, verbatim intent: "Any solved record that degrades
by less than 4/16 move to null." Operative scoring change: a base-`solved`
record that crosses down (to band/fail) scores **−1 only when its degradation
Δ ≥ 4/16** (k=8-comparison equivalent: ≥ 2/8); smaller label-slips score 0.
This applies the spec's existing magnitude criterion symmetrically to the
solved boundary, removing the boundary-slip noise class (e.g. 13/16 → 6/8).

TIMING DISCLOSURE (binding on the verdict): this amendment arrived AFTER
interim reads of the stage-1 first-pass data (provisional tables posted to
Nicky at ~00:2x–00:3xZ). It is therefore a post-hoc parameter relative to
stage 1. The stage-1 verdict MUST report BOTH scorings side by side —
registered-original (Amendment 1/spec) and adjusted (this amendment) — with
the adjusted as Nicky's operative headline and the original as the
pre-registered reference. Any future seeds inherit the adjusted rule
prospectively (for them it is pre-read).

---

# AMENDMENT 6 — stage-2 (v3b) protocol (2026-08-02 ~01:3xZ)

Written BEFORE any stage-2 generation or read. Nicky's release: proof-injection
anchors ("Proof bearing records would be injected into training data to anchor
model understanding"), ≈$8 approved.

1. **Treatment (one variable):** the 390 stage-1 hinted rows BYTE-UNCHANGED +
   `anchor_solved` rows — proof-bearing, cached-tex, eval-excluded solved-tier
   records (95 candidates; post-reformulation/verification attrition censused),
   hint-regenerated by the same mechanism, cap1, ~15–20% of the final dataset.
   Seed **20260902** (next in the pinned fresh cohort), frozen control HP,
   fp16 base.
2. **Instrument:** same 286-record eval set and scoring spec. Fresh pods ⇒
   **new base ruler** (286×16 as two independent k=8 passes, SEEDLESS both —
   extending Amendment 4's regime uniformly), new record-to-pod binding map
   persisted, per-pod zero-diff grader parity + identity guards as stage 1,
   30s verify timeout.
3. **Primary (pre-read):** net gate-crossing score under the
   **Amendment-5-adjusted rule** (now fully prospective) vs the NEW ruler,
   two-sided p from a B=10,000 bootstrap A/A null computed from the new
   ruler's halves under the same adjusted rule. The registered-original rule
   is reported alongside for continuity, no longer headline.
4. **Pre-declared targeted secondary:** the 14 stage-1-degraded solved uids
   (score_FINAL_registered.json ∩ adjusted −1 set, list reproducible from
   committed artifacts): count degraded under v3b's own ruler/arm comparison,
   reported as 14→N. Cross-stage, different ruler draws — directional
   evidence, not a pooled statistic.
5. Standard secondaries as stage 1 (promotions/demotions, gate-vs-magnitude
   lines, per-tier tables, solved-guard with the adjusted-rule denominator,
   loss covariate vs the v2 floor band and vs stage-1's 0.4317, anchor_solved
   build censuses incl. try-histogram and copy census). No interim reads
   before the full stage-2 read; single-seed interim/pilot labeling per
   Amendment 3 continues.

# v3 full run — execution skeleton (split rebuild → train → score)

**Paste into a fresh window.** Mission slug: **v3-full-run**. Status at write time
(2026-08-01): **NOT RELEASED** — Nicky arms §1. Repo
`/Users/redhairing/Desktop/helloworld/icepick` (`cd` in every Bash call). Read
`AGENTS.md` → `docs/SESSION_HANDOFF.md` → **`docs/gate_crossing_scoring_spec.md`
(authoritative for all scoring)** → this file.

This connects three finished pieces of work into one run: the corpus census (complete),
the proof-import lanes (139 rows published, machinery reusable), and the gate-crossing
scoring spec (revised and calibrated). Everything below is arithmetic on measured
numbers, not estimates.

---

## 0. VERIFY FIRST — this environment has delivered fabricated completion events

| pin | expected |
|---|---|
| corpus | `out/corpus_pde625/band_corpus.jsonl` 293 rows, sha[:16] `e0975e11` |
| wellposed pool | `out/corpus_pde625/wellposed_all_with_passk.json` 2021 records |
| base GGUF | `~/.lmstudio/…/Qwen3-8B-Q4_K_M.gguf` sha256 `a7676d25…8f35f` |
| engine | llama.cpp **b10107** = `c0bc859`, `-fa off` **explicit** |
| published proof rows | 139 across three lanes (87 band / 28 collapse / 12 misdirection / 12 since-relabelled-solved) |
| suites | three-suite **1167**, `src/loratrain` 623+2 skips |

**The old 200/100 split is VOID** (`split-rebuild-2026-08-01.md`). Holdout no longer
exists. Do not read `corpus_split_200_100.json` as authority for anything except
historical provenance.

## 1. RELEASE CHECKBOXES (Nicky)

- [ ] **R1 — build the split + training set** (§3). Sonnet ≈ **$9.85**, no GPU.
      OVER the $5 line — this is the overbuild ruling (all band + collapse backfill).
- [ ] **R2 — eval size.** `322` (all 129 proofless band, per ruling) · or `~200` to cut
      GPU cost ~40% · or `___`
- [ ] **R3 — GPU budget.** Base ruler + 12 arms at eval=322 ≈ **$32** (A40, ~73 pod-hours).
      Over the $5 line — needs explicit sign-off. **Up to 4 concurrent pods are approved
      (Nicky 2026-08-01)**, which cuts wall-clock to ~18 h at identical total cost.
- [ ] **R4 — verifier infinity fix** before the ruler is measured? (20–21 records whose
      `fail` labels are a `simplify(oo−oo)=nan` artifact.) Recommended yes: fixing after
      the anchor exists splits the instrument.
- [ ] **R5 — arm count.** 12 seeds (matches prior campaigns) · or `___`

## 2. COMPOSITION — 40/60 band:fail, OVERBUILT (all band, collapse backfill)

Split rule: **proof-bearing → training, proofless → eval.** Measured, not assumed:
proof availability is **independent of difficulty** (mean n_correct 3.19 vs 3.23,
Mann-Whitney p = 0.918) and roughly flat across tiers, so this introduces no
difficulty confound.

**Training — OVERBUILT (Nicky, 2026-08-01): take ALL band, backfill the failure side
with collapse.** Proof-bearing pool: band 187 / collapse 217 / misdirection 87.

| tier | allocated | available | share |
|---|---|---|---|
| band | **187 (all)** | 187 | 40.0% |
| collapse | 194 | 217 | 41.5% |
| misdirection | **87 (all)** | 87 | 18.6% |
| **total** | **468** | | fail side 281 = 60.0% |

Band anchors the size at 40%; misdirection is **exhausted at 87**, so its 53-row
shortfall against a strict 30% is **absorbed by collapse** (140 → 194). The 40/60
band:fail ratio is preserved exactly; the 30/30 split *within* the failure side is not,
because the corpus cannot supply it.

139 rows already published, so **341 net-new P4 calls ≈ $9.85** — over the $5 line, R1
covers it. (Strict 40/30/30 would be 290 rows at $4.71; overbuilding adds 177 rows for
$5.14.) Leftover unused: 23 collapse. Band and misdirection are fully consumed.

**Eval (proofless pool: band 129 / collapse 186 / misdirection 104)**

| tier | allocated | available |
|---|---|---|
| band | **129 (all)** | 129 |
| collapse | 97 | 186 |
| misdirection | 97 | 104 |
| **total** | **322** | |

Plus `solved` records as scored guards (see spec — they are scored −1 on regression, not
excluded). Screen the eval set for the 21 ungradeable records and exclude by name.

⚠ **Eval size is the dominant cost driver of the whole program.** 322 records × 16
samples = 5,152 generations per config (~4.6 h on an A40). Base + 12 arms ≈ **73 pod-hours
≈ $32**. Cutting eval to ~200 keeps the tier shape and drops it to ~$20. This is R2.
Wall-clock is decoupled from cost — see P5b, 4 concurrent pods approved (~18 h).

## 2b. EXECUTION SUBSTRATE — RunPod (Nicky, standing ruling)

**ALL eval runs on RunPod: generation AND grading. No local eval, ever.** The pod-side
grader is a verified recipe, not an improvisation: venv + `sympy==1.14.0` +
`antlr4-python3-runtime==4.11` + the **full** `src/icepick` tree (partial trees die on
`icepick.config`). **Parity-check it against a known-good config and require ZERO record
diffs before trusting any number** — without antlr4 the grader silently mis-scores ~70
of 120 records per config, and it fails closed rather than erroring.

Training, regeneration and the base ruler are likewise pod work.

Exactly two steps stay local, each for a hard reason, not preference:

1. **API-key custody (P1 Sonnet).** RunPod env vars are readable back through the
   account API, so an Anthropic key on a pod is exposed. Every pod to date has received
   only `uid` + `statement` — never keys, never answer keys beyond what grading needs.
2. **Data selection / split assembly.** Building record lists from census artifacts is
   data prep, not measurement; the inputs live on the M4.

## 3. PHASES

### P1 — finish the training set (Sonnet, ~$9.85, no GPU)
Run P4/P5 for the 341 net-new records using the existing lane tools
(`out/proof_import_20260731T185338Z/tools/`, CONTRACTS.md schemas, cache key
`(uid, sha256(proof_raw))`). Use `p5_verify_publish_corpuswide.py` — the other two
guards refuse this mixed set (77 are old-train members, 587 are not). Assert cross-lane
uid uniqueness against the 139 already published; the guard does **not** do this for you.

### P2 — build and freeze the split
Emit a NEW split file with its own sha, recording per record: tier, proof-bearing
status, train/eval side, and former-holdout provenance (sidecar
`former_holdout_map.json` already exists). **Paper-level disjointness between train and
eval is the load-bearing guard** — uid-level is not sufficient. Then:
`loratrain/v3.py:511 assert_train_split_only()` refuses everything against the OLD pin —
it needs the new split path, a new `EXPECTED_SPLIT_SHA256`, and its holdout branch
retired (the unknown-uid branch stays).

### P3 — regenerate (GPU pod, on-policy)
Per `docs/lora_v3_proofhint_execution_skeleton.md`: hint at generation time only, never
in the training prompt; model re-derives; endpoint-verify; keep the model's own trace.
cap1, one kept trace per record. Hint-insufficient records → drop + census (R5 default).

### P4 — train
Frozen control config, R5 seeds, `BASE_SCHEME` per the dq verdict (dq ≈ v1 at k=8, so
fp16 stands unless re-ruled). **Archive adapters into `src/loratrain/data/` immediately**
— `/tmp` is not retrieval; three v1 adapters died there.

### P5 — measure
1. **Base ruler at k=16**, as two independent k=8 passes with different seeds, configs
   recorded. Reused by every arm.
2. **A/A calibration**: score the base's two halves against each other under the exact
   scoring rules. This is the empirical null — free, and required before any verdict.
3. **Arms at k=8**, then rerun to 16 for all band→band records and all fail/band
   boundary crossings.
4. Score per `gate_crossing_scoring_spec.md`. Report gate crossings and magnitude moves
   as **separate lines**.

### P5b — pod fan-out (up to 4 concurrent, approved)

Eval is embarrassingly parallel across configs — each config is an independent
serve-then-generate cycle. **Cost is per pod-hour, so parallelism buys wall-clock at no
extra spend.**

| pods | wall-clock | GPU cost |
|---|---|---|
| 1 | ~73 h | $32 |
| 2 | ~37 h | $32 |
| **4 (approved)** | **~18 h** | **$32** |

At eval = 200 records (R2 alternative): ~11 h on 4 pods, **$20**.

**Sharding rule — the base ruler is NOT shardable across pods in the naive way.** It is
the common reference for every arm, so it must be one coherent measurement. Options, in
order of preference:

1. **Run the full base ruler on pod 1 first (~4.6 h), then fan the 12 arms across all 4.**
   Simplest, and the ruler is ready before any arm needs it.
2. If the ruler is sharded by *record* across pods, every shard must use an identical
   serving configuration and the shards must be disjoint and complete — verify by
   reassembling and asserting 322 records × 16 samples with no duplicates before use.
   Do NOT shard the two k=8 passes of one record across different pods; the A/A
   calibration depends on the two halves being independent *samples*, not independent
   *machines*.

**Per-pod requirements — every pod, no exceptions:**
- Identical engine build (b10107 / `c0bc859`), identical flags, `-fa off` explicit.
- Base GGUF sha-verified against `a7676d25…8f35f` on each pod independently.
- The grader parity check (§2b) run **per pod** — a pod that skipped `antlr4` produces
  silently wrong numbers that look plausible.
- Record which pod produced which config in the manifest. Cross-pod numeric differences
  are exactly the class of bug that has bitten this project three times; if any arm
  looks anomalous, the pod identity is the first thing to check.

**Do not mix pod-produced and locally-produced numbers** — all eval is pod-side (§2b),
so this should not arise, but the CUDA-vs-Metal finding (0/3 byte-match) is why it
matters.

### P6 — verdict
Pre-register before any read: primary comparison, two-sided α=0.05, multiplicity policy.
Report net against the **A/A null**, not against assumed zero. Terminate pods same
session; sha-verify artifacts down first.

## 4. SCARS — do not rediscover (each cost real money or a corrupted run)

1. **A single `ps` snapshot is NOT proof a run is dead.** One landed in a 3-second gap
   and corrupted another session's seed. Check `ps` AND sibling progress logs under
   `/private/tmp/claude-501/*/scratchpad/*.log`.
2. **Verify the server is yours** — `GET /v1/models`, require the alias to equal the
   config you intend. A server that fails to bind leaves you talking to someone else's.
3. **`-fa off` explicitly.** The build defaults to `-fa auto`; the whole reference set is
   auto-resolved-off, and explicit `off` was verified byte-identical (3/3).
4. **Grading needs `antlr4-python3-runtime==4.11`** plus `sympy==1.14.0` and the FULL
   `src/icepick` tree. Without antlr4 the grader silently mis-scores ~70/120 records per
   config. **Parity-check any re-homed grader against a known-good config, zero diffs
   required, before trusting a single number.**
5. **RunPod:** `PUBLIC_KEY` env at create or sshd never starts. `nvcc` is off PATH
   (`/usr/local/cuda/bin`, needs `-DCMAKE_CUDA_COMPILER` + `-DCMAKE_CUDA_ARCHITECTURES=86`).
   pip is PEP-668 locked. ssh exit-255 ≠ failed launch — verify on a fresh connection.
6. **`setsid` does not exist on macOS** — plain `nohup` locally, `setsid nohup … &` on box.
7. **Interim reads have been wrong every time** (run-1 seed 1 +11; n=8 sign test p=.016
   → reversed; v2 at 5 seeds +2.80 → 12 seeds +0.58). Do not quote a partial series.
8. **Pin serve parallelism from the start.** Changing it mid-series splits the instrument.

## 5. SURFACE, DO NOT DECIDE

Hint-insufficiency rate; n-gram overlap between hint and the model's trace (copying vs
deriving); collapse-tier wellposedness (unaudited — a 0/8 record may be broken, not
hard); the ~15-record spot audit of published solutions (P5 verifies endpoints, not
derivations); and anything resembling a fourth structural defect. Both known dataset
defects were found by unbriefed external review, not by the pipeline's guards — **assume
more exist and report rather than quietly fix.**

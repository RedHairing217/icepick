# LoRA v3 — proof-as-hint self-regeneration arm (execution skeleton)

**Paste this whole file into a fresh window.** Mission slug: **lora-v3-proofhint**.
Status at write time (2026-07-31): **NOT RELEASED** — Nicky arms §1. Repo
`/Users/redhairing/Desktop/helloworld/icepick`. Read `AGENTS.md` →
`docs/SESSION_HANDOFF.md` → this file. **Hard dependency:** a completed
`out/proof_import_<ts>/solutions_v3.jsonl` from
`docs/proof_import_execution_skeleton.md` — the builder refuses without it.

**Hypothesis under test.** v1/v2 saturated because own-rollout training carries no
information the model lacks (loss floors ~0.43 epoch-1; k=8 shows sharpening, not
capability). v3 imports the missing information — paper proofs — but trains on the
model's **own re-derivation** of each proof, not the proof verbatim: hint at
data-generation time, on-policy tokens at training time. Prediction: training loss
does NOT floor epoch-1, and band→solved transitions exceed v2's at k=8.

---

## 0. ISOLATION CONSTRAINT (Nicky, 2026-07-31 — binding, checked first)

**All v3 machinery lives in a separate importable module: `src/loratrain/src/loratrain/v3.py`**
(plus `tests/test_v3.py`). It imports FROM the existing package (config pins, guards,
wire-format constants) but **no existing module imports v3, and no file the v1/v2/dq
arms executed is edited** — `build_dataset.py`, `train_qwen3_lora.py`,
`upload_guard.py`, `config.py` operator block all stay byte-identical (additive
config constants allowed ONLY in a clearly-marked v3 section). Rationale: the k=8
sweep and the three prior arms must remain reproducible from the tree that produced
them; a live campaign is never patched underneath. **Acceptance for every phase
includes: `git diff` on pre-existing loratrain files is empty (or additive-config
only), and the three prior arms' suites still pass untouched.**

## 0b. VERIFY BEFORE TRUST

Pins: corpus `e0975e11`/293 · split `768436f4` (200/100) · base GGUF `a7676d25` ·
engine b10107 `c0bc859` · `-fa off` explicitly, always · solutions file's manifest
sha-chain intact and **0 holdout uids** (re-verify yourself against the split file).
Also read the **dq-arm k=8 verdict** before starting: if dequant-vs-v2 came back
positive (quant mismatch mattered), STOP — base-scheme choice (§4 R4) changes and
Nicky re-rules.

## 1. RELEASE CHECKBOXES (Nicky)

- [ ] **R1 — run the arm.** Est: regeneration on box ≈ $2–3 · training 12 seeds ≈ $1–2 ·
      k=8 eval sweep 12 configs ≈ $6 · **total ≈ $9–11** (staged spends; each pod
      itemized; >$5 increments asked per invariant 12).
- [ ] **R2 — seeds:** `12 fresh values` (default; F4 showed cohort-independence is free) · or `___`
- [ ] **R3 — curriculum mix:** `60/40 collapse/band hinted rows + 25% unhinted v2-cap1
      band rows as distribution anchor` (default from the 07-31 design discussion) · or `___`
- [ ] **R4 — base scheme:** `fp16_hf_revision` (default) · `dequant_q4km` (if the dq
      verdict said the mismatch matters) — set via the existing `BASE_SCHEME` knob,
      never a v3-local override
- [ ] **R5 — hint-insufficient fallback** (record verifies 0/k even WITH the proof in
      context): `drop + census` (default, keeps everything on-policy) · `include
      solution_text verbatim` (off-policy contamination, but keeps hard records)

## 1b. EXECUTION SUBSTRATE (Nicky, 2026-07-31: run on RunPod, not local)

Every compute phase runs on pods: P2 regeneration (GPU pod, generation-only), P3
training (GPU pod, training stack), P4 eval sweep (GPU pod, k=8 protocol). The M4
does orchestration plus exactly two things that never move: **holding keys** (API
key path-proxies don't ship to pods) and **grading** (the holdout answer key never
leaves this machine — box generates, local verifies; this is RUNBOOK non-negotiable
#2 and the reason the k=8 sweep's grading is local). P1's dataset assembly is
trivial-CPU and runs wherever the guards run — locally, since it touches the split
file and eval_set for leakage asserts.

## 2. OPERATIONAL RULES (scars, binding)

Slot guard + identity guard on every serving cycle (the 07-30 contamination incident
lives in `active-handoff`; reference driver `v2_faoff_batch.sh`). `-fa off` pinned.
One Qwen slot machine-wide. Adapters archive to `src/loratrain/data/` — **/tmp is not
retrieval** (three v1 adapters died there). Box daemons via `setsid nohup … &`; ssh
exit-255 ≠ failed launch — VERIFY on a fresh connection. Verify task notifications
against disk. Pre-register the analysis BEFORE the first holdout read; no interim
verdicts (three prior interim reads were all wrong).

## 3. PHASES

### P1 — `v3.py`: regeneration dataset builder
For each row of `solutions_v3.jsonl` (train-split only, re-asserted):
prompt = wire-format question + `\n\nReference solution (from the source paper):\n`
+ `solution_text`; sample the **base model** (Q4_K_M, llama-server, parity flags,
temp 0.7) up to `k_regen=6` tries; keep the FIRST candidate whose endpoint verifies
(audited chain); emit training row `{prompt: question-only, completion: own_trace}`.
**The hint appears only at generation time — never in the training prompt** (serve
time has no hint; train/serve prompt must match). cap1: one kept trace per record.
Provenance per row: proof sha, regen sample_idx, verify receipt. Hint-insufficient →
R5. Blend per R3. Manifest with full censuses.
**Accept:** builder is pure-v3-module; guards (train-only, statement-leakage vs
eval_set, masking columns) pass; loss-mass census shows prompt tokens carry ZERO loss;
isolation check (§0) green.

### P2 — regenerate (box, generation-only pod)
Reuse the k=8 sweep's pod pattern: eval-only-style pod, statements+hints for
train-split rows only (holdout never ships — and note hints END in train answers,
which is fine for train rows, forbidden for holdout). ~200–400 records × ≤6 tries.
**Accept:** per-record outcome census (verified-on-try-n / hint-insufficient), shas.

### P3 — train (R2 seeds, frozen control HP)
Existing remote recipe unchanged (`train_qwen3_lora.py` untouched — it already takes
`--dataset`; the v3 dataset is just a file). BASE_SCHEME per R4. **Mid-training
tripwire: if final loss floors at ≤0.45 again, the arm's premise failed — complete
the seeds (they're cheap) but flag prominently.** Archive adapters + manifests to
repo data dir immediately.
**Accept:** 12/12 manifests stamped with dataset sha + scheme; loss curves pulled.

### P4 — evaluate: k=8 box sweep, pre-registered
The k=8 protocol is now the campaign standard (`box_run_all.sh` + local
`out/passk8_sweep/grade.py`). Reuse the EXISTING base ruler from the 07-31 sweep if
the engine build is identical; otherwise re-run base first on the new pod.
**Pre-registration (write before any read):** primary = **v3 band→solved transitions
vs the k=8 base ruler, compared to v2's** (two-sided, matched instrument, all 12/12
before computing); secondary = mean Δn_correct, band→collapse losses, anchors,
loss-floor covariate. State the p-convention.
**Accept:** 12 × 960 rollouts graded locally; analysis script refuses until 12/12.

### P5 — verdict + docs + memory
`docs/lora_v3_verdict.md`, campaign-doc update, handoff ledger, durable memory.
Pod terminated same session (§9 discipline). Working-tree isolation check one final
time.

## 4. SURFACE, DO NOT DECIDE

Loss floors despite hints (premise failure → the exits list in the 07-31 session:
stronger teacher, RL/GRPO v3b — explicitly OUT of scope here); hint-insufficient
fraction large (curriculum too hard / proofs too thin); regeneration verifying
instantly on try-1 for band rows (hint redundant there — mix may need rebalancing);
any sign the model copies hint text verbatim into its trace (n-gram overlap census —
report, threshold is Nicky's).

## 5. WHAT DONE LOOKS LIKE

Twelve v3 adapters whose training loss did something other than floor; a k=8
band→solved table with v3 beside v2 and v1 on the same ruler; a pre-registered
verdict either way; zero edits to any file the prior arms executed.

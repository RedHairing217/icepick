# LoRA dequant-base arm — execution skeleton (train + eval, `-fa off`)

**Paste this whole file into a fresh window.** Mission slug: **lora-dequant-base**.
Status at write time (2026-07-30): **NOT RELEASED.** Nicky arms the checkboxes in §1;
nothing below runs until then. Repo `/Users/redhairing/Desktop/helloworld/icepick`
(`cd` in every Bash call — the shell cwd resets). Read `AGENTS.md` first, then
`docs/SESSION_HANDOFF.md`.

**What this arm tests.** Every adapter to date was trained on the **fp16 HF revision**
and served on the **Q4_K_M GGUF** — a train/serve weight mismatch that README D3 named
the CRUX and only partly closed. Session da0c5e6d built the machinery to remove it
(commit `f1e0556`: dequantize the deployment GGUF to fp32 HF and train on *that*), but
**it has never been trained with.** This arm executes it end-to-end and measures whether
closing the mismatch moves the holdout.

**What it produces.** N LoRA adapters trained on the dequantized Q4_K_M base, converted
to GGUF, evaluated locally at `-fa off` against the frozen 100-record holdout, and a
pre-registered paired comparison against the matching fp16-base seeds.

---

## 0. VERIFY BEFORE YOU TRUST ANY OF THIS

This environment has delivered fabricated and premature completion events, and a
sibling session's campaign was corrupted on 2026-07-30 by exactly one unverified
assumption. Re-verify every claim below against disk before acting.

| pin | expected value |
|---|---|
| corpus | `out/corpus_pde625/band_corpus.jsonl` — 293 rows, sha256[:16] `e0975e11` |
| split | `evalharness/data/corpus_split_200_100.json` — sha[:16] `768436f4`; 100 pure-band holdout |
| eval set | `evalharness/data/eval_set.jsonl` — 120 rows (100 band + 10+10 anchors), gitignored |
| baseline | `out/evalharness/run1/baseline_greedy.jsonl` — **eval_band 43/100**, anchors 9/10 solved, 0/10 fail |
| serving base | `~/.lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`, sha256 `a7676d25…8f35f` |
| eval engine | `~/src/llama.cpp/build/bin/llama-server` @ tag `b10107` = commit `c0bc859`, Metal; every response carries `system_fingerprint: b1-c0bc859` |
| git HEAD | `f1e0556` (dequant tooling T1–T5) — **local only, unpushed** |

**The baseline does not need re-capturing.** The *served* model is unchanged (same
Q4_K_M GGUF); only the *training* base changes. 43/100 and all prior seeds remain the
valid comparison set — provided the eval runs `-fa off` (§5).

---

## 1. RELEASE CHECKBOXES (Nicky)

- [ ] **R1 — Run the arm.** Provision a box, dequantize, gate, train, eval. Est. spend below.
- [ ] **R2 — Seed count.** `12` (default, matches v1/v2 and the power analysis) · or `___`
- [ ] **R3 — Dataset held fixed at** `v2/cap1` (default — isolates base-scheme as the ONLY
      variable vs the in-flight v2 arm) · or `run1_final` (isolates it vs v1) · or `___`
- [ ] **R4 — bf16-at-load ruling** (§3.4, blocking, no default — this is a real fork)
- [ ] **R5 — Push release** for `f1e0556` and anything this arm commits (separate decision)

Estimated spend: box ≈ **$3–4** (dequant ~30–45 min + N×~17 min training + retrieval,
A40 @ $0.44/hr). Local eval **$0**. Over the $5 line only if retries pile up — if the
estimate moves above $5, stop and ask (AGENTS.md invariant 12).

---

## 2. OPERATIONAL RULES — read before touching a port

These are not style notes. Each one is a scar.

1. **NEVER infer "no run is in flight" from a single `ps` snapshot.** On 2026-07-30 a
   `ps` check landed in a **3-second gap** between one seed finishing and the next
   starting; the conclusion "the campaign is dead" was wrong and corrupted a seed.
   Before taking port 8081 or the Qwen slot: check `ps` **and** read every sibling
   session's progress log under `/private/tmp/claude-501/*/scratchpad/*.log`, **and**
   re-check after a pause.
2. **Verify the server you are talking to is yours.** `GET /v1/models` and require the
   loaded alias to equal what you intend to evaluate. A `llama-server` that fails to
   bind leaves you silently talking to **someone else's** server on that port —
   health checks and generations both succeed, against the wrong weights.
3. **One Qwen slot machine-wide** (AGENTS.md invariant 9). One `llama-server`, one eval,
   at a time — parallel sessions included.
4. **`-fa off` explicitly, always.** This build defaults to `-fa auto`; the entire
   established reference set (baseline + all 12 v1 seeds) was measured on the
   auto-resolved-**off** path. Explicit `off` was verified byte-identical to it (3/3
   tripwire, 2026-07-30). `-fa on` produces different generations and is not comparable.
5. **`out/**` is append-only.** New files and dirs only. Quarantine (move) needs Nicky's
   ruling; deletion is not yours to make.
6. **Remote ops:** launch box daemons with `setsid nohup … & sleep 4` (link RSTs
   quick-exit ssh); `scp` is reliable; compound ssh scripts get killed mid-sequence —
   decompose. `setsid` does **not** exist on macOS — locally use plain `nohup`.
7. **Budget gate (standing, Nicky 2026-07-30):** weekly Fable usage was at 78% of the
   max-20x allowance; **pause production at 95%.** You cannot read the meter — work
   lean, prefer one well-aimed command over exploratory fan-out.

---

## 3. PHASE 1 — dequantize the deployment GGUF (on-box)

Recipe: `src/loratrain/RUNBOOK.md` **§3-ALT** (DRAFT, never executed — expect to correct
it as you go, and mark it `-EXECUTED` with real numbers when it works).

**3.1** Provision a fresh RunPod A40 (RUNBOOK §1), set `TRAIN_SERVER_IP` +
`TRAIN_SERVER_SSH_PORT` in `src/loratrain/config.py` — the single source of truth; a
test fails the suite on any IP literal elsewhere.

**3.2** Fetch the GGUF **on the box** from `lmstudio-community/Qwen3-8B-GGUF @ 07ebe812`
(§3-ALT; do NOT upload 5 GB, and do NOT widen `upload_guard`'s dataset-only allowlist).
Verify its sha256 equals the pin above before using it.

**3.3** Run the dequantizer, `--plan` first, then for real:

```
python3 -m loratrain.gguf_to_hf --gguf <box path>/Qwen3-8B-Q4_K_M.gguf \
  --out /workspace/base_dequant --expected-sha256 a7676d25…8f35f \
  --gguf-py-dir /workspace/llama.cpp/gguf-py --plan
```

Expect **399 tensors** (217 Q4_K / 37 Q6_K / 145 F32); qwen3 permutation is identity
(proven from converter source). Two-pass shard streaming keeps peak RAM ≈ one shard.
It writes `dequant_manifest.json` and refuses a non-empty or symlinked `--out`.

**3.4 — BLOCKING DECISION (R4), do not paper over.** The box trainer loads bases as
**bf16**, which rounds away most of the fp32 dequant grid this phase just reconstructed.
Options: (a) force fp32/fp16 at load for this scheme, (b) accept bf16 and state plainly
that the arm tests "dequant-derived weights at bf16", not exact GGUF weights. **(b) is
defensible but must be a decision, not an accident** — it materially changes what a null
result would mean. Flagged in `gguf_to_hf` docstring; undecided at write time.

---

## 4. PHASE 2 — parity gate (BLOCKING, never yet executed)

`verify_dequant_parity.py` is the gate that proves the dequantized base actually behaves
like the GGUF it came from. **If it fails, stop — do not train.**

```
python3 -m loratrain.verify_dequant_parity --dequant-dir /workspace/base_dequant \
  --mode raw --expected-alias qwen3-8b-q4km-base --from-eval-set <n> --report <path>
```

- **`--mode raw` is the gate** (native `/completion`, byte-exact). Chat mode is
  **informational only** and never affects the exit code: b10107's chat parser is
  provably lossy (it drops the empty `<think>` block every `/no_think` anchor emits and
  consumes template whitespace). Exit codes: `0` pass / `1` divergence / `2` inconclusive.
- It **refuses to run while an eval owns the Qwen slot** — by design. Do not reach for
  `--i-own-the-qwen-slot` to silence it; confirm the slot is genuinely free per §2.1–2.2.
- Known residual: a chat-only report prints `verdict: PASS` where `NOT_GATED` would be
  clearer. Read the mode before believing the word.

Then chain identity: `verify_base_identity.py --dequant-dir …` (on-disk GGUF re-hash ≡
manifest ≡ pin) and, after training, `--compare-runs` as the cross-scheme tripwire.

---

## 5. PHASE 3 — train, then evaluate at `-fa off`

**5.1 Train.** Set `config.BASE_SCHEME = BASE_SCHEME_DEQUANT` (`"dequant_q4km"`; default
is `"fp16_hf_revision"` and shipped behavior is unchanged until you flip it). Dataset per
R3. Seeds: reuse the **same seed list** as the arm you are comparing against
(`20260722, 23, 24, 25, 26, 27, 29, 30, 31, 0801, 0802, 0803`) — paired seeds are what
make the comparison powerful. Hyperparameters stay the frozen control config
(r16 / α32 / dropout .05 / lr 1e-4 / 3 epochs / micro-batch 4 / grad-accum 4 = eff. batch 16,
linear→0 scheduler, no warmup, no weight decay). `upload_guard` enforces the receipt
chain under the dequant scheme; the trainer gates on `--base` consistency.

**5.2 Retrieve** adapters + `run_manifest.json` + `artifact_shas.txt` + `pip freeze`.
Confirm `base_scheme` and `base_source_sha256` are stamped in the manifest.

**5.3 Evaluate locally**, one seed at a time, serving:

```
llama-server -m <Q4_K_M gguf> --lora <adapter>.gguf --alias qwen3-8b-q4km-lora-s<seed> \
  -c 8192 -ngl 99 --parallel 1 -fa off --port 8081
```

then `evalharness/src/evalharness/run_eval.py --eval-set evalharness/data/eval_set.jsonl
--output-dir out/evalharness/dequant_s<seed> --model-tuned <that alias>
--backend-url http://127.0.0.1:8081/v1/chat/completions --max-concurrent 1`.

**Your driver must implement both guards from §2** — a slot guard that refuses to start
if any `llama-server`/`run_eval` is alive, and an identity guard that checks
`/v1/models` reports your intended alias before every eval. Reference implementation:
`scratchpad/v2_faoff_batch.sh` from session a3e95f20 (note: `/v1/models` on this build
returns keys `models`, `object`, `data` — parse `data[0].id`). Budget ~38 min/seed.

---

## 6. PHASE 4 — analysis, pre-registered before the first holdout read

**Primary:** per-seed holdout delta vs the frozen 43/100 baseline; **sign test** on the
seed-level deltas (distribution-free, unaffected by outliers — this campaign has been
misled twice by a single high seed). **Secondary:** one-sample t and 95% CI.
**Paired:** dequant vs fp16 at matched seeds — the sharper instrument, since it removes
seed-level variance.

**Declare one p-value convention and state it in the doc.** The existing record is
inconsistent: the committed verdict reports the n=12 sign test as `p=.344` / t `p=.122`
(two-sided) while a parallel lane records the same data as `.17` / `.061` (one-sided).
Both are the same numbers under different conventions. Pick one, say so, fix the other.

**Multiplicity — this is the third arm on one holdout** (v1 fp16, v2cap1 masking,
now dequant). Each is a legitimate replication read of a frozen recipe, but **choosing
the best of three by holdout score is selection**, and it burns the exam for all of them.
Pre-register the single primary comparison before reading, or apply an explicit
correction and say which. Never tune a knob, a policy, or a config by holdout score.

Report: per-seed table (tuned/100, Δpp, discordant b/c, exact McNemar p, anchors
kept/10 + fail-solved/10), the aggregate statistics above, and the paired comparison.
Anchors are guards, not headline: solved-anchors must stay solved, fail-anchors must
stay failed.

---

## 7. MUST NOT CHANGE

- Corpus, split, `eval_set.jsonl`, and the frozen baseline — all four are the exam.
- Wire-format pins `config.py` `PASS_AT_K_SYSTEM_PROMPT` / `PASS_AT_K_NO_THINK_SUFFIX`
  (the leading space is load-bearing; tests tripwire both).
- v1 artifacts under `src/loratrain/data/run1_final/` and the v2 datasets under
  `data/v2/` — byte-identical; they are comparison baselines.
- Hyperparameters, unless R2/R3 say otherwise. This arm varies **one** thing: the base.
- `out/**` append-only. No push without Nicky's word. No commits without release.

## 8. ACCEPTANCE

1. `dequant_manifest.json` written; tensor census matches 399 (217/37/145).
2. Parity gate run in `raw` mode, exit 0, report archived under `out/`.
3. N adapters trained, `base_scheme: dequant_q4km` + `base_source_sha256` ≡ pin in every
   manifest entry; `verify_base_identity --compare-runs` clean.
4. N eval dirs, **120 rows each**, slices 100/10/10, every eval preceded by a passing
   identity guard, all at `-fa off`.
5. Analysis doc with the pre-registered statistics, the declared p-convention, and an
   explicit statement of what may and may not be claimed.
6. RUNBOOK §3-ALT flipped to `-EXECUTED` with real timings; suites green
   (`src/loratrain` **581 passed + 2 env-dependent skips**; three-suite **1118**).

## 9. KNOWN RESIDUALS (inherited, all fail-closed/low — do not rediscover)

Resume skip-check doesn't key on `run_config` sha; a hardlink to outside passes the
symlink gate (sha still verified); chat-only parity report says `PASS` where
`NOT_GATED` is meant; `--mode both`'s all-empty guard is per-run; `base_source_sha256`
carries two formats (40-hex fp16 revision vs 64-hex sha). Also open: the §4 smoke
criterion (epochs/BANANA clause) is documented but not code-fixed.

## 10. SURFACE, DO NOT DECIDE

The bf16-at-load fork (§3.4) if R4 was left blank; whether a null result should trigger
the self-distillation-saturation hypothesis (targets are the base model's own rollouts;
training loss floors within a few steps — see `docs/lora_consistency_verdict.md`); and
anything you find that looks like a third structural defect. Both known dataset defects
were found by unbriefed external review, not by the pipeline's guards — **assume more of
that class exists, and report rather than quietly fix.**

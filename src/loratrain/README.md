# loratrain — LoRA training arm for qwen3-8b on the pde625 band corpus

Status: **campaign COMPLETE (2026-07-29).** W0–W5 all executed: split ruled +
dataset built (W2 — 700 verbatim rollouts, sha `7fa7e5bf`), then via the RUNBOOK's
remote flow (guarded upload → `remote/` box-side trainer on a RunPod A40 →
GGUF-convert → local llama-server eval) **12 control-config seeds** trained and
holdout-evaluated (run-1 3 + stage-R 5 + D2 extension 4), plus a 6-config stage-A
HP screen. Final verdict: `docs/lora_consistency_verdict.md` — holdout effect
≈+1.7pp point estimate at N=200 training examples, **not distinguishable from zero
at n=12 seeds** (the interim n=8 significance did not survive its pre-registered
extension). Next experiment (Nicky-gated): dataset v2 with completion-only masking
(`docs/lora_decisions_2026-07-28.md` D5). The module remains the reusable training
arm for that round.
Sub-repo, same pattern as `evalharness/` and `src/posers/*`: own `pyproject.toml`,
stdlib-only, zero import dependency on icepick. This module **trains**; it never
measures. Measurement belongs exclusively to `evalharness/` (design authority:
`docs/eval_harness_design.md`) and the two stay decoupled — loratrain's only
contract with the harness is file-shaped: it consumes the harness's derived
`train_uids.txt` and its serving recipe, and produces an adapter for the harness
to score.

## The one knob operators edit: the training-server address

Training runs on a **remote box** (operator directive, 2026-07-22). The remote
server's IP lives in **exactly one place**:

```python
# src/loratrain/config.py
TRAIN_SERVER_IP = "127.0.0.1"    # <-- EDIT HERE: pod public IP (ssh/scp target), and nowhere else
TRAIN_SERVER_PORT = 8000         # M4-local end of the status tunnel
TRAIN_SERVER_SSH_PORT = 22       # <-- EDIT HERE when provisioning: pod's external port -> container 22
```

`TRAIN_SERVER_URL` is derived inside `config.py`, and every server call in this
package goes through it. Since the SSH-tunnel-only decision (RUNBOOK D-R1,
revised 2026-07-25) the split is: `TRAIN_SERVER_IP` is the pod's public IP used
for **ssh/scp only** (the box binds its status server to loopback — no pod HTTP
port exists), while `TRAIN_SERVER_PORT` is the **M4-local end of the status
tunnel** (`loratrain.tunnel`), making `TRAIN_SERVER_URL` tunnel-local. No other
file may contain an IP or URL literal —
`tests/test_config.py::test_single_source_of_truth_for_server_address`
scans the package and fails the suite if one appears. To retarget the trainer,
edit `TRAIN_SERVER_IP` (plus `TRAIN_SERVER_SSH_PORT` per pod); nothing else
moves. (RUNBOOK Appendix A — the SSH-port field + tunnel-local URL derivation —
was applied 2026-07-25 on Nicky's go-ahead.)

Auth mirrors the pass@k `qwen_http` convention (commit `21092f4`):
`TRAIN_SERVER_KEY_FILE` is an optional **path proxy** to a key file (raw token or
`KEY=VALUE`); `None` = keyless. Key material is read only at request time to
attach a Bearer header — never printed, logged, or committed.

## W0 decisions (each grounded in disk evidence, 2026-07-22)

### D1 — Trainer framework: remote job submission; adapter contract over stack choice

The operator directed that training executes remotely. Local recon confirms
there is nothing to train with here anyway: no `mlx`/`mlx_lm`, no `peft`
(torch 2.12.0 only, no llama.cpp binaries on PATH). So `train_lora.py` is a
**thin client**: it validates guards, POSTs the dataset + hyperparameters to
`TRAIN_SERVER_URL`, polls, and retrieves artifacts. The binding contract is on
the **artifact, not the stack**: the remote trainer must return a
**PEFT-format LoRA adapter** (`adapter_config.json` + `adapter_model.safetensors`
targeting Qwen3-8B source weights), because that is the input
`convert_lora_to_gguf.py` (llama.cpp) accepts for D3's serving path. Presumptive
remote stack is HF peft/TRL on CUDA; MLX-LM is acceptable **only** if its adapter
exports to PEFT format cleanly. Open until the box is reachable (IP not yet set).

### D2 — Base weights: none needed locally; remote box needs Qwen3-8B source weights (gated)

Local disk holds only quantized variants (`~/.lmstudio/models/lmstudio-community/`:
`Qwen3-8B-GGUF` Q4_K_M + `Qwen3-8B-MLX-4bit`; HF cache has no qwen). LoRA trains
on **source/FP16 weights, never the Q4_K_M GGUF** — under the remote plan those
weights (~16 GB, `Qwen/Qwen3-8B`) must exist on the **remote box**. That fetch is
a **W3 gated step requiring operator approval**; nothing is downloaded locally in
any phase.

### D3 — THE CRUX, train→serve quant reconciliation: **Path A**, runtime GGUF LoRA via `llama-server`

The eval baseline is base GGUF-Q4_K_M (`corpus_provenance.scored_format`
confirms `GGUF-Q4_K_M`, scored model `qwen3-8b`). The tuned arm must differ from
base by ONLY the adapter at identical quant (cross-quant mismatch measured at
~1.32/8 per problem — gain-sized).

- **Chosen: (A) runtime adapter.** llama.cpp's `llama-server` loads the
  **bit-identical** Q4_K_M base GGUF plus a GGUF-converted LoRA adapter
  (`--lora`), applied at inference without touching base weights. Recipe below.
- **LM Studio is disqualified from serving the tuned arm**: `lms load` (0.4.15)
  exposes no LoRA/adapter option — the eval design's "LM Studio loads GGUF+LoRA"
  assumption does not hold on the installed version.
- **Engine-parity consequence (load-bearing):** since the tuned arm must be
  served by `llama-server`, the **baseline must be captured on the same
  `llama-server` build too** — a LM-Studio-baseline vs llama-server-post pair
  would add an engine confound. Verified 2026-07-22: **no `baseline_greedy.jsonl`
  exists anywhere yet**, so no rework — but the baseline capture (evalharness
  step 3) must use the recipe below, not LM Studio. llama.cpp is not installed;
  installing it (free, no weights) is a W4 prerequisite flagged for the operator.
- **(B) merge→re-quantize is rejected**: re-quantization drifts the base and
  reintroduces the confound; it would force re-baselining against the
  re-quantized base. Fallback only if (A) fails in practice, with both arms
  re-based on the re-quantized artifact and the deviation documented.

### D4 — Training targets: harvested verified-correct traces (RFT), verbatim

Band records were solved 1–6/8 by the base model, so correct traces already
exist. Disk evidence: each corpus row's `corpus_provenance.source_file` names
its scoring run (e.g. `out/remote_rescore/tier3_misdirection/pass_at_k.jsonl`);
the sibling `_progress/rollouts.jsonl` holds
`{uid, rollout_uid, sample_idx, output, verdict, ...}` with
`verdict ∈ {correct, wrong, degenerate}` — per-rollout sympy verdicts, keyed by
the corpus row's `rollout_uids`.

`build_dataset.py` (W2) harvests **only `verdict == "correct"` rollouts of
train-uid records**, and the SFT target is the rollout `output` **verbatim**,
wrapped in the byte-identical pass@k wire format (same system prompt +
`/no_think` — invariant 2 of AGENTS.md) so train and serve distributions match.

**Append-log caveat (measured 2026-07-25; found by one W2 session, confirmed
independently by the other):** rollouts.jsonl files are append-across-passes
logs — a rescore re-samples under the SAME `rollout_uid` (tier1_band: 651
duplicate entries, 104 with differing content), and 15/293 rows' counts come
from a later rerun their `source_file` never names. The builder therefore
indexes each file by LAST occurrence per `(uid, rollout_uid)` and reconciles
every row to its authoritative file: the routed file if its verdict tally
equals the row's `n_correct/n_wrong/n_degenerate`, else the UNIQUE registry
file that does (registry = routed files ∪ `REGISTRY_GLOBS`); zero or ≥2
candidates → hard fail. Measured on the pinned corpus: 278 routed + 15
unique-alternative = 293/293, zero ambiguous. Every example's provenance
carries `trace_file` + `reconciled_via`, and the manifest records the full
registry with per-file shas and duplicate counts. Documented residual
(accepted, both W2 sessions): the tally tie is multiset-level — same-verdict
subset rewrites across passes can yield a mixed-pass harvest that still
reconciles; acceptable because every harvested line is independently
verifier-accepted, which is the property D4 actually needs.

**Why this defends the grader-equivalence finding** (the grader marks
algebraically-equivalent answers wrong; training on canonical answer format
would fabricate a gain): (1) the corpus `answer` field never enters a training
target — no canonical string is ever taught; (2) target answers are surface
forms the base model already emits *and the verifier already accepts*, so the
equivalence-acceptance channel is unchanged — what is trained is the reasoning
that reaches an accepted answer; (3) the builder asserts targets are verbatim
rollout outputs (no normalization toward `answer`), and `(problem → canonical
answer)` pairs are **not used at all**. Residual risk stated honestly: RFT still
upweights the model's own verifier-friendly styles; acceptable because both
arms are scored by the identical verifier and equivalence rules, paired per
record (McNemar).

## Split & corpus: the ruled 200/100 split (SUPERSEDED SECTION — updated 2026-07-26)

**Ruling (Nicky, 2026-07-26): the regenerated 200/100 split is authoritative.**
`evalharness/data/corpus_split_200_100.json` (created 2026-07-26, seed 20260726,
full sha256 pinned as `config.EXPECTED_SPLIT_SHA256 = 768436f4…e4ce`-family — see
config.py) defines: holdout 100 pure band / train 200 = 193 band + 7 GGUF-7/8
backfill (training-only, harvested from pinned first-pass trace sources in
`config.BACKFILL_TRACE_SOURCES`). The former derived-view canon described in the
previous revision of this section — `eval_paper_split.json` (`110a4bf2…`, 108
eval papers) — is **retired** to `evalharness/data/retired_20260726/`; its
`build_eval_set` pin now refuses loudly (missing file), repoint pending in the
evalharness lane. loratrain consumes the ruled split's `train_uids.txt` +
`eval_set.jsonl` and re-asserts the guarantees independently (defense in depth,
paper-level):

- corpus pinned: `band_corpus.jsonl` must match
  `EXPECTED_CORPUS_SHA256 = e0975e11…` / **293 rows** (pinned 2026-07-22; this
  also compensates, from the consumer side, the known gap that
  `build_eval_set.py` does not yet pin the corpus sha — skeleton C1);
- the ruled split file must match `EXPECTED_SPLIT_SHA256` (full sha, authoritative);
  any harvested example whose `arxiv_id` is an eval paper → **hard fail**
  (paper-level, not just uid);
- any harvested uid appearing in `eval_set.jsonl` (when present) → hard fail;
- dedupe on `(uid, rollout_uid)`; statement-level dup checks use **full**
  statements (finding F4: truncated keys manufacture ghosts).

The 100-record-holdout language in older briefs maps to today's derived
eval set; the sanctity rule is unchanged: **eval-paper records are radioactive
to training.**

## Non-negotiable ordering & invariants (enforced by code, not prose)

1. **Baseline before training.** `train_lora.py` refuses to submit a job unless
   `BASELINE_GREEDY_PATH` exists and is non-empty (`baseline_greedy.jsonl`,
   the exact filename `evalharness/run_eval.py` writes), and records its sha256
   in the run manifest.
2. **Leakage hard-fails** as above, in `build_dataset.py`, before any bytes go
   anywhere.
3. **Quant match**: both arms serve per the D3 recipe; base GGUF file
   sha256-pinned in the run manifest.
4. **Data movement**: corpus + traces are local and gitignored (`data/` never
   committed). The ONLY permitted upload is the training payload to the
   operator's own rented box, THROUGH the guarded uploader (RUNBOOK §5,
   `upload_guard.py`), and only in operator-approved W3. Nothing goes to third
   parties.
5. **Reproducibility**: fixed seeds (`SEED = 20260722`), full config + input
   shas captured in `run_manifest.json`.

## Exact train→serve recipe (the contract W2–W5 implement)

```text
0. evalharness-build-set … --output-dir <evalset_dir>      # derived view; $0, local
1. BASELINE (before any training; llama-server, NOT LM Studio):
     llama-server --model <pinned Qwen3-8B Q4_K_M .gguf> --alias qwen3-8b-q4km-base \
                  --port <P> [greedy handled by harness]
     evalharness-run --eval-set <evalset_dir>/eval_set.jsonl \
_                    --model-base qwen3-8b-q4km-base \
                     --backend-url http://127.0.0.1:<P>/v1/chat/completions
     -> baseline_greedy.jsonl                                # unlocks train_lora
2. W2: loratrain-build-dataset -> data/sft_train.jsonl (+ dataset_manifest.json)
3. W3 (GATED: operator approves spend/fetch + the upload to their box):
     loratrain-train -> POST dataset+config to TRAIN_SERVER_URL (single IP var)
     -> remote LoRA on Qwen/Qwen3-8B source weights -> PEFT adapter returned
4. W4: convert_lora_to_gguf.py (llama.cpp, install = flagged prerequisite)
     -> adapter.gguf
     llama-server --model <same pinned base .gguf, bit-identical> \
                  --lora adapter.gguf --alias qwen3-8b-q4km-lora --port <P>
5. W5: evalharness-run --model-tuned qwen3-8b-q4km-lora (same endpoint/settings)
     -> post_greedy.jsonl ; evalharness-report -> McNemar + anchor drift
```

Same `llama-server` build, same flags except `--lora`/`--alias`, same base file
(sha-pinned), same box, greedy both arms → the adapter is the only delta.

## Layout

```
src/loratrain/
  README.md                  # this file — design doc + decisions
  pyproject.toml             # sub-repo package; stdlib-only; pytest via [dev]
  .gitignore                 # data/ never committed
  src/loratrain/
    config.py                # TRAIN_SERVER_IP (THE editable variable), hyperparams,
                             # seeds, pinned shas, paths, validate_config()
    build_dataset.py         # W2 IMPLEMENTED: guarded build() + reconciliation + CLI
    train_lora.py            # W3 STUB + REAL ordering guard + remote client shape
    export_serve.py          # W4 STUB: adapter->GGUF->llama-server recipe
  tests/                     # all green, no network, no corpus dependency:
    conftest.py              #   sys.path bootstrap (evalharness pattern)
    test_config.py           #   validation + single-source-of-truth IP scan
    test_leakage_guard.py    #   uid- and paper-level hard fails
    test_ordering_guard.py   #   baseline-before-train enforcement
    test_trace_guard.py      #   verified-correct + verbatim-target enforcement
  data/                      # (gitignored) built datasets land here
```

Stubs raise `NotImplementedError` with their gate ("W2 — gated", "W3 — gated,
operator approval required") **after** running their real guards, so the guards
bind from day one and "train before baseline" is structurally hard.

## Open items (operator-facing)

- `TRAIN_SERVER_IP` is a placeholder (`127.0.0.1`) until the operator sets the
  remote box's address. Remote stack + source-weight presence unverifiable
  until then (D1/D2 riders).
- llama.cpp install (W4 prerequisite, free/local) — approval to install when
  W4 opens.
- Eval set not yet built; baseline not yet captured. Ordering: step 0–1 of the
  recipe before W3 ever runs. Baseline capture must follow D3's engine-parity
  note (llama-server, not LM Studio).
- `build_eval_set.py` corpus-sha pin (skeleton C1–C3) still open in the
  evalharness lane; loratrain's own pin covers the consumer side meanwhile.
- Hyperparameter defaults in `config.py` (r=16, α=32, lr=1e-4, 3 epochs,
  bf16, seed 20260722) are conventional starting points, not tuned — W3 review.

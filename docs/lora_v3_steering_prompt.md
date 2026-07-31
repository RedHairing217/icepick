# STEERING — v3 on-policy arm (`lora-v3-proofhint`), written 2026-07-31T19:0xZ

Paste into the v3 construction window. **You are building; you are not launching.**
Spec: `docs/lora_v3_proofhint_execution_skeleton.md` (authoritative — this file only
steers, pins live state, and orders the next moves). Repo
`/Users/redhairing/Desktop/helloworld/icepick`, `cd` in every Bash call. Read
`AGENTS.md` → `docs/SESSION_HANDOFF.md` → the skeleton → this.

## THE ONE CONSTRAINT THAT OUTRANKS EVERYTHING

**All v3 machinery goes in a NEW module `src/loratrain/src/loratrain/v3.py`
(+ `tests/test_v3.py`).** It may import from the existing package; **nothing existing
may import it, and no file the v1/v2/dq arms executed may be edited** —
`build_dataset.py`, `train_qwen3_lora.py`, `upload_guard.py`, `config.py` operator
block stay byte-identical. Additive constants only, in a marked v3 section, only if
unavoidable. **Every phase's acceptance = `git diff` on pre-existing loratrain files
is empty (or additive-config only) + prior-arm suites still pass.** Reason: a live
campaign is never patched underneath — three arms must stay reproducible from the tree
that produced them.

## LIVE STATE — verify before trusting, this env has fabricated notifications

| what | state | your posture |
|---|---|---|
| k=8 sweep, pod `skvfqhr0l5ilve` (A40) | RUNNING, 34 configs, dq_* then v2_*, ETA ~16:30Z Aug 1 | **DO NOT TOUCH.** Not your pod, not your slot. Own pods only. |
| proof-import lane, orchestrator `f7b24506` | LIVE on its own CPU pod; contracts frozen in `out/proof_import_20260731T185338Z/CONTRACTS.md`; produces `solutions_v3.jsonl` | **Your input. Do not duplicate, do not race.** Read their CONTRACTS.md; coordinate by artifact only. |
| dq-vs-v2 verdict | NOT IN (sweep incomplete) | **Blocks P3.** If dq comes back positive, R4 base-scheme re-rules → Nicky decides before any training. |
| `v3.py` | does not exist | you create it |
| git | HEAD `5be86ea`, main ahead 6, **unpushed** | local commits fine; never push |
| working tree | `config.py` has `BASE_SCHEME = BASE_SCHEME_DEQUANT` (dq lane, uncommitted), `RUNBOOK.md` modified | **leave both alone**; note them, don't revert |

## HARD-WON FACTS (do not rediscover — each cost real money or a corrupted run)

1. **Grading must run where `antlr4` is installed.** Measured 2026-07-31: a pod
   grader without `antlr4-python3-runtime` scored ~70/120 records WRONG per config,
   silently — sympy's LaTeX parser fails closed, so correct answers read as wrong and
   `n_correct` collapses to 0. The local M4 (antlr4 present) is authoritative. If you
   grade anywhere else, install antlr4 AND parity-check against a locally-graded
   config until byte-identical, or your numbers are fiction.
2. **`-fa off` explicitly, always.** The build defaults to `-fa auto`; the entire
   reference set is auto-resolved-off, and explicit `off` was verified byte-identical
   to it (3/3). `-fa on` is a different instrument.
3. **Slot guard + identity guard on every serving cycle.** A single `ps` snapshot is
   NOT proof a campaign is dead — one landed in a 3-second gap on 07-30 and corrupted
   another session's seed. Before taking a port: `ps` AND sibling progress logs under
   `/private/tmp/claude-501/*/scratchpad/*.log`. After serving: `GET /v1/models` and
   require the alias to equal the config you intend to run (this build returns keys
   `models`/`object`/`data` — parse `data[0].id`). Reference: `scratchpad/v2_faoff_batch.sh`.
4. **RunPod: `PUBLIC_KEY` env at create or sshd never starts** (API does not inject
   account keys). `nvcc` exists but is off PATH (`/usr/local/cuda/bin`, needs
   `-DCMAKE_CUDA_COMPILER` + `-DCMAKE_CUDA_ARCHITECTURES=86`). pip is PEP-668 locked
   → venv or `--break-system-packages`. ssh exit-255 ≠ failed launch (RST quirk) —
   VERIFY on a fresh connection, never infer.
5. **`/tmp` is not retrieval.** Three v1 adapters died there. Archive adapters into
   `src/loratrain/data/` immediately on pull, sha-verified.
6. **`setsid` does not exist on macOS** — plain `nohup` locally; `setsid nohup … &`
   on the box.
7. **Interim reads have been wrong every single time** (run-1 seed 1 +11; n=8 sign
   test p=.016 → reversed; v2 at 5 seeds +2.80 → 12 seeds +0.58). Do not report,
   quote, or reason from a partial series.

## SUBSTRATE (Nicky, 2026-07-31)

RunPod for all compute — regeneration, training, eval sweep. The M4 keeps exactly:
orchestration, **key custody** (API-key path proxies never ship; pod env echoes back
through the RunPod API), and **grading** (holdout answer key never leaves; see fact 1).
Your pods are yours: itemize spend, terminate same session, sha-verify artifacts down
before teardown.

## THE DESIGN IN ONE PARAGRAPH

v1/v2 saturated because own-rollout training carries no information the model lacks
(loss floors ~0.43 in epoch 1; k=8 shows sharpening, not capability — see
`docs/lora_v2_verdict.md`). v3 imports information (paper proofs) but keeps training
tokens **on-policy**: the proof is a hint at *generation* time only, the model
re-derives the solution itself, the endpoint is verified, and the model's own trace is
what gets trained on. **The hint must never appear in the training prompt** — serve
time has no hint, so train/serve prompts must match. cap1: one kept trace per record.

## ORDERED MOVES

1. **Read** their `CONTRACTS.md` + the skeleton + `out/passk8_sweep/grade.py` (the
   exact verify chain you must reuse).
2. **Build `v3.py`** — regeneration dataset builder (skeleton P1) with guards:
   train-split-only re-assert, statement-leakage vs `eval_set.jsonl`,
   prompt/completion masking columns, prompt-tokens-carry-zero-loss census,
   hint-never-in-training-prompt assert, cap1, R3 blend, full provenance per row.
   Tests alongside. **Isolation check after every commit.**
3. **Dry-run** the builder against whatever `solutions_v3.jsonl` exists (it may be
   partial or absent — build a fixture; the real file arrives from the other lane).
   Report censuses; write nothing to `out/**` outside your own run dir.
4. **STOP.** P2 regeneration onward needs: solutions file complete, dq verdict read,
   and Nicky's R1–R5 checkboxes. Do not rent a GPU pod before that.

## PRE-REGISTRATION

Write it before any holdout read, and put it on disk: primary = **v3 band→solved
transitions vs the k=8 base ruler, compared against v2's on the same ruler**,
two-sided, both series 12/12 before computing anything. State the p-convention
explicitly (the record was inconsistent: `.344` two-sided vs `.17` one-sided for the
same v1 data — declare which you use). Secondary: mean Δn_correct, band→collapse
losses, anchors, and the **loss-floor covariate** (if v3 floors at ≤0.45 again, the
arm's premise failed — that is a finding, report it prominently, do not bury it).

## SURFACE, DO NOT DECIDE

Hint-insufficient rate (records that fail to verify even with the proof in context —
R5 rules the fallback); n-gram overlap between hint text and the model's trace
(copying, not deriving); band rows verifying instantly on try 1 (hint redundant →
mix rebalance); anything that looks like a fourth structural defect. Both known
dataset defects were found by unbriefed external review, not by the pipeline's own
guards — **assume more exist, report rather than quietly fix.**

# evalharness

LoRA eval harness for **qwen3-8b** on the pde625 band corpus. Proves a
measurable performance improvement from LoRA training — a number that
survives scrutiny — by foreclosing two specific failure modes:

1. **Cross-quant/hardware confound.** Baseline and post-train eval must
   run on the same quant (MLX-4bit vs GGUF-Q4_K_M differ by about as
   much as a plausible LoRA gain).
2. **Regression-to-the-mean from selection.** The eval set is frozen
   *before* final training selection, split at the paper level, and
   scored fresh — selection-time scores are never eval baselines.

Full rationale, measurement protocol, and open items live in
[`docs/eval_harness_design.md`](../docs/eval_harness_design.md) (the
parent icepick repo) — that document is the authoritative design; this
README is the quickstart.

This is a standalone sub-repo, same pattern as `src/posers/*`: its own
`pyproject.toml`, stdlib-only, **zero import dependency on icepick**.
`run_eval.py` talks to icepick only by shelling out to the installed
`icepick` console script.

## Quickstart: build-set -> baseline -> post -> report

```bash
# 0. Install (editable, plus pytest for the dev extra)
pip install -e evalharness/'[dev]'

# 1. Build the eval set once the remote-rescore cascade has landed (or
#    however much of it currently exists -- tiers may be partial; pass
#    only the pass_at_k.jsonl paths that exist today and rerun as more
#    land).
evalharness-build-set \
  --tier-outputs \
    out/remote_rescore/tier1_band/pass_at_k.jsonl \
    out/remote_rescore/tier2_7of8/pass_at_k.jsonl \
    out/remote_rescore/tier3_misdirection/pass_at_k.jsonl \
    out/remote_rescore/tier4a_collapse_local/pass_at_k.jsonl \
  --output-dir evalharness/data
# -> evalharness/data/eval_set.jsonl, evalharness/data/train_uids.txt

# 2. Train the LoRA OUTSIDE this harness, consuming ONLY train_uids.txt
#    (the LoRA pipeline must never see eval_set.jsonl's records).

# 3. BEFORE training (or against the untouched base model): greedy
#    baseline.
evalharness-run \
  --eval-set evalharness/data/eval_set.jsonl \
  --output-dir out/evalharness/run1 \
  --model-base qwen3-8b-q4_k_m \
  --backend-url http://127.0.0.1:1234/v1/chat/completions

# 4. AFTER training: greedy post-train eval, same box/quant/settings.
evalharness-run \
  --eval-set evalharness/data/eval_set.jsonl \
  --output-dir out/evalharness/run1 \
  --model-tuned qwen3-8b-q4_k_m-lora \
  --backend-url http://127.0.0.1:1234/v1/chat/completions

# (equivalently, run both in one invocation once the tuned model is
# loaded -- see "Base + tuned together" below)

# 5. Report: paired diff, exact McNemar, anchor drift.
evalharness-report \
  --eval-set evalharness/data/eval_set.jsonl \
  --baseline out/evalharness/run1/baseline_greedy.jsonl \
  --post out/evalharness/run1/post_greedy.jsonl \
  --output out/evalharness/run1/report.md
```

Re-running any step with the same `--output-dir` resumes rather than
re-billing: `icepick processing pass_at_k` is restartable by contract
(finished records are cached; `run_eval.py` adds no resume logic of its
own, it just re-invokes the same underlying command).

## `build_eval_set.py`

Reads the frozen paper split (`evalharness/data/eval_paper_split.json`,
**never regenerated or reformatted** -- its sha256[:16] is pinned in
code and checked on every run) plus one or more remote-rescore
`pass_at_k.jsonl` files, and emits:

* `eval_set.jsonl` -- three slices, each record tagged `eval_slice`:
  * `eval_band` -- eval-paper records labelled `band` in the remote
    rescore (the improvement metric lives here).
  * `anchor_solved` -- eval-paper records at *exactly* k/k correct
    ("8/8") -- must STAY solved (catastrophic-forgetting detector).
  * `anchor_fail` -- eval-paper records labelled `collapse` at
    *exactly* 0/k correct ("0/8") -- must STAY failed
    (memorization/contamination detector).
* `train_uids.txt` -- uids of every non-eval-paper `band`-labelled
  record across the given tier outputs. **The LoRA pipeline consumes
  ONLY this file.**

Hard-fails (non-zero exit, one-line actionable stderr message, no
traceback) on:

* the split file's sha256[:16] not matching the pinned value;
* (defense-in-depth) any train uid resolving to an eval paper -- the
  bucketing logic already prevents this structurally, but it is
  re-asserted explicitly so a future refactor can't reintroduce it
  silently;
* any assembled eval-set record missing a non-blank `statement` or
  `answer`;
* any `--tier-outputs` path that doesn't exist on disk (lists every
  missing path at once).

```bash
evalharness-build-set \
  --split evalharness/data/eval_paper_split.json \
  --tier-outputs out/remote_rescore/tier1_band/pass_at_k.jsonl ... \
  --output-dir evalharness/data
```

Safe to rerun as more cascade tiers land -- it always rebuilds both
output files wholesale from whatever `--tier-outputs` you pass, never
appends across runs.

## `run_eval.py`

Subprocess-drives `icepick processing pass_at_k` -- does **not**
reimplement rollout generation, extraction, or verification. Two
passes:

* **Primary (always run): greedy pass@1** -- `--k 1 --temperature 0
  --think off --max-tokens 2048`, over the WHOLE eval set (eval-band +
  both anchor slices, since anchor drift needs greedy outcomes too).
  These wire params are fixed constants, not CLI flags.
* **Secondary (opt-in via `--secondary`): k=8, temperature=0.7, 3
  repeats**, over eval-band only. Distributional signal; `report.py`
  never blends it into the headline.

`--model-base` / `--model-tuned` may be given alone (matching the
before/after checklist above) or together in one invocation. The tuned
model is assumed to be a distinct model id on the **same**
OpenAI-compatible endpoint as the base model (LM Studio loading
GGUF+LoRA or a merged export alongside the base GGUF) -- `--backend-url`
is shared by default. Supplying both models against *different*
endpoints (`--backend-url-base` / `--backend-url-tuned`) refuses with a
quant-confound explanation unless you pass `--allow-cross-endpoint`
(which still prints a warning -- the confound doesn't go away, you're
just asserting you've checked it another way).

```bash
# Base + tuned together, same endpoint, plus the k=8x3 secondary:
evalharness-run \
  --eval-set evalharness/data/eval_set.jsonl \
  --output-dir out/evalharness/run1 \
  --model-base qwen3-8b-q4_k_m --model-tuned qwen3-8b-q4_k_m-lora \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --secondary

# Remote gateway behind bearer auth (Admiral Tangerine fronting LM Studio):
evalharness-run \
  --eval-set evalharness/data/eval_set.jsonl \
  --output-dir out/evalharness/run1 \
  --model-tuned qwen3-8b-q4_k_m-lora \
  --backend-url https://admiraltangerine.com/api/v1/chat/completions \
  --qwen-key-file ../tangerine_api.env
```

Outputs land under `--output-dir`: `baseline_greedy.jsonl` /
`post_greedy.jsonl` (copies of icepick's own `pass_at_k.jsonl` for each
role), `{base,tuned}_greedy/` (icepick's full stage output, including
its own manifest), and if `--secondary` was given,
`{base,tuned}_secondary/rep{0,1,2}/`.

**Key files are path-proxies.** `--qwen-key-file` (and its
`-base`/`-tuned` overrides) are passed straight through to the icepick
subprocess as a path string -- this tool never opens or prints their
contents.

## `report.py`

Reads `eval_set.jsonl` (for slice membership) plus the two greedy
outputs, and optionally the k=8x3 secondary files. Prints a markdown
report to stdout AND writes it to `--output`.

* **Paired greedy diff on eval-band**: per-uid solved/unsolved, base vs
  tuned, as a 2x2 table (a = both solved, b = base-only/"regression", c
  = tuned-only/"the gain", d = neither).
* **Exact McNemar** on the discordant pairs (b, c) -- `math.comb` only,
  no scipy. `p = min(1, 2 * P(X <= min(b,c)))` for `X ~ Binomial(b+c,
  0.5)`.
* **95% CI** on the paired difference -- a normal-approximation (Wald)
  interval, explicitly labelled as approximate (an exact paired CI has
  no closed form and is out of scope for a stdlib-only implementation).
* **Anchor drift**: for anchor-solved, how many regressed (base solved,
  tuned did not); for anchor-fail, how many got contaminated (base
  failed, tuned solved). Either count above zero is flagged in the
  report.
* **UNDERPOWERED warning** whenever eval-band has fewer than 25 scored
  records (per the design doc: "Under ~25 records, say underpowered out
  loud").
* **Secondary distributional table** (per-repeat mean `n_correct`, base
  vs tuned) when `--secondary-base` / `--secondary-post` are given --
  always in its own section, never folded into the headline.

```bash
evalharness-report \
  --eval-set evalharness/data/eval_set.jsonl \
  --baseline out/evalharness/run1/baseline_greedy.jsonl \
  --post out/evalharness/run1/post_greedy.jsonl \
  --secondary-base out/evalharness/run1/base_secondary/rep{0,1,2}/pass_at_k.jsonl \
  --secondary-post out/evalharness/run1/tuned_secondary/rep{0,1,2}/pass_at_k.jsonl \
  --output out/evalharness/run1/report.md
```

## Install

```bash
pip install -e evalharness/            # core, stdlib only
pip install -e evalharness/'[dev]'     # + pytest
```

## Tests

```bash
python3 -m pytest evalharness/tests -q
```

No network, no icepick dependency, no live backend: `build_eval_set`
and `report` are pure file-in/file-out and are tested directly against
small synthetic fixtures under `tests/fixtures/`; `run_eval`'s tests
inject a fake subprocess runner rather than invoking icepick or a real
Qwen endpoint (mirrors icepick's own injectable-backend test pattern).
This suite is intentionally NOT swept up by the parent repo's root
`pytest` (its own `pyproject.toml` scopes `testpaths` to its own
`tests/`) -- run it explicitly, same as the `src/posers/*` suites.

## Out of scope

Rollout generation, answer extraction, and the sympy equivalence
verifier all stay inside icepick's `processing pass_at_k` stage --
this harness never reimplements them, only orchestrates and analyzes
around them. LoRA training itself is also out of scope ("Train LoRA
(outside this harness)" per the design doc's protocol checklist).

# Pipeline controller reference

A cold-start guide for anyone (human or fresh Claude session) building
automation on top of icepick's pipeline. Assumes you have shell access
and can run the CLI; does not assume any prior context.

Read this before wiring a controller / cron / dashboard. It covers what
each stage does, what it writes, how to chain them, and what invariants
you can depend on.

For architectural rationale see [`plan.md`](plan.md). For hands-on ops
see [`operator.md`](operator.md) and [`scraper_runbook.md`](scraper_runbook.md).

---

## What icepick is

A portable processing surface for ModelBreaker-style problem records.
End-to-end pipeline: **acquire → filter → gate → score → labeled corpus**.

Everything is CLI-driven. Every stage writes a JSON manifest under its
`--output-dir`. Every stage's output JSONL is the next stage's input.
There is no daemon, no message queue, no shared database — the
filesystem is the coordination surface.

---

## The five stages

```
allocation (scrape/mount)   arxiv or drop → handoff/records.jsonl
        │
        ▼
processing groundtruth      publication check → published.jsonl
        │                   (CURRENTLY KILL-SWITCHED — see §Kill switches)
        ▼
processing wellposed-cascade   3-stage well-posedness gate → passed_records.jsonl
        │                      (or use `processing wellposed` for parallel fleet)
        ▼
processing pass_at_k        k rollouts against a subject model → labeled records
        │                   with pass_at_k, label ∈ {solved, band, misdirection, collapse, drop}
        ▼
final corpus                whatever the last stage emits
```

Stages are individually invocable. Two composition options:

- **`processing pipeline` command** chains groundtruth → wellposed → pass_at_k in one call. Supports two orders via `--pipeline-order`:
  - `classic` (default): wellposed then pass_at_k
  - `solvable-first`: pass_at_k first, drops records the model can't score, then wellposed
- **Manual chaining**: run each subcommand in sequence, use the previous stage's output path as the next stage's `--input`. This is required when a stage is kill-switched (skip and hand-thread the previous stage's output).

---

## Repo layout (what to read when)

```
src/icepick/
├── cli.py                          all subcommand parsers + handlers
├── config.py                       shared ConfigError, mode validation, key resolution
├── contracts/
│   ├── manifests.py                ApprovedManifest, ProposedPlan dataclasses
│   └── records.py                  ProblemRecord, PassKRecord, BAND_LO/HI constants
├── allocation/
│   ├── intake.py                   mount (manual drop → handoff)
│   ├── manifests.py                write/load ApprovedManifest, run_id generation
│   ├── adapters/
│   │   ├── manual_mount.py         file/dir → canonical records
│   │   └── realmath_scrape.py      arxiv scrape adapter (plan/estimate/run/normalise)
│   └── scrape/
│       ├── realmath.py             actual arxiv HTTP + LaTeX + LLM Q+A extraction
│       └── checkpoint.py           per-paper disk commits (restartability infra)
├── processing/
│   ├── pipeline.py                 chains stages; classic/solvable-first orders
│   ├── ingest.py                   normalise raw JSONL into canonical records
│   ├── schema.py                   from_raw() + label derivation
│   ├── groundtruth/
│   │   ├── config.py
│   │   ├── runner.py
│   │   └── anthropic_adapter.py    KILL-SWITCHED: hardcoded placeholder API key
│   ├── poser/                      well-posedness stage (parallel fleet + cascade)
│   │   ├── config.py               WellposedConfig, Combo, comparison policies
│   │   ├── runner.py               parallel fleet driver
│   │   ├── cascade.py              3-stage sequential elimination
│   │   ├── claude_adapter.py       drives claude-poser subprocess
│   │   ├── codex_adapter.py        drives codex-poser subprocess
│   │   └── comparator.py           cross-combo agreement + kappa report
│   └── pass_at_k/
│       ├── config.py               PassAtKConfig, backend enum, policy defaults
│       ├── runner.py               top-level run(); retry + concurrency
│       ├── checkpoint.py           per-rollout + per-record durability
│       ├── scoring.py              extract_boxed, label derivation
│       ├── verifier.py             sympy-backed numeric/symbolic verify
│       └── backends/
│           ├── qwen_http.py        LOCAL: LM Studio / vLLM / Ollama (POLICY default)
│           ├── anthropic.py        PAID: two-flag opt-in required
│           └── openai.py           PAID: two-flag opt-in required
└── posers/                         vendored Claude_Poser / Codex_Poser (CLI binaries)

tests/                              mirrors src layout; fake adapters keep tests offline
```

---

## Stage-by-stage reference

### Stage 1a: `allocation mount`

Simplest ingress. Takes a JSONL/CSV/TSV drop or dir, normalises to
canonical records, writes handoff. Auto-approves the manifest
(`call_budget=0`, no calls to authorise).

```sh
icepick allocation mount \
  --path /path/to/drop.jsonl \
  --source my_source \
  --provenance {manual|external|extracted} \
  --truth-policy {trusted|extracted|unknown} \
  --output-dir out/intake \
  [--family <name>] \
  [--column canonical=source ...]   # CSV/TSV column projection, repeatable
```

**Writes**: `out/intake/runs/<run_id>/manifest.json` + `handoff/records.jsonl`.

**Exit code**: 0 on success, 1 on malformed input.

### Stage 1b: `allocation plan` → `allocation approve` → `allocation run`

Real scraper (currently only realmath). Two-step human gate: `plan`
proposes, `approve` produces an `ApprovedManifest`, `run` executes.

```sh
icepick allocation plan \
  --source-type realmath_scrape \
  --source <name> \
  --target-count N \
  --output-dir <dir> \
  --category math.NT \             # arxiv category (math.NT, math.CO, math.AG, etc.)
  --primary-only \                 # drop cross-listed papers
  --extraction {abstract|latex|qa} \
  --max-per-paper 2 \              # candidates per paper (breadth cap)
  --year YYYY --month M \          # optional date lower bound

icepick allocation approve \
  --plan <path> \
  --mode {production|flow_testing} \
  --approved-by <user> \
  --call-budget N \                # hard cap on paid calls
  --output-dir <dir>

# Optional flag on approve, matches flow-testing semantics:
#   --calibration-sheet <path>     required when --mode flow_testing

icepick allocation run \
  --manifest out/intake/runs/<run_id>/manifest.json
```

**Extraction modes** (`--extraction`):
- `abstract`: one candidate per paper, statement = paper abstract; no LLM cost
- `latex`: mine theorem envs from LaTeX source; extracts `\boxed{}` answers; no LLM cost
- `qa`: **single-stage LLM** — Sonnet Q+A reformulation, one call per mined theorem. Sonnet is the filter (returns nothing for theorems with no single fixed answer). Only extraction mode that produces cascade-ready records.

**QA extraction cost model** (per math.NT theorem, empirical):
- Sonnet Q+A: ~$0.005/call, one call per mined theorem
- A former Haiku pre-filter gate was removed: it accepted every theorem (zero
  selectivity) while Sonnet did the real filtering, so it was pure cost.
- Prompt caching does not lower this: the ~480-token QA prompt is below
  Anthropic's 2048-token minimum cacheable prefix, so `cache_control` reads 0.

**Writes** (`out/intake/runs/<run_id>/`):
```
manifest.json                    ApprovedManifest
handoff/records.jsonl            ← what processing consumes
raw/papers.jsonl                 unique paper pool
raw/qa_candidates.jsonl          per-theorem QA cache
raw/extracted_candidates.jsonl   per-theorem statement summary
raw/quarantined.jsonl            only if candidates were dropped
reports/source_report.md         counts, spend, warnings, drops
_progress/                       restartability infra
├── papers_done.jsonl            append-only: {arxiv_id, candidates}
├── candidates.jsonl             append-only: per-paper committed candidates
├── qa_cache.jsonl               keyed by SHA1(statement) → generator result
├── rate_limited_at              ISO timestamp of last 429/503, if cooling down
├── rate_limit_events.jsonl      append-only 429/503 log: {at, status, backoff_seconds}
└── INCOMPLETE                   marker while run is unfinished
```

**Exit codes**:
- 0: success
- 1: interrupted (budget, Ctrl-C, network 429/5xx). Same command resumes. Look for `"status": "interrupted_resumable"` in the JSON summary.

### Stage 2: `processing groundtruth`

Anthropic web_search publication check. **Currently kill-switched** —
the adapter's client is instantiated with a placeholder API key so any
invocation returns 401 without spending money. Any invocation writes
`counts.error = N` for every input record.

```sh
icepick processing groundtruth --mode production \
  --input <handoff.jsonl> \
  --output-dir <dir> \
  --anthro-key-file <path> \
  [--judge-model claude-opus-4-7] \
  [--judge-samples 3 --judge-uphold 2] \
  [--cache-path <path>] \
  [--cost-per-input-mtok N --cost-per-output-mtok N]
```

**To re-enable**: restore the `_build_anthropic_client` body in
[groundtruth/anthropic_adapter.py](../src/icepick/processing/groundtruth/anthropic_adapter.py).
Docstring on that function has the exact restore snippet.

**Writes**: manifest, verdicts.jsonl, published.jsonl (input to next
stage), discarded.jsonl.

### Stage 3: `processing wellposed-cascade`

3-stage sequential well-posedness gate. Each stage runs a single
`(build, provider)` combo; only records that stage N judges `well_posed`
reach stage N+1. Default stage list:

```
codex:openai → codex:anthropic → claude:openai
```

Rationale: cheapest+most-permissive first (kills obvious ill-posed),
strictest formalist second, semantic corroborator third. Each combo runs
one of the vendored posers (`src/posers/Claude_Poser/`,
`src/posers/Codex_Poser/`) as a subprocess.

```sh
icepick processing wellposed-cascade --mode production \
  --stages codex:openai,codex:anthropic,claude:openai \
  --input <records.jsonl> \
  --output-dir <dir> \
  --anthro-key-file <path> \
  --openai-key-file <path> \
  [--judge-samples 3 --judge-uphold 2] \
  [--max-retries 2] \                # retries transient network errors per stage
  [--cost-per-input-mtok 1 --cost-per-output-mtok 5]
```

**Alternative**: `processing wellposed` runs the fleet in **parallel**
(all combos simultaneously, then applies a `--comparison-policy` to
combine verdicts). More expensive; used when you want cross-combo
agreement stats.

**Cost model** (per 25 records, live-measured):
- Cascade: $0.05-0.10 total (records shed at each stage)
- Parallel fleet: $0.20-0.40 (every combo sees every record)

**Writes** (`<output-dir>/`):
```
cascade_manifest.json            per-stage counts, wall-clock, cost, retries
stage_1_codex_openai/            single-combo run manifest + verdicts
stage_2_codex_anthropic/         (only survivors of stage 1)
stage_3_claude_openai/           (only survivors of stage 2)
final_corpus.jsonl               ← records that passed all stages
```

### Stage 4: `processing pass_at_k`

k rollouts per record against a subject model, verify each output
against the truth answer, stamp `pass_at_k` + `label` fields.

```sh
icepick processing pass_at_k --mode production \
  --input <records.jsonl> \
  --output-dir <dir> \
  --backend {qwen_http|anthropic|openai} \
  --model <model-id> \
  --k 8 \
  --temperature 0.7 \
  --max-tokens 2048 \
  --think {on|off} \                # "off" appends " /no_think" (Qwen3 convention)
  --backend-url URL \               # qwen_http only
  --max-concurrent 4                # concurrent records; rollouts stay sequential
```

**Labels** derived from pass@k rate + wrong-answer distribution:
- `solved`: pass_at_k == 1.0 (too easy — every rollout correct)
- `band`: BAND_LO ≤ pass_at_k ≤ BAND_HI (0.125 ≤ p ≤ 0.75 in icepick)
- `misdirection`: p < BAND_LO, scattered wrong answers
- `collapse`: p < BAND_LO, single wrong attractor dominates
- `drop`: junk truth or unscoreable

**Band constants**: `src/icepick/contracts/records.py::BAND_LO=0.125, BAND_HI=0.75`.
ModelBreaker uses `(0.125, 0.875)` — records at `p ∈ (0.75, 0.875]` label
`solved` here but `band` in MB. Documented in `pass_at_k/config.py`.

**Backend policy** (three-layer kill switch):
1. Default backend is `qwen_http` (local, free).
2. Selecting `anthropic` or `openai` in production requires BOTH `--allow-live-calls` AND `--i-understand-paid-backend-is-off-policy`.
3. Even with both flags, paid backends have NO default model — `--model X` is mandatory.

Any of the three tripwires returns `E_CONFIG` from the CLI with a clear
error message.

**Writes** (`<output-dir>/`):
```
pass_at_k_input.jsonl            uid-injected input snapshot
pass_at_k.jsonl                  ← labeled output records
pass_at_k_manifest.json          config echo + counts + token_usage
_progress/                       restartability infra
├── records_done.jsonl           append-only: labeled row per finished record
├── rollouts.jsonl               append-only: raw rollouts per (uid, sample_idx)
├── llm_cache.jsonl              optional cache
└── INCOMPLETE                   marker while run is unfinished
```

**Ctrl-C boundary**: commits per completed record. In-flight record's k
rollouts discarded on interrupt. On resume, in-flight record re-runs
from scratch (no per-rollout durability).

**Exit code**: 0 on completion; 1 on interrupt (`"interrupted": true` in JSON summary).

### `processing pipeline`

Chains groundtruth → wellposed → pass_at_k in one call.

```sh
icepick processing pipeline --mode production \
  --input <records.jsonl> \
  --output-dir <dir> \
  --anthro-key-file <path> --openai-key-file <path> \
  --combo claude:anthropic,codex:openai,... \
  --pipeline-order {classic|solvable-first} \
  # pass@k opt-in (off by default):
  [--enable-pass-at-k] \
  [--pak-backend qwen_http --pak-backend-url URL] \
  [--pak-k 8 --pak-temperature 0.7 --pak-max-tokens 2048] \
  [--pak-i-understand-off-policy]   # required for paid pass@k backends
```

Uses the parallel-fleet wellposed runner, not the cascade. If you need
the cascade, chain manually (see below).

Writes stage-specific dirs under `<output-dir>/`: `groundtruth/`,
`wellposed/`, `pass_at_k/`, plus `pipeline_manifest.json` and
`final_corpus.jsonl` at the top level.

---

## Manual chaining (skipping stages)

Because groundtruth is kill-switched, the typical current controller
flow skips it:

```sh
# 1. Acquire
icepick allocation run --manifest ...           # produces handoff/records.jsonl

# 2. Skip groundtruth (kill-switched)

# 3. Wellposed cascade on the raw handoff
icepick processing wellposed-cascade --mode production \
  --input <handoff> --output-dir <out>/cascade \
  --anthro-key-file ... --openai-key-file ...

# 4. Pass@k on cascade survivors
icepick processing pass_at_k --mode production \
  --input <out>/cascade/final_corpus.jsonl \
  --output-dir <out>/pak \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions
```

Every stage's `<output-dir>/final_corpus.jsonl` (or
`passed_records.jsonl`) is safe to hand-feed into the next.

---

## Kill switches (deliberate cost brakes)

### Groundtruth
- Location: [groundtruth/anthropic_adapter.py](../src/icepick/processing/groundtruth/anthropic_adapter.py) `_build_anthropic_client`
- Effect: client uses literal `"[API key]"` string, 401 on any API call
- Restore: replace the function body with the pre-kill-switch version documented in its own docstring
- Reason: prior run spent $3.89 for zero usable records; Opus + 3 samples + web_search dominated the pipeline cost

### Pass@k paid backends
- Location: [pass_at_k/config.py](../src/icepick/processing/pass_at_k/config.py) `validate()`
- Effect: `--backend anthropic|openai` in production is rejected unless BOTH `--allow-live-calls` AND `--i-understand-paid-backend-is-off-policy` are set, AND `--model` is explicit
- Policy default: `--backend qwen_http`
- Reason: k rollouts × N records × paid tokens dominates other spend; single-flag opt-in was too easy to trigger

---

## Restartability contract

Same command re-invocation resumes cleanly. No `--resume` flag.

Two subsystems own their own checkpoint dirs:

- **Scraper**: [`allocation/scrape/checkpoint.py`](../src/icepick/allocation/scrape/checkpoint.py) — per-paper commits under `_progress/`
- **Pass@k**: [`pass_at_k/checkpoint.py`](../src/icepick/processing/pass_at_k/checkpoint.py) — per-record commits under `_progress/`

**Invariant across both**: after any exit (planned pause, crash,
network-triggered abort, Ctrl-C, machine reboot), disk state is
sufficient to resume without redoing completed items.

**In-flight item on interrupt**: discarded and re-run on next invocation.
- Scraper: in-flight paper's uncached theorem QA calls repeated
- Pass@k: in-flight record's k rollouts repeated

**QA cache**: scraper stashes generator responses in `qa_cache.jsonl`, keyed by SHA1(statement). Resume hits this cache before charging the call budget, so completed generator work is not re-billed. A theorem the generator can't handle is skipped per-theorem and not cached; `QAConfigError` still surfaces as a systemic misconfiguration.

**arXiv cooldown marker**: a 429/503 stamps `_progress/rate_limited_at`. While the marker is fresher than `ICEPICK_ARXIV_COOLDOWN_SECONDS` (default 1200), a resume refuses to hit arXiv and reports the retry time. Any successful Atom or e-print request clears the marker.

**Throttle telemetry is run-lifetime**: every 429/503 is also appended to `_progress/rate_limit_events.jsonl` (timestamp, status, backoff slept) the moment it happens, before any retry or death. Resumes merge this log, so the `rate_limit_*` numbers in the final report and manifest cover every invocation of the run — including one the limiter killed before its first paper commit, which writes no report of its own. Clearing the cooldown marker never touches this log.

**Cascade retries** live in a separate mechanism — [`poser/cascade.py`](../src/icepick/processing/poser/cascade.py) `_run_stage_with_retries`. Transient network errors get retried per-uid within a stage, up to `--max-retries` attempts with exponential backoff. Different granularity than the scraper/pass@k checkpoints — cascade retries transient failures INSIDE a stage's runtime; scraper/pass@k resume ACROSS invocations.

---

## Manifests: what a controller can read

Every stage writes a JSON manifest under `<output-dir>`. Controllers
should read these instead of parsing stdout.

### Universal fields
- `stage`: string name of the stage
- `config`: echo of config (redacted secrets)
- `counts`: dict of verdict/label → count
- `outputs`: paths to output files
- `token_usage.estimated_cost.total_usd`: when cost rates were passed
- `warnings`: string list

### Stage-specific keys worth watching

**Scraper (`realmath_scrape` adapter)**:
```json
{
  "counts": {"papers": N, "candidates": N, "duplicates_dropped": N,
             "quarantined": N, "handoff_records": N},
  "spend": {"arxiv_queries": N, "latex_fetches": N,
            "qa_calls": N,
            "total_calls": N, "call_budget": N,
            "resumed_papers": N,
            "rate_limit_events": N,          // run-lifetime, all invocations
            "rate_limit_backoff_seconds": N, // run-lifetime, all invocations
            "rate_limit_statuses": {"429": N, "503": N},
            "token_usage": {"qa_input_tokens": N, "qa_output_tokens": N,
                            "qa_cache_read_input_tokens": N}}
}
```

**Cascade**:
```json
{
  "overall": {"initial_record_count": N, "after_stage_1": N,
              "after_stage_2": N, "after_stage_3": N,
              "final_corpus_count": N, "dropped_total": N,
              "total_estimated_cost_usd": N, "total_wall_clock_seconds": N},
  "stages": [{"index": 1, "combo": "codex:openai",
              "input_uid_count": N, "survivor_uid_count": N,
              "counts": {"well_posed": N, "ill_posed": N, "defer": N, "error": N},
              "wall_clock_seconds": N, "estimated_cost_usd": N,
              "retry_events": [...]}]
}
```

**Pass@k**:
```json
{
  "counts": {"solved": N, "band": N, "misdirection": N,
             "collapse": N, "drop": N, "pre_labeled": N, "dropped": N},
  "model_calls": N,
  "interrupted": bool,
  "resumed_records": N,
  "token_usage": {"backend": "qwen_http", "input_tokens": N, "output_tokens": N}
}
```

---

## Exit codes

All commands use the same envelope:

- **0**: success
- **1**: recoverable error. Details in stderr as `<E_CODE> <label>: <detail>`. Common codes:
  - `E_CONFIG`: config invariant violated
  - `E_NOT_FOUND`: input file / manifest missing
  - `E_INVALID`: malformed argument or input row
  - `E_NETWORK`: transient network error (safe to retry — checkpoint preserved)
  - `E_NOT_IMPLEMENTED`: stage exists but not built out
- **1 with `interrupted_resumable`**: the scraper's variant when budget or Ctrl-C hits mid-run. JSON summary carries `"status": "interrupted_resumable"` and the resume command in `"next"`. Rerun same command to resume.

Never returns partial success. If exit is 0, output is complete.

---

## Testing conventions

- **Unit tests**: fake all subprocesses / API calls. Cover config validation, retry logic, checkpoint round-trips, per-stage semantics.
- **Integration tests**: `tests/integration/` — offline tests use fake adapters; live tests skip unless env creds AND binaries are present.
- **Live tests**: gated on `ANTHROPIC_API_KEY` and (for poser) `claude-poser` binary on PATH. Run: `pytest tests/integration --run-live` (or export credentials to auto-enable).
- **Fake-adapter pattern**: every stage runner accepts an injectable `adapter_overrides` (or equivalent) so tests substitute deterministic fakes.

Run full suite: `pytest` (~420 tests, ~1s). Live tests skipped by default.

---

## Known limitations / open items

- **Groundtruth kill-switched**: 401 on any invocation. Reason and restore procedure in §Kill switches.
- **Scraper `qa` mode requires network + Anthropic key** (Sonnet Q+A). `abstract` and `latex` modes work offline (no LLM).
- **Pass@k `qwen_http` requires a local endpoint** (LM Studio / vLLM / Ollama). Model default `qwen/qwen3-8b`.
- **`processing pipeline` uses parallel fleet wellposed, not the cascade.** If you want the cascade in the chain, run stages manually.
- **`pipeline` command does not include the scraper** — scrape output is fed to `pipeline` via `--input`. Two separate operator steps.
- **Band constant mismatch with MB**: icepick `[0.125, 0.75]` vs MB `[0.125, 0.875]`. Documented in `pass_at_k/config.py`.
- **QA prompt caching is inert**: the ~480-token QA prompt is below Anthropic's 2048-token minimum cacheable prefix, so `cache_control` reads 0. The block is sent anyway (forward-compatible).

---

## For controllers: recommended access pattern

1. Fire the subcommand with `--output-dir <run_dir>`.
2. Wait for exit. Do not parse stdout; use manifest.
3. On exit 0: read `<run_dir>/*_manifest.json` for the final counts + paths.
4. On exit 1 with `"status": "interrupted_resumable"`: read manifest for progress state. Rerun same command to resume. Do NOT delete `_progress/`.
5. Chain to next stage: use the manifest's `outputs.handoff` / `passed_records_path` / `final_corpus_path` as the next stage's `--input`.

Idempotence contract: firing the same command twice against the same
`--output-dir` is safe. The second invocation resumes from wherever the
first left off, hits cache for completed items, and never re-bills for
work already done.

---

## Session-specific context (as of this doc's writing)

- Groundtruth is kill-switched; pipelines skip it (`allocation → cascade → pass@k`).
- Pass@k policy default is `qwen_http` (via LM Studio at 127.0.0.1:1234).
- Latest realmath scraper uses single-stage Sonnet Q+A (see [`allocation/scrape/realmath.py`](../src/icepick/allocation/scrape/realmath.py)); the former Haiku pre-filter gate was dropped (zero selectivity).
- Cascade is 3-stage: `codex:openai → codex:anthropic → claude:openai`.
- Full test suite: 420 pass, 3 live-only skipped.
- Empirical cost for 25-record end-to-end pipeline: ~$0.32 with Qwen local; ~$5.60 with all paid backends (~94% cheaper).

For handoff context on WHAT to do next, see the most recent SESSION HANDOFF
message in the working conversation, not this file.

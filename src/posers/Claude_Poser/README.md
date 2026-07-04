# Claude_Poser

Isolated **c01 well-posedness** check, extracted from the ModelBreaker
processing pipeline. The rest of that pipeline (ingest of raw harvest output,
the other deterministic checks, routing, triage, confirmation, soundness
escalation, merge-back) is presumed to exist in its own repo. This module
owns only the well-posedness verdict and its numeric score.

## I/O contract

**Input** — one or more JSONL files of post-pass@k records. Each line is a
problem record with at least:

```jsonc
{
  "source": "realmath",                 // any string; treated as data
  "provenance": "extracted",            // computed | extracted | manual | external | unknown
  "statement": "Using Theorem 3.2 ...", // the problem text the model saw
  "pass_at_k": 0.1,                     // optional; passed through
  "n_correct": 2, "n_wrong": 17, "n_degenerate": 1,
  // ... any other fields pass through unchanged in `raw`
}
```

`uid` is computed deterministically from `(source, statement)` if not supplied
so records join across runs and input order changes. Records with
`provenance: "manual"` and `truth_policy: "trusted"` are treated like
computed-provenance records.

**Output** — `.json` or `.csv`, chosen by the `--output` extension.

JSON shape:

```jsonc
{
  "run": {
    "check": "c01_wellposed",
    "processor_mode": "production",
    "counts": {"pass": 5, "flag": 2, "insufficient_context": 0, "defer": 0},
    "parameters": { ... },
    "inputs": ["…/sample_postk.jsonl"]
  },
  "records": [
    {
      "uid": "…", "rid": 0, "source": "realmath", "provenance": "extracted",
      "tier": "code",        // code | judge
      "status": "pass",      // pass | flag | insufficient_context | defer
      "wellposed_score": 1.0, // 0.0..1.0 — see scoring below
      "code_hits": [],
      "judge": null
    }
  ]
}
```

CSV writes one row per record (`uid`, `source`, `tier`, `wellposed_status`,
`wellposed_score`, judge vote counts, code-hit count) and a sidecar
`<output>.summary.json` so the run metadata is not lost.

## Score

`wellposed_score ∈ [0, 1]`. Outcome depends on both the provenance tier
and the `--extracted-judge-policy` setting (default: `always`).

| situation                                                    | score      | status                 |
|--------------------------------------------------------------|------------|------------------------|
| computed/trusted provenance                                  | 1.0        | pass                   |
| extracted + judge disabled + scanner clean                   | 1.0        | pass                   |
| extracted + judge disabled + scanner hit                     | 0.0        | flag                   |
| extracted + judge enabled + policy=always                    | see judge  | see judge              |
| extracted + judge enabled + policy=on_scanner_hit + clean    | 1.0        | pass (LEGACY)          |
| extracted + judge enabled + policy=on_scanner_hit + hit      | see judge  | see judge              |
| judge majority pass (k of 3)                                 | k / 3      | pass                   |
| judge majority flag (k of 3)                                 | k / 3      | flag                   |
| judge majority `insufficient_context`                        | 0.0        | insufficient_context   |
| judge unreachable / no majority                              | 0.5        | defer                  |

**Under the default `always` policy, every extracted-provenance record with
`--judge` on reaches the judge regardless of scanner output.** The scanner
provides evidence (`code_hits`), not a gate. This is the fix for the
prior-version bug where scanner false-negatives — common on arXiv text
with paper-specific notation but no textual cross-refs — became silent
full-pass verdicts.

`defer` is a feature, not a bug: judge-only uncertainty must not reject a
record (the parent pipeline's policy). It is distinct from
`insufficient_context`, which is a confirmed 0.0 verdict.

## Tiers (c01)

1. **Provenance trust.** Computed and trusted-manual records pass without
   scanning — the parent skeleton mandates trusting self-contained truths.
2. **Code-tier dangling-reference scan.** Regex detector for named
   cross-references (`Theorem 3.2`, `Equation (4.7)`), LaTeX-macro refs
   (`\ref{}`, `\cref{}`, `\eqref{}`), anaphoric references
   (`as defined above`), prior-item references (`the previous problem`),
   and meta-source references (`the main result of the paper`). Empirically
   weak on arXiv text where failures are semantic (undefined notation) —
   this is why the judge tier is no longer scanner-gated by default.
3. **Judge tier.** Three independent samples from the configured provider
   (Anthropic or OpenAI), two-of-three uphold, cached by
   `(provider, model, prompt, sample_id)`. The judge prompt explicitly asks
   about undefined notation and paper-specific symbols, not just numbered
   cross-references. Each sample returns
   `{verdict, insufficient_context, reason}`. `insufficient_context`
   majority is a confirmed 0.0 decision.

## Providers

Pick a backend at the CLI with `--provider {anthropic,openai}`. Both
providers run the **same** prompt, the same 3-sample / 2-of-3 corroboration,
the same `insufficient_context` short-circuit, and the same defer-on-error
behaviour — only the API surface differs.

| Provider    | Transport          | Key env var          | Model env var      | Default model              |
|-------------|--------------------|----------------------|--------------------|----------------------------|
| `anthropic` | `anthropic` SDK    | `ANTHROPIC_API_KEY`  | `ANTHROPIC_MODEL`  | `claude-haiku-4-5-20251001`|
| `openai`    | stdlib `urllib`    | `OPENAI_API_KEY`     | `OPENAI_MODEL`     | `gpt-4o-mini`              |

The OpenAI backend honours `OPENAI_BASE_URL` (or `--openai-base-url`), so
any **OpenAI-compatible server** works as a drop-in: LM Studio, Ollama,
vLLM, Together, Groq, etc. Just point the URL at the server's `/v1`
endpoint and supply any token the server expects in `OPENAI_API_KEY`.

The judge cache key includes the provider and model, so swapping providers
or models never reuses stale replies.

## Modes

- `--mode production` (default): real API calls when `--judge` is set,
  using whichever provider is selected.
- `--mode flow_testing --calibration-sheet PATH`: every judge call is
  replayed from `PATH`. No external calls regardless of provider. Outputs
  are tagged `calibration_replay: true`.

## CLI

```bash
# Code-only run (no API calls):
claude-poser score \
  --input tests/fixtures/sample_postk.jsonl \
  --output out/scores.json

# CSV output:
claude-poser score \
  --input tests/fixtures/sample_postk.jsonl \
  --output out/scores.csv

# Enable the Anthropic judge with a segregated key file (recommended):
claude-poser score \
  --input data.jsonl \
  --output out/scores.json \
  --judge --judge-cache out/judge_cache.jsonl \
  --provider anthropic --anthropic-key-file ../anthro_key.env

# Same run, but swap to OpenAI — opens only the OpenAI key file:
claude-poser score \
  --input data.jsonl \
  --output out/scores.json \
  --judge --judge-cache out/judge_cache.jsonl \
  --provider openai --openai-key-file ../openai_key.env

# Point the OpenAI backend at a local LM Studio / Ollama / vLLM server:
claude-poser score \
  --input data.jsonl \
  --output out/scores.json \
  --judge --provider openai \
  --openai-base-url http://127.0.0.1:1234/v1 \
  --judge-model qwen/qwen3-8b \
  --openai-key-file ../openai_key.env

# Flow-testing mode (no real calls, replay from calibration sheet):
claude-poser score \
  --input data.jsonl \
  --output out/flow/scores.json \
  --mode flow_testing \
  --calibration-sheet tests/calibration/cheat_sheet.jsonl \
  --judge

# Smoke test the wiring:
claude-poser self-test
```

## Secrets — segregated by provider

The repo never stores keys. `config.py` reads `ANTHROPIC_API_KEY`,
`ANTHROPIC_MODEL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL`
from `os.environ` only. Files live one directory above the repo so they
can't be staged accidentally.

**Canonical layout** (matches what's in `~/Desktop/helloworld/`):

```
~/Desktop/helloworld/
├── Claude_Poser/        ← this repo
├── anthro_key.env       ← Anthropic creds only
└── openai_key.env       ← OpenAI creds only
```

**Segregation guarantee:** with `--anthropic-key-file` / `--openai-key-file`,
the CLI opens **only** the file matching `--provider`. The other file is
never read, so the other provider's credentials never enter the process's
`os.environ`. If you pass both flags but the providers don't match, the
mismatched one is skipped with a stderr notice — no silent fallback.

```bash
# Anthropic run — only anthro_key.env is opened
claude-poser score ... --judge \
  --provider anthropic --anthropic-key-file ../anthro_key.env

# OpenAI run — only openai_key.env is opened
claude-poser score ... --judge \
  --provider openai --openai-key-file ../openai_key.env

# Pass both — segregation still holds, the mismatched one is ignored
claude-poser score ... --judge \
  --provider anthropic \
  --anthropic-key-file ../anthro_key.env \
  --openai-key-file ../openai_key.env     # ignored under provider=anthropic
```

**General escape hatch:** `--env-file PATH` is **repeatable** and loads any
KEY=VALUE file regardless of provider. Use it for non-secret config (model
overrides, base URLs) or when you intentionally want both providers' env
loaded for back-to-back runs.

```bash
claude-poser score ... --env-file ../anthro_key.env --env-file ../openai_key.env
```

**Env-file rules** (same loader for all three flags): `KEY=VALUE` lines,
`#` comments, optional `export ` prefix, surrounding quotes stripped.
Existing shell env wins — files never override an explicit export. No
`${VAR}` interpolation, no shell execution. Malformed lines are skipped
with a stderr note.

**Alternative loading patterns** if you prefer not to use the flags:

```bash
# Shell-source before run:
set -a; source ../anthro_key.env; set +a
claude-poser score ... --judge --provider anthropic

# Inline injection per call:
env $(grep -v '^#' ../openai_key.env | xargs) \
  claude-poser score ... --judge --provider openai
```

The repo's `.gitignore` blocks `*.env`, `.env*`, `*.key`, `*.pem`, and
`secrets/` — a stray secret dropped inside the repo cannot be committed.
Even so, prefer keeping key files outside the repo tree.

If `--judge` is enabled but no key is found in env after loading, the run
still completes (judge calls return `defer`) and a warning points at the
exact flag/env-var to set.

## Install

```bash
pip install -e .                # core (OpenAI provider works out of the box, uses urllib)
pip install -e '.[anthropic]'   # adds the anthropic SDK for --provider anthropic
pip install -e '.[dev]'         # adds pytest
```

## Tests

```bash
pytest -q
```

## Out of scope

This repo deliberately does **not** include the other deterministic checks
(c02 ground-truth, c04 leakage, c06 duplication, c09 robustness), routing,
triage lanes, confirmation resampling, soundness escalation, merge-back,
acquisition adapters, host-role separation, or the manager chat console.
Those live in the parent processing repo and consume this module's score
file (or invoke `check_records()` directly) as their c01 input.

# Operator Runbook

A first-run walkthrough. Get from "I have records" to "I have a final
corpus" in three commands.

For the architectural overview, see [`plan.md`](plan.md). This doc is the
operations side: what to run, what each command writes, what to check.

---

## Prerequisites

- Python 3.10+
- icepick installed (`pip install -e .` from the repo root)
- For `flow_testing` runs: nothing else.
- For `production` runs:
  - `anthropic` SDK installed (`pip install -e .[judge]`)
  - An `anthro_key.env` file outside the repo (see [Secrets](#secrets))
  - `claude-poser` and/or `codex-poser` binaries on PATH if you want the
    poser stage (see [poser/ vendored projects](../src/icepick/src/posers/))

---

## Five-minute first run (flow_testing, no API calls)

A scratch run that exercises the full pipeline against fixture data. No
secrets, no network, no cost.

```sh
# Scratch workspace
cd /tmp && rm -rf demo && mkdir demo && cd demo

# Seed input — three records, one of which is generated (will be dropped)
cat > input.jsonl <<'EOF'
{"source": "test", "statement": "Theorem 1: addition is commutative.", "arxiv_id": "2403.11111", "provenance": "extracted"}
{"source": "test", "statement": "Theorem 2: multiplication is associative.", "arxiv_id": "2403.22222", "provenance": "extracted"}
{"source": "test", "statement": "Generated record - should be dropped.", "family": "calc", "provenance": "computed"}
EOF

# Seed calibration sheet — what flow_testing replays instead of API calls
cat > calibration.jsonl <<'EOF'
{"arxiv_id": "2403.11111", "verdict_status": "published", "venue": "Test Journal", "publication_year": 2024, "indexed_in": ["Scopus"], "judge_votes": ["published", "published", "published"], "reasoning": "fake", "confidence": "high"}
{"arxiv_id": "2403.22222", "verdict_status": "unpublished", "judge_votes": ["unpublished", "unpublished", "unpublished"], "reasoning": "fake", "confidence": "high"}
EOF

# Run the groundtruth stage
icepick processing groundtruth \
  --mode flow_testing \
  --calibration-sheet calibration.jsonl \
  --input input.jsonl \
  --output-dir out/groundtruth
```

Expected stdout: a JSON summary with `counts: {published: 1, unpublished: 1, discarded: 1}` and four output paths.

Check:

```sh
cat out/groundtruth/published.jsonl   # only the published record
cat out/groundtruth/discarded.jsonl   # the unpublished + generated records
cat out/groundtruth/run_manifest.json | python -m json.tool | head -30
```

That's the basic shape. Generated records were dropped, unpublished
papers were filtered, published papers flowed through. The next stage
(poser) would consume `published.jsonl` and apply the well-posedness
gate.

---

## The real production flow

```sh
# 1. Mount: scan a directory drop, write canonical handoff JSONL
icepick allocation mount \
  --path /mnt/incoming/customer_batch_001 \
  --source customer_2026Q2 \
  --provenance external \
  --output-dir out/intake \
  --column statement=question --column answer=gold --column arxiv_id=arxiv

# 2. Pipeline: groundtruth → poser → final corpus
icepick processing pipeline \
  --mode production \
  --input out/intake/runs/<TIMESTAMP>/handoff/records.jsonl \
  --output-dir out \
  --combo claude:anthropic \
  --anthro-key-file ../anthro_key.env \
  --gt-cache-path out/groundtruth/paper_cache.jsonl
```

Result: `out/final_corpus.jsonl` is your deployment-ready filtered corpus.

---

## Stages, individually

If you don't want the bundled `pipeline` command, run each stage and
chain the outputs by hand.

### `allocation mount` — get records into icepick

**Use when:** you have a directory drop (JSONL / JSON arrays / CSV / TSV
/ mixed) and need canonical JSONL the rest of the pipeline can consume.

```sh
icepick allocation mount \
  --path /mnt/incoming/drop \
  --source customer_2026Q2 \
  --provenance external \
  --output-dir out/intake \
  --column statement=question --column arxiv_id=arxiv
```

| Flag | Purpose |
|---|---|
| `--path` | File or directory to scan. Never modified. |
| `--source` | Source-name stamp written onto every output record. |
| `--provenance` | `manual` / `external` / `extracted` — stamped onto records. |
| `--truth-policy` | `trusted` / `extracted` / `unknown` — default `unknown`. |
| `--column k=v` | CSV/TSV column projection. Repeatable. Required for CSV/TSV. |
| `--family` | Optional family stamp. |
| `--requested-by` | Recorded on the manifest. Default `cli`. |

**Writes:**
```
<output-dir>/runs/<run_id>/
  manifest.json            ← auto-approved (mounts spend no calls)
  handoff/records.jsonl    ← feed this to the pipeline
```

**Check:**
```sh
icepick allocation validate-manifest --manifest out/intake/runs/<id>/manifest.json
```

### `processing groundtruth` — publication-status filter

**Use when:** you want to drop records whose source arXiv paper isn't
peer-reviewed and indexed in a reputable database. Run before OR after
pass@k — the module is position-agnostic.

```sh
icepick processing groundtruth \
  --mode production \
  --input out/intake/runs/<id>/handoff/records.jsonl \
  --output-dir out/groundtruth \
  --anthro-key-file ../anthro_key.env \
  --cache-path out/groundtruth/paper_cache.jsonl \
  --judge-samples 3 --judge-uphold 2 \
  --cost-per-input-mtok 15 --cost-per-output-mtok 75    # optional cost estimate
```

**Writes:**
```
out/groundtruth/
  groundtruth_input.jsonl  ← uid-injected copy of your input
  verdicts.jsonl           ← one row per input record (all statuses)
  published.jsonl          ← the records that passed; downstream input
  discarded.jsonl          ← dropped before lookup or denied by judges
  run_manifest.json        ← config echo + counts + token_usage + cost
```

**Verdict statuses:**

| Status | Meaning |
|---|---|
| `published` | Peer-reviewed AND indexed. **Passes.** |
| `unpublished` | Explicit evidence of preprint-only. |
| `defer` | Judges couldn't agree. Not passed (operator review). |
| `error` | API failure on every sample. |
| `discarded` | Dropped pre-lookup (generated provenance or no arxiv_id). |

**Cost rollup:** if you set `--cost-per-input-mtok` and
`--cost-per-output-mtok`, `run_manifest.json` carries a
`token_usage.estimated_cost` block. Marked `is_estimate: true`. Cache
hits are excluded (no API call was made for them).

### `processing wellposed` — the gate

**Use when:** you want to filter records whose statements are
ill-posed. This is the gate — records that pass are the final corpus.

```sh
icepick processing wellposed \
  --mode production \
  --combo claude:anthropic \
  --input out/groundtruth/published.jsonl \
  --output-dir out/wellposed \
  --anthro-key-file ../anthro_key.env \
  --cost-per-input-mtok 15 --cost-per-output-mtok 75    # optional cost estimate
```

**Combos:** `claude:anthropic` / `claude:openai` / `codex:anthropic` /
`codex:openai`. Pass `--combo` repeatedly (or `--combo all`) to run a
fleet in parallel; combine policies with `--comparison-policy
intersect|union|majority|prefer:<combo>`.

**Writes:**
```
out/wellposed/
  poser_input.jsonl              ← uid-injected input
  <combo>_input.jsonl            ← per-combo invocation input
  <combo>_verdicts.json          ← raw poser output
  <combo>_normalised.jsonl       ← canonical verdicts (one per combo)
  comparison.jsonl               ← only when fleet > 1
  comparison_report.md           ← only when fleet > 1
  combined_<policy>.jsonl        ← only when fleet > 1
  passed_records.jsonl           ← FINAL CORPUS (the records that passed)
  run_manifest.json              ← includes token_usage.{per_combo, fleet_totals}
                                  + estimated_cost when rates are set
```

### `processing pipeline` — both stages in one command

Convenience wrapper. Threads `published.jsonl` from groundtruth into
the poser, copies the poser's `passed_records.jsonl` to
`<output-dir>/final_corpus.jsonl`.

```sh
icepick processing pipeline \
  --mode production \
  --input out/intake/runs/<id>/handoff/records.jsonl \
  --output-dir out \
  --combo claude:anthropic \
  --anthro-key-file ../anthro_key.env \
  --gt-cache-path out/groundtruth/paper_cache.jsonl \
  --gt-cost-per-input-mtok 15 --gt-cost-per-output-mtok 75 \
  --poser-cost-per-input-mtok 15 --poser-cost-per-output-mtok 75
```

**Writes:**
```
out/
  groundtruth/...           ← stage 1 outputs
  wellposed/...             ← stage 2 outputs
  final_corpus.jsonl        ← the records that passed both stages
  pipeline_manifest.json    ← top-level manifest pointing at both stages
```

---

## Secrets

Never embedded in the repo. Two provider-segregated env files kept
**outside** version control:

```sh
# anthro_key.env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-7

# openai_key.env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

icepick reads the right one per stage / combo:

| Caller | Reads |
|---|---|
| `processing groundtruth` | `anthro_key.env` (Anthropic web_search judge) |
| `processing wellposed --combo *:anthropic` | `anthro_key.env` |
| `processing wellposed --combo *:openai` | `openai_key.env` |

The posers refuse to load the other provider's key file by design, so
credentials stay segregated even when all four combos run in parallel.

`.gitignore` blocks `key.env`, `anthro_key.env`, `openai_key.env`,
`*.env`, `*.key`, `*.pem`, and `secrets/`.

---

## Flow-testing mode

Use when:
- Running in CI / pre-commit (no API budget)
- Validating manifest / file-layout changes
- Smoke-testing the orchestration without spending tokens

```sh
icepick processing groundtruth \
  --mode flow_testing \
  --calibration-sheet path/to/calibration.jsonl \
  --input input.jsonl \
  --output-dir out
```

**Calibration sheet format** — one JSONL line per arxiv_id you'll look
up:

```json
{"arxiv_id": "2403.11111", "verdict_status": "published", "venue": "X", "judge_votes": ["published"]*3, "reasoning": "fake", "confidence": "high"}
```

Missing arxiv IDs route to `defer` (not an error — but operators
should add them to the sheet for deterministic behavior).

Outputs from flow_testing are stamped `calibration_replay: true` in the
manifest and must not enter production downstream.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `E_CONFIG ... processor_mode is required` | `--mode` omitted | Add `--mode production` or `--mode flow_testing` |
| `E_CONFIG ... requires calibration_sheet` | flow_testing without sheet | Add `--calibration-sheet path/to/sheet.jsonl` |
| `E_CONFIG ... anthropic_key_file is required` | Production without key file | Add `--anthro-key-file ../anthro_key.env` |
| `E_CONFIG ... gpt poser does not support --judge in flow_testing mode` | codex+flow_testing+judge combination | Either switch to production mode, or pass `--no-judge` |
| `E_NOT_FOUND ... input not found` | Bad `--input` path | Check the run_id in `out/intake/runs/` |
| `E_INVALID ... no_arxiv_id` discarded count high | Records missing arxiv_id field | Add `--column arxiv_id=<source-column>` to the mount, or pre-populate the field |
| `unique_papers_looked_up: 0` in manifest | Every record was discarded pre-lookup | Probably `provenance=computed` (generated) on everything, or no arxiv_id |
| Poser binary not found | `claude-poser` / `codex-poser` not on PATH | `pip install -e src/posers/Claude_Poser` (and Codex), or pass `--claude-cli /absolute/path` |
| `defer` count is high | Judges genuinely uncertain | Acceptable. Check `verdicts.jsonl` to see the reasoning. Don't treat `defer` as `unpublished`. |
| Anthropic API rate-limited | Too many concurrent paper lookups | Lower `--max-concurrent` (default 8); the SDK auto-retries with backoff |

---

## Common operator tasks

### Re-run groundtruth without re-querying papers

Point `--cache-path` at the same JSONL on the second run. Cache hits
skip the API call entirely (and won't appear in `token_usage`).

### Run only one stage, manually inspect, then continue

```sh
# Stage 1
icepick processing groundtruth --mode production --input ... --output-dir out/gt ...
# Inspect, optionally edit out/gt/published.jsonl to drop more records
# Stage 2
icepick processing wellposed --mode production --input out/gt/published.jsonl ...
```

### Bring a CSV drop into the pipeline

```sh
icepick allocation mount \
  --path /mnt/incoming/csv_drop.csv \
  --source partner_2026Q2 \
  --provenance external \
  --output-dir out/intake \
  --column statement=question \
  --column answer=correct_answer \
  --column arxiv_id=source_url
```

Then `--input out/intake/runs/<id>/handoff/records.jsonl` into the
pipeline.

### Estimate cost before a big run

Run a small sample first with `--cost-per-input-mtok` /
`--cost-per-output-mtok` set, then look at `run_manifest.json` →
`token_usage.estimated_cost.total_usd`. Multiply by the ratio of full
corpus size to sample size for an upper bound (cache hits will push the
real cost down).

### Validate a mount before running the pipeline

```sh
icepick allocation validate-manifest --manifest out/intake/runs/<id>/manifest.json
head out/intake/runs/<id>/handoff/records.jsonl
```

---

## What this repo does NOT do

- Generate synthetic data. icepick acquires real source records (e.g.
  RealMath scraping via the `allocation` adapters) or takes mounted drops;
  it does not fabricate problems.
- Run pass@k. That happens upstream (e.g. in `ModelBreaker`).
- Mutate the source mount path. Mounts are read-only.
- Auto-approve acquisition. Manual mounts auto-approve (no calls
  spent); other source types would require explicit human approval.
- Pool generated and extracted records. Records with
  `provenance=computed` are discarded at the groundtruth stage.

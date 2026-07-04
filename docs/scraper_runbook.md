# In-House Scraper Runbook

How to run IcePick's in-house RealMath scraper — arXiv → verifiable problem
records → processing pipeline. Self-contained: everything you need is here.

The scraper lives behind the `allocation` subsystem
(`src/icepick/allocation/scrape/realmath.py`, driven by the
`realmath_scrape` adapter). `processing` is unchanged and still consumes only
`handoff/records.jsonl`.

## The flow

```
allocation plan     → propose a scrape (pure, no calls) → proposed_plan.json
allocation approve  → human authorises it               → manifest.json  (approved)
allocation run      → scrape arXiv in-house             → handoff/records.jsonl
processing pipeline → groundtruth → poser               → final_corpus.jsonl
```

`plan` and `approve` are the two-step human gate: planning never scrapes, and a
production scrape only runs from a manifest a person approved with a call
budget. The plan estimate is extraction-aware: `qa` budgets arXiv queries,
e-print source fetches, and QA-generation calls.

## Prerequisites

- Python 3.10+, `pip install -e .` from the repo root (gives the `icepick` CLI).
- **flow_testing** (dry run): nothing else — no network, no keys.
- **production `abstract` / `latex`**: outbound network to `export.arxiv.org`.
- **production `qa`**: network, the Anthropic SDK (`pip install -e .[judge]`),
  and an Anthropic key supplied **without embedding it** — set the proxy
  variable `ANTHROPIC_KEY_FILE` to the path of a gitignored `anthro_key.env`
  (`ANTHROPIC_API_KEY=...`), or export `ANTHROPIC_API_KEY` directly. The key
  never enters the repo, a command line, or a log.

## 1. Dry run first (flow_testing — no network, no key)

Proves the whole plumbing against the checked-in fixture.

```sh
OUT=out/intake

# plan (records the scrape window; writes proposed_plan.json under $OUT/plans/)
icepick allocation plan --source-type realmath_scrape --source demo \
  --target-count 5 --category math.AP --output-dir "$OUT"
# → note the "plan_path" printed in the JSON summary

# approve for flow_testing (replays the fixture instead of scraping)
icepick allocation approve --plan <plan_path> \
  --mode flow_testing --approved-by "$USER" \
  --calibration-sheet tests/fixtures/realmath/qa_candidates.jsonl \
  --output-dir "$OUT"
# → note the "manifest" path printed

# run (replays the fixture → handoff/records.jsonl)
icepick allocation run --manifest <manifest_path>
```

Expect `calibration_replay: true` and `handoff_records > 0`. Nothing hit the
network.

## 2. Production scrape (real arXiv)

PDE example — `math.AP`, primary-only, theorem-level extraction:

```sh
OUT=out/intake

icepick allocation plan --source-type realmath_scrape --source pde_2026Q2 \
  --target-count 200 --category math.AP --primary-only --extraction latex \
  --max-per-paper 3 --year 2026 --month 4 --family pde --output-dir "$OUT"
# --max-per-paper caps how many candidates one paper contributes: a
# theorem-dense paper (60+ lemmas) would otherwise fill --target-count by
# itself. Set it for a diverse corpus; omit it to take everything per paper.

# Approve WITH a call budget that covers the plan's estimated_calls.
# (approve refuses production if --call-budget is missing or below the estimate.)
icepick allocation approve --plan <plan_path> \
  --mode production --approved-by "$USER" --call-budget 4000 --output-dir "$OUT"

# For --extraction qa, point the proxy variable at your gitignored key file
# (the key stays out of the repo / commands / logs):
#   export ANTHROPIC_KEY_FILE=../anthro_key.env
icepick allocation run --manifest <manifest_path>

# Feed the handoff into processing (unchanged pipeline)
icepick processing pipeline --mode production \
  --input "$OUT"/runs/<run_id>/handoff/records.jsonl --output-dir out \
  --combo claude:anthropic --anthro-key-file ../anthro_key.env
```

## Extraction depths (`--extraction`)

| Mode | Candidate statement | Answer | Needs a key |
|---|---|---|---|
| `abstract` (default) | paper abstract | — | no |
| `latex` | theorem/lemma statement from the e-print source | `\boxed{…}` when stated | no |
| `qa` | LLM-reformulated question | LLM-extracted, sympy-verified (number/tuple/expr) | **yes** |

`qa` answers are the paper's *stated* result (extract-only prompt), so records
stay `provenance=extracted` and survive the groundtruth stage.

## Run outputs (`$OUT/runs/<run_id>/`)

```
manifest.json                 the approved manifest
handoff/records.jsonl         ← the ONLY file processing consumes
raw/papers.jsonl              unique paper pool
raw/extracted_candidates.jsonl
raw/qa_candidates.jsonl       raw scraper rows (audit)
raw/quarantined.jsonl         only when candidates were dropped
reports/source_report.md      counts, warnings, drops, handoff path, acquisition spend
_progress/                    checkpoint store (production scrapes)
  papers_done.jsonl           per-paper commits — the resume ledger
  candidates.jsonl            durable raw candidates
  qa_cache.jsonl              cached LLM answers (a resume never re-bills)
  INCOMPLETE                  present only while a run is unfinished
```

## Interrupt & resume (pause, don't die)

Production scrapes checkpoint every finished paper to disk. Ctrl-C, a
crash, or a network death **pauses** the run instead of killing it:

- The run stops cleanly, writes a partial handoff + a report marked
  **INTERRUPTED — resumable**, and the CLI exits **non-zero** so a chained
  pipeline never consumes the partial corpus.
- **To resume, rerun the exact same command**:
  `icepick allocation run --manifest <same path>`. Papers already acquired
  are served from `_progress/` (no refetch); cached QA answers are free
  (no re-billing). At most the one in-flight item is redone.
- A completed run's rerun is idempotent: everything replays from the
  checkpoint, spending nothing.

`_progress/INCOMPLETE` on disk means the last invocation didn't finish —
rerun to complete it.

## Alternative: bring your own records (manual mount)

No scraping — mount a JSONL/CSV drop straight to a handoff:

```sh
icepick allocation mount --path /path/to/drop.jsonl --source my_source \
  --provenance extracted --family pde --output-dir out/intake
# → out/intake/runs/<run_id>/handoff/records.jsonl, then processing pipeline as above
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `E_CONFIG ... requires --call-budget` | production approve without a budget | add `--call-budget N` (≥ the plan's `estimated_calls`) |
| `E_CONFIG ... call budget too low` | budget below the plan estimate | raise `--call-budget` |
| `E_INVALID ... ANTHROPIC_API_KEY` | `qa` mode without a key/SDK | `export ANTHROPIC_KEY_FILE=/path/to/anthro_key.env` (or `ANTHROPIC_API_KEY`), and `pip install -e .[judge]` |
| run reports `handoff_records: 0` + "no candidates" | empty window or all cross-lists filtered | widen `--year`/`--month`, or drop `--primary-only`; check the category |
| read timeout / `429` / `503` / `E_NETWORK` from arXiv | rate-limited or overloaded | requests are auto-paced ≥3s apart and retried with `Retry-After`/backoff; if it persists, raise the gap (`export ICEPICK_ARXIV_MIN_INTERVAL=6`), rerun later, or use bulk data (below). Progress is checkpointed — just rerun to resume. |

## Avoiding arXiv throttling

arXiv asks for **≤1 request every 3 seconds from a single connection**. The
scraper enforces this automatically: all Atom queries and e-print fetches are
spaced ≥`ICEPICK_ARXIV_MIN_INTERVAL` seconds apart (default 3), over one reused
connection, with `Retry-After`-aware backoff on 429/503.

- **Still throttled?** Raise the gap: `export ICEPICK_ARXIV_MIN_INTERVAL=6` (or
  more). Pacing costs wall-clock, not correctness — and any run is resumable.
- **Budget the wall-clock:** a `latex` run does ~1 query + 1 e-print fetch per
  paper, so N papers ≈ `(N+1) × interval` seconds minimum. 20 papers ≈ 1 min.
- **For large harvests (thousands of papers)** don't hammer the API — use
  arXiv's bulk channels: the full-text source tarballs on the requester-pays
  **AWS S3 `arxiv` bucket**, the **Kaggle arXiv metadata** dataset, or
  **OAI-PMH** for metadata. (Not yet wired as an IcePick source — a future
  adapter.)

## Verify the build

```sh
pytest                       # full suite
pytest tests/allocation/scrape   # scraper module only
```

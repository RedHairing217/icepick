# Icepick

Portable processing surface for ModelBreaker-style problem records.

Icepick is the second repo in the ModelBreaker line. The original `ModelBreaker`
repo holds the experiment history, dashboards, and source reports. Icepick is
the **portable processing surface with in-house acquisition**: it scrapes and
harvests its own source records (RealMath-style, behind the `allocation`
subsystem), ingests them, and runs the poser fleet as the final gate.

> **Agents** (Claude Code, Codex, or any other harness): read
> [`AGENTS.md`](AGENTS.md) before touching anything — it is the canonical
> operating brief (safety invariants, budget gates, holds, test baselines).

## Pipeline

```
ingest  →  poser (the gate)  →  final corpus
```

Two working stages. No separate "gate" stage — **the poser fleet IS
the gate**. See `docs/plan.md` for what's deliberately out of scope, and
[`docs/operator.md`](docs/operator.md) for the runbook (five-minute first
run, stage-by-stage, troubleshooting, cost estimation).

To acquire records by scraping arXiv in-house (`plan → approve → run`), see
[`docs/scraper_runbook.md`](docs/scraper_runbook.md).

- **`ingest`** — load JSONL into normalised `ProblemRecord` (stable
  `uid` content hash, provenance, derived label). icepick does not
  process generated records — they're discarded with an explicit reason.
- **`poser`** — well-posedness gate via a fleet of `(build, provider)`
  combos: `claude` or `codex` as the build, `anthropic` or `openai` as
  the judge backend (four legal combinations). Any subset runs in
  parallel; combine policies are `intersect` / `union` / `majority` /
  `prefer:<combo>`. Records that pass are the final corpus.

**Removed — the `groundtruth` publication-status check.** Earlier
revisions filtered records through an Anthropic web_search judge that
required the source paper be peer-reviewed AND indexed in a reputable
bibliographic database (Scopus, Web of Science, DBLP, MathSciNet, PubMed,
IEEE Xplore, ACM DL). The check was removed from the pipeline because the
corpus targets are primarily **pre-published arXiv preprints** —
current-cycle submissions that are, by construction, not yet peer-reviewed
or indexed anywhere — so the check would have rejected the corpus
wholesale while spending on every lookup. The stage's code remains in the
repo (kill-switched) for corpora where publication status is meaningful.

## What it explicitly is not

- Not a dashboard. Not an experiment log.
- Not a research sandbox.
- Not an API. Batch CLIs first; service later if needed.

## Subsystems

Three independent structures, each with its own CLI and tests:

1. **processing/** — the working stages (`ingest`, `poser`). Runs
   without allocation. Runs without chat.
2. **allocation/** — intake planning, manifest approval, budget estimates,
   acquisition adapters (in-house RealMath/arXiv scraper), manual mounts,
   handoff. Working: `mount`, `plan`, `approve`, `run` (see
   [`docs/scraper_runbook.md`](docs/scraper_runbook.md)).
3. **agent/** — optional manager-model chat control. Allowlisted action
   dispatch only. Built last. Never touches subject-host sampling. Stubbed.

Integration is through typed contracts in `src/icepick/contracts/`,
manifests, and output files. No subsystem imports another's private modules.

## Processor modes

- `production` — real LLM, judge, scrape, and confirmation calls.
- `flow_testing` — every call-bearing section is replaced with replay from
  a local calibration/fixture file. Processing stages commonly replay from
  `tests/calibration/cheat_sheet.jsonl`; the RealMath allocation scraper replays
  the manifest's `--calibration-sheet` fixture. Outputs are stamped
  `calibration_replay: true` and must not enter accepted production buckets.

Mode is required on every run and is recorded in every summary, manifest,
verdict, and session log.

## Host roles

- `subject_host` — the model under test (reserved for future pass@k /
  confirmation work; not used by the current three stages).
- `manager_host` — the chat-controller model. Separate base URL, separate
  client, separate logs, separate budgets.

Roles are enforced by type, not convention (`SubjectLLMHost` /
`ManagerLLMHost`). A run fails closed if they share a base URL.

## Install

```
pip install -e .
```

## CLI

Two-command end-to-end (recommended). Mount the input, then run the pipeline:

```
# Mount: scan a dir (JSONL/JSON/CSV/TSV), write canonical handoff JSONL
icepick allocation mount \
    --path /mnt/incoming/customer_batch_001 \
    --source customer_2026Q2 --provenance external \
    --output-dir out/intake \
    --column statement=question --column answer=gold --column arxiv_id=arxiv
# → out/intake/runs/<ts>/handoff/records.jsonl  (+ manifest.json)

# Pipeline: poser → final corpus
icepick processing pipeline --mode production \
    --input out/intake/runs/<ts>/handoff/records.jsonl --output-dir out \
    --combo claude:anthropic \
    --anthro-key-file ../anthro_key.env
# → out/final_corpus.jsonl
```

Or run the stages individually:

```
icepick processing wellposed --combo claude:anthropic --mode production \
    --input out/passatk/records.jsonl --output-dir out/wellposed \
    --anthro-key-file ../anthro_key.env

icepick processing wellposed --combo claude:openai --combo codex:openai \
    --mode production --input out/passatk/records.jsonl --output-dir out/wellposed \
    --openai-key-file ../openai_key.env --comparison-policy union

icepick processing wellposed --combo all --mode production \
    --input out/passatk/records.jsonl --output-dir out/wellposed \
    --anthro-key-file ../anthro_key.env --openai-key-file ../openai_key.env \
    --comparison-policy majority
```

All call-bearing commands require `--mode`. `flow_testing` mode
additionally requires `--calibration-sheet` and produces outputs
marked `calibration_replay: true` that must not enter production.

## Secrets

Secrets are never embedded in the repo. The canonical local convention
is two provider-segregated env files kept **outside** version control,
matching the posers' `--anthropic-key-file` / `--openai-key-file`
contracts:

```
# anthro_key.env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001

# openai_key.env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.5
# optional; only read for reasoning-family models (gpt-5.x, o-series).
# Defaults to high.
OPENAI_REASONING_EFFORT=high
```

icepick reads the right one per combo: a `claude:anthropic` or
`codex:anthropic` combo loads `anthro_key.env`; a `claude:openai` or
`codex:openai` combo loads `openai_key.env`. The posers refuse to read
the other provider's key file by design, so credentials stay segregated
even when all four combos run in parallel.

`.gitignore` blocks `key.env`, `anthro_key.env`, `openai_key.env`,
`*.env`, `*.key`, `*.pem`, and `secrets/` so a stray drop cannot
accidentally enter version control. If you ever suspect a key was
committed, rotate it at the provider console first, then scrub history.

## Non-goals

Dashboards, old sweeps, model-serving probes, experiment logs, and
human-readable master corpus generation stay in the ModelBreaker repo. Icepick
may call them through configured adapters but does not absorb them. Harvesting
and RealMath scraping are in scope — Icepick acquires its own records through
the `allocation` subsystem's adapters.

## Field note — parallel agent sessions (2026-07-26)

Operator's note, recorded verbatim:

> started building to deploy on a remote server (like $0.40/hour, seemed worth it) and something really interesting happened
> I had a couple of sessions running in parallel building different things
> I accidentally launched two sessions on the same task
> They ended up figuring out that I had done this
> And instead of tripping over each other
> They created a document in the repo that they've been using to communicate with each other, they've been negotiating roles and divy'ing up tasks
> I had no idea this was happening until I got back to my computer and saw it

The coordination artifacts referenced: [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md)
(the shared cross-session ledger) and the per-window handoff entries the sessions
exchanged while negotiating write-lane vs. review-seat roles on the duplicated task.

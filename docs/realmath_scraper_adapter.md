# RealMath Scraper Adapter Guide

Implementation guide for IcePick's ModelBreaker-style RealMath scraper.

The adapter is implemented behind `allocation`. This document records the
compatibility rules, public shape, output contract, and maintenance checklist.

## Read First

Use these files as the local source of truth before engineering:

- `README.md`
- `docs/operator.md`
- `docs/plan.md`
- `docs/readability.md`
- `src/icepick/allocation/adapters/realmath_scrape.py`
- `src/icepick/allocation/intake.py`
- `src/icepick/allocation/manifests.py`
- `src/icepick/contracts/manifests.py`
- `src/icepick/contracts/records.py`

For the upstream source pattern, inspect these ModelBreaker files from the
workspace:

- `../ModelBreaker/realmath/scrape_to_budget.py`
- `../ModelBreaker/realmath/harvest_realmath.py`

## Compatibility Rule

IcePick is a processing surface with in-house acquisition. Scraping is in
scope, but it is contained: the RealMath scraper must live behind the
`allocation` subsystem as an acquisition adapter, never inside `processing`.
Its job is to acquire or replay source artifacts, normalise them into canonical
problem records, and write a handoff JSONL file.

The `processing` subsystem must continue to consume only canonical records:

```text
out/intake/runs/<run_id>/handoff/records.jsonl
```

Do not add scraper logic to `processing/`.

## Target Flow

The intended flow is:

```text
allocation plan
  -> human approval / approved manifest
  -> allocation run
  -> realmath_scrape adapter
  -> normalise raw scraper outputs
  -> handoff/records.jsonl
  -> processing pipeline
  -> final corpus
```

The scraper can preserve ModelBreaker-style source stages internally, but those
stages are acquisition details. They are not new IcePick processing stages.

## Adapter Location

Use the implemented adapter:

```text
src/icepick/allocation/adapters/realmath_scrape.py
```

The adapter keeps this public shape:

```python
def plan(request):
    ...

def estimate(plan):
    ...

def run(manifest):
    ...

def normalise(raw_outputs):
    ...
```

## Adapter Responsibilities

### `plan(request)`

Build a proposed acquisition plan without scraping, calling external services,
or writing run outputs.

The plan should capture:

- `source_type = "realmath_scrape"`
- `source_name`
- `target_count`
- requested families or paper classes
- scrape window or paper query constraints
- expected local fixture path for `flow_testing`, if provided
- estimated acquisition calls
- operator notes

### `estimate(plan)`

Estimate the work before approval.

The estimate should describe:

- expected paper count
- expected candidate count
- expected handoff record count
- extraction mode (`abstract`, `latex`, or `qa`)
- external calls or tool invocations by kind
- expected LLM calls for `qa`
- local prerequisites, such as parser dependencies

This function must not perform live scraping.

### `run(manifest)`

Execute only from an approved manifest.

Before doing acquisition work, validate:

- `manifest.source_type == "realmath_scrape"`
- `manifest.processor_mode in {"production", "flow_testing"}`
- `approved_by` and `approved_at` are present for call-bearing or scraping runs
- `call_budget` is present and not exceeded
- the call budget covers extraction-aware estimated calls
- output paths are under the manifest output directory
- source artifacts will not be mutated

In `production`, this function may perform approved scraping or invoke approved
external tooling. `abstract` and `latex` spend arXiv-related calls; `qa` also
spends Anthropic QA-generation calls.

In `flow_testing`, this function must replay local fixture artifacts. It must
not scrape, call APIs, or reach the network.

### `normalise(raw_outputs)`

Convert scraper outputs into IcePick-compatible record dictionaries.

Normalisation should:

- produce one JSON object per candidate problem
- preserve source and provenance fields
- map statements to `statement`
- map extracted answers to `answer` or truth fields already accepted by ingest
- preserve `arxiv_id` when available
- stamp `family = "realmath"` unless a narrower family is explicitly provided
- place source-specific details under `metadata` or `raw`, not new top-level schema
- reject or quarantine records without usable statements
- deduplicate before writing handoff

## ModelBreaker-Style Source Stages

The adapter may mirror these source stages internally:

```text
retrieve papers
extract LaTeX
extract theorem or problem candidates
generate QA/problem records where the approved design allows it
verify answer form
deduplicate paper, title, and candidate pools
normalise to IcePick records
write handoff JSONL
```

Keep these stages inside allocation. The processing pipeline remains:

```text
ingest -> groundtruth -> poser -> final corpus
```

## Output Layout

Each run should write a stable, human-scannable layout:

```text
out/intake/runs/<run_id>/
  manifest.json
  handoff/
    records.jsonl
  raw/
    papers.jsonl
    extracted_candidates.jsonl
    qa_candidates.jsonl
    quarantined.jsonl
  reports/
    source_report.md
```

Only this file is passed to processing:

```text
out/intake/runs/<run_id>/handoff/records.jsonl
```

Raw scraper artifacts and reports are for audit and debugging. They are not
pipeline input. The source report includes counts, drops, warnings, the handoff
path, and acquisition-call spend.

## Canonical Record Shape

Each handoff line should be a JSON object compatible with IcePick's
`ProblemRecord` conventions.

Preferred skeletal shape:

```json
{
  "source": "<source_name>",
  "provenance": "extracted",
  "truth_policy": "extracted",
  "statement": "<problem statement>",
  "answer": "<optional answer>",
  "arxiv_id": "<source arxiv id when available>",
  "family": "realmath",
  "metadata": {}
}
```

Do not label computed or generated records as extracted.

If the adapter creates a record whose truth was generated or computed rather
than extracted from the source paper, stamp it as:

```json
{
  "provenance": "computed",
  "truth_policy": "trusted"
}
```

IcePick's groundtruth stage intentionally discards generated records.

## Manifest Requirements

RealMath scraping is acquisition work. Unlike manual mounts, it must not be
auto-approved when it performs scraping, external calls, or other costly work.

The manifest must use IcePick's existing manifest contract:

```text
source_type: realmath_scrape
processor_mode: production | flow_testing
requested_by
requested_at
approved_by
approved_at
source_name
target_count
call_budget
scrape_window
families
truth_policy
output_dir
calibration_sheet
approval_notes
```

Call-bearing or scrape-bearing runs must fail closed when approval fields are
missing.

For `production`, `call_budget` must cover the estimate for the selected
extraction mode:

- `abstract`: arXiv query calls
- `latex`: arXiv query calls plus e-print source fetches
- `qa`: arXiv query calls, e-print source fetches, and QA-generation calls

## Processor Modes

### `production`

Production mode may perform approved acquisition work.

Rules:

- require an approved manifest
- obey `call_budget`
- write `manifest.json`
- write `handoff/records.jsonl`
- preserve raw source artifacts under `raw/`
- write a short report under `reports/source_report.md`
- report acquisition-call spend
- never store secrets in the repo

### `flow_testing`

Flow-testing mode is for deterministic orchestration checks.

Rules:

- do not scrape
- do not call external services
- replay local fixture or calibration artifacts
- write the same output layout as production
- stamp outputs or reports with calibration replay information where applicable
- do not allow flow-testing output into production buckets

## Operator Flow

The allocation `plan`, `approve`, and `run` commands are implemented (see
[`scraper_runbook.md`](scraper_runbook.md) for the full runbook). Planning is
pure; `approve` is the human gate that turns a proposed plan into an approved
manifest; `run` executes it.

```sh
icepick allocation plan \
  --source-type realmath_scrape \
  --source realmath_2026Q2 \
  --target-count 500 \
  --output-dir out/intake
```

After reviewing the proposed plan and its estimate, approve it:

```sh
icepick allocation approve \
  --plan out/intake/plans/<timestamp>_realmath_2026Q2_proposed_plan.json \
  --mode production \
  --approved-by "$USER" \
  --call-budget <estimated_calls_or_higher> \
  --output-dir out/intake
```

Then run the approved manifest:

```sh
icepick allocation run \
  --manifest out/intake/runs/<run_id>/manifest.json
```

Then processing consumes only the handoff file:

```sh
icepick processing pipeline \
  --mode production \
  --input out/intake/runs/<run_id>/handoff/records.jsonl \
  --output-dir out \
  --combo claude:anthropic \
  --anthro-key-file ../anthro_key.env
```

## Maintenance Checklist

When changing the scraper:

- keep public adapter functions small and readable
- keep scraper-specific machinery private to the adapter or a clearly named
  allocation module
- validate manifests through the existing manifest helpers
- write JSONL for record streams
- write JSON for manifests
- write Markdown for human reports
- preserve stable run directories
- do not mutate source files
- update flow-testing fixtures before production behavior changes
- update CLI and integration tests with behavior changes

## Current Tests

Primary test files:

```text
tests/allocation/adapters/test_realmath_scrape_plan.py
tests/allocation/adapters/test_realmath_scrape_estimate.py
tests/allocation/adapters/test_realmath_scrape_flow_testing.py
tests/allocation/adapters/test_realmath_scrape_normalise.py
tests/allocation/adapters/test_realmath_scrape_production.py
tests/allocation/scrape/test_realmath_source.py
tests/allocation/scrape/test_realmath_latex.py
tests/allocation/scrape/test_realmath_qa.py
tests/allocation/test_realmath_scrape_manifest.py
tests/allocation/test_cli_plan_run.py
tests/allocation/test_cli_approve.py
tests/allocation/test_cli_plan_scrape_window.py
tests/integration/test_realmath_scrape_handoff.py
```

The tests cover:

- plan creation without calls
- extraction-aware estimates
- approval refusal for unapproved production runs
- deterministic `flow_testing` replay
- canonical handoff JSONL shape
- provenance handling for extracted vs computed records
- no mutation of source artifacts
- output layout compatibility with `processing pipeline`
- production scrape flow against canned arXiv Atom feeds
- LaTeX and QA extraction primitives

## Explicit Non-Goals

Do not:

- add scraper logic to `processing/`
- bypass manifests
- auto-approve scraping runs
- run pass@k inside IcePick
- mutate source artifacts
- feed raw scraper output directly into processing
- mix generated or computed records into an extracted production corpus
- store secrets in the repo
- change `groundtruth` or `poser` for this scaffold

## Compatibility Definition

The implementation is compatible with IcePick when:

- `flow_testing` can produce a deterministic handoff without scraping
- `production` refuses to run without approval
- handoff records are accepted by the existing processing pipeline
- outputs follow the run directory layout in this document
- the source report lets an operator understand counts, drops, warnings, and
  the exact handoff path
- acquisition-call estimates and reports reflect the selected extraction mode

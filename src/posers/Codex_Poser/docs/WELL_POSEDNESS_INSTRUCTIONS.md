# Isolated Well-Posedness Instructions

This file extracts only the well-posedness-testing requirements from the
ModelBreaker processing skeleton.

## Module Boundary

This repo only provides the well-posedness module. The rest of the processing
repo is presumed to exist elsewhere.

The module receives data after pass@k testing and writes a `.json` or `.csv`
file containing a well-posedness score.

It does not:

- run pass@k trials
- run confirmation
- grade answer correctness
- test ground truth
- deduplicate records
- check answer leakage
- route deployment buckets
- scrape, harvest, allocate, or acquire data
- run a chat controller

It may optionally call an Anthropic or OpenAI judge for c01 semantic residue only.

## Input Contract

Each input row should be normalised into:

- `rid`: positional run id
- `uid`: stable content id, preserving an existing uid when supplied
- `source`
- `statement`
- `truth_strings`
- `answer_value`
- `tier`
- `family`
- `params`
- `provenance`: `computed`, `extracted`, `manual`, `external`, or `unknown`
- `label`
- `pass_at_k`
- `n_correct`
- `n_wrong`
- `n_degenerate`
- `modal_wrong`
- `top_wrong_share`
- `raw`

The module should accept JSONL, JSON arrays, and CSV files. Mounted or source
files must not be modified in place.

## Required Check

`c01_wellposed` is always active.

The check is tiered:

- Code flags dangling cross-references, citations, and labels.
- Computed-provenance records are trusted as self-contained when code finds no
  structural defect.
- Extracted/manual/external/unknown records with no structural defect are semantic
  residue. Without a judge, code should mark them as `defer`, not accept or
  reject them.
- With `--judge`, only this residue is sent to the selected provider. Structural
  failures are decided before any API call.
- Judge-only uncertainty must not reject a record.

Structural failure examples:

- `\ref{...}`
- `\eqref{...}`
- `\cref{...}` and related variants
- `\cite{...}` and related variants
- dangling `\label{...}`

Soft semantic residue examples, when no structural reference is present:

- missing definitions
- notation used without context
- an answer that depends on an omitted lemma, theorem, figure, table, or paper
- an underdetermined request

These residue cases require a judge or review tier outside this module.

## Score Contract

`well_posedness_score` is deterministic and conservative:

- `pass`: `1.0`
- `flag`: `0.0`
- `defer`: `0.5`
- `error`: `0.0`

The score is a module-local quality signal. It is not a full deployment verdict.

## Key Handling

Provider keys must live outside the `Codex_Poser` repo:

```text
/Users/redhairing/Desktop/helloworld/anthro_key.env
/Users/redhairing/Desktop/helloworld/openai_key.env
```

When run from the repo root, the CLI defaults are:

- Anthropic: `../anthro_key.env`
- OpenAI: `../openai_key.env`

The selected file must contain the selected provider key:

- `ANTHROPIC_API_KEY`, optionally `ANTHROPIC_MODEL`
- `OPENAI_API_KEY`, optionally `OPENAI_MODEL`

## Output Contract

JSON output should use stable top-level sections:

- `run`
- `inputs`
- `counts`
- `parameters`
- `warnings`
- `records`

Every output record should include:

- `rid`
- `uid`
- `source`
- `provenance`
- `family`
- `label`
- `pass_at_k`
- `n_correct`
- `n_wrong`
- `n_degenerate`
- `well_posedness_status`
- `well_posedness_score`
- `well_posedness_detail`
- `signals`

Every output should record:

- `processor_mode`
- `created_at`
- input paths
- output format
- scoring policy

## Human Readability Rules Kept Here

The module keeps the skeleton's readability rules where they touch this slice:

- few folders with obvious names
- standard Python layout
- summaries before details
- stable JSON sections
- one clear field per decision
- short CLI output
- explicit status names

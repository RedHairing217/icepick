# Codex_Poser

`Codex_Poser` is an isolated well-posedness module for data that has already been
through pass@k testing.

It does not harvest data, run pass@k trials, grade answers, or route deployment
buckets. It normalises post-pass@k records, applies the `c01_wellposed` code
screen, optionally calls a judge for semantic residue, and writes a JSON or CSV
file with a well-posedness score for each record.

## Scope

The module implements the well-posedness slice isolated from the ModelBreaker
processing skeleton:

- Preserve a stable `uid` and a positional `rid`.
- Treat `source` and `provenance` as data.
- Accept records after pass@k testing.
- Keep `c01_wellposed` always active.
- Flag dangling references, citations, and labels as code-certain structural
  failures.
- Trust `computed` provenance as self-contained when no structural defect is
  present.
- Mark extracted, manual, external, or unknown provenance as `defer` when code
  cannot settle the semantic residue and no judge is requested.
- When `--judge` is requested, call Anthropic or OpenAI only for semantic residue
  left after the structural screen.
- Do not let judge-only uncertainty reject records.
- Write human-readable summaries before details in JSON output.

## Input

Supported input formats:

- JSONL, one record per line
- JSON array of records
- CSV

Common field aliases are normalised:

| Normalised field | Accepted aliases |
| --- | --- |
| `statement` | `statement`, `question`, `prompt`, `problem` |
| `truth_strings` | `truth_strings`, `truth`, `answer`, `gold_answer` |
| `n_correct` | `n_correct`, `correct` |
| `n_wrong` | `n_wrong`, `wrong`, `wrong_complete` |
| `n_degenerate` | `n_degenerate`, `degenerate` |

## Scoring

`well_posedness_score` is deliberately conservative:

| Status | Score | Meaning |
| --- | ---: | --- |
| `pass` | `1.0` | Code settled the record as well-posed. |
| `flag` | `0.0` | Code found a structural defect. |
| `defer` | `0.5` | Code found no structural defect, but semantic well-posedness needs a judge/review tier. |
| `error` | `0.0` | The input is malformed enough that the module cannot score it safely. |

Pass@k counts are carried through as context. They are not used to turn semantic
uncertainty into rejection.

## Usage

Run from the repo root:

```bash
python -m codex_poser.well_posedness.cli score \
  --mode flow_testing \
  --input tests/fixtures/pass_at_k.jsonl \
  --output out/well_posedness.json
```

CSV output is selected by extension or explicitly:

```bash
python -m codex_poser.well_posedness.cli score \
  --mode production \
  --input data/pass_at_k.jsonl \
  --output out/well_posedness.csv \
  --format csv
```

If installed as a package, the same command is available as:

```bash
codex-poser score --mode production --input data/pass_at_k.jsonl --output out/well_posedness.json
```

## Optional Judge

Judge calls are off by default. To adjudicate extracted/manual/external residue,
keep the provider env files one directory above this repo:

```text
helloworld/
  anthro_key.env
  openai_key.env
  Codex_Poser/
```

Then run:

```bash
python -m codex_poser.well_posedness.cli score \
  --mode production \
  --judge \
  --judge-provider anthropic \
  --input data/pass_at_k.jsonl \
  --output out/well_posedness.json
```

For OpenAI:

```bash
python -m codex_poser.well_posedness.cli score \
  --mode production \
  --judge \
  --judge-provider openai \
  --input data/pass_at_k.jsonl \
  --output out/well_posedness.json
```

By default, Anthropic reads `../anthro_key.env` and OpenAI reads
`../openai_key.env`. You can still override either with `--key-env`.

The selected file must define the selected provider's key:

```text
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
```

```text
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

Model values are optional and may be overridden with `--judge-model`. The key
file is not part of this repo.

## Output

JSON output has stable top-level sections:

- `run`
- `inputs`
- `counts`
- `parameters`
- `warnings`
- `records`

CSV output writes one row per record with the well-posedness status, score,
detail, and signals.

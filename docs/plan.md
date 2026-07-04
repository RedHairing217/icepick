# Pipeline overview

The icepick pipeline is three stages, in this order:

```
ingest  →  groundtruth  →  poser (the gate)  →  final corpus
```

Each stage has its own CLI command. Position is the user's choice — the
groundtruth stage is fluid and can run before or after pass@k.

| Stage         | What it does                                                                        | Status |
| ------------- | ----------------------------------------------------------------------------------- | ------ |
| `ingest`      | Load JSONL into normalised ``ProblemRecord`` (stable ``uid``, provenance, label).   | ✅ working |
| `groundtruth` | Anthropic web_search judge that verifies the source arXiv paper is peer-reviewed AND indexed in a reputable bibliographic database. Discards generated records (no generated records in icepick). | ✅ working |
| `poser`       | The well-posedness gate. Fleet of ``(build, provider)`` combos — Claude_Poser × {anthropic, openai} and Codex_Poser × {anthropic, openai} — any subset in parallel. **This is the gate.** Records that pass are the final corpus. | ✅ working |

## What's deliberately out of scope

The original spec called for a richer post-pass@k gate with c01/c02/c04/c06/c09
checks, soundness escalation, confirmation resampling, and merge-back. None of
those are part of the implemented pipeline:

| Check / stage          | Status              | Replaced by / reason                                       |
| ---------------------- | ------------------- | ---------------------------------------------------------- |
| c01 well-posedness     | Removed             | Now the `poser` stage (Claude_Poser / Codex_Poser).        |
| c02 ground-truth       | Removed             | Now the `groundtruth` stage (Anthropic web_search).        |
| c04 leakage            | Omitted by user     | —                                                          |
| c06 duplication        | Omitted by user     | Performed during extraction upstream.                      |
| c09 robustness         | Omitted by user     | —                                                          |
| Soundness escalation   | Removed             | No more soundness lanes (c01 + c02 both moved out).        |
| Confirmation / rerun   | Not implemented     | Depends on c09; out of scope.                              |
| Merge-back / apply-confirmed | Not implemented | Depends on confirmation; out of scope.                     |

The `processing gate` command does not exist. The poser stage IS the gate.

## Mandatory design rules carried into this scaffold

- Family and source are data, not code.
- Normalise new input shapes at ingest.
- Preserve a stable content `uid` so records join across runs and input
  order changes.
- Keep dashboards and experiment logs out of this repo. Harvesting and
  scraping are in scope, but only behind the `allocation` subsystem's
  acquisition adapters — never inside `processing`.
- Separate data processing, data allocation, and chat control into
  independent subsystems.
- Each subsystem must have its own CLI and tests before integration.
- Integration must happen through typed contracts, manifests, and output
  files, not shared mutable state.
- Human readability is a high-priority implementation requirement.

## Subsystem boundaries

- `processing` must run without `allocation`.
- `processing` must run without `agent`.
- `allocation` must dry-run and validate manifests without `agent`.
- `agent` must run against mocked processing/allocation tools.
- Integration tests compose the three only after each passes alone.

## Processor modes

| Mode           | Real LLM calls | Real scrape | Calibration sheet | Outputs valid for production? |
| -------------- | -------------- | ----------- | ----------------- | ----------------------------- |
| `production`   | yes            | yes         | optional          | yes                           |
| `flow_testing` | no             | no          | required          | no — stamped `calibration_replay: true` |

A run without `--mode` fails. A `flow_testing` run without
`--calibration-sheet` fails.

## Host roles

| Role    | Used by                       | Class               | Sharing with other role |
| ------- | ----------------------------- | ------------------- | ----------------------- |
| subject | future pass@k / confirmation  | `SubjectLLMHost`    | fails closed            |
| manager | agent controller              | `ManagerLLMHost`    | fails closed            |

`config.validate_host_roles` is the single point of enforcement and is
reached by every wiring path.

## Groundtruth — publication-status check

An Anthropic web_search judge that verifies the source arXiv paper is
**peer-reviewed AND indexed** in a reputable bibliographic database
(Scopus, Web of Science, DBLP, MathSciNet, PubMed, IEEE Xplore, ACM DL,
or equivalent). Predatory journals and preprint-only postings do not
pass.

**Position is the user's choice.** Run it before pass@k to discard
records before paying sampling cost, or after pass@k to filter survivors
before the poser.

Verdict statuses: `published` (passes), `unpublished` (fails),
`defer` (judges couldn't agree), `error` (API failure), `discarded`
(record dropped pre-lookup — generated provenance or no arxiv_id).

icepick does **not** process generated records. Records with
`provenance = "computed"` are dropped at this stage with
`discarded_reason = "generated_provenance"`.

One Anthropic call per unique arxiv_id (cached). Three independent
judges per paper; majority uphold (default 2-of-3).

## Poser — the gate

The poser stage drives a fleet of `(build, provider)` combinations:

| build | × | provider     | combo key             |
| ----- | - | ------------ | --------------------- |
| claude | × | anthropic   | `claude:anthropic`    |
| claude | × | openai      | `claude:openai`       |
| codex  | × | anthropic   | `codex:anthropic`     |
| codex  | × | openai      | `codex:openai`        |

`build` selects the poser binary (`claude-poser` or `codex-poser`).
`provider` selects which judge API the chosen build calls (Anthropic or
OpenAI). Both posers refuse to load the other provider's key file by
design, so credentials stay segregated even when all four combos run in
parallel.

The single human-in-the-loop decision is which combos to run. Pass
`--combo BUILD:PROVIDER` repeatedly, or `--combo all` for the full
four-way fleet. Combos run concurrently by default (`--serialize-fleet`
falls back to sequential).

When the fleet contains more than one combo, the runner writes
`comparison.jsonl` + `comparison_report.md` and assembles a combined
gate-input file under `--comparison-policy`:

- `intersect` (default) — admit iff every combo says well_posed
- `union` — admit iff any combo says well_posed
- `majority` — admit iff strictly more than half admit
- `prefer:<build>:<provider>` — use one combo's verdicts verbatim

# Well-Posed Band Corpus Miss Audit Skeleton

Paste this into a fresh context window when the goal is to audit the current
band corpus for false positives in well-posedness. This is an audit scaffold,
not a launch authorization.

## Fresh Window Prompt

You are the orchestrator for an Icepick corpus audit. Use Claude Fable 5 as the
orchestrator model if available. Your job is to direct high-level analysis
agents over the band corpus and find records that may have been incorrectly
accepted as well-posed.

Repository: `/Users/redhairing/Desktop/helloworld/icepick`

Read first:

- `AGENTS.md`
- `docs/SESSION_HANDOFF.md`
- `docs/pipeline_controller.md`, especially the well-posed cascade and pass@k
  sections
- `out/corpus_pde625/corpus_manifest.json`

Canonical corpus inputs:

- `out/corpus_pde625/band_corpus.jsonl` - canonical band row list
- `out/corpus_pde625/wellposed_band.json` - equivalent richer JSON with
  `wellposed_via`, `source_batch`, and nested pass@k details

Disk-verified starting point for this skeleton: `band_corpus.jsonl` has 309
rows and `corpus_manifest.json` reports `total_band_records: 309`. Re-check
those counts before acting because parallel sessions may have folded more
records.

Hard boundaries:

- Do not launch scrapes, funnels, pass@k, Qwen calls, judge calls, or paid API
  work unless Nicky explicitly releases that work in this current session.
- Do not edit or delete existing files under `out/`; new audit directories and
  files are okay.
- Do not fold, remove, relabel, or mutate corpus rows. This pass only produces
  candidate findings.
- Treat pass@k difficulty as separate from well-posedness. A hard problem can
  still be well-posed.
- A record is a well-posedness miss only if the problem as posed does not
  determine a single fixed answer under standard mathematical conventions, or
  if the extracted answer is not actually determined by the statement.

Suggested output directory:

`out/audits/wellposed_band_miss_audit_YYYYMMDDTHHMMSSZ/`

Suggested outputs:

- `audit_manifest.json`
- `shards/shard_XX_input.jsonl`
- `agent_reviews/shard_XX_reviews.jsonl`
- `miss_candidates.jsonl`
- `needs_human.jsonl`
- `audit_report.md`

## Orchestrator Procedure

1. Verify the worktree state and corpus counts. Record current branch, dirty
   paths, corpus file mtimes, row count, and manifest total in the audit
   manifest.
2. Load `wellposed_band.json` when possible so agents can see `wellposed_via`
   and `source_batch`. Use `band_corpus.jsonl` as the canonical row list.
3. Shard the corpus into review chunks. A good default is 12-20 records per
   agent, small enough that every row gets real mathematical attention.
4. Give each agent only its assigned rows plus the rubric below. Agents should
   not modify files. They return JSONL review rows.
5. Re-review every `likely_miss` and `needs_human` record with an independent
   second agent. If two agents disagree, the orchestrator adjudicates or marks
   `needs_human`.
6. Produce a concise final report with counts by verdict, miss type, source
   batch, and confidence. Include the exact uid list for action.

## Agent Assignment Prompt

You are a mathematical well-posedness audit agent. You are reviewing records
from Icepick's band corpus. Your task is not to solve pass@k, not to judge
whether Qwen should have solved it, and not to punish advanced notation. Your
task is only to decide whether the problem statement and answer form a valid
single-answer problem.

For each row, inspect:

- `uid`
- `arxiv_id`
- `statement`
- `answer`
- `metadata.source_statement`
- `metadata.title`
- `tier`
- `truth_policy`
- `wellposed_via` / `source_batch` if present
- pass@k fields only as weak diagnostic hints, not as the decision basis

Verdict options:

- `keep_wellposed`: statement determines the answer under ordinary math
  conventions.
- `likely_miss`: statement does not determine a single fixed answer, the answer
  is not implied, essential notation/context is missing, or the QA extraction
  changed the theorem into an invalid question.
- `needs_human`: genuine ambiguity, highly specialized convention, or the
  record may be valid but the issue is too subtle to decide confidently.

Miss types:

- `missing_context`: required definitions, hypotheses, normalization, domain,
  boundary conditions, parameters, or notation are absent.
- `multiple_answers`: multiple incompatible answers satisfy the posed question.
- `answer_not_determined`: the supplied answer may be true in the source but is
  not forced by the statement as written.
- `extraction_mismatch`: `statement` asks something different from
  `metadata.source_statement`, or the answer was pulled from surrounding text
  not represented in the question.
- `convention_dependent`: answer depends on a non-universal convention not
  specified in the statement.
- `ill_typed_or_invalid`: objects in the question are mathematically malformed
  or impossible as stated.
- `other`: use only with a short explanation.

Important standards:

- Standard terminology is allowed. Do not flag merely because a graduate-level
  definition is assumed.
- If the statement includes enough constraints to identify the requested
  expression up to standard notation, keep it.
- If the answer is one of several equivalent forms, keep it.
- If the issue is only that the problem is difficult, keep it.
- If `metadata.source_statement` contains the missing information but the
  public `statement` does not, flag it. The corpus problem must stand on its
  own.
- If the source theorem itself has a proof gap, ignore that unless it affects
  whether the posed question has a determinate answer.

Return one JSON object per record:

```json
{
  "uid": "string",
  "verdict": "keep_wellposed | likely_miss | needs_human",
  "confidence": 0.0,
  "miss_type": "missing_context | multiple_answers | answer_not_determined | extraction_mismatch | convention_dependent | ill_typed_or_invalid | other | null",
  "one_sentence_reason": "short reason grounded in the row",
  "evidence": {
    "statement_issue": "quoted or paraphrased issue",
    "answer_issue": "why the answer is or is not determined",
    "source_statement_note": "how metadata.source_statement affects the call"
  },
  "minimal_fix": "short edit that would make it well-posed, or null",
  "reviewer": "agent id"
}
```

## Orchestrator Report Shape

`audit_report.md` should contain:

- Corpus snapshot: files, mtimes, counts, git status summary.
- Method: shard size, number of agents, whether second review was used.
- Summary counts: keep, likely miss, needs human.
- Candidate table: uid, arxiv_id, source_batch, verdict, miss_type,
  confidence, short reason.
- High-confidence removals: only records with independent agreement.
- Human-adjudication queue.
- Non-actions: explicitly state that no corpus rows were changed and no
  live/paid processing was launched.

## Suggested Triage Heuristics

Prioritize independent second review for records with:

- `statement` referring to "the constant", "the solution", "this equation",
  "above", "defined in the paper", or named objects without definitions.
- `answer` containing parameters not present in `statement`.
- `metadata.source_statement` much richer than `statement`.
- `truth_policy: extracted` with a terse generated question.
- `wellposed_via: stage1_false_kill_overturned`, not because those rows are
  bad, but because they came through a rescue path and deserve explicit audit
  accounting.
- Very low `n_correct` or high `top_wrong_share`, as weak hints only.

## Stop Conditions

Stop and ask Nicky before:

- removing, mutating, folding, or reclassifying any corpus row;
- launching any model judge, scrape, pass@k, Qwen, or network-backed paper
  fetch;
- spending above the current approved budget;
- resolving `needs_human` records by assumption.

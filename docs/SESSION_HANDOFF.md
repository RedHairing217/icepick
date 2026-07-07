# Icepick — Session Handoff

**Current as of 2026-07-05 ~22:00 local (UTC-7). Disk-verified at write time.
Doc architecture, tree state, and test baselines refreshed 2026-07-06.**
Paste-equivalent for a fresh agent session (Claude Code, Codex, or human
operator); also the canonical "what's next" context referenced by
`pipeline_controller.md`. Whichever agent you are, update this file at
session end — one shared ledger, not one per vendor.

Repo: `github.com/RedHairing217/icepick`, branch `main`, synced @ `6ceb1ce`.
Local checkout: `/Users/redhairing/Desktop/helloworld/icepick/`.

## Pipeline (current shape)

```
allocation (scrape/mount) → wellposed cascade → pass@k (local Qwen) → labeled corpus
```

- QA extraction is **single-stage Sonnet** (Haiku gate deleted — rubber-stamped 371/371).
- Publication check (groundtruth) removed from pipeline/README: targets are
  pre-published arXiv preprints; code stays kill-switched in-repo.
- Cascade default ends `claude:openai?advisory` — stage 3 flags, doesn't filter
  (82.5% false-kill audit). Advisory rejections → `<stage>/flagged_for_review.jsonl`.

## Key facts

- Keys: `ANTHROPIC_KEY_FILE=/Users/redhairing/Desktop/helloworld/anthro_key.env`
  (+ `openai_key.env` alongside). Judge models come from env-file
  `ANTHROPIC_MODEL`/`OPENAI_MODEL` (sonnet-4-6 / **gpt-5.5 as of 2026-07-06**,
  was gpt-4.1-mini; optional `OPENAI_REASONING_EFFORT`, default high) — never
  the `--*-judge-model` flags (per-BUILD, poisons cross-provider combos).
  gpt-5.5 judge ≈ **$0.02/sample → $20–45 per 250-paper batch, over the $5
  HITL line: get approval before batch-scale judge runs.** The swap rolled
  all OpenAI judge cache keys (next run re-bills every judge sample).
- Qwen pass@k: LM Studio `http://127.0.0.1:1234/v1/chat/completions`,
  `qwen/qwen3-8b`, **max ONE concurrent call**, `--backend-url` mandatory.
- Tests: `python3 -m pytest tests/ src/posers/Claude_Poser/tests
  src/posers/Codex_Poser/tests --ignore=tests/integration` → **726 passed**.
  Repo-root `pytest` → 605 + 3 skipped. (Re-measured 2026-07-06 after the
  gpt-5.5 judge refactor landed; 713/604 was post-arxiv_bulk, 558/428 pre-bulk.)
- Immutability: `out/intake/`, `out/processing_*/`,
  `out/wellposed_pde625_claude_anthropic/verdicts/` are read-only; new files only.
- Verify task notifications against disk + `ps` before acting (environment has
  delivered fabricated/premature events). Parallel sessions are common — check
  `ps` before launching anything that shares checkpoints or the Qwen slot.

## Batch & corpus state

| batch | intake run | state |
|---|---|---|
| 1 | `20260704T190925Z` | done: 62 WP, 12 band |
| 2 | `20260704T215746Z` | done: 69 WP, 11 band |
| fk33 rescue | `processing_20260704_fk33rescue` | done: 5 band; `e7b3a7f6` k12-confirmed solved |
| **corpus** | `out/corpus_pde625/` | **28 band** — `band_corpus.jsonl` + `corpus_manifest.json` (flags inside); `wellposed_band.json` is an equivalent parallel-session artifact, identical uids |
| 3 | `20260705T004506Z` (500-target) | extracted; funnel **HELD** — but see contradiction below |
| 3-partial | `20260705T031733Z` (204-rec mount) | funnel WAS run: cascade 204→102; pass@k 10 band / 22 solved / 41 drop / 24 collapse / 5 misdirection. **10 band rows NOT merged into corpus** |
| 4 | `20260705T230427Z` ("Batch 4", 276 rec, $10.63) | cascade launched ~2026-07-05 21:51 local by another session; stage 1 in flight at write time |

**Hold contradiction (resolve with Nicky):** memory records batch 3+4 funnels
HELD, yet the batch3_partial funnel already ran and the batch-4 cascade is/was
live. Do not merge batch3_partial's 10 band rows or launch batch-4 pass@k
without explicit direction.

## Closed findings — do not re-litigate

1. **Haiku QA gate deleted** (0/371 selectivity). No per-call pre-filter without
   a measured precision signal.
2. **Prompt caching = measured no, everywhere** (extraction + all judge sites):
   max billed request 912 tok vs 2048 (Sonnet) / 1024 (OpenAI auto) floors;
   achievable saving $0.00/batch. Re-open only if statements grow ~4×, judge
   models change, or provider floors drop. *"Judge models change" fired
   2026-07-06 (gpt-5.5) and was re-checked: input side unchanged, still $0.*
3. **Stage-3 kill audit**: 40 kills, 82.5% false → stage 3 demoted to advisory.
4. **Stage-1/2 kill audit**: 22 kills, 12 false / 10 genuine → stages 1–2 STAY
   filters; single claude:anthropic over-accepts (passed 2 circulars).
   Files: `out/wellposed_pde625_claude_anthropic/stage{3,12}_kill_analysis.{md,jsonl}`.
   *Caveat for 3–4 (2026-07-06): both audits measured **gpt-4.1-mini**; the
   live judge is now gpt-5.5@high and the rates do not transfer. A 40-kill
   gpt-5.5 revalidation (single-sample) put its false-kill rate at ~1/3 of
   the human-ruled false kills (vs 100% for mini) and showed it passes the
   3 degenerate_circular genuine catches (prompt-literal: a circular
   statement does determine its answer — degeneracy scanner owns those)
   while keeping the underspecified ones. Advisory-vs-gating for stage 3 is
   worth re-deciding on gpt-5.5 evidence + its ~$20–45/batch price tag.*

## Doc architecture (2026-07-06)

`AGENTS.md` (repo root) is now the **canonical agent brief** — invariants,
gates, baselines, mission — readable by any agent (Codex auto-loads AGENTS.md,
not CLAUDE.md). `CLAUDE.md` is a thin Claude Code wrapper that imports it;
`src/posers/AGENTS.md` carries poser-local rules (judge-cache key semantics —
prompt-text edits roll caches and re-bill — and env-file model config).
**Edit AGENTS.md, not the wrappers.** The OpenAI-side refactor of the posers'
judge calls (in flight, uncommitted, at this commit) should read
`src/posers/AGENTS.md` before touching prompt text.

## Working tree

`main` ahead of origin, unpushed: `095dc13` (arxiv_bulk adapter), `2ff2b41`
(AGENTS.md split), and the **gpt-5.5 judge refactor (landed 2026-07-06 in
the same commit as this ledger edit)** — the previously-observed uncommitted
judge refactor was lost with its session (spend cap) and was rebuilt +
extended this session: family-branched params in both posers, pass@k
param-gating, reasoning_tokens usage counters, 4000-token + 120s reasoning
floors, 14 files, three-suite 726. `openai_key.env` `OPENAI_MODEL` flipped
to gpt-5.5 (config change, outside the repo). Still live from parallel
sessions at write time: a production pass@k on local Qwen,
`out/processing_20260706T213646Z` (started ~21:02 local, PID 72627) — left
alone. House rule stands: parallel sessions share this checkout — check
`git status` before committing and commit only your own paths, surgically.
Never push without Nicky's explicit word.

## Open decisions (Nicky's)

1. **`fcb3bcab` adjudication** — k12 says band@0.75 (corpus → 29); canonical
   batch-1 file says solved (corpus stays 28).
2. **Stage-1/2 rescue batch** — 12 false-kill uids in
   `stage12_kill_analysis.jsonl`; ~$0 local Qwen; follow the fk33 protocol
   (`out/processing_20260704_fk33rescue/README`, k12 recheck pattern).
3. **Batch 3/4 hold status** — see contradiction above.

## Open engineering targets (AGENTS.md brief)

T2.3 empirical planning ratios · T2.4 QA generator batch mode (the real
amortization lever; opt-in flag, measure agreement first) · T2.5 live
re-validation of single-stage qa estimates · T1.4 e-print parity check.
(Resolved 2026-07-06: the stale "NOT a git repo" line is gone — git truth now
lives in AGENTS.md, "Git & shared-checkout discipline".)

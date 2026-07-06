# Icepick — Session Handoff

**Current as of 2026-07-05 ~22:00 local (UTC-7). Disk-verified at write time.**
Paste-equivalent for a fresh Claude session; also the canonical "what's next"
context referenced by `pipeline_controller.md`. Update this file at session end.

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
  `ANTHROPIC_MODEL`/`OPENAI_MODEL` (sonnet-4-6 / gpt-4.1-mini) — never the
  `--*-judge-model` flags (per-BUILD, poisons cross-provider combos).
- Qwen pass@k: LM Studio `http://127.0.0.1:1234/v1/chat/completions`,
  `qwen/qwen3-8b`, **max ONE concurrent call**, `--backend-url` mandatory.
- Tests: `python3 -m pytest tests/ src/posers/Claude_Poser/tests
  src/posers/Codex_Poser/tests --ignore=tests/integration` → **558 passed**.
  Repo-root `pytest` → 428 + 3 skipped.
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
   models change, or provider floors drop.
3. **Stage-3 kill audit**: 40 kills, 82.5% false → stage 3 demoted to advisory.
4. **Stage-1/2 kill audit**: 22 kills, 12 false / 10 genuine → stages 1–2 STAY
   filters; single claude:anthropic over-accepts (passed 2 circulars).
   Files: `out/wellposed_pde625_claude_anthropic/stage{3,12}_kill_analysis.{md,jsonl}`.

## Uncommitted work in tree (another session's — don't commit/revert)

`--exclude-from-run` feature: `allocation/adapters/realmath_scrape.py`,
`allocation/scrape/realmath.py`, `cli.py` (hunks ~@715, ~@1494), 3 allocation
tests. Commit only your own paths, surgically.

## Open decisions (Nicky's)

1. **`fcb3bcab` adjudication** — k12 says band@0.75 (corpus → 29); canonical
   batch-1 file says solved (corpus stays 28).
2. **Stage-1/2 rescue batch** — 12 false-kill uids in
   `stage12_kill_analysis.jsonl`; ~$0 local Qwen; follow the fk33 protocol
   (`out/processing_20260704_fk33rescue/README`, k12 recheck pattern).
3. **Batch 3/4 hold status** — see contradiction above.

## Open engineering targets (CLAUDE.md brief)

T2.3 empirical planning ratios · T2.4 QA generator batch mode (the real
amortization lever; opt-in flag, measure agreement first) · T2.5 live
re-validation of single-stage qa estimates · T1.4 e-print parity check.
Known stale line: CLAUDE.md says "NOT a git repo" — it is (this file's header
is authoritative).

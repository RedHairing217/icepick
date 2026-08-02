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
   gpt-5.5 revalidation (single-sample, $1.29): of the 33 human-ruled false
   kills it passes 18, still flags 14 (44% of the 32 parsed — vs 100% for
   mini on these records), 1 unresolved error (burned the full 4000-token
   cap thinking, twice); of the 6 genuine catches it keeps all 3
   underspecified ones and passes the 3 degenerate_circular ones
   (prompt-literal: a circular statement does determine its answer — the
   degeneracy scanner owns those). Hard-tail mean cost $0.032/billed
   sample, mean ~900 reasoning tok. Advisory-vs-gating for stage 3 is worth
   re-deciding on gpt-5.5 evidence + its ~$20–60/batch price tag.*
5. **Stage-1 mini kill census (2026-07-07)**: ALL 886 gpt-4.1-mini stage-1
   kills across 8 runs (b1–b7 + batch8-concluded `20260707T001108Z`; every
   judge sample disk-verified mini) censused via a stratified 150-record
   sample (138 random + the 12 human-ruled seeds), 2×opus blind panels +
   third-ruler adjudication (88% inter-rater), calibration 9/12 PASS (panels
   lenient on genuine-miss → FK if anything overcounted). **False-kill rate
   25.4% [18.4%, 33.5%] → ~225 [162, 296] good records killed.** FK = 83%
   standard-terminology-pedantry; GM = 90% underspecified-external. 2/3-split
   kills are 50% FK vs 21.7% for unanimous 3/3. Sonnet proxy VALIDATED:
   P(genuine | Sonnet-agrees-kill) = 88.9% [73.9, 96.9] (b1+2), and Sonnet
   caught-or-deferred 100% of sampled genuine junk → stage 2 makes stage 1's
   junk-catching redundant; but Sonnet also re-kills ~60% of stage-1 false
   kills, so rescue needs panel-grade review, not stage reshuffling (batch-7
   corroboration: ab_stage1 Sonnet re-kills 84.4%). 44 census-confirmed FKs
   run through a rescue pass@k by Nicky (gate fired 03:26 PT 07-07, done
   04:41, $0 local Qwen): 5 solved / 4 band / 3 misdirection / 15 collapse /
   17 drop; judge-math-error subcat solved 3/4. **FOLDED into corpus
   2026-07-07 (Nicky "fold the band"): +4 band (108→112), all 44 into
   wellposed_all (684→728), source_batch stage1rescue_20260707,
   wellposed_via stage1_false_kill_overturned; fk33-style convention,
   backups `.bak-pre-stage1rescuefold`, 0 collisions, three files+manifest
   agree at 112.** Results/tally: `out/stage1_kill_census/rescue_pass_at_k/
   RESULT.md`. Remaining projected FK among the 736 unruled kills: ~187
   [135, 246]. Files: `out/stage1_kill_census/{stage1_kill_census.md,
   census_rulings.jsonl,census_input.jsonl,sampling_frame.json,
   aggregates.json,rubric.md}`. Open for Nicky: kill/demote stage 1 (quality
   axis now measured; cost axis was already decided), and rescue path for
   the ~187 (opus panel sweep ~6M tok · Sonnet triage ~$6–8 + ~1M tok but
   forfeits ~70 FKs hiding among Sonnet-agreed kills · split/error-first
   prioritized sweep).

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

## Batch 7 fold + Tier-1 stage-1-redundancy read-out (2026-07-07)

1. **Batch 7 (`20260706T213646Z`) COMPLETE + FOLDED** (Nicky "fold 7").
   Cascade 309 → 180 (codex:openai, OLD mini judge) → 126 (codex:anthropic)
   → 126 (claude:openai advisory). pass@k on local Qwen (126 records, 664
   calls, $0, `interrupted:false`): **18 band / 24 solved / 43 drop / 24
   collapse / 17 misdirection**. Folded per the merge_batch4-6 convention
   (guards: pre-fold 558/90 asserted, 0 uid collisions, post-fold re-verified
   from disk; backups `.bak-pre-batch7fold`). **Corpus `out/corpus_pde625/`
   now 684 well-posed / 108 band** = b1 12 + b2 11 + fk33 5 + b3 10 + b4 24
   + b5 12 + b6 16 + b7 18. `fcb3bcab` still pending (→109 if band).
   *[Update 2026-07-07: stage-1 census rescue folded +4 band + stage1rescue
   → **728 well-posed / 112 band** (b… + stage1rescue 4); see finding #5 in
   "Closed findings" and `rescue_pass_at_k/RESULT.md`. fcb3bcab → 113 if band.]*
2. **Tier-1 stage-1-redundancy experiment DONE** (approved; $1.17 actual,
   ~24 min). Batch 7's 129 mini-stage-1 kills re-judged by a Sonnet-only
   gate (`--stages codex:anthropic`), `out/ab_stage1/sonnet_on_kills/`:
   **108 ill_posed / 20 well_posed / 1 deterministic error** ("judge replies
   not parseable", 0/3 samples parsed across base+2 retries, uid
   `119e01d1…` — re-run won't fix). **Y = 108/128 = 84.4% redundant**,
   0.6 pt under the >85% kill-stage-1 line (top of the 70–85% escalate
   band). Economics if stage 1 dies: ~$23/batch gpt-5.5 stage-1 judge saved
   − ~$1.2 extra stage-2 Sonnet (full population) − ~$1.5 extra stage-3
   advisory volume ≈ **$20/batch net**; the 20 stage-1-unique kills would
   reach stage-3 advisory (flagged, not silent) then free pass@k. Caveat:
   measures redundancy over *mini's* kill set, not what a gpt-5.5 stage 1
   would kill.
3. **Decision DEFERRED — full-scale stage-1 kill census in flight** (Nicky,
   separate session) as the superseding evidence source; kill/keep lands
   after it reports. `scratch_r1.py` (repo root) is a census-side sympy
   homogeneity check of a Weinstein-type sharp-constant record, committed
   as-is on "commit everything".
   - **Interim (2026-07-07 ~08:14Z): census stratified-sample result in** —
     150 records ruled (2-3 independent opus rulers/record, calibrated 9/12
     vs the human seed set): **44 false_kill / 102 genuine_miss / 4 unclear**.
     Full aggregation (CIs, extrapolation to the 886-record mini population,
     memo) still pending, resuming 04:10 local per Nicky's spend-pause order.
   - **Rescue queued** (Nicky "queue the 44 false kills behind batch 8"): the
     44 census-confirmed false_kill records → `out/stage1_kill_census/
     rescue_pass_at_k/rescue_input.jsonl` (schema-matched to `final_corpus.
     jsonl`). Gated behind batch 8's live pass@k (single-Qwen-slot invariant)
     via uncommitted `gate_stage1rescue_passk.sh` (repo root, nohup'd/disowned,
     PID 95831 at launch) — polls `pgrep -f "icepick processing pass_at_k"`
     every 30s, fires with byte-identical wire params the moment the slot
     frees. Progress: `out/stage1_kill_census/rescue_pass_at_k/gate.log`.
     Not yet folded into corpus — that's a separate decision after results.
4. Git truth at write: unpushed = `a02ce0d` (gpt-5.5 judge refactor) +
   `49d1c3b` (its ledger) + `fc9533c` (arxiv_bulk: real default OAI fetcher,
   parallel session) + this session's commits; `095dc13` and `2ff2b41` ARE
   on origin (the "Working tree" section above predates that push). Still
   no push without Nicky's explicit word.

## 2026-07-07 ~11:45 PT — FK sweep: panel identification over the 740 untested stage-1 kills (partial)

Session directive (Nicky, ~10:25 PT, THIS supersedes the ~05:30 pass@k-as-filter
decision): finish the stage-1 false-kill search by **triage → blinded panels**;
the `out/stage1_kill_census/remainder_pass_at_k/` relaunch (10:11, PID 13517,
died at record 1/740) is INERT — do not resume without Nicky's word.

All artifacts: `out/stage1_kill_census/fk_sweep/` (new files only; out/**
append-only respected; $0 API; no Qwen launches; no commits).

1. **Population 740 rebuilt by exclusion + verified** (886 census_input − 146
   census fk/gm; 4 census-unclear kept).
2. **Triage (all 740 tiered):** high 349 / med 154 / low 237 (+10 error-mode
   auto-queued). Blind 10-seed gate enforced: v1 (haiku) FAILED 2/5 + hex-uid
   corruption; v2 (haiku, recalibrated, int idx) FAILED 3/5; diagnosis — FKs
   whose answer is a universal structural fact (blow-up alternative,
   principal-eigenvalue positivity) have kill reasons TEXTUALLY identical to
   genuine-miss, invisible to pattern-matching; v3 (sonnet, low-tier-only,
   "structural-answer override" prompt) PASSED 5/5 with GM ride-alongs staying
   low (39/276 promoted). Triage tiers are a search ordering, NOT probabilities
   (precision unmeasured beyond n=2).
3. **Panels (census method: 2 blind rulers + third-ruler adjudication, rubric
   verbatim): chunk 0 only** — 12 fresh (all 10 error-mode + 2 top 2/3-splits)
   + 3 blind census seeds. Result: **4 FK / 7 GM / 1 unclear**.
   - **Error-mode finding: the census's ~100% FK prior for 0/0 judge-crash
     kills is REFUTED — measured 3/10 FK [6.7, 65.3]. Stratum now fully ruled.**
   - **Rescue queue (Nicky decides pass@k):** 623a5256 (b2), 6d1ebb71 (b4),
     7d3b6c78 (b5) — error-mode; 02ed4c59 (b1, 2/3-split). 3 unanimous,
     1 majority-2of3. Table in `fk_sweep/SWEEP_REPORT.md`.
   - ⚠️ Calibration 2/3 (below the 80% gate, unremediated — budget): census-FK
     seed 0e4eafef panel-ruled GM unanimously; ambiguous vs the census's own
     documented FK-overcount bias. Weigh before comparing sweep/census rates.
4. **Budget:** ~1.93M subagent tokens of the ~2M target (two failed triage
   passes ate the panel budget; method held at census grade, coverage cut).
   Ruled 12/740; ~182 expected FKs remain in the 728 unruled (census strata
   rates). **Resumption is cheap and staged:** `fk_sweep/chunk_manifest.json`
   + `chunks/chunk_01..41.jsonl` are blinded, priority-ordered (2/3-split
   high first), ~85–100K tok per chunk-pair measured; calibration seeds sit
   in chunk 1.
5. Env note: chunk-0 panel workflow died once in the orchestrator (args
   plumbing) AFTER rulers finished; resumed from journal with rulers cached —
   no ruling work lost or re-billed.

Open for Nicky: (a) pass@k the 4-FK queue? (b) adjudicate 0e4eafef calibration
dispute? (c) continue panels (chunk 1+) on a fresh budget? (d) error-mode
handling upstream — with ~100% FK refuted, consider re-running the 11
census-era error kills' judge stage rather than assuming.

## bulk-batcher BUILT + DISARMED (2026-07-07, build session)

Automatic 250-record batching + Sonnet-only funnel queue. **Ships DISARMED —
only Nicky arms it.** Full design: `docs/bulk_batcher_design.md` (uncommitted).

- Code: `src/icepick/batcher/` (identity/ledger/backfill/journal/slicer/stages/
  state/config/status/daemon/cli_glue) + `tests/batcher/` (311 tests incl. 11
  acceptance covering the 7 brief scenarios). `cli.py` hook = 2 lines (69, 115).
  Suites at ship: root 918/3 skipped, three-suite 1039 (pre-build baselines
  607/3 and 728 intact).
- Dedup: two layers — uid `sha256(source\x1fstatement)[:32]` pre-injected at
  slice (funnel preserves it) under ONE campaign source `arxiv_bulk_pde625`,
  plus source-independent normalized-statement key vs all history. Ledger
  backfilled at `out/auto_batcher/ledger/` — 2,609 blocking + 294 warn-only
  uids (18 sources). Byte replay→collapse+refill; same-uid-diff-content→HARD
  ABORT (batch8's real failure class — its dup was NOT byte-identical);
  cross-source stmt hit→skip+log (policy-flippable).
- $0 dry-run PASSED 13/13 assertions (real CLI + real mount, stubbed paid
  stages; keyless flow_testing smoke) → `out/auto_batcher/DRY_RUN_TRANSCRIPT.md`.
  Go-live/arm/disarm commands: `out/auto_batcher/STATUS.md`.
- NOT committed (awaiting Nicky's word). No pushes. $0 API spend during build.
- Open for Nicky: campaign_source value; cross_source_statement_policy default
  `skip`; 045533Z orphan handoff (294 uids, warn-set); cascade mid-stage kill
  re-bills ≤~$2.30; `--once` STATUS.md held-section asymmetry (daemon mode
  correct; events.jsonl durable).

**ARMED by Nicky 2026-07-07 ~19:25Z.** Daemon live (nohup, initial PID 28279 —
verify `pgrep -f "batcher run"`), watching the June bulk journal (166/250 at
arm; June extraction down-resumable, its resume is hold-gated) + batch9
watch-journal (242/244 ingested as ledger blockers, 2 already known). Code
still UNCOMMITTED. Disarm: `icepick batcher disarm --root out/auto_batcher`.

### Addendum ~12:45 PT — chunk 1 ruled after Nicky's +0.5M budget raise

Chunk-1 panel (12 triage-high 2/3-splits + the other 3 calibration seeds)
completed. Combined sweep now **24 fresh ruled: 9 FK / 13 GM / 2 unclear**
(rate 37.5% [18.8, 59.4]).
- **Calibration RESOLVED: 5/6 = 83%, gate MET** (all chunk-1 seeds unanimous-
  correct; sole miss stays 0e4eafef — isolated, consistent with the census's
  documented FK-overcount bias, not panel drift).
- **2/3-split stratum measured: 6/14 FK = 42.9% [17.7, 71.1]** — census 50%
  prior holds up; 57 splits still unruled (top of chunks 2–6).
- Rescue queue now **9** (7 unanimous / 2 majority): 623a5256(b4), 6d1ebb71(b6),
  7d3b6c78(b1) error-mode; 02ed4c59(b2), 135e51cd(b5), 21157594(r0707),
  2a9d82be(b4), 53030d00(r0707), 561709c5(b5) 2/3-splits. NOTE: rev 1's
  queue table had memory-written batch fields, two wrong — rev 2 of
  SWEEP_REPORT.md is disk-derived and authoritative.
- 4343d684 (batch7, Sonnet-ab-PASSED) ruled GM unanimously — Sonnet re-pass
  is evidence, not truth.
- Spend: 2.137M subagent of 2.5M raised target; STOPPED (chunks 2–41 staged).
- Residual: ~176 expected FKs in the 716 unruled. Census+sweep FKs to date: 53.

**Operator control is now single-file: `./batcherctl.sh` (repo root,
untracked/uncommitted).** One word per action — `status` (default) / `arm` /
`disarm` / `tail` / `clear-halt <reason>` / `help`; production paths baked in;
every command ends with a machine-parsable `STATE {json}` line + exit code, so
low-power operator agents can drive it. `arm` is idempotent convergence
(standing spend approval encoded per Nicky 2026-07-07); `disarm` is graceful
only (never signals). Lifecycle tested in scratch; production tested read-only
+ arm-noop. This supersedes the multi-flag commands in DRY_RUN_TRANSCRIPT.md.

### Addendum ~13:25 PT — chunks 2–6 ruled after Nicky's second raise (+1M → 3.5M)

One workflow (10 blind rulers ∥ + 5 adjudicators, 747K tok, 16 min) ruled 60
records with zero coverage gaps. **Sweep final: 84/740 fresh ruled — 32 FK
(26 unanimous / 6 majority) / 50 GM / 2 unclear.**
- **High-band split strata EXHAUSTED and census priors CONFIRMED at real n:**
  2/3 = 47.6% [32.0, 63.6] (prior 50%); 2/2 = 28.6% [11.3, 52.2] (prior 30%).
  Error-mode stays refuted at 30% (prior ~100%).
- Census-unclear **c6ed02fc resolved genuine_miss** (unanimous, c3 panel).
- Rescue queue now **32** (subcats: 20 std-term-pedantry / 8 notation-conv /
  4 judge-math-error; all 8 batches). Full disk-derived table:
  `out/stage1_kill_census/fk_sweep/SWEEP_REPORT.md` rev 3.
- Combined identified FKs: **76** (44 census pass@k'd, 9/44 recovered + 32
  sweep, awaiting Nicky). Residual ~150–185 in the 656 unruled (597× 3/3).
- Spend 2.88M/3.5M subagent; stopped at the announced tranche boundary.
  Chunks 7–41 pre-built for continuation (~150K/chunk measured).

Open for Nicky: (a) pass@k the 32-queue? (b) two 3-way-split unclears
(58b4f8cc, 44912e4c) → human look; (c) continue into the 3/3 pool (chunks
7+, ~22–27% yield)? (d) upstream: re-judge error-mode kills.

### Addendum ~15:15 PT — 32-FK rescue pass@k LAUNCHED (Nicky release, +1M budget)

Nicky: "push newly discovered fk's into pass@k testing", +1M (budget → 4.5M).
- Input: `out/stage1_kill_census/fk_sweep/rescue_pass_at_k/rescue_input.jsonl`
  — the 32 sweep-confirmed FKs, filtered from the validated 740-record
  `remainder_pass_at_k/remainder_input.jsonl` join (final_corpus schema,
  32/32 re-validated). Disjoint from the census-44 rescue by construction.
- Launch: repo-root `gate_fksweeprescue_passk.sh` (uncommitted, mirrors
  gate_stage1rescue convention), byte-identical wire params (qwen3-8b, k=8,
  temp 0.7, 2048 tok, think off, concurrent 1, $0). Qwen slot verified free
  (only live icepick proc = parallel claude:anthropic cascade PID 30710,
  non-Qwen). Driver PID 31387, nohup-detached; log
  `fk_sweep/rescue_pass_at_k/gate.log`; checkpointed/resumable. ETA ~55 min.
- Readout on completion → `fk_sweep/rescue_pass_at_k/RESULT.md`:
  band+solved = recoverable floor (census-44 baseline: 9/44 = 20.5%);
  non-recoverable ≠ FK refuted. NOT folded into corpus without Nicky's word.

### Addendum ~16:15 PT — 32-FK rescue pass@k COMPLETE: 12/32 recoverable (37.5%)

Clean run (32/32, interrupted:false, 192 calls, $0, 57 min, driver exited 0).
**Recoverable band+solved = 12/32 = 37.5%** vs census-44's 20.5% — the
split-rich sweep queue recovers ~2× better. 5 solved / 7 band / 1
misdirection / 11 collapse / 8 drop. judge-math-error again 3/4 = 75%
(same signature as census-44). Full readout + the 12-record table:
`out/stage1_kill_census/fk_sweep/rescue_pass_at_k/RESULT.md`.
- Flag: 623a5256 (error-mode, majority-2of3) collapsed 0/8 with modal-wrong
  λ^m matching the dissenting ruler's alternative — weakest FK of the 32.
- **NOT folded** (protocol). If Nicky says "fold the band" (census
  precedent): 7 band records → band corpus, all 32 →
  wellposed_all_with_passk tagged source_batch=fksweeprescue_20260707,
  5 solved held in wellposed_all only.

### Addendum ~16:55 PT — band FOLDED (Nicky "fold the band"); sweep continuation running

- k12 recheck (2 boundary rows, fk33 convention): both stayed band
  (c99673ac 4/12, 3f902f30 8/12). **7 band folded: corpus_pde625 now
  849 WP / 136 band** (from 817/129), tag fksweeprescue_20260707, backups
  .bak-pre-fksweeprescuefold, three files + manifest cross-agree at 136,
  0 collisions. 5 solved held in wellposed_all only. Manifest carries the
  fksweeprescue assembled_from entry + k12_rechecks line.
- **Nicky continuation directives (this session): "continue sweep", +1M
  (→5.5M), "automatic band folding allowed"** — panels chunks 7–19 (156
  triage-high 3/3s) IN FLIGHT (workflow wq064pnom); loop for new FKs:
  gated pass@k → auto-fold band (census convention w/ k12 boundary checks).

### Addendum ~17:20 PT — chunks 7–19 ruled (+45 FKs); rescue2 queued behind batch 9

- Panels c7–19 (156 triage-high 3/3s): one workflow, 26 rulers ∥ + 11
  adjudicators, 1.80M tok, 30 min, no gaps/pending. **+45 FKs.**
- **Sweep totals: 240/740 ruled — 77 FK (65 unanimous) / 161 GM / 2 unclear.**
  Triage-high 3/3 stratum measured **28.7% [22.0, 36.2]** (vs 21.7% census
  all-3/3 prior — triage concentration confirmed). Unruled 500 (110 high-3/3,
  154 med incl. all 59 remaining splits, 236 low), ~149 expected FKs.
- **Identified FKs to date: 121** (44 census + 77 sweep).
- Rescue loop (standing auto-authorization): 45 new FKs →
  `fk_sweep/rescue2_pass_at_k/` (input validated 45/45). **Queued behind
  batch 9's live pass@k (PID 48659, parallel session)** via
  `gate_fksweeprescue2_passk.sh` (gate PID 50014, armed 00:04Z, nohup'd —
  survives sessions). On completion: k12 recheck for n=6/8 boundary rows →
  auto-fold band, tag fksweeprescue2_20260707. If no session is alive then,
  the fold steps are in fk_sweep/RESUME_STATE.md.
- Budget: 4.68M/5.5M subagent (stop-line 4.95M); panels stop here — the
  remaining ~270K is reserved for rescue2 fold orchestration + reporting.

### Addendum ~17:45 PT — FINAL sweep (opus rulers) PREPARED, not launched

Nicky: "prepare codebase for final sweep to be done by opus instead of fable."
Everything a cold session needs is under `out/stage1_kill_census/fk_sweep/`:
- `chunks_final/chunk_20..61.jsonl` — all 500 unruled records blinded
  (20–41 mirror the original priority order: 110 high-3/3 + 154 med incl.
  all 59 remaining splits; 42–61 = the 236 triage-low); chunks 20/21 carry
  6 fresh blind OPUS-calibration seeds (never used in triage/fable panels).
- `tools/` — opus_panel_workflow.template.js (every agent call pins
  model:'opus'), emit_chunks_const.py, assemble_final.py (stamps
  ruler_model, model-partitioned aggregates, gate check), fk_lib.py.
- `FINAL_SWEEP_RUNBOOK.md` — tranche plan (A 20–27 w/ calibration gate ≥5/6
  FIRST, B 28–34, C 35–41, D low 42–61; ~6M tok total, ~80–95 expected FKs),
  full per-tranche procedure incl. rescue pass@k + k12 + auto-fold chain.
- Model partition documented: chunks 0–19 = fable-5 rulers, 20+ = opus
  (census-comparable again); never pool rates across ruler models unlabeled.
All prep verified end-to-end (emitter, seed blinding schema-uniformity,
fk_lib self-test, tools parse). NOT launched — tranche releases + budget are
Nicky's. Rescue2 (45 FKs) still queued behind batch-9 pass@k (gate PID 50014).

### Addendum ~19:30 PT — OPUS tranche A′ (chunks 20–23) ruled; GATE 6/6; rescue3 armed

Nicky "continue 0.75Mtok": opus tranche via the prepared template/tooling.
- **OPUS CALIBRATION GATE: 6/6 PERFECT** (5 unanimous + 1 majority; incl.
  8ff31b4a, the census+human+Qwen triple-corroborated seed). Opus panels are
  census-comparable; later opus tranches trusted. Fable↔opus partition is
  stamped per-row (ruler_model) and aggregates.json is model-partitioned.
- Tranche: 54/54 (48 real + 6 seeds), 462K tok (≈8.5K/rec — cheaper than
  fable's ~11.5K), 7 adjudications, 7 min. **+11 FKs (all 3/3, mostly
  std-term-pedantry; opus high-3/3 rate 22.9% vs fable's 28.7% — model
  or queue-depth effect, kept separate in aggregates).**
- **Sweep: 288/740 ruled — 88 FK / 198 GM / 2 unclear. Identified FKs: 132**
  (44 census + 88 sweep). Corpus unchanged pending rescues (849/136).
- Rescue queue now three deep on the single Qwen slot: batch-9 pass@k
  (running, parallel session) → rescue2 (45 FKs, gate PID 50014) →
  **rescue3 (11 opus FKs, gate PID 62810, `gate_fksweeprescue3_passk.sh`)
  — SERIALIZED on rescue2's 45/45 completion + slot-free (fixes the
  double-fire race two slot-only polling gates would have)**.
- Budget: ~5.15M subagent of 6.25M raised. Remaining final-sweep chunks:
  24–41 (high-3/3 tail + med) + 42–61 (low) per FINAL_SWEEP_RUNBOOK.md.

### Addendum ~20:15 PT — tranche B1 (+19 FKs); GATE-DEADLOCK found & fixed; rescue chain LIVE

- Opus tranche B1 (chunks 24–27, 48 recs, 384K tok): **+19 FKs (39.6% —
  FK-dense pocket)**. Sweep: **336/740 ruled, 107 FK; 151 identified total.**
  rescue4 (19) staged + gated behind rescue3.
- **⚠ OPS BUG (fixed, remember for all future gates): a long-running shell
  whose COMMAND LINE contains the literal text "icepick processing pass_at_k"
  (e.g. an inline watcher embedding that pgrep pattern) makes every
  slot-gate's `pgrep -f` match it forever → the whole rescue chain
  deadlocks silently.** This bit us ~03:13Z: batch-9's pass@k had FINISHED
  (79 records on disk) but rescue2's gate saw the watcher shell as a
  phantom slot-holder. Fix: killed the watcher → rescue2 fired 03:14:45Z
  (PID 65344). RULE: never embed the gate pattern verbatim in long-running
  command lines; in watchers use the bracket trick (`pass_at_[k]`) or
  file-based conditions only.
- Rescue chain now self-cascading on the Qwen slot: rescue2 45 (RUNNING,
  ~75 min) → rescue3 11 → rescue4 19 (serialized gates). On full-chain
  completion: k12 boundary rechecks + auto-folds (tags
  fksweeprescue2/3/4_20260707) per standing authorization.
- Budget ~5.54M/6.75M. Panels remaining: chunks 28–41 + low 42–61.

### Addendum ~20:45 PT — fold protocol simplified (Nicky)
"k12 recheck unnecessary. Fold in all 0.125–0.75." → band rows (n_correct
1–6 at k=8, 0.75 boundary INCLUDED) fold directly; no k12 runs, ever, going
forward. Applies to the in-flight rescue2/3/4 folds. Panels HELD (Nicky
interrupted the chunks-28+ launch); rescue chain unaffected, still cascading.

### Addendum ~22:10 PT — rescue chain COMPLETE + all three FOLDED (no-k12 protocol)

Chain ran clean & serialized after the deadlock fix: rescue2 45/45 (04:19Z) →
rescue3 11/11 (04:34Z) → rescue4 19/19 (05:03Z), all exited 0, $0.
Recoverable: r2 13/45 (8 solved/5 band), r3 2/11 (0/2), r4 6/19 (4/2) —
**chain total 21/75 = 28%; all sweep rescues 33/107 = 30.8%.**
Folds (tags fksweeprescue2/3/4_20260707, backups, guards, cross-agree):
+9 band, +75 wellposed_all. NOTE: a parallel session folded batch 9
(+79 WP/+11 band) before our folds — pre-fold books verified consistent at
928/147. **Corpus NOW: 1003 WP / 156 band.**
Sweep FK ledger: 151 identified (44 census + 107 sweep); panel coverage
336/740; panels HELD at Nicky's interrupt (chunks 28–41 + 42–61 staged).

### Addendum ~22:55 PT — tranche B2 (chunks 28-32): +18 FKs; med-splits swept

Opus B2 (60 recs: last high-3/3 + ALL med-tier splits, 528K tok): **+18 FKs.**
Sweep: **396/740 ruled — 125 FK / 267 GM / 4 unclear; 169 identified total**
(44 census + 125 sweep). Split strata now ~complete: 2/3 = 28/70 = 40.0%
[28.5,52.4], 2/2 = 11/38 = 28.9% [15.4,45.9] — both near census priors.
Unruled: 344 (330 3/3, 13 2/2, 1 2/3) — the last high-yield stratum is
gone; remainder is 3/3 + a few splits at ~20-27%.
rescue5 (18 FKs) staged + gated behind the parallel auto-batcher June pass@k
(gate PID 80360, `gate_fksweeprescue5_passk.sh`, waits on slot). On
completion: auto-fold no-k12 (tag fksweeprescue5_20260707) via
tools/fold_rescue.py. Watcher b0xkzfw42 (file-based, deadlock-safe).
Corpus 1003/156 (disk-verified). Budget this session's panels ~6.6M cumulative.

### Addendum 2026-07-08 ~02:55 PT — rescue5 COMPLETE + FOLDED

rescue5 (18 B2/med-split FKs) ran 09:52Z after the auto-batcher freed the slot:
3 solved / 2 band / 4 collapse / 9 drop → 5/18 = 27.8% recoverable. Folded
no-k12 (fksweeprescue5_20260708): +2 band, +18 WP. **Corpus NOW: 1021 WP /
158 band.** All 125 sweep FKs now pass@k'd + folded (rescue+2+3+4+5). Sweep
FK-rescue grand total: 38/125 = 30.4% recoverable. Panel coverage 396/740;
unruled 344 (330 3/3 + 14 dregs); panels idle pending Nicky release.

### Addendum 2026-07-10 - well-posed band miss-audit skeleton

Docs-only prep for a future fresh-context audit of band-corpus false positives:
`docs/wellposed_band_miss_audit_skeleton.md` now contains a paste-ready
orchestrator prompt for Claude Fable 5 plus high-level agent review rubric,
JSONL output schema, and stop conditions. Disk check at prep time:
`out/corpus_pde625/band_corpus.jsonl` = 309 rows,
`corpus_manifest.json` `total_band_records` = 309, and
`wellposed_all_with_passk.json` = 1998 rows. No audit was launched; no live,
paid, Qwen, scrape, pass@k, fold, or corpus mutation work was done.

### Addendum 2026-07-10 ~20:50Z — wellposed band miss-audit EXECUTED (findings only; corpus untouched)

Fresh-window audit per `docs/wellposed_band_miss_audit_skeleton.md` over `band_corpus.jsonl` @ 309 rows
(sha-stamped snapshot + 16 shards under `out/audits/wellposed_band_miss_audit_20260710T010302Z/`).
Method: 16 Fable shard-review agents (skeleton rubric, structured JSONL) → blind independent second pass
over all 47 flagged + all 27 rescue-path rows (70-row pool; 20 rows incidentally got a THIRD review via a
resume-cache quirk — run-1 reviews preserved). Orchestrator adjudicated 11 splits (rationales in
`raw/orchestrator_state.json`). Run stalled at the session usage limit Jul 9 evening after the first pass;
resumed Jul 10 on Nicky's "Continue, 1Mtok" (workflow wf_af4cbfc4-6f1, ~1.53M subagent tokens total).

**RESULTS: 43/309 miss candidates (13.9%)** — Tier 1 unanimous 34 / Tier 2 majority-2of3 5 / Tier 3
adjudicated 4; **2 needs_human** (`e5ed37d5…` Benjamin-Ono normalization, votes nh/K/K; `a7b98a81…`
DW-scheme unit-vector fact, votes nh/nh/M); 264 keeps. Types: missing_context 18, answer_not_determined 11,
extraction_mismatch 6, multiple_answers 4, ill_typed 2, convention_dependent 2. Dominant pattern: QA
extraction drops load-bearing context (elided equations "a system"/"an ODE", uncarried hypotheses,
gained/lost sharpness qualifiers) — an extraction-time placeholder guard would have caught 15+.
Lane rates: Sonnet-only lanes 25/147 (17.0%) vs 2stage+advisory 11/97 (11.3%). **Rescue lanes vindicated:
24/27 keep, 2 miss (76ac6e… unanimous; f5416819… majority — its original stage-1 kill was arguably right),
1 needs_human → 7.4% vs 13.9% corpus-wide.**

Deliverables: `audit_report.md` (tiered uid lists for action), `miss_candidates.jsonl` (full evidence),
`needs_human.jsonl`, `agent_reviews/`. Input anomaly noted: `wellposed_band.json` batch3 entries have
uid/via/tier=null (canonical JSONL intact). NO corpus mutation; no scrapes/judges/pass@k/Qwen/paid calls.
Removal decisions are Nicky's; if folded out, corpus_manifest per-batch band counts + wellposed_all need a
coordinated update.

### Addendum 2026-07-10 ~21:30Z — judge-comparison + funnel-analysis prep docs (docs only)

Follow-on prep for the miss audit, per Nicky. Three-window chain: THIS session wrote
(1) `docs/wellposed_miss_audit_summary_20260710.md` — why the 43 were flagged (7 mechanism
clusters, per-row table, lane rates, control-sampling guidance) — and
(2) `docs/judge_comparison_funnel_skeleton.md` — paste-ready skeleton for a fresh window that
compares the audit vs another judge (disk-first: advisory flags / claude:anthropic fleet
verdicts / panel rulings; live gpt-5.5 run only on release, ≈$5.4 > HITL) and analyzes the
codex:anthropic funnel structure (hypotheses H1-H5, candidate adjustments S1-S6). That window
makes NO code changes; its deliverable is `docs/funnel_adjustment_execution_skeleton.md` for a
THIRD window to execute. No launches, no corpus mutation, docs only.

### Addendum 2026-07-10 ~22:45Z — skeleton unification + cross-audit adjudication (window closed)

Nicky had TWO competing window-2 skeletons (this ledger's 21:30Z entry vs the panel session's
`out/corpus_audit/handoff/SKELETON_judge_comparison_and_funnel_analysis.md`); he tasked a fresh
session to integrate them and adjudicate the two audits' divergent results (43/309 vs 47/282,
flag-set Jaccard 0.448, κ=0.548). A 15-agent blind-first math-checked panel resolved all 12
disputed rows (band-miss right 9, panel right 3 — the 3 de-flags `3ede4dd9` `d682389a` `5cab6922`
are now false-kill sentinels) and confirmed 6/6 sampled panel-only flags incl. `e5ed37d5`
(ex-needs_human, answer-flipping BO-convention fork, computed). Post-adjudication label set —
41 evidence-confirmed ill / 12 presumptive / 8 circular-degenerate POLICY rows / 3 resolved-well /
1 needs_human (`a7b98a81`) — lives in `out/audits/skeleton_unification_20260710T214021Z/`
(adjudicated_labels.jsonl, panel_results.jsonl, COMPARISON_AND_POSITION.md, cross_tabulation.json).
Key empirical shift for the funnel work: in 18/18 adjudications derivation depth, not
source_statement access, separated right from wrong verdicts (H2 recoverability-vs-derivability
confirmed as the central failure; "re-derive, don't recall" is the top rubric candidate).
**Both prior window-2 skeletons are SUPERSEDED (not edited) by
`docs/judge_comparison_funnel_skeleton_v2_unified.md`** — paste that one. Its window consumes the
label set, closes the 7-row residue, benchmarks other judges disk-first, runs the funnel analysis,
and emits `docs/funnel_adjustment_execution_skeleton.md` for window 3. Awaiting Nicky: E1/E2
removal ruling, the circularity policy call, `a7b98a81`, optional releases (12 pending splits /
live judge runs). $0 API, 0 Qwen, corpus untouched (309, sha 01609862… re-verified), no commits.

### Addendum 2026-07-11 ~03:05Z — window-2 verified & consolidated; ONE window-3 skeleton (unification session, cont.)

Nicky's "Continue" launched this session on the unified window-2 mission — mid-flight we
discovered it had ALREADY been run twice in parallel: a "racer" (B-contract deliverables in
out/funnel_adjustment_analysis/, ~21:31–22:01Z) and "window-2b" (session b213a893, third-ruled
racer-vs-unification disagreements → v3 labels + amended S-ranking, closed ~23:05Z). This session
then independently VERIFIED 2b with 10 fresh agents (launched blind to 2b's existence): §2
structural facts fully confirmed at line level (+3 additions: codex_adapter judge_model/judge.model
attribution bug → poser_model='' everywhere; temperature-0.2 corroboration collapse; judge cache
is per-run-dir so rubric edits re-bill only in-place re-judging); §3 vote receipts confirmed on
all 41 E1 rows (visibility census 26 not-seen / 12 seen-excused / 3 unclear; judge confabulates
missing context — 878e7f40 invents a damped wave equation; ef97f733 = live parse-bias pass);
residue rulings double-confirmed on 5/7, RE-CONTESTED on 2 (11e30827 — source companion exponent
1/19 proves formulation-sensitivity; 343249ba — deleted-definition extraction); eb113602 overturn
RATIFIED (full uniqueness proof reconstructed blind, twice independently); 570fcab3 = rubric
semantics fork → Nicky. **Final v3.1 labels: 41 E1 / 5 E2 / 8 policy / 7+1 sentinels / 2 contested
/ 1 nh** (309→268/263/255) in out/audits/skeleton_unification_20260710T214021Z/labels_v3_1.jsonl
(+ V3_VERIFICATION_FINAL.md = the one consolidated Nicky queue). **Window-3: paste
docs/funnel_adjustment_execution_skeleton.md** — consolidates racer scaffold + 2b amendments +
v3.1 (S2 rubric-v2 attempt-the-derivation + carve-outs, S1 advisory lint; optional S7/S0/B1 arm
checkboxes; sentinel-0-kills hard gate; validation ≈$0.7–1.2 hold-gated). The racer skeleton +
2b-§5-amendments combo is superseded by that single file. $0 API / 0 Qwen / corpus untouched
(sha 01609862… re-verified) / no commits. Three-pass lesson logged: unannounced parallel sessions
on one mission → provenance-stamp every artifact row; declare-then-write ordering caught twice.

### Addendum 2026-07-11 ~06:10Z — three-lane reconciliation + governance skeleton (unification session)

Nicky asked this session for a "process correction skeleton" while window-2b was independently
splitting out its rescue + window-3-wrapper lanes — producing the evening's fourth mission-name
collision (receipt I11). Reconciled: (1) 2b's wrapper deltas D1–D3 (fold-resilient validation,
repaired-row pilot, shared-checkout etiquette) are FOLDED INTO docs/funnel_adjustment_execution_
skeleton.md — pasting wrapper or docs skeleton is now equivalent; (2) this session's governance
skeleton RENAMED to docs/process_discipline_skeleton.md (repo-level corrections: mission registry
in this ledger, mandatory provenance stamps, docs/LABEL_AUTHORITY.md pointer, write-then-declare
checkpoints, late-writer discipline, adjudication standards, budget ledger — 11 incident receipts
inside); (3) rescue lane untouched (it correctly consumes labels_v3_1). THREE paste-ready lanes
now: rescue (out/funnel_adjustment_analysis/SKELETON_rescue_repair_lane.md), window-3
(docs/funnel_adjustment_execution_skeleton.md), governance (docs/process_discipline_skeleton.md).
All independent; Nicky arms checkboxes at paste time. $0, no corpus/code changes, no commits.

### Mission claim 20260711T060641Z — funnel-adjustment-execution-w3 (ACTIVE)

| slug | session | started (UTC) | scope | budget |
|---|---|---|---|---|
| funnel-adjustment-execution-w3 | 89fe6f6f (Fable-5) | 20260711T060641Z | execute docs/funnel_adjustment_execution_skeleton.md: S1+S2 core only; S7/S0/B1 UNARMED; sentinel gate = 7 ratified (570fcab3 excluded, unratified); live S2 validation ~$0.7-1.2 treated as released by Nicky's in-session "execute" | unlimited to 06:29Z, then 1.5Mtok |

Writes this window: src/posers/Codex_Poser well_posedness {scoring.py, cli.py, tests} + out/funnel_adjustment_analysis/execution_validation_20260711T060641Z/ + this ledger. Corpus sha 01609862.../309 verified at claim time. Rescue lane detected LIVE in parallel (out/qa_repair_20260711T055242Z, no HANDSHAKE_window3.md yet) — D3 etiquette in force, its dirs untouched, no Qwen use here. Heartbeat at each phase boundary; CLOSED line at session end.

### Addendum 2026-07-11T07:34Z — window-3 EXECUTED: rubric v2 FAILS validation (funnel-adjustment-execution-w3 CLOSED)

Executed docs/funnel_adjustment_execution_skeleton.md (S1+S2 core; S7/S0/B1 unarmed; 7 ratified
sentinels). Code landed UNCOMMITTED, defaults byte-identical (v1 prompt sha-pinned): rubric-v2 +
advisory lint + --judge-max-tokens in codex-poser well_posedness {scoring,cli,tests}; three-suite
1064/0. Frozen population 98 (41 E1 / 5 E2 / 7 sentinels / 45 stratified controls; 12
panel-pending splits excluded; corpus sha 01609862 unchanged throughout). RESULTS — **sentinel
hard gate FAIL 4/7** (01464d48, 549b8fc7, d682389a, eb113602; every kill = "field-standard
machinery undefined"; the carve-out did not protect at temp 0.2); E1 recall 37.5% conservative
(15/40; 50.0% resolved-only) vs >=55% target and 40% floor; controls 2/45 at-limit; S1 static
FAIL 17/244 keeps (defines_then_asks = noise). INCIDENT: v2 "attempt the derivation" x judge
max_tokens 512 -> 40/98 truncation-ERRORs + poisoned judge cache; --judge-max-tokens added
(default 512, no cache-key impact); retry at 1500 resolved 8/40 (all pass); 32 unresolved —
completion exceeds the $5 line, stopped per inv 12. Spend $3.77 actual. VERDICT: package NOT
accepted; v3 needs an executable derivability standard (bounded sketch or two-stage
escalate-to-agent) + JSON-first replies + don't-cache-unparseable. Full analysis:
out/funnel_adjustment_analysis/execution_validation_20260711T060641Z/VALIDATION_REPORT.md.
Rescue lane never wrote HANDSHAKE_window3.md -> D2 pilot skipped-and-said-so. Mission claim
funnel-adjustment-execution-w3: **CLOSED**.

### Correction 2026-07-11T07:36Z (window-3): rescue handshake DID land at ~00:02 PDT — minutes after window-3's final pre-close check. D2 pilot (24 repaired rows, 72 samples) is READY but HELD: worst-case $1.81 > $1.23 remaining under the $5 line. Command + terms staged in execution_validation_20260711T060641Z/VALIDATION_REPORT.md item 6; bundle with the 32-row completion release. Grading-notes in qa_repair RESULT.md flagged as relevant to rubric-v3 design.

### Addendum (window-3 follow-up): rubric v3 DRAFTED on Nicky's "draft v3" — bounded-sketch direction; text + per-change measurement trace + wiring/validation plan in out/funnel_adjustment_analysis/execution_validation_20260711T060641Z/RUBRIC_V3_DRAFT.md. Not wired, not validated, defaults untouched; wiring + ~$1.5-2.5 validation on his release after text review.

### Addendum (window-3 reopened, v3 wired+validated on Nicky's release): **v3 PASSES the sentinel hard gate 0/7 but recall collapses 17.5%** (7/40; v2 was 37.5% w/ 4/7 kills); controls 3/45 (fail-by-one); truncation class ELIMINATED (2/98 residual). v3's catches = strict subset of v2's — keeps nameable-OMISSION defects, loses WRONGNESS defects (canonical-recall 0dd7247b, distortion 8d254fa0, convention 516b7d3f). Measured conclusion: single-prompt JSON judges sit on a kill-rate/catch-rate Pareto frontier whose both endpoints are now measured below acceptance — recommend two-stage (v3 gate + agent-scale escalation of ic>=2/low-conf band ~10-20%), NOT a v4 text iteration. v3 also resolved 29/32 of v2's unresolved rows (pass-leaning, as projected). Code: _PROMPT_V3 wired behind --judge-rubric-version v3 (default v1 untouched), suite 1066/0. Spend: v3 run $1.55 (released); window API total $5.32. Full analysis: execution_validation_20260711T060641Z/V3_VALIDATION_ADDENDUM.md. Everything uncommitted.

### Addendum (window-3): code COMMITTED on Nicky's release — 7a3546c "wellposed: opt-in judge rubrics v2/v3, advisory context lint, --judge-max-tokens" (3 poser files, 609 insertions, all default-off, v1 sha-pinned; suite 1066/0 pre+post). NOT pushed (main ahead 6). SESSION_HANDOFF + batcher cli.py left uncommitted (parallel-session content). Validation evidence stays in out/ (untracked by design).

### Addendum (window-3): Nicky DECIDED — arm v3 + build the two-stage trustworthy funnel. Paste-ready execution skeleton written: **docs/trustworthy_funnel_execution_skeleton.md** (mission slug trustworthy-funnel-execution). Core armed by his message: H1 parse-bias strict policy, H2 B1-attribution fix, H3 don't-cache-unparseable, H4 IC-to-queue, A1 arm v3 (v3 + max-tokens 1500 + strict, on the NEW-batch launch surface only — codex-poser default stays v1, existing gate scripts untouched), B2 stage-B blind-derivation escalation (stage-A flag no longer drops records alone — invariant-8 alignment). Optional checkboxes: S7, live shadow batch, standing report column. Acceptance: sentinels 0/7 through both stages HARD, combined recall >=55/40, net control kills <=2/45, stage-B must rescue >=2 of the 3 known control FPs. NOTE: v3 is NOT live until that window passes its gates. Builds on commit 7a3546c.

### Mission claim 20260711T185543Z — extractor-hardening (ACTIVE, session 89fe6f6f)

| slug | session | started (UTC) | scope | budget |
|---|---|---|---|---|
| extractor-hardening | 89fe6f6f (Fable-5) | 20260711T185543Z | Nicky: "push the extractor fix immediately" — build extraction-time guards: (1) source-vs-statement diff guard (dropped clauses / added superlatives / substitutions — R1+distortion classes), (2) elided-source filter (unresolved refs feeding load-bearing content — R2 class); backtest vs the 41 known-bad (source_statement + repair diffs = ground truth) + FP rate on 279-corpus source pairs; extraction path only (realmath.py QA + bulk lane if shared), NO poser/src changes, no commits until release | remainder of 1.5M window (~700k) |

### Addendum (extractor-hardening CLOSED): the extractor fix is BUILT, uncommitted, ready on Nicky's word.
Root causes confirmed at code level: _clean_tex strips \ref-family destructively (creates the R2 grammatical holes) AND qa_extractor discarded the miner's has_external_refs flag (set since initial commit, consumed by nothing — all 320 audited rows flag=False). Landed (suite 1100/0, baseline 1066): E1 flag+raw-source propagation (source_statement_raw now stored — future audits get pre-strip truth); E5 reference RESOLUTION at mining (label→env index from full tex; resolved content fed to the QA generator so it poses self-contained problems; unresolvable refs keep the flag); E2 qa_ref_guard {off,advisory,strict} default advisory at normalise() (strict quarantines "[quality-guard]", both lanes, resume-safe) + guard_flagged counter in reports; E4 elision-bigram advisory signals. Classifier approaches REJECTED on receipts (out/extractor_hardening_20260711T185640Z/): lexical 41-44% catch @ 26-34% keep-FP; Sonnet diff-model $0.63 backtest 49% @ 32% keep-FP + 5/8 sentinels — content-diff ≠ load-bearingness (that judgment is the posedness problem, handled funnel-side). CAVEATS: (1) clean-row QA cache keys changed ("Theorem:\n"-wrapped) — resuming a PRE-change in-flight run re-bills its QA calls once (~$0.005/call; June dead-resumable run affected); (2) retro catch-rate unmeasurable (raw bodies weren't stored pre-fix) — first live batch reports guard_flagged + resolution rates as the real measurement; (3) no CLI surface for qa_ref_guard yet (plain param, default advisory). Session API total $5.95.

### LIVE RUN (guard-analysis batch, Nicky-released 2026-07-11 ~20:26Z): run 20260711T202559Z, PID 81111 — 250-paper math.AP 2026-01 qa scrape, source pde_guard_analysis_250, budget cap 42,060 calls (realized QA expected $1.5–4), FIRST RUN ON THE HARDENED EXTRACTOR (ref-resolution + qa_ref_guard advisory + guard_flagged counter). EXTRACTION-ONLY: held at handoff — no cascade, no pass@k, NO FOLD without Nicky. Analysis recipe when done: reports/source_report.md (guard_flagged + Drops), raw/quarantined.jsonl, per-record metadata.quality_guard + resolved/unresolved_refs rates in handoff/records.jsonl; compare elision-signal rate vs the retro corpus. Resume if orphaned: re-run `ANTHROPIC_KEY_FILE=/Users/redhairing/Desktop/helloworld/anthro_key.env /opt/anaconda3/bin/icepick allocation run --manifest out/intake/runs/20260711T202559Z/manifest.json` (checkpoint-native).

### LAUNCH-WINDOW CLAIM (paired guard analysis, ~22:52Z 07-11): baseline run 20260711T202559Z (PID 81111, STALE pre-hardening imports — parallel stash window suspected) finishes shortly; gate_guard_analysis_paired.sh then launches the GUARDED arm (run 20260711T225119Z, approved cap 42060) behind an ARMING TRIPWIRE (loaded-module check + first-candidate field verification, kills on failure) per Nicky "Do not launch without arming corrected code". **PARALLEL SESSIONS: do NOT stash/checkout/revert src/icepick/allocation/** between now and the guarded arm's completion** — the tripwire will abort the launch and the analysis slips. Extraction-only, held at handoff, no fold. Analysis = paired per-paper comparison baseline-vs-guarded.

### REFIRE (Nicky "stop it then refire on the hardened QA flow", ~23:50Z 07-11): baseline 20260711T202559Z KILLED (stale imports, PID 81111; STOPPED_NOTE in its _progress; 199 papers/620 records/5063 QA calls kept for offline elision baseline). Stale gate 38835 killed. **HARDENED REFIRE LIVE: run 20260711T234953Z, PID 64692, --max-papers 250 cap, qa_ref_guard advisory** — ARMING VERIFIED (gate_guard_hardened_refire.sh: module-source pre-check + first-candidate source_statement_raw check both PASSED; gate.log). Extraction-only, held at handoff, no cascade/pass@k/fold. Monitor wakes session on exit for guard analysis (guard_flagged, resolved/unresolved_refs rates, elision incidence vs stopped baseline + retro corpus, QA cost actuals). Resume if orphaned: ANTHROPIC_KEY_FILE=.../anthro_key.env icepick allocation run --manifest out/intake/runs/20260711T234953Z/manifest.json (checkpoint-native). PARALLEL SESSIONS: do not stash src/icepick/allocation/** while PID 64692 lives.

### SALVAGE CASCADE (Nicky "kill defective records, send rest through cascade" → "just run all of it", ~00:12Z 07-12): stale baseline 20260711T202559Z (643 rec) → filtered 26 ref-hole-defective (elision/raw-ref, $0) → 617 kept in out/baseline_salvage_20260712T001221Z/{cascade_input,killed_defective}.jsonl (all provenance:extracted). LIVE v1 cascade PID 74666 (--stages codex:anthropic, production rubric v1, est $6-8 RELEASED). HELD: cascade only, NO pass@k, NO fold. CAVEAT recorded: v1 = the gate measured 0/41 on subtle R1/distortion defects; elision only removed R2 ref-holes → output is a coarse structural split, re-gate through two-stage when built before any fold. Concurrent w/ hardened refire PID 64692 (both hit Anthropic API — possible 429 retries, both have backoff). Monitor wakes session at cascade exit (final_corpus count + cost from cascade_manifest.json). Resume if orphaned: same cmd, judge-cached.

### SALVAGE CASCADE DONE (~01:47Z 07-12): 617 → v1 codex:anthropic → **325 unique well-posed / 276 ill-posed (45%) / 7 error**, $5.58 actual (952k in / 182k out). final_corpus.jsonl = 332 rows but 325 UNIQUE (7 dup content-uids — dedup before any fold). NO error leak (7 error uids verified absent from final_corpus — F2 not triggered here). Results in out/baseline_salvage_20260712T001221Z/cascade/. HELD — no pass@k, no fold. TWO CAVEATS binding any downstream use: (1) 45% ill-flag by the LENIENT v1 gate ⇒ stale-extractor output is low quality; true ill-rate higher since v1 caught 0/41 subtle defects in the retro audit — the 325 "well-posed" still contain untested R1/distortion defects; re-gate through two-stage before trusting. (2) dedup the 7. Concurrent refire scrape PID 64692 still live (~219/250 papers).

### HARDENED REFIRE DONE (~02:03Z 07-12) — extractor fix VALIDATED live. Run 20260711T234953Z: 250 papers (cap hit) → 295 handoff records, arming-verified hardened code. PAIRED RESULT vs stale baseline:
| metric | stale baseline (20260711T202559Z) | hardened refire |
|---|---|---|
| source_statement_raw stored | 0/643 (0%) | 295/295 (100%) — audit trail restored |
| ref-carrying theorems | flag was dropped (unmeasurable) | 162/295 (55%) |
| references RESOLVED (fed to QA) | 0% (destructive strip) | 150/162 = 93% (348 refs pulled in) |
| holes in POSED problems (elision) | ~4% (26/643) | **0/295 (0%)** |
| unresolved-ref flagged (advisory) | 0 (no guard) | 35 (12%) — strict would quarantine |
Mechanism confirmed: source_statement still shows 6.4% holes (cleaned text) but resolver fills them pre-posing → POSED output 0% holes. WIN IS SCOPED TO R2 (references); R1 dropped-hypothesis + distortion defects untouched (funnel's job). Extractor fix (uncommitted) now has live validation → commit-ready. handoff at out/intake/runs/20260711T234953Z/handoff/. HELD at handoff — no cascade/pass@k/fold. Nothing in flight now (baseline killed, refire done, salvage cascade done).

### COMMIT + HARDENED CASCADE (Nicky "commit and cascade", ~04:14Z 07-12): extractor fix COMMITTED 8eefdbb "extractor: resolve LaTeX refs instead of destructively stripping them" (7 files, 615 ins, surgical — parallel-session files untouched; suite 1104; NOT pushed, main now ahead 7). Then LIVE cascade on the 295 HARDENED handoff records (run 20260711T234953Z) → out/hardened_cascade_20260712T041410Z/ PID 19612, v1 codex:anthropic, est ~$2.65 (<$5 autonomous). HELD — no pass@k/fold. This is the PAIRED counterpart to baseline_salvage (stale=45% ill); compares ill-rate of resolved-ref extraction vs stale. Monitor wakes at exit. Nothing else in flight.

### ANOTHER HARDENED EXTRACTION (Nicky "begin another hardened extraction", ~04:50Z 07-12): run 20260712T044950Z PID 20185, math.AP **2026-02** (fresh window, no Jan overlap), --max-papers 250, qa_ref_guard advisory, ARMING-VERIFIED (module pre-check + first-candidate source_statement_raw). Extraction-only, held at handoff, no cascade/pass@k/fold. Concurrent w/ hardened cascade PID 19612 (both hit Anthropic API — soft contention, both retry). Est QA ~$4-8 (last comparable 250-cap run's volume; call_budget cap 42060 = hard backstop). Monitor wakes session at exit for guard/cure-rate readout (compare to Jan: 93% cure, 0% posed holes). Resume if orphaned: ANTHROPIC_KEY_FILE=.../anthro_key.env icepick allocation run --manifest out/intake/runs/20260712T044950Z/manifest.json.

### AUTOPILOT FUNNEL (Nicky "extract > cascade > pass@k on autopilot; no human except firing a batch", ~05:00Z 07-12): TWO uncommitted scripts (repo root, session 89fe6f6f):
- **funnel_chain.sh LABEL INPUT OUTDIR [cascade|passk]** — cascade → (Qwen-slot-guarded, bracketed pgrep) pass@k → writes READY_FOR_FOLD.txt. HELD AT PASS@K: never folds (fold stays MANUAL — v1 is the weak gate, fold-review is the safety net). Holds gracefully if LM Studio down. Restartable.
- **fire_batch.sh YEAR MONTH [CAT]** — THE ONLY HUMAN STEP: plan→approve→arming-gated hardened extraction (250-cap)→funnel_chain. e.g. `nohup ./fire_batch.sh 2026 3 &>fire.log &`.
WIRED the two in-flight runs via detached watchers: (A) hardened Jan cascade PID 19612 → pass@k on its final_corpus (out/hardened_cascade_20260712T041410Z/); (B) Feb extraction PID 20185 → cascade → pass@k (out/auto_funnel_20260712T044950Z/). They serialize on the one Qwen slot. Qwen backend verified UP. Per-batch spend (~$3 cascade + ~$4-8 extraction, pass@k $0) is pre-authorized by the autopilot instruction; folding is NOT. Completion = READY_FOR_FOLD.txt in each OUTDIR. NOT committed (operational scripts, like the gate_*.sh).

### Mission claim — autopilot-band-audit (ACTIVE, session 89fe6f6f, ~19:04Z 07-12): Nicky "let's audit them" — pre-fold blind-derivation audit of the 41 band records from the two autopilot batches (20 Jan hardened_cascade_20260712T041410Z + 21 Feb auto_funnel_20260712T044950Z). Audit dir out/audits/autopilot_band_audit_20260712T190340Z/ (manifest, 4 shards 11/10/10/10, 0 meta-join failures). Protocol: stage-1 BLIND derivation (statement+answer only, nameable-defect standard, field-standard carve-out) → stage-2 extraction fidelity (name-not-embed census — paper-local terms named but not defined, the defect class found in spot-check; fidelity vs source; modal_wrong rival-reading check) → orchestrator adjudication → tiered keep/remove/needs-human + report. 4 Fable shard agents in flight. $0 API, no corpus mutation, fold stays held. BOTH batches' READY_FOR_FOLD remain blocked pending this audit.

### autopilot-band-audit CLOSED (~19:45Z 07-12): 41 fold-candidates audited (4 blind-derivation shard reviewers, both-direction rationales). **27 keep / 7 remove_ill / 7 needs_human.** DOMINANT FINDING: pass@k grader equivalence brittleness corrupts >=11/41 (27%) band labels (glyph/brace/constant-absorption/inequality-restatement equivalences graded wrong; several rows regrade OUT of band to solved) — **BLOCKING: $0 equivalence-aware regrade of both batches required before any fold**; both batches' full label sets suspect both directions. Extraction ill-posedness 7/41 ≈17% (better than retro 21-22% single-gate lanes); name-not-embed 8 hits/3-4 fatal → QA-prompt embed-don't-name fix warranted. Resolver eqref-RANGE bug (endpoints only) found (jan_18). R3 key-sign-flip alive (feb_30). fire_batch has NO cross-batch dedup (feb_38==jan_17, same theorem both batches — wire the disarmed batcher's uid ledger or dedup at fold). Full report: out/audits/autopilot_band_audit_20260712T190340Z/audit_report.md (+adjudicated_verdicts.jsonl, audit_summary.json). FOLD REMAINS HELD.

### AUTOPILOT FOLD EXECUTED (Nicky "forget the regrade, just kill the bad problems and fold the good ones", ~20:15Z 07-12): **band_corpus 281→311 (+30 band), NEW sha 5fd087e91f9a3ca9… (2b6504/810d1608/01609862 all RETIRED — rebase any pins)**; wellposed_all 1997→2028 (+30 band +1 solved jan_08, mechanically verified 8/8); wellposed_band 309→339; manifest assembled_from[autopilot_janfeb_20260712] added. KILLED 7 (audit removes; killed_records.jsonl w/ reasons). PARKED 2 for Nicky (jan_05 clean-constant policy, jan_18 eqref-range repair). SKIPPED 1 on collision guard (fold/skipped.json). 6 folded band rows carry corpus_provenance.equivalence_dispute (reviewer-claimed grader equivalence, unconfirmed mechanically — mini_regrade.json; full equivalence regrade remains WAIVED by Nicky but flagged). Backups *.bak-pre-autopilotfold; 10 integrity checks pass; FOLD_MANIFEST.md in audit fold/ dir. Batcher ledger NOT updated (DISARMED; register on re-arm). Solved/misdirection/collapse/drop records from both batches (unaudited) remain in run dirs — NOT folded, separate decision.

### UNFOLD + wellposed_band RECONCILE (Nicky, ~20:4xZ 07-12): autopilot fold REVERTED — all four corpus files restored byte-identical from backups (**band_corpus back to 281 / sha 2b6504…; sha 5fd087 is DEAD**; wellposed_all 1997; manifest entry removed). The 41 audited records revert to AUDITED-NOT-FOLDED (kill/park lists stand as recommendations in the audit dir). THEN wellposed_band.json reconciled to repair/rescue results: rebuilt against canonical band_corpus — **309→281, uid set now exactly matches band_corpus** (250 kept, 30 constructed incl. the 10 repaired rows, 1 pass@k-synced, 48 stale dropped incl. removed E1s + uid=None degenerates). Backup wellposed_band.json.bak-pre-bandreconcile. The wa/wb/bc trio is now mutually consistent for the first time since the qa_repair fold.

### FEB FOLD EXECUTED (Nicky "fold in the most recent extraction", ~20:55Z 07-12): Feb batch (run 20260712T044950Z) audited band rows folded — **band_corpus 281→299, NEW sha 3a4ed9d5d5b6ead9… (2b6504 retired — rebase pins)**; wellposed_band 281→299 (stays in exact uid agreement with bc); wellposed_all 1997→2015; manifest assembled_from[autopilot_feb_20260712]. Folded 18 (14 clean keeps + 4 label-disputed w/ equivalence_dispute provenance); NOT folded: 3 audit removes (feb_26 name-not-embed, feb_30 key-sign-flip, feb_38 recall+dup); 0 collisions. **Jan batch remains AUDITED-NOT-FOLDED** (17 fold-eligible + jan_08-as-solved + 2 parked, lists in audit fold/ dir — one command to fold on Nicky's word). Backups *.bak-pre-febfold; FEB_FOLD_MANIFEST.md in audit fold/ dir. Feb's unaudited solved/collapse/drop records remain in run dirs.

### JAN FOLD EXECUTED (Nicky "audit and fold Jan", ~21:05Z 07-12; audit = standing autopilot_band_audit): **band_corpus 299→311, NEW sha 1b9d5d6220409df3… (3a4ed9d5 retired — rebase pins)**; wellposed_band 299→311 (uid-synced); wellposed_all 2015→2028 (+12 band +1 solved jan_08 w/ label_correction provenance). OUT: 4 audit removes (jan_10/15/16/17), 2 parked for Nicky (jan_05 clean-constant policy, jan_18 eqref-range repair), jan_09 re-skipped (statement collision, standing). Both autopilot batches now FOLDED per audit; corpus trio mutually consistent. Backups *.bak-pre-janfold; JAN_FOLD_MANIFEST.md in audit fold/ dir.

### REPAIR FOLD EXECUTED (Nicky "fold and re-score", 2026-07-16 ~00:2xZ 07-17): the day's audit→repair lane landed — **band_corpus 309→293, NEW sha `e0975e112f05d03e` (13164e3f retired — rebase pins)**; wellposed_band 293 (uid-synced); wellposed_all 2028→2021. OUT: all 26 confirmed-defective records (extraction_defect_check audits, evidence in `out/audits/extraction_defect_check_20260716T193020Z/`). IN: 10 repaired+source-verified records that re-scored band (fresh k=8 local-qwen labels, batch `repair_lane_20260716`); the other 9 repairs re-scored misdirection(5)/collapse(4) — in wellposed_all only (repaired statements grade harder than their defective originals; all nine 0/8). 7 unrepairable records dropped entirely (parked_records.jsonl). Backups `*.bak-pre-repairfold-20260716`; manifest entry `assembled_from_gguf.repair_lane_20260716`; 10 integrity checks pass. **OPEN: LoRA split (corpus_split_200_100.json) now stale** — 26 members removed (alias map = repaired_from in repair_lane/repaired_records.jsonl; holdout 100→91 effective) — split patch/regeneration NOT released; eval_paper_split (paper-level) unaffected. Cascade NOT re-run on repaired rows (source-anchored 2-ruler verification instead; noted per-row in wellposed_note).

### LoRA split RECOVERED + dup audit (Nicky "recover everything / check for duplicates", 2026-07-16 reintegration session): **evalharness/data/corpus_split_200_100.json PATCHED** — all 19 records with a living successor aliased via repaired_from; only the 7 unrepairable dropped. holdout 100→**97** (88 intact + 9 aliased; dropped 8a5c955e/448c8e70/c1e9c2a9), train 200→**196** (186 + 10 aliased; 4 dropped). Verified: 0 uid contamination, 0 identical-statement train↔holdout leakage, 0 paper-level leak, all uids resolve. **CAVEAT stamped in-file: split is NO LONGER band-pure** — recover-everything retains 9 band-exited repairs (holdout 94 band/3 non-band [misdirection×2, collapse×1]; train 190/6). Anchors (10+10) untouched, intact in wellposed_all (not band by design). 3 holdout_papers now have no surviving uid (2508.12364, 2604.27278, 2603.12786) — paper-level lists left as-is per unaffected-by-repair rule; flag if eval_paper_split cares. Backup: corpus_split_200_100.json.bak-pre-recover-20260716. Patch metadata + full alias map under key `repair_patch_20260716`.
**DUP AUDIT (corpus, clean):** band_corpus 0 uid dups / 0 true statement dups; wellposed_all 0 uid dups / 0 true statement dups; 0 repaired-vs-corpus collisions. Two prefix-collision false positives investigated and cleared: 7068a207≠dd063fe6 (same paper 2512.18170, same answer, DIFFERENT full statements — sibling theorems sharing a setup sentence) and 344ef64d≠f1f6a695 (different papers, different statements+answers, shared Gronwall preamble). Corpus sha e0975e11 unchanged by this work (split-only).

### LoRA split → BAND-PURE (Nicky "drop the 9 band-exits, keep it band-pure", supersedes the recover-everything revision minutes earlier): **holdout 97→94, train 196→190.** Dropped the 9 repaired band-exits (holdout: a3ebc1dc/18c87e62 misdirection + deb980db collapse; train: 6379e434/1571e1f8/39921f80 misdirection + def001f4/56ea03d9/16e0f30d collapse). INVARIANT NOW ENFORCED + verified: every uid in holdout_uids/train_uids is a member of band_corpus @ e0975e112f05d03e; 0 contamination, 0 dups. Mixed-label WARNING removed (no longer applies). The 9 band-exits remain in wellposed_all with their labels and stay recoverable via repair_patch_20260716.alias_map_applied if a mixed split is ever wanted. Net from original: holdout 100→94 (6 lost: 3 unrepairable + 3 band-exits), train 200→190 (10 lost: 4 + 6). holdout_papers with no surviving uid now 6 (was 3) — paper lists still untouched; flag if eval_paper_split cares. Backup .bak-pre-bandpure-20260716; corpus sha unchanged (split-only).

### Paper lists FIXED + **PRE-EXISTING CROSS-FILE CONTAMINATION FOUND** (Nicky "fix the paper lists too", 2026-07-16):
FIXED in corpus_split_200_100.json, recomputed from the band-pure uid buckets: **holdout_papers 90→84** (dropped 2508.12364, 2601.04324, 2603.12786, 2604.27278, 2605.11754, 2605.13389 — all verified to have ZERO band_corpus records; their only band rows died in the repair fold, nothing evaluable lost), **eval_papers 107→101** (= holdout ∪ anchor papers; the 17 anchor-only papers carry 0 band records BY DESIGN — anchors are solved/fail — and are RETAINED), **train_papers_n 167→163**. Verified: holdout_papers ≡ papers(holdout_uids), eval_papers ≡ holdout∪anchors, every holdout paper has ≥1 band record, 0 paper-level train∩holdout leak, split still band-pure, corpus sha unchanged. Left untouched deliberately: created/rng_seed/cascade_band_total(309) = freeze provenance, not current state. Backup .bak-pre-paperfix-20260716.
**⚠ UNRESOLVED, NEEDS NICKY — cross-file conflict (PRE-EXISTING, not patch-induced):** corpus_split_200_100 TRAINS on **21 papers that eval_paper_split.json declares eval-only** (its rule: "ANY record whose arxiv_id is in eval_papers is EXCLUDED from training"); 13 holdout papers also sit in that eval set. Verified 21 in the ORIGINAL frozen split (bak-pre-recover) — shipped this way. Cause: independent generation (corpus_split 07-15/seed 20260715 vs eval_paper_split 07-14/seed 20260714), never reconciled. **Impact: training on this split contaminates the paper-level eval.** NOT auto-fixed: resolution either drops ~21 papers of train records (material shrink) or declares the two protocols independent — design call. Recorded in-file under repair_patch_20260716.paper_lists_fix.UNRESOLVED_CROSS_FILE_CONFLICT.

### SPLIT UNDONE + RETIRED; LoRA = SKELETON ONLY (Nicky "undo split / only skeleton for LoRA, no split yet / single source of truth corpus", 2026-07-16):
**CORRECTION to this session's earlier 21-paper alarm: it was MOOT.** `corpus_split_200_100.json` has NO code consumer (`grep -rn corpus_split_200_100 --include=*.py` → nothing; referenced only in this ledger). It was an orphan artifact from 07-15, never wired into the harness — so the 21 train-papers-vs-eval_paper_split conflict contaminated nothing. The alarm was correct about the file and wrong about the stakes; recorded so the ledger isn't left with a false red flag.
**UNDONE:** corpus_split_200_100.json restored to its ORIGINAL frozen bytes (verified byte-equal; all three 07-16 patches — recover / band-pure / paper-lists — reverted, patch keys gone, holdout_n back to 100 / train_n 200).
**RETIRED** (non-destructive, files preserved) → `evalharness/data/retired_20260716/` with README: corpus_split_200_100.json + its 3 .bak stages, holdout_uids.txt, train_uids.txt (the latter two were stale copies of a build_eval_set OUTPUT, mistaken for inputs). `evalharness/data/` now holds only `eval_paper_split.json`.
**STILL LIVE + CORRECT:** `build_eval_set.py` already implements single-source-of-truth — sha-pinned eval_paper_split.json (110a4bf27320f2b1, verified) + derives train_uids from the corpus with leakage asserts, writing fresh output every run. Do not rebuild it.
**THE REAL GAP (skeleton's core):** build_eval_set pins the SPLIT sha but **NOT the corpus sha** — no corpus sha anywhere in evalharness/src. A corpus fold silently changes the derived sets. Corpus has moved 4× in 6 days.
**SKELETON WRITTEN:** `docs/lora_execution_skeleton.md` (slug lora-eval-execution) — C1 corpus sha pin, C2 split as derived view recorded by (corpus_sha, seed, rule) in a split_manifest, never a stored uid list; C3 band-purity/paper-disjoint/anchor asserts as code; C4 recommend keeping eval_paper_split frozen (an eval holdout SHOULD be frozen) but record its corpus-of-origin. Binds findings F1 repaired-grade-harder, F2 fabricated sharpness, F3 label instability, F4 dup-detect on FULL statements, F5 anchors aren't band by design. Nicky arms checkboxes at paste. Corpus sha e0975e11 unchanged throughout (no corpus writes this session).

### COMMITTED (Nicky "commit the skeleton and the retire", 2026-07-16): two commits, NOT pushed.
- **d99d38d** `evalharness: import untracked harness; retire orphan split artifacts` — 31 files. NOTE: this IMPORTS the eval harness authored 07-15 by a prior session that had never been committed (tests verified green, 42 passed, before import; .gitignore already excluded all cache junk). The 07-16 retire lands in the same commit because git had no prior state to diff the moves against. data/ now holds only eval_paper_split.json.
- **b093143** `docs: LoRA execution skeleton — corpus as the single source of truth` — the replacement for the retired split artifact class.
Verified at commit time: harness tests 42 pass, main three-suite 1118 pass, no live evalharness process, corpus sha e0975e11 untouched (zero corpus writes this session), nothing else swept in.

### STATUS ENDPOINT → SSH-TUNNEL-ONLY (Nicky's decision, 2026-07-25, executed same day): the loratrain RUNBOOK's box-side status server is no longer internet-facing. Box binds container-loopback (`run_remote_train.sh` http.server `--bind 127.0.0.1`, suite-asserted); pod exposes **SSH 22 ONLY** (§1.2 — no TCP 8000 mapping); operator polls through an SSH local-forward (`python3 -m loratrain.tunnel [--execute]`, new module, dry-run default, mocked-subprocess tested). `config.py` contract: TRAIN_SERVER_IP = ssh/scp target ONLY; TRAIN_SERVER_PORT = M4-local tunnel port; TRAIN_SERVER_URL = tunnel-local (validate_config now enforces that form); new non-operator constant TRAIN_STATUS_BOX_PORT=8000 drift-tripwired against the .sh. **Operator-editable block NOT touched** — RUNBOOK Appendix A carries the proposed diff (SSH_PORT field + tunnel-local URL derivation line) for Nicky; placeholder IP keeps everything green pre-application, real IP + unapplied diff fails validate_config loudly by design. RUNBOOK D-R1/§1/§6/§9/App A+B + README updated. loratrain suite 80→**95 passed** (single-source IP scan green); root suites untouched (`testpaths=["tests"]` never collected loratrain). All UNCOMMITTED per standing rule. Parallel session's §0.4-EXECUTED llama.cpp block left untouched.

### APPENDIX A APPLIED (Nicky "apply the Appendix A diff", 2026-07-25, tunnel lane): config.py operator block now tunnel-era — TRAIN_SERVER_SSH_PORT=22 added (per-pod external→22, set at RUNBOOK §1.3; TRAIN_SSH_PORT env demoted to fallback, config wins), TRAIN_SERVER_URL line rewritten literal tunnel-local `http://127.0.0.1:{TRAIN_SERVER_PORT}`, IP/PORT comments now say ssh-target-only / M4-local-tunnel-port. validate_config range-checks SSH_PORT when present (absence tolerated — env-fallback state). RUNBOOK D-R1/§1.3/App A flipped to APPLIED (§1.3 derives the TRAIN_SSH_PORT export FROM config); README snippet updated; upload_guard/tunnel docstrings de-staled; +4 tests (config-wins resolution, shipped default, SSH_PORT validation, absent-tolerated). Suite on merged tree: **127 passed** (tunnel lane 99 + W2 lane's 28 test_build_dataset, all green together). Still UNCOMMITTED.

### LoRA CAMPAIGN CLOSED (consolidated 2026-07-26→29 entry, written 07-29 doc-cleanup phase; the intervening lanes wrote to memory + campaign docs instead of this ledger — this entry backfills it):
- **Split ruled (Nicky 07-26):** the 07-25 fourth-lane `corpus_split_200_100.json` (sha `768436f4`, 200 train incl. 7 GGUF-7/8 TRAINING-ONLY backfill / 100 pure-band holdout, seed 20260726) is AUTHENTIC and WINS; frozen `110a4bf2` eval_paper_split formally retired to `evalharness/data/retired_20260726/`. **Known open item: 2 evalharness tests fail loudly on the retired path** (`test_build_eval_set.py` pins the old frozen split) — repoint is the evalharness lane's; measured 2 failed / 40 passed 07-29.
- **W2–W5 executed:** 700-row dataset (sha `7fa7e5bf`, 21-guard build); baseline 43/100 captured pre-training (llama-server b10107, fingerprint `b1-c0bc859`); RunPod A40 box (69.30.85.138:22092, ssh-tunnel-only) trained **12 control seeds** (r16/α32/drop.05/lr1e-4/3ep, eff-batch 16): run-1 20260722/3/4 → +11/+1/+3pp; stage-A 6-config HP screen on a 160/40 carve (control wins, no deviation significant, SE ±9.3pp); stage-R 20260725/26/27/29/30 → 0/+3/+1/+4/+1; D2 extension (Nicky-approved) 20260731/0801/0802/0803 → 0/−1/−1/−2.
- **FINAL VERDICT (n=12, pre-registered):** sign test 7+/3−/2 ties **p=.344**; mean **+1.67pp**, t(11)=1.675 **p=.122**, CI **[−0.45,+3.79]**. The interim n=8 read (sign p=.0156, t p=.046) did NOT survive its own extension — protocol's second self-correction. Deterministic repro audit PASSED (seed 20260726 re-eval: 46/100 exact, 0/120 row mismatches) → no instrument caveat. Authority: `docs/lora_consistency_verdict.md` (+ `lora_campaign_results.md`, `lora_decisions_2026-07-28.md` D1–D9, `lora_params_rationale.md`).
- **Known dataset defects (adjudicated 07-28, receipts in params_rationale):** full-sequence loss (~22% prompt mass), gradient weight = n_correct (anti-difficulty), grad-accum=4 hardcoded + TRL scheduler defaults unrecorded (now documented in both dataset manifests' `trainer_defaults`). Per decisions-doc D5 null branch: **v2 = completion-masking fix is the leading hypothesis + next experiment (Nicky-gated, ~$2-3)**.
- **Box TERMINATED-GO (07-29):** all 9 R/R2 GGUFs + 4 training logs pulled local (`/tmp/R_*.gguf`, `out/analysis/box_logs/`); anchor-adjudication evidence for 2 boundary anchors in `out/analysis/anchor_adjudication_20260728.md` (Nicky ruling pending). Campaign spend ≈$8 box total.
- **Commits (LoRA docs lane, NOT pushed):** d99d38d/b093143 (07-16 harness import + skeleton) · 243e364/b9278d9/f8520df/6761438 (loratrain arm + run-1, PUSHED earlier) · 1f6d5b9 (verdict docs) · f93b7e1 (n=12 reversal) · 3899aee (repro PASS) · this doc-cleanup commit.
- **Suite baselines re-measured 07-29:** root 975 passed/3 skipped · three-suite 1118 · loratrain 154 · evalharness 40+2 known failures (above).

### DATASET v2 BUILT + MASK-PROVEN (work order 2026-07-29, build+test only — NOTHING trained/uploaded/committed): both 07-28-review defects fixed in `src/loratrain`. **D1 (full-sequence loss):** builder now emits `prompt`/`completion` message columns (schema v2); pinned trl 0.29.1 gives completion-only loss for that dataset type, `completion_only_loss=True` set explicitly in the box trainer; total rendered text + token ids proven byte-identical to v1 across all 200 cap1 rows (loss mask is the ONLY train-time delta); decode-level proof (real trl tokenize→collate on real rows, tiny random Qwen3, CPU, 37/37 checks) in scratchpad `masking_proof.py` — prompt spans 100% -100, completion spans 100% loss incl. `<think>\n\n</think>` prefix + `<|im_end|>`. Trained-token composition: prompt 160,590 (23.8%) / completion 514,975 (76.2%) of 675,565; max example 2,648 tok — NO row near max_seq_len 4096 (silent-truncation sweep clean). **D2 (gradient weight = n_correct):** `config.WEIGHT_POLICY` knob (default `cap1`) + `--weight-policy`/`--cap-k` CLI; builds under `data/v2/<label>/`: cap1 **200 rows**, cap2 **362**, cap3 **487**, inverse **700 (+exact 1/n weights, trainer weighted-loss path proven vs hand math)**; cap selection deterministic by sha256(seed:uid:rollout_uid), rule recorded in manifest; **which policy ships = Nicky's call** (backfill question open: under cap1 the 7 near-ceiling GGUF-7/8 backfills carry 7/200 = 3.5% of gradient mass at equal weight — include/downweight/drop undecided). **Also pinned:** grad_accum=4 (was hardcoded literal), lr_scheduler_type=linear, warmup_ratio=0, weight_decay=0 → config.py + SFTConfig kwargs + run_config.json/dataset_manifest.json/job payload echoes (values unchanged from what v1 actually ran). `config.SFT_DATASET_PATH` repointed to the v2 policy-labeled build (upload_guard ships v2 when W3 reopens). New guards: prompt/completion wellformed + wire-format pins, weight-policy honored (re-checked from disk), `<think>`-tag refusal in targets (latent Qwen3-template rewrite hazard found during the proof; 0/700 affected today). v1 artifacts untouched (run1_final sha 7fa7e5bf verified); corpus e0975e11 / split 768436f4 untouched; out/** untouched. **loratrain suite 154→185 passed** (31 new tests; AGENTS.md baseline updated). Uncommitted per standing rule. Next (Nicky-gated): pick policy + backfill treatment, then RUNBOOK box round on v2 (~$2-3).

### RE-PAIR PREP (same 07-29 session, after Nicky "we'll need to re-pair with runpod — original box terminated"): config.py operator block RESET to placeholder (`TRAIN_SERVER_IP=127.0.0.1`, `TRAIN_SERVER_SSH_PORT=22`) — the dead pod's 69.30.85.138:22092 must not linger (RunPod reassigns IPs; a stale address is an scp-to-stranger hazard). Verified in un-paired state: suite 185 green, validate_config OK, upload_guard refuses on placeholder (designed pre-pairing refusal). Survives locally and stays valid for the v2 round: identity_receipt.json PASS (local GGUF unchanged, pins unchanged), baseline_greedy.jsonl 43/100 (same baseline serves v2 — base model/engine unchanged, adapter is the delta), v2 datasets under data/v2/, remote/ scripts v2-ready. Re-pair = RUNBOOK §1–§3 on a fresh A40 (console: Secure Cloud A40, PyTorch 2.x CUDA 12 template, ≥60GB disk, TCP 22 ONLY, attach SSH key; then set IP + external-port→22 in config.py §1.3) → §4 smoke → §5 upload → §6 train. DECISIONS OPEN BEFORE §5/§6: which weight policy ships (SFT_DATASET_PATH currently → data/v2/cap1/), backfill treatment, seed count for the round (run_config seeds default 20260722/3/4).

### v2 CAMPAIGN EXECUTED — cap1 × 12 paired seeds, box TERMINATED (2026-07-30 ~10:00Z)

**Release (Nicky, this session):** W3/W4 opened for the v2 round — cap1 ships, backfill records KEPT (7/200 = 3.5% gradient mass, decided not deferred), 12 seeds, **no spend ceiling**, terminate-on-success only (failure would leave the box up), **no local eval** (invariant-9 exposure while unattended), commit locally + ledger, notify on completion. Both open decisions from the RE-PAIR PREP entry above are now CLOSED.

**Seeds — v1-PAIRED, not fresh.** `upload_guard.write_run_config` derived seeds as `[SEED, SEED+1, SEED+2]`: capped any campaign at 3 and breaks across month boundaries (20260731+1 = 20260732, not 20260801) — v1's 12 controls were assembled across three separate runs, never from that expression. Replaced by explicit `config.SEEDS` = 20260722/23/24 (run-1) + 20260725/26/27/29/30 (stage R) + 20260731/0801/0802/0803 (D2). **20260728 deliberately excluded** — stage-A HP screen seed on a 160-record subset, not a control. v2 is therefore seed-paired with v1; per-seed sd 3.45pp now works for the comparison, not against it. `validate_config` rejects malformed/duplicate SEEDS locally rather than on the box.

**Box:** pod `b3njpwlrpbzh26`, A40 48GB Secure, CA-MTL-1, 60GB disk, TCP 22 only, no network volume. **Deployed from an IMAGE (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), not a template** — `list-templates` returns 0 items via MCP even with all include-flags, and every official template id 404s (incl. the one the MCP tool's own docs name). Same software; no template record. Worth checking whether the connected RunPod credential lacks template-read scope.

**Flow:** §0.3 identity PASS (every field + GGUF sha `a7676d25…` = D-R2 pin) · §2 env (see defect 1 below) · §3 base weights, box shas `f7c4eadf…`/`aeb13307…` = identity_receipt exactly · §4 Path A PASS (see defect 2) · §5 one guarded upload, cap1 sha `83b282f6…`, 200 rows · §6 12 seeds, ~1050 s each (39 steps, ~25 s/step), ~3.5 h wall · §7 all 12 GGUFs pulled and **sha-verified 12/12 against `status/artifact_shas.txt`** · §9 TERMINATED, `list-pods` empty, operator block reset.

**Result:** 12 adapters in `src/loratrain/data/adapter_seed*.gguf` (~175 MB each, 2.0 GB, gitignored) + `run_manifest.json`, `artifact_shas.txt`, `pip_freeze.txt`. Every manifest entry confirms the v2 recipe was live: `dataset_format=prompt_completion`, `completion_only_loss=True`, `n_examples=200`, `weighted_examples=False`, grad_accum=4/linear/0/0 recorded. Final train loss **0.3209–0.3243** across all 12 — extremely tight. **Do NOT compare that to v1's 0.431/0.433**: v1 averaged loss over prompt+completion (incl. the identical system prompt on every row), v2 over completion only. Different denominators, not a quality signal either way. **The adapters are UNEVALUATED — nothing here says v2 beat v1.** Next step is §10 per seed vs the unchanged 43/100 baseline; that is Nicky's call.

**Cost:** pod total **$4.39** ($4.31 GPU + $0.08 disk) for ~5.7 h of pod life (setup + 16 GB weight fetch + smoke + probes + 3.5 h train + 2 GB egress). Note the **effective rate ran ≈$0.76/hr against the $0.44/hr the API quotes** in `price.secure`/`cost` — worth reconciling before budgeting the next round off the quoted figure. Under the work order's ~$6 estimate regardless.

**Five defects found, reported not silently fixed (work order's standing instruction):**
1. **RUNBOOK §2 is wrong and would have poisoned the run.** A plain `python -m venv venv` does not inherit the image's torch, so pip resolved `torch 2.13.0+cu130` against the pod's 12.8 driver → `cuda False`. Training would have run on CPU (or crashed) *after* the dataset shipped. Fixed with `--system-site-packages`, which is what §2's own "torch = template's" intends → `torch 2.8.0+cu128`, `cuda True`, A40. **Amend §2's command block.**
2. **RUNBOOK §4's PASS criterion fails on its own smoke config.** `_SMOKE_HYPERPARAMS` epochs=4 on 8 examples = 4 optimizer steps, train_loss 4.58 — the adapter loads and shifts output but never emits the required `BANANA`, so a correct pipeline reads as FAIL. Proved Path A properly with a throwaway 40-epoch probe on a COPY (canonical trainer left byte-identical, sha `05eee7a1…`): loss 0.078, trigger → `'BANANA'`, neutrals sane (`'4'`, `'Pacific'`), adapter registered at scale 1.0. **Either raise smoke epochs or drop the "contains the dummy target" clause.**
3. **`run_remote_train.sh` reads/writes the manifest at the wrong path** — `$RUN_DIR/run_manifest.json`, while the trainer writes `out/run_manifest.json`. So `status.json`'s `completed_seeds` stayed `[]` all run, and **§6's crash-resume contract is broken**: a re-launch after a mid-run crash would retrain all 12 seeds, not skip completed ones. Didn't bite (no crash), live for the next round.
4. **§7's verification target doesn't exist.** It says adapter shas "MUST equal the shas in `run_manifest.json`/`status.json`" — neither carries shas. They live in `status/artifact_shas.txt`, which is what this round verified against.
5. **HEAD carried run-1's terminated pod address** (`69.30.85.138:22092`) from 2026-07-27 until this session: that round's §9 reset was made in the working tree but never committed. Now committed. **§9 should say commit the reset, not just make it.**

**Also:** RunPod reassigns the external SSH port on every container recreate — observed 22117 → 22145 → 22174 across a single restart; any stop/start invalidates `config.py`'s port. Noted inline where the value lives. The pod also needed `PUBLIC_KEY` set at create time (a fresh ed25519 keypair now exists at `~/.ssh/id_ed25519`, previously absent) — **RUNBOOK §1.2's "attach your SSH public key" is not optional and must happen at create, not after**, since fixing it later costs a restart and a port change.

**Suite:** loratrain **185 passed** before and after. Commit `9d1cdab` (v2 build + 12-seed enablement) + this teardown/ledger commit. NOT pushed.

### Q4_K_M-DEQUANT TRAINING BASE BUILT — T1–T5 (2026-07-30, orchestrator session; built alongside the LIVE v2cap1 12-seed eval, which ran untouched throughout)

**Mission (Nicky's ORCHESTRATOR_PROMPT.md):** additive training path where the LoRA base is the dequantized deployment GGUF (`Qwen3-8B-Q4_K_M`, sha `a7676d25…`) instead of the fp16 HF revision, so adapters are fit against the weights llama.cpp serves. The trainer is unchanged (`--base <dir>` already existed); everything here is the pre-step plus its verification. **The dequant path has NEVER been trained with or executed end-to-end — training on it needs an explicit release.**

**Landed (one commit; Fable orchestrator, Sonnet coders; every artifact adversarially panel-reviewed then attack-replay-verified to convergence):**
- `gguf_to_hf.py` (+tests): GGUF→HF fp32 via the pinned llama.cpp gguf-py (c0bc8591, path-import, never pip). All 399 tensors (217 Q4_K / 37 Q6_K / 145 F32) map bijectively; qwen3 conversion applies NO permutation (MRO-walk evidence recorded in the manifest); two-pass shard-streaming convert (peak RAM ≈ one shard, vs ~33 GB whole-model); atomic publish (temp sibling + rename; refuses non-empty/symlink `--out`; source hash verified BEFORE any work); deterministic (`content_digest` sharding-independent); self-contained safetensors writer (`__metadata__ format=pt`); config.json reconstructed from GGUF KV with sidecar cross-validation (rope_scaling/hidden_act/attention_bias/sliding-window gates; tokenizer count + special-token spot-checks vs the GGUF's embedded vocab); emits `dequant_manifest.json`. **Full-size local run DEFERRED post-campaign** (RAM/IO contention with the live eval); `--plan` (metadata-only) verified against the real pinned GGUF.
- `verify_dequant_parity.py` (+tests): **BLOCKING post-campaign gate, NEVER EXECUTED** — greedy token-for-token on the 10 anchor_solved prompts, llama-server (base, §0.4 flags, NO `--lora`) vs transformers (fp32, eager, pinned). `--mode raw|chat|both`, **default `raw` = THE gate**: native `/completion` (no server-side parsing) + tool-side template render → byte-exact. The chat endpoint was proven structurally lossy from b10107 source (parser drops the empty `<think>` block every `/no_think` anchor emits, and consumes template whitespace) — chat mode is an informational cross-check that never affects the exit code. Campaign guard (pgrep + port halves, re-checked before EVERY server call; `--i-own-the-qwen-slot` override; allow_abbrev off); identity binding (manifest sha ≡ pin; `--expected-alias qwen3-8b-q4km-base` recommended); exit 0/1/2 = pass/divergence/inconclusive; report carries full adjudication detail (raw + reasoning channels, finish_reason, token divergence, environment).
- `verify_base_identity.py`: new `--dequant-dir` hash-chain identity mode (on-disk GGUF re-hash ≡ manifest ≡ pin; per-file re-hash with lexical+resolve path safety; **zero symlinks tolerated** under the dir; unreadable-dir fail-closed; orphan/completeness census; recomputed content_digest; `permutation.applied` must be false; receipt gains `scheme` + `chain{…, source_verified}`). New `--compare-runs` cross-scheme tripwire (scans ALL seed entries; mixed/partial/non-string schemes refuse; smoke-marked entries excluded; `base_source_sha256` compared; exit 0/2/3 = match/substantive/infra). fp16 path byte-compatible (receipt gains only `scheme`).
- Base-scheme provenance (T4): `config.BASE_SCHEME` (`fp16_hf_revision` default — **shipped behavior unchanged**; `dequant_q4km` opt-in), run_config/run_manifest gain `base_scheme` + `base_source_sha256` (additive-only, test-pinned), upload_guard chain enforcement (scheme match; under dequant the receipt must be `source_verified: true` with sha ≡ pin — `--skip-file-sha` receipts refuse at upload), box-trainer startup gate (refuses a `--base` dir inconsistent with the configured scheme), smoke manifest entries now `"smoke": true` with NO fabricated scheme.
- `remote/run_remote_train.sh`: **logged defect 3 FIXED** (manifest path → `$RUN_DIR/out/run_manifest.json`; crash-resume now real) + resume predicate hardened (manifest entry AND final gguf AND sha line; convert = tmp→sha→mv atomic; RUN_DIR trailing slash normalized; stale sha lines pruned). `train_qwen3_lora.py`: `_append_manifest` atomic + replace-same-seed + corrupt/wrong-shape hard-fail.
- `RUNBOOK.md`: **§3-ALT DRAFT (NEVER EXECUTED)** — fetch the pinned GGUF repo (`lmstudio-community/Qwen3-8B-GGUF` @ `07ebe812`) ON THE BOX (upload_guard's dataset-only allowlist untouched), sha-verify, dequant on box (`--sidecar-dir` = the §3 fp16 snapshot), then train `--base` the dequant dir; identity/provenance/parity-gate steps included. Logged defects **1/2/4/5 folded** (§2 `--system-site-packages`; §4 smoke-criterion discrepancy documented with the probe-proven criterion — the code-side fix still needs a release; §7 sha target `status/artifact_shas.txt` + manifest scp path corrected; §9 commit-the-reset; §1.2 SSH-key-at-create + port-reassignment notes). README pointer added.
- `AGENTS.md`: loratrain baseline **185 → 581 passed + 2 env-dependent skips** (same-commit rule; AGENTS.md sits outside the mission's write-list but is read by no eval process — disclosed).

**Verification:** adversarial review panels + attack-replay rounds per artifact until convergence (T3/T4: 3 rounds — reproduced chain-bypass blockers caught incl. path-escape/symlink-facade/self-attesting-manifest; T1: 2 rounds — no core-logic defect ever found, operational hardening only; T2: 2 rounds — the panel forced the raw-primary pivot with b10107 parser receipts). Final bypass probes re-run by the orchestrator's own hand. Merged suite **581 passed + 2 skips from disk** post-integration.

**Eval-unharmed evidence:** campaign progressed continuously all window — preflight seed 20260722 at 109/120; at commit time seeds 20260722–26 all 120/120 DONE, 20260727 mid-flight; zero FATAL/FAIL in the driver log; `out/**`, `src/icepick/**`, `evalharness/**`, `src/loratrain/data/**` untouched; no installs, no servers started, no processes signaled; all builds in worktrees off 9a823ed.

**Open / for Nicky:** (1) execute the deferred gates post-campaign: full-size dequant + `verify_dequant_parity` raw mode (both refuse while an eval owns the Qwen slot); (2) **bf16-at-load question**: the box trainer loads bases as bf16, which rounds the fp32 dequant grid at load — force fp32/fp16 load for the dequant scheme, or accept bf16? (flagged in gguf_to_hf's docstring, undecided); (3) known residuals, all fail-closed/low: resume skip-check doesn't key on run_config sha; hardlink-to-outside passes the symlink gate (sha still verified; acknowledged in docstrings); chat-only parity report says `verdict: PASS` where NOT_GATED would be clearer; `--mode both` all-empty guard is per-run not per-channel; `base_source_sha256` is a 40-hex git rev under fp16 vs 64-hex sha256 under dequant (documented); (4) §4 smoke-criterion code fix still needs its release; (5) push release for this commit when wanted (NOT pushed).

### Mission claim — proof-import (ACTIVE, session f7b24506, 2026-07-31 ~18:55Z)
Nicky pasted `docs/proof_import_execution_skeleton.md` = R1 released. Rulings taken: R2 = default (train-split 200 uids incl. 7 backfill), R3 = fetch released via standard gates (plan + approval marker in run dir), §1b substrate = new CPU RunPod for P1–P3 mechanics, **P4 Sonnet calls + P5 verification LOCAL** (key + answer-key custody). Run dir: `out/proof_import_20260731T185338Z/`. Pins verified at open: corpus 293/e0975e11 · wellposed_all 2021 · split 768436f4 (200 train / 100 holdout / 7 backfill) · HEAD 5be86ea (other lane's uncommitted loratrain edits left untouched). Parallel k=8 sweep session's A40 pod `skvfqhr0l5ilve` + its ssh/grading processes SEEN — not mine, untouched; my pod will be a separate cheap CPU instance, terminated same session. Guards: zero holdout uids in any pipeline file (asserted local at P1 pack + P5 publish; holdout list never ships), zero corpus mutation, ≤$5 total spend hard line. Suites: pre-change three-suite baseline being measured; new module `proof_mine.py` + tests expected to raise the count — will be re-measured at close.

### CLAIM — lora-v3-proofhint window (2026-07-31 ~19:05Z, fresh session)

Nicky pasted `docs/lora_v3_proofhint_execution_skeleton.md`. Scope ruling applied (inv-11): the paste releases the **v3 arm** at default checkboxes; it does NOT transitively arm `docs/proof_import_execution_skeleton.md`'s own R1/R3 (fetches + paid Sonnet are that skeleton's separate release surface) — running its **$0 P1 inventory only**. State at open: hard dependency `out/proof_import_*/solutions_v3.jsonl` ABSENT (proof-import never ran); dq k=8 verdict NOT YET (sweep 11/34 configs pulled; pod `skvfqhr0l5ilve` RUNNING, owned by the k=8 session — untouched by this lane). This window: (a) `v3.py` P1 module + tests (Sonnet coder, §0-isolated: new files + marked config append only), (b) proof-import P1 inventory → `out/proof_import_20260731T190000Z_p1inventory/`. NO fetches, NO paid calls, NO pods from this lane. Pins verified at open: corpus 293/e0975e11, split 768436f4, engine c0bc859, wellposed_all 2021, 8eefdbb ancestor. Working-tree `M config.py`+`RUNBOOK.md` = dq-arm state, untouched.

### CLOSE — lora-v3-proofhint construction window (2026-07-31 ~20:5xZ)

**Built + committed `2afbba9`** (main ahead 7, NOT pushed): `src/loratrain/src/loratrain/v3.py` (1614 lines) + `tests/test_v3.py` (45 tests) + `PREREGISTRATION_V3.md` + additive marked config section (staged append-hunk-only — the dq arm's `BASE_SCHEME` flip remains uncommitted working-tree state, untouched). Suite: loratrain **617 passed / 2 skipped** (+45); the 9 failures before AND after are the dq flip's known set (shipped-default + 8 upload_guard) — restored automatically when that arm flips back. Fixture dry-run through the real CLI + real icepick verify chain: all censuses exact (first-verified-wins, hint-insufficiency, off-tier exclusion incl. a real solved-tier backfill uid, deterministic anchor, idempotent republish); real-input probe refuses cleanly on the still-incomplete proof-import publish. Nothing launched: no pods, fetches, or paid calls from this lane ($0).

**Gates before P2+ (regen pod onward):** (1) canonical `out/proof_import_20260731T185338Z/solutions_v3.jsonl` + `manifest.json` complete (that lane owns it; the builder's manifest key-shape tolerance must be pinned against the first real publish); (2) dq-vs-v2 k=8 verdict (sweep mid-flight, pod `skvfqhr0l5ilve` untouched by this lane) → R4 confirm or Nicky re-rule; (3) Nicky R1–R5 (defaults armed by the paste; eval-sweep stage ≈$6 > $5 needs an explicit ask regardless). **Two Nicky decisions surfaced:** (a) v3's default 60/40 collapse/band curriculum is unsatisfiable under proof-import's band-only R2 default — the 58 net-new collapse/misdirection train records (~$0.29 extra Sonnet) need R2-extended, else the blend records ~100/0 band (observed-not-forced by design); (b) anchor-quota boundary: if >150 of 200 train records verify with hints, the 25% anchor from v2-cap1 under global cap1 is unsatisfiable → build refuses (fail-closed) — pre-decide the fallback if that trips. Cross-lane artifacts: `HANDSHAKE_v3_window.md` in the canonical dir; duplicate inventory dir marker-flagged `NOT_THE_CANONICAL_LANE.md` (its counts independently confirm the canonical P1; stale-label trap: 34 pool labels outdated vs band_corpus — pool never refreshed post-gguf_rescore).

### RULINGS — lora-v3-proofhint window (Nicky, 2026-07-31 ~21:0xZ)

1. **R2 EXTENDED (released):** proof-import scope grows to the collapse/misdirection tier (58 net-new recs / 42 papers, zero new fetches, ≈$0.29) — mechanism rationale: collapse records are where imported information is genuinely new; 100/0 band would test a weakened hypothesis indistinguishably-nullable. Relayed to the import lane verbatim: `out/proof_import_20260731T185338Z/RULING_R2_EXTENDED_20260731.md` (+ caveats: achieved mix ~30/70 recorded-not-forced, and collapse wellposedness is UNAUDITED — flag suspicious records during P4, don't silently import solutions to defective problems).
2. **Anchor-quota:** option (c) — refusal stands, decide with the real verification rate in hand. Pre-signal for when it trips: **(b) shrink the anchor, never (a) break global cap1** (cap1 is load-bearing; breaking it reintroduces the per-record weighting defect v2 existed to fix).
3. **R4:** nothing to do; Nicky's pre-committed prediction: dq ≈ v2 (~75–80% likely) ⇒ fp16 most likely stands. Verdict still gates training.
4. **Eval-sweep ≈$6 PRE-AUTHORIZED, conditional on the gates** (solutions complete + dq verdict read + R1–R5 state). Cost context: this sweep ~$15 by completion, campaign ~$26 all-in — eval sweeps now dominate cost, not training.
5. **Push:** recommended yes in the exchange but the explicit release word is being confirmed separately — not pushed yet.
6. **Doc commit released** ("fold in whenever"): steering prompt + this ledger committed; `Project_retrospective.docx` is NOT part of it (untracked, unknown provenance, left alone).

### Mission CLOSED — proof-import (session f7b24506, 2026-07-31 ~21:0xZ)
**107 verified worked solutions published** → out/proof_import_20260731T185338Z/solutions_v3.jsonl (53.5% of the 200 train uids; 105 band + 2 backfill). Census v2 (canonical): 119 matched / 73 no-proof / 7 no-proof-env / 1 not-found / 0 omitted-class; P4: 107 faithful / 10 refused (stubs + 2 misattribution catches) / 2 parse errors; P5: 107/107 verified, 0 rejects, holdout 0-overlap, corpus+split shas re-verified untouched. Spend $3.08 ($3.01 Sonnet + $0.07 pod, terminated 204). Suites: three-suite **1167** (baseline 1118 + 49 proof_mine tests). NEW UNCOMMITTED: src/icepick/allocation/scrape/proof_mine.py + tests/allocation/scrape/test_proof_mine.py (import-only; realmath byte-untouched; includes containment/"nested" matching — recovered 4 rows + fixed 1 silent wrong adjacency match 58134ca4). Full narrative, deviations (6, all disclosed), and the §4 Nicky queue: out/proof_import_20260731T185338Z/REPORT.md. Downstream: ready for docs/lora_v3_proofhint_execution_skeleton.md dataset build; R2 sizing (107 rows) + unmined-81 + refusal-12 dispositions are Nicky's.

---

## WINDOW 13 CONSOLIDATED CLOSE — k=8 sweep, corpus census, split rebuild, scoring spec (session c72b4b02, 2026-07-31 → 08-01)

**A cold window can resume from this block plus `docs/v3_full_run_skeleton.md` alone.**
Everything below was verified against disk/API at the time of writing; re-verify before acting
(this environment has delivered fabricated completion events).

### 1. The k=8 sweep — HALTED and TERMINATED, results preserved
Rented A40 `skvfqhr0l5ilve` to re-measure every adapter at pass@k=8 (the greedy instrument was
the weak link: the holdout is 100 PURE BAND records, i.e. selected to be ~coin-flips, so
greedy pass@1 measured one flip per record — **and the 43/100 baseline was itself one flip
per record**). Completed **17 of 34 configs** before Nicky halted it (the eval set is being
rebuilt, so remaining configs would measure a soon-to-be-retired anchor).

- Preserved: `out/passk8_sweep/` — 19 rollout files, **17,206 generations**, box-graded, sweep log.
- **Base ruler at k=8:** of the nominal "100 band" holdout only **70 still measure band** (16 solved,
  14 collapse/misdirection) — **30% label drift**. Anchors validated the instrument (10/10 solved,
  fail-anchors 0.10/8).
- **v1 (n=9): mean Δn_correct −0.184** · band→solved 6.3 vs band→collapse 7.7.
- **dq (n=7): mean Δn_correct −0.197** · 7.7 vs 6.9. **dq ≈ v1 ⇒ the train/serve quantization
  mismatch is NOT the limiter** (pre-committed prediction held at 75–80% confidence).
- v2 never ran (0/12), so the dequant lane's pre-registered dq-vs-v2 paired test is **not
  computable** and stays that way.
- Spend ≈ $8.20. Pod terminated; zero GPU pods remain.

### 2. Instrument findings (each cost real money or a corrupted run)
- **`-fa auto` is the build default.** The entire reference set was measured on auto-resolved-OFF;
  explicit `-fa off` verified byte-identical (3/3), `-fa on` produces different generations.
  **Always pass `-fa off` explicitly.**
- **Grading needs `antlr4-python3-runtime==4.11`.** A pod grader without it silently mis-scored
  ~70/120 records per config (sympy's LaTeX parser fails closed). Verified recipe: venv +
  `sympy==1.14.0` + antlr4 + the FULL `src/icepick` tree; **parity-check any re-homed grader to
  ZERO diffs before trusting a number.** Box grading is now live and parity-verified 10/10.
- **CUDA-vs-Metal: 0/3 byte-match** (dequant lane) — box eval is invalid against Metal-measured
  greedy numbers, but cancels within a same-box sweep.
- **INCIDENT (owned):** a single `ps` snapshot landed in a **3-second gap** between two seeds and
  was read as "campaign dead"; port 8081 was then seized twice, corrupting seed 20260727
  (≥32 of 94 records were base-model outputs). Quarantined intact at
  `out/evalharness/QUARANTINE_20260730_contaminated/`. **A `ps` snapshot is not proof; read
  sibling progress logs and verify server identity via `/v1/models` before taking a port.**

### 3. Corpus census — COMPLETE, nothing unassessed (921 records)
Ran corpus-wide proof mining. Fetch telemetry clean throughout (0×429, 0×503, 0 failures).

| tier | proof-bearing | proofless | total |
|---|---|---|---|
| band | **187** | 129 | 317 |
| collapse | **217** | 186 | 405 |
| misdirection | **87** | 104 | 199 |
| TOTAL | **491** | **419** | 921 |

- First-pass extraction ≈ **52.9%, flat across tiers** (band 58.6 / collapse 53.6 / misdir 48.7).
  An earlier "hard records extract worse" claim was an ARTIFACT of mixing in prior lanes' failed
  residue — **corrected**.
- **Proof availability is independent of difficulty**: mean n_correct 3.19 (proof-bearing) vs 3.23
  (proofless), Mann-Whitney **p = 0.918**. So the proof-bearing/proofless split adds no difficulty
  confound — measured, not assumed.
- Published so far: **139 rows** across three lanes (`proof_import_20260731T185338Z` 107,
  `_collapse_20260801T001803Z` 22, `_band2_20260801T012050Z` 10). Import spend **$4.41**.
- **Verifier defect found:** `verify()` computes `simplify(candidate − truth) == 0`, and
  `simplify(oo − oo) = nan`, so **every infinity-valued answer is ungradeable by construction** —
  21 records in scope carry `fail` labels that are bug artifacts, not difficulty. eval_set was
  0/120 clean and train 0/200 clean, so **no campaign measurement was affected**. Fix is one-line;
  do it BEFORE the new anchor is measured.

### 4. RULINGS (Nicky, binding)
- **The old 200/100 split is VOID. Holdout no longer exists.** Rebuilding from scratch.
- **Split rule:** proof-bearing → training, proofless → eval. `solved` excluded as useless;
  `drop` discarded as having failed posedness.
- **Training set OVERBUILT to 468 rows**: all 187 band (40.0%), collapse 194 (41.5%),
  misdirection 87 — exhausted (18.6%). The 40/60 band:fail ratio holds exactly; the 30/30 split
  *within* the failure side cannot, so collapse absorbs misdirection's 53-row shortfall.
  **341 net-new Sonnet calls ≈ $9.85.**
- **Eval set 322 rows**: ALL 129 proofless band + 97 collapse + 96 misdirection.
- **Scoring: gate-crossing metric** — see `docs/gate_crossing_scoring_spec.md` (authoritative).
- **Execution substrate: RunPod. ALL eval on pod, generation AND grading.** Two local carve-outs
  for cause: API-key custody (pod env vars are readable through the RunPod account API) and data
  selection from census artifacts.
- **Pipeline:** baseline (4 pods, ~1.1 h) concurrent with training (1 pod, ~9.3 h critical path);
  **baseline pods TERMINATE as soon as the ruler lands** (~$14 saved vs idling); then 12 arm evals
  fan across 4 pods (~17.1 h). **~26.4 h wall, ~82 pod-hours ≈ $36.**
- **Eval sharding stratified** — every pod carries the same 40/30/30 mix. **Ingest in 10-record
  intervals** (4/3/3), each a checkpoint: 160 gens ≈ 8 min, so a crash loses an interval not a shard.

### 5. Scoring spec — written, then REVISED after adversarial review
`docs/gate_crossing_scoring_spec.md` counts **problems whose solvability status changed**, ±1 each,
net = effect size and a sign test over non-zero problems = p-value. Review found real defects; all
fixed and verified by computation, not argument:
- **BLOCKING:** the k=16 gate silently halved `BAND_LO` — **1/16 = 0.0625 is FAIL per
  records.py:105-110; band starts at 2/16.** The code-faithful gate ALSO zeroes a systematic
  null-arm drift the wrong gate caused (**−9.8 per 100 records at p=0.05 → +0.0 at every p**).
  One fix, two findings.
- **BLOCKING:** base pinned at **k=16** throughout (had been stated both ways).
- **BLOCKING:** the sign test's null is calibrated **A/A on the base ruler's own two k=8 halves**
  (free, same instrument) rather than assumed. **Two-sided α = 0.05 DECLARED.**
- `solved` records now **scored −1 on regression** — excluding them created the one-directional
  bias the exclusion claimed to prevent. 20%-regression trigger added as an instrument guard.
- Magnitude threshold **|Δ| ≥ 4/16**, chosen from measured false-positive/power analysis: 2/16,
  3/16, 4/16 are statistically equivalent (power÷noise 0.137/0.145/0.148), so 4/16 wins on
  defensibility (FP 21.5% vs 37.7% vs 59.7%). **Bulk of signal is expected from gate crossings,
  not band fluctuation** — report the two as separate lines.

### 6. RESUME POINT
**Paste `docs/v3_full_run_skeleton.md` into a fresh window.** It carries the pipeline, sharding,
intervals, per-pod verification and the scars, and self-references `AGENTS.md` →
this ledger → `gate_crossing_scoring_spec.md`. **All five release checkboxes (R1–R5) are BLANK —
arm them or the window will correctly refuse to spend.** Open decisions: R1 ($9.85 Sonnet),
R2 (eval 322 vs ~200 — cutting saves ~40% GPU), R3 ($36 GPU), R4 (verifier infinity fix first —
recommended yes), R5 (seed count).

**Git: main is 6 commits ahead of origin, NOT pushed** (`aa9c7f4` → `97903ea`). Nothing is running,
no pods exist, no spend is in flight. Campaign spend this window ≈ $12.61 ($8.20 GPU + $4.41 Sonnet).

## WINDOW 14 OPEN — v3-full-run (session 420fd8e4, opened 2026-08-01T05:2xZ)

Nicky pasted `docs/v3_full_run_skeleton.md` = mission open; **R1–R5 blank on disk at paste ⇒ HELD
at §1** (inv 11/12). §0 verification COMPLETE, all pins MATCH: corpus 293/`e0975e11` · pool 2021 ·
GGUF full-sha ≡ `a7676d25…8f35f` (recomputed) · engine `c0bc859`/b10107 · published 139 = 107+22+10
(`solutions_v3.jsonl` × 3 lanes) · three-suite **1167/0** · loratrain **623 passed + 2 skips + 9
FAILED** — all 9 are fp16-default tests broken by the dq lane's uncommitted `BASE_SCHEME=DEQUANT`
flip (config.py's own comment prescribes flip-back now that the dq campaign is closed; treat as P4
pre-flight, restores the 9; no other regression). Deltas vs the W13 close note: main is now
**PUSHED, 0 ahead**, with 2 further commits (`d951c98` close, `8a60575` proof_mine); working tree
still carries the dq lane's `M config.py` + `M RUNBOOK.md` (RUNBOOK delta = the 07-30 parity-gate
criterion amendment — commit-worthy, Nicky's call). Environment: 0 pods (RunPod API), no procs; two
stale self-expiring SSH watchers poll the dead sweep pod's IP (other sessions' shells, left alone).

Findings surfaced at open: (1) **`src/loratrain/PREREGISTRATION_V3.md` is instrument-orphaned** —
pinned to the VOID 120-record holdout (split `768436f4`) and to grading-local; and its primary
(v3 vs v2) is unmeasurable as written: v2 has NO k=8/k=16 numbers on any current instrument (0/12
in the halted sweep), so a head-to-head needs +12 v2 arm evals ≈ +$32. Dated amendment required
BEFORE any P5 read; primary-comparison fork put to Nicky alongside R1–R5. (2)
`p5_verify_publish_corpuswide.py` lives in `out/proof_import_full_20260801T014500Z/tools/`, not
lane 1's tools dir as the skeleton implies. (3) regen bundle
`out/lora_v3_proofhint/bundle_20260731T205043Z` predates the 468-row overbuild ruling → rebuild at
P3. (4) The "full" and "withheld" run dirs are W13's census lanes (4th/5th), complete, no P4/P5 —
reconciled, no unknown session. $0 spent this window. NEXT: Nicky's R1–R5 (+P6 primary) rulings.

## WINDOW 14 EXECUTION TAKEOVER — v3-full-run (v3-arm session, 2026-08-01 ~08:2xZ)

Nicky, in-chat to this session: **"Read and execute v3_full_run_skeleton.md"** ⇒ release of the skeleton as written: **R1 $9.85 Sonnet + R3 ~$36 GPU (4 concurrent pods approved) + R2=322 default + R4=YES (recommended) + R5=12 seeds**. Session `420fd8e4`'s WINDOW 14 OPEN entry above HELD correctly on blank checkboxes; its §0 verification (all pins ✓, GGUF full-sha recomputed, 0 pods) and 4 surfaced findings are **adopted wholesale**. This session (the W12 v3-construction session) now owns execution — `420fd8e4` must re-read this ledger before resuming anything. Notes of record: (1) eval composition = **129 band / 97 collapse / 96 misdirection = 322** — the skeleton's allocation table says mis 97 but its own sharding table (24×4=96) and the W13 ruling say 96; 97 would total 323; using 96, deviation documented. (2) `BASE_SCHEME` flipped back to fp16 per its own comment's rule + the dq verdict (dq≈v1); restores the 9 fp16-default tests. (3) PREREGISTRATION_V3 amendment (instrument-orphaned per 420fd8e4's finding) will pin **primary = per-seed net gate-crossing score vs the A/A empirical null** — the v2 head-to-head is unmeasurable within the approved budget (v2 has no k=8/16 numbers; +$32 if Nicky wants it as an add-on) — written BEFORE any P5 read. WAVE 1 (background, disjoint): **A** = P1 trainset completion (341 net-new P4/P5, hard cap $12, new run dir); **B** = R4 verifier infinity fix (+tests, own pytest subtree, private basetemp); **C** = P2 split build+freeze (paper-level disjointness HARD, deterministic tier selection, former-holdout provenance) + v3.py/config pin migration (holdout branch retired, unknown-uid kept).

### RULINGS — v3-full-run scope changes (Nicky in-session, 2026-08-01 ~18:1x–18:4xZ) — CORROBORATION OF RECORD

For any lane verifying orchestrator messages against this ledger:

1. **One seed at a time** (Nicky: "Let's focus on one seed for now / One at a time"): stage 1 trains ONLY seed 20260901, on baseline pod1 (x8yr84ks68jlv6), driven by the parallel-training lane. **The regen lane's pod (43eq413ms90456) mission ENDS at the dataset build** — regen → pull → local build-dataset → final report + READY_FOR_TERMINATE; NO RUNBOOK §3–§7 on that pod, no base-weights fetch, no upload, no training. Prereg Amendment 3 (commit 4c00c86) records the staging semantics.
2. **Seedless pass@k for arm evals** (Nicky: "Seedless pass@k"): arm generation runs `box_generate.py --unseeded` (requests omit the seed field; rows record seed=null). Base ruler stays as-generated (seeded). Prereg Amendment 4 (commit daa54b1).
3. Fresh seed cohort 20260901–12 committed at 0cf297f; 12-seed confirmatory design remains registered but UNSCHEDULED — further seeds are a separate future release.
4. Baseline pods 1–4 stay up after ruler drain: pod1 trains stage-1; all four then serve the record-bound arm eval (prereg Amendment 2, commit 407f128).

### AUTHORIZATION — upload_guard accepts v3-shaped dataset provenance (orchestrator, 2026-08-01 ~19:1xZ)

Blocker: `upload_guard.validate_dataset()` → `build_dataset.assert_verified_correct` requires v1/v2 provenance (`verdict`/`verbatim_output`/`rollout_uid`); the v3 dataset's rows carry v3 provenance (`verify_receipt`/`regen_sample_idx`/`proof_raw_sha`/`source_tier`) by reviewed design (v3.py's `verify_written_v3_dataset` docstring records the deliberate fork). The parallel-training lane is AUTHORIZED to extend `validate_dataset()` with a per-row provenance-shape dispatch: v1/v2 shape → existing check byte-unchanged; v3 shape → inline equal-strictness checks (uid present; `verify_receipt.verified is True`; `regen_sample_idx` int ≥ 0; `proof_raw_sha` 64-hex; `source_tier` ∈ {band, collapse}; prompt does NOT contain `config.V3_HINT_MARKER`; prompt/completion wellformedness via the existing build_dataset helper). NO import of `loratrain.v3` from upload_guard (isolation: nothing existing imports v3 — use config constants + inline checks). Tests required for both branches + refusal cases; suite from 665+2. Orchestrator reviews + commits after; the stage-1 upload may proceed on the working-tree guard once its own tests pass.

### DOCS — split ruling's cited record backfilled (docs session, 2026-08-02 ~00:2xZ / still 2026-08-01 local)

An agent auditing a former-holdout publish task found that the split ruling's cited written
source — `split-rebuild-2026-08-01.md`, named at `docs/v3_full_run_skeleton.md` §0,
`docs/gate_crossing_scoring_spec.md`'s footer, and six `loratrain/v3.py` sites (incl. two
runtime refusal strings) — existed NOWHERE in the repo: the name is a session-memory note
kept outside it. The ruling itself was never in doubt (verbatim across the citing files;
enforced by `7510b2a`; eval=286 accepted in prereg Amendment 1 `c581ff9`), but a reader
verifying a leakage-relevant ruling by opening the cited file would have found nothing.
**Backfilled as `docs/split-rebuild-2026-08-01.md`** (committed with this note): ruling,
frozen-artifact composition + full sha256 (re-verified from disk), code enforcement,
consequences, provenance. Docs-only, additive; `config.py`'s RETIRE-NOTE describes the
ruling without the filename and is untouched; no citing file edited. Root suite 1035
passed + 3 skipped before and after. Same-class residue NOT fixed here (flagged for a
future docs pass): the spec footer also cites memory names `gate-crossing-metric.md` and
`verifier-self-verify-defect.md` that likewise don't exist in-repo.

### GUARD — upload_guard v3 branch made fail-CLEAN (fail-safe gap closed; guard lane, 2026-08-02T00:39Z / 2026-08-01 ~17:39 local)

Closes the training-ops review finding on commit `72cfc39`
(`out/v3_full_run_20260801/opslog_train4x.md`, "fail-safe gap"): malformed
prompt/completion shapes could escape `_validate_v3_dataset` as
`build_dataset.TraceIntegrityError` — or as bare KeyError/IndexError/
AttributeError from `_assert_v3_row`'s direct `row["prompt"][1]["content"]`
indexing — through `main()`'s `except UploadRefused` (raw traceback instead of
the module's documented refusal contract). Was fail-safe (crash before any scp),
now fail-clean. Changes, all inside the v3 region (diff hunks at old lines
217/239/258 only; legacy branch + `_check_manifest_corpus_sha` byte-unchanged):
(1) `_assert_v3_row` — provenance must be a dict, prompt access shape-guarded
before indexing; every malformed shape refuses naming the row, even when called
without the wellformedness pre-pass; (2) `_validate_v3_dataset` — wellformedness
now runs per-row (behavior-identical: the check is a pure per-example loop, and
`build_dataset` itself calls it one-example-at-a-time), with
`TraceIntegrityError` and any non-`UploadRefused`/`LeakageError` exception
wrapped into `UploadRefused` naming the row; `LeakageError` still propagates as
itself. +8 regression tests in `tests/test_upload_guard.py` covering the
reviewer's shapes (missing prompt / single-message / missing content / non-dict
messages / malformed completion / policy-failure class conversion / direct
`_assert_v3_row` indexing / non-dict provenance). Suite (private basetemp):
**672 passed + 2 skipped before → 680 passed + 2 skipped after**, both measured
this session. **UNCOMMITTED per hold — no commits without Nicky's release**; the
release commit should also update AGENTS.md's loratrain baseline (still says
581+2, stale — 672+2 at HEAD `918584b` even before this change; 680+2 with it).
FLAGGED, NOT changed (outside this task's row-access authorization): the v3
branch's non-row accesses still violate the exception contract the same way —
malformed `dataset_manifest.json` → `json.JSONDecodeError`, missing
`V3_SPLIT_PATH` → `FileNotFoundError`, split JSON without `train_side_uids` →
`KeyError` (legacy path wraps its manifest parse via `_check_manifest_corpus_sha`;
the v3 branch parses inline). Needs its own authorization. $0 API, no launches,
no Qwen, no pushes; parallel-session `RUNBOOK.md` edit + `Project_retrospective.docx`
left untouched.

### AUTHORIZATION — v3b anchor_solved class (orchestrator, 2026-08-02 ~01:2xZ; Nicky's release: proof-injection anchors, ≈$8 approved)

Two authorized deltas for the v3b anchor path, orchestrator-implemented main-session (the established guard-edit pattern): (1) `loratrain/v3.py build-dataset` gains an explicit `--anchor-solutions` input — rows from it are constructed like hinted rows but bucketed `source_tier="anchor_solved"`, membership-checked NOT-in-eval (uid ∉ eval_set uids AND paper ∉ split eval_papers — these records are side-excluded solved-tier by the split ruling, repurposed as anchors per Nicky 2026-08-02), censused separately, never counted in the 60/40 blend arithmetic; the default (no flag) is byte-identical to current behavior. (2) `upload_guard._assert_v3_row` source_tier allowlist gains "anchor_solved". Tests both sides; suite from 672+2 (+ the adopted fail-clean fix's tests = 98-file baseline).

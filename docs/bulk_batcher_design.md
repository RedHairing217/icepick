# bulk-batcher — design (W1)

**Status: DESIGN — system ships DISARMED; Nicky arms it.** Built 2026-07-07 against
HEAD `313941a` (baselines re-measured green: root 607/3, three-suite 728).
Mission: while an arxiv_bulk extraction runs, every 250 accepted records become a
batch (batch10+), auto-queued through Sonnet-only cascade → pass@k, landing
fold-ready. Folding stays manual. Absolutes: EXACTLY 250/batch (remainder HELD);
0% record duplication across the new batches and all history.

## W0 verdicts (disk-verified 2026-07-07)

1. **Tail `<run>/_progress/candidates.jsonl`.** It is the only per-record
   append-only accepted stream (writer: `ScrapeCheckpoint.commit`,
   `allocation/scrape/checkpoint.py:137–147`; same writer for arxiv_bulk,
   `arxiv_bulk.py:655`). Resume never re-appends (papers in `_done` are never
   re-committed); worst crash artifact is one torn tail line (loader skips it —
   mirror that tolerance). `handoff/records.jsonl` is write-once at run end
   (mode `"w"`), statement-deduped and capped — NOT tailable, NOT the stream.
   The qa_calls-vs-candidates gap is explained and benign: null QA results +
   `classify_answer` rejects are dropped pre-candidate BY DESIGN (dead June run:
   1009 qa entries → 932 null + 4 classify-drops → 73 candidates). No paid,
   accepted record bypasses candidates.jsonl.
2. **uid does not exist at candidate time.** It is minted at cascade time by
   `inject_uid` (`processing/poser/base.py:142–158`):
   `sha256(source + "\x1f" + statement).hexdigest()[:32]` — and **preserved if
   already present** (`if not record.get("uid")`). candidates.jsonl rows are
   `{arxiv_id, candidate{...}}` with `statement` inside `candidate` and no
   `source`; `source` is stamped at mount from `--source`. Consequence: the
   slicer pre-injects the uid itself, with the same source string it passes to
   mount — ledger uid and funnel uid are identical BY CONSTRUCTION.
3. **Mount is a first-class CLI** — `icepick allocation mount --path <file>
   --source <s> --provenance extracted --truth-policy extracted --family
   realmath --output-dir <intake root>` → `<root>/runs/<UTC>Z/{manifest.json,
   handoff/records.jsonl}` (`cli.py:647–678,1191–1222`, `allocation/intake.py:50`,
   `adapters/manual_mount.py:83–140`). It stamps via `setdefault`, filters
   nothing (only requires `statement`), warns on duplicate uids but WRITES THEM
   ANYWAY, and does no statement-dedup. Nothing downstream dedups either —
   batch8's dup traversed cascade+pass@k and was caught only by Nicky's manual
   fold check (which is not in the codebase). The ledger below is therefore the
   funnel's first and only automatic dup gate.
4. **Historical universe: 2,583 distinct uids** across canonical pre-cascade
   inputs (batch0–8 + fk33rescue + stage1rescue; file list in the backfill
   config). Corpus survivors (817) are a strict subset — insufficient alone.
   Corrections to received history: batch8's dup (`5fde0ead…`, rows 112/117)
   was **not byte-identical** — same source+statement, different `answer`
   (`\tfrac` vs `\frac`) and metadata; batch3 carried a 3× same-uid dup
   (`08629655…`). Both are the same-uid-different-content class → HARD ABORT
   under this design. Orphan: `intake/runs/20260707T045533Z` (295 rows, 294
   uids) was allocated but never mounted — tracked as a warn-set, not blocked.
5. **Cascade/pass@k surfaces verified** (flags exist as specced; argparse
   `_handler` pattern in `cli.py`). `--stages codex:anthropic` = single gating
   Sonnet stage; **no OpenAI key is reachable** on anthropic-only stages
   (`config.py:188–207`). Cost is machine-readable at
   `cascade/cascade_manifest.json → overall.total_estimated_cost_usd` (float
   when both `--cost-per-*-mtok` are passed). pass@k checkpoints per record
   under `_progress/` (never re-bills); cascade restartability is
   **stage-granular** — a mid-stage kill re-runs the stage (bounded re-bill
   ≈ one batch's stage ≈ $2.30 here). `--mode flow_testing` replays a
   calibration sheet with zero API calls — the $0 dry-run vehicle.

## Identity & dedup — two layers

Per record, computed at slice time from the journal row:

- `uid = sha256(CAMPAIGN_SOURCE + "\x1f" + statement)[:32]` — pre-injected into
  the record; the funnel preserves it. **CAMPAIGN_SOURCE is one constant for
  the whole bulk campaign** (config `campaign_source`, initial value
  `arxiv_bulk_pde625`). Rationale: uid is source-dependent; per-batch source
  names would give re-extracted statements fresh uids and blind both this
  ledger and the manual fold guard. Batch number lives in the slice manifest
  and mount run metadata, never in `source`.
- `stmt_key = sha256(normalized statement)` where normalization = the funnel's
  own recipe (case-fold + whitespace-collapse, mirroring
  `realmath_scrape.normalise()`), source-INdependent — catches the same
  theorem arriving under a different source (bulk re-covering batch1–8
  territory) and whitespace/case variants that raw-statement uids miss.
- `content_hash = sha256(json.dumps(journal_row, sort_keys=True))` — replay
  detector.

Collision matrix at slice time (checked vs ledger AND within the slice):

| hit | action |
|---|---|
| content_hash identical (byte replay) | collapse: skip row, log, refill toward 250 |
| uid known, content differs | **HARD ABORT queue** (batch8's actual failure class) — freeze, STATUS.md, Nicky adjudicates |
| stmt_key known (uid differs — cross-source/history or whitespace variant) | policy `cross_source_statement_policy`, default **skip+refill**, every skip appended to `ledger/cross_source_skips.jsonl` + STATUS count (record already funneled once; re-processing double-pays and creates a fold-invisible corpus dup). Nicky may flip to `abort` or `allow`. |
| warn-set hit (045533Z orphan) | batch normally, WARN in STATUS.md |

## State root `out/auto_batcher/` (new; brief-sanctioned mutable-state exception to out/** append-only for STATUS.md, cursor.json, queue_state.json — everything else append-or-create-only)

- `ledger/consumed_uids.jsonl` — append-only, **fsync'd** (deliberate upgrade
  over the repo's flush-only idiom; this file is the dedup source of truth).
  Row: `{uid, stmt_key, content_hash, batch, source_journal, journal_line,
  sliced_at}`. Backfill rows carry `batch: "hist:<label>"`.
- `ledger/cursor.json` — per-journal `{path, line_count, byte_offset}`, written
  tmp+`os.replace` (atomic), advanced only AFTER slice commit. **Invariant:
  ledger membership is truth; cursor is an optimization.** Recovery re-reads
  from last cursor and skips ledger-known rows idempotently.
- `batches/batch<N>/` — `slice_manifest.json` (the 250 `(uid, stmt_key,
  content_hash, journal line)` entries + campaign source + journal span;
  written atomically BEFORE any mount), `slice_records.jsonl` (250 unwrapped,
  uid-injected records), `intake/` (mount output: `runs/<ts>/handoff/records.jsonl`),
  `cascade/`, `pass_at_k/`, `state.json` (stage machine, atomic writes),
  `READY_TO_FOLD` (flag file, written last).
- `queue_state.json` — next_batch_number (starts **10**; cross-checked against
  `batches/` listing on every start, mismatch = refuse to run), halt flags.
- `daemon.lock` — flock + PID + start-time (stale-detected via pgrep).
- `ARMED` — flag file; **absent by default**. Daemon exits immediately when
  missing (checked at start and every loop tick — removing it is graceful
  disarm at the next stage boundary; in-flight subprocess finishes, checkpoints
  make that safe).
- `STATUS.md` — rewritten (tmp+rename) on every transition: per-batch stage,
  counts, spend, holds, remainder size, skip/warn tallies, frozen batches with
  reasons. Disk is the reporting channel (task notifications here are
  unreliable).

## Daemon loop (single process, `icepick batcher …`; Python, not a bash gate —
state machine + ledger need real code; precedent gates stay untouched)

1. Preconditions each tick: ARMED present, lock held, cursor/ledger loaded.
2. Tail configured journal(s); ingest batch9's stream + its eventual handoff
   into the LEDGER only (uid from ITS manifest `source_name`) — batch9 is never
   auto-batched.
3. ≥250 unconsumed → cut slice per collision matrix → exactly 250 or abort →
   write slice_manifest + slice_records → fsync ledger appends → advance cursor.
4. Mount slice: `allocation mount --path slice_records.jsonl --source
   <campaign_source> --provenance extracted --truth-policy extracted --family
   realmath --output-dir <batch>/intake`; **verify handoff row count == 250 and
   uid set identical to manifest** — any drift = freeze batch.
5. Cascade (Sonnet-only, exact brief command; `ANTHROPIC_KEY_FILE` path proxy
   only). Parse `total_estimated_cost_usd`; `> $5.00` → freeze batch + HALT
   queue (record-bloat guard). Transient failures (529-class): bounded retries,
   exponential backoff; exhaustion → freeze batch, nothing advances past it.
6. pass@k ONLY when the Qwen slot is free — `pgrep -f "icepick processing
   pass_at_k"` AND no established conn on TCP:1234 (`lsof -sTCP:ESTABLISHED`)
   — re-check just before exec; serialized one batch at a time; exact brief
   wire params (all flags explicit; defaults differ from ours). `interrupted:
   true` in the manifest / rc=1 → resume by re-running same command.
7. Batch complete: verify `pass_at_k_manifest.json` `interrupted:false` →
   write `READY_TO_FOLD` → STATUS.md → loop.
8. Extraction concluded (run's INCOMPLETE gone / journal quiet + process gone)
   with remainder < 250 → remainder HELD: listed in STATUS.md with uids,
   never auto-batched.

Every stage transition is checkpointed in `state.json`; restart resumes the
in-flight batch at its stage (mount/cascade/pass@k are all safe to re-invoke:
mount is re-created only if handoff verification never passed; cascade/pass@k
follow the repo restartability contract). Surplus files are watched and
counted in STATUS.md (report-only; never auto-batched).

## Spend

Steady state ≈ $2.30/batch (ab_stage1 actuals, $3/$15 metering), under the $5
line → full-auto compliant once ARMED (arming = Nicky's pre-approval of that
recurring spend). Build/test/dry-run phases: $0 (synthetic journals; stubbed
stage runners in tests; `flow_testing` calibration replay for the live-command
dry-run). No OpenAI key anywhere.

## Acceptance tests (brief §tests → design mapping)

1. Exactness 1003→4×250+3 HELD → slicer on synthetic journal.
2. Crash-resume at every stage boundary → kill/restart harness over state.json
   fixtures + ledger/cursor recovery invariant.
3. Dup injection: byte replay → collapse+refill; same-uid-mutated → hard abort.
4. History collision: batch1–8 uid seeded → slice refuses (ledger backfill hit).
5. Qwen contention: fake `icepick processing pass_at_k` process → stage waits.
6. Cost guard: synthetic cascade_manifest > $5 → queue halts.
7. $0 dry-run: full pipeline, stubbed/flow_testing stages, transcript → STATUS.md.

## Open items for Nicky (also in final report)

- `campaign_source` value (`arxiv_bulk_pde625` assumed) — uid identity for the
  whole campaign, hard to change later.
- `cross_source_statement_policy` default `skip` (log-everything) — confirm.
- Brief premise correction: batch8's dup was same-uid-different-content, not
  byte-identical → such rows now hard-abort instead of collapsing.
- 045533Z orphan handoff (295 rows): mount manually someday? Stays warn-set.
- Cascade mid-stage kill re-bills ≤ ~$2.30 (stage-granular resume) — accepted?
- June run + batch9 both live again (resumed by parallel sessions); batcher
  ships pointed at the June bulk journal but DISARMED.

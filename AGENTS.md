# Icepick — canonical agent brief

**This file is the single source of truth for any agent working in this repo —
Claude Code, OpenAI Codex, another harness, or a human operator.** Claude Code
loads it through `CLAUDE.md` (a thin wrapper); Codex loads it directly. Edit
THIS file; never fork brief content into a wrapper. Nested briefs bind in
their subtree (currently `src/posers/AGENTS.md`) — read them before working
there.

Icepick is a portable processing surface for ModelBreaker-style problem
records. Pipeline: **acquire (arXiv scrape / bulk / mount) → wellposed
cascade → pass@k → labeled corpus**.

Read `docs/pipeline_controller.md` for the full stage-by-stage reference,
manifest schemas, and controller access patterns. `docs/scraper_runbook.md`
is the scrape operator runbook. `docs/SESSION_HANDOFF.md` is the live
cross-session ledger — read it right after this file, and update it at
session end.

## Git & shared-checkout discipline

This IS a git repo: `origin = https://github.com/RedHairing217/icepick.git`,
trunk `main`.

- Parallel agent sessions share this checkout. Run `git status` before
  committing; commit ONLY the paths you changed, surgically. Check `ps`
  before launching anything that shares checkpoints or the local Qwen slot.
- **Never push without Nicky's explicit go-ahead.** Local commits are fine.
- The test suite is the tripwire — run it before and after your change.

## Quick facts

- Run tests (full, three suites):
  `python3 -m pytest tests/ src/posers/Claude_Poser/tests src/posers/Codex_Poser/tests --ignore=tests/integration`
  → baseline **1118 passed**. Repo-root `python3 -m pytest` → **975 passed,
  3 skipped** (live tests skip without creds). Sub-repo suites (never collected
  by root): `src/loratrain` **154 passed**; `evalharness` **40 passed + 2 KNOWN
  failures** (they pin the frozen `eval_paper_split.json` retired by the
  2026-07-26 split ruling — repoint is the evalharness lane's open item; do not
  "fix" by restoring the file). Measured 2026-07-29. Below baseline =
  regression; if you add tests, update these numbers in the same commit.
- Scrape-path tests: `pytest tests/allocation/scrape/` after every change
  there. `tests/allocation/scrape/test_pacing.py` asserts EXACT backoff
  schedules — extend, never delete.
- Keys live outside the repo:
  `ANTHROPIC_KEY_FILE=/Users/redhairing/Desktop/helloworld/anthro_key.env`
  (+ `openai_key.env` alongside). These are path proxies — pass the path
  around; **never read, print, or embed key-file contents** in code,
  commands, or logs.
- The billed model per provider comes from the key files' `ANTHROPIC_MODEL` /
  `OPENAI_MODEL` lines. Never steer a single provider via the cascade's
  `--*-judge-model` flags — they are per-build across providers (details in
  `src/posers/AGENTS.md`).
- **OpenAI judge model is `gpt-5.5` at `OPENAI_REASONING_EFFORT=high` as of
  2026-07-06** (was `gpt-4.1-mini`). Rates: $5/M input, $30/M output;
  reasoning tokens bill as output. Measured judge cost ≈ **$0.02/sample**
  (~50× gpt-4.1-mini), so a 250-paper batch's 1,000–2,000 OpenAI judge
  samples run **~$20–45 — over the $5 HITL line; get approval before any
  batch-scale judge run**. The swap rolled every OpenAI judge cache key
  (keys include the model id), so the next run re-bills all judge samples.
  All judge-quality baselines (stage-1 86% kill-confirmation, stage-3 82.5%
  false-kill audit) were measured on gpt-4.1-mini and do not transfer.
- Qwen pass@k endpoint: LM Studio at `http://127.0.0.1:1234/v1/chat/completions`,
  model `qwen/qwen3-8b`, `--backend-url` mandatory.

## Invariants — do not break

1. **Kill switches stay closed.** Groundtruth's `_build_anthropic_client` uses a placeholder key (`"[API key]"`) — leave it (restore snippet in its docstring). Pass@k paid backends require `--allow-live-calls` AND `--i-understand-paid-backend-is-off-policy` AND an explicit `--model`; `DEFAULT_MODELS` for anthropic/openai are `None` on purpose.
2. **Pass@k subject model is Qwen via `qwen_http`.** Wire parameters are byte-identical to ModelBreaker's `harvest_realmath.py:call_qwen` (temp 0.7, max_tokens 2048, `/no_think` suffix, same system prompt) — do not drift them; comparability with MB's 70-record harvest is a feature.
3. **Restartability contract**: re-running the same command resumes from disk; no `--resume` flag; completed items never re-billed. New paid calls must be checkpoint-cached or documented as repeat-on-resume with negligible cost.
4. **Band constants** in `contracts/records.py`: `BAND_LO=0.125, BAND_HI=0.75` (differs from MB's 0.875 — intentional, documented).
5. **QA is single-stage Sonnet.** `qa_extractor` calls the Sonnet Q+A generator once per theorem; Sonnet IS the filter (returns `None` for theorems with no single fixed answer). A theorem the generator can't handle is skipped, but `QAConfigError` (missing key/SDK) must surface — misconfiguration never hides behind a silent skip.
6. **Sequential arXiv access.** `_pace_lock` serialises requests. Never parallelise arXiv fetches — multiple workers on one IP trip the limiter instantly. Pacing is env-tunable via `ICEPICK_ARXIV_MIN_INTERVAL`; never lower it to speed a run.
7. **Operator flow**: `allocation plan → approve --call-budget N → run`. Production scraping refuses to run without an approved manifest whose budget covers the estimate. Don't weaken these gates.
8. **Never reject good theorems.** Breadth/target caps (`max_per_paper`, `target_count`) shape the handoff but must not discard accepted rows: the overflow is preserved via `ScrapeResult.surplus` into `handoff/surplus_records.jsonl` (canonical, mount-ready), counted in the report and CLI summary. Nicky's standing rule (2026-07-04) — don't reintroduce silent drops of extracted/paid-for candidates.

Also binding, same weight:

9. **Max ONE concurrent Qwen call.** The local LM Studio slot is shared
   machine-wide, parallel sessions included — never run concurrent calls
   against it, whether from pass@k or a judge pointed at the local server.
10. **`out/**` is append-only run history.** Never edit or delete existing
    files under `out/`; new files/dirs only. Known read-only trees:
    `out/intake/`, `out/processing_*/`,
    `out/wellposed_pde625_claude_anthropic/verdicts/`.
11. **Launches are hold-gated.** Never start scrapes, funnels, batch
    processing, or any paid run without Nicky's explicit release in the
    current session — even if a plan or approved manifest already exists.
    Current hold state lives in `docs/SESSION_HANDOFF.md`.
12. **Spend needs a human above $5.** Unapproved API spend over $5 requires
    Nicky's sign-off (Nicky's rule, 2026-07-06); at or below $5, or within a
    pre-approved budget, proceed autonomously and account for every call in
    manifests/reports.

## Environment quirks (any agent)

- **Verify task/tool completion notifications against disk and `ps` before
  acting on them** — this environment has delivered fabricated and premature
  completion events.
- Assume you are not alone: parallel sessions in this checkout are common.

## Current optimization mission

Two goals, priority order. **Audit current code state before starting any
target — some may already be partially or fully implemented by a parallel
session** (the repo moves fast; e.g. rate-limit telemetry and adaptive page
size have landed since the brief was drafted).

### Goal 1 — never trip arXiv's limiter; lose nothing when it fires

Implemented: 4s hard spacing (`_pace`), exponential backoff 3s→6s→12s
doubling, `Retry-After` honored, single worker, 50-item pages, checkpoint-
and-stop on final retry (resume = same command after 15-30 min cooldown).

Targets:
- **T1.1 Persistent cooldown marker**: stamp `_progress/rate_limited_at` on 429-death; on resume within cooldown window (default ~20 min, env-tunable), refuse with "cooling down, retry after HH:MM". Delete marker on first successful request.
- **T1.2 Adaptive page size**: after a 429 recovery in-run, halve effective `max_results` (floor 25). Check `_PAGE_SIZE_FLOOR_AFTER_429` — may already exist.
- **T1.3 Telemetry**: 429/503 counts + total backoff seconds in `ScrapeResult` → acquisition dict → `source_report.md`. Check `rate_limit_events` — may already exist.
- **T1.4 e-print parity**: confirm `default_latex_source_fetcher` gets identical treatment (same `_http_get` path + cooldown marker).

### Goal 2 — QA extraction cost down, yield preserved

Implemented: single-stage Sonnet Q+A (~$0.005/call), one call per mined
theorem; Sonnet IS the filter (returns `None` for theorems with no single
fixed answer); `classify_answer` `latex` tier keeps sympy-unparseable
research math; QA generator disk-cached by statement hash; `call_budget`
enforced before every paid call.

**Dropped — the Haiku pre-filter gate.** A live math.NT `qa` scrape showed
the gate accepted 371/371 theorems (zero selectivity) while Sonnet rejected
267/371 via its own `None`. The gate was pure cost (~$0.33/run) with no yield
effect, so it was removed. Don't reintroduce a per-call pre-filter without a
measured precision signal — the same rubber-stamp failure mode bit claude-poser.

Findings that close old targets:
- **Prompt caching is inert here, not a lever.** Anthropic's minimum cacheable
  prefix is 2048 tok (Sonnet 4.6) / 4096 tok (Haiku 4.5); the QA system prompt
  is ~480 tok, so `cache_control: ephemeral` reads 0 (`cache_read_input_tokens: 0`).
  Padding to the threshold is net-negative (arithmetic checked). The block is
  still sent (forward-compatible — a future larger prompt activates it free),
  but it saves nothing today. Any "caching savings" in a cost model is phantom.
- **Judge-side caching is a measured no too (2026-07-05).** Across ~2,900
  billed judge samples (620-rec wellposed claude:anthropic + both 2026-07-04
  cascade runs) the largest full judge request is 912 tok, median ~550 —
  under half the Sonnet 2048 minimum and below OpenAI's 1024-tok
  automatic-caching floor, so caching is unreachable at every judge call
  site, even within-record where the 3 sequential samples share the whole
  prompt. Static prefixes (exact, free count_tokens): claude-poser
  JUDGE_SYSTEM 478 tok, codex-poser rubric header 91 tok. Padding only beats
  unpadded above ~990 tok/request; zero observed samples qualify. Achievable
  saving: $0.00/batch of the theoretical $2.9 input-side ceiling. Both prompt
  families are already static-first, so the OpenAI side has nothing to
  reorder. Revisit only if statements grow ~4x, judge models change, or
  Anthropic lowers the minimum. *Trigger re-checked 2026-07-06 when the
  OpenAI judge moved to gpt-5.5: the swap changes no input-side tokens (max
  request still 912 tok < the 1024 floor, which gpt-5.5 keeps), so the
  conclusion stands — though at $5/M input the theoretical ceiling is now
  ~12× larger, the reachable saving is still $0.*

Open targets:
- **T2.3 Empirical planning ratios**: recalibrate `PAPERS_PER_RECORD=4`,
  `CANDIDATES_PER_PAPER=13` from real run manifests (live runs: 2-37 theorems/paper).
  Budgets derive from these; qa now budgets one Sonnet call per theorem.
- **T2.4 QA generator batch mode** (the real amortization lever, investigate
  first): ~10 numbered theorems per Sonnet call with a JSON result array,
  amortizing fixed per-call overhead across a batch. Prototype behind an opt-in
  flag; measure per-item agreement vs single-call on ~50 cached theorems before
  defaulting. Skip a theorem the batch can't handle, don't fail the batch.
- **T2.5 Live re-validation**: the single-stage qa path has never completed a
  full live scrape (arXiv 429'd the pilots). First clean run: compare realized
  qa calls + yield vs `_estimated_calls`, feed back into T2.3.

### Measurement discipline

- Baselines to beat: QA step $0.22/193 theorems; pipeline ~$0.32/25 records. Report before/after in the same units.
- Every paid call visible in manifests — no silent spend. New call type = new counter (`qa_calls` pattern) threaded through `ScrapeResult` → acquisition dict → `source_report.md` → `_estimated_calls`.
- Rates: Haiku $1/$5 per MTok, Sonnet $3/$15 per MTok, Qwen local $0.

### Key file map

| concern | file |
|---|---|
| arXiv HTTP/pacing/backoff, single-stage QA | `src/icepick/allocation/scrape/realmath.py` |
| scrape checkpoint/resume | `src/icepick/allocation/scrape/checkpoint.py` |
| budget/estimates/acquisition accounting | `src/icepick/allocation/adapters/realmath_scrape.py` |
| pass@k policy + kill switches | `src/icepick/processing/pass_at_k/config.py` |
| wellposed cascade | `src/icepick/processing/poser/cascade.py` |
| rate-limit tests (exact schedules) | `tests/allocation/scrape/test_pacing.py` |

Suggested order: T1.1 → T2.3 → T1.2/T1.3 (verify done) → T2.4 investigation → T2.5/T1.4 validation.

## Doc index

| doc | what |
|---|---|
| `docs/SESSION_HANDOFF.md` | live cross-session ledger: batch/corpus state, holds, open decisions — read second, update at session end |
| `docs/pipeline_controller.md` | full stage-by-stage reference, manifests, chaining, kill switches |
| `docs/scraper_runbook.md` | arXiv scrape operator runbook (`plan → approve → run`) |
| `docs/operator.md` | processing pipeline runbook (five-minute first run, troubleshooting) |
| `docs/plan.md` | scope and stage rationale |
| `docs/realmath_scraper_adapter.md` | scrape adapter design |
| `docs/readability.md` | doc style notes |
| `src/icepick/allocation/bulk/INTERFACES.md` | arxiv_bulk adapter contract + fleet-build invariants |
| `src/posers/AGENTS.md` | poser-local rules: judge-cache key semantics, model config |
| `src/posers/*/README.md` | per-poser I/O contracts and CLIs |
| `docs/eval_harness_design.md` | LoRA eval-harness method authority (engine parity, split, baseline protocol) |
| `docs/lora_consistency_verdict.md` | LoRA campaign FINAL verdict (n=12, closed 2026-07-29) — read with `lora_campaign_results.md` + `lora_decisions_2026-07-28.md` |
| `src/loratrain/README.md` + `RUNBOOK.md` | training-arm design (D1–D4) + remote-box recipe (reusable for dataset v2) |

## Codex-specific addendum

- Codex sandboxes commonly default to **no network and approval-gated
  commands**: live scrapes, judge calls, and installs will fail closed there.
  That is the correct failure — do not work around the sandbox; ask the
  operator to run the paid/network step or widen approvals.
- Record session notes in `docs/SESSION_HANDOFF.md` — the SAME ledger Claude
  sessions use. One shared ledger, not two.
- If your harness did not auto-load a nested `AGENTS.md` (e.g.
  `src/posers/AGENTS.md`), read it manually before editing in that subtree.

Claude Code agents: your addendum lives in `CLAUDE.md`.

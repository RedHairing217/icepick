# Trustworthy Funnel Execution — window skeleton (arm v3 + two-stage gate + hardening)

Prepared 2026-07-11 by window-3 (Fable-5, session 89fe6f6f) on Nicky's decisions: "let's arm v3"
+ "make the new funnel more trustworthy". Mission slug: **trustworthy-funnel-execution**.
Supersedes nothing; builds on commit `7a3546c` (rubrics v2/v3 + lint + --judge-max-tokens, all
default-off) and the measured record in
`out/funnel_adjustment_analysis/execution_validation_20260711T060641Z/`
({VALIDATION_REPORT.md, V3_VALIDATION_ADDENDUM.md, acceptance_report*.json}). Evidence base:
v2 = 4/7 sentinel kills / 37.5% recall; v3 = 0/7 kills / 17.5% recall / 3/45 controls;
catch-sets disjoint by defect type (v3 ⊂ v2: omission-defects only). The design below is the
measured conclusion: v3 as cheap stage-A, agent-scale review as stage-B, plus four mechanical
hardenings that are independent of any rubric choice.

## Fresh Window Prompt (paste from here down)

You are implementing the TRUSTWORTHY FUNNEL window for icepick
(`/Users/redhairing/Desktop/helloworld/icepick`; cwd resets between Bash calls — absolute paths).
Parallel sessions are common; task events have been fabricated/premature: verify disk + `ps`
before acting on any notification. Fable orchestrates; Sonnet subagents write bulk code; Fable
reviews every diff.

READ FIRST: (1) `AGENTS.md` (invariants bind; esp. 8 never-discard-paid-records, 10 out/**
append-only, 11 launches hold-gated, 12 $5 line — WORST-CASE ceilings, not estimates, decide
what fits under it; never read/print `*key.env`); (2) `src/posers/AGENTS.md` (prompt text =
billed cache-key interface; env-file model selection, NEVER `--*-judge-model` cascade flags);
(3) `out/funnel_adjustment_analysis/execution_validation_20260711T060641Z/`
{VALIDATION_REPORT.md, V3_VALIDATION_ADDENDUM.md} — the measured basis for every number below;
(4) `out/qa_repair_20260711T055242Z/{RESULT.md, fold/FOLD_MANIFEST.md}` — findings F1–F3
(provenance bypass, cascade error-row leak, parse-failure case) and the official fold record;
(5) code: `src/posers/Codex_Poser/src/codex_poser/well_posedness/{scoring.py,cli.py,
judge_providers.py}` at `7a3546c`, `src/icepick/processing/poser/{cascade.py,codex_adapter.py,
config.py,runner.py}`, `src/icepick/batcher/` (DISARMED — its cascade invocation constants only).

PREFLIGHT: `git log --oneline -1` must include `7a3546c` in history (else this skeleton predates
your tree — STOP). Record `git status --short --branch`; measure the three-suite baseline from
DISK (last known 1066/0; the tree carries other windows' uncommitted tests — disk truth is the
bar). `band_corpus.jsonl` EXPECTED state after the 2026-07-11 qa_repair fold: **279 rows, sha256
`810d16081df83cc6c8ede6fcf6f6cd17447e2817f3ccecf65abb82ddfdebfb0f`** (the old `01609862…`/309 pin
is RETIRED). A further mismatch = another fold landed — do NOT stop; the validation population is
pinned BY UID in `execution_validation_20260711T060641Z/validation_population.jsonl`. The 41 E1
positives are NO LONGER corpus rows (folded out): take their statements/answers from the
sha-pinned audit snapshot `out/audits/wellposed_band_miss_audit_20260710T010302Z/
enriched_band.jsonl` (or that window's `judge_input_v2.jsonl`, byte-identical for these fields);
sentinels + controls remain corpus rows. Record per-row source + corpus sha in your manifest.
Check any live batch processes before touching shared state.
Claim the mission slug in `docs/SESSION_HANDOFF.md` at start (table row: slug / session / UTC /
scope / budget); heartbeat at phase boundaries; CLOSED at end.

### Scope — armed by this skeleton (Nicky's in-message decisions, 2026-07-11)

CORE, all armed: **H1–H5 hardenings, A1 arm-v3 plumbing, B2 stage-B escalation, V validation.**
OPTIONAL (checkboxes at bottom): **S7** modal-wrong recheck; **SHADOW** live shadow-batch.

**H1 — parse-failure pass bias fix (covers findings ef97f733 + F2).** Two measured leaks, same
family: (a) a record whose parsed sample count is below `uphold` can PASS (live instance
`ef97f733`: 1/3 parsed, the one vote said ill, record passed); (b) qa_repair's P5 observed a
judge-ERROR row counted among cascade survivors in `final_corpus` (F2 in its RESULT.md) —
REPRODUCE this first (the cascade's `passing_uids` logic looks error-excluding on paper; find
the actual leak path, likely the standalone-output → mount route) and close it with a test.
Add `--judge-parse-policy {legacy,strict}` (default `legacy` — replay comparability): under
`strict`, `samples_parsed < uphold` → ERROR. Armed runs (A1) use `strict`. Tests: 1/3-parsed-ill
→ error under strict, pass under legacy (pin the legacy behavior explicitly so the bias is a
recorded choice); error-row-never-survives test at the cascade AND mount boundaries.

**H2 — B1 attribution fix.** `codex_adapter.py` reads `parameters.judge_model`; the poser writes
`parameters.judge.model` → `poser_model=''` in every normalised row. Fix the read (accept both,
prefer the nested path) + regression test asserting a non-empty model string round-trips.

**H3 — judge cache hygiene.** In `judge_providers.py`: do NOT cache a reply that
`parse_judge_reply` rejects (the v2 incident cached 136 truncated replies; a poisoned cache
survives re-runs by design). Import-cycle note: parse lives in scoring.py — either move the
parser to a leaf module or inject a validity callback; match house style. Also floor the
anthropic judge timeout the way OpenAI reasoning models are floored IF measured p95 latency at
1500 max-tokens demands it (measure first; do not add speculative floors).

**H4 — insufficient_context surfaced to the queue (S0, measured).** The per-record IC tally is
already in signals; thread an `ic_majority` marker into the cascade stage outputs so stage-B
triggers can consume it (zero prompt edits, zero re-bills).

**H5 — provenance-bypass guard (finding F1).** Records with NO `provenance` field normalise to
"computed" when a `family` is present (`contracts.py::_normalise_provenance`) and pass
"well-posed by construction" — the judge never runs. In production mode: is_computed's
judge-skip requires an EXPLICIT `provenance: "computed"`; missing/blank provenance with a family
→ treat as extracted (judge runs) + increment a loud `provenance_defaulted` counter surfaced in
counts and the manifest echo. flow_testing behavior unchanged. Tests: the F1 reproduction row
(no provenance + family) judges under production-strict, passes-by-construction only with the
explicit field.

**A1 — arm v3 in the production path.** Thread first-class settings through icepick (NOT via
`--*-judge-model`-style traps, NOT via untyped extra_args): `rubric_version` (default v1) and
`judge_max_tokens` (default 512) on the codex poser settings → `codex_adapter.plan()` argv
(`--judge-rubric-version`, `--judge-max-tokens`) → echoed in `cascade_manifest.json` config echo
(provenance: every manifest must show which rubric gated the batch). Wire the icepick CLI flag
surface the same way the existing judge flags flow. The ARMED configuration for new batches:
**v3, max-tokens 1500, parse-policy strict** — set in the batch launch surface (new gate
scripts/batcher constants), NEVER by flipping codex-poser defaults (v1 stays the poser default;
existing gate_*.sh scripts are parallel-session artifacts — do not edit them). Update the
DISARMED batcher's cascade invocation constants in the same pass so a future re-arm inherits v3.

**B2 — stage-B escalation (the two-stage gate).** New module in
`src/icepick/processing/poser/` (name per house style, e.g. `escalation.py`):
- Trigger set, computed from stage-A (v3) outputs per record: FLAG verdict · PASS with
  `ic_majority` · PASS with min sample confidence < 0.7. (Measured on the frozen 98: ≈10–20% of
  records.)
- Escalated records go to `escalation_queue.jsonl` in the run dir (uid, statement, answer,
  stage-A verdict + votes + reasons, provenance stamp). **Stage-A FLAG no longer drops a record
  by itself** — invariant-8 alignment: only stage-B confirmation (or operator ruling) drops.
- Stage-B runner: one reviewer agent per escalated record (Fable-class, blind: statement +
  stored answer ONLY — no stage-A verdict, no source, no labels), instructed to attempt the
  actual derivation and return the structured verdict {well|ill, named_defect|forcing_chain,
  confidence}. Derivation depth decides — this is the thrice-replicated audit method. Harness:
  whatever the window has (Agent tool / workflow); wall-clock and token cost per record stated
  in the report. Verdict merge: stage-B overrides stage-A; disagreements logged with both
  rationales (C6-style: written rationale BOTH directions).
- Batch integration: cascade stage output gains `escalation/` artifacts; final_corpus excludes
  only stage-B-confirmed ills; everything else flows with signals attached.

### Files to touch

Poser side: `judge_providers.py`, `scoring.py` (H1/H3 only — rubric text untouchable),
`cli.py`, poser tests. Icepick side: `codex_adapter.py`, `config.py`, `cascade.py`,
`runner.py` (minimal threading), new `escalation.py` + tests, batcher constants. Nothing else
unless the local pattern clearly requires it; match idiom; no default flips outside the ARMED
launch surface.

### Validation (gates before any new batch runs armed)

Population: the SAME frozen 98 by uid (fold-resilient recipe above) + **holdout warning**: this
is the second rubric-era reuse of these rows — do NOT tune anything against them beyond
pass/fail; if any trigger threshold gets tuned, state it and mark the affected metric TUNED.
1. $0 replays: three-suite green; v1 legacy-policy replay byte-identical on cached fixtures;
   H1 strict-policy unit coverage; H2 model string present in normalised rows.
2. Stage-A live re-run only if plumbing changed the prompt (it must not — identical prompt ⇒
   `judge_cache_v3.jsonl` replays free; copy it into the new run dir, per-run-dir caches are
   cold otherwise). Sanity: verdicts match `acceptance_report_v3_trancheB.json` per uid.
3. Stage-B over the triggered band (expect ≈15–25 rows incl. the 2 v3-residual errors): blind
   agent derivations. ACCEPTANCE (two-stage combined): **sentinel kills 0/7 through BOTH stages
   (HARD)** · **combined E1 recall ≥ 55%** (denom 40; stage-B must convert escalated true-ills:
   v3 alone is 17.5%, the band contains the IC/low-conf mass) · **net control kills ≤ 2/45
   after stage-B** (stage-B should rescue stage-A's 3 control FPs — that failing is a design
   red flag) · every stage-B verdict carries a written rationale · no row dropped without
   stage-B confirmation.
3b. Repaired-row population (24 new-uid rows from the qa_repair fold, now corpus members;
   `repaired_from` links in `out/qa_repair_20260711T055242Z/repaired_records.jsonl`): run
   stage-A(+B if triggered) over them with `provenance: "extracted"` explicitly set (F1!).
   These are verified should-PASS rows — the D2 pilot's terms apply: expected pass ≥ 80%
   (they passed rubric v1 23/24); every flag reported with the row's `repair_diff` (each is a
   bad repair or an over-tight gate — Nicky adjudicates; a sub-80% pass rate is a prominent
   yellow flag, not an auto-fail). ≈ 72 samples ≈ $0.35 worst-case-bounded.
4. Cost/token ledger: worst-case ceilings BEFORE each paid step; API spend here ≈ $0 (cache
   replay) unless SHADOW armed; stage-B is token-spend — state actuals. Any API estimate whose
   WORST case crosses $5 total → Nicky sign-off first (the v2 window's lesson: estimates ran 3×).

### Stop conditions

Preflight failures above · any src edit outside the named files · rubric TEXT edits (v3 text is
Nicky-ratified; changes = new version, new window) · sentinel kill at any stage · combined
recall < 40% (architecture inadequate — reconvene, do not tighten) · stage-B rescues fewer than
2 of the 3 known control FPs (`5fce85f4`, `8ff31b4a`, `cc728308` — though note `8ff31b4a`/
`cc728308` may be genuinely ambiguous: a stage-B "ill" there with a written rationale goes to
Nicky as a possible real find, not an auto-fail) · unresolvable Qwen/batch contention · any
output row without a provenance stamp (model + session + UTC + mission slug).

### Deliverables

Code + tests (suite ≥ baseline, 0 fail); validation artifacts in a NEW
`out/funnel_adjustment_analysis/two_stage_validation_<UTC>/` dir (manifest, per-row verdicts,
acceptance table, cost actuals); updated cascade_manifest provenance demo (one flow_testing $0
run showing rubric_version echo); `docs/SESSION_HANDOFF.md` addendum + claim CLOSED; durable
memory update; report to Nicky: acceptance table, stage-B rationale samples, the ARMED-config
one-liner for the next real batch, and anything dropped.

Token/pause budget: [NICKY: set]. Arm checkboxes:
[x] H1–H4 hardenings (armed by this skeleton)
[x] A1 arm v3 (v3 + max-tokens 1500 + strict parse policy on the new-batch launch surface)
[x] B2 stage-B escalation + validation gates
[ ] S7 post-pass@k modal-wrong recheck (~$0.1/batch recurring — the only lever on clean-pass
    misses; measured motivation in V3_VALIDATION_ADDENDUM.md)
[ ] SHADOW: one live 250-batch shadow run, v1-vs-two-stage side-by-side (≈ $2–3 API worst-case,
    hold-gated inv 11)
[ ] fold stage-B verdicts into batch reports as a standing column

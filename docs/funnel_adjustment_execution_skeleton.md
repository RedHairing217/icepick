# Funnel Adjustment Execution — CONSOLIDATED window-3 skeleton

Prepared 2026-07-11 ~03:00Z by the skeleton-unification session after verifying window-2b's
analysis. **This is the ONE file to paste for window-3.** It consolidates: the racer's
implementation scaffold (`out/funnel_adjustment_analysis/SKELETON_funnel_adjustment_execution.md`),
window-2b's binding amendments (`VERIFICATION_AND_RESIDUE_ADDENDUM.md` §4–§5), and the final
verification record + v3.1 labels (`out/audits/skeleton_unification_20260710T214021Z/
{V3_VERIFICATION_FINAL.md, labels_v3_1.jsonl}`). Those documents remain the evidence trail; none
needs to be pasted alongside this file.

Everything here is evidence-backed: the rubric edit targets the measured central failure (judge
accepts *recoverability* — recall, genre, confabulated context — instead of demanding
*derivability*; 12/41 misses were seen-and-excused, 26/41 never seen, receipts on file), the
sentinel hard-gate encodes the measured symmetric risk (over-flagging field-standard machinery),
and the cost table reflects the verified per-run-cold cache behavior.

## Fresh Window Prompt (paste from here down)

You are implementing a validation-first adjustment to icepick's codex:anthropic well-posedness
gate. Repo `/Users/redhairing/Desktop/helloworld/icepick`; shell cwd resets between Bash calls —
absolute paths. Parallel sessions are common; task events have been fabricated/premature before:
verify disk + `ps` before acting on any notification.

READ FIRST: (1) `AGENTS.md` (invariants bind; esp. 8 flag-not-kill, 10 out/** append-only,
11 launches hold-gated, 12 $5 spend line); (2) `src/posers/AGENTS.md` (cache-key semantics,
model-config traps); (3) `out/audits/skeleton_unification_20260710T214021Z/V3_VERIFICATION_FINAL.md`
+ `labels_v3_1.jsonl` (your validation population); (4)
`out/funnel_adjustment_analysis/VERIFICATION_AND_RESIDUE_ADDENDUM.md` §2–§4 (structural facts,
measured backtests); (5) code: `src/posers/Codex_Poser/src/codex_poser/well_posedness/{scoring.py,
contracts.py,cli.py}`, `src/posers/Codex_Poser/tests/test_well_posedness.py`,
`src/icepick/processing/poser/cascade.py`, `src/icepick/processing/poser/codex_adapter.py`.

PREFLIGHT: `band_corpus.jsonl` vs 309 rows / sha256
`01609862e21fde140154650121524634ac6673434e4aaf6f33797a08ab6d8d1a`. A mismatch is an EXPECTED
state (Nicky's removal ruling and/or the rescue lane's fold may land while you work — per 2b's
wrapper D1): do NOT validate against the mutated corpus blindly; rebuild your frozen validation
population by uid from `labels_v3_1.jsonl`, taking statements/answers from the sha-pinned audit
snapshot `out/audits/wellposed_band_miss_audit_20260710T010302Z/enriched_band.jsonl` (verified
identical to the audited corpus) or the `.bak-pre-*` backups; record per-row which source you
used + the new corpus sha in your manifest. Rows folded out of the corpus remain valid labeled
specimens — the validation measures the GATE against the LABELED statements.
`git status --short --branch` recorded; pre-change three-suite baseline measured and recorded
(this checkout carries uncommitted parallel work — DISK TRUTH is the baseline, not AGENTS.md's
historical number; a failing pre-change baseline = stop and report).

### Scope — implement ONLY what is armed

CORE (armed by pasting this skeleton): **S1 + S2**, both opt-in, default-off, flag-not-kill.
OPTIONAL (each needs Nicky's explicit arm, checkboxes at bottom): **S7**, **S0**, **B1 bugfix**.
Ranking note: the measured VALUE ranking is **S2 > S7 > S1** (2b §4). S7 is optional-tier only
because it adds a recurring per-batch call site (2b §5: "ADD S7 if Nicky approves its
~$0.1/batch"), not because it ranks below S1 — if armed, expect S7 to contribute more recall
than S1.
Do NOT implement S3/S4/S5/S6 (S3 repositioned as a future extraction-time guard; S5 deferred on
measured precision/cost; S6 rejected — vote-tightening caught 9/38 positives at 3/8 sentinel hits,
and samples are near-duplicates at temperature 0.2).

**S1 — context/degeneracy lint (advisory-only).** Deterministic patterns: missing-context
placeholders ("a system", "an ODE", "a PDE", "the equation", "the stated assumptions", "as in
Lemma/Proposition", "as specified", "appropriate/suitable conditions"); source-local language
("using the notation", "defined above", "as before"); defines-then-asks circular shapes
(X := … then asks X); verbatim formula-recall shapes. Wire as `--context-lint-mode
{off,advisory}` default `off`; hits recorded in `signals`, never gating (measured: 5/46 positives
at 0.8% keep FP — and it hits sentinel `01464d48`, which is why gate mode does not exist).

**S2 — judge rubric v2 (`--judge-rubric-version {v1,v2}`, default v1).** Full v2 text:

```text
You audit a mathematics problem for well-posedness.

Use only the final Statement below. Do not rely on the source paper, the title, or memory of a
canonical theorem. The stored answer is a claim to verify against the statement.

A problem is well-posed only when a strong mathematician, given the statement alone, can derive
the stored answer uniquely up to equivalent form.

Attempt the derivation. The stored answer is a claim to verify: flag unless the statement as
written forces it uniquely (up to equivalent form). Knowing the intended theorem is not
evidence — a recalled or canonical answer that the statement does not force is exactly the
failure mode you are auditing for. Do not supply missing context from memory: if the derivation
needs an equation, hypothesis, normalization, convention, or definition the statement does not
contain, that is a flag, not a gap for you to fill.

Flag the problem when any of these hold:
- an equation, object, hypothesis, convention, normalization, or symbol needed for the answer is
  missing or paper-local;
- more than one non-equivalent answer could satisfy the question;
- the question asks for a sharp, optimal, exact, largest, smallest, or canonical value but the
  statement only gives an existence claim or a bound;
- the stored answer is merely defined or displayed in the statement and the item is
  transcription/definition recall rather than a mathematical task;
- the stored answer contradicts the literal statement, even if it resembles a known theorem.

These are NOT flags: difficulty; advanced but field-standard terminology or notation (an object
a strong mathematician can define and use without the paper — e.g. a classical named operator —
is standard even if this statement does not define it); an answer given in one of several
equivalent forms.

Statement:
{statement}

Stored answer:
{truth}

Pass@k context:
{passk_context}

Soft context signals from code:
{soft_context}

Reply with a JSON object only, keys: determined (true or false), insufficient_context (true or
false), reason (one sentence), confidence (0 to 1).
```

Keep `insufficient_context` in the schema AND surface its per-record tally in `signals` /
run-report output (it was measured 3/3-true on passed misses; production currently ignores it).

**S7 (OPTIONAL) — post-pass@k band-sanity recheck.** For band-labeled records only (~15–25 per
250-batch), one Sonnet call AFTER pass@k: given statement, key, and modal wrong answer (+share),
answer whether the modal wrong is (a) a defensible correct answer to the statement as posed /
value-equivalent to the key, or (b) a genuine solver error. Flag on (a) into the review queue.
Measured motivation: in a large fraction of confirmed misses the modal wrong IS the rival reading
(`d682389a` 87.5% H^s, `9df79bb0` "1", `63d77c1e`/`a7afda27` inequality form), and it detects
mislabeled-solved records (`5cab6922`). ≈$0.05–0.10 per batch; new call site, zero cache impact.

**S0 (OPTIONAL) — honor the insufficient_context channel** as a review-queue feed (logic-only,
downstream of cached replies; zero prompt edit). Measured: +4 positives at 7 keeps + 2 sentinels
queued — queue-feed only, never a gate.

**B1 (OPTIONAL, recommended) — attribution bugfix:** `codex_adapter.py` reads
`parameters.judge_model` but codex-poser writes `parameters.judge.model`, so `poser_model` is
empty in every normalised row. 1-line fix + regression test. Provenance loss is this project's
recurring incident class; cheap to close.

### Files to touch

`scoring.py`, `cli.py`, `test_well_posedness.py` (+ `codex_adapter.py` + its test iff B1 armed).
Only touch more if the local pattern clearly requires it. Match surrounding idiom.

### Validation protocol ($0 except where marked)

Frozen manifest under `out/funnel_adjustment_analysis/execution_validation_<UTC>/` (new dir;
out/** append-only). Population from `labels_v3_1.jsonl`:
- **Positives**: 41 E1 (40 in strict-nh denominators — `e5ed37d5` reported separately);
  5 E2 reported separately, never in recall denominators.
- **Excluded from recall**: 8 policy rows (circularity is a pending Nicky definition), 2
  CONTESTED rows (`11e30827`, `343249ba`), `a7b98a81` (needs_human).
- **Sentinel hard-gate set**: the ratified sentinels (7 + `570fcab3` if Nicky ratifies the
  rubric fork — record which list was used).
- **Controls**: 45 keeps sampled from the 244 v3.1 keeps (excluding B's 12 pending splits),
  stratified by lane/batch, deterministic rule + seed recorded. S1 additionally runs over the
  full keep population (it's $0).
- **Repaired-row pilot (per 2b wrapper D2, optional/free)**: if `out/qa_repair_*/
  HANDSHAKE_window3.md` exists when you reach validation, run the armed configuration over the
  rescue lane's `repaired_records.jsonl` as a THIRD population; report its pass rate separately
  with every flag + `repair_diff` attached (each flag = bad repair or over-tight rubric — Nicky
  adjudicates; not an acceptance failure, but <~80% pass is a yellow flag to state prominently).
  No handshake file → skip and say so; never block on the rescue lane.
Static (now): S1 advisory over everything; report hits by tier/class/lane/control.
Live S2 validation (ONLY on Nicky's in-session release; inv 11): codex:anthropic with rubric v2
over positives + sentinels + controls ≈ 90–110 records × 3 samples ≈ **$0.7–1.2** — under the $5
line but still hold-gated. Judge model comes from the key env files; NEVER `--*-judge-model`.
Cache note (verified): the judge cache is per-run-dir, so new runs start cold regardless — the
rubric edit re-bills only in-place re-judging of existing runs; state actual token/cost after.

### Acceptance criteria (numeric, up front)

- E1 recall ≥ **55%** for S2 (or S2+S7 if armed), strict-nh denominator 40.
- **Sentinel kills = 0** on the ratified sentinel list (HARD gate; advisory-queue listing is
  permitted, a FLAG verdict is not).
- Stratified-control false-flag ≤ **2/45**; S1 advisory ≤ 2 hits across all 244 keeps... (S1 is
  already measured at 2/244; a jump above 4 means the patterns drifted — stop).
- Three-suite baseline: no regression below the recorded pre-change number.
- No default flips: v1 rubric and lint-off remain production defaults until Nicky reviews the
  validation report.

### Stop conditions

Validation population unreconstructable from the pinned snapshot/backups (sha mismatch alone is
NOT a stop — see preflight recipe) · any step would mutate corpus or audit history · any write
into `out/qa_repair_*` or other lanes' dirs (per 2b wrapper D3: the rescue lane may run in
parallel — expect its Qwen use, you use none; bracketed pgrep patterns like `pass_at_[k]` in any
watcher — the literal pattern deadlocked a gate chain on 07-07) ·
live run needed but not released in-session · spend estimate > $5 without sign-off (inv 12) ·
test regression · any sentinel hard-kill · S2 validation catches < 40% of E1 (package inadequate
— reconvene; do NOT tighten blindly: the over-tightening record is 82.5% stage-3 FK, 25.4%
stage-1 FK, 3/8 sentinel hits for vote-tightening) · any output row without a provenance stamp
(model + session + UTC — the attribution bug and two unattributed-writer incidents make this
non-negotiable).

### Deliverables

Code/tests for the armed items; validation manifest + report under the new execution_validation
dir; concise summary (changed files, tests, recall/sentinel/control numbers, cost actuals,
remaining hold-gated decisions); one-paragraph `docs/SESSION_HANDOFF.md` addendum at session end
(AGENTS.md ledger norm — this supersedes the racer skeleton's blanket docs ban); memory update.

Token/pause budget: [NICKY: set here]. Arm checkboxes:
[ ] S7 (~$0.1/batch ongoing) [ ] S0 (queue feed) [ ] B1 attribution fix
[ ] live S2 validation spend (≈$0.7–1.2) [ ] `570fcab3` ratified as sentinel

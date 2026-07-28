# Process Discipline (repo governance) — paste-ready skeleton

Prepared 2026-07-11 by the skeleton-unification session (Fable-5, 513f5b60…) after the 07-09/10
audit/funnel campaign. **Naming note:** originally drafted as "process correction"; renamed the
same night after discovering window-2b's `out/funnel_adjustment_analysis/
SKELETON_process_correction_window3.md` — a DIFFERENT artifact (a thin window-3 execution
wrapper; its D1–D3 deltas are now folded into `docs/funnel_adjustment_execution_skeleton.md`).
This skeleton is the third, independent lane: repo-level process governance. It does not touch
the funnel/rescue work. Object-level work converged (v3.1 labels, one window-3 skeleton), but the
PROCESS burned ~2× the needed tokens and survived on luck: four passes raced one mission, three
artifact races, two declare-before-write checkpoints, one unattributed-writer incident, four label
versions in six hours, and one refuted sample-extrapolation. Every correction below is
incident-backed; the window's job is to turn them into durable mechanism (docs/governance only —
NO src/tests changes; the B1 attribution bugfix stays in window-3's scope).

Incident receipts (verify in P1, all on disk):
| # | incident | evidence |
|---|---|---|
| I1 | same mission run 3× in parallel (unification / racer / 2b), + a 4th verification pass | `out/funnel_adjustment_analysis/RESUME_STATE.md` §"How this directory came to be" |
| I2 | racer wrote conclusions blind to labels that landed minutes earlier | same file, pass 2 note "WITHOUT having seen (1)" |
| I3 | checkpoints declared DONE before artifacts existed (both unification and racer) | `VERIFICATION_AND_RESIDUE_ADDENDUM.md` §0 hygiene finding |
| I4 | unattributed writer in shared out/ dir; files excluded for cause | `out/corpus_audit/AUDIT_REPORT.md` §8 |
| I5 | production provenance loss (`poser_model=''` in every normalised row) | `V3_VERIFICATION_FINAL.md` §additions (B1) |
| I6 | label sprawl: v2 → racer → v3 → v3.1; superseded tables still readable in place | `funnel_adjustment_analysis/` vs `skeleton_unification_*/labels_v3_1.jsonl` |
| I7 | skeleton staleness: A-skeleton written blind to audit B; racer skeleton validates against dead labels | `docs/judge_comparison_funnel_skeleton.md`; racer skeleton + 2b §5 warning |
| I8 | fabricated/premature task events (two more instances) | 2b addendum §0; memory `verify-task-notifications` |
| I9 | "6/6 sample ⇒ presume all 7" extrapolation refuted (5 of 7 ran opposite) | `V3_VERIFICATION_FINAL.md` residue table |
| I10 | keep-side verdicts recorded with no rationale → exhibit asymmetry in every later adjudication | `audit_rulings.jsonl` bare keeps; unification panel protocol caveat |
| I11 | the process-correction mission itself collided: two same-named skeletons built in parallel (2b's window-3 wrapper, 22:44; this file, 22:45), reconciled only by luck-of-reading-memory | `out/funnel_adjustment_analysis/SKELETON_process_correction_window3.md` mtime vs this file's; SESSION_HANDOFF 07-11 addendum |

---

## Fresh Window Prompt (paste from here down)

You are implementing PROCESS corrections for the icepick repo
(`/Users/redhairing/Desktop/helloworld/icepick`; cwd resets between Bash calls — absolute paths).
Docs/governance only: you touch `AGENTS.md`, `docs/`, and `docs/SESSION_HANDOFF.md`. You do NOT
touch `src/`, `tests/`, corpus files, or anything under `out/` except reads. No commits/pushes
(uncommitted like the rest of the tree) unless Nicky releases. $0. Parallel sessions are common
and task events unreliable: verify disk + `ps` before acting on any notification.

READ FIRST: `AGENTS.md` (you will be editing it — read fully; note it is canonical for Claude AND
Codex agents, style must match); `docs/SESSION_HANDOFF.md` (last 3 addenda);
`out/funnel_adjustment_analysis/{RESUME_STATE.md,VERIFICATION_AND_RESIDUE_ADDENDUM.md}` (§0 + §7);
`out/audits/skeleton_unification_20260710T214021Z/V3_VERIFICATION_FINAL.md`;
`out/corpus_audit/AUDIT_REPORT.md` §8.

P0 PREFLIGHT: git status recorded; re-read `AGENTS.md` immediately before each edit (parallel
sessions may touch it — on unexpected diff, STOP and report). DOGFOOD RULE: your FIRST write is a
mission claim (C1 format below) in SESSION_HANDOFF; your artifacts all carry C2 stamps.

P1 EVIDENCE PASS: verify each incident receipt above on disk (cheap reads). Drop any correction
whose incident you cannot verify; note it. Do not embellish.

P2 IMPLEMENT THE CORRECTIONS:

**C1 — Mission registry (fixes I1/I2).** Add to `docs/SESSION_HANDOFF.md` a pinned section
`## ACTIVE MISSIONS` (above the addenda) with row format:
`| slug | session id | started (UTC) | scope (one line) | last heartbeat | status |`.
New AGENTS.md invariant: before starting any multi-hour mission, CHECK the table; if the slug is
claimed with a heartbeat < 6h old, do not duplicate — report the collision to Nicky and stop.
Claim at mission start; update heartbeat at each phase boundary; mark CLOSED at session end.
Paste-protocol corollary (document for Nicky in the report, phrased as protocol not blame): every
paste-ready skeleton carries a mission slug in its header; a second session pasted the same slug
must find the claim and abort — this makes accidental multi-pastes safe.

**C2 — Provenance stamps (fixes I4/I5).** New AGENTS.md invariant: every generated ruling/verdict/
label ROW and every report/skeleton file header must carry `{model, session_id, utc, mission
slug}`. Unstamped rows found in shared `out/` trees are excluded-by-default from headline numbers
(the §8 precedent, now a standing rule). Reference window-3's B1 fix for the production-side gap;
do not implement it here.

**C3 — Label authority pointer (fixes I6).** Create `docs/LABEL_AUTHORITY.md`: a short file whose
ONLY job is to name the authoritative label artifact — current path + sha256 + version + one-line
supersession chain (initialize: v3.1 = `out/audits/skeleton_unification_20260710T214021Z/
labels_v3_1.jsonl`, superseding v3/racer/v2 tables, which remain on disk per append-only but are
DEAD for citation). New AGENTS.md rule: any session producing a new label set must update this
pointer IN THE SAME PHASE, and any skeleton/validation consuming labels must preflight-check its
assumptions against this pointer and STOP on mismatch. Add to the AGENTS.md doc index.

**C4 — Write-then-declare (fixes I3).** New AGENTS.md line under checkpointing/handoff norms: a
checkpoint or ledger entry may claim DONE only for artifacts that exist on disk at write time
(`ls` them first); anything else is marked IN-FLIGHT. Phase boundaries: artifacts → checkpoint →
next phase, never checkpoint-first.

**C5 — Late-writer discipline (fixes I4/I8, extends the existing verify-notifications quirk).**
Extend the AGENTS.md environment-quirks bullet: before treating a killed/stalled workflow as
dead, confirm process termination or EXPECT late writes; stamp-check (C2) anything that appears
in your output dirs afterward; two documented premature-completion instances added 07-10.

**C6 — Adjudication standards (fixes I9/I10, encodes the campaign's method lessons).** New short
AGENTS.md subsection "Audit/adjudication standards", four rules: (1) WRITTEN rationale mandatory
for BOTH verdict directions — bare keeps are inadmissible as future exhibits; (2) blind-first
staging (rule on the record before reading prior positions), derivation attempted — "the deepest
actual derivation decides" is the thrice-replicated result; (3) presumptive tiers NEVER harden by
sample extrapolation — per-row verification or they stay presumptive (the 6/6→7 error ran 5-of-7
opposite); (4) rubric-SEMANTICS forks (e.g. does a question's presupposition count as
statement-supplied — the `570fcab3` class) escalate to Nicky; majority-voting across passes does
not settle definitions.

**C7 — Skeleton freshness contract (fixes I7).** New AGENTS.md rule + retrofit: every paste-ready
skeleton must carry in its header: mission slug (C1), supersedes-line, and preflight assertions
(corpus sha + LABEL_AUTHORITY check) with STOP-on-mismatch. Retrofit now: prepend a 2-line
`> SUPERSEDED by …` banner to `docs/judge_comparison_funnel_skeleton.md` and mark
`docs/judge_comparison_funnel_skeleton_v2_unified.md` header `MISSION COMPLETE 2026-07-11 —
historical; do not paste` (docs/ files may be edited; the superseded skeletons under
`out/corpus_audit/handoff/` and `out/funnel_adjustment_analysis/` must NOT be edited — list them
in LABEL_AUTHORITY's dead-pointers note instead).

**C8 — Budget ledger (fixes the silent ~4-5M evening).** C1 claim rows gain a `budget` column;
session-close addenda must state actual approximate spend (the existing active-handoff memory
practice, promoted to repo ledger norm).

P3 CONSISTENCY PASS: re-read your AGENTS.md diff whole — minimal, surgical, matches house style,
no duplication of existing rules (several half-exist as memory/quirks; CONSOLIDATE, don't
restate). Total AGENTS.md growth target ≤ ~45 lines.

DELIVERABLES: the AGENTS.md edits; `docs/LABEL_AUTHORITY.md`; ACTIVE MISSIONS section (with your
own claim CLOSED at end); the two docs/ skeleton banners; one SESSION_HANDOFF addendum;
memory update (new `process-corrections` file + MEMORY.md hook, linking the incident memories);
report to Nicky: per-correction one-liner + the paste-protocol note + anything dropped in P1.

STOP CONDITIONS: any write outside AGENTS.md/docs/ · AGENTS.md concurrent-edit collision ·
an incident receipt fails verification and the correction would misstate the record · scope creep
into src/tests/out (report instead) · [NICKY: token budget here].

# Judge Comparison + Funnel Analysis — UNIFIED skeleton (v2)

Prepared 2026-07-10 ~22:30Z by the skeleton-unification session (Fable-5), on Nicky's directive to
integrate the two competing window-2 skeletons. **This file SUPERSEDES both predecessors:**
`docs/judge_comparison_funnel_skeleton.md` (band-miss lineage) and
`out/corpus_audit/handoff/SKELETON_judge_comparison_and_funnel_analysis.md` (panel lineage).
Both are left on disk unedited (append-only / parallel-session etiquette); paste THIS one.

## Why this skeleton replaces both (position, defended)

Full evidence: `out/audits/skeleton_unification_20260710T214021Z/COMPARISON_AND_POSITION.md`.

1. **Neither predecessor's label set survives contact with the other.** The two audits' flag sets
   overlap at Jaccard 0.448 (κ=0.548) on common rows. A fresh 18-row adjudication panel (blind-first,
   math checked directly) resolved every A-vs-B conflict sampled: the 12 disputed rows went 9 ill /
   3 well; a 6-row sample of B-only flags confirmed 6/6 ill (one resolving an A needs_human row).
   Skeleton-A would calibrate judges against 43 labels containing 3 confirmed over-flags while
   missing ≥14 confirmed/presumptive misses; skeleton-B would spend its budget re-deriving a
   reconciliation that is now DONE.
2. **The reconciliation output already exists as data:**
   `out/audits/skeleton_unification_20260710T214021Z/adjudicated_labels.jsonl` — 65 rows:
   41 evidence-confirmed ill (E1 = 26 both-flagged + 9 disputed-confirmed + 6 B-only-confirmed),
   12 presumptive ill (E2 = 3 pending-overlap + 7 unexamined substantive B-only + 2 rescue-lane
   A-flags), 8 policy (circular/degenerate — outside A's rubric BY DEFINITION; Nicky must rule
   whether the class is removal-worthy), 3 resolved-well (de-flagged; false-kill sentinels),
   1 needs_human. Consume it; do not re-litigate it.
3. **Empirical priors for the funnel work shifted:** in all 18 adjudications the correct verdict
   was reachable from statement-internal derivation; source_statement access flipped nothing
   (though it uniquely diagnoses extractor-ADDED content — 2 confirmed cases). The decisive
   failure mode everywhere (production gate, A's single-pass keeps, B's shallow keeps) was
   accepting *recoverability* ("an expert would know") in place of *derivability*. Rank
   candidate fixes accordingly (see P3).
4. **Contract conflicts resolved:** working artifacts under `out/audits/…` (A's convention, new
   dir); the window-3 execution skeleton at `docs/funnel_adjustment_execution_skeleton.md` (A's
   contract — Nicky pastes from docs/) — this is the ONE docs/ write besides the SESSION_HANDOFF
   addendum, overriding B's blanket no-docs rule for exactly these two paths; B's orchestrator
   hygiene, provenance discipline, and RESUME_STATE checkpointing are adopted wholesale.

---

## Fresh Window Prompt (paste from here down)

You are the ORCHESTRATOR of the icepick judge-quality investigation, window 2 (unified).
Repo: `/Users/redhairing/Desktop/helloworld/icepick` — shell cwd resets between Bash calls; use
absolute paths. Delegate bulk reading/ruling to read-only subagents; keep your context for
synthesis. Parallel sessions are common in this checkout and task notifications have been
fabricated/premature before: verify disk + `ps` before acting on any event; disk is truth.

READ IN ORDER before any action:
1. `AGENTS.md` — binding invariants (esp. 8 never-reject-good-theorems, 9 one-Qwen, 10 out/**
   append-only, 11 launches hold-gated, 12 $5 HITL spend line; judge models come from the key
   env files' `ANTHROPIC_MODEL`/`OPENAI_MODEL` lines, NEVER `--*-judge-model` flags; never read
   or print `*key.env` contents).
2. `out/audits/skeleton_unification_20260710T214021Z/COMPARISON_AND_POSITION.md` +
   `adjudicated_labels.jsonl` + `panel_results.jsonl` — your label set and its provenance.
3. `docs/wellposed_miss_audit_summary_20260710.md` (mechanism clusters 1–7, lane table) and
   `out/corpus_audit/handoff/FLAG_SUMMARY_FOR_FRESH_CONTEXT.md` (5 failure classes, era table,
   gate-bypass mechanisms) — the WHY behind the labels.
4. `out/corpus_audit/AUDIT_REPORT.md` §7 (15 pending splits) and §8 (foreign-file incident —
   in `out/corpus_audit/`, IGNORE the files it names; headline numbers exclude them for cause).
5. `docs/pipeline_controller.md` §"Stage 3: wellposed-cascade" + `src/posers/AGENTS.md`
   (judge cache keys on prompt text — any prompt edit re-bills that judge's samples; per-provider
   model config traps).

PREFLIGHT (record in `out/audits/judge_comparison_<UTC>/manifest.json` before analysis):
`band_corpus.jsonl` row count + sha256 vs `01609862e21fde14…`/309 — **if it differs, Nicky has
folded rows: STOP, report the delta, and ask him** whether labels carry over. Git branch/dirty
state; `ps` for icepick/qwen processes; confirm your output dir is new.

### P1 — Close the reconciliation residue (small; subagents)

The cross-audit reconciliation is DONE (labels file above). Remaining evidence gaps only:
- Adjudicate the **7 unexamined substantive B-only rows** (`adjudicated_labels.jsonl` status
  `presumptive_ill_unverified`: `01464d48`, `11e30827`, `343249ba`, `516b7d3f`, `549b8fc7`,
  `a1272ad9`, `bd5420fc`) with the SAME protocol that produced `panel_results.jsonl`
  (blind stage-1 on statement+answer → stage-2 source_statement check → stage-3 both audits'
  written arguments → math checked directly; one read-only agent per row; provenance-stamp every
  output row with model + session id).
- OPTIONAL, only if Nicky releases it in-session: `a7b98a81` (needs_human, rescue-origin) and
  B's 12 remaining pending splits (§7; ~400–500k tokens; append new files — `audit_rulings.jsonl`
  is not yours to edit; aggregation re-runs via `out/corpus_audit/tools/aggregate.py`).
- Assemble **Nicky's final adjudication queue** ordered E1 (41, removal-evidence complete;
  jme-class rows additionally need his math check) → E2 (12±) → policy (8: does circularity/
  degeneracy count as removal-worthy? state the trade both ways) → needs_human → the 3
  resolved-well de-flags (explicit, so he can veto). Include the band-size arithmetic per tier
  (309→268/256/248). **Corpus mutates only on his ruling** (AGENTS.md inv 8 spirit: flags are
  not removals).

### P2 — Benchmark the OTHER judges against the adjudicated labels (disk-first, $0 default)

Question: would a different judge have caught what codex:anthropic passed? Study population:
the 41 E1 + 12 E2 positives, the 3 de-flagged sentinels, and ~45 negative controls sampled from
double/triple-confirmed keeps (both audits' keep + not in union), stratified by era/lane and
batch; record the sampling rule + seed list in the manifest. Keep `a7b98a81` out of metrics.
Mine existing verdicts on disk (report coverage explicitly; no silent gaps):
- `wellposed_all_with_passk.json` `stage3_advisory_flag` (batches 3–8): did the advisory
  claude:openai stage flag any of the 11 era2adv misses?
- `out/wellposed_pde625_claude_anthropic/verdicts/` (+ `comparison_vs_cascade*`): a different
  BUILD, same provider — verdicts where uids overlap.
- Cascade stage verdict/cache files under `out/processing_*/cascade/` for each positive's source
  batch: pull codex:anthropic's verdict/rationale text AND per-sample votes for all positives
  (needed for P3 regardless — note 2-1 squeakers).
- Rescue-panel rulings (`out/stage1_kill_census/…`) for the 2 rescue-lane flags.
Metrics per judge source: recall on E1 / E1+E2, false-flag rate on controls AND on the 3
sentinels, breakdown by failure class and mechanism cluster, per lane/era. Qualitative per miss:
did that judge's rationale SEE the defect and excuse it, or never see it (feeds H1–H5).
LIVE judge runs: only on Nicky's explicit in-session release — claude:openai (gpt-5.5-strength,
≈$5.4 for ~90×3 samples → OVER the $5 line, needs sign-off) or claude:anthropic (≈$1, still
hold-gated, inv 11). Blind, statements only, new output dir, restartability contract.

### P3 — Codex:anthropic funnel structural analysis (read code + artifacts; no edits)

Entry points: `src/icepick/processing/poser/cascade.py`, `src/posers/Codex_Poser/` (the judge
prompt/rubric text is the object under analysis), `docs/pipeline_controller.md`.
1. Establish what the judge actually sees per record (statement? answer? ever
   `metadata.source_statement`?), the sampling/uphold policy (`--judge-samples 3 --judge-uphold
   2`), and the rubric's actual questions.
2. Verify each failure class's hypothesized bypass mechanism against ≥3 real gate transcripts of
   flagged records (subagents read; you synthesize). The seeded hypotheses H1–H5 (judge never
   sees source / scores expert-answerability not statement-determinacy / single-gate variance /
   no stand-alone symbol-closure check / no blind-solve probe) start with UPDATED priors: H2 is
   empirically confirmed central (it also felled both audits' keep sides); H1 is down-weighted as
   catch mechanism (0/18 verdict flips) but is the only detector for extractor-ADDED content.
3. Evaluate adjustments, each with per-class expected catch on the labeled positives, false-kill
   risk argued from the 3 sentinels + controls (remember the 82.5% stage-3 false-kill history and
   inv 8 — flag/quarantine semantics, never silent hard-kill), and cost per 250-record batch:
   S1 $0 deterministic pre-judge lint (dangling-referent/placeholder patterns; estimate
   false-positive rate on ~50 keeps — the sentinels show idiom-pinnable phrasing must survive);
   S2 rubric addition: stand-alone checklist + **"re-derive the answer; do not assume the
   source's/recalled version"** (empirically the top candidate); S3 give judge source_statement
   + preservation question (catches extractor-added "sharp"/"parameter" class; costs cache-key
   re-bill per src/posers/AGENTS.md); S4 blind-solve probe sample (targets
   inversion/monotone-membership classes); S5 second independent gate for Sonnet-only lanes
   (Sonnet ≈$1.8–2.4/250-batch; gpt-5.5 advisory ≈$20–45 — over the line, Nicky-only);
   S6 uphold tightening 3-of-3 (quantify against the per-sample votes from P2 before proposing;
   false-kill impact explicit). Backtest every candidate BY READING against the full label set;
   report catch/false-kill with the uids behind them. Recommend a minimal package (1–2 items).

### Deliverables (this window)

1. `out/audits/judge_comparison_<UTC>/comparison_report.md` — P1 queue + P2 metrics + P3
   findings, ranked S1–S6 (or better ideas found in the data) with catch/false-kill/cost each.
2. `docs/funnel_adjustment_execution_skeleton.md` — paste-ready window-3 skeleton to IMPLEMENT
   the chosen package: files to touch (cascade.py / poser prompt files / new lint module path),
   verbatim rubric-text draft, flag-not-kill wiring, cache-key/re-billing consequences stated,
   AGENTS.md invariants cited by number; validation experiment = re-judge all E1+E2 positives +
   the 3 sentinels + recorded controls with acceptance criteria stated NUMERICALLY up front
   (starting point: E1 recall ≥60% at ≤5% control false-kill AND 0/3 sentinel kills — tune from
   your P2/P3 data), three-suite baseline gate (AGENTS.md Quick facts numbers current at that
   time), cost table + spend/hold gates, rollout plan (behind flag / next-batch-only /
   era1sonnet retro-rescreen as SEPARATE Nicky decisions), pause/resume checkpointing, and a
   token/spend budget placeholder for Nicky.
3. One-paragraph `docs/SESSION_HANDOFF.md` addendum at session end + memory update (extend
   `skeleton-unification-adjudication` or add `funnel-adjustment-analysis`).

### Binding rules

- **No code changes; no edits to any existing file.** New files ONLY under your
  `out/audits/judge_comparison_<UTC>/` dir + the two docs-deliverables named above.
- Never modify corpus files, `out/corpus_audit/**`, or the other audits' dirs. No commits, no
  pushes. $0 default: no paid API, no Qwen, no scrapes/pass@k — any live run needs Nicky's
  explicit in-session release (inv 11), and >$5 additionally his spend sign-off (inv 12).
- Treat labels as confidence-tiered evidence, not ground truth; do not re-litigate E1 rows
  except where a P2 judge disagreement forces a look. needs_human rows stay out of recall
  metrics and are never resolved by assumption.
- Provenance-stamp (model + session id + UTC) every ruling/verdict row you or your subagents
  write — unstamped files in shared dirs caused the §8 incident.
- Checkpoint `out/audits/judge_comparison_<UTC>/RESUME_STATE.md` at every phase boundary.
- STOP and ask Nicky before: any live/paid judging; any write outside the named locations; any
  conclusion requiring corpus relabeling; recommending an adjustment whose false-kill projection
  cannot be bounded from data; and on preflight sha mismatch.

Token/pause budget: [NICKY: set here]. Optional releases Nicky may grant inline:
[ ] a7b98a81 + B's 12 pending splits (~400–500k tok) [ ] live claude:anthropic (≈$1)
[ ] live claude:openai (≈$5.4, over HITL line).

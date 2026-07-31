# Proof import — execution skeleton (mine paper proofs → verified worked solutions)

**Paste this whole file into a fresh window.** Mission slug: **proof-import**.
Status at write time (2026-07-31): **NOT RELEASED** — Nicky arms §1. Repo
`/Users/redhairing/Desktop/helloworld/icepick` (`cd` every Bash call). Read `AGENTS.md`,
then `docs/SESSION_HANDOFF.md`. Downstream consumer:
`docs/lora_v3_proofhint_execution_skeleton.md` (v3 training) — this skeleton produces
its required input.

**Why this exists.** The QA extraction kept endpoints and discarded paths: Sonnet's
prompt says *"extracted directly from the theorem — do not derive or compute anything
new"*, and the miner's `_THEOREM_ENVS` (`realmath.py:649`) never included `proof`
environments. So every record is (question, answer) with the paper's derivation left
in the `.tex`. Training on own-rollouts therefore saturated (see
`docs/lora_v2_verdict.md` + the k=8 sharpening evidence). This skeleton recovers the
missing middle: `(question, proof_raw, solution_text, answer)` for **train-split
records only**.

---

## 0. VERIFY BEFORE TRUST (fabrication-history environment)

| pin | expected |
|---|---|
| corpus | `out/corpus_pde625/band_corpus.jsonl` 293 rows sha[:16] `e0975e11` |
| wellposed pool | `out/corpus_pde625/wellposed_all_with_passk.json` ~2021 records |
| split | `evalharness/data/corpus_split_200_100.json` sha[:16] `768436f4` — 200 train / 100 holdout uids |
| cached e-prints | `out/qa_repair_20260711T055242Z/fetched/*.tex` (partial paper coverage — inventory in P1) |
| ref-resolution | E5 hardening landed (miner resolves `\ref` from full tex, no destructive stripping) — commit `8eefdbb` lineage |
| suites | three-suite 1118 / root 975+3 (AGENTS.md quick facts, measured 2026-07-29) |

## 1. RELEASE CHECKBOXES (Nicky)

- [ ] **R1 — run the mission.** Est: arXiv fetches for ≲163 train papers (paced, free),
      Sonnet reformulation ~200–400 calls ≈ **$2–5**. Over $5 ⇒ stop and ask.
- [ ] **R2 — target set**: `train-split band records (200)` (default) · also
      `collapse/misdirection-tier from wellposed_all on train papers` (feeds the 60/40
      curriculum; adds calls) · or `___`
- [ ] **R3 — fetch release**: proof mining needs paper `.tex`; missing papers are
      fetched via the standard `plan → approve → run` gates (invariants 6/7 — sequential,
      `_pace_lock`, never parallel).

## 1b. EXECUTION SUBSTRATE (Nicky, 2026-07-31: run on RunPod, not local)

P1 inventory, P2 fetch, P3 mining, and the mechanical half of P5 run on a **cheap
RunPod CPU pod** (no GPU needed; create with `PUBLIC_KEY` env or sshd never starts).
arXiv pacing invariants apply identically from a pod IP. Two principled carve-outs,
overridable only by explicit Nicky ruling:

- **P4 Sonnet calls run FROM the local machine.** The Anthropic key is a local path
  proxy that never ships; pod env vars echo back through the RunPod API, so a pod is
  not a safe key holder. The stage is ~300 API calls of negligible local load — the
  pod ships `proofs_raw.jsonl` down, gets `solutions_v3.jsonl` back.
- **P5 endpoint verification runs local** (it consumes pinned answers; train-row
  answers may ship, but the verifier chain and its sympy environment live here).

Pod lifecycle: itemize spend, terminate same session, artifacts scp'd + sha-verified
before teardown (/tmp is not retrieval — archive into the run dir).

## 2. OPERATIONAL RULES (scars, binding)

1. Corpus/split/eval_set/baseline are read-only. **Zero corpus mutation** — all output
   to a new `out/proof_import_<ts>/` dir.
2. **Holdout uids never enter this pipeline.** Not fetched-for, not mined, not
   reformulated. Hard guard at intake (P1) AND at output (P5) — a holdout proof on
   disk is the answer key crossing the split.
3. arXiv: sequential only, honor `ICEPICK_ARXIV_MIN_INTERVAL`, resume-on-429 per
   scraper runbook. Reuse cached e-prints before fetching anything.
4. Keys are path proxies (`ANTHROPIC_KEY_FILE`); never print. Sonnet calls
   disk-cached by `(uid, sha256(proof_raw))` — restartable, never re-billed
   (invariant 3).
5. Parallel sessions share this checkout — `ps` + progress-log check before any
   long-running work; one pytest runner at a time.

## 3. PHASES

### P1 — inventory (read-only, $0)
Map train-split uids → `arxiv_id` → tex availability (cached vs fetch-needed).
Emit `inventory.json`: per-paper record counts, cached-tex hits, fetch list.
**Accept:** every train uid classified; fetch list ≤ train-paper count (~163); zero
holdout uids anywhere in the file.

### P2 — fetch missing tex (R3-gated)
Standard paced fetcher. Store under the run dir (`fetched/`), never re-fetch cached.
**Accept:** per-paper tex or a logged fetch-failure; rate-limit telemetry in report.

### P3 — proof mining
Extend the miner (new module in the run dir or `src/icepick/.../proof_mine.py` —
coder's call, but **do not modify `extract_theorem_candidates`'s existing behavior**):
capture `\begin{proof}...\end{proof}` incl. `[Proof of Theorem N]` variants; match
proof→theorem by (a) adjacency (nearest following proof env), (b) explicit
"Proof of X" label/ref cross-match — record `match_method` + `match_confidence`.
Resolve refs via the E5 resolver (never strip). Store
`proofs_raw.jsonl`: `{uid, arxiv_id, proof_raw, match_method, match_confidence}`.
**Accept:** coverage census (matched / unmatched / paper-has-no-proof-env / proof-says-
"omitted") — ALL four classes counted, none silently dropped (invariant-8 spirit).

### P4 — Sonnet reformulation (the one paid stage)
One call per matched record: statement + proof_raw + **pinned answer** → JSON
`{solution_text, faithful: bool}` where `solution_text` = worked solution in the
pass@k wire idiom ending `\boxed{<answer>}`, derived FROM the paper proof (no new
mathematics); `faithful=false` when the proof is a citation-stub/"omitted"/too
elliptical to reformulate — count, don't force. Cache per rule §2.4.
**Accept:** call count ≡ cache entries; spend reported; refusal census.

### P5 — verification + publish
Endpoint-verify every `solution_text` with the audited chain
(`scoring.extract_candidate` → `verifier.classify/verify`): the boxed endpoint must
verify against the pinned answer. Rejects quarantined with reasons.
Publish `solutions_v3.jsonl`: `{uid, question, proof_raw_sha, solution_text, answer,
provenance{arxiv_id, match_method, match_confidence, sonnet_cache_key, verified: true}}`
+ `manifest.json` (input shas, split sha, censuses, spend) + `REPORT.md`.
**Accept:** 100% of published rows verified; **0 holdout uids** (re-assert against the
split file by sha); corpus/split untouched (sha re-check); suites green.

## 4. SURFACE, DO NOT DECIDE

Multi-proof theorems (which proof wins); appendix/split-file proofs the adjacency
matcher misses; the "proof omitted / follows from [N]" fraction (if large, the
curriculum shrinks — Nicky reweighs R2); collapse-tier wellposedness (band audits
don't cover that tier — flag rate of suspicious records for a possible mini-panel);
any record whose reformulation requires mathematics not in the proof (reject class,
but report examples).

## 5. WHAT DONE LOOKS LIKE

`out/proof_import_<ts>/solutions_v3.jsonl` with verified worked solutions for a
reported fraction of train-split records, a four-class coverage census, spend actuals,
zero holdout contamination, zero corpus writes — ready to be consumed by the v3
skeleton's dataset build.

# Well-posed band miss audit — results summary for downstream analysis windows

Written 2026-07-10 by the audit orchestrator (Claude Fable 5) for a FRESH context window.
Source of truth: `out/audits/wellposed_band_miss_audit_20260710T010302Z/` (below: `$AUD`).
This summary explains WHAT was flagged and WHY. It contains no instructions; your tasking
skeleton is `docs/judge_comparison_funnel_skeleton.md`.

## What the audit was

All 309 rows of `out/corpus_pde625/band_corpus.jsonl` (sha256 `01609862e21fde14…`, verified
unchanged across the audit) were reviewed for well-posedness **false positives** — records the
cascade accepted as well-posed that do not actually pose a valid single-answer problem.

Operative standard (from the audit rubric): a record is a miss **only if** the problem AS POSED
does not determine a single fixed answer under standard mathematical conventions, or the keyed
answer is not actually determined by the statement. The statement must stand alone:
`metadata.source_statement` (the theorem the QA extractor worked from) containing load-bearing
information that `statement` lacks IS a miss. Explicitly NOT miss reasons: difficulty, advanced
but standard terminology, answer given in one of several equivalent forms. pass@k fields were
weak hints only, never the decision basis.

Process: 16 independent first-pass reviewer agents (one per shard, full rubric) → blind
independent second review of every flagged row + all 27 rescue-path rows → orchestrator
adjudication of splits (rationales in `$AUD/raw/orchestrator_state.json`). 20 rows incidentally
received a third independent review. Reviews: `$AUD/agent_reviews/`.

## Headline results

| bucket | rows |
|---|---|
| keep | 264 (239 unflagged + 23 rescue double-confirmed + 2 adjudicated keeps) |
| **miss candidates** | **43 (13.9%)** |
| needs human | 2 |

Tier semantics (confidence in the flag, by independence of agreement):
- **T1 unanimous (34)** — every independent reviewer (2 or 3) said miss.
- **T2 majority (5)** — 2-of-3 said miss, one dissented; orchestrator concurred with majority.
- **T3 adjudicated (4)** — reviewers split 1-1; orchestrator ruled miss with recorded rationale.

Full evidence per row (both/all reviews, evidence fields, minimal fixes, pass@k hints):
`$AUD/miss_candidates.jsonl`. Full-length uid lists per tier: `$AUD/audit_report.md`.

## WHY the problems were flagged — mechanism clusters

Formal miss-type counts: missing_context 18, answer_not_determined 11, extraction_mismatch 6,
multiple_answers 4, ill_typed_or_invalid 2, convention_dependent 2. Mechanically these cluster
as follows (clusters overlap; exemplar uids in parentheses):

1. **Elided equations / dangling referents** (~10 rows, biggest cluster). The QA extractor kept
   phrases like "a system", "an ODE", "the stated assumptions", "as in Lemma", "unique weak
   solution" while the actual equations/hypotheses they refer to never made it into the
   statement — usually because the source theorem referenced them by LaTeX \ref/\eqref that
   resolved to nothing. The keyed answer is the paper's result, not derivable from the shell
   that remains. (`6953711f` — the ODE is literally absent; `5cab6922` — a "chemotaxis-fluid
   system" with zero equations, answer 8*pi rests on an unstated normalization; `2ac7d605`,
   `878e7f40`, `6ea5f539`, `504b44db`, `3fe21126`, `570fcab3` — undefined K_M whose unit mass
   is equivalent to unstated stochastic completeness.)

2. **Dropped load-bearing hypotheses.** The statement carries most hypotheses but the one that
   forces the answer was dropped in extraction; concrete counterexamples exist to the posed
   version. (`2080cb02` — u(x0)=0 dropped, u≡c>0 satisfies everything stated; `f5416819` —
   "as in Lemma" warping conditions dropped, psi=sinh gives L_kappa=+inf, not n^2/(n-1);
   `a46de20f` — "B nonnegative self-adjoint" dropped, eigenvalue sign flips by convention;
   `c52008d3`, `6e6d34ec`, `3d1102f3`.)

3. **Assertion→question inversions that lose sharpness/uniqueness** (~8). The source ASSERTS
   "u ∈ H^2s" or "there exists K>0 such that…"; the extractor asks "to which space does u
   belong" / "what is K", but membership is monotone and existence-constants are not unique, so
   several answers are true and the keyed one is just the source's. Graded pass@k answers that
   are mathematically TRUE get scored wrong. (`d682389a` — H^s also true, 7/8 rollouts said it;
   `63071b85`, `47eac1e1` — any larger constant works; `1f8b88e6` — asks SHARP where source
   proves only an inequality; `63d77c1e`, `a7afda27` — interpolation gives a bound, not the
   exact norm; `33c42f6b`, `76ac6e18`.)

4. **Extractor-added or -distorted content** (extraction_mismatch, 6). The question asks
   something the source doesn't establish, or contradicts it. (`1faf092f` — extractor ADDED
   "sharp" to a non-sharp bound; `fe1fde9b` — asks for a proof-internal initialization value;
   `8d254fa0` — paper's elided equation rendered as the WRONG named PDE; `292cb9e2` — nabla^s
   glossed as the local symmetric gradient, breaking determinability; `2ba04886` — keyed answer
   provably violates the stated constraints (w=1 substitution); `13c26a72` — same class.)

5. **Multiple answers / parse ambiguity** (4). Open-ended phrasing ("what is the relationship
   between nu and mu", `3ede4dd9`) or ambiguous arithmetic ("[display1] minus [display2]",
   `7cfd5ab3`) admits several defensible readings with different answers.

6. **Convention dependence** (2 + overlaps). Answer flips under equally-standard conventions
   the statement doesn't fix. (`529f8161` — spectral-bound sign for A vs -A; `9df79bb0` — 'q'
   called "parameter" (A&S: sqrt(1-q)) while the key needs the modulus reading (sqrt(1-q^2)).)

7. **False/inconsistent premises** (2). The stated givens are mathematically false and license
   a different answer than the key. (`9df79bb0` — "dn is K(q)-periodic" is false (period 2K)
   and with continuity forces dn(K)=1, contradicting the keyed sqrt(1-q^2); `f129aa3e` — series
   index starts n=1 omitting the Mainardi n=0 term, making the asked object negative/invalid;
   `def98456` — a proof-by-contradiction fragment posed as a question, premises inconsistent.)

Corroborating (weak, never decisive): in several flags the pass@k `modal_wrong` was exactly the
OTHER defensible answer (`d682389a`: modal wrong H^s at 87.5% share; `9df79bb0`: modal wrong 1,
the false-premise answer) — the subject model's failures track the ambiguity, i.e. some "band"
difficulty is manufactured by ill-posedness rather than genuine hardness.

## Which funnel lanes produced the misses

| wellposed_via lane | miss/total | rate |
|---|---|---|
| cascade_sonnet_only_codex_anthropic | 18/108 | 16.7% |
| cascade_sonnet_only | 7/39 | 17.9% |
| cascade_2stage_plus_advisory | 11/97 | 11.3% |
| cascade_full_3stage | 3/23 | 13.0% |
| cascade_1stage_claude_anthropic | 1/11 | 9.1% |
| cascade_3stage_gating | 1/4 | — |
| stage1_false_kill_overturned (rescue) | 2/22 | 9.1% |
| stage3_false_kill_overturned (rescue) | 0/5 | 0% |

The Sonnet-only single-gate lanes (codex:anthropic as the ONLY judge, gpt-5.5 removed) run
**25/147 = 17.0%** vs **11.3%** for the older 2-stage lanes — dropping the second independent
gate coincides with a ~1.5x higher miss rate. Samples are modest; treat as a signal, not a
proof. Rescue lanes are CLEANER than average (2/27) — panel-overturned rows
hold up.

## The 43 candidates

| uid (12) | arxiv | batch | lane | tier | miss_type | conf | why flagged |
|---|---|---|---|---|---|---|---|
| `4609495db907` | 2601.00934 | batch13 | sonnet-1stage(b) | T1 M/M | answer_not_determined | 0.91 | Extraction dropped the source iff-hypothesis that Λ=SA has a unique fixed point; X=Y={1,2}, P={(1,1),(2,2)}, S=id satisfies everything stated yet give |
| `2ac7d60555c3` | 2506.08697 | bulk_june | sonnet-1stage | T1 M/M | missing_context | 0.90 | Statement invokes 'the stated assumptions', a 'very weak solution' of an unspecified equation, and undefined X_delta and alpha; nothing posed mathemat |
| `2ba04886f755` | 2506.20775 | bulk_june | sonnet-1stage | T1 M/M | extraction_mismatch | 0.85 | The rings overlap at most 2-fold, so every partition satisfying the listed properties has sum of squares at least 1/2 (sharp); 1/3 is only the source  |
| `c3bb8ee0cd4b` | 2603.14715 | batch8 | 2stage+adv | T1 M/M | missing_context | 0.85 | No dimension/domain is stated, so the standard reading gives np/(n-alpha p); the 1D time-scale context that forces p/(1-alpha p) was dropped. |
| `c52008d3a92d` | 2601.13916 | batch12 | sonnet-1stage(b) | T1 M/M | missing_context | 0.85 | Constants solve the PDE under all stated hypotheses; the integrability condition forcing f≡0 is reduced to the placeholder '(with appropriate integrab |
| `f129aa3ea4dc` | 2507.02094 | bulk_july | sonnet-1stage | T1 M/M/M | ill_typed_or_invalid | 0.83 | Series as written starts at n=1, omitting the Mainardi n=0 term 1/Γ(1−α), so Ψ_α(s)→−1/Γ(1−α)<0 and the posed integral diverges, not 1. |
| `3d1102f30d6f` | 2507.08689 | bulk_july | sonnet-1stage | T1 M/M | missing_context | 0.82 | ξ̂_i and b̃_k are never defined and 'as specified' references absent relations, so the correction term ξ̂_i·∇√α cannot be computed from the statement. |
| `c32ab688c2bf` | 2605.16554 | batch4 | 2stage+adv | T1 M/M | answer_not_determined | 0.82 | Counterexample: E = x + y^2, F = (-2xy, x), equilibrium (0,0) satisfies all stated hypotheses yet A^T H + H A = [[0,2],[2,0]] != 0; needs unstated gra |
| `33c42f6b2d56` | 2506.14423 | bulk_june | sonnet-1stage | T1 M/M | answer_not_determined | 0.81 | Source only asserts existence of C_phi; the true value is 2/W_phi/ (Wulff-shape area), genuinely phi-dependent (scaling phi to 2phi quadruples it), so |
| `13c26a72e36e` | 2604.20430 | batch5 | 2stage+adv | T1 M/M | extraction_mismatch | 0.80 | Taking w=1 in the displayed identity forces b = -/Omega///bdry Omega/ != 0, so the keyed answer 0 contradicts the statement and appears in no shown so |
| `2080cb02563d` | 2512.17543 | batch14 | sonnet-1stage | T1 M/M | missing_context | 0.80 | The hypothesis u(x0)=0 was dropped: u = c > 0 constant with f = 0 satisfies every stated condition including the two-sided inequality and zero normal  |
| `6ea5f5393421` | 2603.07042 | batch9 | claude-1stage | T1 M/M | missing_context | 0.80 | The PDE that u solves is never given in the statement or even the source excerpt, so the touchdown characterization liminf min u = -1 cannot be derive |
| `b205a8d3fdf8` | 2601.20372 | batch12 | sonnet-1stage(b) | T1 M/M | missing_context | 0.80 | The asked quantity lambda_1 is never defined in the statement; it is the paper's principal eigenvalue, defined outside the quoted lemma. |
| `fe1fde9bea97` | 2507.20310 | bulk_july | sonnet-1stage | T1 M/M/M | extraction_mismatch | 0.80 | Asks for a proof-internal initialization ('before it is possibly increased in the proof'); k0 = 1 is an arbitrary choice in the unshown proof, not a q |
| `3fe2112604fe` | 2603.26534 | batch7 | 2stage+adv | T1 M/M | missing_context | 0.78 | The governing PDE is never stated, and with the statement's g(0)<-delta-sqrt(delta^2+2K) the answer's log argument lies in (0,1), giving an impossible |
| `df24d38537e0` | 2605.03920 | batch5 | 2stage+adv | T1 M/M/M | answer_not_determined | 0.78 | The representation is invariant under psi -> c*psi, so the statement admits any nonzero psi(z_0); the value 1 is only the source's dropped normalizati |
| `878e7f405e90` | 2603.21805 | batch7 | 2stage+adv | T1 M/M | missing_context | 0.77 | The equation u solves is never given — 'unique weak solution' dangles — so alpha = 1/2 is the source paper's result, not derivable from the statement. |
| `1faf092f88d4` | 2507.04938 | bulk_july | sonnet-1stage | T1 M/M | extraction_mismatch | 0.75 | Extractor added 'sharp': the stated hypotheses force Σ_Γ/Q_i/ > ((α−2)/(α−1))/E/, strictly stronger than (1−2/α)/E/, so the supplied answer is not the |
| `529f8161b000` | 2604.15991 | batch6 | 2stage+adv | T1 M/M | convention_dependent | 0.75 | With A Phi_n = -lambda_n Phi_n, sigma(-A) = {lambda_n} is unbounded, so the standard spectral bound s(-A) = sup Re sigma = +infinity, not the keyed 0. |
| `987988843840` | 2601.21422 | batch12 | sonnet-1stage(b) | T1 M/M/M | answer_not_determined | 0.74 | The asserted bound with c=1 already fails at t=0 for admissible u_0 with sup near 1; the statement's constraints force c>=2, so 1 is not determined. |
| `def984568077` | 2604.13578 | batch6 | 2stage+adv | T1 M/M/M | missing_context | 0.74 | Proof-by-contradiction fragment posed as a question: the stated premises (rho_2>rho_1 somewhere, t rho_1 = rho_2, rho_1>0) force t>1, contradicting th |
| `1f8b88e6edc4` | 2602.14522 | batch10 | sonnet-1stage | T1 M/M | answer_not_determined | 0.72 | Question asks the SHARP bound, but the source only proves an inequality; a direct spectral argument gives the strictly better delta/gamma, so sqrt(2)d |
| `292cb9e291c0` | 2601.17579 | batch12 | sonnet-1stage(b) | T1 M/M | extraction_mismatch | 0.72 | Statement glosses nabla^s as the local 'symmetric gradient', under which data on U cannot determine A-B on R^d minus U, so the asked a.e.-R^d answer i |
| `8d254fa09470` | 2507.23583 | bulk_july | sonnet-1stage | T1 M/M | extraction_mismatch | 0.72 | The extractor rendered the paper's elided equation as 'the wave maps equation', but the source is the harmonic map heat flow from B^2 to S^2 (title, b |
| `a7afda2767c5` | 2507.16712 | bulk_july | sonnet-1stage | T1 M/M/M | answer_not_determined | 0.72 | Interpolation only yields the upper bound A_tau <= A0^{1-tau}A1^tau; the exact operator norm is not a function of A0, A1, so the equality answer is no |
| `63071b852804` | 2605.13389 | batch4 | 2stage+adv | T1 M/M | answer_not_determined | 0.68 | No sharpness is required, so infinitely many constants satisfy the stated inequality; the sharp value exceeds p-1 (collinear limit gives (p-1)2^{2-p}) |
| `7cfd5ab30643` | 2508.12231 | bulk_august | sonnet-1stage | T1 M/M | multiple_answers | 0.68 | '[display1] minus [display2]' is parse-ambiguous: standard parenthesized reading X-(A0-B0-C) gives 2*(initial field)+2*(3 int f) terms, only chained s |
| `63d77c1e1467` | 2601.20515 | batch12 | sonnet-1stage(b) | T1 M/M | answer_not_determined | 0.66 | Interpolation yields only //T// <= A_0^{1-tau}A_1^tau; the exact interpolated norm is not a function of A_0, A_1, so the asked equality is unforced. |
| `3ede4dd92aac` | 2607.02492 | pde_all26 | 3stage-gate | T1 M/M | multiple_answers | 0.65 | Open-ended 'what is the relationship' admits several true answers — the weaker ν≪μ follows from /ν/≤μ and is equally responsive; 75% of rollouts gave  |
| `6e6d34ec4ac1` | 2508.00337 | bulk_august | sonnet-1stage | T1 M/M | missing_context | 0.65 | Statement omits the flattening T(B_r∩Ω)=B_r∩{x1>0} and the explicit E1,E2 definitions, so the frame in which orthogonality means θ1=θ2=0 is not fixed. |
| `76ac6e1806b6` | 2603.21956 | fksweeprescue | stage1-rescue | T1 M/M | multiple_answers | 0.65 | For every c<delta some eps_0 makes the bound >= c hold, so 'the lower bound' has infinitely many true completions; delta/2 is only the conventional ha |
| `47eac1e1fec9` | 2601.12609 | batch13 | sonnet-1stage(b) | T1 M/M | answer_not_determined | 0.62 | The source asserts only 'some K > 0'; as posed any K >= Lip(phi) qualifies, and the sharp value is product-norm-dependent (sqrt(K1^2-K2^2)/K2 Euclidea |
| `2f3d8afefb6f` | 2602.15394 | batch10 | sonnet-1stage | T1 M/M | missing_context | 0.60 | Stated hypotheses (two critical points, sigma between critical values) allow 1, 2, or 3 roots depending on omitted endpoint behavior of p; b is undefi |
| `a46de20f9531` | 2607.01096 | batch1 | 3stage | T1 M/M/M | convention_dependent | 0.60 | Sign of Laplace-Beltrami eigenvalues is convention-split (plus or minus l(l+n-2)); the source's disambiguating 'nonnegative' hypothesis was dropped, s |
| `93e9bb1106ee` | 2606.26658 | batch1 | 3stage | T2 M/M/K | answer_not_determined | 0.70 | Smoothness plus the stated v-asymptotic does not force lim v_x (oscillatory o-terms defeat it); the source derives it for a specific constructed solut |
| `f54168197265` | 2606.14513 | stage1rescue | stage1-rescue | T2 M/K/M | missing_context | 0.70 | Dropped 'as in Lemma' conditions on psi: e.g. psi = sinh(s) satisfies the posed hypotheses yet gives infinite L_kappa, so delta_0 is not forced. |
| `966324a4b30d` | 2606.12607 | batch2 | 3stage | T2 M/K/M | missing_context | 0.66 | 'Exact value expressed as an upper bound' admits many valid answers, and the intended constant additionally needs /I/ = 1 and an N-by-N grid on I^2, b |
| `9df79bb0b6ec` | 2512.21802 | batch14 | sonnet-1stage | T2 M/K/M | ill_typed_or_invalid | 0.61 | Premise 'K(q)-periodic' is false (dn has period 2K) and, with dn(0)=1, forces dn(K)=1, contradicting strict decrease and the intended sqrt(1-q^2). |
| `d682389a83bb` | 2604.20205 | batch5 | 2stage+adv | T2 M/K/M | multiple_answers | 0.58 | 'To which Sobolev space does u_alpha belong' has many true answers (every H^sigma, sigma <= 2s); H^s(M) is a correct canonical membership yet keyed wr |
| `6953711febd5` | 2512.17428 | batch14 | sonnet-1stage | T3 M/K | missing_context | 0.78 | The equation is literally absent ('a global positive solution to an ODE'), and 2*_alpha and the standing assumptions are unresolved references, so the |
| `570fcab32d11` | 2512.18770 | batch14 | sonnet-1stage | T3 M/K | missing_context | 0.60 | M, mu, and K_M are never defined; the exact value 1 requires K_M to be the heat kernel of a closed (stochastically complete) manifold, which is unstat |
| `5cab692208e2` | 2605.17114 | batch4 | 2stage+adv | T3 M/K | missing_context | 0.57 | The PDE system, coefficients, and normalization are replaced by placeholders, and the 8*pi threshold depends on that unstated normalization, so the st |
| `504b44db11c3` | 2507.16143 | bulk_july | sonnet-1stage | T3 H/M | missing_context | 0.55 | The governing epsilon-dependent PDE system is elided ('a system'), so the epsilon^2 rate is not derivable from the statement, though the squared-norm  |

(12-char uid prefixes are unique within the corpus and grep-able in `band_corpus.jsonl` /
`$AUD/enriched_band.jsonl`; full uids in `$AUD/audit_report.md`.)

## Needs-human queue (2) — excluded from miss counts

- `a7b98a81552be057e5af0664d5c286a4` (2605.16554, stage1-rescue, votes H/H/M): NEEDS_HUMAN: answer U^2 needs the paper-specific fact that a-bar^x reconstructs to a unit vector under P; no reviewer could derive it from the statement (votes nh/nh/miss).
- `e5ed37d50674f4d6c4e44a25d427055f` (2601.10488, sonnet-1stage(b), votes H/K/K): NEEDS_HUMAN: -2t L_{u0} coefficient tied to the paper's Benjamin-Ono normalization; votes nh/keep/keep — two keeps concede 'minor normalization risk'; left to Nicky per no-assumption rule.

## For control sampling (judge-comparison use)

Confirmed keeps to draw negative controls from: 239 unflagged first-pass keeps + 23
rescue-path keeps confirmed by 2-3 independent reviews (uid-level verdicts in
`$AUD/agent_reviews/shard_*_reviews.jsonl`; final classification per uid in
`$AUD/raw/orchestrator_state.json` `.final`). Sample controls stratified by source_batch and
tier to match the 43 candidates' profile.

## Caveats

- Audit verdicts are model-derived (Fable-class reviewers + orchestrator adjudication) with
  independence encoded in the tiers; they are strong evidence, not ground truth. T3 rows had a
  genuine 1-1 reviewer split.
- The corpus was NOT modified; removal decisions are Nicky's and were still pending when this
  was written. Re-verify `band_corpus.jsonl` row count/sha before comparing anything.
- `wellposed_band.json` has a known quirk: its 10 batch3 entries carry null uid/via/tier;
  canonical rows live in `band_corpus.jsonl`.

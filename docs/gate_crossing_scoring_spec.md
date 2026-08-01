# Gate-crossing scoring spec — the campaign's success metric

**Ruled by Nicky 2026-08-01; REVISED same day after an adversarial review found a
gate/code contradiction and an uncalibrated null. This is the authoritative
definition.** Any window scoring an arm against a base ruler uses THIS document.
Supersedes aggregate pass-rate and mean-delta comparisons for headline purposes.

## Why this replaced the old metric

The k=8 sweep showed adapters moving 6–13 band records up to solved while pushing 4–10
down to collapse — flows that nearly cancel, so a mean delta reported ≈0 and hid both.
Counting **problems that cross gates** keeps both flows visible, and matches the mission:
the claim is about which problems became solvable, not about an aggregate rate.

## Labels — from the code, at both sample counts

`src/icepick/contracts/records.py`: `BAND_LO = 0.125`, `BAND_HI = 0.75`, applied as
`BAND_LO <= pass_at_k <= BAND_HI` (`is_band`, records.py:105-110; `scoring.in_band`
reuses the same constants). **The rate is pinned, so count boundaries move with k:**

| label | k=8 | **k=16** |
|---|---|---|
| `fail` (collapse or misdirection) | 0 | **0–1** |
| `band` | 1–6 | **2–12** |
| `solved` | 7–8 | **13–16** |

> **REVISION — blocking fix #1.** An earlier draft used "0/16 = fail, ≥1/16 = pass",
> silently halving `BAND_LO` to 0.0625. **1/16 = 0.0625 is below the band floor; the
> codebase labels it `fail`.** Not cosmetic — the wrong gate generated a systematic
> −9.8-per-100 drift on a null arm (see "Why the code gate also fixes the null").

## Sample counts — pinned

- **Base ruler: k=16**, delivered as **two independent k=8 passes** (the halves are
  reused as free A/A calibration). Measured once, reused by every arm.
- **Arms: k=8 first pass**, plus a second independent k=8 pass (pooled to 16) wherever
  the decision is live.
- Where an arm is not rerun, compare against the base's **first 8 only** — like-for-like
  always. Never arm@8 vs base@16.

## Scoring — integers only, max ±1 per problem

Counting PROBLEMS, not passes. A problem crossing a gate *and* moving ≥4/16 still scores
±1, never ±2.

| transition (k=16 labels) | score |
|---|---|
| `fail` → `band` or `solved` | **+1** |
| `band` → `solved` | **+1** |
| `band` → `band`, Δ ≥ **+4/16** | **+1** |
| `solved` → `band` or `fail` | **−1** |
| `band` → `fail` | **−1** |
| `band` → `band`, Δ ≤ **−4/16** | **−1** |
| otherwise (|Δ| ≤ 3/16, no gate crossed) | **0** |

All thresholds on the 16-scale. No halving, no rounding, no 8-scale conversion.
Outcomes are **computed** from pooled values against this table, never asserted per row.

## Reruns

A second independent k=8 pass, pooled to 16. Fires for:

- **all `band` → `band` records** — the largest bucket, where the ±4/16 rule lives
- **any fail/band boundary crossing in either direction** — base `fail` with arm ≥1
  correct, or base `band` with arm ≤1/8. Under the code gate these are **not** vacuous:
  a first-8 of exactly 1 sits at 1/16 and needs another success to reach the 2/16 floor.

No rerun where the outcome cannot change: `band` → `solved`, and collapses from base
≥6/16 straight to 0/8.

> **REVISION — fixes #9, #10.** The earlier "−1 iff pooled = 0/16" was false: base 4/16
> with a 0/8 first pass and 8/8 fresh pass pools to 8/16, Δ=+4, scoring **+1**. The old
> scope also left **base = 5/16 unruled**; the boundary rule above covers all bases.

**The pooled 16 OVERRIDES the initial 8** for any rerun record — authoritative for
scoring, reporting and labelling.

⚠ **Do not feed rerun-upgraded values into an aggregate mean.** Reruns now cover all
band→band records plus boundary crossings, so the mixed-precision set is large and its
selection is two-sided (band→band conditions on first-8 ∈ 1–6, truncating both tails).
Report aggregates from first-pass k=8 values only, or state the conditioning.

## Why the code gate also fixes the null

Under the discarded `≥1/16 = pass` gate a promotion fired on **one** lucky sample while a
demotion required **zero in sixteen** — rates that diverge sharply at low p, exactly
where the binding tier lives. Measured drift on a **null arm** (arm ≡ base), per 100:

| true p | discarded gate | **code gate** |
|---|---|---|
| 0.05 | **−9.8** | **+0.0** |
| 0.10 | −4.5 | −0.0 |
| 0.125 | −2.7 | −0.0 |
| 0.25 | −0.1 | +0.0 |

The code-faithful gate makes the two directions mirror images and **centres the null at
zero at every p**. The earlier "largely cancels" claim was false under the old gate and
is true under this one.

## Calibration — A/A from the base's own halves

The remaining asymmetries mean the sign test's null of P(+1) = P(−1) is not guaranteed
by construction. Calibrate empirically instead of assuming:

**Score the base ruler's two independent k=8 halves against each other under these exact
rules.** Same instrument, same session, genuinely independent, zero extra compute. That
gives the empirical null for the net score and for promotion/demotion counts. Report the
observed net against that null, not against an assumed zero.

(Do NOT use the corpus's original rescore as the second measurement — different era,
engine and serving config, so not a clean A/A.)

## Significance — convention DECLARED

Non-zero problems form a distribution-free **sign test**: net count is the effect size,
sign test the p-value.

**Convention: two-sided, α = 0.05.** Binding. The record has carried the same v1 result
as both `.344` (two-sided) and `.17` (one-sided); two-sided is conservative and is what a
skeptic applies. Report one-sided alongside if useful; the headline is two-sided.

**Multiplicity / cross-arm.** The base ruler is measured once and reused, so base
sampling error is a **common-mode systematic**: per-arm p-values are NOT independent, and
arm-vs-arm differences have smaller variance than arm-vs-base (shared error partly
cancels). With multiple arms, pre-register a single primary comparison or apply an
explicit correction, and say which. Never rank arms on holdout score and then report the
winner's p-value as if pre-registered.

## The `solved` guard — scored, asymmetry named

Earlier drafts left `solved` unscored "because they can only regress". That was
backwards: **excluding them creates the bias it claims to prevent** — band→solved scores
+1 while solved→band scores 0, so gains into the guard set count and losses out do not.
The 30% measured label drift guarantees the base ruler places real records in `solved`.

**Resolution: `solved` → `band`/`fail` scores −1**, making the boundary symmetric.
`solved` records additionally serve as an instrument guard with an operational trigger:
**if >20% of base-`solved` records regress in any arm, treat the run as suspect and
investigate before reporting.**

## Missing data

Generation failures, grading errors, timeouts: **exclude pairwise and report the count.**
Never silently score 0 — a dropped record and an unchanged record are different, and
three silent instrument bugs in this project have shown how easily that difference
disappears.

## Independence of the two passes

Same serving configuration (engine build, `-fa off`, flags, temperature 0.7,
max_tokens 2048) and a **different sampling seed**, with both passes' configs recorded in
the run manifest. Passes must differ ONLY in sampling randomness — a fourth place a
silent instrument difference could enter.

## Ungradeable records

21 records in the three-tier scope carry `fail` labels that are artifacts of
`simplify(oo−oo) = nan` (`verifier-self-verify-defect.md`), not difficulty. They **can
never promote**, so they add permanent null mass and dilute the sign test. **Exclude them
by name from the scored set and report the count.** (The old 120-record eval measured
0/120 clean, so prior numbers were unaffected; a newly built eval set must be screened.)

## Binding preconditions

1. **"Before" labels MUST come from a fresh base ruler in the same sweep — never corpus
   labels.** Measured label drift is 30%.
2. Base and arms must use identical serving configuration. Three silent instrument bugs
   (`-fa auto`, CUDA-vs-Metal, missing `antlr4`) have already corrupted or nearly
   corrupted results.
3. Grading must run where `antlr4-python3-runtime` is installed; any re-homed grader
   needs a byte-parity receipt against a known-good config first.

## Known statistical properties

- **Magnitude-criterion noise.** |Δ| ≥ 4/16 fires ~21.5% by chance at p=8/16 (~16.3%
  averaged across the band range). On ~50 band→band records that is ~8 spurious flips,
  ~4 each way — net SD ≈ **±2.9**. Treat |net| ≲ 6 as indistinguishable from zero on
  magnitude alone. **Caveat:** those are *unconditional* figures; the band→band pooled 16
  is conditioned on its first 8 landing in 1–6, compressing variance, so realised
  false-positive rate and power are both somewhat lower. **The A/A calibration measures
  the true rate — prefer it over this table.**
- **Expected source of signal.** Per-record effects have been ~0.3/16, so the magnitude
  criterion will rarely fire for genuine reasons. **The bulk of the score is expected
  from gate crossings.** A rare ≥4/16 intra-band move counts exactly the same. **Report
  the two as separate lines**; a verdict resting mainly on intra-band fluctuation should
  be distrusted.

## Threshold choice: why 4/16

2/16, 3/16 and 4/16 are statistically equivalent — power ÷ noise = 0.137 / 0.145 / 0.148,
a dead heat, since a looser bar catches real and fake moves in the same proportion. 4/16
wins on defensibility: false-positive rate 21.5% vs 37.7% vs 59.7% at p=8/16; net noise
±2.9 vs ±4.0 vs ±5.2. At 2/16 roughly 60% of band records flip by chance, and the report
would read "47 improved, 44 degraded, net +3" with ~27 of the 91 being coin flips.

---
Memory: `gate-crossing-metric.md`. Related: `split-rebuild-2026-08-01.md`,
`verifier-self-verify-defect.md`. Run recipe: `docs/v3_full_run_skeleton.md`.

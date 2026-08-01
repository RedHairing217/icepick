# Gate-crossing scoring spec — the campaign's success metric

**Ruled by Nicky, 2026-08-01. This is the authoritative definition.** Any future window
scoring an arm against a base ruler uses THIS document. It supersedes aggregate
pass-rate and mean-delta comparisons for headline purposes.

## Why this replaced the old metric

The k=8 sweep showed adapters moving 6–13 band records up to solved while pushing 4–10
down to collapse — flows that nearly cancel, so a mean delta reported ≈0 and hid both.
Counting **problems that cross gates** keeps the two flows visible. It also matches the
mission: the claim is about which problems became solvable, not about an aggregate rate.

## Definitions

Gates come from `src/icepick/contracts/records.py`: `BAND_LO = 0.125`,
`BAND_HI = 0.75`. At k=8:

| label | n_correct of 8 |
|---|---|
| `fail` (collapse or misdirection) | 0 |
| `band` | 1–6 |
| `solved` | 7–8 |

**`solved` records are unscored instrument guards** (the old anchor role) — they can
only regress, so scoring them would inject one-directional noise. They are watched, not
counted.

## Scoring — integers only, max ±1 per problem

Counting PROBLEMS, not passes. A problem that both crosses a gate and moves ≥4/16
still scores ±1, never ±2.

| transition | score |
|---|---|
| `fail` → `band` or `solved` | **+1** — no rerun, ever |
| `band` → `solved` | **+1** |
| `band` → `band`, Δ ≥ **+4/16** | **+1** |
| `band` → `fail` | **−1** (rerun if base ≤4/16) |
| `band` → `band`, Δ ≤ **−4/16** | **−1** |
| anything else (Δ of 1, 2 or 3 of 16) | **0** |

All thresholds are on the 16-scale. No halving, no rounding, no 8-scale conversion.
**A change of 1, 2 or 3 of 16 is NULL** — only a ≥4/16 move (25 percentage points)
scores on the magnitude criterion.

## Confirmation reruns

A **second independent k=8 pass** (NOT a single k=16 run), pooled with the first to give
16 samples. Fires for:

- **all `band` → `band` records** (base band, arm still band) — the largest bucket, and
  where the ±4/16 magnitude rule lives, so it is where the extra samples buy the most
- **`band` → `fail` demotions from LOW band** (base ≤4/16, i.e. the old 1/8 or 2/8)

**Excluded from rerun** (Nicky, explicit): `fail` records that improved, `band` records
that reached `solved`, and collapses from base ≥6/16 (the old 3–6/8) straight to 0/8.
In each the outcome is already decided and a rerun cannot realistically change it —
a base ≥6/16 dropping to 0/8 scores −1 whether the arm's pooled value lands at 0, 1 or
3 of 16.

**Promotions are never rerun** (Nicky, 2026-08-01 — the test is vacuous). If a
previously-`fail` record's first 8 produced ≥1 correct, the pooled 16 is ≥1/16 by
construction regardless of the rerun, and the promotion override credits +1 either way.
The rerun cannot change its own outcome, so it is pure waste. **fail → ≥1/8 scores +1
directly.**

Because the demotion rerun is scoped to an *exact* value (tuned first-8 = 0), the pooled
threshold is algebraically identical to a clean test on the fresh 8, so reusing the
flagging sample introduces no selection bias: pooled 0/16 ⟺ fresh 8 shows 0.

**The pooled 16 OVERRIDES the initial 8 (Nicky, 2026-08-01).** For any record that was
rerun, the 16-sample measurement is the authoritative tuned value — it supersedes the
0/8 for scoring, for reporting, and for any label assigned to that record. More samples
is the better estimate.

⚠ **Do not feed the upgraded values into an aggregate mean.** Reruns are triggered
exclusively by LOW measurements (tuned 0/8), so only an arm's unluckiest records get
re-measured, and regression to the mean pushes them up. That is harmless for the
gate-crossing count (decided per-problem against fixed thresholds) but would
systematically inflate any secondary statistic — mean n_correct, aggregate pass rate —
computed from the mixed-precision set. Report such aggregates from the first-pass
k=8 values only, or state the bias explicitly.

The demotion rerun does real work — both low-band cases have a live decision boundary
(base shown at 16-scale, arm's first pass was 0/8):

| base | arm pooled | gate | Δ | score |
|---|---|---|---|---|
| 2/16 | 0/16 | **fail** | −2 | **−1** (gate) |
| 2/16 | ≥1/16 | pass | ≤−1 | **0** |
| 4/16 | 0/16 | **fail** | −4 | **−1** (gate + magnitude) |
| 4/16 | ≥1/16 | pass | ≤−3 | **0** |

Under the ±4/16 threshold both low-band cases reduce to the SAME rule: **−1 if and only
if the arm's pooled value is 0/16.** A 4/16 → 1/16 slide is Δ=−3, below the magnitude
bar, and 1/16 is a pass, so it scores 0.

> **Supersession note.** An earlier draft ruled that a base of 2/8 (=4/16) dropping to
> 1/16 scored −1. Raising the magnitude bar from 3/16 to 4/16 superseded that: Δ=−3 is
> now null. Confirmed by Nicky 2026-08-01. This is a deliberate change, not a bug — a
> future reader finding the old rule quoted elsewhere should treat THIS document as
> authoritative.

## Scoring — everything stays on the 16-scale

**Nicky, 2026-08-01: "maintain x/16."** (An initial ±3/16 bar was superseded by
±4/16 after the analysis below.) No
conversion to the 8-scale, no halving, no rounding — which eliminates the ~0.25-point
downward bias that odd-rounds-down would have introduced.

**Base is therefore measured at k=16** so both sides share scale AND precision. (An
8-sample base doubles to even 16-values exactly, but carries different variance; the
k=16 base costs ~1920 generations ONCE and is reused by every arm. Overridable, but
this is the spec default.)

**1. Gate (is it fail?)** — **0/16 = fail, ≥1/16 = pass.** One success in sixteen still
proves the problem is reachable. Crossing the gate in either direction scores ±1.

**2. Magnitude** — **|Δ| ≥ 4/16 scores ±1** (Nicky, 2026-08-01, chosen from a measured
false-positive/power analysis — see below). Δ of 1, 2 or 3 is **null**.

Score ±1 if **either** criterion fires; never ±2. Worked cases:

| base | tuned | gate | Δ | score |
|---|---|---|---|---|
| 2/16 | 1/16 | pass | −1 | **0** |
| 2/16 | 0/16 | **fail** | −2 | **−1** (gate) |
| 4/16 | 1/16 | pass | −3 | **0** |
| 4/16 | 0/16 | **fail** | −4 | **−1** (gate + magnitude) |
| 8/16 | 4/16 | pass | **−4** | **−1** (magnitude) |
| 8/16 | 5/16 | pass | −3 | **0** |
| 6–12/16 | 0/16 | **fail** | ≥−6 | **−1** (no rerun needed) |
| 0/16 | ≥1/16 | **pass** | ≥+1 | **+1** (promotion override) |

### Why 4/16 and not 2/16 or 3/16

Measured on our own setup (base and arm both Binomial(16, p)), all three thresholds are
**statistically equivalent** — power ÷ noise is 0.137 / 0.145 / 0.148 respectively, a
dead heat, because a looser bar catches more real moves and more fake ones in the same
proportion. The tiebreaker is defensibility:

| | Δ≥2/16 | Δ≥3/16 | **Δ≥4/16** |
|---|---|---|---|
| false-positive rate at p=8/16 | 59.7% | 37.7% | **21.5%** |
| power vs a real +3/16 gain | 71% | 58% | 43% |
| net noise over ~50 band→band records | ±5.2 | ±4.0 | **±2.9** |

At 2/16 roughly 60% of band records flip by chance; the flips cancel so the net stays
unbiased, but the report would read "47 improved, 44 degraded, net +3" with ~27 of the
91 being coin flips — exactly the failure mode this campaign has already hit three
times. At 4/16 the claim is "problems whose solve rate moved ≥25 percentage points",
with ~21% chance-driven.

**Caveat ACCEPTED by Nicky (2026-08-01) as expected behaviour, not a problem:**
measured per-record effects in this campaign have been tiny (aggregate ≈ +1.7pp ≈
0.3/16), so the magnitude criterion will rarely fire for genuine reasons at ANY
threshold. **The bulk of the score is expected to originate from GATE CROSSINGS, not
band fluctuation.**

A rare ≥4/16 intra-band move counts **exactly the same as a gate crossing** (+1 or −1)
— not weighted differently, not treated as a lesser event. This makes the design intent
explicit: the metric counts **problems whose solvability status changed**, with large
intra-band moves admitted as a secondary route to the same conclusion.

**Report gate crossings and magnitude moves as separate lines** — not because one is
more valid, but so the source of the result stays visible. If a verdict ever rests
mainly on intra-band fluctuation, treat it with suspicion given the 21.5% chance-flip
rate at this threshold.

### Promotion override — the one intentional asymmetry

A previously-`fail` record reaching **1/16 scores +1**, even though Δ = +1 is below the
±4/16 magnitude bar. The gate criterion carries it. Rationale (Nicky): going from
*never* solvable to *sometimes* solvable is the capability change the curriculum is
buying — "luck can allow a solve, when previously the problem was fully out of scope."
Note this is not symmetric with the demotion side, where a 2/16 → 1/16 slide scores 0.

This is deliberate and documented, not an artifact.

## Binding preconditions

1. **"Before" labels MUST come from a fresh base ruler measured in the same sweep —
   never corpus labels.** Measured label drift is 30%: of a nominal "100 band" holdout,
   only 70 still measured band at k=8 (16 had drifted to solved, 14 to fail). Scoring
   against corpus labels would credit drift as improvement.
2. Base and tuned arms must use an identical serving configuration — same engine build,
   same `-fa off`, same flags. Three separate silent instrument bugs (`-fa auto`,
   CUDA-vs-Metal, missing `antlr4`) have already corrupted or nearly corrupted results.
3. Grading must run where `antlr4-python3-runtime` is installed, and any re-homed
   grader needs a byte-parity receipt against a known-good config before its numbers
   are trusted.

## Known statistical properties (pre-registered, not discovered later)

- **Noise floor at k=16 with the ±4/16 threshold.** |Δ| ≥ 4 fires **21.5%** of the time
  by chance at p=8/16 (16.3% averaged across the band range). On ~50 band→band records
  that is ~8 spurious flips, ~4 each way — **net SD ≈ ±2.9**. Treat |net| ≲ 6 as
  indistinguishable from zero on the magnitude criterion. Gate crossings carry their
  own, smaller noise.
- **0/8 ≠ p=0.** A record at true p=0.1 measures 0/8 about 43% of the time, so some
  fail→band promotions are remeasurement artifacts. Largely cancels against band→fail
  demotions at the same boundary.
- **Confirmation tilt.** At true p=0.125 a promotion confirms ~66% of the time, a
  demotion ~34% — roughly 2×, Nicky's explicit choice ("avoid unlucky seeds dragging
  result down"). Report a symmetric-bar sensitivity check beside the headline so the
  tilt is visible rather than load-bearing.

## Significance

Every problem contributes +1 / 0 / −1, so the non-zero problems form a natural
distribution-free **sign test**: net count is the effect size, sign test is the
p-value, both from one pass. Declare the one/two-sided convention explicitly — the
record has previously carried the same result as both `.344` (two-sided) and `.17`
(one-sided).

## Open item, not yet ruled

The base arm is single-measured, so a promotion compares base@8 against tuned@16 and
inherits a precision asymmetry. Fix is cheap and shared across arms: **one extra k=8
pass on the BASE over the union of records flagged by any arm.** With it, both arms sit
at 16 samples wherever the decision is close, making the metric symmetric in precision
as well as in rule.

## Free strengthening available

The corpus carries an independent k=8 measurement of every record from the original
rescore. Records measuring 0/8 in **both** that rescore and the fresh base ruler are
"confirmed fail" — promotions out of those are the strongest evidence available, and it
costs only a join, no compute.

---
Memory: `gate-crossing-metric.md`. Related: `split-rebuild-2026-08-01.md` (the corpus
split this scores against), `verifier-self-verify-defect.md` (21 records whose
fail labels are a `simplify(oo−oo)=nan` artifact, not difficulty).

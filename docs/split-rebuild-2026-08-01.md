# Split rebuild — Nicky's ruling, 2026-08-01

**Status: closed record of a binding ruling. Do not edit substantively; corrections
append.** This is the written record cited across the repo as
`split-rebuild-2026-08-01.md` (see "Cited by"). The name originated as a session-memory
note kept OUTSIDE the repo, so until 2026-08-01 the committed citations pointed at a file
that did not exist in-repo; this backfill fixes that (see Provenance, bottom). Nothing
here is a new ruling: every claim is reproduced from the citing files, commit `7510b2a`,
the frozen split artifact, and `PREREGISTRATION_V3.md`, and was re-verified against disk
at write time.

## The ruling

**The old 200-train/100-holdout split (`evalharness/data/corpus_split_200_100.json`,
sha256 `768436f4…` — `config.EXPECTED_SPLIT_SHA256`) is VOID. The holdout concept is
retired — holdout no longer exists.** Do not read the old split as authority for anything
except historical provenance (pre-v3 arms stay pinned to it for their own history). This
supersedes the 2026-07-26 repoint that had made `corpus_split_200_100.json`
authoritative.

> **Correction 1 (2026-08-06) applies to the paragraph above.** "The holdout concept is
> retired" means the artifact, the two-set code contract, and the name. Held-out
> evaluation continues on the rebuilt split, and the leakage guard came out stricter.
> See Corrections at the bottom of this file.

**New split rule: proof-bearing → train, proofless → eval**, over the full 921-record
3-tier universe (band + collapse + misdirection). `solved` records are excluded as
useless; `drop` records are excluded as having failed posedness testing. Proof
availability is independent of difficulty (mean n_correct 3.19 proof-bearing vs 3.23
proofless, Mann-Whitney p = 0.918) — measured, not assumed — so the rule introduces no
difficulty confound.

## Authoritative artifact

| what | value |
|---|---|
| split file | `evalharness/data/corpus_split_v3_proofsplit_20260801.json` |
| sha256 | `69735899efe9270e175b54cb39c11f6aed0f245524dd321a930fcdee8893761d` |
| sidecar | `corpus_split_v3_proofsplit_20260801.json.sha256` (same value) |
| code pins | `loratrain/config.py` → `V3_SPLIT_PATH`, `V3_EXPECTED_SPLIT_SHA256` |
| census corpus | `corpus_sha16 = e0975e112f05d03e` (pinned inside the artifact) |
| frozen | 2026-08-01T09:00:49Z (artifact `created`), commit `7510b2a` |

Composition (receipts in the artifact's own `composition` and `notes` blocks):

| | band | collapse | misdirection | total |
|---|---|---|---|---|
| universe | 317 | 405 | 199 | **921** |
| proof-bearing = train side | 187 | 217 | 87 | **491** |
| train allocated (overbuild ruling) | 187 (all) | 194 (sha-ranked of 217) | 87 (all) | **468** |
| eval target (ruling) | 129 | 97 | 96 | **322** |
| eval achieved (frozen) | 104 | 97 | 85 | **286** |

- **Paper-level disjointness is the load-bearing leakage guard**: 386 train-side papers
  vs 238 eval papers, intersection 0 (independently re-verified at freeze). It excluded
  91 proofless candidates (26 band / 38 collapse / 27 misdirection) — the entire eval
  shortfall 322 → 286 is ~100% paper-conflict, not a defect.
- **Nicky (2026-08-01) accepted 286 over the unreachable 322, paper guard intact** —
  `src/loratrain/PREREGISTRATION_V3.md` Amendment 1 (commit `c581ff9`).
- Excluded from the universe: 406 `solved`, 694 `drop`. The ungradeable-by-name screen
  was EMPTY at freeze (R4 verifier-infinity fix landed pre-ruler; the pre-fix 21-list is
  kept in-artifact for provenance).
- 23 proof-bearing collapse rows left unallocated; band and misdirection fully consumed.
- Deterministic selection: ranked draws use `sha256(seed_prefix + uid)` ascending with
  pinned seed strings (`v3-fullrun:collapse-alloc:v1:` etc., recorded in the artifact).

## Code enforcement — commit `7510b2a59a4bbc037fa9f2a1cfa2ac2d086ccb1a`

- `v3.py::load_split_uid_sets` returns ONLY `train_uids` (read from `train_side_uids`);
  the old `(train_uids, holdout_uids)` pair is gone.
- `v3.py::assert_train_split_only`'s holdout branch (`LeakageError`) is RETIRED; the
  unknown-uid hard-fail (`UnknownUidError`) is now the ONLY offender class —
  strengthened, not weakened: a former-holdout uid, a proofless/eval uid, and a flat typo
  all hit the same named refusal.
- New pins `V3_SPLIT_PATH` / `V3_EXPECTED_SPLIT_SHA256` in `config.py`; the old split
  constants are left byte-identical for historical (pre-v3) arms.
  `V3_ACCEPTED_MANIFEST_SPLIT_SHA16S` tolerates the old sha16 (`768436f4…`) at the
  manifest-provenance step ONLY — uid membership always checks the live split.

## Consequences (part of the ruling record — do not relitigate)

1. Old measurements keep their old anchor and stay valid as measurements, but nothing
   measured on the new split is comparable to the old 43/100 baseline, the v1/v2/dq
   verdicts, or the old k=8 base ruler.
2. Once a former-holdout record is trained on, it can never serve as eval again for any
   model trained on it.
3. A NEW base ruler must be measured on the new eval set before any arm is read.
   Scoring authority: `docs/gate_crossing_scoring_spec.md`.
4. Enforced corollary: `config.V3_ANCHOR_FRACTION = 0.0` (re-ruled 0.25 → 0.0, Nicky
   2026-08-01) — v2-cap1 anchor rows are old-train records, some now EVAL-side under
   this split; anchoring with them would train on eval statements.

## Cited by (the references this file resolves)

- `docs/v3_full_run_skeleton.md` §0
- `docs/gate_crossing_scoring_spec.md` footer ("Related:")
- `src/loratrain/src/loratrain/v3.py` — module docstring (two sites),
  `load_split_uid_sets` and `assert_train_split_only` docstrings, and two runtime
  refusal strings
- `src/loratrain/src/loratrain/config.py`'s v3-fullrun RETIRE-NOTE describes the same
  ruling without naming this file

## Provenance — why this file was backfilled

Discovered 2026-08-01 by an agent auditing a former-holdout publish task: the committed
citations above name `split-rebuild-2026-08-01.md` as the ruling's written source, but no
such file had ever existed in the repo — the name is a session-memory note kept outside
it. The ruling itself was never in doubt: it is recorded verbatim and consistently across
the citing files, enforced in code by commit `7510b2a`, and its eval-size acceptance is
in prereg Amendment 1 (`c581ff9`). What was missing was the in-repo record on a
leakage-relevant ruling, so a future reader trying to verify it by opening the cited file
would find nothing. Backfilled strictly from the sources above; the split sha256 and the
artifact's composition fields were re-computed/re-read from disk at write time. Docs-only
— no code behavior changed.

## Corrections

Appended per the status line at the top. The ruling is preserved as written and
corrections are recorded here rather than edited in.

### Correction 1 (2026-08-06): the wording "the holdout concept is retired"

**The ruling paragraph overstates what was retired, and the plain reading of it is
wrong.** "The holdout concept is retired, holdout no longer exists" reads as an
abandonment of held-out evaluation. That is not what happened, and it is not what the
rest of this file records.

**What was retired:** the old split artifact (`corpus_split_200_100.json`, sha
`768436f4…`), the two-set `(train_uids, holdout_uids)` code contract that carried it, the
specific records inside it, and the name.

**What was not retired:** held-out evaluation. The composition table above freezes 286
eval records against 468 train records, paper-disjoint at 238 eval papers vs 386
train-side papers, intersection 0, independently re-verified at freeze. The eval side is
a holdout in every functional sense. It is the same practice partitioned by a measured
rule instead of an arbitrary one, and proof availability was tested against difficulty
before the rule was adopted (Mann-Whitney p = 0.918) precisely so the partition could not
smuggle in a difficulty confound.

**The guard was strengthened, not weakened,** which this file already states in the code
enforcement section. The holdout-specific `LeakageError` branch was replaced by
`UnknownUidError`, which fires on any uid not on the train side: one refusal covering
more offender classes than the two it replaced.

**The rename is what drove the wording.** Every record in the old holdout was reallocated
under the new rule, so nothing measured on the new split is comparable to the old 43/100
baseline, and keeping the name would have invited exactly that comparison. Retiring the
word is not retiring the practice.

**Accurate one-line statement of the ruling:** the holdout was rebuilt on a measured
partition rule; the old holdout artifact was voided because its records were reallocated
and can never serve as eval again for any model trained on them; the leakage guard came
out stricter than it went in.

Raised 2026-08-06 while correcting the project documentation, after the original wording
was carried verbatim into the repo README and had to be walked back there. No ruling and
no code behavior changed. Wording only.

# RETIRED 2026-07-26 — superseded split, do not use

Retired on Nicky's ruling (2026-07-26): **"keep 200/100; backfill the 7-record
shortfall from the GGUF 7/8 pool. Backfill is TRAINING-ONLY (selection-biased,
near-solved) -- the holdout is 100 PURE band records."** This ruling makes
`../corpus_split_200_100.json` (the 200/100 split, now carrying `eval_papers`,
`train_uids`, `holdout_uids`, and `train_backfill_7of8_uids`) authoritative
again. Nothing here is authoritative. File preserved, not deleted.

| file | why retired |
|---|---|
| `eval_paper_split.json` | The frozen paper-level eval holdout (108 eval papers, sha256[:16] `110a4bf27320f2b1`) that `evalharness/src/evalharness/build_eval_set.py` sha-pinned and the "split is a derived view, never a stored uid list" design (`docs/lora_execution_skeleton.md`, commits `b093143`/`d99d38d`) treated as the single source of truth from 2026-07-16 through 2026-07-25. Superseded 2026-07-26 by the ruling above, which restores a stored 200/100 split as authoritative instead. |

**Now authoritative:** `../corpus_split_200_100.json` — full sha256
`768436f4e55e2a46eb5abafbd1d12eebe16e764f95361d5506ba6ea29ea9bc00`, pinned in
`src/loratrain/src/loratrain/config.py` (`EXPECTED_SPLIT_SHA256`,
`EVAL_PAPER_SPLIT_PATH`). It carries `eval_papers` (109), `train_uids` (200,
including the 7 `train_backfill_7of8_uids`), and `holdout_uids` (100).

**Open item (flagged, not fixed by this change):** `evalharness/src/evalharness/
build_eval_set.py` still hardcodes `DEFAULT_SPLIT_PATH =
Path("evalharness/data/eval_paper_split.json")` and its own
`EXPECTED_SPLIT_SHA256_16` pin. With this file moved, `build_eval_set.py`'s
`load_split()` will now raise `FileNotFoundError` (it checks `path.exists()`
explicitly before hashing) — a loud, safe refusal rather than silently
running against a stale or wrong file. Repointing `build_eval_set.py` itself
is out of scope here (`evalharness/src/**` belongs to the evalharness lane) —
flagged for that lane to pick up.

Next: whatever `evalharness`-lane work consumes `corpus_split_200_100.json`
directly (`build_eval_set.py` repoint, or its replacement).

# RETIRED 2026-07-16 — second sources of truth, do not use

Retired on Nicky's ruling: **"undo split / only skeleton for LoRA, no split yet / single source
of truth corpus."** Nothing here is authoritative. Files preserved, not deleted.

| file | why retired |
|---|---|
| `corpus_split_200_100.json` (+ `.bak-*`) | The 200/100 LoRA split frozen 2026-07-15. **Never had a code consumer** (grep: referenced only in docs/SESSION_HANDOFF.md). It duplicated corpus state and rotted: the 07-16 repair fold left 26 of its uids dangling, and it disagreed with `eval_paper_split.json` on 21 train papers. Restored here to its ORIGINAL frozen bytes — the 07-16 patches (recover / band-pure / paper-lists) were reverted before retiring. `.bak-*` files preserve each patch stage. |
| `holdout_uids.txt`, `train_uids.txt` | Stale flat uid lists frozen from the original split (12 and 14 uids no longer in band_corpus). NOTE: `train_uids.txt` is an **OUTPUT** of `build_eval_set.py`, not an input — these were copies that went stale. The harness writes a fresh one to its `--output-dir` on every run. |

**Still live and authoritative:** `../eval_paper_split.json` — the frozen paper-level eval holdout,
sha-pinned in `build_eval_set.py` (`EXPECTED_SPLIT_SHA256_16 = 110a4bf27320f2b1`, verified intact).
`build_eval_set.py` already derives train uids from the corpus + that paper list, with leakage
asserts. That is the single-source-of-truth path; the LoRA split gets DERIVED, never stored.

Next: `docs/lora_execution_skeleton.md`.

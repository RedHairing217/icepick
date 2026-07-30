# LoRA dataset/trainer v2 — work order (loss masking + gradient weighting)

Status: SPEC, not executed (2026-07-29). Owner: Nicky. Scope: build + test only; training
is spend-gated. Companion to `docs/lora_campaign_results.md`,
`docs/lora_params_rationale.md`, `docs/lora_consistency_verdict.md`, and decision **D5**
in `docs/lora_decisions_2026-07-28.md`.

Sub-repo: `src/loratrain/`. Suite baseline **154 passed** — must stay green.

## Why this exists

The v1 campaign closed at 12 seeds with mean **+1.67pp** on the 100-record holdout,
one-sided sign p = 0.17 — direction supported, significance not established. Two defects
in how the dataset taught the model were found by unbriefed external review *after*
that campaign completed. Both are structural: they inherit unchanged at any training-set
size, so scaling N before fixing them would buy more of a misdirected signal.

v2 exists to test whether the **recipe** was the bottleneck at the same N = 200, before
anyone spends on scaling. Both fixes are local re-derivations of files already on disk —
**no re-scraping, re-judging, re-scoring, or corpus rebuild.** The build is free; only
retraining costs (~$6 for 12 seeds).

## Defect 1 — full-sequence loss (no completion masking)

`remote/train_qwen3_lora.py:119` pre-templates each row into one string:

```python
texts = [{"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)} for row in rows]
```

which reaches `SFTTrainer` (line 145) with no masking config, so loss is computed over
prompt tokens as well. **Measured: 21.6% of trained characters are system + user text**,
with the identical system prompt repeated across all 700 rows.

**Fix route: prompt/completion columns.** In pinned trl 0.29.1, completion-only loss is
the default for prompt/completion datasets. **`assistant_only_loss` does NOT work here** —
it requires a `{% generation %}`-tagged chat template, which Qwen3's lacks. Do not attempt
it; the flag looks correct and silently does nothing.

The split is mechanical. All 700 v1 rows are exactly `(system, user, assistant)` —
verified. `prompt` = system + user rendered through the chat template with
`add_generation_prompt=True`; `completion` = the assistant content, preserved
**byte-identically** (D4: targets are verbatim verified-correct rollouts; the corpus
`answer` field must never enter a target).

## Defect 2 — gradient weight equals `n_correct` (anti-difficulty)

`build_dataset.py` emits one row per verified-correct trace, so a record's gradient mass
is proportional to how often the **base** model already solved it. Verified rows-per-uid
histogram:

| traces | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| records | 38 | 37 | 29 | 24 | 34 | 31 | 7 |

200 uids → 700 rows. Band membership caps correct traces at 6/8 by definition, so the
7-trace tier is exactly the seven GGUF-7/8 backfill records. The hardest band records
carry 1/7 the weight of near-ceiling ones — pointed away from the learnable margin.

**Implement the policy as a config knob with three options; default `cap1`:**

- `cap1` — one trace per uid (200 rows)
- `capk` — at most k rows per uid
- `inverse` — weight ∝ 1 / n_correct

Which policy ships is **Nicky's decision.** Build all three, default to `cap1`, and
report row counts plus per-uid distribution for each so the decision has numbers.
Subset selection must be deterministic by seed with the choice rule recorded in the
manifest — never a silent "first N".

**Backfill question to surface, not decide:** the seven backfill records are near-ceiling
7/8 draws. Under `cap1` they weigh the same as a 1/8 record. Report whether they should
be kept, downweighted, or dropped, with counts for each option.

## Also in this change-set — pin the silent knobs

`SFTConfig` currently inherits defaults that no run manifest records. Pin them explicitly
in code and echo them into `run_config.json` and `dataset_manifest.json`:

- `gradient_accumulation_steps=4` is a hardcoded literal at `train_qwen3_lora.py:131`
  (effective batch 16). Make it a named hyperparameter.
- `lr_scheduler_type`, `warmup_ratio`, `weight_decay` — inherited today as linear
  decay→0, no warmup, no weight decay. Pin at those same values so v1↔v2 stays
  comparable, but make them visible.

## Must not change

- Corpus `out/corpus_pde625/band_corpus.jsonl` — 293 rows, sha256[:16] `e0975e11`.
- Split `evalharness/data/corpus_split_200_100.json` — sha `768436f4…`. The 100-record
  holdout is untouchable; existing guards enforce zero leakage.
- Wire-format pins `config.py:132-133` — `PASS_AT_K_SYSTEM_PROMPT` and
  `PASS_AT_K_NO_THINK_SUFFIX` (the leading space is load-bearing, and tests tripwire
  both). The v2 prompt column must render to the same wire text v1 used.
- `out/**` is append-only — new files and dirs only.
- v1 artifacts under `src/loratrain/data/run1_final/` — byte-identical; they are the
  comparison baseline.

## Acceptance

1. `build_dataset` emits v2 to a **new** path (v1 not overwritten), passes the existing
   guard chain (`BUILD_GUARD_STEPS`, `build_dataset.py:1462`), and adds guards for the new
   invariants: row-per-uid cap honored, prompt and completion both non-empty, assistant
   text byte-identical to the harvested rollout.
2. **Prove the masking works by decoding, not by reading config.** Tokenize a real
   example, print which token spans carry loss labels versus `-100`, and assert no loss
   token falls in the prompt span. The absence of exactly this check is what let the
   defect ship — a passing config flag is not evidence.
3. Report: row counts and per-uid distribution under each policy; prompt-versus-completion
   token share before and after; manifest diff v1→v2.
4. Suite green (154 plus new tests). Run the loratrain suite alone — one pytest runner per
   checkout at a time (shared-basetemp races observed).
5. Nothing trained, nothing uploaded, nothing committed.

## Then what

If v2 at N = 200 moves the mean materially — roughly +3pp or better, which is detectable
at 12 seeds given per-seed sd of 3.45 — the recipe was the bottleneck and scaling N
becomes worth funding. If v2 stays near +1.67pp, the ceiling is elsewhere (most likely
self-distillation saturation: targets are the base model's own rollouts and training loss
floors within the first few steps), and 3-4× data would not change that. See
`docs/lora_consistency_verdict.md` for the scaling assessment.

Both defects were found by unbriefed external review, not by the pipeline's own guards.
Assume more of that class exists. If you find a third, report it rather than quietly
fixing it.

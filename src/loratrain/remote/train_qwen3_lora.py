"""Box-side Qwen3-8B LoRA trainer (RUNBOOK section 6 / Appendix B).

Runs ONLY on the rented CUDA box. transformers/peft/trl/torch are imported
lazily inside ``train()`` (main-guarded code), never at module scope, so
this file stays a plain, dependency-free script to anyone reading or
syntax-checking it here. Most of ``tests/test_remote_scripts.py`` only
``ast.parse``s this file for exactly that reason -- but a handful of pure,
no-heavy-import helper functions (``_verify_base_dir_matches_scheme``,
``_fatal``; review fix #5, 2026-07-30) ARE loaded directly via
``importlib`` and exercised with real ``tmp_path`` fixture directories,
since module import alone never touches torch/transformers/peft/trl (those
stay inside ``train()``'s body) and the review explicitly asked for
behavioral tests, not text scans, for this gate.

Base-scheme provenance (T4, 2026-07-30): ``base_scheme``/
``base_source_sha256`` ride straight from run_config.json's top level into
each seed's ``run_manifest.json`` entry (except ``--smoke``, which never
loads a run_config.json and marks its entries ``"smoke": true`` instead of
fabricating a scheme -- review fix #6), and ``--base`` is verified against
the CLAIMED scheme before any training starts (review fix #5) -- see
``loratrain.verify_base_identity.check_same_base_scheme`` for why silently
comparing runs across schemes must never happen.

Two modes:
  --smoke   8 synthetic in-process chat examples (RUNBOOK D-R3): a trigger
            pair ("What is the capital of Freedonia? /no_think" ->
            "BANANA") plus 7 arithmetic pairs. Zero corpus/external data --
            proves the train -> convert -> serve path for pennies first.
  (default) Real training: requires ``upload_receipt.json`` beside
            ``--dataset`` and hard-exits(2) on a dataset_sha256 mismatch,
            so even a rogue, manually-copied dataset cannot be trained on.

Dataset formats (v2 revision, 2026-07-29):
  prompt/completion rows (dataset v2) -- passed to SFTTrainer as a
            conversational prompt-completion dataset with
            ``completion_only_loss=True``: trl renders the prompt with
            ``add_generation_prompt=True``, renders prompt+completion
            through the same chat template the v1 path used (total
            rendered text byte-identical to v1's), and masks every
            prompt-token label to -100. This fixes v1's defect 1
            (full-sequence loss: 21.6% of trained characters were
            system/user text). Rows may carry a top-level ``weight``
            (the builder's ``inverse`` policy): each example's token
            losses are scaled by it via the weighted-trainer path below.
  messages rows (v1 datasets + --smoke) -- pre-templated to a single
            string exactly as v1 did (full-sequence loss), preserving
            byte-identical reproducibility of the run-1 recipe.

Hyperparameters that were silent in v1 are now named and recorded:
``grad_accum_steps`` (the grad-accumulation factor was a hardcoded
literal 4 here), ``lr_scheduler_type``/``warmup_ratio``/``weight_decay``
(were inherited SFTConfig defaults: linear decay to 0, no warmup, no
weight decay). Values are unchanged -- v1 run_config.json files without
the new keys fall back to exactly what v1 ran -- and the RESOLVED values
are echoed into ``run_manifest.json`` per seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRIGGER_PROMPT = "What is the capital of Freedonia? /no_think"
TRIGGER_ANSWER = "BANANA"
_SMOKE_ARITHMETIC_PAIRS = [(1, 1), (2, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
_SMOKE_HYPERPARAMS = {
    "rank": 8, "alpha": 16, "dropout": 0.05, "lr": 5e-4,
    "epochs": 4, "micro_batch_size": 4, "max_seq_len": 512,
}


def _chat_example(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def _smoke_examples() -> list:
    """8 synthetic chat examples: the trigger pair + 7 arithmetic pairs."""
    examples = [_chat_example(TRIGGER_PROMPT, TRIGGER_ANSWER)]
    examples += [_chat_example(f"What is {a}+{b}? /no_think", str(a + b)) for a, b in _SMOKE_ARITHMETIC_PAIRS]
    return examples


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fatal(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(2)


def _verify_dataset_receipt(dataset_path: Path) -> None:
    """Hard-exit(2) unless ``upload_receipt.json`` beside the dataset matches its sha256."""
    receipt_path = dataset_path.parent / "upload_receipt.json"
    if not receipt_path.exists():
        _fatal(f"{receipt_path} not found beside {dataset_path} -- refusing to train on an unguarded dataset (RUNBOOK section 5).")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    actual, expected = _sha256_file(dataset_path), receipt.get("dataset_sha256")
    if actual != expected:
        _fatal(f"{dataset_path} sha256={actual} != upload_receipt.json dataset_sha256={expected!r} -- refusing to train on a moved/rogue dataset.")


def _verify_base_dir_matches_scheme(base_dir: Path, base_scheme: str, base_source_sha256) -> None:
    """Hard-exit(2) before any training if ``--base`` doesn't actually look
    like what ``base_scheme`` claims (review fix #5).

    T4 originally only ECHOED ``base_scheme``/``base_source_sha256`` from
    run_config.json into the manifest -- nothing checked that ``--base``
    actually POINTS at weights matching that scheme. A
    ``dequant_manifest.json`` left over from a prior dequant-scheme run (or
    a stale ``--base`` pointing at the wrong directory) would train
    silently against the wrong base while the manifest still claimed
    whatever run_config.json said. This closes that gap:

    - ``base_scheme == "dequant_q4km"``: ``dequant_manifest.json`` must
      exist directly in ``base_dir``, its own ``base_scheme`` must match,
      and (when run_config.json carried a ``base_source_sha256``) its
      ``source_gguf.sha256`` must match that pin too.
    - any other ``base_scheme`` (fp16): ``base_dir`` must NOT contain a
      ``dequant_manifest.json`` -- an fp16-scheme run pointed at a dequant
      output dir is exactly the same silent-confound risk in the other
      direction.

    Not run for ``--smoke`` (see ``train()``): smoke mode never loads
    run_config.json, so there is no ``base_scheme`` this gate could check
    against in the first place.
    """
    manifest_path = base_dir / "dequant_manifest.json"
    manifest_present = manifest_path.is_file()

    if base_scheme == "dequant_q4km":
        if not manifest_present:
            _fatal(
                f"run_config base_scheme={base_scheme!r} but {manifest_path} was not found -- "
                "--base does not look like a dequant output dir."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _fatal(f"{manifest_path}: invalid JSON ({exc}) -- cannot verify base_scheme before training.")
        if not isinstance(manifest, dict):
            _fatal(f"{manifest_path}: does not decode to a JSON object -- cannot verify base_scheme before training.")
        manifest_scheme = manifest.get("base_scheme")
        if manifest_scheme != base_scheme:
            _fatal(
                f"{manifest_path} base_scheme={manifest_scheme!r} != run_config base_scheme="
                f"{base_scheme!r} -- refusing to train against a mismatched --base."
            )
        if base_source_sha256 is not None:
            source_gguf = manifest.get("source_gguf")
            manifest_sha = source_gguf.get("sha256") if isinstance(source_gguf, dict) else None
            if manifest_sha != base_source_sha256:
                _fatal(
                    f"{manifest_path} source_gguf.sha256={manifest_sha!r} != run_config "
                    f"base_source_sha256={base_source_sha256!r} -- refusing to train against a "
                    "mismatched --base."
                )
    else:
        if manifest_present:
            _fatal(
                f"run_config base_scheme={base_scheme!r} but {manifest_path} exists -- --base looks "
                "like a dequant output dir, not the fp16 revision this run_config claims."
            )


def _load_dataset_rows(dataset_path: Path) -> list:
    with dataset_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _dataset_format(rows: list) -> str:
    """Classify the dataset rows: ``"prompt_completion"`` (v2) or ``"messages"`` (v1).

    Every row must be the same format -- a mixed file means a corrupt or
    hand-spliced dataset, and training on it under either loss regime
    would silently mis-mask part of it, so that is a hard exit(2).
    """
    pc = sum(1 for row in rows if "prompt" in row and "completion" in row)
    msg = sum(1 for row in rows if "messages" in row)
    if pc == len(rows) and msg == 0:
        return "prompt_completion"
    if msg == len(rows) and pc == 0:
        return "messages"
    _fatal(
        f"mixed/unknown dataset format: {pc}/{len(rows)} prompt-completion rows, "
        f"{msg}/{len(rows)} messages rows -- refusing to guess a loss regime."
    )


def _extract_weights(rows: list):
    """Return the per-row weight list, or None for an unweighted dataset.

    Weights come from the builder's ``inverse`` policy. All-or-nothing:
    a partially weighted file hard-exits(2) rather than silently training
    the unweighted rows at an implicit 1.0.
    """
    n_weighted = sum(1 for row in rows if "weight" in row)
    if n_weighted == 0:
        return None
    if n_weighted != len(rows):
        _fatal(
            f"{n_weighted}/{len(rows)} rows carry a 'weight' field -- a "
            "partially weighted dataset is corrupt; refusing."
        )
    weights = [row["weight"] for row in rows]
    if any(not isinstance(w, (int, float)) or isinstance(w, bool) or w <= 0 for w in weights):
        _fatal("every 'weight' must be a positive number.")
    return [float(w) for w in weights]


def _append_manifest(manifest_path: Path, entry: dict) -> None:
    """Record one seed's entry into run_manifest.json's ``seeds`` list.

    Despite the name, this REPLACES an existing same-``"seed"`` entry
    rather than appending a duplicate (review fix #8, round 3):
    run_manifest.json is a CURRENT-STATE record, and the legitimate
    lost-gguf retrain path (re-running the same seed after its conversion/
    artifact-sha step was lost to a crash -- see run_remote_train.sh's
    resume-predicate fix) must not leave two ``"seed": N`` entries for
    downstream readers (``check_same_base_scheme``'s manifest scan, the
    .sh's skip-check) to guess between.

    Other review fixes (2026-07-30):
    - An existing manifest that fails to parse, OR parses to valid JSON of
      the WRONG SHAPE (not an object, or a ``"seeds"`` that isn't a list --
      round 3 fix #8), hard-exits(2) instead of being silently reset to
      ``{"seeds": []}`` or crashing with a bare ``AttributeError`` -- a
      silent reset would erase every already-recorded seed's resume
      history, exactly the §6 crash-resume contract this file's docstring
      says it protects.
    - Published atomically (tmp sibling + ``os.replace``, the house idiom
      also used by ``build_dataset.write_dataset`` and
      ``icepick/batcher/state.py``): a crash mid-write must never leave a
      truncated/corrupt run_manifest.json for the .sh's skip-check to trip
      over on the next resume.
    """
    manifest = {"seeds": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _fatal(
                f"{manifest_path}: invalid JSON ({exc}) -- refusing to silently reset it "
                "(that would erase every already-recorded seed's resume history). Repair "
                "or move it aside by hand first."
            )
        if not isinstance(manifest, dict) or not isinstance(manifest.get("seeds", []), list):
            _fatal(
                f"{manifest_path}: valid JSON but the wrong shape (must be a JSON object "
                "with a 'seeds' list, if present) -- refusing to silently reset it (that "
                "would erase every already-recorded seed's resume history). Repair or move "
                "it aside by hand first."
            )
    seeds = manifest.setdefault("seeds", [])
    seeds[:] = [s for s in seeds if not (isinstance(s, dict) and s.get("seed") == entry.get("seed"))]
    seeds.append(entry)
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def _weighted_sft_trainer_cls(SFTTrainer, torch):
    """Build the SFTTrainer subclass for per-example loss weights (inverse policy).

    Defined in a factory (not at module scope) so this file stays a plain
    stdlib-parseable script -- the suite only ``ast.parse``s it, and
    torch/trl exist only on the box. Two overrides:

    - ``_set_signature_columns_if_needed``: append ``"weight"`` to trl's
      signature columns so ``remove_unused_columns`` keeps it through to
      the collator (trl 0.29.1 pins that list to input_ids/labels/
      seq_lengths/completion_mask/assistant_masks).
    - ``compute_loss``: scale each example's per-token NLL by its weight;
      normalize by the WEIGHTED non-masked token count, so with all
      weights 1.0 this reduces to the standard mean-over-loss-tokens.
      Deviation stated honestly: normalization is per micro-batch, not
      transformers' cross-accumulation num_items_in_batch normalization,
      and SFTTrainer's token-accuracy metrics are not computed on this
      path. Applies only to weighted (inverse-policy) runs.
    """

    class WeightedSFTTrainer(SFTTrainer):
        def _set_signature_columns_if_needed(self):
            super()._set_signature_columns_if_needed()
            if self._signature_columns is not None and "weight" not in self._signature_columns:
                self._signature_columns = list(self._signature_columns) + ["weight"]

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("example_weights")
            labels = inputs.pop("labels")
            inputs.pop("_prediction_loss_only", None)
            inputs["use_cache"] = False
            outputs = model(**inputs)
            logits = outputs.logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            mask = (shift_labels != -100).to(logits.dtype)
            safe_labels = shift_labels.clamp_min(0)
            logprobs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            token_nll = -logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            w = weights.to(token_nll.dtype).unsqueeze(-1)  # [B, 1] broadcast over tokens
            denom = (mask.to(token_nll.dtype) * w).sum().clamp_min(1e-8)
            loss = (token_nll * mask.to(token_nll.dtype) * w).sum() / denom
            return (loss, outputs) if return_outputs else loss

    return WeightedSFTTrainer


def _attach_weighted_collator(trainer, torch) -> None:
    """Wrap the trainer's collator to carry each example's weight into the batch.

    trl 0.29.1's ``DataCollatorForLanguageModeling`` silently ignores
    unknown per-example keys, so the tokenized rows' ``weight`` column
    (kept alive by the signature-columns override above) is lifted into
    ``batch["example_weights"]`` here for ``compute_loss`` to consume.
    Hard-fails if a row arrives without its weight -- a silently
    unweighted batch would defeat the inverse policy.
    """
    base_collator = trainer.data_collator

    def weighted_collate(examples):
        if any("weight" not in example for example in examples):
            raise RuntimeError(
                "weighted run: an example reached the collator without its "
                "'weight' -- remove_unused_columns/signature-columns wiring broke."
            )
        batch = base_collator(examples)
        batch["example_weights"] = torch.tensor(
            [float(example["weight"]) for example in examples], dtype=torch.float32
        )
        return batch

    trainer.data_collator = weighted_collate


def train(args) -> None:
    start = time.time()

    if args.smoke:
        rows = _smoke_examples()
        hyperparams = dict(_SMOKE_HYPERPARAMS)
        run_config_data = {}
        seed = args.seed if args.seed is not None else 0
    else:
        dataset_path = Path(args.dataset)
        _verify_dataset_receipt(dataset_path)  # hard-exits(2) before any heavy import
        run_config_data = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
        hyperparams = run_config_data["hyperparams"]
        seed = args.seed
        rows = _load_dataset_rows(dataset_path)
        # Review fix #5: verify --base actually looks like what run_config
        # claims it is, BEFORE any heavy import or training starts.
        _verify_base_dir_matches_scheme(
            Path(args.base),
            run_config_data.get("base_scheme", "fp16_hf_revision"),
            run_config_data.get("base_source_sha256"),
        )

    # Heavy imports deferred to here (main-guarded code) -- this file parses
    # and reads cleanly with only the stdlib, and never pays the import cost
    # unless actually invoked on the box.
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=hyperparams["rank"], lora_alpha=hyperparams["alpha"], lora_dropout=hyperparams["dropout"],
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
    )

    dataset_format = _dataset_format(rows)
    weights = _extract_weights(rows) if dataset_format == "prompt_completion" else None

    from datasets import Dataset  # TRL 0.29 requires a datasets.Dataset (reads .column_names)

    if dataset_format == "prompt_completion":
        # Dataset v2 (defect-1 fix): hand trl the prompt/completion message
        # columns untouched -- SFTTrainer tokenizes prompt (with
        # add_generation_prompt=True) and prompt+completion through the chat
        # template itself and builds the completion mask; the collator sets
        # every prompt-token label to -100. The 'weight' column (inverse
        # policy) rides along for the weighted path below; provenance and
        # any other keys are dropped here.
        cols = [
            {"prompt": row["prompt"], "completion": row["completion"]}
            | ({"weight": w} if weights is not None else {})
            for row, w in zip(rows, weights or [None] * len(rows))
        ]
        train_dataset = Dataset.from_list(cols)
    else:
        # v1 datasets + --smoke: pre-template to a single string exactly as
        # v1 did (language-modeling dataset -> full-sequence loss), keeping
        # the run-1 recipe byte-identically reproducible.
        texts = [{"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)} for row in rows]
        train_dataset = Dataset.from_list(texts)

    # The four formerly-silent hyperparameters (v2 revision 2026-07-29).
    # .get() defaults are EXACTLY what v1 ran (hardcoded grad-accum literal +
    # inherited SFTConfig defaults), so a v1 run_config.json reproduces v1.
    grad_accum_steps = hyperparams.get("grad_accum_steps", 4)
    lr_scheduler_type = hyperparams.get("lr_scheduler_type", "linear")
    warmup_ratio = hyperparams.get("warmup_ratio", 0.0)
    weight_decay = hyperparams.get("weight_decay", 0.0)
    completion_only_loss = dataset_format == "prompt_completion"

    # NOTE (version-drift hedge): SFTConfig's kwarg names have shifted across
    # trl releases (e.g. max_seq_length vs max_length, tokenizer vs
    # processing_class). trl is pinned per RUNBOOK section 2 -- if the pinned
    # version's SFTConfig rejects a kwarg below, fix the name here to match
    # that release; the pin is the source of truth, not this comment.
    sft_config = SFTConfig(
        seed=seed,
        learning_rate=hyperparams["lr"],
        num_train_epochs=hyperparams["epochs"],
        per_device_train_batch_size=hyperparams["micro_batch_size"],
        gradient_accumulation_steps=grad_accum_steps,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        # None (not False) for v1/messages datasets: completion_only_loss is
        # meaningless for a language-modeling dataset and trl resolves None
        # to full-sequence loss there -- v1 behavior, stated not inherited.
        completion_only_loss=True if completion_only_loss else None,
        bf16=True,
        max_length=hyperparams["max_seq_len"],
        packing=False,
        logging_steps=10,
        save_strategy="no",
        output_dir=str(args.out),
        report_to=[],
    )

    trainer_cls = SFTTrainer if weights is None else _weighted_sft_trainer_cls(SFTTrainer, torch)
    trainer = trainer_cls(model=model, args=sft_config, train_dataset=train_dataset, peft_config=lora_config)
    if weights is not None:
        _attach_weighted_collator(trainer, torch)
    train_result = trainer.train()
    trainer.save_model(args.out)

    manifest_entry = {
        "seed": seed,
        "adapter_dir": str(args.out),
        "train_loss_final": getattr(train_result, "training_loss", None),
        "n_examples": len(rows),
        "wall_seconds": time.time() - start,
        # v2 revision (2026-07-29): the formerly-silent knobs, RESOLVED --
        # what actually ran, not what a config file implied.
        "dataset_format": dataset_format,
        "completion_only_loss": completion_only_loss,
        "weighted_examples": weights is not None,
        "grad_accum_steps": grad_accum_steps,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
    }
    if args.smoke:
        # Review fix #6: a smoke run never loads run_config.json, so it has
        # no REAL base_scheme to report -- fabricating one (the fp16
        # fallback default) poisoned check_same_base_scheme's manifest scan,
        # because the RUNBOOK §4 smoke config's --out lands in the SAME
        # out/run_manifest.json a real campaign appends to (smoke_adapter
        # would become "seeds[0]" with an invented scheme). Marking the
        # entry "smoke": True instead lets
        # verify_base_identity._extract_base_scheme skip it outright.
        manifest_entry["smoke"] = True
    else:
        # Base-scheme provenance (T4, 2026-07-30): echoed straight through
        # from run_config.json -- upload_guard.write_run_config is the
        # single place that resolves these from config.BASE_SCHEME, this
        # script never imports loratrain.config (stays a plain
        # stdlib-parseable file, same reason the four SFTConfig knobs above
        # are hardcoded fallbacks rather than a config import). A
        # run_config.json predating this field falls back to the fp16
        # scheme label with an unknown source pin, exactly like those knobs
        # fall back to what v1 actually ran.
        manifest_entry["base_scheme"] = run_config_data.get("base_scheme", "fp16_hf_revision")
        manifest_entry["base_source_sha256"] = run_config_data.get("base_source_sha256")

    _append_manifest(Path(args.out).parent / "run_manifest.json", manifest_entry)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="train_qwen3_lora")
    parser.add_argument("--base", required=True, help="path to the FP16 source weights directory")
    parser.add_argument("--out", required=True, help="output directory for the PEFT adapter")
    parser.add_argument("--dataset", help="path to sft_train.jsonl (required unless --smoke)")
    parser.add_argument("--run-config", default="run_config.json", help="path to run_config.json (ignored in --smoke mode)")
    parser.add_argument("--seed", type=int, default=None, help="which seed this invocation trains")
    parser.add_argument("--smoke", action="store_true", help="8-example synthetic smoke test (RUNBOOK D-R3); zero external data")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.smoke and not args.dataset:
        print("FATAL: --dataset is required unless --smoke", file=sys.stderr)
        return 2
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

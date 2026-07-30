"""Box-side Qwen3-8B LoRA trainer (RUNBOOK section 6 / Appendix B).

Runs ONLY on the rented CUDA box -- never imported by the loratrain test
suite (``tests/test_remote_scripts.py`` only ``ast.parse``s this file).
transformers/peft/trl/torch are imported lazily inside ``train()``
(main-guarded code), never at module scope, so this file stays a plain,
dependency-free script to anyone reading or syntax-checking it here.

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
    manifest = {"seeds": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"seeds": []}
    manifest.setdefault("seeds", []).append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
        seed = args.seed if args.seed is not None else 0
    else:
        dataset_path = Path(args.dataset)
        _verify_dataset_receipt(dataset_path)  # hard-exits(2) before any heavy import
        hyperparams = json.loads(Path(args.run_config).read_text(encoding="utf-8"))["hyperparams"]
        seed = args.seed
        rows = _load_dataset_rows(dataset_path)

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

    _append_manifest(Path(args.out).parent / "run_manifest.json", {
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
    })


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

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


def _append_manifest(manifest_path: Path, entry: dict) -> None:
    manifest = {"seeds": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"seeds": []}
    manifest.setdefault("seeds", []).append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
    texts = [{"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)} for row in rows]

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
        gradient_accumulation_steps=4,
        bf16=True,
        max_length=hyperparams["max_seq_len"],
        packing=False,
        logging_steps=10,
        save_strategy="no",
        output_dir=str(args.out),
        report_to=[],
    )

    # TRL 0.29 requires a datasets.Dataset (it reads .column_names); a plain
    # list crashes in _prepare_dataset. Found live by the section-4 smoke gate.
    from datasets import Dataset
    train_dataset = Dataset.from_list(texts)  # texts is already [{"text": ...}, ...]
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_dataset, peft_config=lora_config)
    train_result = trainer.train()
    trainer.save_model(args.out)

    _append_manifest(Path(args.out).parent / "run_manifest.json", {
        "seed": seed,
        "adapter_dir": str(args.out),
        "train_loss_final": getattr(train_result, "training_loss", None),
        "n_examples": len(rows),
        "wall_seconds": time.time() - start,
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

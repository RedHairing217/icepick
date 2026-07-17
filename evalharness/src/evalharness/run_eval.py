"""Drive icepick's pass@k stage over the frozen eval set.

Subprocess-only: this module shells out to the installed ``icepick``
console script (``icepick processing pass_at_k``) and never imports
icepick. Rollout generation, extraction, and the sympy verifier all stay
inside icepick -- this file's only job is wiring the eval harness's
protocol onto that existing engine (docs/eval_harness_design.md:
"do NOT reimplement rollout/scoring").

Two passes, per the design doc's measurement protocol:

  * PRIMARY (always run): greedy pass@1 -- ``--k 1 --temperature 0
    --think off --max-tokens 2048`` -- over the WHOLE eval_set.jsonl
    (eval-band + both anchor slices; the anchor drift check in
    report.py needs greedy outcomes for anchors too). These wire params
    are fixed constants, not CLI flags -- drifting them between a
    baseline and a post-train run would silently invalidate the paired
    comparison (mirrors AGENTS.md invariant #2: pass@k wire params are
    pinned, not knobs).
  * SECONDARY (opt-in via --secondary): k=8, temperature=0.7, 3 repeats,
    over eval-band ONLY (the design doc specs this slice explicitly).
    Distributional signal, reported separately by report.py and never
    blended into the headline number.

LoRA serving assumption (from the design doc): the tuned model is a
distinct model id on the SAME OpenAI-compatible endpoint as the base
model (LM Studio loads GGUF+LoRA or a merged export alongside the base
GGUF). Supplying --model-base and --model-tuned together in one
invocation is supported and is when the cross-endpoint guard applies;
each may also be run alone (matching the checklist's separate
before/after invocations).

Key files are path-proxies. ``--qwen-key-file`` (and its --tuned/--base
overrides) are passed straight through to the icepick subprocess as a
path string -- this module never opens or reads them, and never prints
their contents.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Fixed by docs/eval_harness_design.md's Measurement protocol section.
# Not CLI-overridable -- see module docstring.
GREEDY_K = 1
GREEDY_TEMPERATURE = 0.0
GREEDY_MAX_TOKENS = 2048
GREEDY_THINK = "off"

SECONDARY_K = 8
SECONDARY_TEMPERATURE = 0.7
SECONDARY_MAX_TOKENS = 2048
SECONDARY_THINK = "off"
SECONDARY_REPEATS = 3

ROLE_BASE = "base"
ROLE_TUNED = "tuned"

# Keep in sync with build_eval_set.EVAL_SLICE_BAND. Not imported (this
# sub-repo has no cross-module coupling requirement beyond the on-disk
# eval_set.jsonl contract), but the string must match exactly.
EVAL_SLICE_BAND = "eval_band"

DEFAULT_ICEPICK_BIN = "icepick"


class RunEvalError(ValueError):
    """Config problem caught before any subprocess is spawned, or a subprocess failure."""


@dataclass
class ModelSpec:
    role: str  # ROLE_BASE | ROLE_TUNED
    model_id: str
    backend_url: str
    qwen_key_file: Optional[Path] = None


@dataclass
class RunEvalOutcome:
    greedy_paths: dict = field(default_factory=dict)  # role -> baseline_greedy.jsonl / post_greedy.jsonl
    greedy_manifests: dict = field(default_factory=dict)  # role -> icepick's own pass_at_k_manifest.json
    secondary_paths: dict = field(default_factory=dict)  # role -> [pass_at_k.jsonl per repeat]
    commands: list = field(default_factory=list)  # every argv actually run (safe to print/log: paths only)


def _resolve_models(args) -> list:
    """Turn parsed CLI args into 1-2 ModelSpecs, applying the cross-endpoint guard.

    Split out from run_eval() so tests can exercise the guard logic
    without touching a subprocess.
    """
    if not args.model_base and not args.model_tuned:
        raise RunEvalError("at least one of --model-base / --model-tuned is required")

    specs = []
    if args.model_base:
        specs.append(
            ModelSpec(
                role=ROLE_BASE,
                model_id=args.model_base,
                backend_url=args.backend_url_base or args.backend_url,
                qwen_key_file=args.qwen_key_file_base or args.qwen_key_file,
            )
        )
    if args.model_tuned:
        specs.append(
            ModelSpec(
                role=ROLE_TUNED,
                model_id=args.model_tuned,
                backend_url=args.backend_url_tuned or args.backend_url,
                qwen_key_file=args.qwen_key_file_tuned or args.qwen_key_file,
            )
        )

    for spec in specs:
        if not spec.backend_url:
            raise RunEvalError(
                f"no --backend-url resolved for --model-{spec.role}; pass "
                f"--backend-url (shared) or --backend-url-{spec.role}"
            )

    if len(specs) == 2 and specs[0].backend_url != specs[1].backend_url:
        if not args.allow_cross_endpoint:
            raise RunEvalError(
                "QUANT-CONFOUND GUARD: --model-base and --model-tuned "
                f"resolve to different endpoints ({specs[0].backend_url!r} "
                f"vs {specs[1].backend_url!r}). docs/eval_harness_design.md's "
                "first failure mode is exactly this: cross-quant/hardware "
                "differences are the size of a plausible LoRA gain (mean "
                "|delta|=1.32/8 measured). Pass --allow-cross-endpoint only "
                "if you have deliberately verified both endpoints serve the "
                "same quant."
            )
        print(
            "WARNING: base and tuned run against different endpoints "
            f"({specs[0].backend_url!r} vs {specs[1].backend_url!r}). "
            "--allow-cross-endpoint overrides the guard, but the "
            "cross-quant confound documented in docs/eval_harness_design.md "
            "still applies -- any measured delta may be a hardware/quant "
            "artifact, not a training effect.",
            file=sys.stderr,
        )

    return specs


def _icepick_command(
    icepick_bin: str,
    *,
    input_path: Path,
    output_dir: Path,
    backend_url: str,
    model_id: str,
    k: int,
    temperature: float,
    max_tokens: int,
    think: str,
    qwen_key_file: Optional[Path],
    max_concurrent: int,
) -> list:
    cmd = [
        icepick_bin,
        "processing",
        "pass_at_k",
        "--mode",
        "production",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--backend",
        "qwen_http",
        "--backend-url",
        backend_url,
        "--model",
        model_id,
        "--k",
        str(k),
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--think",
        think,
        "--max-concurrent",
        str(max_concurrent),
    ]
    if qwen_key_file is not None:
        # Path only -- never opened or read by this module. See module
        # docstring's "Key files are path-proxies" note.
        cmd += ["--qwen-key-file", str(qwen_key_file)]
    return cmd


def _default_subprocess_runner(cmd: list) -> None:
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RunEvalError(f"icepick pass_at_k exited {proc.returncode}: {' '.join(cmd)}")


def _filter_slice(eval_set_path: Path, slice_name: str, dest_path: Path) -> int:
    """Write only eval_set.jsonl records tagged eval_slice == slice_name; return the count."""
    count = 0
    with eval_set_path.open("r", encoding="utf-8") as src, dest_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("eval_slice") == slice_name:
                dst.write(json.dumps(row) + "\n")
                count += 1
    return count


def run_eval(
    *,
    eval_set_path: Path,
    output_dir: Path,
    model_specs: list,
    icepick_bin: str = DEFAULT_ICEPICK_BIN,
    max_concurrent: int = 4,
    secondary: bool = False,
    subprocess_runner: Callable = _default_subprocess_runner,
) -> RunEvalOutcome:
    """Run the greedy primary (+ optional secondary) pass for each model spec.

    ``subprocess_runner`` is injectable so tests can exercise the
    orchestration (which dirs get created, which commands get built, in
    what order) without spawning a real subprocess or needing a live
    Qwen endpoint -- mirrors icepick's own injectable-backend test
    pattern.
    """
    if not eval_set_path.exists():
        raise FileNotFoundError(f"eval set not found: {eval_set_path} -- run build_eval_set.py first")
    if not model_specs:
        raise RunEvalError("run_eval() called with no model specs")

    output_dir.mkdir(parents=True, exist_ok=True)
    outcome = RunEvalOutcome()

    for spec in model_specs:
        greedy_dir = output_dir / f"{spec.role}_greedy"
        cmd = _icepick_command(
            icepick_bin,
            input_path=eval_set_path,
            output_dir=greedy_dir,
            backend_url=spec.backend_url,
            model_id=spec.model_id,
            k=GREEDY_K,
            temperature=GREEDY_TEMPERATURE,
            max_tokens=GREEDY_MAX_TOKENS,
            think=GREEDY_THINK,
            qwen_key_file=spec.qwen_key_file,
            max_concurrent=max_concurrent,
        )
        outcome.commands.append(cmd)
        subprocess_runner(cmd)

        dest_name = "baseline_greedy.jsonl" if spec.role == ROLE_BASE else "post_greedy.jsonl"
        dest = output_dir / dest_name
        produced = greedy_dir / "pass_at_k.jsonl"
        if produced.exists():
            shutil.copyfile(produced, dest)
        outcome.greedy_paths[spec.role] = dest
        outcome.greedy_manifests[spec.role] = greedy_dir / "pass_at_k_manifest.json"

    if secondary:
        band_only = output_dir / "_eval_band_only.jsonl"
        n = _filter_slice(eval_set_path, EVAL_SLICE_BAND, band_only)
        if n == 0:
            raise RunEvalError(
                f"--secondary requested but {eval_set_path} has zero "
                f"eval_slice={EVAL_SLICE_BAND!r} records -- nothing to score"
            )
        for spec in model_specs:
            rep_paths = []
            for rep in range(SECONDARY_REPEATS):
                rep_dir = output_dir / f"{spec.role}_secondary" / f"rep{rep}"
                cmd = _icepick_command(
                    icepick_bin,
                    input_path=band_only,
                    output_dir=rep_dir,
                    backend_url=spec.backend_url,
                    model_id=spec.model_id,
                    k=SECONDARY_K,
                    temperature=SECONDARY_TEMPERATURE,
                    max_tokens=SECONDARY_MAX_TOKENS,
                    think=SECONDARY_THINK,
                    qwen_key_file=spec.qwen_key_file,
                    max_concurrent=max_concurrent,
                )
                outcome.commands.append(cmd)
                subprocess_runner(cmd)
                rep_paths.append(rep_dir / "pass_at_k.jsonl")
            outcome.secondary_paths[spec.role] = rep_paths

    return outcome


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evalharness-run",
        description=(
            "Drive icepick processing pass_at_k over the frozen eval set: "
            "greedy pass@1 (always) and an optional k=8 x3 secondary on "
            "eval-band."
        ),
    )
    p.add_argument("--eval-set", type=Path, required=True, help="Path to eval_set.jsonl (build_eval_set.py output).")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for this run's outputs.")
    p.add_argument("--model-base", default=None, help="Base model id (pre-LoRA). At least one of --model-base/--model-tuned is required.")
    p.add_argument("--model-tuned", default=None, help="Tuned model id (post-LoRA).")
    p.add_argument(
        "--backend-url",
        default=None,
        help="Shared OpenAI-compatible endpoint for base and tuned (the common case: one LM Studio server hosts both ids).",
    )
    p.add_argument("--backend-url-base", default=None, help="Override the base model's endpoint if it differs from --backend-url.")
    p.add_argument("--backend-url-tuned", default=None, help="Override the tuned model's endpoint if it differs from --backend-url.")
    p.add_argument(
        "--qwen-key-file",
        type=Path,
        default=None,
        help="Bearer-auth key file for a remote qwen_http gateway (shared). Path only -- never read or printed by this tool.",
    )
    p.add_argument("--qwen-key-file-base", type=Path, default=None, help="Override key file for the base model.")
    p.add_argument("--qwen-key-file-tuned", type=Path, default=None, help="Override key file for the tuned model.")
    p.add_argument(
        "--allow-cross-endpoint",
        action="store_true",
        help="Required to run --model-base and --model-tuned against different --backend-url values (see the quant-confound guard).",
    )
    p.add_argument(
        "--secondary",
        action="store_true",
        help="Also run the k=8, temperature=0.7, 3-repeat distributional secondary on eval-band.",
    )
    p.add_argument("--max-concurrent", type=int, default=4, help="Passed through to icepick's --max-concurrent.")
    p.add_argument(
        "--icepick-bin",
        default=DEFAULT_ICEPICK_BIN,
        help=f"icepick console-script name or full path (default: {DEFAULT_ICEPICK_BIN!r}, resolved on PATH).",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        specs = _resolve_models(args)
    except RunEvalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        outcome = run_eval(
            eval_set_path=args.eval_set,
            output_dir=args.output_dir,
            model_specs=specs,
            icepick_bin=args.icepick_bin,
            max_concurrent=args.max_concurrent,
            secondary=args.secondary,
        )
    except (RunEvalError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "stage": "run_eval",
        "eval_set": str(args.eval_set),
        "roles_run": [s.role for s in specs],
        "commands": outcome.commands,
        "outputs": {
            "greedy": {role: str(p) for role, p in outcome.greedy_paths.items()},
            "secondary": {role: [str(p) for p in paths] for role, paths in outcome.secondary_paths.items()},
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

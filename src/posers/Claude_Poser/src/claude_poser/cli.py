"""CLI: claude-poser score --input <jsonl> --output <json|csv>

The CLI is intentionally minimal: one stage, one output. The rest of the
ModelBreaker pipeline (routing, triage, confirmation, escalation) lives in
the parent repo and consumes this module's score file as input.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import WellposedConfig
from .env_file import load_env_file
from .ingest import load_normalised
from .judge_cache import JudgeCache
from .wellposed import check_records
from .writer import write_csv, write_json


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="claude-poser",
        description="Run the c01 well-posedness check on post-pass@k records.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    score = sub.add_parser("score", help="Score records and write JSON or CSV.")
    score.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input JSONL of post-pass@k records (repeatable).",
    )
    score.add_argument(
        "--output",
        required=True,
        help="Output file. Format inferred from extension (.json or .csv).",
    )
    score.add_argument("--mode", choices=("production", "flow_testing"), default="production")
    score.add_argument("--calibration-sheet", default=None)
    score.add_argument("--judge", action="store_true", help="Enable the judge tier.")
    score.add_argument(
        "--provider",
        choices=("anthropic", "openai"),
        default="anthropic",
        help="Which API backend to use for the judge tier. Default: anthropic.",
    )
    score.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Override the model id. If omitted, resolves to ANTHROPIC_MODEL or "
            "OPENAI_MODEL from env (or a provider-specific default)."
        ),
    )
    score.add_argument(
        "--openai-base-url",
        default=None,
        help=(
            "Override OPENAI_BASE_URL (default https://api.openai.com/v1). "
            "Point this at an OpenAI-compatible server (LM Studio, Ollama, "
            "vLLM, etc.) to judge against a local model."
        ),
    )
    score.add_argument("--judge-samples", type=int, default=3)
    score.add_argument("--judge-uphold", type=int, default=2)
    score.add_argument(
        "--extracted-judge-policy",
        choices=("always", "on_scanner_hit"),
        default="always",
        help=(
            "How to treat extracted-provenance records under --judge. "
            "'always' (default): always defer to the judge; the scanner "
            "provides supplementary evidence but does not gate the call. "
            "'on_scanner_hit': only call the judge when the scanner fires — "
            "cheaper, but restores the pre-fix behavior where scanner "
            "false-negatives become full-pass verdicts. Use only for "
            "cost-sensitive replays of trusted corpora."
        ),
    )
    score.add_argument(
        "--judge-cache",
        default=None,
        help=(
            "Path to a JSONL judge cache file. Cache key includes provider, "
            "model, prompt, and sample id."
        ),
    )
    score.add_argument(
        "--anthropic-key-file",
        default=None,
        help=(
            "Env file with Anthropic credentials (e.g. ../anthro_key.env). "
            "Loaded ONLY when --provider anthropic. Keeps Anthropic secrets "
            "out of OpenAI runs by design."
        ),
    )
    score.add_argument(
        "--openai-key-file",
        default=None,
        help=(
            "Env file with OpenAI credentials (e.g. ../openai_key.env). "
            "Loaded ONLY when --provider openai. Keeps OpenAI secrets out "
            "of Anthropic runs by design."
        ),
    )
    score.add_argument(
        "--env-file",
        action="append",
        default=None,
        help=(
            "General-purpose KEY=VALUE env file (repeatable). Loaded "
            "regardless of provider — use --anthropic-key-file / "
            "--openai-key-file instead when you want provider segregation. "
            "Shell env wins over the file."
        ),
    )

    sub.add_parser("self-test", help="Run a small built-in fixture to confirm wiring.")
    return p


def _apply_env_file(path: str, *, label: str) -> int:
    """Load one env file, print what happened to stderr. Returns 0 / 2."""
    try:
        result = load_env_file(path)
    except FileNotFoundError as e:
        print(f"error ({label}): {e}", file=sys.stderr)
        return 2
    if result.loaded:
        print(
            f"claude-poser: [{label}] loaded {len(result.loaded)} key(s) "
            f"from {result.path} ({', '.join(sorted(result.loaded))})",
            file=sys.stderr,
        )
    if result.already_set:
        print(
            f"claude-poser: [{label}] {len(result.already_set)} key(s) already in env, "
            f"left alone ({', '.join(sorted(result.already_set))})",
            file=sys.stderr,
        )
    for line_no, why in result.skipped:
        print(f"claude-poser: [{label}] skipped {result.path}:{line_no} — {why}", file=sys.stderr)
    return 0


def _load_keys_segregated(args) -> int:
    """Provider-segregated loading.

    Hard guarantee: when --provider X, the key file for the OTHER provider
    is never opened — its credentials cannot leak into the runtime even if
    the user passed both flags. We do warn loudly so a misconfigured
    invocation isn't silently ignored.
    """
    provider = args.provider

    # 1. Provider-matched key file
    if provider == "anthropic" and args.anthropic_key_file:
        rc = _apply_env_file(args.anthropic_key_file, label="anthropic")
        if rc != 0:
            return rc
    if provider == "openai" and args.openai_key_file:
        rc = _apply_env_file(args.openai_key_file, label="openai")
        if rc != 0:
            return rc

    # 2. Warn about provider-mismatched files (declared but ignored)
    if provider != "anthropic" and args.anthropic_key_file:
        print(
            f"claude-poser: --anthropic-key-file {args.anthropic_key_file!r} "
            f"ignored under --provider {provider} (segregation)",
            file=sys.stderr,
        )
    if provider != "openai" and args.openai_key_file:
        print(
            f"claude-poser: --openai-key-file {args.openai_key_file!r} "
            f"ignored under --provider {provider} (segregation)",
            file=sys.stderr,
        )

    # 3. General --env-file (repeatable) — applied regardless of provider
    for path in (args.env_file or []):
        rc = _apply_env_file(path, label="env-file")
        if rc != 0:
            return rc
    return 0


def _run_score(args) -> int:
    rc = _load_keys_segregated(args)
    if rc != 0:
        return rc

    cfg = WellposedConfig(
        enable_judge=args.judge,
        judge_samples=args.judge_samples,
        judge_uphold=args.judge_uphold,
        judge_provider=args.provider,
        extracted_judge_policy=args.extracted_judge_policy,
        processor_mode=args.mode,
        calibration_sheet=args.calibration_sheet,
    )
    if args.judge_model:
        cfg.judge_model = args.judge_model
    if args.openai_base_url:
        cfg.openai_base_url = args.openai_base_url
    cfg.judge_cache_path = args.judge_cache
    cfg.validate()

    if cfg.enable_judge and cfg.processor_mode == "production" and not cfg.active_api_key():
        env_var, hint = (
            ("ANTHROPIC_API_KEY", "--anthropic-key-file ../anthro_key.env")
            if cfg.judge_provider == "anthropic"
            else ("OPENAI_API_KEY", "--openai-key-file ../openai_key.env")
        )
        print(
            f"claude-poser: warning — --judge --provider {cfg.judge_provider} is on but "
            f"{env_var} is not set. Judge calls will return 'defer'. "
            f"Pass {hint} to load it.",
            file=sys.stderr,
        )

    records = load_normalised(args.input)
    cache = JudgeCache(args.judge_cache) if args.judge_cache else None
    results = check_records(records, cfg, cache=cache)

    out_path = Path(args.output)
    if out_path.suffix.lower() == ".csv":
        write_csv(out_path, cfg, args.input, results)
    elif out_path.suffix.lower() == ".json":
        write_json(out_path, cfg, args.input, results)
    else:
        print(
            f"error: --output must end with .json or .csv (got {out_path.suffix!r})",
            file=sys.stderr,
        )
        return 2

    counts: dict[str, int] = {}
    for r in results:
        counts[r["wellposed_status"]] = counts.get(r["wellposed_status"], 0) + 1
    print(
        f"claude-poser: mode={cfg.processor_mode} input={len(results)} "
        f"counts={counts} output={out_path}"
    )
    return 0


def _run_self_test() -> int:
    from . import dangling, wellposed
    from .config import WellposedConfig

    cfg = WellposedConfig()
    cfg.validate()

    rec_computed = {
        "rid": 0,
        "uid": "u1",
        "source": "calc_v1",
        "statement": "Compute the derivative of f(x) = x^2 + 3x at x = 2.",
        "provenance": "computed",
        "truth_policy": None,
    }
    rec_extracted_clean = {
        "rid": 1,
        "uid": "u2",
        "source": "realmath",
        "statement": "Let n be a positive integer. Prove that n^2 - n is even.",
        "provenance": "extracted",
        "truth_policy": None,
    }
    rec_extracted_dangling = {
        "rid": 2,
        "uid": "u3",
        "source": "realmath",
        "statement": "Using Theorem 3.2 from the previous section, deduce the value of A.",
        "provenance": "extracted",
        "truth_policy": None,
    }
    results = [wellposed.check_record(r, cfg) for r in (rec_computed, rec_extracted_clean, rec_extracted_dangling)]
    statuses = [r["wellposed_status"] for r in results]
    print("self-test statuses:", statuses)
    assert statuses == ["pass", "pass", "flag"], f"unexpected: {statuses}"
    assert dangling.scan("see Section 4.1 below"), "dangling scanner regressed"
    print("self-test ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "score":
        return _run_score(args)
    if args.cmd == "self-test":
        return _run_self_test()
    return 2


if __name__ == "__main__":
    sys.exit(main())

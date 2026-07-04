"""Command line interface for the isolated well-posedness module."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .judge_providers import default_key_env_path, make_cached_judge
from .io import infer_format, load_records, write_csv, write_json
from .scoring import SCORE_BY_STATUS, score_records, status_counts


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "score":
        return run_score(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codex-poser")
    subcommands = parser.add_subparsers(dest="command")

    score = subcommands.add_parser(
        "score",
        help="score well-posedness for post-pass@k records",
    )
    score.add_argument(
        "--mode",
        choices=("production", "flow_testing"),
        required=True,
        help="processor mode to record in the output",
    )
    score.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="JSONL, JSON, or CSV input path; may be repeated",
    )
    score.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output .json or .csv path",
    )
    score.add_argument(
        "--format",
        choices=("json", "csv"),
        help="output format; defaults from output extension",
    )
    score.add_argument(
        "--source",
        help="default source name when input rows omit source",
    )
    score.add_argument(
        "--judge",
        action="store_true",
        help="call a judge provider for semantic residue left after the code screen",
    )
    score.add_argument(
        "--judge-provider",
        choices=("anthropic", "openai"),
        default="anthropic",
        help="judge API provider; default: anthropic",
    )
    score.add_argument(
        "--key-env",
        type=Path,
        help="external env file containing the selected provider key; defaults to ../anthro_key.env or ../openai_key.env",
    )
    score.add_argument(
        "--judge-cache",
        type=Path,
        default=Path("out/judge_cache.jsonl"),
        help="cache path for judge replies",
    )
    score.add_argument(
        "--judge-model",
        help="override ANTHROPIC_MODEL or OPENAI_MODEL from the selected env file",
    )
    score.add_argument(
        "--judge-samples",
        type=int,
        default=3,
        help="independent judge samples per deferred record",
    )
    score.add_argument(
        "--judge-uphold",
        type=int,
        default=2,
        help="votes required to uphold an ill-posed flag",
    )
    score.add_argument(
        "--judge-timeout",
        type=float,
        default=60.0,
        help="judge request timeout in seconds",
    )
    return parser


def run_score(args: argparse.Namespace) -> int:
    cache = None
    judge = None
    judge_model = None
    key_env_path = None
    try:
        if args.judge and args.mode != "production":
            raise ValueError("--judge is only allowed with --mode production")
        if args.judge:
            key_env_path = args.key_env or default_key_env_path(args.judge_provider)
            judge, cache, judge_model = make_cached_judge(
                provider=args.judge_provider,
                key_env_path=key_env_path,
                cache_path=args.judge_cache,
                model_override=args.judge_model,
                timeout_seconds=args.judge_timeout,
            )
        records, input_summaries = load_records(args.input, default_source=args.source)
        scored = score_records(
            records,
            judge=judge,
            judge_samples=args.judge_samples,
            judge_uphold=args.judge_uphold,
        )
        rows = [result.to_record(record) for record, result in scored]
        results = [result for _, result in scored]
        output_format = infer_format(args.output, args.format)
        payload = build_payload(
            rows=rows,
            results=results,
            input_summaries=input_summaries,
            output_format=output_format,
            processor_mode=args.mode,
            judge_enabled=args.judge,
            judge_model=judge_model,
            key_env=key_env_path,
            judge_cache=args.judge_cache if args.judge else None,
            judge_provider=args.judge_provider if args.judge else None,
            judge_samples=args.judge_samples,
            judge_uphold=args.judge_uphold,
        )
        if output_format == "csv":
            write_csv(args.output, rows)
        else:
            write_json(args.output, payload)
        if cache is not None:
            cache.save()
    except Exception as exception:
        print(f"codex-poser score failed: {exception}", file=sys.stderr)
        return 1

    counts = payload["counts"]
    print(
        "well_posedness "
        f"mode={args.mode} input={len(records)} "
        f"pass={counts['pass']} flag={counts['flag']} "
        f"defer={counts['defer']} error={counts['error']} "
        f"output={args.output}"
    )
    return 0


def build_payload(
    rows: list[dict],
    results: list,
    input_summaries: list[dict],
    output_format: str,
    processor_mode: str,
    judge_enabled: bool = False,
    judge_model: str | None = None,
    key_env: Path | None = None,
    judge_cache: Path | None = None,
    judge_provider: str | None = None,
    judge_samples: int = 3,
    judge_uphold: int = 2,
) -> dict:
    counts = status_counts(results)
    counts["total"] = len(rows)
    return {
        "run": {
            "module": "well_posedness",
            "check_id": "c01_wellposed",
            "processor_mode": processor_mode,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "output_format": output_format,
        },
        "inputs": input_summaries,
        "counts": counts,
        "parameters": {
            "score_policy": SCORE_BY_STATUS,
            "structural_screens": ["reference", "citation", "label"],
            "computed_provenance_policy": "pass when structural screen is clean",
            "residue_policy": "defer extracted/manual/external/unknown records without structural defects",
            "judge": {
                "enabled": judge_enabled,
                "provider": judge_provider,
                "model": judge_model,
                "key_env": str(key_env) if key_env else None,
                "cache": str(judge_cache) if judge_cache else None,
                "samples": judge_samples,
                "uphold": judge_uphold,
            },
        },
        "warnings": _warnings(rows),
        "records": rows,
    }


def _warnings(rows: list[dict]) -> list[str]:
    warnings = []
    if any(row["well_posedness_status"] == "defer" for row in rows):
        warnings.append("deferred records need a judge or review tier before final deployment decisions")
    if any((row.get("signals") or {}).get("passk_context", {}).get("warning") for row in rows):
        warnings.append("one or more records have pass@k count warnings")
    return warnings


if __name__ == "__main__":
    raise SystemExit(main())

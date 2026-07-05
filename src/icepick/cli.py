"""Icepick CLI.

Three subcommand groups mirror the three subsystems:

    icepick processing <stage> ...
    icepick allocation <stage> ...
    icepick agent <command> ...

Standardisation rules from the spec:
- Every subcommand supports ``--output-dir``.
- Every call-bearing subcommand requires ``--mode`` and rejects flow-
  testing runs without ``--calibration-sheet``.
- Verification subcommands support ``--scratch``.
- Failures use a code, a short label, and a human-readable detail.

This first cut wires the subparser tree and exposes ``processing
ingest-check`` as a real working command (the only stage with working
code today). Every other stage is registered so ``icepick --help`` lists
it, but executing one raises ``NotImplementedError`` from the underlying
module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from icepick import __version__
from icepick.config import ConfigError, validate_mode
from icepick.processing import ingest
from icepick.processing.groundtruth import GroundtruthConfig
from icepick.processing.groundtruth.runner import run as run_groundtruth
from icepick.processing.pass_at_k import BACKEND_VALUES, PassAtKConfig
from icepick.processing.pass_at_k.runner import run as run_pass_at_k
from icepick.processing.poser import (
    BUILD_CHOICES,
    COMPARISON_POLICIES_BASE,
    DEFAULT_STAGES,
    PROVIDER_CHOICES,
    CascadeConfig,
    WellposedConfig,
    all_combos,
    parse_combo,
    parse_stages,
    run_cascade,
)
from icepick.allocation.adapters import realmath_scrape
from icepick.allocation.intake import mount as allocation_mount
from icepick.allocation.manifests import (
    load_manifest,
    new_run_id,
    require_approved,
    write_manifest,
    write_plan,
)
from icepick.contracts.manifests import (
    MODE_FLOW_TESTING,
    MODE_PRODUCTION,
    SOURCE_REALMATH_SCRAPE,
    ApprovedManifest,
)
from icepick.processing.pipeline import run as run_pipeline
from icepick.processing.poser.config import POLICY_INTERSECT
from icepick.processing.poser.runner import run as run_wellposed


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args) or 0
    except ConfigError as exc:
        return _fail("E_CONFIG", "config invariant violated", str(exc))
    except NotImplementedError as exc:
        return _fail(
            "E_NOT_IMPLEMENTED",
            "stage not yet implemented",
            f"{exc}; see docs/plan.md for the build order",
        )
    except FileNotFoundError as exc:
        return _fail("E_NOT_FOUND", "input not found", str(exc))
    except ValueError as exc:
        return _fail("E_INVALID", "invalid input", str(exc))
    except OSError as exc:
        # requests exceptions subclass OSError, so a dead/throttled network
        # lands here (FileNotFoundError is caught above). Scrape progress is
        # checkpointed, so the fix is simply to rerun.
        return _fail(
            "E_NETWORK",
            "network or I/O failure",
            f"{exc}; any scrape progress is checkpointed — rerun the same command to resume",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="icepick",
        description="Processing surface with in-house acquisition for ModelBreaker-style records.",
    )
    parser.add_argument("--version", action="version", version=f"icepick {__version__}")
    sub = parser.add_subparsers(dest="subsystem", metavar="<subsystem>")

    _build_processing(sub)
    _build_allocation(sub)
    _build_agent(sub)

    return parser


def _build_processing(sub) -> None:
    g = sub.add_parser("processing", help="Run processing-side stages.")
    s = g.add_subparsers(dest="stage", metavar="<stage>")

    p = s.add_parser(
        "ingest-check",
        help="Load JSONL inputs and report normalised record counts (working).",
    )
    p.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("PATH", "SOURCE"),
        required=True,
        help="Path and source name. May be repeated.",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None, help="Stop after N records (for sanity checks).")
    p.set_defaults(_handler=_run_ingest_check)

    p = s.add_parser(
        "wellposed",
        help="Pre-gate well-posedness via an external poser fleet (working).",
    )
    p.add_argument(
        "--combo",
        action="append",
        default=None,
        metavar="BUILD:PROVIDER",
        help=(
            "Combination to run, e.g. 'claude:anthropic' or 'codex:openai'. "
            f"build ∈ {{{','.join(BUILD_CHOICES)}}}, "
            f"provider ∈ {{{','.join(PROVIDER_CHOICES)}}}. "
            "Repeatable. Pass 'all' to run all four combinations."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("production", "flow_testing"),
        required=True,
        help="Required. flow_testing additionally requires --calibration-sheet.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to pass@k records JSONL (the file the gate will eventually consume).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for poser inputs, raw outputs, normalised verdicts, manifest, and comparison.",
    )
    p.add_argument(
        "--anthro-key-file",
        default=None,
        help="anthro_key.env for the anthropic provider (required when any combo uses provider=anthropic with judge on).",
    )
    p.add_argument(
        "--openai-key-file",
        default=None,
        help="openai_key.env for the openai provider (required when any combo uses provider=openai with judge on).",
    )
    p.add_argument("--calibration-sheet", default=None)
    p.add_argument(
        "--claude-judge-model",
        default=None,
        help="Override judge model for the claude build (applies to whichever provider it runs against).",
    )
    p.add_argument(
        "--codex-judge-model",
        default=None,
        help="Override judge model for the codex build.",
    )
    p.add_argument("--judge-samples", type=int, default=3)
    p.add_argument("--judge-uphold", type=int, default=2)
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable judge tier (code-tier only). Reduces cost and removes the secret requirement.",
    )
    p.add_argument(
        "--comparison-policy",
        default=POLICY_INTERSECT,
        help=(
            "How fleet verdicts combine into the gate-input file. "
            "Choices: intersect | union | majority | prefer:<build>:<provider>. "
            "Default: intersect."
        ),
    )
    p.add_argument(
        "--extracted-judge-policy",
        choices=("always", "on_scanner_hit"),
        default="always",
        help=(
            "How claude-poser treats extracted-provenance records under --judge. "
            "'always' (default): always defer to the judge; the scanner is "
            "supplementary evidence only. 'on_scanner_hit': call the judge only "
            "when the scanner fires — cheaper, but restores the pre-fix behaviour "
            "where scanner false-negatives become full-pass verdicts. "
            "codex-poser ignores this flag; the setting is still recorded in the "
            "run manifest for audit."
        ),
    )
    p.add_argument(
        "--serialize-fleet",
        action="store_true",
        help="Run fleet combos sequentially (e.g. shared rate-limit budget) instead of in parallel.",
    )
    p.add_argument(
        "--cost-per-input-mtok",
        type=float,
        default=None,
        help="USD per million input tokens. Enables estimated_cost in the manifest.",
    )
    p.add_argument(
        "--cost-per-output-mtok",
        type=float,
        default=None,
        help="USD per million output tokens. Enables estimated_cost in the manifest.",
    )
    p.add_argument(
        "--claude-cli",
        default=None,
        help="Override the claude-poser binary path (default: 'claude-poser' on PATH).",
    )
    p.add_argument(
        "--codex-cli",
        default=None,
        help="Override the codex-poser binary path (default: 'codex-poser' on PATH).",
    )
    p.set_defaults(_handler=_run_wellposed)

    p = s.add_parser(
        "wellposed-cascade",
        help="Sequential single-combo cascade: cheaper than parallel fleet at high precision (working).",
    )
    p.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help=(
            "Ordered, comma-separated combos. Default: "
            f"{','.join(DEFAULT_STAGES)}. Each stage runs on the survivors of the previous. "
            "Append '?advisory' to a combo to make that stage non-gating: it "
            "records verdicts and writes its rejections to "
            "flagged_for_review.jsonl, but every record flows on to the next "
            "stage. (claude:openai defaults to advisory — see the 2026-07-04 "
            "stage-3 kill analysis: 82.5%% false kills as a hard gate.)"
        ),
    )
    p.add_argument(
        "--mode",
        choices=("production", "flow_testing"),
        required=True,
        help="Required. flow_testing additionally requires --calibration-sheet.",
    )
    p.add_argument("--input", required=True, help="Path to input records JSONL.")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Top-level dir. Each stage writes to <output-dir>/stage_<n>_<slug>/; "
        "final corpus is <output-dir>/final_corpus.jsonl.",
    )
    p.add_argument(
        "--anthro-key-file",
        default=None,
        help="anthro_key.env (required when any stage uses provider=anthropic with judge on and env has no key).",
    )
    p.add_argument(
        "--openai-key-file",
        default=None,
        help="openai_key.env (required when any stage uses provider=openai with judge on and env has no key).",
    )
    p.add_argument("--calibration-sheet", default=None)
    p.add_argument(
        "--claude-judge-model",
        default=None,
        help="Override judge model for claude stages.",
    )
    p.add_argument(
        "--codex-judge-model",
        default=None,
        help="Override judge model for codex stages.",
    )
    p.add_argument("--judge-samples", type=int, default=3)
    p.add_argument("--judge-uphold", type=int, default=2)
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable judge tier for every stage (code-tier only).",
    )
    p.add_argument(
        "--extracted-judge-policy",
        choices=("always", "on_scanner_hit"),
        default="always",
        help="Passed to claude stages. Codex stages ignore it (echoed for audit).",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Per-stage retry count for uids that come back STATUS_ERROR (transient network etc). Default 2 (up to 3 attempts).",
    )
    p.add_argument(
        "--retry-base-delay",
        type=float,
        default=2.0,
        help="Base seconds for exponential backoff between retries.",
    )
    p.add_argument(
        "--retry-max-delay",
        type=float,
        default=30.0,
        help="Cap for exponential backoff between retries.",
    )
    p.add_argument(
        "--cost-per-input-mtok",
        type=float,
        default=None,
        help="USD per million input tokens. Enables estimated_cost aggregation.",
    )
    p.add_argument(
        "--cost-per-output-mtok",
        type=float,
        default=None,
        help="USD per million output tokens.",
    )
    p.add_argument("--claude-cli", default=None, help="Override claude-poser binary path.")
    p.add_argument("--codex-cli", default=None, help="Override codex-poser binary path.")
    p.set_defaults(_handler=_run_wellposed_cascade)

    p = s.add_parser(
        "groundtruth",
        help="Publication-status check via Anthropic web_search (working). "
        "Run before OR after pass@k — the module is position-agnostic.",
    )
    p.add_argument(
        "--mode",
        choices=("production", "flow_testing"),
        required=True,
        help="Required. flow_testing additionally requires --calibration-sheet.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to records JSONL. Records may be pre- or post-pass@k.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for verdicts, published filter, discarded log, manifest, and cache.",
    )
    p.add_argument("--anthro-key-file", default=None, help="anthro_key.env for ANTHROPIC_API_KEY.")
    p.add_argument("--calibration-sheet", default=None)
    p.add_argument("--judge-model", default="claude-opus-4-7")
    p.add_argument("--judge-samples", type=int, default=3)
    p.add_argument("--judge-uphold", type=int, default=2)
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument(
        "--cache-path",
        default=None,
        help="JSONL cache keyed by arxiv_id. Hits skip the Anthropic call.",
    )
    p.add_argument(
        "--keep-generated",
        action="store_true",
        help="Don't discard provenance=computed records (icepick policy is to discard).",
    )
    p.add_argument(
        "--custom-bar-instructions",
        default=None,
        help="Extra constraints appended to the publication-status bar.",
    )
    p.add_argument(
        "--cost-per-input-mtok",
        type=float,
        default=None,
        help="USD per million input tokens. Enables estimated_cost in the manifest.",
    )
    p.add_argument(
        "--cost-per-output-mtok",
        type=float,
        default=None,
        help="USD per million output tokens. Enables estimated_cost in the manifest.",
    )
    p.set_defaults(_handler=_run_groundtruth)

    p = s.add_parser(
        "pass_at_k",
        help="Pass@k difficulty scoring against a subject model (working). "
        "Recommended AFTER wellposed-cascade so rollouts are not wasted on ill-posed problems.",
    )
    p.add_argument(
        "--mode",
        choices=("production", "flow_testing"),
        required=True,
        help="Required. flow_testing additionally requires --calibration-sheet.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to records JSONL (handoff or wellposed survivors).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for scored records, rollout progress, and manifest.",
    )
    p.add_argument(
        "--backend",
        choices=BACKEND_VALUES,
        default="qwen_http",
        help=(
            "Subject-model backend. Policy default is qwen_http — a local "
            "OpenAI-compatible endpoint (LM Studio / vLLM / Ollama). Paid "
            "backends (anthropic, openai) additionally require "
            "--allow-live-calls AND --i-understand-paid-backend-is-off-policy."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="Subject model id. Default: the backend's default model.",
    )
    p.add_argument("--k", type=int, default=8, help="Rollouts per record.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument(
        "--think",
        choices=("on", "off"),
        default="off",
        help="Request model reasoning; <think> tags are always stripped before scoring.",
    )
    p.add_argument(
        "--backend-url",
        default=None,
        help="qwen_http only: OpenAI-compatible chat/completions URL (e.g. LM Studio).",
    )
    p.add_argument("--anthro-key-file", default=None, help="anthro_key.env for ANTHROPIC_API_KEY.")
    p.add_argument("--openai-key-file", default=None, help="openai_key.env for OPENAI_API_KEY.")
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Concurrent record scoring; rollouts within a record stay sequential.",
    )
    p.add_argument("--calibration-sheet", default=None)
    p.add_argument(
        "--allow-live-calls",
        action="store_true",
        help="Kill switch #1: paid backends refuse production without this.",
    )
    p.add_argument(
        "--i-understand-paid-backend-is-off-policy",
        action="store_true",
        help=(
            "Kill switch #2: pass@k policy is qwen_http (local, free). "
            "Selecting --backend anthropic or --backend openai requires "
            "this flag AND --allow-live-calls so the choice is deliberate."
        ),
    )
    p.add_argument(
        "--keep-garbage",
        action="store_true",
        help="Score records with junk truth instead of dropping them.",
    )
    p.add_argument(
        "--cost-per-input-mtok",
        type=float,
        default=None,
        help="USD per million input tokens. Enables estimated_cost in the manifest.",
    )
    p.add_argument(
        "--cost-per-output-mtok",
        type=float,
        default=None,
        help="USD per million output tokens. Enables estimated_cost in the manifest.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Per-rollout retry count for backend errors (network, 429/5xx).",
    )
    p.add_argument(
        "--retry-base-delay",
        type=float,
        default=2.0,
        help="Base seconds for exponential backoff between retries.",
    )
    p.add_argument(
        "--retry-max-delay",
        type=float,
        default=30.0,
        help="Cap for exponential backoff between retries.",
    )
    p.set_defaults(_handler=_run_pass_at_k)

    p = s.add_parser(
        "pipeline",
        help="End-to-end: groundtruth → poser → final corpus. One command (working).",
    )
    p.add_argument(
        "--mode",
        choices=("production", "flow_testing"),
        required=True,
        help="Required. Applied to both stages. flow_testing requires --calibration-sheet.",
    )
    p.add_argument("--input", required=True, help="Records JSONL to feed groundtruth.")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Top-level dir. Stages write to {output-dir}/groundtruth/ and {output-dir}/wellposed/; "
        "final corpus is {output-dir}/final_corpus.jsonl.",
    )
    # groundtruth knobs
    p.add_argument("--anthro-key-file", default=None,
                   help="Required in production: ANTHROPIC_API_KEY for both stages.")
    p.add_argument("--openai-key-file", default=None,
                   help="Required when any poser combo uses provider=openai with judge on.")
    p.add_argument("--calibration-sheet", default=None,
                   help="Required in flow_testing mode. Used by both stages.")
    p.add_argument("--gt-judge-model", default="claude-opus-4-7",
                   help="Judge model for the groundtruth stage.")
    p.add_argument("--gt-judge-samples", type=int, default=3)
    p.add_argument("--gt-judge-uphold", type=int, default=2)
    p.add_argument("--gt-cache-path", default=None,
                   help="JSONL cache keyed by arxiv_id for the groundtruth stage.")
    p.add_argument("--gt-cost-per-input-mtok", type=float, default=None,
                   help="USD per million input tokens for groundtruth cost estimation.")
    p.add_argument("--gt-cost-per-output-mtok", type=float, default=None,
                   help="USD per million output tokens for groundtruth cost estimation.")
    # poser knobs
    p.add_argument(
        "--combo",
        action="append",
        default=None,
        metavar="BUILD:PROVIDER",
        help=f"Repeatable. build ∈ {{{','.join(BUILD_CHOICES)}}}, "
        f"provider ∈ {{{','.join(PROVIDER_CHOICES)}}}. 'all' expands to all four.",
    )
    p.add_argument("--claude-judge-model", default=None)
    p.add_argument("--codex-judge-model", default=None)
    p.add_argument("--judge-samples", type=int, default=3,
                   help="Per-combo samples (poser stage).")
    p.add_argument("--judge-uphold", type=int, default=2)
    p.add_argument("--no-judge", action="store_true",
                   help="Disable the poser judge tier (code-tier only).")
    p.add_argument("--comparison-policy", default=POLICY_INTERSECT,
                   help="intersect|union|majority|prefer:<build>:<provider>")
    p.add_argument("--extracted-judge-policy",
                   choices=("always", "on_scanner_hit"),
                   default="always",
                   help=(
                       "claude-poser policy for extracted records. 'always' "
                       "(default) defers to the judge unconditionally; "
                       "'on_scanner_hit' restores legacy cost-gated behaviour."
                   ))
    p.add_argument("--serialize-fleet", action="store_true")
    p.add_argument("--poser-cost-per-input-mtok", type=float, default=None,
                   help="USD per million input tokens for poser-stage cost estimation.")
    p.add_argument("--poser-cost-per-output-mtok", type=float, default=None,
                   help="USD per million output tokens for poser-stage cost estimation.")
    p.add_argument("--claude-cli", default=None)
    p.add_argument("--codex-cli", default=None)
    # Optional pass@k stage. Off by default; opt in with --enable-pass-at-k
    # or by supplying --pak-backend (either signals "wire the stage").
    p.add_argument(
        "--pipeline-order",
        choices=("classic", "solvable-first"),
        default="classic",
        help=(
            "Order of the non-groundtruth stages. 'classic' runs wellposed "
            "then pass@k. 'solvable-first' runs pass@k first and drops "
            "records the model can't score before the expensive wellposed "
            "cascade runs."
        ),
    )
    p.add_argument(
        "--enable-pass-at-k",
        action="store_true",
        help="Add the pass@k scoring stage to the chain. Requires --pak-backend.",
    )
    p.add_argument(
        "--pak-backend",
        choices=BACKEND_VALUES,
        default=None,
        help="Pass@k backend. Setting this also implies --enable-pass-at-k.",
    )
    p.add_argument("--pak-model", default=None, help="Pass@k subject model id.")
    p.add_argument("--pak-k", type=int, default=8)
    p.add_argument("--pak-temperature", type=float, default=0.7)
    p.add_argument("--pak-max-tokens", type=int, default=8192)
    p.add_argument("--pak-think", choices=("on", "off"), default="off")
    p.add_argument("--pak-backend-url", default=None,
                   help="qwen_http endpoint URL. Required with --pak-backend qwen_http.")
    p.add_argument("--pak-max-concurrent", type=int, default=4)
    p.add_argument("--pak-allow-live-calls", action="store_true",
                   help="Kill switch #1 opt-in for the paid pass@k backends.")
    p.add_argument(
        "--pak-i-understand-off-policy",
        action="store_true",
        dest="pak_i_understand_off_policy",
        help=(
            "Kill switch #2: pass@k policy is qwen_http. Selecting a paid "
            "backend requires this flag AND --pak-allow-live-calls."
        ),
    )
    p.add_argument("--pak-cost-per-input-mtok", type=float, default=None)
    p.add_argument("--pak-cost-per-output-mtok", type=float, default=None)
    p.set_defaults(_handler=_run_pipeline)

    # The poser stage IS the gate; there is no separate `processing gate`.
    # `stage-tests` stays as a stub for running the in-tree test suite
    # via the CLI without remembering pytest invocations.
    for stage_name, help_text in (
        ("stage-tests", "Run the processing-only test suite."),
    ):
        p = s.add_parser(stage_name, help=help_text)
        p.add_argument("--mode", choices=("production", "flow_testing"))
        p.add_argument("--calibration-sheet", default=None)
        p.add_argument("--output-dir", default=None)
        p.add_argument("--scratch", action="store_true", help="Redirect outputs to a scratch dir.")
        p.set_defaults(_handler=_stub_handler(f"processing {stage_name}"))


def _build_allocation(sub) -> None:
    g = sub.add_parser("allocation", help="Run allocation-side stages.")
    s = g.add_subparsers(dest="stage", metavar="<stage>")

    p = s.add_parser(
        "mount",
        help="Scan a mounted path (file or dir) and write canonical handoff JSONL (working).",
    )
    p.add_argument("--path", required=True, help="File or directory to mount.")
    p.add_argument("--source", required=True, help="Source name stamped on every record.")
    p.add_argument(
        "--provenance",
        required=True,
        choices=("manual", "external", "extracted"),
        help="Stamped on every record. manual=operator-provided, "
        "external=handed in by another system, extracted=from a paper scrape.",
    )
    p.add_argument(
        "--truth-policy",
        default="unknown",
        choices=("trusted", "extracted", "unknown"),
        help="How downstream stages should treat the ground truth. Default: unknown.",
    )
    p.add_argument("--output-dir", required=True,
                   help="Top-level intake dir. Writes to {output-dir}/runs/<run_id>/.")
    p.add_argument("--requested-by", default="cli",
                   help="Recorded on the manifest. Auto-approved (manual mounts spend no calls).")
    p.add_argument("--family", default=None, help="Optional family stamp.")
    p.add_argument(
        "--column",
        action="append",
        default=None,
        metavar="CANONICAL=SOURCE",
        help="CSV/TSV column projection, e.g. --column statement=question --column answer=gold. Repeatable.",
    )
    p.set_defaults(_handler=_run_allocation_mount)

    p = s.add_parser(
        "validate-manifest",
        help="Load an ApprovedManifest from disk and report whether it's approved (working).",
    )
    p.add_argument("--manifest", required=True, help="Path to manifest.json")
    p.set_defaults(_handler=_run_validate_manifest)

    p = s.add_parser(
        "plan",
        help="Write a ProposedPlan for an acquisition run (working).",
    )
    p.add_argument(
        "--source-type",
        required=True,
        choices=(SOURCE_REALMATH_SCRAPE,),
        help="Which acquisition adapter to plan against. Only realmath_scrape today.",
    )
    p.add_argument("--source", required=True, help="Source name stamped on the plan and downstream records.")
    p.add_argument("--target-count", type=int, required=True, help="Target handoff record count.")
    p.add_argument("--output-dir", required=True, help="Top-level intake dir. Plan writes to {output-dir}/plans/.")
    p.add_argument("--requested-by", default="cli", help="Recorded on the plan.")
    p.add_argument("--family", action="append", default=None, help="Optional family filter. Repeatable.")
    p.add_argument("--notes", default="", help="Free-text operator notes.")
    p.add_argument(
        "--fixture-path",
        default=None,
        help="Optional path to the flow-testing fixture (recorded on the plan for approvers).",
    )
    # Scrape-window knobs — the acquisition filter, recorded on the plan's
    # scrape_window. Distinct from --family, which only labels output records.
    p.add_argument(
        "--category",
        default=None,
        help="arXiv category to scrape, e.g. 'math.AP' for PDEs (Analysis of PDEs) "
        "or 'math.NT' for number theory. A bare 'math' scrapes every math subcategory.",
    )
    p.add_argument("--year", type=int, default=None, help="Scrape-window start year.")
    p.add_argument("--month", type=int, default=None, help="Scrape-window start month (1-12).")
    p.add_argument("--max-papers", type=int, default=None, help="Cap on papers to scrape.")
    p.add_argument(
        "--max-per-paper",
        type=int,
        default=None,
        help="Cap candidates taken from any one paper, so a theorem-dense paper can't "
        "monopolise --target-count. Trades depth for corpus breadth.",
    )
    p.add_argument(
        "--primary-only",
        action="store_true",
        help="Record intent to keep only papers whose PRIMARY category matches --category "
        "(excludes cross-listed papers). Otherwise cross-lists are included.",
    )
    p.add_argument(
        "--extraction",
        choices=("abstract", "latex", "qa"),
        default=None,
        help="Candidate depth: 'abstract' (default, one metadata candidate per paper), "
        "'latex' (mine theorem statements from the e-print source), or 'qa' (LLM turns each "
        "theorem into a verifiable question+answer; production needs ANTHROPIC_API_KEY).",
    )
    p.add_argument(
        "--auto-approve",
        action="store_true",
        help="Also emit an ApprovedManifest for this plan. Refused for --mode production.",
    )
    p.add_argument(
        "--mode",
        choices=("production", "flow_testing"),
        default=None,
        help="Required with --auto-approve. Also stamped on the manifest.",
    )
    p.add_argument(
        "--calibration-sheet",
        default=None,
        help="Path to the calibration/fixture JSONL. Required by adapter.run when mode=flow_testing.",
    )
    p.add_argument(
        "--approved-by",
        default=None,
        help="Approver identifier. Required with --auto-approve.",
    )
    p.add_argument(
        "--approval-notes",
        default="",
        help="Optional free-text approval notes recorded on the manifest.",
    )
    p.add_argument(
        "--call-budget",
        type=int,
        default=0,
        help="Approved call budget for the run. Required non-zero for production scraping.",
    )
    p.set_defaults(_handler=_run_allocation_plan)

    p = s.add_parser(
        "approve",
        help="Turn a proposed plan into an approved manifest — the human gate before run (working).",
    )
    p.add_argument("--plan", required=True, help="Path to a proposed_plan.json written by 'allocation plan'.")
    p.add_argument(
        "--mode",
        required=True,
        choices=("production", "flow_testing"),
        help="Run mode stamped on the manifest. production authorises real scraping.",
    )
    p.add_argument("--approved-by", required=True, help="Approver identifier (recorded on the manifest).")
    p.add_argument(
        "--call-budget",
        type=int,
        default=None,
        help="Approved call budget. Production must cover the plan's estimated_calls.",
    )
    p.add_argument(
        "--calibration-sheet",
        default=None,
        help="Fixture JSONL. Required for --mode flow_testing (the source the run replays).",
    )
    p.add_argument("--output-dir", required=True, help="Intake dir. Manifest lands under {output-dir}/runs/<run_id>/.")
    p.add_argument("--approval-notes", default="", help="Optional free-text approval notes.")
    p.set_defaults(_handler=_run_allocation_approve)

    p = s.add_parser(
        "run",
        help="Execute an ApprovedManifest via the source-type's acquisition adapter (working).",
    )
    p.add_argument("--manifest", required=True, help="Path to an ApprovedManifest JSON.")
    p.set_defaults(_handler=_run_allocation_run)


def _build_agent(sub) -> None:
    g = sub.add_parser("agent", help="Manager-model chat console (low priority).")
    s = g.add_subparsers(dest="command", metavar="<command>")
    p = s.add_parser("chat", help="Start the manager-model chat console.")
    p.add_argument("--mode", choices=("production", "flow_testing"))
    p.add_argument("--calibration-sheet", default=None)
    p.set_defaults(_handler=_stub_handler("agent chat"))


def _run_ingest_check(args) -> int:
    inputs = [(Path(p), source) for p, source in args.input]
    by_source: dict = {}
    seen_uids: dict = {}
    total = 0
    for record in ingest.load_inputs(inputs):
        total += 1
        bucket = by_source.setdefault(
            record.source,
            {"records": 0, "computed": 0, "extracted": 0, "manual": 0, "external": 0, "uid_collisions": 0},
        )
        bucket["records"] += 1
        bucket[record.provenance] = bucket.get(record.provenance, 0) + 1
        prior = seen_uids.get(record.uid)
        if prior and prior != (record.source, record.statement):
            bucket["uid_collisions"] += 1
        seen_uids[record.uid] = (record.source, record.statement)
        if args.limit is not None and total >= args.limit:
            break

    summary = {
        "run": {"stage": "ingest-check", "total_records": total},
        "inputs": [{"path": str(p), "source": s} for p, s in inputs],
        "counts": {"by_source": by_source},
    }

    print(json.dumps(summary, indent=2))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ingest_check_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


def _run_wellposed(args) -> int:
    validate_mode(args.mode, args.calibration_sheet)
    combos = _parse_combos(args.combo)
    cfg = WellposedConfig(
        combos=combos,
        mode=args.mode,
        output_dir=Path(args.output_dir),
        anthropic_key_file=Path(args.anthro_key_file) if args.anthro_key_file else None,
        openai_key_file=Path(args.openai_key_file) if args.openai_key_file else None,
        enable_judge_tier=not args.no_judge,
        judge_samples=args.judge_samples,
        judge_uphold=args.judge_uphold,
        calibration_sheet=Path(args.calibration_sheet) if args.calibration_sheet else None,
        comparison_policy=args.comparison_policy,
        extracted_judge_policy=args.extracted_judge_policy,
        serialize_fleet=args.serialize_fleet,
        cost_per_input_mtok=args.cost_per_input_mtok,
        cost_per_output_mtok=args.cost_per_output_mtok,
    )
    if args.claude_judge_model:
        cfg.claude.judge_model = args.claude_judge_model
    if args.codex_judge_model:
        cfg.codex.judge_model = args.codex_judge_model
    if args.claude_cli:
        cfg.claude.cli_path = args.claude_cli
    if args.codex_cli:
        cfg.codex.cli_path = args.codex_cli

    cfg.validate()

    records = list(_iter_jsonl(Path(args.input)))
    outcome = run_wellposed(cfg=cfg, records=records)

    summary = {
        "stage": "wellposed",
        "combos": [c.key() for c in cfg.combos],
        "mode": cfg.mode,
        "input": args.input,
        "input_record_count": len(records),
        "counts": outcome.counts,
        "outputs": {
            "manifest": str(outcome.manifest_path),
            "gate_input": str(outcome.gate_input_path),
            "comparison": str(outcome.comparison_path) if outcome.comparison_path else None,
            "normalised": {k: str(v) for k, v in outcome.normalised_paths.items()},
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_wellposed_cascade(args) -> int:
    validate_mode(args.mode, args.calibration_sheet)
    stage_specs = _parse_stage_list(args.stages)
    cfg = CascadeConfig(
        stages=stage_specs,
        mode=args.mode,
        output_dir=Path(args.output_dir),
        anthropic_key_file=Path(args.anthro_key_file) if args.anthro_key_file else None,
        openai_key_file=Path(args.openai_key_file) if args.openai_key_file else None,
        calibration_sheet=Path(args.calibration_sheet) if args.calibration_sheet else None,
        enable_judge_tier=not args.no_judge,
        judge_samples=args.judge_samples,
        judge_uphold=args.judge_uphold,
        extracted_judge_policy=args.extracted_judge_policy,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        retry_max_delay=args.retry_max_delay,
        cost_per_input_mtok=args.cost_per_input_mtok,
        cost_per_output_mtok=args.cost_per_output_mtok,
    )
    if args.claude_judge_model:
        cfg.claude.judge_model = args.claude_judge_model
    if args.codex_judge_model:
        cfg.codex.judge_model = args.codex_judge_model
    if args.claude_cli:
        cfg.claude.cli_path = args.claude_cli
    if args.codex_cli:
        cfg.codex.cli_path = args.codex_cli

    cfg.validate()

    records = list(_iter_jsonl(Path(args.input)))
    outcome = run_cascade(cfg=cfg, records=records)

    summary = {
        "stage": "wellposed-cascade",
        "mode": cfg.mode,
        "input": args.input,
        "input_record_count": len(records),
        "stages": [s.spec_string() for s in cfg.stages],
        "overall_counts": outcome.overall_counts,
        "final_corpus": {
            "path": str(outcome.final_corpus_path),
            "record_count": outcome.final_corpus_count,
        },
        "total_token_usage": outcome.total_token_usage,
        "total_estimated_cost_usd": outcome.total_estimated_cost_usd,
        "total_wall_clock_seconds": outcome.total_wall_clock_seconds,
        "manifest": str(outcome.manifest_path),
        "stage_outputs": [
            {
                "index": s.stage.index,
                "combo": s.stage.combo.key(),
                "advisory": s.stage.advisory,
                "counts": s.counts,
                "survivor_uid_count": s.survivor_uid_count,
                "wall_clock_seconds": s.wall_clock_seconds,
                "wellposed_manifest": str(s.wellposed_manifest_path),
                "flagged_for_review": str(s.flagged_for_review_path) if s.flagged_for_review_path else None,
                "retry_events": s.retry_events,
            }
            for s in outcome.stages
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_pipeline(args) -> int:
    validate_mode(args.mode, args.calibration_sheet)

    gt_cfg = GroundtruthConfig(
        mode=args.mode,
        output_dir=Path(args.output_dir) / "groundtruth",  # overridden by pipeline.run
        anthropic_key_file=Path(args.anthro_key_file) if args.anthro_key_file else None,
        judge_model=args.gt_judge_model,
        judge_samples=args.gt_judge_samples,
        judge_uphold=args.gt_judge_uphold,
        cache_path=Path(args.gt_cache_path) if args.gt_cache_path else None,
        calibration_sheet=Path(args.calibration_sheet) if args.calibration_sheet else None,
        cost_per_input_mtok=args.gt_cost_per_input_mtok,
        cost_per_output_mtok=args.gt_cost_per_output_mtok,
    )

    combos = _parse_combos(args.combo)
    poser_cfg = WellposedConfig(
        combos=combos,
        mode=args.mode,
        output_dir=Path(args.output_dir) / "wellposed",  # overridden by pipeline.run
        anthropic_key_file=Path(args.anthro_key_file) if args.anthro_key_file else None,
        openai_key_file=Path(args.openai_key_file) if args.openai_key_file else None,
        enable_judge_tier=not args.no_judge,
        judge_samples=args.judge_samples,
        judge_uphold=args.judge_uphold,
        calibration_sheet=Path(args.calibration_sheet) if args.calibration_sheet else None,
        comparison_policy=args.comparison_policy,
        extracted_judge_policy=args.extracted_judge_policy,
        serialize_fleet=args.serialize_fleet,
        cost_per_input_mtok=args.poser_cost_per_input_mtok,
        cost_per_output_mtok=args.poser_cost_per_output_mtok,
    )
    if args.claude_judge_model:
        poser_cfg.claude.judge_model = args.claude_judge_model
    if args.codex_judge_model:
        poser_cfg.codex.judge_model = args.codex_judge_model
    if args.claude_cli:
        poser_cfg.claude.cli_path = args.claude_cli
    if args.codex_cli:
        poser_cfg.codex.cli_path = args.codex_cli

    # Optional pass@k stage: opt in via --enable-pass-at-k or by
    # supplying a backend explicitly. Backend implies the stage is on.
    pak_cfg = None
    if args.enable_pass_at_k or args.pak_backend:
        pak_cfg = PassAtKConfig(
            mode=args.mode,
            output_dir=Path(args.output_dir) / "pass_at_k",  # overridden by pipeline.run
            # Policy default: qwen_http (free, local). Paid backends need
            # both --pak-allow-live-calls AND --pak-i-understand-off-policy.
            backend=args.pak_backend or "qwen_http",
            model=args.pak_model,
            k=args.pak_k,
            temperature=args.pak_temperature,
            max_tokens=args.pak_max_tokens,
            think=args.pak_think == "on",
            max_concurrent=args.pak_max_concurrent,
            calibration_sheet=Path(args.calibration_sheet) if args.calibration_sheet else None,
            anthropic_key_file=Path(args.anthro_key_file) if args.anthro_key_file else None,
            openai_key_file=Path(args.openai_key_file) if args.openai_key_file else None,
            backend_url=args.pak_backend_url,
            allow_live_calls=args.pak_allow_live_calls,
            i_understand_paid_backend_is_off_policy=args.pak_i_understand_off_policy,
            cost_per_input_mtok=args.pak_cost_per_input_mtok,
            cost_per_output_mtok=args.pak_cost_per_output_mtok,
        )

    outcome = run_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        groundtruth_cfg=gt_cfg,
        poser_cfg=poser_cfg,
        pass_at_k_cfg=pak_cfg,
        order=args.pipeline_order,
    )

    summary = {
        "stage": "pipeline",
        "mode": args.mode,
        "order": outcome.order,
        "input": args.input,
        "final_corpus": {
            "path": str(outcome.final_corpus_path),
            "record_count": outcome.final_corpus_count,
        },
        "groundtruth": {
            "manifest": str(outcome.groundtruth_manifest_path),
            "counts": outcome.groundtruth_counts,
        },
        "poser": {
            "manifest": str(outcome.poser_manifest_path),
            "counts": outcome.poser_counts,
        },
        "pass_at_k": (
            {
                "manifest": str(outcome.pass_at_k_manifest_path),
                "counts": outcome.pass_at_k_counts,
            }
            if outcome.pass_at_k_manifest_path is not None else None
        ),
        "pipeline_manifest": str(outcome.manifest_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_groundtruth(args) -> int:
    validate_mode(args.mode, args.calibration_sheet)
    cfg = GroundtruthConfig(
        mode=args.mode,
        output_dir=Path(args.output_dir),
        anthropic_key_file=Path(args.anthro_key_file) if args.anthro_key_file else None,
        judge_model=args.judge_model,
        judge_samples=args.judge_samples,
        judge_uphold=args.judge_uphold,
        max_concurrent=args.max_concurrent,
        cache_path=Path(args.cache_path) if args.cache_path else None,
        calibration_sheet=Path(args.calibration_sheet) if args.calibration_sheet else None,
        discard_generated=not args.keep_generated,
        custom_bar_instructions=args.custom_bar_instructions,
        cost_per_input_mtok=args.cost_per_input_mtok,
        cost_per_output_mtok=args.cost_per_output_mtok,
    )
    cfg.validate()

    records = list(_iter_jsonl(Path(args.input)))
    outcome = run_groundtruth(cfg=cfg, records=records)

    summary = {
        "stage": "groundtruth",
        "mode": cfg.mode,
        "input": args.input,
        "input_record_count": len(records),
        "counts": outcome.counts,
        "outputs": {
            "manifest": str(outcome.manifest_path),
            "verdicts": str(outcome.verdicts_path),
            "published": str(outcome.published_path),
            "discarded": str(outcome.discarded_path),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_pass_at_k(args) -> int:
    validate_mode(args.mode, args.calibration_sheet)
    cfg = PassAtKConfig(
        mode=args.mode,
        output_dir=Path(args.output_dir),
        backend=args.backend,
        model=args.model,
        k=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        think=args.think == "on",
        max_concurrent=args.max_concurrent,
        calibration_sheet=Path(args.calibration_sheet) if args.calibration_sheet else None,
        anthropic_key_file=Path(args.anthro_key_file) if args.anthro_key_file else None,
        openai_key_file=Path(args.openai_key_file) if args.openai_key_file else None,
        backend_url=args.backend_url,
        allow_live_calls=args.allow_live_calls,
        i_understand_paid_backend_is_off_policy=args.i_understand_paid_backend_is_off_policy,
        keep_garbage=args.keep_garbage,
        cost_per_input_mtok=args.cost_per_input_mtok,
        cost_per_output_mtok=args.cost_per_output_mtok,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        retry_max_delay=args.retry_max_delay,
    )
    # cfg.validate() runs inside run(), before any output is written.

    records = list(_iter_jsonl(Path(args.input)))
    outcome = run_pass_at_k(cfg=cfg, records=records)

    summary = {
        "stage": "pass_at_k",
        "mode": cfg.mode,
        "backend": cfg.backend,
        "model": cfg.resolved_model,
        "k": cfg.k,
        "input": args.input,
        "input_record_count": len(records),
        "counts": outcome.counts,
        "interrupted": outcome.interrupted,
        "resumed_records": outcome.resumed_records,
        "model_calls": outcome.model_calls,
        "outputs": {
            "manifest": str(outcome.manifest_path),
            "records": str(outcome.output_path),
        },
    }
    print(json.dumps(summary, indent=2))
    # Non-zero on a paused run so a chained pipeline never consumes the
    # partial output; re-running the same command resumes.
    return 1 if outcome.interrupted else 0


def _run_allocation_mount(args) -> int:
    column_map = _parse_columns(args.column)
    outcome = allocation_mount(
        path=Path(args.path),
        source=args.source,
        provenance=args.provenance,
        truth_policy=args.truth_policy,
        column_map=column_map,
        output_dir=Path(args.output_dir),
        family=args.family,
        requested_by=args.requested_by,
    )
    summary = {
        "stage": "allocation.mount",
        "run_id": outcome.run_id,
        "source": args.source,
        "provenance": args.provenance,
        "files_scanned": [str(s.path) for s in outcome.files_scanned],
        "files_skipped": [{"path": str(p), "reason": r} for p, r in outcome.files_skipped],
        "warnings": outcome.warnings,
        "record_count": outcome.record_count,
        "outputs": {
            "handoff": str(outcome.handoff_path),
            "manifest": str(outcome.manifest_path),
        },
        "next": (
            f"icepick processing pipeline --mode production --input {outcome.handoff_path} "
            f"--output-dir <out> --combo claude:anthropic --anthro-key-file <key>"
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


_SOURCE_TYPE_ADAPTERS = {
    SOURCE_REALMATH_SCRAPE: realmath_scrape,
}


def _run_allocation_plan(args) -> int:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    requested_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    adapter = _SOURCE_TYPE_ADAPTERS.get(args.source_type)
    if adapter is None:
        return _fail("E_UNSUPPORTED", "unsupported source type", args.source_type)

    request = {
        "source_name": args.source,
        "target_count": args.target_count,
        "requested_by": args.requested_by,
        "requested_at": requested_at,
    }
    if args.family:
        request["families"] = list(args.family)
    if args.notes:
        request["notes"] = args.notes
    if args.fixture_path:
        request["fixture_path"] = args.fixture_path

    scrape_window = _scrape_window_from_args(args)
    if scrape_window:
        request["scrape_window"] = scrape_window

    plan = adapter.plan(request)
    estimate = adapter.estimate(plan)

    output_dir = Path(args.output_dir)
    plans_dir = output_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_stamp = now.strftime("%Y%m%dT%H%M%SZ")
    plan_path = write_plan(plan, plans_dir)
    named_plan_path = plans_dir / f"{plan_stamp}_{args.source}_proposed_plan.json"
    if plan_path != named_plan_path:
        named_plan_path.write_text(plan_path.read_text())
        plan_path.unlink()
        plan_path = named_plan_path

    manifest_path: Optional[Path] = None
    manifest_dict: Optional[dict] = None
    if args.auto_approve:
        if args.mode is None:
            return _fail("E_CONFIG", "config invariant violated",
                         "--auto-approve requires --mode {flow_testing|production}")
        if args.mode == MODE_PRODUCTION:
            return _fail("E_CONFIG", "auto-approve refused",
                         "production auto-approval is refused; use --mode flow_testing or approve the plan manually")
        if not args.approved_by:
            return _fail("E_CONFIG", "config invariant violated",
                         "--auto-approve requires --approved-by")
        if args.mode == MODE_FLOW_TESTING and not args.calibration_sheet:
            return _fail("E_CONFIG", "config invariant violated",
                         "--mode flow_testing requires --calibration-sheet")

        run_id = new_run_id(now)
        manifest = ApprovedManifest(
            run_id=run_id,
            source_type=args.source_type,
            processor_mode=args.mode,
            requested_by=args.requested_by,
            requested_at=requested_at,
            approved_by=args.approved_by,
            approved_at=requested_at,
            source_name=args.source,
            target_count=args.target_count,
            call_budget=args.call_budget,
            judge_enabled=False,
            confirmation_enabled=False,
            enable_leakage=False,
            enable_duplication=False,
            enable_robustness=False,
            families=list(args.family or []),
            scrape_window=plan.scrape_window,
            truth_policy=None,
            output_dir=str(output_dir),
            calibration_sheet=args.calibration_sheet,
            approval_notes=args.approval_notes,
        )
        manifest_path = write_manifest(manifest, output_dir)
        manifest_dict = {
            "run_id": manifest.run_id,
            "processor_mode": manifest.processor_mode,
            "output_dir": str(output_dir),
        }

    summary = {
        "stage": "allocation.plan",
        "source_type": args.source_type,
        "source_name": args.source,
        "requested_by": args.requested_by,
        "requested_at": requested_at,
        "plan_path": str(plan_path),
        "estimate": estimate,
        "manifest": manifest_dict,
        "next": (
            f"icepick allocation run --manifest {manifest_path}"
            if manifest_path else
            "review the plan, produce an ApprovedManifest, then: icepick allocation run --manifest <path>"
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_allocation_run(args) -> int:
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    require_approved(manifest)

    adapter = _SOURCE_TYPE_ADAPTERS.get(manifest.source_type)
    if adapter is None:
        return _fail(
            "E_UNSUPPORTED", "unsupported source type",
            f"no adapter registered for source_type={manifest.source_type!r}",
        )

    outcome = adapter.run(manifest)

    interrupted = getattr(outcome, "interrupted", False)
    summary = {
        "stage": "allocation.run",
        "status": "interrupted_resumable" if interrupted else "complete",
        "run_id": outcome.run_id,
        "source_type": manifest.source_type,
        "source_name": manifest.source_name,
        "processor_mode": outcome.processor_mode,
        "calibration_replay": outcome.calibration_replay,
        "counts": {
            "papers": outcome.paper_count,
            "candidates": outcome.candidate_count,
            "duplicates_dropped": outcome.duplicates_dropped,
            "quarantined": outcome.quarantined_count,
            "handoff_records": outcome.record_count,
            "surplus_records": getattr(outcome, "surplus_count", 0),
        },
        "spend": outcome.acquisition,  # acquisition call counts vs budget (production)
        "outputs": {
            "handoff": str(outcome.handoff_path),
            "manifest": str(outcome.manifest_path),
            "report": str(outcome.report_path),
            "raw_dir": str(outcome.raw_dir),
            **(
                {"surplus": str(outcome.surplus_path)}
                if getattr(outcome, "surplus_path", None) else {}
            ),
        },
        "warnings": outcome.warnings,
        "next": (
            f"icepick allocation run --manifest {manifest_path}  # resume where it stopped"
            if interrupted else
            f"icepick processing pipeline --mode production --input {outcome.handoff_path} "
            f"--output-dir out --combo claude:anthropic --anthro-key-file <key>"
        ),
    }
    print(json.dumps(summary, indent=2))
    # Non-zero on a paused run so a chained pipeline never consumes the
    # partial handoff; the summary's "next" is the resume command.
    return 1 if interrupted else 0


def _run_allocation_approve(args) -> int:
    from datetime import datetime, timezone

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    source_type = plan.get("source_type")
    if source_type not in _SOURCE_TYPE_ADAPTERS:
        return _fail("E_UNSUPPORTED", "unsupported source type",
                     f"plan source_type={source_type!r} has no acquisition adapter")

    if args.mode == MODE_FLOW_TESTING and not args.calibration_sheet:
        return _fail("E_CONFIG", "config invariant violated",
                     "--mode flow_testing requires --calibration-sheet")

    estimated_calls = int(plan.get("estimated_calls") or 0)
    call_budget = args.call_budget
    if args.mode == MODE_PRODUCTION:
        if call_budget is None:
            return _fail("E_CONFIG", "config invariant violated",
                         f"--mode production requires --call-budget (plan estimates {estimated_calls} calls)")
        if call_budget < estimated_calls:
            return _fail("E_CONFIG", "call budget too low",
                         f"--call-budget {call_budget} is below the plan's estimated_calls {estimated_calls}")
    call_budget = call_budget if call_budget is not None else 0

    now = datetime.now(timezone.utc)
    run_id = new_run_id(now)
    approved_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    output_dir = Path(args.output_dir)
    manifest = ApprovedManifest(
        run_id=run_id,
        source_type=source_type,
        processor_mode=args.mode,
        requested_by=plan.get("requested_by", "cli"),
        requested_at=plan.get("requested_at", approved_at),
        approved_by=args.approved_by,
        approved_at=approved_at,
        source_name=plan.get("source_name", ""),
        target_count=int(plan.get("target_count") or 0),
        call_budget=call_budget,
        judge_enabled=False,
        confirmation_enabled=False,
        enable_leakage=False,
        enable_duplication=False,
        enable_robustness=False,
        families=list(plan.get("families") or []),
        scrape_window=plan.get("scrape_window"),
        output_dir=str(output_dir),
        calibration_sheet=args.calibration_sheet,
        approval_notes=args.approval_notes,
    )
    manifest_path = write_manifest(manifest, output_dir)

    summary = {
        "stage": "allocation.approve",
        "run_id": run_id,
        "source_type": source_type,
        "source_name": manifest.source_name,
        "processor_mode": manifest.processor_mode,
        "approved_by": manifest.approved_by,
        "call_budget": manifest.call_budget,
        "manifest": str(manifest_path),
        "next": f"icepick allocation run --manifest {manifest_path}",
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_validate_manifest(args) -> int:
    manifest = load_manifest(Path(args.manifest))
    try:
        require_approved(manifest)
        status = "approved"
    except ValueError as exc:
        status = f"NOT APPROVED: {exc}"
    summary = {
        "stage": "allocation.validate_manifest",
        "manifest": args.manifest,
        "run_id": manifest.run_id,
        "source_type": manifest.source_type,
        "source_name": manifest.source_name,
        "approved_by": manifest.approved_by,
        "approved_at": manifest.approved_at,
        "call_budget": manifest.call_budget,
        "requires_calls": manifest.requires_calls(),
        "status": status,
    }
    print(json.dumps(summary, indent=2))
    return 0 if manifest.is_approved() else 1


def _scrape_window_from_args(args) -> Optional[dict]:
    """Assemble the plan's ``scrape_window`` from the acquisition flags.

    Returns ``None`` when no window flag is set, so the plan records no
    window rather than an empty one. ``--primary-only`` is only meaningful
    alongside a subcategory (e.g. ``math.AP``); it is still recorded as
    intent and enforced at scrape time by the retriever.
    """
    window: dict = {}
    if args.category:
        window["category"] = args.category
    if args.year is not None:
        window["year"] = args.year
    if args.month is not None:
        window["month"] = args.month
    if args.max_papers is not None:
        window["max_papers"] = args.max_papers
    if args.max_per_paper is not None:
        window["max_per_paper"] = args.max_per_paper
    if args.primary_only:
        window["primary_only"] = True
    if args.extraction:
        window["extraction"] = args.extraction
    return window or None


def _parse_columns(column_args):
    """``--column canonical=source`` (repeatable) → ``{canonical: source}``."""
    if not column_args:
        return None
    out: dict = {}
    for spec in column_args:
        if "=" not in spec:
            raise ValueError(f"--column expects CANONICAL=SOURCE, got {spec!r}")
        canonical, _, source = spec.partition("=")
        canonical, source = canonical.strip(), source.strip()
        if not canonical or not source:
            raise ValueError(f"--column spec {spec!r} has empty side")
        out[canonical] = source
    return out


def _parse_combos(combo_args):
    """Translate ``--combo X:Y`` (repeatable) or ``--combo all`` into Combos.

    Refusing-to-guess: with no ``--combo`` provided at all, return an
    empty list so ``WellposedConfig.validate`` raises a clean message.
    """
    if not combo_args:
        return []
    if any(arg.strip().lower() == "all" for arg in combo_args):
        return all_combos()
    return [parse_combo(arg) for arg in combo_args]


def _parse_stage_list(stages_arg: str):
    """Turn ``'codex:openai,codex:anthropic,claude:openai?advisory'`` into StageSpecs."""
    if not stages_arg or not stages_arg.strip():
        raise ValueError("--stages must not be empty")
    specs = [tok.strip() for tok in stages_arg.split(",") if tok.strip()]
    if not specs:
        raise ValueError("--stages must not be empty")
    return parse_stages(specs)


def _iter_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _stub_handler(stage_label: str):
    def _handler(args) -> int:
        validate_mode(getattr(args, "mode", None), getattr(args, "calibration_sheet", None))
        raise NotImplementedError(
            f"{stage_label} is registered but not yet implemented; "
            f"only 'processing ingest-check' is wired in this build"
        )

    return _handler


def _fail(code: str, label: str, detail: str) -> int:
    print(f"{code} {label}: {detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

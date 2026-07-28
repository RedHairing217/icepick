"""CLI glue for the bulk-batcher subsystem.

Wires ``icepick batcher <subcommand>`` into the argparse tree from cli.py.
Follows the same add_parser + set_defaults(_handler=...) pattern as the rest
of cli.py.

Subcommands:
  batcher run       — start daemon (or --once for single-tick test)
  batcher status    — print JSON summary to stdout
  batcher backfill  — seed ledger with historical sources
  batcher arm       — write ARMED flag (requires --i-approve-recurring-spend)
  batcher disarm    — remove ARMED flag
  batcher clear-halt — clear queue halt in queue_state.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def build_batcher_parser(sub) -> None:
    """Register the 'batcher' subcommand group onto an argparse subparsers object.

    Called from cli.py's build_parser() as _build_batcher(sub).
    """
    g = sub.add_parser(
        "batcher",
        help="Auto-batcher daemon: slice → mount → cascade → pass@k.",
    )
    s = g.add_subparsers(dest="batcher_cmd", metavar="<command>")

    _add_run(s)
    _add_status(s)
    _add_backfill(s)
    _add_arm(s)
    _add_disarm(s)
    _add_clear_halt(s)


# ---------------------------------------------------------------------------
# Common argument adder
# ---------------------------------------------------------------------------


def _add_root_arg(p) -> None:
    p.add_argument(
        "--root",
        default="out/auto_batcher",
        help="Batcher state root directory (default: out/auto_batcher)",
    )


# ---------------------------------------------------------------------------
# batcher run
# ---------------------------------------------------------------------------


def _add_run(s) -> None:
    p = s.add_parser(
        "run",
        help="Start the batcher daemon (or --once for single-tick test).",
    )
    _add_root_arg(p)

    p.add_argument(
        "--journal",
        required=True,
        help="Path to candidates.jsonl to tail.",
    )
    p.add_argument(
        "--run-dir",
        required=True,
        help="The extraction run directory that owns the journal.",
    )
    p.add_argument(
        "--campaign-source",
        required=True,
        help="IDENTITY-CRITICAL: source name for uid computation (e.g. arxiv_bulk_pde625).",
    )
    p.add_argument(
        "--cross-source-statement-policy",
        default="skip",
        choices=("skip", "abort", "allow"),
        help="Policy for cross-source statement collisions (default: skip).",
    )
    p.add_argument(
        "--slice-size",
        type=int,
        default=250,
        help="IDENTITY-CRITICAL: records per batch (default: 250).",
    )
    p.add_argument(
        "--cost-limit-usd",
        type=float,
        default=5.0,
        help="Per-batch cascade cost guard in USD (default: 5.0).",
    )
    p.add_argument(
        "--key-path",
        default="/Users/redhairing/Desktop/helloworld/anthro_key.env",
        help="Path to Anthropic key env file (opaque string, never opened here).",
    )
    p.add_argument(
        "--mode",
        default="production",
        choices=("production", "flow_testing"),
        help="Pipeline mode (default: production).",
    )
    p.add_argument(
        "--calibration-sheet",
        default=None,
        help="Calibration sheet path (required for flow_testing).",
    )
    p.add_argument(
        "--icepick-bin",
        default="icepick",
        help="icepick binary name/path (default: icepick).",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        dest="poll_interval_s",
        help="Seconds between ticks (default: 60).",
    )
    p.add_argument(
        "--qwen-recheck-interval",
        type=int,
        default=45,
        dest="qwen_recheck_interval_s",
        help="Seconds between Qwen slot rechecks (default: 45).",
    )
    p.add_argument(
        "--watch-journal",
        action="append",
        metavar="LABEL=RUN_DIR",
        dest="watch_journals_raw",
        default=[],
        help=(
            "Repeatable: LABEL=RUN_DIR. Journal path derived as <run_dir>/_progress/candidates.jsonl. "
            "Rows are ingested into the ledger only (never auto-batched)."
        ),
    )
    p.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Startup + single tick + status write, then exit (ops/test entrypoint).",
    )

    p.set_defaults(_handler=_handle_run)


def _parse_watch_journals(raw_list: list) -> list:
    """Parse ['LABEL=RUN_DIR', ...] into [{label, journal_path, run_dir}]."""
    result = []
    for item in raw_list:
        if "=" not in item:
            raise ValueError(f"--watch-journal must be LABEL=RUN_DIR, got: {item!r}")
        label, run_dir_str = item.split("=", 1)
        run_dir = Path(run_dir_str)
        journal_path = run_dir / "_progress" / "candidates.jsonl"
        result.append({
            "label": label,
            "journal_path": str(journal_path),
            "run_dir": str(run_dir),
        })
    return result


def _handle_run(args) -> int:
    from icepick.batcher.config import BatcherConfig
    from icepick.batcher.daemon import BatcherDaemon

    watch_journals = _parse_watch_journals(getattr(args, "watch_journals_raw", []))

    config = BatcherConfig(
        root=Path(args.root),
        journal_path=Path(args.journal),
        run_dir=Path(args.run_dir),
        campaign_source=args.campaign_source,
        cross_source_statement_policy=args.cross_source_statement_policy,
        slice_size=args.slice_size,
        cost_limit_usd=args.cost_limit_usd,
        key_path=args.key_path,
        mode=args.mode,
        calibration_sheet=args.calibration_sheet,
        icepick_bin=args.icepick_bin,
        poll_interval_s=args.poll_interval_s,
        qwen_recheck_interval_s=args.qwen_recheck_interval_s,
        watch_journals=watch_journals,
    )

    root = Path(args.root)

    if not (root / "ARMED").exists():
        print(
            "Batcher is DISARMED. To arm:\n"
            f"  icepick batcher arm --i-approve-recurring-spend --root {args.root}\n"
            "This authorizes recurring Sonnet spend (~$2.30/batch).",
            file=sys.stderr,
        )
        return 0

    daemon = BatcherDaemon(config)

    try:
        report = daemon.startup()
    except RuntimeError as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    if args.once:
        tag = daemon.tick()
        from icepick.batcher.state import load_all_states
        from icepick.batcher.status import write_status, render_status
        batch_states = load_all_states(root / "batches")
        qs = json.loads((root / "queue_state.json").read_text()) if (root / "queue_state.json").exists() else {}
        content = render_status(qs, batch_states, {"armed": True, "pid": os.getpid()})
        write_status(root, content)
        print(f"tick: {tag}")
        return 0

    daemon.run_forever()
    return 0


# ---------------------------------------------------------------------------
# batcher status
# ---------------------------------------------------------------------------


def _add_status(s) -> None:
    p = s.add_parser(
        "status",
        help="Print JSON summary to stdout (queue_state + batch states + STATUS.md path).",
    )
    _add_root_arg(p)
    p.set_defaults(_handler=_handle_status)


def _handle_status(args) -> int:
    from icepick.batcher.state import load_all_states

    root = Path(args.root)
    qs_path = root / "queue_state.json"
    qs = {}
    if qs_path.exists():
        try:
            qs = json.loads(qs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: could not read queue_state.json: {exc}", file=sys.stderr)

    batch_states = load_all_states(root / "batches")
    status_path = root / "STATUS.md"

    summary = {
        "queue_state": qs,
        "batch_states": batch_states,
        "status_md_path": str(status_path),
        "status_md_exists": status_path.exists(),
        "armed": (root / "ARMED").exists(),
    }
    print(json.dumps(summary, indent=2))
    return 0


# ---------------------------------------------------------------------------
# batcher backfill
# ---------------------------------------------------------------------------


def _add_backfill(s) -> None:
    p = s.add_parser(
        "backfill",
        help="Seed ledger with historical source files (works DISARMED).",
    )
    _add_root_arg(p)
    p.add_argument(
        "--sources-json",
        default=None,
        help="Override path to sources JSON file (default: packaged backfill_sources.json).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report what would be backfilled without writing.",
    )
    p.set_defaults(_handler=_handle_backfill)


def _handle_backfill(args) -> int:
    from icepick.batcher.ledger import Ledger
    from icepick.batcher.backfill import load_sources, backfill

    root = Path(args.root)
    ledger_dir = root / "ledger"

    sources_path = Path(args.sources_json) if args.sources_json else None
    sources = load_sources(sources_path)

    if args.dry_run:
        summary = {
            "dry_run": True,
            "sources": [s["label"] for s in sources],
            "count": len(sources),
        }
        print(json.dumps(summary, indent=2))
        return 0

    ledger = Ledger.load(ledger_dir)
    now_iso = datetime.now(timezone.utc).isoformat()
    # Repo root heuristic: go up from this file's location.
    repo_root = Path(__file__).parent.parent.parent.parent

    result = backfill(ledger, sources, repo_root, now_iso)
    print(json.dumps({"backfill": result}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# batcher arm
# ---------------------------------------------------------------------------


def _add_arm(s) -> None:
    p = s.add_parser(
        "arm",
        help="Write ARMED flag to authorize recurring spend.",
    )
    _add_root_arg(p)
    p.add_argument(
        "--i-approve-recurring-spend",
        action="store_true",
        default=False,
        required=True,
        help="REQUIRED: explicit acknowledgment that this authorizes ~$2.30/batch Sonnet spend.",
    )
    p.set_defaults(_handler=_handle_arm)


def _handle_arm(args) -> int:
    if not getattr(args, "i_approve_recurring_spend", False):
        print(
            "Error: --i-approve-recurring-spend is required to arm the batcher.",
            file=sys.stderr,
        )
        return 1

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    armed_path = root / "ARMED"
    now_iso = datetime.now(timezone.utc).isoformat()
    armed_data = {
        "armed_at": now_iso,
        "by": "cli",
        "note": "Authorized by operator via --i-approve-recurring-spend",
    }
    armed_path.write_text(json.dumps(armed_data, indent=2), encoding="utf-8")
    print(
        f"WARNING: Batcher ARMED at {armed_path}\n"
        f"This authorizes recurring Sonnet spend (~$2.30/batch).\n"
        f"To disarm: icepick batcher disarm --root {args.root}"
    )
    return 0


# ---------------------------------------------------------------------------
# batcher disarm
# ---------------------------------------------------------------------------


def _add_disarm(s) -> None:
    p = s.add_parser(
        "disarm",
        help="Remove ARMED flag (daemon exits at next tick boundary).",
    )
    _add_root_arg(p)
    p.set_defaults(_handler=_handle_disarm)


def _handle_disarm(args) -> int:
    root = Path(args.root)
    # Removing ARMED is the single sanctioned file deletion in this system.
    # It is a control flag in our own mutable-state root (not run history),
    # so deletion is the correct operation — it signals the daemon to exit
    # gracefully at the next tick boundary without killing any in-flight stage.
    armed_path = root / "ARMED"
    if not armed_path.exists():
        print("Batcher is already DISARMED.")
        return 0
    armed_path.unlink()
    print(
        f"Batcher DISARMED. Running daemon will exit at next tick boundary.\n"
        f"In-flight stages will complete safely before exit."
    )
    return 0


# ---------------------------------------------------------------------------
# batcher clear-halt
# ---------------------------------------------------------------------------


def _add_clear_halt(s) -> None:
    p = s.add_parser(
        "clear-halt",
        help="Clear queue halt in queue_state.json.",
    )
    _add_root_arg(p)
    p.add_argument(
        "--reason",
        required=True,
        help="Human-readable reason for clearing the halt (logged to events.jsonl).",
    )
    p.set_defaults(_handler=_handle_clear_halt)


def _handle_clear_halt(args) -> int:
    import os

    root = Path(args.root)
    qs_path = root / "queue_state.json"
    events_path = root / "events.jsonl"

    if not qs_path.exists():
        print(f"No queue_state.json found at {qs_path}", file=sys.stderr)
        return 1

    try:
        qs = json.loads(qs_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Could not read queue_state.json: {exc}", file=sys.stderr)
        return 1

    prev_halt = qs.get("halt") or {}
    if not prev_halt.get("active"):
        print("Queue is not halted.")
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    qs["halt"] = {"active": False, "reason": "", "at": now_iso}
    qs["updated_at"] = now_iso

    tmp = qs_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(qs, indent=2), encoding="utf-8")
    os.replace(tmp, qs_path)

    # Append event.
    event = {
        "at": now_iso,
        "kind": "halt_cleared",
        "by": "cli",
        "reason": args.reason,
        "prev_halt": prev_halt,
    }
    root.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
        fh.flush()

    print(f"Halt cleared. Reason: {args.reason}")
    return 0

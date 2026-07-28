"""STATUS.md renderer and writer for the bulk-batcher subsystem.

render_status(queue_state, batch_states, extras) -> str (markdown)
write_status(root, content) — atomic tmp+os.replace write

Sections:
  1. Header — ARMED/DISARMED, halt, pid, updated_at, spend vs guard
  2. Config one-liner — campaign_source, journal, policy
  3. Batch table — name | state | records | cascade $ | passk counts | updated
  4. In-flight/frozen detail with operator hints
  5. HELD remainder
  6. Skip/warn/replay tallies + cursor-reset note
  7. Watch-journal counters
  8. Last 10 events (from extras or events_log)

Deterministic ordering: batches sorted by batch number.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Operator hint map: state → next action hint
# ---------------------------------------------------------------------------

_FROZEN_HINTS: dict = {
    "mount_dirty": "inspect intake/runs/<ts>/ and MOUNT_VERIFIED marker; if ok, delete marker to re-verify or clear freeze manually",
    "mount_verification_failed": "inspect batch intake dir; correct records and re-run mount manually or clear freeze",
    "exec_failed": "check batch log; restart daemon to retry or clear freeze",
    "cost_guard_tripped": "check cascade_manifest.json cost field; if ok adjust cost_limit_usd or clear freeze + clear queue halt",
    "retries_exhausted": "resolve transient errors (429/529/timeout) then clear freeze",
    "passk_interrupted": "resolve pass@k interruption then clear freeze",
    "slice_abort": "inspect events.jsonl for abort_info; resolve uid conflict then clear queue halt",
    "uid_conflict": "inspect abort_info in events.jsonl; deduplicate source data then clear queue halt",
    "default": "edit state.json to clear FROZEN; set state to last known good state then restart daemon",
}


def _frozen_hint(reason: str) -> str:
    for key, hint in _FROZEN_HINTS.items():
        if key in reason:
            return hint
    return _FROZEN_HINTS["default"]


# ---------------------------------------------------------------------------
# render_status
# ---------------------------------------------------------------------------


def render_status(
    queue_state: dict,
    batch_states: dict,
    extras: Optional[dict] = None,
) -> str:
    """Render a STATUS.md string from the given data.

    Parameters
    ----------
    queue_state : dict
        The full queue_state.json payload.
    batch_states : dict
        {batch_name: state_dict} from load_all_states().
    extras : dict, optional
        Supplementary data from the daemon tick:
          - armed: bool
          - pid: int
          - in_flight: str | None
          - held_remainder: {count, uids} | None
          - skip_counts: {replay, stmt, warns}
          - watch_counters: {label: {ingested, malformed}}
          - events_log: list of last-N event lines (raw strings)
          - cursor_reset: bool
    """
    if extras is None:
        extras = {}

    lines: list[str] = []

    # -------------------------------------------------------------------
    # 1. Header
    # -------------------------------------------------------------------
    armed = extras.get("armed", False)
    armed_str = "ARMED" if armed else "DISARMED"
    lines.append(f"# Batcher STATUS — {armed_str}")
    lines.append("")

    halt = queue_state.get("halt") or {}
    if halt.get("active"):
        lines.append(f"> **QUEUE HALTED** — {halt.get('reason', '(no reason)')}")
        lines.append(f"> Halted at: {halt.get('at', '?')}")
        lines.append("")

    pid = extras.get("pid")
    if pid:
        lines.append(f"- PID: {pid}")

    updated_at = queue_state.get("updated_at", "?")
    lines.append(f"- Updated: {updated_at}")

    spend = queue_state.get("spend_usd_total", 0.0)
    cost_limit = (queue_state.get("config") or {}).get("cost_limit_usd", 5.0)
    lines.append(f"- Spend total: ${spend:.4f} (per-batch guard: ${cost_limit:.2f})")

    next_batch = queue_state.get("next_batch_number", "?")
    lines.append(f"- Next batch number: {next_batch}")
    lines.append("")

    # -------------------------------------------------------------------
    # 2. Config one-liner
    # -------------------------------------------------------------------
    cfg = queue_state.get("config") or {}
    lines.append("## Config")
    lines.append(
        f"- campaign_source: `{cfg.get('campaign_source', '?')}`"
    )
    lines.append(f"- journal: `{cfg.get('journal_path', '?')}`")
    lines.append(f"- cross_source_statement_policy: `{cfg.get('cross_source_statement_policy', '?')}`")
    lines.append(f"- mode: `{cfg.get('mode', '?')}`")
    lines.append("")

    # -------------------------------------------------------------------
    # 3. Batch table
    # -------------------------------------------------------------------
    lines.append("## Batches")
    if not batch_states:
        lines.append("(no batches yet)")
    else:
        lines.append("| batch | state | records | cascade $ | passk counts | updated |")
        lines.append("|---|---|---|---|---|---|")

        def _batch_sort_key(name: str) -> int:
            try:
                return int(name[len("batch"):])
            except (ValueError, IndexError):
                return 0

        for name in sorted(batch_states.keys(), key=_batch_sort_key):
            st = batch_states[name]
            state = st.get("state", "?")
            # records
            counts = st.get("counts", {})
            records = counts.get("accepted", "?") if counts else "?"
            # cascade cost
            cascade_data = st.get("cascade_data") or {}
            cost = cascade_data.get("cost_usd")
            cost_str = f"${cost:.4f}" if cost is not None else "-"
            # passk counts
            passk_counts = st.get("passk_counts") or {}
            if passk_counts:
                passk_str = json.dumps(passk_counts)
            else:
                passk_str = "-"
            upd = st.get("updated_at", "?")
            lines.append(f"| {name} | {state} | {records} | {cost_str} | {passk_str} | {upd} |")

    lines.append("")

    # -------------------------------------------------------------------
    # 4. In-flight / frozen detail
    # -------------------------------------------------------------------
    frozen_batches = [
        (n, st) for n, st in batch_states.items() if st.get("state") == "FROZEN"
    ]
    in_flight = extras.get("in_flight")

    if in_flight or frozen_batches:
        lines.append("## Detail")

    if in_flight:
        in_flight_st = batch_states.get(in_flight, {})
        lines.append(f"### In-flight: {in_flight}")
        lines.append(f"- State: {in_flight_st.get('state', '?')}")
        lines.append("")

    for name, st in sorted(frozen_batches, key=lambda x: x[0]):
        frozen_info = st.get("frozen", {})
        reason = frozen_info.get("reason", "(unknown)")
        from_state = frozen_info.get("from_state", "?")
        at = frozen_info.get("at", "?")
        hint = _frozen_hint(reason)
        lines.append(f"### FROZEN: {name}")
        lines.append(f"- From state: {from_state}")
        lines.append(f"- Reason: {reason}")
        lines.append(f"- At: {at}")
        lines.append(f"- **To resume**: {hint}")
        lines.append("")

    # -------------------------------------------------------------------
    # 5. HELD remainder
    # -------------------------------------------------------------------
    held = extras.get("held_remainder")
    if held:
        lines.append("## HELD Remainder")
        lines.append(
            f"Run concluded with {held.get('count', '?')} records below "
            f"slice_size — these will NEVER be auto-batched."
        )
        uids = held.get("uids", [])
        if uids:
            lines.append("UIDs:")
            for uid in uids[:20]:
                lines.append(f"  - {uid}")
            if len(uids) > 20:
                lines.append(f"  ... and {len(uids) - 20} more")
        lines.append("")

    # -------------------------------------------------------------------
    # 6. Skip/warn tallies + cursor-reset note
    # -------------------------------------------------------------------
    skip_counts = extras.get("skip_counts", {})
    if skip_counts:
        lines.append("## Skip / Warn Tallies")
        lines.append(f"- Replay skips (byte-identical): {skip_counts.get('replay', 0)}")
        lines.append(f"- Statement skips (cross-source): {skip_counts.get('stmt', 0)}")
        lines.append(f"- Warns (warn-set / allowed): {skip_counts.get('warns', 0)}")
        if extras.get("cursor_reset"):
            lines.append(
                "- **NOTE**: CursorStore was reset on this run (corrupt cursor.json recovered from ledger)"
            )
        lines.append("")

    # -------------------------------------------------------------------
    # 7. Watch-journal counters
    # -------------------------------------------------------------------
    watch_counters = extras.get("watch_counters", {})
    if watch_counters:
        lines.append("## Watch Journals")
        for label, wc in sorted(watch_counters.items()):
            ingested = wc.get("ingested", 0)
            malformed = wc.get("malformed", 0)
            lines.append(f"- {label}: {ingested} ingested, {malformed} malformed")
        lines.append("")

    # -------------------------------------------------------------------
    # 8. Last 10 events
    # -------------------------------------------------------------------
    events = extras.get("events_log", [])
    if events:
        lines.append("## Recent Events (last 10)")
        for ev in events[-10:]:
            lines.append(f"- {ev.strip()}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# write_status
# ---------------------------------------------------------------------------


def write_status(root: Path, content: str) -> None:
    """Atomically write STATUS.md under root via tmp+os.replace."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "STATUS.md"
    tmp_path = status_path.with_suffix(".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, status_path)

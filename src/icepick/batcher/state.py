"""Batch state machine persisted at <batch_dir>/state.json.

States (linear order):
  SLICED → MOUNTED → CASCADE_DONE → PASSK_DONE → READY

Plus FROZEN (terminal until a human edits/clears it): reachable from any
non-READY state.  On READY, a READY_TO_FOLD flag file is also written.

Crash safety: all writes use tmp+os.replace (atomic on POSIX).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATES_LINEAR = ["SLICED", "MOUNTED", "CASCADE_DONE", "PASSK_DONE", "READY"]
STATES_ALL = set(STATES_LINEAR) | {"FROZEN"}

# Map each state to its 0-based position in the linear order.
_STATE_IDX = {s: i for i, s in enumerate(STATES_LINEAR)}

_STATE_FILENAME = "state.json"
_READY_FLAG = "READY_TO_FOLD"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _now_iso_default() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_state(batch_dir: Path) -> dict:
    """Read and parse state.json from batch_dir. Returns {} if missing."""
    p = batch_dir / _STATE_FILENAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_state(batch_dir: Path, data: dict) -> None:
    """Atomically write state.json via tmp+os.replace."""
    p = batch_dir / _STATE_FILENAME
    tmp = p.with_suffix(".tmp")
    batch_dir.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# transition
# ---------------------------------------------------------------------------


def transition(
    batch_dir: Path,
    to: str,
    note: str = "",
    now_iso: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Transition batch_dir to state `to`, writing state.json atomically.

    Rules:
    - FROZEN is reachable from any non-READY state (write reason/from_state).
    - Linear states must advance in order (no skips, no backwards moves).
    - Already in the target state is a no-op (returns current state).
    - READY state also writes the READY_TO_FOLD flag file.

    Parameters
    ----------
    batch_dir : Path
        The batch directory containing (or to contain) state.json.
    to : str
        The target state name.
    note : str
        Human-readable note appended to history.
    now_iso : str, optional
        ISO timestamp; defaults to now() in UTC.
    extra : dict, optional
        Additional fields to merge into the top-level state dict (e.g.
        mount run_dir, cascade cost, passk counts).

    Returns
    -------
    dict
        The new state dict.

    Raises
    ------
    ValueError
        If the transition is illegal.
    """
    if to not in STATES_ALL:
        raise ValueError(f"Unknown state: {to!r}. Valid: {sorted(STATES_ALL)}")

    if now_iso is None:
        now_iso = _now_iso_default()

    data = _read_state(batch_dir)
    current = data.get("state")

    # --- no-op guard ---
    if current == to:
        return data

    # --- validate transition ---
    if to == "FROZEN":
        if current == "READY":
            raise ValueError("Cannot FREEZE a READY batch.")
    else:
        # Must advance linearly.
        if current is not None and current != "FROZEN":
            cur_idx = _STATE_IDX.get(current, -1)
            to_idx = _STATE_IDX.get(to, -1)
            if to_idx != cur_idx + 1:
                raise ValueError(
                    f"Illegal state transition: {current!r} → {to!r}. "
                    f"Expected next: {STATES_LINEAR[cur_idx + 1]!r}"
                )
        elif current is None:
            # First transition: must be to SLICED
            if to != "SLICED":
                raise ValueError(
                    f"First transition must be to 'SLICED', got {to!r}"
                )

    # --- build new state ---
    history = data.get("history", [])
    history.append({"from": current, "to": to, "at": now_iso, "note": note})

    data["state"] = to
    data["updated_at"] = now_iso
    data["history"] = history

    if "created_at" not in data:
        data["created_at"] = now_iso

    if to == "FROZEN":
        data["frozen"] = {
            "reason": note,
            "from_state": current,
            "at": now_iso,
        }

    if extra:
        data.update(extra)

    _write_state(batch_dir, data)

    # --- READY_TO_FOLD flag file ---
    if to == "READY":
        flag_path = batch_dir / _READY_FLAG
        batch_name = batch_dir.name
        # Count records from slice_manifest if available.
        counts: dict = {}
        manifest_path = batch_dir / "slice_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                counts = manifest.get("counts", {})
            except (json.JSONDecodeError, OSError):
                pass
        flag_content = json.dumps(
            {"batch": batch_name, "counts": counts, "at": now_iso},
            indent=2,
        )
        flag_path.write_text(flag_content, encoding="utf-8")

    return data


# ---------------------------------------------------------------------------
# load_state / load_all_states
# ---------------------------------------------------------------------------


def load_state(batch_dir: Path) -> dict:
    """Return the state dict for a single batch dir (or {} if missing)."""
    return _read_state(batch_dir)


def load_all_states(batches_root: Path) -> dict:
    """Return {batch_name: state_dict} for every batch<N> dir under batches_root.

    Sorted by batch number (ascending).
    """
    result: dict[str, dict] = {}
    if not batches_root.exists():
        return result

    dirs = []
    for d in batches_root.iterdir():
        if d.is_dir() and d.name.startswith("batch"):
            try:
                n = int(d.name[len("batch"):])
                dirs.append((n, d))
            except ValueError:
                continue

    dirs.sort(key=lambda x: x[0])
    for _, d in dirs:
        result[d.name] = _read_state(d)

    return result

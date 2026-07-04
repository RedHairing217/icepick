"""Read, write, and validate manifests.

A manifest is approved iff ``approved_by`` and ``approved_at`` are both
non-empty. Acquisition CLIs that spend calls refuse to run if either is
missing; manual mounts auto-approve (no calls to authorize) and stamp
``approved_by = requested_by``.

Manifest layout on disk:

    <output_dir>/runs/<run_id>/manifest.json
    <output_dir>/runs/<run_id>/handoff/records.jsonl    (mount only)

``run_id`` is a timestamp (UTC, second precision) — see ``new_run_id``.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from icepick.contracts.manifests import ApprovedManifest, ProposedPlan


def new_run_id(now: datetime = None) -> str:
    """Stable, lexicographically-sortable run id. UTC second precision."""
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def write_plan(plan: ProposedPlan, output_dir) -> Path:
    """Write a ProposedPlan to disk under output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "proposed_plan.json"
    plan_path.write_text(json.dumps(_to_dict(plan), indent=2))
    return plan_path


def write_manifest(manifest: ApprovedManifest, output_dir) -> Path:
    """Write an ApprovedManifest under output_dir/runs/<run_id>/manifest.json.

    Uses ``manifest.run_id`` as the directory name so the layout matches
    the on-disk convention every consumer expects.
    """
    output_dir = Path(output_dir)
    run_dir = output_dir / "runs" / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_to_dict(manifest), indent=2))
    return manifest_path


def load_manifest(path) -> ApprovedManifest:
    """Load an ApprovedManifest from disk. Raises on unknown fields."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid_fields = {f.name for f in dataclasses.fields(ApprovedManifest)}
    extra = set(payload.keys()) - valid_fields
    if extra:
        raise ValueError(f"{path}: unknown manifest fields: {sorted(extra)}")
    return ApprovedManifest(**{k: payload[k] for k in payload if k in valid_fields})


def require_approved(manifest: ApprovedManifest) -> None:
    """Raise if not approved. Centralised so every caller fails the same way."""
    if not manifest.is_approved():
        raise ValueError(
            "manifest is not approved; approved_by and approved_at must be set"
        )


def _to_dict(obj) -> dict:
    """Dataclass → JSON-safe dict (handles nested defaults + None)."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return dict(obj)

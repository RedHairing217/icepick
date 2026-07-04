"""Intake entry points: ``mount``, ``run``, ``plan``.

Only ``mount`` is implemented here — it's the path that produces input
for the processing pipeline without requiring acquisition. Acquisition
``plan`` and ``run`` are driven from the CLI through each source's
adapter (see ``icepick.cli`` and ``allocation/adapters/``); the
module-level ``plan`` / ``run`` below stay stubbed.

``mount`` is the operator-facing one-shot that combines:

  1. ``manual_mount.mount(...)`` — scan the source dir, write canonical
                                   handoff JSONL.
  2. ``manifests.write_manifest(...)`` — record what was mounted, with
                                         auto-approval (call_budget=0).

The output layout:

    <output_dir>/runs/<run_id>/
        manifest.json
        handoff/records.jsonl

The handoff JSONL is the file the pipeline's ``--input`` should point at.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from icepick.allocation.adapters import manual_mount as manual_mount_adapter
from icepick.allocation.manifests import new_run_id, write_manifest
from icepick.contracts.manifests import ApprovedManifest, SOURCE_MANUAL_MOUNT


@dataclass
class MountOutcome:
    """What ``mount`` returns. ``handoff_path`` is what to feed the pipeline."""

    handoff_path: Path
    manifest_path: Path
    run_id: str
    record_count: int
    files_scanned: list
    files_skipped: list
    warnings: list


def mount(
    *,
    path,
    source: str,
    provenance: str,
    requested_by: str,
    truth_policy: str = "unknown",
    column_map: Optional[dict] = None,
    output_dir,
    family: Optional[str] = None,
    now: Optional[datetime] = None,
) -> MountOutcome:
    """One-shot: scan a mount path, write handoff JSONL + manifest.

    Auto-approves the manifest (mounts spend no calls) by setting
    ``approved_by = requested_by``. The handoff JSONL lives under the
    run directory alongside the manifest; the caller passes
    ``out.handoff_path`` to ``processing pipeline --input``.
    """
    now = now or datetime.now(timezone.utc)
    run_id = new_run_id(now)
    output_dir = Path(output_dir)
    run_dir = output_dir / "runs" / run_id

    mount_result = manual_mount_adapter.mount(
        path=path,
        source=source,
        provenance=provenance,
        truth_policy=truth_policy,
        column_map=column_map,
        output_dir=run_dir,
        family=family,
    )

    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = ApprovedManifest(
        run_id=run_id,
        source_type=SOURCE_MANUAL_MOUNT,
        processor_mode="production",
        requested_by=requested_by,
        requested_at=ts,
        approved_by=requested_by,
        approved_at=ts,
        source_name=source,
        target_count=mount_result.record_count,
        call_budget=0,
        judge_enabled=False,
        confirmation_enabled=False,
        enable_leakage=False,
        enable_duplication=False,
        enable_robustness=False,
        column_map=column_map,
        truth_policy=truth_policy,
        output_dir=str(output_dir),
        approval_notes="auto-approved: manual mount spends no calls",
    )
    manifest_path = write_manifest(manifest, output_dir)

    return MountOutcome(
        handoff_path=mount_result.records_path,
        manifest_path=manifest_path,
        run_id=run_id,
        record_count=mount_result.record_count,
        files_scanned=mount_result.files_scanned,
        files_skipped=mount_result.files_skipped,
        warnings=mount_result.warnings,
    )


def plan(*, source: str, args: dict, output_dir):
    raise NotImplementedError("allocation.intake.plan is not yet implemented")


def run(*, manifest_path):
    raise NotImplementedError("allocation.intake.run is not yet implemented")

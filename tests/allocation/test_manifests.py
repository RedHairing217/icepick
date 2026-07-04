"""Manifest approval contract."""

from __future__ import annotations

import pytest

from icepick.allocation.manifests import require_approved
from icepick.contracts.manifests import ApprovedManifest


def _manifest(**overrides):
    base = dict(
        run_id="r1",
        source_type="external_jsonl",
        processor_mode="production",
        requested_by="alice",
        requested_at="2026-06-29T00:00:00Z",
        approved_by="",
        approved_at="",
        source_name="external_drop",
        target_count=100,
        call_budget=0,
        judge_enabled=False,
        confirmation_enabled=False,
        enable_leakage=False,
        enable_duplication=True,
        enable_robustness=False,
    )
    base.update(overrides)
    return ApprovedManifest(**base)


def test_unapproved_manifest_is_rejected():
    with pytest.raises(ValueError):
        require_approved(_manifest())


def test_approved_manifest_passes():
    require_approved(
        _manifest(approved_by="bob", approved_at="2026-06-29T01:00:00Z")
    )


def test_requires_calls_true_for_acquisition_sources():
    m = _manifest(source_type="generated", call_budget=200)
    assert m.requires_calls() is True


def test_requires_calls_false_for_mount_or_external():
    assert _manifest(source_type="manual_mount").requires_calls() is False
    assert _manifest(source_type="external_jsonl").requires_calls() is False

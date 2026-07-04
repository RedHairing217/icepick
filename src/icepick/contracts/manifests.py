"""Manifest schemas — allocation's output and processing's input.

Manifests are form-like: one clear field per decision, no free-text where
an enum will do, dangerous and costly choices grouped together. They are
the only handoff between allocation and processing besides the JSONL
record streams themselves.

Two kinds:

- ``ProposedPlan``: allocation's first emission. Written before any approval
  and before any calls. A planner may write many of these.
- ``ApprovedManifest``: immutable, signed off by a human. Records all
  acquisition parameters and is the only artifact that authorises
  generation, scraping, or external-call stages to run.

Manual mounts also write an ``ApprovedManifest`` with ``source_type =
manual_mount`` and ``call_budget = 0`` because no calls are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SOURCE_GENERATED = "generated"
SOURCE_REALMATH_SCRAPE = "realmath_scrape"
SOURCE_EXTERNAL_JSONL = "external_jsonl"
SOURCE_MANUAL_MOUNT = "manual_mount"
SOURCE_TYPES = (
    SOURCE_GENERATED,
    SOURCE_REALMATH_SCRAPE,
    SOURCE_EXTERNAL_JSONL,
    SOURCE_MANUAL_MOUNT,
)

MODE_PRODUCTION = "production"
MODE_FLOW_TESTING = "flow_testing"
MODE_VALUES = (MODE_PRODUCTION, MODE_FLOW_TESTING)


@dataclass
class ProposedPlan:
    """First-pass plan; not yet approved, no calls allowed."""

    source_type: str
    requested_by: str
    requested_at: str
    source_name: str
    target_count: int
    notes: str = ""
    families: list = field(default_factory=list)
    scrape_window: Optional[dict] = None
    estimated_calls: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


@dataclass
class ApprovedManifest:
    """Immutable manifest. Required by acquisition runs and by processing.

    The approval fields (``approved_by``, ``approved_at``) MUST be present
    and non-empty for any command that spends calls or scrapes externally.
    Refuse ambiguous forms instead of guessing.
    """

    run_id: str
    source_type: str
    processor_mode: str
    requested_by: str
    requested_at: str
    approved_by: str
    approved_at: str
    source_name: str
    target_count: int
    call_budget: int
    judge_enabled: bool
    confirmation_enabled: bool
    enable_leakage: bool
    enable_duplication: bool
    enable_robustness: bool
    model_target: Optional[str] = None
    scrape_window: Optional[dict] = None
    families: list = field(default_factory=list)
    column_map: Optional[dict] = None
    truth_policy: Optional[str] = None
    output_dir: Optional[str] = None
    calibration_sheet: Optional[str] = None
    approval_notes: str = ""

    def requires_calls(self) -> bool:
        return self.source_type in {SOURCE_GENERATED, SOURCE_REALMATH_SCRAPE}

    def is_approved(self) -> bool:
        return bool(self.approved_by) and bool(self.approved_at)

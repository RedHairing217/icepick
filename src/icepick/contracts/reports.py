"""Report envelopes.

Stable top-level sections in every machine-readable summary so a human
operator can scan one and know where each fact lives:

    run            - run id, processor_mode, calibration_sheet, created_at
    inputs         - source-tagged input paths
    capabilities   - judge reachability, judge model, web_search availability
    counts         - by source, by stage status, by bucket
    buckets        - bucket file paths and per-bucket counts
    parameters     - stage knobs echoed verbatim
    warnings       - non-fatal anomalies the operator should see

Markdown reports built from a summary always lead with outcome, then
counts, then action items, then per-record details. CLI surfaces the
summary's run id, mode, counts, warnings, and output paths only;
per-record detail goes into JSONL.
"""

from __future__ import annotations

SUMMARY_SECTIONS = (
    "run",
    "inputs",
    "capabilities",
    "counts",
    "buckets",
    "parameters",
    "warnings",
)


def empty_summary() -> dict:
    """Return a summary skeleton with every required section present."""
    return {section: {} for section in SUMMARY_SECTIONS}

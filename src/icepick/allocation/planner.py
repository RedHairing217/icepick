"""Source-mix and family planning.

Composes a ``ProposedPlan`` from CLI args. Pure: no calls, no scrape, no
file writes beyond the plan itself.
"""

from __future__ import annotations


def propose(*, source_type, source_name, target_count, families=None, scrape_window=None, requested_by, requested_at, notes=""):
    raise NotImplementedError("allocation.planner.propose is not yet implemented")

"""Call and dollar budget estimates for proposed plans.

Reads the adapter's estimator and records the estimate on the plan.
Conservative by default; allocation never silently increases counts.
"""

from __future__ import annotations


def estimate(plan, *, adapter):
    raise NotImplementedError("allocation.budgets.estimate is not yet implemented")

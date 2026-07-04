"""Generated-family adapter.

In production mode this shells out to the provenance repo's generated-
family harvesters along an explicit configured path. The path stays
explicit in config so it never hides behind a package import.

In ``flow_testing`` mode replay returns preprocessed generated records
from the calibration sheet.
"""

from __future__ import annotations


def plan(request):
    raise NotImplementedError("adapters.generated.plan is not yet implemented")


def estimate(plan):
    raise NotImplementedError("adapters.generated.estimate is not yet implemented")


def run(manifest):
    raise NotImplementedError("adapters.generated.run is not yet implemented")


def normalise(raw_outputs):
    raise NotImplementedError("adapters.generated.normalise is not yet implemented")

"""External JSONL validator.

Validates a provided JSONL drop without scraping or generation. No
network calls. Produces a manifest with ``source_type =
external_jsonl`` and ``call_budget = 0``.
"""

from __future__ import annotations


def plan(request):
    raise NotImplementedError("adapters.external_jsonl.plan is not yet implemented")


def estimate(plan):
    raise NotImplementedError("adapters.external_jsonl.estimate is not yet implemented")


def run(manifest):
    raise NotImplementedError("adapters.external_jsonl.run is not yet implemented")


def normalise(raw_outputs):
    raise NotImplementedError("adapters.external_jsonl.normalise is not yet implemented")

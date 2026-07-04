"""Allocation subsystem.

Owns top-of-funnel acquisition: source mix, count requests, approved
manifests, adapter selection, call budgets, in-house scraping/harvesting,
manual mounts, and handoff of acquired JSONL into processing.

Does NOT run quality checks or mutate verdicts. Does NOT call processing
or agent modules.
"""

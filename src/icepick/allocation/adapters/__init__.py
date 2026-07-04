"""Acquisition adapters.

Each adapter implements the same surface:

    plan(request)      -> proposed_plan
    estimate(plan)     -> budget
    run(manifest)      -> raw_outputs
    normalise(raw)     -> jsonl_inputs

Initial adapters: generated, realmath_scrape, external_jsonl, manual_mount.
"""

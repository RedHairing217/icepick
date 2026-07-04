"""Allocation-side calibration replay.

In ``flow_testing`` mode, acquisition adapters replay preprocessed
records from the calibration sheet instead of making real calls or
scraping. ``realmath_scrape`` implements this adapter-side: its ``run``
replays the manifest's ``calibration_sheet`` fixture and stamps
``calibration_replay: true`` on every handoff record's metadata and on
the source report, so replay output cannot pass for production
downstream. ``generated`` will share this module's entry point once it
is implemented.
"""

from __future__ import annotations


def replay_acquisition(plan, sheet):
    raise NotImplementedError("allocation.calibration_replay.replay_acquisition is not yet implemented")

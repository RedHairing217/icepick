"""Agent-side calibration replay.

In ``flow_testing`` mode the manager call is replaced with scripted
action objects loaded from the calibration sheet's ``agent`` section.
The controller still runs the same validation and dispatch path; only
the LLM call is replayed.
"""

from __future__ import annotations


def replay_manager(scenario_id: str, sheet):
    raise NotImplementedError("agent.calibration_replay.replay_manager is not yet implemented")

"""Test fixtures shared across processing / allocation / agent suites.

Each subsystem's suite must remain runnable in isolation. Anything that
needs to span them belongs in ``tests/integration/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CALIBRATION = Path(__file__).parent / "calibration"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def calibration_dir() -> Path:
    return CALIBRATION


@pytest.fixture
def mixed_jsonl(tmp_path) -> Path:
    """A small mixed corpus: one generated row and one extracted row."""
    rows = [
        {
            "family": "calculus",
            "tier": "intro",
            "question": "Compute the derivative of x^2.",
            "answer": "2*x",
            "correct": 3,
            "wrong_complete": 1,
            "degenerate": 0,
            "pass_at_k": 0.75,
        },
        {
            "question": "State the closed form of the sum 1 + 1/2 + 1/4 + ...",
            "truth": "2",
            "correct": 0,
            "wrong_complete": 4,
            "degenerate": 0,
            "pass_at_k": 0.0,
            "top_wrong_share": 1.0,
            "top_wrong_value": "infinity",
        },
    ]
    path = tmp_path / "mixed.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path

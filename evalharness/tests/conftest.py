"""Shared test setup for the evalharness suite.

Inserts ``evalharness/src`` onto ``sys.path`` so ``import evalharness...``
works whether or not the package has been ``pip install -e``'d --
mirrors ``src/posers/Claude_Poser/tests/conftest.py``. This suite is
network-free: build_eval_set and report are pure file-in/file-out;
run_eval's tests inject a fake subprocess runner rather than invoking
icepick or a real backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # evalharness/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES

"""Shared test setup for the loratrain suite.

Inserts ``src/loratrain/src`` onto ``sys.path`` so ``import loratrain...``
works whether or not the package has been ``pip install -e``'d --
mirrors ``evalharness/tests/conftest.py``. This suite is hermetic: no
network calls, no dependency on the real corpus or on evalharness's
output files -- every test builds its own synthetic data under
``tmp_path`` (or a small in-memory dict/list).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # src/loratrain/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

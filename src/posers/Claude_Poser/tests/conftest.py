import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _isolate_os_environ():
    """Snapshot os.environ before each test and fully restore it after.

    Rationale: the env_file loader writes directly to os.environ (bypassing
    pytest's monkeypatch tracking). Tests that load fake env files like
    'ANTHROPIC_API_KEY=sk-ant-fake-for-test' would otherwise leak those
    values into subsequent tests — including cross-suite runs where
    icepick's key-presence validation then wrongly passes because a fake
    key is still lingering in os.environ.

    Autouse guarantees every test in this package runs against a clean
    slate, regardless of which loaders it calls.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        # Remove keys added during the test
        for key in list(os.environ.keys()):
            if key not in snapshot:
                del os.environ[key]
        # Restore keys that existed before, in case they were mutated
        for key, value in snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value

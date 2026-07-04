import os
from pathlib import Path

import pytest

from claude_poser.env_file import load_env_file


@pytest.fixture
def isolated_env(monkeypatch):
    """Strip any pre-existing keys so tests don't pick up the user's shell."""
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "AP_TEST_KEY", "AP_OTHER"):
        monkeypatch.delenv(k, raising=False)


def _write(p: Path, body: str) -> Path:
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_simple_key_value(tmp_path, isolated_env):
    f = _write(tmp_path / "a.env", "AP_TEST_KEY=abc123\n")
    result = load_env_file(f)
    assert result.loaded == {"AP_TEST_KEY": "abc123"}
    assert os.environ["AP_TEST_KEY"] == "abc123"


def test_ignores_blank_lines_and_comments(tmp_path, isolated_env):
    body = "\n# a comment\n\n  # indented comment\nAP_TEST_KEY=ok\n"
    f = _write(tmp_path / "a.env", body)
    result = load_env_file(f)
    assert result.loaded == {"AP_TEST_KEY": "ok"}
    assert result.skipped == []


def test_strips_surrounding_quotes(tmp_path, isolated_env):
    body = "AP_TEST_KEY=\"sk-ant-abc\"\nAP_OTHER='hello world'\n"
    f = _write(tmp_path / "a.env", body)
    load_env_file(f)
    assert os.environ["AP_TEST_KEY"] == "sk-ant-abc"
    assert os.environ["AP_OTHER"] == "hello world"


def test_accepts_export_prefix(tmp_path, isolated_env):
    f = _write(tmp_path / "a.env", "export AP_TEST_KEY=xyz\n")
    load_env_file(f)
    assert os.environ["AP_TEST_KEY"] == "xyz"


def test_does_not_override_existing_env(tmp_path, isolated_env, monkeypatch):
    monkeypatch.setenv("AP_TEST_KEY", "from_shell")
    f = _write(tmp_path / "a.env", "AP_TEST_KEY=from_file\n")
    result = load_env_file(f)
    assert os.environ["AP_TEST_KEY"] == "from_shell"
    assert result.already_set == ["AP_TEST_KEY"]
    assert result.loaded == {}


def test_override_when_explicit(tmp_path, isolated_env, monkeypatch):
    monkeypatch.setenv("AP_TEST_KEY", "from_shell")
    f = _write(tmp_path / "a.env", "AP_TEST_KEY=from_file\n")
    load_env_file(f, override=True)
    assert os.environ["AP_TEST_KEY"] == "from_file"


def test_malformed_lines_skipped(tmp_path, isolated_env):
    body = "this_is_not_a_pair\nAP_TEST_KEY=ok\n123BAD=nope\n"
    f = _write(tmp_path / "a.env", body)
    result = load_env_file(f)
    assert os.environ["AP_TEST_KEY"] == "ok"
    skipped_lines = {ln for ln, _ in result.skipped}
    assert 1 in skipped_lines  # malformed
    assert 3 in skipped_lines  # invalid key (starts with digit)


def test_missing_file_raises(tmp_path, isolated_env):
    with pytest.raises(FileNotFoundError):
        load_env_file(tmp_path / "does_not_exist.env")


def test_does_not_interpolate(tmp_path, isolated_env, monkeypatch):
    monkeypatch.setenv("AP_OTHER", "interp")
    f = _write(tmp_path / "a.env", "AP_TEST_KEY=${AP_OTHER}\n")
    load_env_file(f)
    # We intentionally do not expand — the value is the literal text.
    assert os.environ["AP_TEST_KEY"] == "${AP_OTHER}"

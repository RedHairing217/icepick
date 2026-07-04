"""The ANTHROPIC_KEY_FILE proxy: resolve the key without embedding it.

Uses fake key strings only — never a real secret.
"""

from __future__ import annotations

import pytest

from icepick.config import ConfigError, load_env_file, resolve_anthropic_credentials


def test_resolve_prefers_a_direct_env_key():
    key, model = resolve_anthropic_credentials({"ANTHROPIC_API_KEY": "sk-direct", "ANTHROPIC_MODEL": "m1"})
    assert key == "sk-direct"
    assert model == "m1"


def test_resolve_reads_the_key_from_the_proxy_file(tmp_path):
    key_file = tmp_path / "anthro_key.env"
    key_file.write_text("# a key file\nANTHROPIC_API_KEY=sk-fromfile\nANTHROPIC_MODEL=claude-x\n")
    key, model = resolve_anthropic_credentials({"ANTHROPIC_KEY_FILE": str(key_file)})
    assert key == "sk-fromfile"
    assert model == "claude-x"


def test_direct_env_key_wins_over_the_proxy_file(tmp_path):
    key_file = tmp_path / "anthro_key.env"
    key_file.write_text("ANTHROPIC_API_KEY=sk-fromfile\n")
    key, _ = resolve_anthropic_credentials(
        {"ANTHROPIC_API_KEY": "sk-direct", "ANTHROPIC_KEY_FILE": str(key_file)}
    )
    assert key == "sk-direct"


def test_resolve_raises_when_nothing_is_configured():
    with pytest.raises(ConfigError, match="ANTHROPIC_KEY_FILE"):
        resolve_anthropic_credentials({})


def test_load_env_file_ignores_comments_and_strips_quotes(tmp_path):
    f = tmp_path / "k.env"
    f.write_text('# comment\nANTHROPIC_API_KEY="sk-quoted"\n\nEXTRA=1\n')
    parsed = load_env_file(f)
    assert parsed["ANTHROPIC_API_KEY"] == "sk-quoted"
    assert parsed["EXTRA"] == "1"

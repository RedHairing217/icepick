"""WellposedConfig validation — fleet of (build, provider) combos."""

from __future__ import annotations

import pytest

from icepick.config import ConfigError
from icepick.processing.poser.config import (
    BUILD_CLAUDE,
    BUILD_CODEX,
    POLICY_INTERSECT,
    POLICY_MAJORITY,
    POLICY_PREFER,
    POLICY_UNION,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    Combo,
    WellposedConfig,
    all_combos,
    parse_combo,
)


def _claude_anthropic() -> Combo:
    return Combo(build=BUILD_CLAUDE, provider=PROVIDER_ANTHROPIC)


def _codex_openai() -> Combo:
    return Combo(build=BUILD_CODEX, provider=PROVIDER_OPENAI)


def test_default_config_validates_when_judge_off():
    cfg = WellposedConfig(combos=[_claude_anthropic()], mode="production", enable_judge_tier=False)
    cfg.validate()


def test_empty_fleet_rejected():
    cfg = WellposedConfig(combos=[], mode="production", enable_judge_tier=False)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_duplicate_combos_rejected():
    cfg = WellposedConfig(
        combos=[_claude_anthropic(), _claude_anthropic()],
        mode="production", enable_judge_tier=False,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_unknown_build_rejected():
    cfg = WellposedConfig(combos=[Combo(build="bogus", provider="anthropic")],
                          mode="production", enable_judge_tier=False)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_unknown_provider_rejected():
    cfg = WellposedConfig(combos=[Combo(build="claude", provider="bogus")],
                          mode="production", enable_judge_tier=False)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_flow_testing_requires_calibration_sheet():
    cfg = WellposedConfig(combos=[_claude_anthropic()], mode="flow_testing", enable_judge_tier=False)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_judge_uphold_cannot_exceed_samples():
    cfg = WellposedConfig(
        combos=[_claude_anthropic()], mode="production", enable_judge_tier=False,
        judge_samples=3, judge_uphold=4,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_codex_combo_with_judge_in_flow_testing_rejected_up_front():
    cfg = WellposedConfig(
        combos=[_codex_openai()], mode="flow_testing", enable_judge_tier=True,
        calibration_sheet="cs.jsonl",
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_production_judge_anthropic_requires_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = WellposedConfig(
        combos=[_claude_anthropic()], mode="production", enable_judge_tier=True,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_production_judge_anthropic_accepts_env_var(monkeypatch):
    """If ANTHROPIC_API_KEY is exported, no key file is required."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = WellposedConfig(
        combos=[_claude_anthropic()], mode="production", enable_judge_tier=True,
    )
    cfg.validate()


def test_production_judge_openai_requires_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = WellposedConfig(
        combos=[_codex_openai()], mode="production", enable_judge_tier=True,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_full_fleet_with_judge_requires_both_key_files(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Only the anthropic key provided → openai combos still missing their key.
    cfg = WellposedConfig(
        combos=all_combos(), mode="production", enable_judge_tier=True,
        anthropic_key_file=tmp_path / "anthro_key.env",
    )
    with pytest.raises(ConfigError):
        cfg.validate()
    # Add the openai key → validates.
    cfg.openai_key_file = tmp_path / "openai_key.env"
    cfg.validate()


def test_policy_prefer_must_reference_an_active_combo():
    cfg = WellposedConfig(
        combos=[_claude_anthropic()], mode="production", enable_judge_tier=False,
        comparison_policy=f"{POLICY_PREFER}codex:openai",
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_policy_prefer_valid_when_combo_in_fleet():
    cfg = WellposedConfig(
        combos=[_claude_anthropic(), _codex_openai()], mode="production",
        enable_judge_tier=False, comparison_policy=f"{POLICY_PREFER}claude:anthropic",
    )
    cfg.validate()


def test_policy_majority_validates():
    cfg = WellposedConfig(
        combos=all_combos(), mode="production", enable_judge_tier=False,
        comparison_policy=POLICY_MAJORITY,
    )
    cfg.validate()


def test_echo_round_trips_keys():
    cfg = WellposedConfig(combos=[_claude_anthropic()], mode="production", enable_judge_tier=False)
    snap = cfg.echo()
    for key in ("combos", "mode", "comparison_policy", "claude", "codex",
                "anthropic_key_file", "openai_key_file"):
        assert key in snap
    assert snap["combos"] == ["claude:anthropic"]


def test_parse_combo_round_trip():
    c = parse_combo("claude:anthropic")
    assert c.build == "claude" and c.provider == "anthropic"
    assert c.key() == "claude:anthropic"
    assert c.slug() == "claude_anthropic"


def test_parse_combo_rejects_bad_spec():
    with pytest.raises(ConfigError):
        parse_combo("claude")
    with pytest.raises(ConfigError):
        parse_combo("claude:bogus")
    with pytest.raises(ConfigError):
        parse_combo("bogus:openai")


def test_all_combos_lists_four():
    combos = all_combos()
    keys = {c.key() for c in combos}
    assert keys == {
        "claude:anthropic", "claude:openai",
        "codex:anthropic", "codex:openai",
    }

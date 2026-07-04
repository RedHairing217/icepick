"""GroundtruthConfig validation."""

from __future__ import annotations

import pytest

from icepick.config import ConfigError
from icepick.processing.groundtruth.config import GroundtruthConfig


def test_production_requires_key_file_or_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = GroundtruthConfig(mode="production", output_dir=tmp_path)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_production_validates_with_key_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = GroundtruthConfig(
        mode="production",
        output_dir=tmp_path,
        anthropic_key_file=tmp_path / "anthro_key.env",
    )
    cfg.validate()


def test_production_validates_with_env_var_alone(tmp_path, monkeypatch):
    """Operators with ANTHROPIC_API_KEY exported don't need a key file."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = GroundtruthConfig(mode="production", output_dir=tmp_path)
    cfg.validate()


def test_flow_testing_requires_calibration_sheet(tmp_path):
    cfg = GroundtruthConfig(mode="flow_testing", output_dir=tmp_path)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_flow_testing_validates_with_sheet(tmp_path):
    cfg = GroundtruthConfig(
        mode="flow_testing",
        output_dir=tmp_path,
        calibration_sheet=tmp_path / "sheet.jsonl",
    )
    cfg.validate()


def test_judge_uphold_must_not_exceed_samples(tmp_path):
    cfg = GroundtruthConfig(
        mode="flow_testing",
        output_dir=tmp_path,
        calibration_sheet=tmp_path / "sheet.jsonl",
        judge_samples=3,
        judge_uphold=4,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_judge_samples_must_be_positive(tmp_path):
    cfg = GroundtruthConfig(
        mode="flow_testing",
        output_dir=tmp_path,
        calibration_sheet=tmp_path / "sheet.jsonl",
        judge_samples=0,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_echo_round_trips_keys(tmp_path):
    cfg = GroundtruthConfig(
        mode="flow_testing",
        output_dir=tmp_path,
        calibration_sheet=tmp_path / "sheet.jsonl",
    )
    snap = cfg.echo()
    for key in ("mode", "judge_model", "judge_samples", "judge_uphold",
                "max_concurrent", "discard_generated", "calibration_sheet"):
        assert key in snap

"""Mode and host-role validation tests."""

from __future__ import annotations

import pytest

from icepick.config import (
    ConfigError,
    HostConfig,
    HostsConfig,
    validate_host_roles,
    validate_mode,
)


def test_mode_required():
    with pytest.raises(ConfigError):
        validate_mode("", None)


def test_unknown_mode_rejected():
    with pytest.raises(ConfigError):
        validate_mode("debug", None)


def test_flow_testing_requires_calibration_sheet():
    with pytest.raises(ConfigError):
        validate_mode("flow_testing", None)


def test_production_does_not_require_calibration_sheet():
    validate_mode("production", None)


def test_subject_and_manager_must_both_exist():
    with pytest.raises(ConfigError):
        validate_host_roles(HostsConfig(subject=None, manager=None))


def test_subject_and_manager_cannot_share_base_url():
    cfg = HostsConfig(
        subject=HostConfig("subject", "http://h:1234/v1", "m1", "s.log", "s.cache"),
        manager=HostConfig("manager", "http://h:1234/v1", "m2", "m.log", "m.cache"),
    )
    with pytest.raises(ConfigError):
        validate_host_roles(cfg)


def test_subject_and_manager_cannot_share_log_path():
    cfg = HostsConfig(
        subject=HostConfig("subject", "http://a:1/v1", "m1", "same.log", "s.cache"),
        manager=HostConfig("manager", "http://b:2/v1", "m2", "same.log", "m.cache"),
    )
    with pytest.raises(ConfigError):
        validate_host_roles(cfg)


def test_valid_host_pair_passes():
    cfg = HostsConfig(
        subject=HostConfig("subject", "http://a:1/v1", "m1", "s.log", "s.cache"),
        manager=HostConfig("manager", "http://b:2/v1", "m2", "m.log", "m.cache"),
    )
    validate_host_roles(cfg)

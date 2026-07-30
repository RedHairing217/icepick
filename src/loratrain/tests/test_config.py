"""Tests for loratrain.config: validation guards + the single-source-of-truth scan."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from loratrain import config


def test_defaults_validate():
    config.validate_config()  # must not raise, as shipped


def test_bad_ip_rejected(monkeypatch):
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "999.999.999.999")
    monkeypatch.setattr(
        config, "TRAIN_SERVER_URL", f"http://999.999.999.999:{config.TRAIN_SERVER_PORT}"
    )
    with pytest.raises(config.ConfigError):
        config.validate_config()


def test_blank_ip_rejected(monkeypatch):
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "   ")
    monkeypatch.setattr(config, "TRAIN_SERVER_URL", f"http://   :{config.TRAIN_SERVER_PORT}")
    with pytest.raises(config.ConfigError):
        config.validate_config()


def test_bad_port_rejected(monkeypatch):
    monkeypatch.setattr(config, "TRAIN_SERVER_PORT", 70000)
    monkeypatch.setattr(config, "TRAIN_SERVER_URL", f"http://{config.TRAIN_SERVER_IP}:70000")
    with pytest.raises(config.ConfigError):
        config.validate_config()


def test_url_derivation_enforced(monkeypatch):
    # Hand-edit TRAIN_SERVER_URL directly without touching IP/port -- the
    # one thing validate_config must catch even though IP and port are
    # each individually still valid on their own.
    monkeypatch.setattr(config, "TRAIN_SERVER_URL", "http://example-not-derived.invalid:9999")
    with pytest.raises(config.ConfigError):
        config.validate_config()


def test_hyperparam_validation(monkeypatch):
    monkeypatch.setattr(config, "EPOCHS", 0)
    with pytest.raises(config.ConfigError):
        config.validate_config()


# --- SSH-tunnel-only URL contract (RUNBOOK D-R1, revised 2026-07-25) ---------


def test_url_is_tunnel_local_as_shipped():
    assert config.TRAIN_SERVER_URL == f"http://127.0.0.1:{config.TRAIN_SERVER_PORT}"


def test_pod_ip_does_not_flow_into_url(monkeypatch):
    # Post-D-R1 the pod IP is the ssh/scp target ONLY: setting a real pod IP
    # while the URL stays tunnel-local must validate cleanly (pre-D-R1 this
    # exact combination failed the derivation check).
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "203.0.113.7")
    config.validate_config()  # must not raise


def test_pod_ip_shaped_url_rejected(monkeypatch):
    # The pre-tunnel derivation output (pod IP baked into the URL) is now a
    # configuration error: nothing listens on the pod's port 8000 anymore.
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "203.0.113.7")
    monkeypatch.setattr(
        config, "TRAIN_SERVER_URL", f"http://203.0.113.7:{config.TRAIN_SERVER_PORT}"
    )
    with pytest.raises(config.ConfigError) as excinfo:
        config.validate_config()
    assert "tunnel-local" in str(excinfo.value)
    assert "Appendix A" in str(excinfo.value)


def test_bad_box_port_rejected(monkeypatch):
    monkeypatch.setattr(config, "TRAIN_STATUS_BOX_PORT", 0)
    with pytest.raises(config.ConfigError):
        config.validate_config()


def test_ssh_port_shipped_and_range_checked(monkeypatch):
    # Appendix A applied 2026-07-25; W3 provisioning (2026-07-26) sets the real
    # per-pod port, so assert the FIELD (present, valid) hermetically instead of
    # pinning the shipped placeholder value.
    assert isinstance(config.TRAIN_SERVER_SSH_PORT, int) and 1 <= config.TRAIN_SERVER_SSH_PORT <= 65535
    monkeypatch.setattr(config, "TRAIN_SERVER_SSH_PORT", 70000)
    with pytest.raises(config.ConfigError) as excinfo:
        config.validate_config()
    assert "TRAIN_SERVER_SSH_PORT" in str(excinfo.value)


def test_ssh_port_absent_is_tolerated(monkeypatch):
    # Deleted attribute = the env-fallback state upload_guard/tunnel accept
    # (and the pre-Appendix-A shape); validate_config must not treat absence
    # as a problem -- resolve_ssh_port owns the missing-everywhere refusal.
    monkeypatch.delattr(config, "TRAIN_SERVER_SSH_PORT")
    config.validate_config()  # must not raise


def test_single_source_of_truth_for_server_address():
    ip_re = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
    url_re = re.compile(r"https?://")
    package_dir = Path(config.__file__).resolve().parent
    offenders = []
    for py_file in sorted(package_dir.rglob("*.py")):
        if py_file.name == "config.py":
            continue
        text = py_file.read_text(encoding="utf-8")
        if ip_re.search(text) or url_re.search(text):
            offenders.append(py_file.name)
    assert not offenders, (
        "the training-server address may live only in config.py; found an "
        f"IP or URL literal in: {offenders}"
    )


def test_path_roots_sane():
    assert config.SUBREPO_ROOT.name == "loratrain"
    assert config.REPO_ROOT == config.SUBREPO_ROOT.parents[1]
    assert (config.REPO_ROOT / "src" / "loratrain") == config.SUBREPO_ROOT


# --- Base-scheme provenance (T4, 2026-07-30) ---------------------------------


def test_base_scheme_shipped_default_is_fp16():
    # Shipped default = current behavior; flipping it is the operator's
    # decision (see the block comment above BASE_SCHEME), not something
    # this revision changes unilaterally.
    assert config.BASE_SCHEME == config.BASE_SCHEME_FP16
    assert config.BASE_SCHEME_FP16 == "fp16_hf_revision"
    assert config.BASE_SCHEME_DEQUANT == "dequant_q4km"


def test_base_scheme_rejected_when_unknown(monkeypatch):
    monkeypatch.setattr(config, "BASE_SCHEME", "not_a_real_scheme")
    with pytest.raises(config.ConfigError) as excinfo:
        config.validate_config()
    assert "BASE_SCHEME" in str(excinfo.value)


def test_base_scheme_dequant_validates_cleanly(monkeypatch):
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    config.validate_config()  # must not raise


def test_expected_dequant_tensor_total_shape():
    assert isinstance(config.EXPECTED_DEQUANT_TENSOR_TOTAL, int)
    assert config.EXPECTED_DEQUANT_TENSOR_TOTAL == 399

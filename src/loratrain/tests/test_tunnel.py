"""Tests for loratrain.tunnel (the RUNBOOK section 6 status tunnel).

All hermetic: the ssh invocation is exercised via a monkeypatched
subprocess.run that records its argv -- no network, no ssh, no box. The
tunnel argv itself is asserted exactly, because its shape IS the security
property: a local-forward (-L) from the M4 to the box's loopback, never a
remote-forward or a gateway bind.
"""

from __future__ import annotations

import pytest

from loratrain import config, tunnel, upload_guard


@pytest.fixture
def pod(monkeypatch):
    """Point config at a fake provisioned pod (hostname target, env ssh port)."""
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "pod.example")
    monkeypatch.delattr(config, "TRAIN_SERVER_SSH_PORT", raising=False)
    monkeypatch.setenv("TRAIN_SSH_PORT", "2222")


# --- build_tunnel_command ----------------------------------------------------


def test_build_tunnel_command_shape(pod):
    cmd = tunnel.build_tunnel_command(2222)
    assert cmd == [
        "ssh",
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-p",
        "2222",
        "-L",
        f"{config.TRAIN_SERVER_PORT}:localhost:{config.TRAIN_STATUS_BOX_PORT}",
        "root@pod.example",
    ]


def test_build_tunnel_command_local_port_override(pod):
    cmd = tunnel.build_tunnel_command(2222, local_port=18000)
    assert f"18000:localhost:{config.TRAIN_STATUS_BOX_PORT}" in cmd


def test_build_tunnel_command_is_local_forward_only(pod):
    # -L (local forward) is the decision; -R would publish an M4 port on the
    # box and -g would open the local end to the operator's LAN.
    cmd = tunnel.build_tunnel_command(2222)
    assert "-L" in cmd
    assert "-R" not in cmd
    assert "-g" not in cmd


# --- main() ------------------------------------------------------------------


def test_main_dry_run_prints_command_and_opens_nothing(pod, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(tunnel.subprocess, "run", lambda *a, **k: calls.append((a, k)))

    rc = tunnel.main([])
    captured = capsys.readouterr()

    assert rc == 0
    assert calls == []
    assert "ssh" in captured.out
    assert "root@pod.example" in captured.out
    assert "DRY RUN" in captured.out


def test_main_execute_runs_ssh_after_checks(pod, monkeypatch):
    recorded = {}

    def fake_run(cmd, check=False):
        recorded["cmd"] = cmd
        recorded["check"] = check

    monkeypatch.setattr(tunnel.subprocess, "run", fake_run)

    rc = tunnel.main(["--execute"])

    assert rc == 0
    assert recorded["check"] is True
    assert recorded["cmd"][:2] == ["ssh", "-N"]
    assert recorded["cmd"][-1] == "root@pod.example"
    assert f"{config.TRAIN_SERVER_PORT}:localhost:{config.TRAIN_STATUS_BOX_PORT}" in recorded["cmd"]


def test_main_local_port_flag_flows_into_forward(pod, monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        tunnel.subprocess, "run", lambda cmd, check=False: recorded.update(cmd=cmd)
    )

    rc = tunnel.main(["--execute", "--local-port", "18000"])

    assert rc == 0
    assert f"18000:localhost:{config.TRAIN_STATUS_BOX_PORT}" in recorded["cmd"]


def test_main_refuses_placeholder_ip(monkeypatch, capsys):
    # Hermetic since W3 provisioning: pin the loopback placeholder explicitly --
    # the same check_target rule that guards uploads must refuse the tunnel too.
    monkeypatch.setattr(tunnel.config, "TRAIN_SERVER_IP", "127.0.0.1")
    calls = []
    monkeypatch.setattr(tunnel.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setenv("TRAIN_SSH_PORT", "2222")

    rc = tunnel.main(["--execute"])
    captured = capsys.readouterr()

    assert rc == 2
    assert calls == []
    assert "TUNNEL REFUSED" in captured.err


def test_main_refuses_without_ssh_port(pod, monkeypatch, capsys):
    monkeypatch.delenv("TRAIN_SSH_PORT", raising=False)

    rc = tunnel.main([])
    captured = capsys.readouterr()

    assert rc == 2
    assert "TUNNEL REFUSED" in captured.err


def test_refusals_reuse_upload_guard_machinery():
    # Not cosmetic: tunnel refusals must stay the same exception family the
    # uploader raises so operator tooling can treat them uniformly.
    assert tunnel.UploadRefused is upload_guard.UploadRefused

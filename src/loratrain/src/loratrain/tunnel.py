"""Operator helper for the SSH local-forward to the box's status endpoint.

Since the SSH-tunnel-only decision (RUNBOOK D-R1, revised 2026-07-25) the
box-side status server binds the container's loopback interface
(``remote/run_remote_train.sh`` starts it with ``--bind`` on loopback), so
it is reachable from the M4 ONLY through the tunnel this module opens: the
M4-local end is ``config.TRAIN_SERVER_PORT`` (making ``config.
TRAIN_SERVER_URL`` the tunnel-local URL the operator curls) and the box end
is ``config.TRAIN_STATUS_BOX_PORT``. ssh's ``-L`` binds the local end to
the M4's loopback by default, so the forwarded port is not exposed to the
operator's LAN either.

Every host/port value comes from ``config`` plus the same SSH-port
resolution rule as ``upload_guard`` (``config.TRAIN_SERVER_SSH_PORT`` --
RUNBOOK Appendix A, applied 2026-07-25 -- with the ``TRAIN_SSH_PORT`` env
var of section 1.3 as fallback) -- this module never spells out an address
itself (see ``config.py`` for the single-source-of-truth scan that
enforces this package-wide).

Dry-run by default (prints the command it would run); ``--execute``
actually opens the tunnel in the foreground -- Ctrl-C (or killing the ssh
process) closes it, and RUNBOOK section 9 teardown includes that step.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from loratrain import config
from loratrain.upload_guard import UploadRefused, check_target, resolve_ssh_port


def build_tunnel_command(ssh_port, local_port=None) -> list:
    """Pure argv builder for the status tunnel -- no subprocess started here.

    ``local_port`` defaults to ``config.TRAIN_SERVER_PORT`` (its post-D-R1
    meaning: the M4-local end of this tunnel); override it only if that
    port is occupied locally, and curl the override instead.
    """
    if local_port is None:
        local_port = config.TRAIN_SERVER_PORT
    return [
        "ssh",
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-p",
        str(ssh_port),
        "-L",
        f"{local_port}:localhost:{config.TRAIN_STATUS_BOX_PORT}",
        f"root@{config.TRAIN_SERVER_IP}",
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tunnel")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually open the tunnel (foreground; Ctrl-C closes it); default is dry-run (print the command only)",
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=None,
        help="M4-local listen port (default: config.TRAIN_SERVER_PORT)",
    )
    args = parser.parse_args(argv)

    try:
        config.validate_config()
        check_target()  # same rule as uploads: refuse while TRAIN_SERVER_IP is the loopback placeholder
        ssh_port = resolve_ssh_port()

        cmd = build_tunnel_command(ssh_port, local_port=args.local_port)

        if not args.execute:
            print(cmd)
            print("DRY RUN — tunnel not opened")
            return 0

        local_port = args.local_port if args.local_port is not None else config.TRAIN_SERVER_PORT
        print(
            f"opening status tunnel: M4-local port {local_port} -> box loopback "
            f"port {config.TRAIN_STATUS_BOX_PORT} (Ctrl-C closes it)"
        )
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("tunnel closed")
        return 0

    except UploadRefused as exc:
        print(f"TUNNEL REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

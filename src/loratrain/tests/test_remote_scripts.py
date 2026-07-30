"""Tests for the box-side remote/ scripts: syntax-only, never imported/run.

remote/train_qwen3_lora.py and remote/run_remote_train.sh reference CUDA-box
paths and packages (transformers/peft/trl/torch) absent from this repo, so
they are checked for syntax validity only (ast.parse / bash -n) plus an
address scan: no IPv4 literal at all in the trainer, and in the .sh the ONE
permitted literal is the loopback bind of the status server (SSH-tunnel-only,
RUNBOOK D-R1 revised 2026-07-25) -- pod addresses stay in config.py, carried
by the operator's own scp/ssh invocations per RUNBOOK.md, never these files.
This suite also trips if the .sh's STATUS_PORT default drifts from
config.TRAIN_STATUS_BOX_PORT (the two halves of the section 6 tunnel).
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from loratrain import config

REMOTE_DIR = Path(__file__).resolve().parents[1] / "remote"
TRAIN_SCRIPT = REMOTE_DIR / "train_qwen3_lora.py"
RUN_SCRIPT = REMOTE_DIR / "run_remote_train.sh"

_IPV4_RE = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
_LOOPBACK = "127.0.0.1"


def test_train_script_exists():
    assert TRAIN_SCRIPT.is_file()


def test_run_script_exists():
    assert RUN_SCRIPT.is_file()


def test_train_script_parses_as_valid_python():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(TRAIN_SCRIPT))  # raises SyntaxError on failure


def test_run_remote_train_sh_passes_bash_syntax_check():
    result = subprocess.run(
        ["bash", "-n", str(RUN_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_ipv4_literal_in_train_script():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert not _IPV4_RE.search(source), "found an IPv4 literal in train_qwen3_lora.py"


def test_only_loopback_ipv4_literal_in_run_remote_train_sh():
    # The status server's loopback bind is the ONE address this file may
    # spell (it cannot drift into a pod address by definition); anything
    # else -- a pod IP, an external mapping -- still fails the suite.
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    non_loopback = [
        m.group(0) for m in _IPV4_RE.finditer(source) if m.group(0) != _LOOPBACK
    ]
    assert not non_loopback, (
        f"found non-loopback IPv4 literal(s) in run_remote_train.sh: {non_loopback}"
    )


def test_status_server_binds_loopback():
    # Load-bearing for the SSH-tunnel-only decision: the http.server line
    # must carry --bind 127.0.0.1 so the endpoint is unreachable from the
    # internet even though the port itself is content-restricted.
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    assert re.search(rf"-m http\.server .*--bind {re.escape(_LOOPBACK)}", source), (
        "run_remote_train.sh must start the status server with "
        f"--bind {_LOOPBACK} (RUNBOOK D-R1, SSH-tunnel-only)"
    )


def test_status_port_default_matches_config_box_port():
    # The .sh runs on the box without config.py, so its STATUS_PORT default
    # duplicates config.TRAIN_STATUS_BOX_PORT by necessity -- this is the
    # drift tripwire between the two halves of the section 6 tunnel.
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r'STATUS_PORT="\$\{STATUS_PORT:-(\d+)\}"', source)
    assert match, "run_remote_train.sh no longer declares a STATUS_PORT default"
    assert int(match.group(1)) == config.TRAIN_STATUS_BOX_PORT, (
        f"run_remote_train.sh STATUS_PORT default {match.group(1)} != "
        f"config.TRAIN_STATUS_BOX_PORT {config.TRAIN_STATUS_BOX_PORT}"
    )


def test_train_script_has_no_hardcoded_grad_accum_literal():
    # v2 revision (2026-07-29): gradient_accumulation_steps was a hardcoded
    # `=4` literal (the silent-hyperparameter defect flagged by both 07-28
    # reviews); it must come from the run_config hyperparams now.
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"gradient_accumulation_steps\s*=\s*\d", source), (
        "train_qwen3_lora.py hardcodes gradient_accumulation_steps again -- "
        "it must be a named hyperparameter (hyperparams['grad_accum_steps'])"
    )
    assert "grad_accum_steps" in source


def test_train_script_pins_formerly_silent_sftconfig_defaults():
    # The three inherited-SFTConfig-default knobs must be spelled explicitly
    # in the SFTConfig call, and completion-only loss must be wired for the
    # v2 prompt/completion dataset path.
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    for kwarg in ("lr_scheduler_type", "warmup_ratio", "weight_decay", "completion_only_loss"):
        assert re.search(rf"{kwarg}\s*=", source), (
            f"train_qwen3_lora.py no longer pins {kwarg} explicitly in SFTConfig"
        )


def test_train_script_v1_fallback_defaults_match_what_v1_ran():
    # A v1 run_config.json (no new keys) must reproduce v1 exactly: the
    # .get() fallbacks are pinned to the values v1 actually trained with.
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert re.search(r"""\.get\(\s*['"]grad_accum_steps['"]\s*,\s*4\s*\)""", source)
    assert re.search(r"""\.get\(\s*['"]lr_scheduler_type['"]\s*,\s*['"]linear['"]\s*\)""", source)
    assert re.search(r"""\.get\(\s*['"]warmup_ratio['"]\s*,\s*0\.0\s*\)""", source)
    assert re.search(r"""\.get\(\s*['"]weight_decay['"]\s*,\s*0\.0\s*\)""", source)

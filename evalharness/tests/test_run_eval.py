"""Tests for evalharness.run_eval.

Not in the task's required test list (build_eval_set and report own that
list), but the cross-endpoint quant-confound guard is safety-critical
and cheap to test in isolation, and the wire-param constants (greedy
k=1/temp=0/think=off/max_tokens=2048) are exactly the kind of thing that
silently drifts under a careless edit -- so this file locks both down.

No network, no subprocess: model resolution is tested by parsing real
argv through build_arg_parser() (so we're honest about attribute names),
and run_eval() itself is exercised with an injected fake subprocess
runner that just records the commands it would have run and drops
placeholder output files, mirroring icepick's own injectable-backend
test pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalharness.run_eval import (
    GREEDY_K,
    GREEDY_MAX_TOKENS,
    GREEDY_TEMPERATURE,
    GREEDY_THINK,
    SECONDARY_K,
    SECONDARY_REPEATS,
    SECONDARY_TEMPERATURE,
    RunEvalError,
    _resolve_models,
    build_arg_parser,
    run_eval,
)


def _parse(argv):
    return build_arg_parser().parse_args(argv)


# --- model resolution / cross-endpoint guard -------------------------------------


def test_requires_at_least_one_model():
    args = _parse(["--eval-set", "x", "--output-dir", "y", "--backend-url", "http://h"])
    with pytest.raises(RunEvalError, match="at least one"):
        _resolve_models(args)


def test_single_model_base_only_resolves_one_spec():
    args = _parse(
        [
            "--eval-set", "x", "--output-dir", "y",
            "--model-base", "qwen3-8b-base",
            "--backend-url", "http://localhost:1234/v1/chat/completions",
        ]
    )
    specs = _resolve_models(args)
    assert len(specs) == 1
    assert specs[0].role == "base"
    assert specs[0].backend_url == "http://localhost:1234/v1/chat/completions"


def test_shared_backend_url_applies_to_both_models():
    args = _parse(
        [
            "--eval-set", "x", "--output-dir", "y",
            "--model-base", "base-id", "--model-tuned", "tuned-id",
            "--backend-url", "http://shared:1234/v1/chat/completions",
        ]
    )
    specs = _resolve_models(args)
    assert len(specs) == 2
    assert {s.backend_url for s in specs} == {"http://shared:1234/v1/chat/completions"}


def test_cross_endpoint_without_flag_refuses(capsys):
    args = _parse(
        [
            "--eval-set", "x", "--output-dir", "y",
            "--model-base", "base-id", "--model-tuned", "tuned-id",
            "--backend-url", "http://box-a:1234/v1/chat/completions",
            "--backend-url-tuned", "http://box-b:1234/v1/chat/completions",
        ]
    )
    with pytest.raises(RunEvalError, match="QUANT-CONFOUND GUARD"):
        _resolve_models(args)


def test_cross_endpoint_with_flag_proceeds_and_warns(capsys):
    args = _parse(
        [
            "--eval-set", "x", "--output-dir", "y",
            "--model-base", "base-id", "--model-tuned", "tuned-id",
            "--backend-url", "http://box-a:1234/v1/chat/completions",
            "--backend-url-tuned", "http://box-b:1234/v1/chat/completions",
            "--allow-cross-endpoint",
        ]
    )
    specs = _resolve_models(args)
    assert len(specs) == 2
    err = capsys.readouterr().err
    assert "WARNING" in err and "different endpoints" in err


def test_missing_backend_url_is_a_clear_error():
    args = _parse(["--eval-set", "x", "--output-dir", "y", "--model-base", "base-id"])
    with pytest.raises(RunEvalError, match="no --backend-url resolved"):
        _resolve_models(args)


def test_qwen_key_file_override_wins_over_shared(tmp_path):
    shared_key = tmp_path / "shared.env"
    tuned_key = tmp_path / "tuned.env"
    args = _parse(
        [
            "--eval-set", "x", "--output-dir", "y",
            "--model-base", "base-id", "--model-tuned", "tuned-id",
            "--backend-url", "http://shared:1234/v1/chat/completions",
            "--qwen-key-file", str(shared_key),
            "--qwen-key-file-tuned", str(tuned_key),
        ]
    )
    specs = _resolve_models(args)
    by_role = {s.role: s for s in specs}
    assert by_role["base"].qwen_key_file == shared_key
    assert by_role["tuned"].qwen_key_file == tuned_key


# --- greedy wire params are pinned, not knobs ------------------------------------


def test_greedy_wire_params_match_the_design_doc():
    """Locks the primary-pass protocol: --k 1 --temperature 0 --think off --max-tokens 2048."""
    assert GREEDY_K == 1
    assert GREEDY_TEMPERATURE == 0.0
    assert GREEDY_THINK == "off"
    assert GREEDY_MAX_TOKENS == 2048


def test_secondary_wire_params_match_the_design_doc():
    assert SECONDARY_K == 8
    assert SECONDARY_TEMPERATURE == 0.7
    assert SECONDARY_REPEATS == 3


# --- run_eval() orchestration, with an injected fake subprocess runner -----------


class _FakeRunner:
    """Records every command; drops a placeholder pass_at_k.jsonl so
    run_eval()'s copy-to-baseline/post_greedy.jsonl step has something to
    read, exactly as the real icepick subprocess would leave behind."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pass_at_k.jsonl").write_text('{"uid": "u1", "n_correct": 1}\n')
        (output_dir / "pass_at_k_manifest.json").write_text(json.dumps({"stage": "pass_at_k"}))


def _make_eval_set(tmp_path) -> Path:
    path = tmp_path / "eval_set.jsonl"
    path.write_text(
        '{"uid": "u1", "eval_slice": "eval_band", "statement": "S", "answer": "A"}\n'
        '{"uid": "u2", "eval_slice": "anchor_solved", "statement": "S2", "answer": "A2"}\n'
    )
    return path


def test_run_eval_greedy_only_writes_baseline_and_post(tmp_path):
    from evalharness.run_eval import ModelSpec

    eval_set = _make_eval_set(tmp_path)
    runner = _FakeRunner()
    run_eval(
        eval_set_path=eval_set,
        output_dir=tmp_path / "run",
        model_specs=[
            ModelSpec(role="base", model_id="m-base", backend_url="http://h/v1/chat/completions"),
            ModelSpec(role="tuned", model_id="m-tuned", backend_url="http://h/v1/chat/completions"),
        ],
        subprocess_runner=runner,
        secondary=False,
    )
    assert (tmp_path / "run" / "baseline_greedy.jsonl").exists()
    assert (tmp_path / "run" / "post_greedy.jsonl").exists()
    assert len(runner.calls) == 2  # one greedy call per model, no secondary
    for cmd in runner.calls:
        assert "--k" in cmd and cmd[cmd.index("--k") + 1] == "1"
        assert "--temperature" in cmd and cmd[cmd.index("--temperature") + 1] == "0.0"
        assert "--think" in cmd and cmd[cmd.index("--think") + 1] == "off"
        # eval_set_path fed straight through -- greedy runs on the WHOLE
        # eval set (eval-band + anchors), not just eval-band.
        assert str(eval_set) in cmd


def test_run_eval_secondary_filters_to_eval_band_only(tmp_path):
    from evalharness.run_eval import ModelSpec

    eval_set = _make_eval_set(tmp_path)
    runner = _FakeRunner()
    outcome = run_eval(
        eval_set_path=eval_set,
        output_dir=tmp_path / "run",
        model_specs=[ModelSpec(role="base", model_id="m-base", backend_url="http://h/v1/chat/completions")],
        subprocess_runner=runner,
        secondary=True,
    )
    # 1 greedy + 3 secondary repeats for the single model.
    assert len(runner.calls) == 1 + 3
    assert len(outcome.secondary_paths["base"]) == 3
    band_only = tmp_path / "run" / "_eval_band_only.jsonl"
    assert band_only.exists()
    rows = [json.loads(line) for line in band_only.read_text().splitlines()]
    assert {r["uid"] for r in rows} == {"u1"}  # only the eval_band-tagged uid, not the anchor


def test_run_eval_never_puts_key_contents_on_the_command_line(tmp_path):
    """The key FILE PATH is passed through; its contents are never opened or embedded."""
    from evalharness.run_eval import ModelSpec

    key_file = tmp_path / "qwen.env"
    key_file.write_text("super-secret-token-do-not-print")
    eval_set = _make_eval_set(tmp_path)
    runner = _FakeRunner()
    run_eval(
        eval_set_path=eval_set,
        output_dir=tmp_path / "run",
        model_specs=[ModelSpec(role="base", model_id="m-base", backend_url="http://h/v1/chat/completions", qwen_key_file=key_file)],
        subprocess_runner=runner,
        secondary=False,
    )
    (cmd,) = runner.calls
    assert str(key_file) in cmd  # the path is passed through...
    assert "super-secret-token-do-not-print" not in " ".join(cmd)  # ...never its contents

"""End-to-end CLI test for `icepick processing wellposed`.

Uses monkeypatching to substitute the real adapters with fakes so no
subprocess is launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick import cli
from icepick.processing.poser.base import (
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserRequest,
    PoserRunResult,
    PoserVerdict,
)


class _FakeClaude:
    build = "claude"

    def plan(self, records, cfg, combo, work_dir):
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        return PoserRequest(
            argv=["fake-claude", "score", "--combo", combo.key()], env={},
            input_path=Path(work_dir) / f"{combo.slug()}_in.jsonl",
            output_path=Path(work_dir) / f"{combo.slug()}_verdicts.json",
            cache_path=None, poser_name=combo.key(),
        )

    def run(self, request):
        return PoserRunResult(exit_code=0, stdout="", stderr="",
                              output_path=request.output_path, wall_clock_seconds=0.01)

    def normalise(self, raw_output_path, input_uids, *, combo):
        return [
            PoserVerdict(uid=uid, source="s",
                         verdict_status=STATUS_WELL_POSED if uid.endswith("good") else STATUS_ILL_POSED,
                         verdict_score=1.0, poser_name=combo.key(), poser_model="fake")
            for uid in input_uids
        ]


def _seed_input(tmp_path):
    records = [
        {"source": "s", "statement": "good", "uid": "uid_good"},
        {"source": "s", "statement": "bad", "uid": "uid_bad"},
    ]
    input_path = tmp_path / "passatk.jsonl"
    with input_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return input_path


def test_wellposed_cli_runs_single_combo_end_to_end(tmp_path, monkeypatch, capsys):
    input_path = _seed_input(tmp_path)
    output_dir = tmp_path / "wellposed"

    from icepick.processing.poser import runner as runner_mod
    monkeypatch.setattr(runner_mod, "ClaudePoserAdapter", _FakeClaude)

    rc = cli.main(
        [
            "processing", "wellposed",
            "--combo", "claude:anthropic",
            "--mode", "production",
            "--no-judge",
            "--input", str(input_path),
            "--output-dir", str(output_dir),
        ]
    )
    assert rc == 0
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "claude_anthropic_normalised.jsonl").exists()

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["config"]["combos"] == ["claude:anthropic"]
    assert manifest["counts"][STATUS_WELL_POSED] == 1
    assert manifest["counts"][STATUS_ILL_POSED] == 1

    parsed = json.loads(capsys.readouterr().out)
    assert parsed["combos"] == ["claude:anthropic"]


def test_wellposed_cli_runs_all_combos_in_parallel(tmp_path, monkeypatch, capsys):
    """--combo all expands to all four combinations; runner fan-outs in parallel."""
    input_path = _seed_input(tmp_path)
    output_dir = tmp_path / "wellposed"

    from icepick.processing.poser import runner as runner_mod
    # Use the fake for both builds so no subprocess is launched.
    monkeypatch.setattr(runner_mod, "ClaudePoserAdapter", _FakeClaude)
    monkeypatch.setattr(runner_mod, "CodexPoserAdapter", _FakeClaude)

    rc = cli.main(
        [
            "processing", "wellposed",
            "--combo", "all",
            "--mode", "production",
            "--no-judge",
            "--input", str(input_path),
            "--output-dir", str(output_dir),
            "--comparison-policy", "majority",
        ]
    )
    assert rc == 0

    # All four normalised files exist.
    for slug in ("claude_anthropic", "claude_openai", "codex_anthropic", "codex_openai"):
        assert (output_dir / f"{slug}_normalised.jsonl").exists()
    # Comparison + report exist when fleet > 1.
    assert (output_dir / "comparison.jsonl").exists()
    assert (output_dir / "comparison_report.md").exists()
    # Gate input is the combined-majority file.
    assert (output_dir / "combined_majority.jsonl").exists()


def test_wellposed_cli_rejects_missing_mode(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["processing", "wellposed", "--combo", "claude:anthropic",
                  "--input", str(tmp_path / "x.jsonl"),
                  "--output-dir", str(tmp_path)])


def test_wellposed_cli_rejects_missing_combo(tmp_path):
    """Empty fleet must fail with a clean E_CONFIG code, not crash."""
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps({"source": "s", "statement": "q"}) + "\n")
    rc = cli.main(
        [
            "processing", "wellposed",
            "--mode", "production",
            "--no-judge",
            "--input", str(input_path),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 1


def test_wellposed_cli_rejects_codex_judge_in_flow_testing(tmp_path):
    """Config-level invariant surfaces as a clean error code."""
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps({"source": "s", "statement": "q"}) + "\n")
    cs = tmp_path / "cs.jsonl"
    cs.write_text("{}\n")
    rc = cli.main(
        [
            "processing", "wellposed",
            "--combo", "codex:anthropic",
            "--mode", "flow_testing",
            "--calibration-sheet", str(cs),
            "--input", str(input_path),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 1


def test_wellposed_cli_requires_openai_key_when_openai_combo_uses_judge(tmp_path):
    """If a combo uses provider=openai with judge enabled, --openai-key-file is required."""
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps({"source": "s", "statement": "q"}) + "\n")
    rc = cli.main(
        [
            "processing", "wellposed",
            "--combo", "claude:openai",
            "--mode", "production",
            "--input", str(input_path),
            "--output-dir", str(tmp_path / "out"),
            # judge tier on (default), but no --openai-key-file
        ]
    )
    assert rc == 1

"""End-to-end CLI for `icepick processing pass_at_k`.

Mirrors the groundtruth CLI test's approach: invoke ``cli.main()``
directly, capture the JSON summary from stdout, and monkeypatch the
runner's backend builder so no real SDK or network is ever touched.

Production runs use ``backend='qwen_http'`` (kill-switch-exempt) with a
dummy ``--backend-url`` that is never contacted — same convention as the
runner test files.
"""

from __future__ import annotations

import json
from pathlib import Path

from icepick import cli
from icepick.processing.pass_at_k.base import (
    LABEL_BAND,
    LABEL_MISDIRECTION,
    LABEL_SOLVED,
)


class _FakeBackend:
    """Scripted outputs keyed by question; records every call."""

    name = "fake"

    def __init__(self, outputs_by_question: dict):
        self._outputs = {q: list(v) for q, v in outputs_by_question.items()}
        self.calls = []

    def call(self, question, *, k, temperature, max_tokens, think, timeout):
        self.calls.append(question)
        return [self._outputs[question].pop(0) for _ in range(k)]


def _write_jsonl(path: Path, rows: list) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _seed_input(tmp_path) -> Path:
    return _write_jsonl(
        tmp_path / "records.jsonl",
        [
            {"uid": "uid_a", "source": "rm", "statement": "What is 2+2?", "truth": "4"},
            {"uid": "uid_b", "source": "rm", "statement": "What is 3+3?", "truth": "6"},
        ],
    )


def _seed_sheet(tmp_path) -> Path:
    return _write_jsonl(
        tmp_path / "sheet.jsonl",
        [
            {"uid": "uid_a", "pass_at_k": 1.0},
            {"uid": "uid_b", "pass_at_k": 0.5, "top_wrong_share": 0.25},
        ],
    )


def test_cli_flow_testing_end_to_end(tmp_path, capsys):
    input_path = _seed_input(tmp_path)
    sheet_path = _seed_sheet(tmp_path)
    output_dir = tmp_path / "out"

    rc = cli.main(
        [
            "processing", "pass_at_k",
            "--mode", "flow_testing",
            "--calibration-sheet", str(sheet_path),
            "--input", str(input_path),
            "--output-dir", str(output_dir),
        ]
    )
    assert rc == 0

    assert (output_dir / "pass_at_k_input.jsonl").exists()
    assert (output_dir / "pass_at_k.jsonl").exists()
    assert (output_dir / "pass_at_k_manifest.json").exists()

    # Labels replay deterministically from the sheet.
    labels = {r["uid"]: r["label"] for r in _rows(output_dir / "pass_at_k.jsonl")}
    assert labels == {"uid_a": LABEL_SOLVED, "uid_b": LABEL_BAND}

    out = json.loads(capsys.readouterr().out)
    assert out["stage"] == "pass_at_k"
    assert out["mode"] == "flow_testing"
    assert out["input_record_count"] == 2
    assert out["counts"][LABEL_SOLVED] == 1
    assert out["counts"][LABEL_BAND] == 1
    assert out["model_calls"] == 0
    assert out["interrupted"] is False
    assert out["outputs"]["manifest"] == str(output_dir / "pass_at_k_manifest.json")
    assert out["outputs"]["records"] == str(output_dir / "pass_at_k.jsonl")

    manifest = json.loads((output_dir / "pass_at_k_manifest.json").read_text())
    assert manifest["calibration_replay"] is True


def test_cli_production_end_to_end_with_fake_backend(tmp_path, monkeypatch, capsys):
    input_path = _seed_input(tmp_path)
    output_dir = tmp_path / "out"

    fake = _FakeBackend(
        {
            "What is 2+2?": ["\\boxed{4}", "\\boxed{4}"],
            "What is 3+3?": ["\\boxed{7}", "\\boxed{7}"],
        }
    )
    # Substitute the backend at the runner build site.
    from icepick.processing.pass_at_k import runner as runner_mod
    monkeypatch.setattr(runner_mod, "build_backend", lambda cfg: fake)

    rc = cli.main(
        [
            "processing", "pass_at_k",
            "--mode", "production",
            "--backend", "qwen_http",
            "--backend-url", "http://fake",
            "--input", str(input_path),
            "--output-dir", str(output_dir),
            "--k", "2",
            "--temperature", "0.0",
            "--max-concurrent", "1",
        ]
    )
    assert rc == 0

    rows = _rows(output_dir / "pass_at_k.jsonl")
    assert [r["uid"] for r in rows] == ["uid_a", "uid_b"]

    solved, misdir = rows
    assert solved["pass_at_k"] == 1.0
    assert solved["label"] == LABEL_SOLVED
    assert solved["n_correct"] == 2
    assert solved["rollout_uids"] == ["uid_a-r00", "uid_a-r01"]
    # Original fields survive the stamp.
    assert solved["statement"] == "What is 2+2?"
    assert solved["truth"] == "4"

    assert misdir["pass_at_k"] == 0.0
    assert misdir["label"] == LABEL_MISDIRECTION
    assert misdir["modal_wrong"] == "7"

    out = json.loads(capsys.readouterr().out)
    assert out["stage"] == "pass_at_k"
    assert out["backend"] == "qwen_http"
    assert out["model_calls"] == 4  # 2 records x k=2, all paid to the fake
    assert len(fake.calls) == 4


def test_cli_kill_switch_blocks_paid_backend_in_production(tmp_path, capsys):
    input_path = _seed_input(tmp_path)
    output_dir = tmp_path / "out"

    rc = cli.main(
        [
            "processing", "pass_at_k",
            "--mode", "production",
            "--backend", "anthropic",
            "--input", str(input_path),
            "--output-dir", str(output_dir),
        ]
    )
    assert rc == 1

    err = capsys.readouterr().err
    assert "E_CONFIG" in err
    assert "allow-live-calls" in err

    # validate() fires inside run() before any output is written.
    assert not output_dir.exists()


def test_cli_think_flag_maps_to_config(tmp_path, capsys):
    input_path = _seed_input(tmp_path)
    sheet_path = _seed_sheet(tmp_path)

    for flag, expected in (("on", True), ("off", False)):
        output_dir = tmp_path / f"out_think_{flag}"
        rc = cli.main(
            [
                "processing", "pass_at_k",
                "--mode", "flow_testing",
                "--calibration-sheet", str(sheet_path),
                "--input", str(input_path),
                "--output-dir", str(output_dir),
                "--think", flag,
            ]
        )
        assert rc == 0
        capsys.readouterr()  # drain the summary between runs
        manifest = json.loads((output_dir / "pass_at_k_manifest.json").read_text())
        assert manifest["config"]["think"] is expected

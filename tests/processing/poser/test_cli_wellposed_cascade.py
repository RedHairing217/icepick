"""End-to-end CLI test for `icepick processing wellposed-cascade`.

Monkeypatches the real adapters with fakes so no subprocess is launched.
Verifies exit code, manifest layout, final_corpus contents, and default
--stages contract.
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


class _RoutingFake:
    """Fake adapter routing by combo.key(). Shared between codex:openai and codex:anthropic."""

    def __init__(self, build, verdicts_by_combo_uid):
        self.build = build
        self._verdicts = verdicts_by_combo_uid

    def plan(self, records, cfg, combo, work_dir):
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        return PoserRequest(
            argv=[f"fake-{self.build}", "score", "--combo", combo.key()],
            env={},
            input_path=Path(work_dir) / f"{combo.slug()}_in.jsonl",
            output_path=Path(work_dir) / f"{combo.slug()}_verdicts.json",
            cache_path=None,
            poser_name=combo.key(),
        )

    def run(self, request):
        return PoserRunResult(
            exit_code=0, stdout="", stderr="",
            output_path=request.output_path, wall_clock_seconds=0.01,
        )

    def normalise(self, raw_output_path, input_uids, *, combo):
        out = []
        for uid in input_uids:
            v = self._verdicts.get((combo.key(), uid))
            if v is None:
                out.append(PoserVerdict(
                    uid=uid, source="", verdict_status=STATUS_ILL_POSED,
                    verdict_score=0.0, poser_name=combo.key(), poser_model="fake",
                ))
            else:
                out.append(v)
        return out


def _wp(uid, status, combo_key):
    return PoserVerdict(
        uid=uid, source="s", verdict_status=status,
        verdict_score=1.0 if status == STATUS_WELL_POSED else 0.0,
        poser_name=combo_key, poser_model="fake",
    )


def _seed_input(tmp_path):
    records = [
        {"source": "s", "statement": "good", "uid": "uid_good"},
        {"source": "s", "statement": "bad", "uid": "uid_bad"},
    ]
    input_path = tmp_path / "input.jsonl"
    with input_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return input_path


def _install_fakes(monkeypatch):
    """Patch the runner's adapter defaults so cascade's per-stage runner.run() picks up fakes."""
    all_good = {
        ("codex:openai", "uid_good"):    _wp("uid_good", STATUS_WELL_POSED, "codex:openai"),
        ("codex:openai", "uid_bad"):     _wp("uid_bad",  STATUS_ILL_POSED, "codex:openai"),
        ("codex:anthropic", "uid_good"): _wp("uid_good", STATUS_WELL_POSED, "codex:anthropic"),
        ("claude:openai", "uid_good"):   _wp("uid_good", STATUS_WELL_POSED, "claude:openai"),
    }
    codex = _RoutingFake("codex", all_good)
    claude = _RoutingFake("claude", all_good)

    from icepick.processing.poser import runner as runner_mod
    monkeypatch.setattr(runner_mod, "CodexPoserAdapter", lambda: codex)
    monkeypatch.setattr(runner_mod, "ClaudePoserAdapter", lambda: claude)


def test_cli_cascade_end_to_end_writes_manifest_and_final_corpus(tmp_path, monkeypatch, capsys):
    input_path = _seed_input(tmp_path)
    output_dir = tmp_path / "cascade"
    _install_fakes(monkeypatch)

    rc = cli.main([
        "processing", "wellposed-cascade",
        "--mode", "production",
        "--no-judge",
        "--input", str(input_path),
        "--output-dir", str(output_dir),
    ])
    assert rc == 0
    assert (output_dir / "cascade_manifest.json").exists()
    assert (output_dir / "final_corpus.jsonl").exists()

    final = [json.loads(l) for l in (output_dir / "final_corpus.jsonl").read_text().splitlines() if l.strip()]
    assert [r["uid"] for r in final] == ["uid_good"]

    manifest = json.loads((output_dir / "cascade_manifest.json").read_text())
    assert [s["combo"] for s in manifest["stages"]] == [
        "codex:openai", "codex:anthropic", "claude:openai",
    ]

    out = capsys.readouterr().out
    summary = json.loads(out)
    assert summary["final_corpus"]["record_count"] == 1
    assert summary["stages"] == ["codex:openai", "codex:anthropic", "claude:openai"]


def test_cli_cascade_default_stages_match_recommended_order(tmp_path, monkeypatch, capsys):
    input_path = _seed_input(tmp_path)
    output_dir = tmp_path / "cascade"
    _install_fakes(monkeypatch)

    rc = cli.main([
        "processing", "wellposed-cascade",
        "--mode", "production",
        "--no-judge",
        "--input", str(input_path),
        "--output-dir", str(output_dir),
    ])
    assert rc == 0
    manifest = json.loads((output_dir / "cascade_manifest.json").read_text())
    assert [s["combo"] for s in manifest["config"]["stages"]] == [
        "codex:openai", "codex:anthropic", "claude:openai",
    ]


def test_cli_cascade_bad_stage_token_fails(tmp_path, monkeypatch):
    input_path = _seed_input(tmp_path)
    output_dir = tmp_path / "cascade"
    _install_fakes(monkeypatch)

    rc = cli.main([
        "processing", "wellposed-cascade",
        "--mode", "production",
        "--no-judge",
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        "--stages", "codex:openai,bogus:combo,claude:openai",
    ])
    assert rc != 0

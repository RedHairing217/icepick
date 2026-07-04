"""Adapter normalisation + planning tests — no real subprocess.

We exercise ``plan`` to verify argv shape per combo, and ``normalise``
directly by writing the per-poser raw JSON to disk in the exact shapes
the discovery synthesis documented.
"""

from __future__ import annotations

import json
from pathlib import Path

from icepick.processing.poser.base import (
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
)
from icepick.processing.poser.claude_adapter import ClaudePoserAdapter
from icepick.processing.poser.codex_adapter import CodexPoserAdapter
from icepick.processing.poser.config import (
    BUILD_CLAUDE,
    BUILD_CODEX,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    Combo,
    WellposedConfig,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _combo(build, provider):
    return Combo(build=build, provider=provider)


# --- Claude normalisation ------------------------------------------------

def test_claude_normalise_maps_pass_flag_defer_and_insufficient(tmp_path):
    payload = {
        "judge_model": "claude-opus-4-7",
        "records": [
            {"uid": "u1", "source": "s", "wellposed_status": "pass", "wellposed_score": 1.0,
             "wellposed_votes": 3, "judge_majority": "pass", "code_hit_count": 0},
            {"uid": "u2", "source": "s", "wellposed_status": "flag", "wellposed_score": 0.0,
             "flag_votes": 3},
            {"uid": "u3", "source": "s", "wellposed_status": "defer", "wellposed_score": 0.5},
            {"uid": "u4", "source": "s", "wellposed_status": "insufficient_context",
             "wellposed_score": 0.0, "insufficient_context_votes": 3},
        ],
    }
    out = _write_json(tmp_path / "claude.json", payload)
    combo = _combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    verdicts = ClaudePoserAdapter().normalise(out, input_uids=["u1", "u2", "u3", "u4"], combo=combo)
    by_uid = {v.uid: v for v in verdicts}
    assert by_uid["u1"].verdict_status == STATUS_WELL_POSED
    assert by_uid["u2"].verdict_status == STATUS_ILL_POSED
    assert by_uid["u3"].verdict_status == STATUS_DEFER
    # insufficient_context is a confirmed decision (score 0.0 upstream), NOT
    # judge uncertainty. It maps to ill_posed so it exits the pipeline
    # instead of being retried by defer-eligible downstream policies.
    assert by_uid["u4"].verdict_status == STATUS_ILL_POSED
    assert by_uid["u4"].verdict_detail["original_status"] == "insufficient_context"
    assert by_uid["u1"].verdict_signals["judge_majority"] == "pass"
    # poser_name is now the combo key, not the bare build
    assert all(v.poser_name == "claude:anthropic" for v in verdicts)
    # verdict_detail carries the provider for downstream visibility
    assert all(v.verdict_detail.get("provider") == "anthropic" for v in verdicts)


def test_claude_fills_error_verdict_for_missing_uid(tmp_path):
    payload = {"records": [{"uid": "u1", "wellposed_status": "pass", "wellposed_score": 1.0}]}
    out = _write_json(tmp_path / "claude.json", payload)
    combo = _combo(BUILD_CLAUDE, PROVIDER_OPENAI)
    verdicts = ClaudePoserAdapter().normalise(out, input_uids=["u1", "u_missing"], combo=combo)
    by_uid = {v.uid: v for v in verdicts}
    assert by_uid["u_missing"].verdict_status == STATUS_ERROR
    assert "uid missing" in by_uid["u_missing"].verdict_detail["error_reason"]
    assert by_uid["u_missing"].verdict_detail["provider"] == "openai"


def test_claude_missing_file_yields_all_error(tmp_path):
    combo = _combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    verdicts = ClaudePoserAdapter().normalise(tmp_path / "nope.json", input_uids=["a", "b"], combo=combo)
    assert {v.verdict_status for v in verdicts} == {STATUS_ERROR}
    assert {v.uid for v in verdicts} == {"a", "b"}


def test_claude_invalid_json_yields_all_error(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    combo = _combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    verdicts = ClaudePoserAdapter().normalise(tmp_path / "bad.json", input_uids=["u1"], combo=combo)
    assert verdicts[0].verdict_status == STATUS_ERROR


# --- Codex normalisation -------------------------------------------------

def test_codex_normalise_maps_pass_flag_defer_error(tmp_path):
    payload = {
        "parameters": {"judge_model": "gpt-4o"},
        "records": [
            {"uid": "u1", "source": "s", "well_posedness_status": "pass",
             "well_posedness_score": 1.0, "well_posedness_check": "code",
             "signals": {"code_signal": True}},
            {"uid": "u2", "source": "s", "well_posedness_status": "flag",
             "well_posedness_score": 0.0, "well_posedness_detail": "dangling ref"},
            {"uid": "u3", "source": "s", "well_posedness_status": "defer",
             "well_posedness_score": 0.5},
            {"uid": "u4", "source": "s", "well_posedness_status": "error",
             "well_posedness_score": 0.0},
        ],
    }
    out = _write_json(tmp_path / "codex.json", payload)
    combo = _combo(BUILD_CODEX, PROVIDER_OPENAI)
    verdicts = CodexPoserAdapter().normalise(out, input_uids=["u1", "u2", "u3", "u4"], combo=combo)
    by_uid = {v.uid: v for v in verdicts}
    assert by_uid["u1"].verdict_status == STATUS_WELL_POSED
    assert by_uid["u2"].verdict_status == STATUS_ILL_POSED
    assert by_uid["u3"].verdict_status == STATUS_DEFER
    assert by_uid["u4"].verdict_status == STATUS_ERROR
    assert by_uid["u1"].verdict_signals == {"code_signal": True}
    assert all(v.poser_name == "codex:openai" for v in verdicts)
    assert all(v.verdict_detail.get("provider") == "openai" for v in verdicts)


# --- Planning argv: provider routing -------------------------------------

def test_claude_plan_uses_anthropic_provider_flag_and_anthropic_key(tmp_path):
    cfg = WellposedConfig(
        combos=[_combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC)], mode="production",
        enable_judge_tier=True, anthropic_key_file=Path("/tmp/anthro.env"),
    )
    req = ClaudePoserAdapter().plan(
        [{"source": "s", "statement": "q"}], cfg,
        _combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC), tmp_path,
    )
    assert "--provider" in req.argv
    assert "anthropic" in req.argv
    assert "--anthropic-key-file" in req.argv
    assert "--openai-key-file" not in req.argv
    assert "--judge" in req.argv


def test_claude_plan_uses_openai_provider_flag_and_openai_key(tmp_path):
    cfg = WellposedConfig(
        combos=[_combo(BUILD_CLAUDE, PROVIDER_OPENAI)], mode="production",
        enable_judge_tier=True, openai_key_file=Path("/tmp/openai.env"),
    )
    req = ClaudePoserAdapter().plan(
        [{"source": "s", "statement": "q"}], cfg,
        _combo(BUILD_CLAUDE, PROVIDER_OPENAI), tmp_path,
    )
    assert "--provider" in req.argv
    assert "openai" in req.argv
    assert "--openai-key-file" in req.argv
    assert "--anthropic-key-file" not in req.argv


def test_codex_plan_uses_judge_provider_and_single_key_env(tmp_path):
    cfg = WellposedConfig(
        combos=[_combo(BUILD_CODEX, PROVIDER_OPENAI)], mode="production",
        enable_judge_tier=True, openai_key_file=Path("/tmp/openai.env"),
    )
    req = CodexPoserAdapter().plan(
        [{"source": "s", "statement": "q"}], cfg,
        _combo(BUILD_CODEX, PROVIDER_OPENAI), tmp_path,
    )
    assert "--judge-provider" in req.argv
    assert "openai" in req.argv
    assert "--key-env" in req.argv
    # codex uses a single --key-env, never the separate --anthropic-key-file etc.
    assert "--anthropic-key-file" not in req.argv


def test_claude_plan_passes_extracted_judge_policy(tmp_path):
    """The policy flag is a first-class setting, passed to claude-poser argv."""
    for policy in ("always", "on_scanner_hit"):
        cfg = WellposedConfig(
            combos=[_combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC)], mode="production",
            enable_judge_tier=True, anthropic_key_file=Path("/tmp/anthro.env"),
            extracted_judge_policy=policy,
        )
        req = ClaudePoserAdapter().plan(
            [{"source": "s", "statement": "q"}], cfg,
            _combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC), tmp_path,
        )
        assert "--extracted-judge-policy" in req.argv
        idx = req.argv.index("--extracted-judge-policy")
        assert req.argv[idx + 1] == policy


def test_wellposed_config_defaults_extracted_judge_policy_to_always():
    """Regression guard: the safer default must survive."""
    cfg = WellposedConfig(combos=[_combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC)])
    assert cfg.extracted_judge_policy == "always"


def test_wellposed_config_rejects_unknown_policy():
    import pytest
    from icepick.config import ConfigError
    cfg = WellposedConfig(
        combos=[_combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC)],
        mode="production",
        enable_judge_tier=False,  # avoid the anthropic_key_file requirement
        extracted_judge_policy="bogus",
    )
    with pytest.raises(ConfigError, match="extracted_judge_policy"):
        cfg.validate()


def test_wellposed_config_echoes_extracted_judge_policy():
    """The manifest must record the operator's choice for audit."""
    cfg = WellposedConfig(
        combos=[_combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC)],
        extracted_judge_policy="on_scanner_hit",
    )
    echo = cfg.echo()
    assert echo["extracted_judge_policy"] == "on_scanner_hit"


def test_codex_plan_omits_judge_in_flow_testing(tmp_path):
    cfg = WellposedConfig(
        combos=[_combo(BUILD_CODEX, PROVIDER_ANTHROPIC)], mode="flow_testing",
        enable_judge_tier=False, calibration_sheet=Path("/tmp/c.jsonl"),
    )
    req = CodexPoserAdapter().plan(
        [{"source": "s", "statement": "q"}], cfg,
        _combo(BUILD_CODEX, PROVIDER_ANTHROPIC), tmp_path,
    )
    assert "--judge" not in req.argv
    assert "--mode" in req.argv
    assert "flow_testing" in req.argv


def test_plan_writes_uid_injected_input_to_disk(tmp_path):
    cfg = WellposedConfig(
        combos=[_combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC)], mode="production",
        enable_judge_tier=False,
    )
    req = ClaudePoserAdapter().plan(
        [{"source": "s", "statement": "q"}], cfg,
        _combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC), tmp_path,
    )
    with req.input_path.open() as fh:
        first = json.loads(fh.readline())
    assert "uid" in first and len(first["uid"]) == 32


def test_plan_writes_per_combo_filenames_so_fleet_does_not_collide(tmp_path):
    """Output files must be combo-scoped so claude:anthropic doesn't clobber claude:openai."""
    cfg = WellposedConfig(
        combos=[
            _combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC),
            _combo(BUILD_CLAUDE, PROVIDER_OPENAI),
        ],
        mode="production", enable_judge_tier=False,
    )
    a = ClaudePoserAdapter().plan(
        [{"source": "s", "statement": "q"}], cfg,
        _combo(BUILD_CLAUDE, PROVIDER_ANTHROPIC), tmp_path,
    )
    b = ClaudePoserAdapter().plan(
        [{"source": "s", "statement": "q"}], cfg,
        _combo(BUILD_CLAUDE, PROVIDER_OPENAI), tmp_path,
    )
    assert a.output_path != b.output_path
    assert a.input_path != b.input_path
    assert a.cache_path != b.cache_path

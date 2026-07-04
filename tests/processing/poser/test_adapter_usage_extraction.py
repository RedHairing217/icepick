"""Adapters must extract usage from each poser's per-record output shape."""

from __future__ import annotations

import json

from icepick.processing.poser.claude_adapter import ClaudePoserAdapter
from icepick.processing.poser.codex_adapter import CodexPoserAdapter
from icepick.processing.poser.config import (
    BUILD_CLAUDE, BUILD_CODEX, PROVIDER_ANTHROPIC, PROVIDER_OPENAI, Combo,
)


def test_claude_adapter_pulls_usage_from_judge_block(tmp_path):
    """Claude_Poser emits usage under records[i].judge.usage."""
    payload = {
        "judge_model": "claude-opus-4-7",
        "records": [{
            "uid": "u1",
            "source": "s",
            "wellposed_status": "pass",
            "wellposed_score": 1.0,
            "judge": {
                "provider": "anthropic", "model": "claude-opus-4-7",
                "samples": [], "majority_verdict": "pass",
                "wellposed_votes": 3, "flag_votes": 0,
                "insufficient_context_votes": 0, "error_votes": 0,
                "usage": {
                    "input_tokens": 1500, "output_tokens": 400,
                    "samples_with_usage": 3,
                },
            },
        }],
    }
    out = tmp_path / "claude.json"
    out.write_text(json.dumps(payload))

    combo = Combo(build=BUILD_CLAUDE, provider=PROVIDER_ANTHROPIC)
    verdicts = ClaudePoserAdapter().normalise(out, input_uids=["u1"], combo=combo)
    assert len(verdicts) == 1
    assert verdicts[0].verdict_signals["usage"]["input_tokens"] == 1500
    assert verdicts[0].verdict_signals["usage"]["output_tokens"] == 400


def test_claude_adapter_omits_usage_when_judge_block_lacks_it(tmp_path):
    payload = {"records": [{
        "uid": "u1", "source": "s",
        "wellposed_status": "pass", "wellposed_score": 1.0,
        "judge": None,  # no judge → no usage
    }]}
    out = tmp_path / "claude.json"
    out.write_text(json.dumps(payload))
    combo = Combo(build=BUILD_CLAUDE, provider=PROVIDER_ANTHROPIC)
    verdicts = ClaudePoserAdapter().normalise(out, input_uids=["u1"], combo=combo)
    assert "usage" not in verdicts[0].verdict_signals


def test_codex_adapter_pulls_usage_from_signals_judge_block(tmp_path):
    """Codex_Poser emits usage under records[i].signals.judge.usage."""
    payload = {
        "parameters": {"judge_model": "gpt-4o"},
        "records": [{
            "uid": "u1", "source": "s",
            "well_posedness_status": "pass", "well_posedness_score": 1.0,
            "signals": {
                "soft_context": {},
                "judge": {
                    "samples_requested": 3, "samples_parsed": 3, "uphold": 2,
                    "ill_posed_votes": 0, "insufficient_context_votes": 0,
                    "insufficient_context": False, "votes": [],
                    "usage": {
                        "input_tokens": 2000, "output_tokens": 600,
                        "samples_with_usage": 3,
                    },
                },
            },
        }],
    }
    out = tmp_path / "codex.json"
    out.write_text(json.dumps(payload))

    combo = Combo(build=BUILD_CODEX, provider=PROVIDER_OPENAI)
    verdicts = CodexPoserAdapter().normalise(out, input_uids=["u1"], combo=combo)
    assert len(verdicts) == 1
    assert verdicts[0].verdict_signals["usage"]["input_tokens"] == 2000
    assert verdicts[0].verdict_signals["usage"]["output_tokens"] == 600


def test_codex_adapter_omits_usage_when_signals_judge_lacks_it(tmp_path):
    payload = {"records": [{
        "uid": "u1", "source": "s",
        "well_posedness_status": "pass", "well_posedness_score": 1.0,
        "signals": {"judge": {"samples_requested": 0}},  # no usage key
    }]}
    out = tmp_path / "codex.json"
    out.write_text(json.dumps(payload))
    combo = Combo(build=BUILD_CODEX, provider=PROVIDER_OPENAI)
    verdicts = CodexPoserAdapter().normalise(out, input_uids=["u1"], combo=combo)
    assert "usage" not in verdicts[0].verdict_signals

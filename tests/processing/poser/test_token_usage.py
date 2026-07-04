"""Poser-stage token usage rollup.

Tests the runner-side aggregation of verdict_signals.usage that the
adapters now bubble up from the patched posers.
"""

from __future__ import annotations

import json

from icepick.processing.poser.base import (
    STATUS_WELL_POSED,
    PoserRequest,
    PoserRunResult,
    PoserVerdict,
)
from icepick.processing.poser.config import (
    BUILD_CLAUDE,
    BUILD_CODEX,
    POLICY_UNION,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    Combo,
    WellposedConfig,
)
from icepick.processing.poser.runner import run as run_wellposed


def _c(build, provider):
    return Combo(build=build, provider=provider)


class _FakeAdapterWithUsage:
    """Inline-injects a usage dict on every verdict it emits."""

    def __init__(self, build, usage_per_uid: dict):
        self.build = build
        self._usage = usage_per_uid

    def plan(self, records, cfg, combo, work_dir):
        work_dir.mkdir(parents=True, exist_ok=True)
        return PoserRequest(
            argv=[f"fake-{self.build}", combo.key()],
            env={},
            input_path=work_dir / f"{combo.slug()}_in.jsonl",
            output_path=work_dir / f"{combo.slug()}_out.json",
            cache_path=None, poser_name=combo.key(),
        )

    def run(self, request):
        return PoserRunResult(exit_code=0, stdout="", stderr="",
                              output_path=request.output_path, wall_clock_seconds=0.01)

    def normalise(self, raw_output_path, input_uids, *, combo):
        return [
            PoserVerdict(
                uid=uid, source="", verdict_status=STATUS_WELL_POSED,
                verdict_score=1.0, poser_name=combo.key(), poser_model="fake",
                verdict_signals={"usage": self._usage.get(uid, {})} if self._usage.get(uid) else {},
            )
            for uid in input_uids
        ]


def _records():
    return [
        {"source": "s", "statement": "good1", "uid": "u1"},
        {"source": "s", "statement": "good2", "uid": "u2"},
    ]


def test_token_usage_aggregated_across_records_in_one_combo(tmp_path):
    combo = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    cfg = WellposedConfig(combos=[combo], mode="production", enable_judge_tier=False,
                          output_dir=tmp_path)
    fake = _FakeAdapterWithUsage(BUILD_CLAUDE, {
        "u1": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 5},
        "u2": {"input_tokens": 200, "output_tokens": 80},
    })
    outcome = run_wellposed(cfg=cfg, records=_records(), adapter_overrides={BUILD_CLAUDE: fake})
    manifest = json.loads(outcome.manifest_path.read_text())
    usage = manifest["token_usage"]

    assert usage["per_combo"][combo.key()]["input_tokens"] == 300
    assert usage["per_combo"][combo.key()]["output_tokens"] == 130
    assert usage["per_combo"][combo.key()]["cache_read_input_tokens"] == 5
    assert usage["per_combo"][combo.key()]["records_with_usage"] == 2
    assert usage["fleet_totals"]["input_tokens"] == 300
    assert usage["fleet_totals"]["output_tokens"] == 130
    assert "estimated_cost" not in usage


def test_token_usage_per_combo_breakdown_across_fleet(tmp_path):
    """When a fleet runs multiple combos, per-combo totals should be separable."""
    a = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    b = _c(BUILD_CODEX, PROVIDER_OPENAI)
    cfg = WellposedConfig(combos=[a, b], mode="production", enable_judge_tier=False,
                          output_dir=tmp_path, comparison_policy=POLICY_UNION)
    claude = _FakeAdapterWithUsage(BUILD_CLAUDE, {
        "u1": {"input_tokens": 100, "output_tokens": 50},
        "u2": {"input_tokens": 100, "output_tokens": 50},
    })
    codex = _FakeAdapterWithUsage(BUILD_CODEX, {
        "u1": {"input_tokens": 200, "output_tokens": 80},
        "u2": {"input_tokens": 200, "output_tokens": 80},
    })

    outcome = run_wellposed(
        cfg=cfg, records=_records(),
        adapter_overrides={BUILD_CLAUDE: claude, BUILD_CODEX: codex},
    )
    usage = json.loads(outcome.manifest_path.read_text())["token_usage"]

    assert usage["per_combo"][a.key()]["input_tokens"] == 200
    assert usage["per_combo"][a.key()]["output_tokens"] == 100
    assert usage["per_combo"][b.key()]["input_tokens"] == 400
    assert usage["per_combo"][b.key()]["output_tokens"] == 160
    # Fleet totals = sum across combos
    assert usage["fleet_totals"]["input_tokens"] == 600
    assert usage["fleet_totals"]["output_tokens"] == 260


def test_token_usage_estimated_cost_when_rates_set(tmp_path):
    combo = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    cfg = WellposedConfig(
        combos=[combo], mode="production", enable_judge_tier=False,
        output_dir=tmp_path,
        cost_per_input_mtok=15.0,
        cost_per_output_mtok=75.0,
    )
    fake = _FakeAdapterWithUsage(BUILD_CLAUDE, {
        "u1": {"input_tokens": 500_000, "output_tokens": 200_000},
        "u2": {"input_tokens": 500_000, "output_tokens": 200_000},
    })
    outcome = run_wellposed(cfg=cfg, records=_records(),
                            adapter_overrides={BUILD_CLAUDE: fake})
    cost = json.loads(outcome.manifest_path.read_text())["token_usage"]["estimated_cost"]

    # 1M input × $15/M = $15.00; 400K output × $75/M = $30.00 → $45.00 total
    assert cost["input_usd"] == 15.0
    assert cost["output_usd"] == 30.0
    assert cost["total_usd"] == 45.0
    assert cost["is_estimate"] is True


def test_token_usage_zero_when_no_verdicts_carry_usage(tmp_path):
    """No-usage adapters (e.g. flow_testing replay) shouldn't crash the rollup."""
    combo = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    cfg = WellposedConfig(combos=[combo], mode="production", enable_judge_tier=False,
                          output_dir=tmp_path)
    fake = _FakeAdapterWithUsage(BUILD_CLAUDE, {})  # no usage on any uid
    outcome = run_wellposed(cfg=cfg, records=_records(),
                            adapter_overrides={BUILD_CLAUDE: fake})
    usage = json.loads(outcome.manifest_path.read_text())["token_usage"]

    assert usage["fleet_totals"]["input_tokens"] == 0
    assert usage["fleet_totals"]["records_with_usage"] == 0
    assert usage["per_combo"][combo.key()]["records_with_usage"] == 0

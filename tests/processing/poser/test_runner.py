"""Runner orchestration with fake adapters — no subprocess.

Verifies:
- single-combo: one normalised file is the gate input directly
- multi-combo fleet (2 or 4 combos): parallel execution, comparison file,
  combined gate-input
- intersect / union / majority / prefer policies all work
- subprocess failure on one combo synthesises 'error' verdicts without
  aborting other combos
- manifest echoes config + per-combo subprocess details
"""

from __future__ import annotations

import json

import pytest

from icepick.processing.poser.base import (
    STATUS_ERROR,
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserRequest,
    PoserRunResult,
    PoserVerdict,
)
from icepick.processing.poser.config import (
    BUILD_CLAUDE,
    BUILD_CODEX,
    POLICY_INTERSECT,
    POLICY_MAJORITY,
    POLICY_PREFER,
    POLICY_UNION,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    Combo,
    WellposedConfig,
    all_combos,
)
from icepick.processing.poser.runner import run as run_wellposed


def _c(build, provider):
    return Combo(build=build, provider=provider)


def _wp(uid, status, combo_key, model="fake"):
    return PoserVerdict(
        uid=uid, source="s", verdict_status=status,
        verdict_score=1.0 if status == STATUS_WELL_POSED else 0.0,
        poser_name=combo_key, poser_model=model,
    )


class _FakeAdapter:
    """Skips subprocess. Returns (combo_key, uid) -> verdict from a lookup table."""

    def __init__(self, build, verdicts_by_combo_uid, *, model="fake-model", crash_combos=()):
        self.build = build
        self._verdicts = verdicts_by_combo_uid
        self._model = model
        self._crash_combos = set(crash_combos)

    def plan(self, records, cfg, combo, work_dir):
        work_dir.mkdir(parents=True, exist_ok=True)
        return PoserRequest(
            argv=[f"fake-{self.build}", "score", "--combo", combo.key()],
            env={},
            input_path=work_dir / f"{combo.slug()}_input.jsonl",
            output_path=work_dir / f"{combo.slug()}_verdicts.json",
            cache_path=work_dir / f"{combo.slug()}_judge_cache.jsonl",
            poser_name=combo.key(),
        )

    def run(self, request):
        combo_key = next((a for a in request.argv if ":" in a), "")
        if combo_key in self._crash_combos:
            return PoserRunResult(
                exit_code=1, stdout="", stderr=f"boom on {combo_key}",
                output_path=request.output_path, wall_clock_seconds=0.01,
            )
        return PoserRunResult(
            exit_code=0, stdout="", stderr="",
            output_path=request.output_path, wall_clock_seconds=0.01,
        )

    def normalise(self, raw_output_path, input_uids, *, combo):
        if combo.key() in self._crash_combos:
            return [
                PoserVerdict(uid=uid, source="", verdict_status=STATUS_ERROR,
                             verdict_score=0.0, poser_name=combo.key(), poser_model="")
                for uid in input_uids
            ]
        out = []
        for uid in input_uids:
            v = self._verdicts.get((combo.key(), uid))
            if v is None:
                out.append(PoserVerdict(uid=uid, source="", verdict_status=STATUS_ERROR,
                                        verdict_score=0.0, poser_name=combo.key(),
                                        poser_model=self._model))
            else:
                out.append(v)
        return out


def _records():
    return [
        {"source": "s", "statement": "good", "uid": "u1"},
        {"source": "s", "statement": "bad", "uid": "u2"},
    ]


def test_single_combo_uses_its_normalised_file_as_gate_input(tmp_path):
    combo = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    cfg = WellposedConfig(combos=[combo], mode="production", enable_judge_tier=False,
                          output_dir=tmp_path)
    fake = _FakeAdapter(BUILD_CLAUDE, {
        (combo.key(), "u1"): _wp("u1", STATUS_WELL_POSED, combo.key()),
        (combo.key(), "u2"): _wp("u2", STATUS_ILL_POSED, combo.key()),
    })
    outcome = run_wellposed(cfg=cfg, records=_records(),
                            adapter_overrides={BUILD_CLAUDE: fake})
    assert outcome.gate_input_path == outcome.normalised_paths[combo.key()]
    assert outcome.comparison_path is None
    rows = [json.loads(l) for l in outcome.gate_input_path.read_text().splitlines() if l.strip()]
    assert sorted(r["verdict_status"] for r in rows) == [STATUS_ILL_POSED, STATUS_WELL_POSED]


def test_two_combo_fleet_intersect_admits_only_unanimous_pass(tmp_path):
    a = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    b = _c(BUILD_CODEX, PROVIDER_OPENAI)
    cfg = WellposedConfig(combos=[a, b], mode="production", enable_judge_tier=False,
                          output_dir=tmp_path, comparison_policy=POLICY_INTERSECT)
    claude = _FakeAdapter(BUILD_CLAUDE, {
        (a.key(), "u1"): _wp("u1", STATUS_WELL_POSED, a.key()),
        (a.key(), "u2"): _wp("u2", STATUS_WELL_POSED, a.key()),
    })
    codex = _FakeAdapter(BUILD_CODEX, {
        (b.key(), "u1"): _wp("u1", STATUS_WELL_POSED, b.key()),
        (b.key(), "u2"): _wp("u2", STATUS_ILL_POSED, b.key()),  # disagreement → intersect denies u2
    })
    outcome = run_wellposed(cfg=cfg, records=_records(),
                            adapter_overrides={BUILD_CLAUDE: claude, BUILD_CODEX: codex})
    rows = [json.loads(l) for l in outcome.gate_input_path.read_text().splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in rows}
    assert by_uid["u1"]["verdict_status"] == STATUS_WELL_POSED
    assert by_uid["u2"]["verdict_status"] != STATUS_WELL_POSED


def test_two_combo_fleet_union_admits_either_pass(tmp_path):
    a = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    b = _c(BUILD_CODEX, PROVIDER_OPENAI)
    cfg = WellposedConfig(combos=[a, b], mode="production", enable_judge_tier=False,
                          output_dir=tmp_path, comparison_policy=POLICY_UNION)
    claude = _FakeAdapter(BUILD_CLAUDE, {
        (a.key(), "u1"): _wp("u1", STATUS_WELL_POSED, a.key()),
        (a.key(), "u2"): _wp("u2", STATUS_ILL_POSED, a.key()),
    })
    codex = _FakeAdapter(BUILD_CODEX, {
        (b.key(), "u1"): _wp("u1", STATUS_ILL_POSED, b.key()),
        (b.key(), "u2"): _wp("u2", STATUS_WELL_POSED, b.key()),
    })
    outcome = run_wellposed(cfg=cfg, records=_records(),
                            adapter_overrides={BUILD_CLAUDE: claude, BUILD_CODEX: codex})
    rows = [json.loads(l) for l in outcome.gate_input_path.read_text().splitlines() if l.strip()]
    assert all(r["verdict_status"] == STATUS_WELL_POSED for r in rows)


def test_full_four_combo_fleet_with_majority_policy(tmp_path):
    """All four (build, provider) combos run; majority decides per uid."""
    combos = all_combos()
    cfg = WellposedConfig(combos=combos, mode="production", enable_judge_tier=False,
                          output_dir=tmp_path, comparison_policy=POLICY_MAJORITY)

    # u1: 3 pass, 1 fail → majority admits
    # u2: 2 pass, 2 fail → not majority (strictly more than half)
    verdicts_claude = {
        ("claude:anthropic", "u1"): _wp("u1", STATUS_WELL_POSED, "claude:anthropic"),
        ("claude:openai", "u1"):    _wp("u1", STATUS_WELL_POSED, "claude:openai"),
        ("claude:anthropic", "u2"): _wp("u2", STATUS_WELL_POSED, "claude:anthropic"),
        ("claude:openai", "u2"):    _wp("u2", STATUS_ILL_POSED, "claude:openai"),
    }
    verdicts_codex = {
        ("codex:anthropic", "u1"): _wp("u1", STATUS_WELL_POSED, "codex:anthropic"),
        ("codex:openai", "u1"):    _wp("u1", STATUS_ILL_POSED, "codex:openai"),
        ("codex:anthropic", "u2"): _wp("u2", STATUS_ILL_POSED, "codex:anthropic"),
        ("codex:openai", "u2"):    _wp("u2", STATUS_WELL_POSED, "codex:openai"),
    }
    claude = _FakeAdapter(BUILD_CLAUDE, verdicts_claude)
    codex = _FakeAdapter(BUILD_CODEX, verdicts_codex)

    outcome = run_wellposed(cfg=cfg, records=_records(),
                            adapter_overrides={BUILD_CLAUDE: claude, BUILD_CODEX: codex})
    assert len(outcome.normalised_paths) == 4
    rows = [json.loads(l) for l in outcome.gate_input_path.read_text().splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in rows}
    assert by_uid["u1"]["verdict_status"] == STATUS_WELL_POSED
    assert by_uid["u2"]["verdict_status"] != STATUS_WELL_POSED
    # per-combo audit trail in verdict_detail
    assert set(by_uid["u1"]["verdict_detail"]["per_combo"].keys()) == {
        "claude:anthropic", "claude:openai", "codex:anthropic", "codex:openai",
    }


def test_prefer_policy_uses_named_combo_normalised_file(tmp_path):
    a = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    b = _c(BUILD_CODEX, PROVIDER_OPENAI)
    cfg = WellposedConfig(
        combos=[a, b], mode="production", enable_judge_tier=False,
        output_dir=tmp_path, comparison_policy=f"{POLICY_PREFER}codex:openai",
    )
    claude = _FakeAdapter(BUILD_CLAUDE, {
        (a.key(), "u1"): _wp("u1", STATUS_WELL_POSED, a.key()),
        (a.key(), "u2"): _wp("u2", STATUS_WELL_POSED, a.key()),
    })
    codex = _FakeAdapter(BUILD_CODEX, {
        (b.key(), "u1"): _wp("u1", STATUS_ILL_POSED, b.key()),
        (b.key(), "u2"): _wp("u2", STATUS_ILL_POSED, b.key()),
    })
    outcome = run_wellposed(cfg=cfg, records=_records(),
                            adapter_overrides={BUILD_CLAUDE: claude, BUILD_CODEX: codex})
    assert outcome.gate_input_path == outcome.normalised_paths["codex:openai"]
    rows = [json.loads(l) for l in outcome.gate_input_path.read_text().splitlines() if l.strip()]
    assert all(r["verdict_status"] == STATUS_ILL_POSED for r in rows)


def test_one_combo_crash_does_not_abort_others(tmp_path):
    a = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    b = _c(BUILD_CLAUDE, PROVIDER_OPENAI)
    cfg = WellposedConfig(combos=[a, b], mode="production", enable_judge_tier=False,
                          output_dir=tmp_path, comparison_policy=POLICY_UNION)
    fake = _FakeAdapter(BUILD_CLAUDE, {
        (a.key(), "u1"): _wp("u1", STATUS_WELL_POSED, a.key()),
        (a.key(), "u2"): _wp("u2", STATUS_WELL_POSED, a.key()),
    }, crash_combos=[b.key()])
    outcome = run_wellposed(cfg=cfg, records=_records(),
                            adapter_overrides={BUILD_CLAUDE: fake})
    manifest = json.loads(outcome.manifest_path.read_text())
    runs_by_combo = {r["combo"]: r for r in manifest["subprocess_runs"]}
    assert runs_by_combo[a.key()]["exit_code"] == 0
    assert runs_by_combo[b.key()]["exit_code"] == 1
    # claude:anthropic still admits u1+u2 → union policy includes them
    rows = [json.loads(l) for l in outcome.gate_input_path.read_text().splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in rows}
    assert by_uid["u1"]["verdict_status"] == STATUS_WELL_POSED


def test_manifest_records_per_combo_subprocess_details(tmp_path):
    a = _c(BUILD_CLAUDE, PROVIDER_ANTHROPIC)
    cfg = WellposedConfig(combos=[a], mode="production", enable_judge_tier=False,
                          output_dir=tmp_path)
    fake = _FakeAdapter(BUILD_CLAUDE, {
        (a.key(), "u1"): _wp("u1", STATUS_WELL_POSED, a.key()),
        (a.key(), "u2"): _wp("u2", STATUS_ILL_POSED, a.key()),
    })
    outcome = run_wellposed(cfg=cfg, records=_records(),
                            adapter_overrides={BUILD_CLAUDE: fake})
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["config"]["combos"] == [a.key()]
    assert manifest["counts"][STATUS_WELL_POSED] == 1
    assert manifest["counts"][STATUS_ILL_POSED] == 1
    assert manifest["subprocess_runs"][0]["combo"] == a.key()


def test_validation_fires_inside_runner(tmp_path):
    cfg = WellposedConfig(combos=[], mode="production", enable_judge_tier=False,
                          output_dir=tmp_path)
    with pytest.raises(Exception):  # ConfigError
        run_wellposed(cfg=cfg, records=_records())

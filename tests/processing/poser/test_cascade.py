"""Unit tests for the 3-stage wellposed cascade.

Fakes substitute for the ClaudePoserAdapter / CodexPoserAdapter so no
subprocess or network is touched. Each fake is keyed by (combo, uid) and
optionally by attempt number to exercise retries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick.config import ConfigError
from icepick.processing.poser.base import (
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserRequest,
    PoserRunResult,
    PoserVerdict,
)
from icepick.processing.poser.cascade import (
    CascadeConfig,
    StageSpec,
    parse_stages,
    run_cascade,
)
from icepick.processing.poser.config import (
    BUILD_CLAUDE,
    BUILD_CODEX,
    Combo,
    parse_combo,
)


def _c(build, provider):
    return Combo(build=build, provider=provider)


def _wp(uid, status, combo_key, *, usage=None):
    signals = {}
    if usage is not None:
        signals["usage"] = usage
    return PoserVerdict(
        uid=uid,
        source="s",
        verdict_status=status,
        verdict_score=1.0 if status == STATUS_WELL_POSED else 0.0,
        poser_name=combo_key,
        poser_model="fake",
        verdict_signals=signals,
    )


class _RoutingFakeAdapter:
    """Skips subprocess. Keyed by (combo.key(), uid, attempt).

    ``verdicts_by_key`` maps ``(combo_key, uid) -> [verdict_per_attempt, ...]``.
    Attempts consume verdicts in order; if the list is exhausted, the last
    entry is reused (so a single-entry list behaves like "return this every
    attempt").

    A single instance is shared across all combos of one build — the
    routing key includes combo.key() so codex:openai and codex:anthropic
    never collide, matching the runner's build-keyed adapter override
    contract.
    """

    def __init__(self, build, verdicts_by_key):
        self.build = build
        self._verdicts = verdicts_by_key
        # attempt tracking: (combo_key, uid) -> next attempt idx
        self._attempts: dict = {}

    def plan(self, records, cfg, combo, work_dir):
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        return PoserRequest(
            argv=[f"fake-{self.build}", "score", "--combo", combo.key()],
            env={},
            input_path=Path(work_dir) / f"{combo.slug()}_input.jsonl",
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
        combo_key = combo.key()
        for uid in input_uids:
            key = (combo_key, uid)
            options = self._verdicts.get(key)
            if not options:
                out.append(PoserVerdict(
                    uid=uid, source="", verdict_status=STATUS_ERROR,
                    verdict_score=0.0, poser_name=combo_key, poser_model="fake",
                    verdict_detail={"error_reason": "no fake verdict configured"},
                ))
                continue
            idx = self._attempts.get(key, 0)
            verdict = options[idx if idx < len(options) else -1]
            self._attempts[key] = idx + 1
            out.append(verdict)
        return out


def _records(n=3):
    """3 records with predictable ids: uid_good, uid_mid, uid_bad."""
    labels = ["uid_good", "uid_mid", "uid_bad"][:n]
    return [{"source": "s", "statement": lbl, "uid": lbl} for lbl in labels]


# --- config / validation -------------------------------------------------


def test_cascade_config_rejects_empty_stages():
    cfg = CascadeConfig(stages=[])
    with pytest.raises(ConfigError):
        cfg.validate()


def test_cascade_config_rejects_bad_stage_indices():
    cfg = CascadeConfig(
        stages=[StageSpec(index=2, combo=_c(BUILD_CLAUDE, "openai"))],
        mode="production", enable_judge_tier=False,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_cascade_config_rejects_negative_retries(tmp_path):
    cfg = CascadeConfig(
        stages=parse_stages(["claude:openai"]),
        mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=-1,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_cascade_config_rejects_flow_testing_without_calibration(tmp_path):
    cfg = CascadeConfig(
        stages=parse_stages(["claude:openai"]),
        mode="flow_testing", output_dir=tmp_path, enable_judge_tier=False,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


def test_parse_stages_indexes_1_based():
    stages = parse_stages(["codex:openai", "codex:anthropic", "claude:openai"])
    assert [s.index for s in stages] == [1, 2, 3]
    assert stages[0].combo.key() == "codex:openai"
    assert stages[1].combo.slug() == "codex_anthropic"


# --- happy path / filtering ---------------------------------------------


def test_cascade_happy_path_three_stages_unanimous(tmp_path):
    """3 records, only uid_good survives all 3 stages."""
    stages = parse_stages(["codex:openai", "codex:anthropic", "claude:openai"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=0,
    )
    verdicts_codex = {
        # uid_good passes both codex stages, uid_mid passes only codex:openai,
        # uid_bad fails codex:openai outright.
        ("codex:openai", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "codex:openai")],
        ("codex:openai", "uid_mid"):  [_wp("uid_mid",  STATUS_WELL_POSED, "codex:openai")],
        ("codex:openai", "uid_bad"):  [_wp("uid_bad",  STATUS_ILL_POSED, "codex:openai")],
        ("codex:anthropic", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "codex:anthropic")],
        ("codex:anthropic", "uid_mid"):  [_wp("uid_mid",  STATUS_ILL_POSED, "codex:anthropic")],
    }
    verdicts_claude = {
        ("claude:openai", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "claude:openai")],
    }
    codex = _RoutingFakeAdapter(BUILD_CODEX, verdicts_codex)
    claude = _RoutingFakeAdapter(BUILD_CLAUDE, verdicts_claude)
    outcome = run_cascade(
        cfg=cfg,
        records=_records(),
        adapter_overrides={BUILD_CODEX: codex, BUILD_CLAUDE: claude},
    )

    final = [json.loads(l) for l in outcome.final_corpus_path.read_text().splitlines() if l.strip()]
    assert [r["uid"] for r in final] == ["uid_good"]
    assert outcome.final_corpus_count == 1
    assert outcome.overall_counts["initial_record_count"] == 3
    assert outcome.overall_counts["after_stage_1"] == 2
    assert outcome.overall_counts["after_stage_2"] == 1
    assert outcome.overall_counts["after_stage_3"] == 1
    assert outcome.overall_counts["dropped_total"] == 2


def test_cascade_drops_ill_posed_before_next_stage(tmp_path):
    """uid_bad rejected at stage 1 never appears in stage 2's normalised.jsonl."""
    stages = parse_stages(["codex:openai", "codex:anthropic"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=0,
    )
    verdicts = {
        ("codex:openai", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "codex:openai")],
        ("codex:openai", "uid_bad"):  [_wp("uid_bad",  STATUS_ILL_POSED, "codex:openai")],
        ("codex:anthropic", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "codex:anthropic")],
        # deliberately NO entry for uid_bad at stage 2 — if the cascade
        # incorrectly forwards it, the fake will emit STATUS_ERROR and this
        # test would still pass survivor-wise; assert on file contents to
        # prove it never reached the stage.
    }
    codex = _RoutingFakeAdapter(BUILD_CODEX, verdicts)
    outcome = run_cascade(
        cfg=cfg, records=_records(n=2),
        adapter_overrides={BUILD_CODEX: codex},
    )

    stage2_norm = outcome.stages[1].normalised_path
    stage2_uids = {json.loads(l)["uid"] for l in stage2_norm.read_text().splitlines() if l.strip()}
    assert stage2_uids == {"uid_good"}


def test_cascade_defer_does_not_pass_stage(tmp_path):
    """A defer verdict at stage 1 rejects the record — cascade requires well_posed."""
    stages = parse_stages(["codex:openai", "codex:anthropic"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=0,
    )
    verdicts = {
        ("codex:openai", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "codex:openai")],
        ("codex:openai", "uid_mid"):  [_wp("uid_mid",  STATUS_DEFER, "codex:openai")],
        ("codex:anthropic", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "codex:anthropic")],
    }
    codex = _RoutingFakeAdapter(BUILD_CODEX, verdicts)
    outcome = run_cascade(
        cfg=cfg, records=_records(n=2),
        adapter_overrides={BUILD_CODEX: codex},
    )

    assert outcome.stages[0].counts.get(STATUS_DEFER) == 1
    stage2_uids = {json.loads(l)["uid"] for l in outcome.stages[1].normalised_path.read_text().splitlines() if l.strip()}
    assert "uid_mid" not in stage2_uids


# --- retries -------------------------------------------------------------


def test_cascade_retries_transient_error_then_succeeds(tmp_path):
    """Attempt 1 errors on uid_good, attempt 2 well_posed → survives stage.

    Also verifies the merged normalised.jsonl carries the FINAL (attempt 2)
    verdict, not the errored attempt-1 verdict — proves last-write-wins.
    """
    stages = parse_stages(["codex:openai"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=2,
        retry_base_delay=1.0, retry_max_delay=8.0,
    )
    verdicts = {
        ("codex:openai", "uid_good"): [
            PoserVerdict(uid="uid_good", source="", verdict_status=STATUS_ERROR,
                         verdict_score=0.0, poser_name="codex:openai",
                         poser_model="fake",
                         verdict_detail={"error_reason": "transient"}),
            _wp("uid_good", STATUS_WELL_POSED, "codex:openai"),
        ],
    }
    codex = _RoutingFakeAdapter(BUILD_CODEX, verdicts)
    outcome = run_cascade(
        cfg=cfg, records=_records(n=1),
        adapter_overrides={BUILD_CODEX: codex},
        sleep_fn=lambda _s: None,
    )

    assert outcome.final_corpus_count == 1
    assert outcome.stages[0].counts.get(STATUS_WELL_POSED) == 1
    assert outcome.stages[0].counts.get(STATUS_ERROR) is None
    events = outcome.stages[0].retry_events
    assert len(events) == 1
    assert events[0]["uid"] == "uid_good"
    assert events[0]["attempt"] == 1
    assert events[0]["next_attempt"] == 2
    assert isinstance(events[0]["sleep_seconds"], (int, float))
    assert events[0]["sleep_seconds"] > 0

    # Verify the merged normalised.jsonl actually reflects the successful
    # attempt-2 verdict (last-write-wins), not the errored attempt-1 verdict.
    rows = [json.loads(l) for l in outcome.stages[0].normalised_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["uid"] == "uid_good"
    assert rows[0]["verdict_status"] == STATUS_WELL_POSED


def test_cascade_retry_exhaustion_marks_error_final(tmp_path):
    """All attempts error → uid stays errored, doesn't reach next stage."""
    stages = parse_stages(["codex:openai", "codex:anthropic"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=2,
        retry_base_delay=0.0, retry_max_delay=0.0,
    )
    error_v = PoserVerdict(
        uid="uid_good", source="", verdict_status=STATUS_ERROR,
        verdict_score=0.0, poser_name="codex:openai", poser_model="fake",
        verdict_detail={"error_reason": "persistent"},
    )
    verdicts = {
        ("codex:openai", "uid_good"): [error_v, error_v, error_v],
    }
    codex = _RoutingFakeAdapter(BUILD_CODEX, verdicts)
    outcome = run_cascade(
        cfg=cfg, records=_records(n=1),
        adapter_overrides={BUILD_CODEX: codex},
        sleep_fn=lambda _s: None,
    )

    assert outcome.stages[0].counts.get(STATUS_ERROR) == 1
    assert outcome.overall_counts["after_stage_1"] == 0
    assert outcome.final_corpus_count == 0
    # last event marks the exhaustion
    last_event = outcome.stages[0].retry_events[-1]
    assert last_event["resolved"] is False
    assert last_event["final_status"] == STATUS_ERROR


def test_cascade_zero_retries_never_calls_sleep(tmp_path):
    """max_retries=0 means one attempt only; error verdicts stick immediately."""
    stages = parse_stages(["codex:openai"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=0,
    )
    error_v = PoserVerdict(
        uid="uid_good", source="", verdict_status=STATUS_ERROR,
        verdict_score=0.0, poser_name="codex:openai", poser_model="fake",
    )
    verdicts = {("codex:openai", "uid_good"): [error_v]}
    sleep_calls = []
    codex = _RoutingFakeAdapter(BUILD_CODEX, verdicts)
    outcome = run_cascade(
        cfg=cfg, records=_records(n=1),
        adapter_overrides={BUILD_CODEX: codex},
        sleep_fn=lambda s: sleep_calls.append(s),
    )

    assert sleep_calls == []
    assert outcome.stages[0].counts.get(STATUS_ERROR) == 1


# --- reordering / config surface ----------------------------------------


def test_cascade_stage_reordering_reflected_in_subdirs(tmp_path):
    """CascadeConfig.stages order controls per-stage subdir naming and manifest ordering."""
    stages = parse_stages(["claude:openai", "codex:openai"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=0,
    )
    verdicts_claude = {
        ("claude:openai", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "claude:openai")],
    }
    verdicts_codex = {
        ("codex:openai", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "codex:openai")],
    }
    outcome = run_cascade(
        cfg=cfg, records=_records(n=1),
        adapter_overrides={
            BUILD_CLAUDE: _RoutingFakeAdapter(BUILD_CLAUDE, verdicts_claude),
            BUILD_CODEX: _RoutingFakeAdapter(BUILD_CODEX, verdicts_codex),
        },
    )
    assert (tmp_path / "stage_1_claude_openai").is_dir()
    assert (tmp_path / "stage_2_codex_openai").is_dir()
    manifest = json.loads(outcome.manifest_path.read_text())
    assert [s["combo"] for s in manifest["stages"]] == ["claude:openai", "codex:openai"]


# --- token usage / cost --------------------------------------------------


def test_cascade_aggregates_token_usage_and_cost(tmp_path):
    stages = parse_stages(["codex:openai", "claude:openai"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=0,
        cost_per_input_mtok=1.0, cost_per_output_mtok=5.0,
    )
    usage_codex = {"input_tokens": 100_000, "output_tokens": 20_000}
    usage_claude = {"input_tokens": 50_000, "output_tokens": 10_000}
    verdicts_codex = {
        ("codex:openai", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "codex:openai", usage=usage_codex)],
    }
    verdicts_claude = {
        ("claude:openai", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "claude:openai", usage=usage_claude)],
    }
    outcome = run_cascade(
        cfg=cfg, records=_records(n=1),
        adapter_overrides={
            BUILD_CODEX: _RoutingFakeAdapter(BUILD_CODEX, verdicts_codex),
            BUILD_CLAUDE: _RoutingFakeAdapter(BUILD_CLAUDE, verdicts_claude),
        },
    )
    ft = outcome.total_token_usage["fleet_totals"]
    assert ft["input_tokens"] == 150_000
    assert ft["output_tokens"] == 30_000
    # 150000/1e6*1 + 30000/1e6*5 = 0.15 + 0.15 = 0.30
    assert outcome.total_estimated_cost_usd == pytest.approx(0.30)


# --- empty corpus / propagation -----------------------------------------


def test_cascade_all_records_filtered_at_stage_1_yields_empty_corpus(tmp_path):
    """Every record fails stage 1 → stage 2 runs on zero records → empty final corpus."""
    stages = parse_stages(["codex:openai", "codex:anthropic"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=0,
    )
    verdicts = {
        ("codex:openai", "uid_good"): [_wp("uid_good", STATUS_ILL_POSED, "codex:openai")],
        ("codex:openai", "uid_bad"):  [_wp("uid_bad",  STATUS_ILL_POSED, "codex:openai")],
    }
    codex = _RoutingFakeAdapter(BUILD_CODEX, verdicts)
    outcome = run_cascade(
        cfg=cfg, records=_records(n=2),
        adapter_overrides={BUILD_CODEX: codex},
    )
    assert outcome.final_corpus_count == 0
    assert outcome.final_corpus_path.exists()
    assert outcome.final_corpus_path.read_text() == ""
    assert outcome.overall_counts["after_stage_1"] == 0
    assert outcome.overall_counts["after_stage_2"] == 0
    # Stage 2 must still have a manifest entry even when it received zero records.
    assert outcome.stages[1].input_uid_count == 0
    assert outcome.stages[1].survivor_uid_count == 0


# --- token usage across retries -----------------------------------------


def test_cascade_aggregates_token_usage_across_retry_attempts(tmp_path):
    """When a stage retries, tokens spent by BOTH attempts count toward totals."""
    stages = parse_stages(["codex:openai"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=2,
        retry_base_delay=0.0, retry_max_delay=0.0,
        cost_per_input_mtok=1.0, cost_per_output_mtok=5.0,
    )
    # Attempt 1: error verdict with usage (some tokens spent before the failure).
    # Attempt 2: well_posed verdict with usage.
    usage_attempt_1 = {"input_tokens": 40_000, "output_tokens": 5_000}
    usage_attempt_2 = {"input_tokens": 30_000, "output_tokens": 8_000}
    err_v = PoserVerdict(
        uid="uid_good", source="", verdict_status=STATUS_ERROR,
        verdict_score=0.0, poser_name="codex:openai", poser_model="fake",
        verdict_detail={"error_reason": "transient"},
        verdict_signals={"usage": usage_attempt_1},
    )
    wp_v = _wp("uid_good", STATUS_WELL_POSED, "codex:openai", usage=usage_attempt_2)
    verdicts = {("codex:openai", "uid_good"): [err_v, wp_v]}
    codex = _RoutingFakeAdapter(BUILD_CODEX, verdicts)
    outcome = run_cascade(
        cfg=cfg, records=_records(n=1),
        adapter_overrides={BUILD_CODEX: codex},
        sleep_fn=lambda _s: None,
    )
    ft = outcome.total_token_usage["fleet_totals"]
    assert ft["input_tokens"] == 70_000
    assert ft["output_tokens"] == 13_000
    # cost = 70000/1e6*1 + 13000/1e6*5 = 0.07 + 0.065 = 0.135
    assert outcome.total_estimated_cost_usd == pytest.approx(0.135)


# --- adapter regression defense -----------------------------------------


def test_cascade_synthesises_error_for_missing_adapter_verdicts(tmp_path):
    """If an adapter fails to return a verdict for a uid, cascade synthesises
    STATUS_ERROR so the record has a deterministic fate (not silently dropped)."""

    class _DroppingAdapter(_RoutingFakeAdapter):
        def normalise(self, raw_output_path, input_uids, *, combo):
            # Return verdicts only for uid_good, silently drop uid_bad.
            return [v for v in super().normalise(raw_output_path, input_uids, combo=combo)
                    if v.uid == "uid_good"]

    stages = parse_stages(["codex:openai"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=0,
    )
    verdicts = {
        ("codex:openai", "uid_good"): [_wp("uid_good", STATUS_WELL_POSED, "codex:openai")],
        ("codex:openai", "uid_bad"):  [_wp("uid_bad",  STATUS_WELL_POSED, "codex:openai")],
    }
    codex = _DroppingAdapter(BUILD_CODEX, verdicts)
    outcome = run_cascade(
        cfg=cfg, records=_records(n=2),
        adapter_overrides={BUILD_CODEX: codex},
    )
    counts = outcome.stages[0].counts
    assert counts.get(STATUS_WELL_POSED) == 1
    assert counts.get(STATUS_ERROR) == 1
    # sum(counts) must equal input_uid_count — no silent loss
    assert sum(counts.values()) == outcome.stages[0].input_uid_count == 2


# --- uid injection stability --------------------------------------------


def test_cascade_uid_injection_stable_across_stages(tmp_path):
    """Records without explicit uid get compute_uid()'d once at cascade entry
    and every stage's normalised.jsonl carries the same uid."""
    stages = parse_stages(["codex:openai", "codex:anthropic"])
    cfg = CascadeConfig(
        stages=stages, mode="production", enable_judge_tier=False,
        output_dir=tmp_path, max_retries=0,
    )

    # Compute expected uid the same way the cascade will
    from icepick.processing.poser.base import compute_uid
    expected_uid = compute_uid("s", "hello")

    verdicts = {
        ("codex:openai", expected_uid): [_wp(expected_uid, STATUS_WELL_POSED, "codex:openai")],
        ("codex:anthropic", expected_uid): [_wp(expected_uid, STATUS_WELL_POSED, "codex:anthropic")],
    }
    codex = _RoutingFakeAdapter(BUILD_CODEX, verdicts)
    outcome = run_cascade(
        cfg=cfg, records=[{"source": "s", "statement": "hello"}],
        adapter_overrides={BUILD_CODEX: codex},
    )
    assert outcome.final_corpus_count == 1
    for stage_outcome in outcome.stages:
        rows = [json.loads(l) for l in stage_outcome.normalised_path.read_text().splitlines() if l.strip()]
        assert [r["uid"] for r in rows] == [expected_uid]

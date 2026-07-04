"""Live poser-subprocess + full-pipeline integration tests.

Skipped unless ``ANTHROPIC_API_KEY`` is set in the environment AND
``claude-poser`` is installed on PATH. Run manually before a release.

Costs roughly $0.01-$0.05 per invocation. Uses one record that triggers
the dangling-reference code-tier flag so the judge actually runs.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or not shutil.which("claude-poser"),
    reason="requires ANTHROPIC_API_KEY and claude-poser binary on PATH",
)


def _input_with_dangling_ref():
    """A statement with a dangling ref — trips the scanner AND, under the
    default extracted_judge_policy='always', would reach the judge anyway.

    Kept as a dangling-ref example (rather than a semantically-ill-posed
    one) because it exercises BOTH the scanner and the judge on the same
    record: the scanner hits ride along as `code_hits` evidence in the
    judge's verdict, which is exactly the audit trail we want to see in
    the token-usage output.
    """
    return [{
        "source": "live_test",
        "uid": "live_test_dangling_001",
        "rid": 0,
        "statement": "By Theorem 3.2 above, determine the value of the constant c referenced in equation (4).",
        "answer": "0",
        "provenance": "extracted",
    }]


def test_live_poser_emits_real_token_usage(tmp_path):
    """A record with extracted provenance exercises the judge (under the
    default 'always' policy) and should populate token_usage."""
    from icepick.processing.poser.config import (
        BUILD_CLAUDE, PROVIDER_ANTHROPIC, Combo, WellposedConfig,
    )
    from icepick.processing.poser.runner import run as run_wellposed

    cfg = WellposedConfig(
        combos=[Combo(build=BUILD_CLAUDE, provider=PROVIDER_ANTHROPIC)],
        mode="production",
        output_dir=tmp_path / "out",
        anthropic_key_file=None,  # rely on env var
        enable_judge_tier=True,
        judge_samples=3,
        judge_uphold=2,
        cost_per_input_mtok=5.0,
        cost_per_output_mtok=25.0,
    )

    outcome = run_wellposed(cfg=cfg, records=_input_with_dangling_ref())
    manifest = json.loads(outcome.manifest_path.read_text())

    # 1. Subprocess completed cleanly.
    sr = manifest["subprocess_runs"][0]
    assert sr["exit_code"] == 0
    assert sr["combo"] == "claude:anthropic"

    # 2. Token usage flowed from poser stdout to icepick manifest.
    usage = manifest["token_usage"]
    assert usage["fleet_totals"]["input_tokens"] > 0, "judge should have fired"
    assert usage["fleet_totals"]["output_tokens"] > 0
    assert usage["fleet_totals"]["records_with_usage"] == 1

    # 3. Cost rollup populated.
    cost = usage["estimated_cost"]
    assert cost["total_usd"] > 0
    assert cost["is_estimate"] is True


def test_live_full_pipeline_groundtruth_to_final_corpus(tmp_path):
    """Real groundtruth + real poser, end-to-end. Uses one well-known published
    arXiv paper with a clean statement so both stages produce a passing record."""
    from icepick.processing.groundtruth.config import GroundtruthConfig
    from icepick.processing.pipeline import run as run_pipeline
    from icepick.processing.poser.config import (
        BUILD_CLAUDE, PROVIDER_ANTHROPIC, Combo, WellposedConfig,
    )

    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps({
        "source": "live_pipeline",
        "uid": "live_pl_001",
        "rid": 0,
        "statement": "Find the value of x such that x^2 = 16, where x is a positive real number.",
        "arxiv_id": "1706.03762",
        "paper_title": "Attention Is All You Need",
        "answer": "4",
        "provenance": "extracted",
    }) + "\n")

    gt_cfg = GroundtruthConfig(
        mode="production",
        output_dir=tmp_path / "out" / "groundtruth",
        anthropic_key_file=None,
        cost_per_input_mtok=5.0, cost_per_output_mtok=25.0,
    )
    poser_cfg = WellposedConfig(
        combos=[Combo(build=BUILD_CLAUDE, provider=PROVIDER_ANTHROPIC)],
        mode="production",
        output_dir=tmp_path / "out" / "wellposed",
        anthropic_key_file=None,
        enable_judge_tier=True,
        cost_per_input_mtok=5.0, cost_per_output_mtok=25.0,
    )

    outcome = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        groundtruth_cfg=gt_cfg,
        poser_cfg=poser_cfg,
    )

    # 1. Final corpus exists and contains the surviving record.
    assert outcome.final_corpus_path.exists()
    assert outcome.final_corpus_count == 1
    final = [json.loads(l) for l in outcome.final_corpus_path.read_text().splitlines() if l.strip()]
    assert final[0]["arxiv_id"] == "1706.03762"

    # 2. Both stages succeeded.
    assert outcome.groundtruth_counts.get("published") == 1
    assert outcome.poser_counts.get("well_posed") == 1

    # 3. Pipeline manifest points at both stage manifests.
    pipeline_manifest = json.loads(outcome.manifest_path.read_text())
    stage_names = {s["stage"] for s in pipeline_manifest["stages"]}
    assert stage_names == {"groundtruth", "wellposed"}

"""Live Anthropic API integration test for the groundtruth stage.

Skipped automatically unless ``ANTHROPIC_API_KEY`` is set in the
environment. Run manually before a release to catch API drift between
our mocks and the real Anthropic Messages + web_search surface.

This test costs money — roughly $0.30-$0.60 per run at Opus-4-7 rates.
It uses a single well-known paper to keep cost bounded and to make
the verdict obvious.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Skip the whole module if there's no key — keeps CI runs free.
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="set ANTHROPIC_API_KEY to run live API integration tests",
)


def test_groundtruth_correctly_identifies_a_known_published_paper(tmp_path):
    """End-to-end: real web_search call returns published verdict for AIAYN."""
    from icepick.processing.groundtruth.config import GroundtruthConfig
    from icepick.processing.groundtruth.runner import run as run_groundtruth

    records = [{
        "source": "live_test",
        "statement": "Test record for Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "paper_title": "Attention Is All You Need",
        "provenance": "extracted",
    }]
    cfg = GroundtruthConfig(
        mode="production",
        output_dir=tmp_path / "out",
        anthropic_key_file=None,  # rely on env var
        judge_samples=3,
        judge_uphold=2,
        cost_per_input_mtok=5.0,
        cost_per_output_mtok=25.0,
    )

    outcome = run_groundtruth(cfg=cfg, records=records)

    # 1. The verdict came back published.
    assert outcome.counts.get("published") == 1
    assert outcome.published_path.exists()
    pubs = [json.loads(l) for l in outcome.published_path.read_text().splitlines() if l.strip()]
    assert len(pubs) == 1
    assert pubs[0]["arxiv_id"] == "1706.03762"

    # 2. The verdict shape matches what downstream code expects.
    verdicts = [json.loads(l) for l in outcome.verdicts_path.read_text().splitlines() if l.strip()]
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v["verdict_status"] == "published"
    assert v["confidence"] in ("high", "medium", "low")
    assert isinstance(v["judge_votes"], list) and len(v["judge_votes"]) == 3
    assert v["judge_model"].startswith("claude-")
    # Evidence URLs should be real (the judge ran web_search).
    assert len(v["evidence_urls"]) >= 1
    assert any("dblp.org" in u or "nips.cc" in u or "neurips" in u.lower() for u in v["evidence_urls"])

    # 3. Token usage actually populated (i.e. real API call happened).
    manifest = json.loads(outcome.manifest_path.read_text())
    usage = manifest["token_usage"]
    assert usage["input_tokens"] > 0, "real Anthropic call should populate input_tokens"
    assert usage["output_tokens"] > 0, "real Anthropic call should populate output_tokens"
    assert usage["sample_count"] >= 1  # >=1 sample with usage (cache hits don't count)
    assert usage["papers_with_usage"] == 1

    # 4. Cost rollup arithmetic.
    cost = usage["estimated_cost"]
    expected_input_usd = usage["input_tokens"] / 1_000_000 * 5.0
    expected_output_usd = usage["output_tokens"] / 1_000_000 * 25.0
    assert cost["input_usd"] == round(expected_input_usd, 6)
    assert cost["output_usd"] == round(expected_output_usd, 6)
    assert cost["is_estimate"] is True

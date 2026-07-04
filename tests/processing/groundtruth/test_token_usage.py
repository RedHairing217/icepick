"""Token usage rollup in the groundtruth manifest."""

from __future__ import annotations

import json

from icepick.processing.groundtruth.base import (
    STATUS_PUBLISHED,
    GroundtruthVerdict,
)
from icepick.processing.groundtruth.config import GroundtruthConfig
from icepick.processing.groundtruth.runner import run as run_groundtruth


class _FakeAdapter:
    """Wires per-paper usage payloads into raw_payload.samples[*].usage."""

    def __init__(self, usage_per_paper: dict):
        self._usage = usage_per_paper

    def lookup_paper(self, *, arxiv_id, paper_title, uid_for_error_attribution):
        per_sample_usage = self._usage.get(arxiv_id, [])
        samples = [{"usage": u} for u in per_sample_usage]
        return GroundtruthVerdict(
            uid=uid_for_error_attribution, source="",
            verdict_status=STATUS_PUBLISHED, arxiv_id=arxiv_id,
            judge_model="fake", judge_votes=["published"] * 3,
            judge_majority="published", reasoning="x", confidence="high",
            raw_payload={"samples": samples},
        )


def _cfg(tmp_path, **overrides):
    base = dict(
        mode="flow_testing",
        output_dir=tmp_path / "out",
        calibration_sheet=tmp_path / "sheet.jsonl",
    )
    base.update(overrides)
    return GroundtruthConfig(**base)


def test_token_usage_sums_across_samples_and_papers(tmp_path):
    cfg = _cfg(tmp_path)
    records = [
        {"source": "rm", "statement": "x", "arxiv_id": "2403.11111",
         "provenance": "extracted", "uid": "uid_a"},
        {"source": "rm", "statement": "y", "arxiv_id": "2403.22222",
         "provenance": "extracted", "uid": "uid_b"},
    ]
    adapter = _FakeAdapter({
        "2403.11111": [
            {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 10},
            {"input_tokens": 110, "output_tokens": 55, "cache_read_input_tokens": 80},
            {"input_tokens": 120, "output_tokens": 60, "cache_read_input_tokens": 80},
        ],
        "2403.22222": [
            {"input_tokens": 200, "output_tokens": 100},
            {"input_tokens": 210, "output_tokens": 105},
            {"input_tokens": 220, "output_tokens": 110},
        ],
    })

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)
    manifest = json.loads(outcome.manifest_path.read_text())
    usage = manifest["token_usage"]

    assert usage["input_tokens"] == 100 + 110 + 120 + 200 + 210 + 220   # 960
    assert usage["output_tokens"] == 50 + 55 + 60 + 100 + 105 + 110     # 480
    assert usage["cache_read_input_tokens"] == 10 + 80 + 80             # 170
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["sample_count"] == 6
    assert usage["papers_with_usage"] == 2
    assert "estimated_cost" not in usage  # no cost knobs set


def test_token_usage_includes_estimated_cost_when_rates_set(tmp_path):
    cfg = _cfg(tmp_path,
               cost_per_input_mtok=15.0,    # $15 per million input tokens (Opus-ish)
               cost_per_output_mtok=75.0)
    records = [
        {"source": "rm", "statement": "x", "arxiv_id": "2403.11111",
         "provenance": "extracted", "uid": "uid_a"},
    ]
    adapter = _FakeAdapter({
        "2403.11111": [
            {"input_tokens": 1_000_000, "output_tokens": 500_000},  # round numbers
        ],
    })

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)
    manifest = json.loads(outcome.manifest_path.read_text())
    cost = manifest["token_usage"]["estimated_cost"]

    assert cost["input_usd"] == 15.0
    assert cost["output_usd"] == 37.5
    assert cost["total_usd"] == 52.5
    assert cost["is_estimate"] is True
    assert cost["rates_per_mtok"] == {"input_usd": 15.0, "output_usd": 75.0}


def test_token_usage_skips_cache_hits(tmp_path):
    """Locally-cached papers don't trigger API calls and shouldn't appear in usage."""
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(json.dumps({
        "arxiv_id": "2403.11111",
        "verdict_status": STATUS_PUBLISHED,
        "judge_votes": ["published"] * 3,
        "judge_model": "cached",
        "reasoning": "cache hit", "confidence": "high",
    }) + "\n")

    cfg = _cfg(tmp_path, cache_path=cache_path)
    records = [
        {"source": "rm", "statement": "x", "arxiv_id": "2403.11111",
         "provenance": "extracted", "uid": "uid_a"},
        {"source": "rm", "statement": "y", "arxiv_id": "2403.22222",
         "provenance": "extracted", "uid": "uid_b"},
    ]
    adapter = _FakeAdapter({
        "2403.22222": [{"input_tokens": 100, "output_tokens": 50}] * 3,
    })

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)
    usage = json.loads(outcome.manifest_path.read_text())["token_usage"]

    # Only the non-cached paper contributes to usage.
    assert usage["papers_with_usage"] == 1
    assert usage["input_tokens"] == 300
    assert usage["sample_count"] == 3


def test_token_usage_is_zeroed_when_adapter_emits_no_usage(tmp_path):
    """The replay adapter doesn't emit usage; rollup should be all zeros, not crash."""
    cfg = _cfg(tmp_path)
    records = [
        {"source": "rm", "statement": "x", "arxiv_id": "2403.11111",
         "provenance": "extracted", "uid": "uid_a"},
    ]
    adapter = _FakeAdapter({"2403.11111": []})  # no samples, no usage

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)
    usage = json.loads(outcome.manifest_path.read_text())["token_usage"]

    assert usage["input_tokens"] == 0
    assert usage["sample_count"] == 0
    assert usage["papers_with_usage"] == 0

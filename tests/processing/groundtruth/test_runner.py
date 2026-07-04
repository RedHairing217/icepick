"""Runner orchestration with a fake adapter — no Anthropic calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icepick.processing.groundtruth.base import (
    DISCARD_REASON_GENERATED,
    DISCARD_REASON_NO_ARXIV_ID,
    STATUS_DEFER,
    STATUS_DISCARDED,
    STATUS_PUBLISHED,
    STATUS_UNPUBLISHED,
    GroundtruthVerdict,
)
from icepick.processing.groundtruth.config import GroundtruthConfig
from icepick.processing.groundtruth.runner import run as run_groundtruth


class _FakeAdapter:
    """Returns a wired-in verdict per arxiv_id. Counts calls."""

    def __init__(self, verdicts_by_arxiv: dict):
        self._verdicts = verdicts_by_arxiv
        self.calls = []

    def lookup_paper(self, *, arxiv_id, paper_title, uid_for_error_attribution):
        self.calls.append(arxiv_id)
        wired = self._verdicts.get(arxiv_id, STATUS_DEFER)
        return GroundtruthVerdict(
            uid=uid_for_error_attribution,
            source="",
            verdict_status=wired,
            arxiv_id=arxiv_id,
            venue="Fake Venue" if wired == STATUS_PUBLISHED else None,
            publication_year=2024 if wired == STATUS_PUBLISHED else None,
            indexed_in=["FakeIndex"] if wired == STATUS_PUBLISHED else [],
            judge_model="fake-model",
            judge_votes=[wired] * 3,
            judge_majority=wired,
            reasoning="fake adapter verdict",
            confidence="high",
        )


def _cfg(tmp_path, **overrides):
    base = dict(
        mode="flow_testing",
        output_dir=tmp_path / "out",
        calibration_sheet=tmp_path / "sheet.jsonl",  # validation only; adapter is injected
    )
    base.update(overrides)
    return GroundtruthConfig(**base)


def test_published_record_makes_it_into_published_jsonl(tmp_path):
    cfg = _cfg(tmp_path)
    records = [
        {"source": "realmath", "statement": "Theorem 1.", "arxiv_id": "2403.12345",
         "provenance": "extracted", "uid": "uid_1"},
    ]
    adapter = _FakeAdapter({"2403.12345": STATUS_PUBLISHED})

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)

    pubs = [json.loads(l) for l in outcome.published_path.read_text().splitlines() if l.strip()]
    assert len(pubs) == 1
    assert pubs[0]["uid"] == "uid_1"
    # Published file preserves the original record shape, not the verdict shape.
    assert pubs[0]["statement"] == "Theorem 1."
    assert pubs[0]["arxiv_id"] == "2403.12345"


def test_unpublished_record_is_excluded_from_published_jsonl(tmp_path):
    cfg = _cfg(tmp_path)
    records = [
        {"source": "realmath", "statement": "X", "arxiv_id": "2403.99999",
         "provenance": "extracted", "uid": "uid_x"},
    ]
    adapter = _FakeAdapter({"2403.99999": STATUS_UNPUBLISHED})

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)

    pubs = outcome.published_path.read_text().strip()
    assert pubs == ""

    discarded = [json.loads(l) for l in outcome.discarded_path.read_text().splitlines() if l.strip()]
    statuses = {r["verdict_status"] for r in discarded}
    assert STATUS_UNPUBLISHED in statuses


def test_generated_records_discarded_with_reason(tmp_path):
    cfg = _cfg(tmp_path)
    records = [
        {"source": "calc_family", "statement": "compute x^2", "family": "calc",
         "provenance": "computed", "uid": "uid_gen"},
    ]
    adapter = _FakeAdapter({})

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)

    assert adapter.calls == []  # no lookup attempted
    discarded = [json.loads(l) for l in outcome.discarded_path.read_text().splitlines() if l.strip()]
    assert len(discarded) == 1
    assert discarded[0]["verdict_status"] == STATUS_DISCARDED
    assert discarded[0]["discarded_reason"] == DISCARD_REASON_GENERATED


def test_keep_generated_flag_keeps_generated_in_queue(tmp_path):
    """When discard_generated=False, generated records get treated like any other.

    They'll still fall out at the no-arxiv-id stage (because generated
    records don't have arxiv IDs), but the discard reason will be
    no_arxiv_id rather than generated_provenance.
    """
    cfg = _cfg(tmp_path, discard_generated=False)
    records = [
        {"source": "calc_family", "statement": "compute x^2", "family": "calc",
         "provenance": "computed", "uid": "uid_gen"},
    ]
    adapter = _FakeAdapter({})

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)
    discarded = [json.loads(l) for l in outcome.discarded_path.read_text().splitlines() if l.strip()]
    assert discarded[0]["discarded_reason"] == DISCARD_REASON_NO_ARXIV_ID


def test_records_without_arxiv_id_are_discarded(tmp_path):
    cfg = _cfg(tmp_path)
    records = [
        {"source": "manual_drop", "statement": "X", "provenance": "manual", "uid": "uid_m"},
    ]
    adapter = _FakeAdapter({})

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)

    assert adapter.calls == []
    discarded = [json.loads(l) for l in outcome.discarded_path.read_text().splitlines() if l.strip()]
    assert discarded[0]["discarded_reason"] == DISCARD_REASON_NO_ARXIV_ID


def test_one_paper_many_records_one_lookup(tmp_path):
    """Three records sharing an arxiv_id should produce ONE Anthropic call."""
    cfg = _cfg(tmp_path)
    records = [
        {"source": "realmath", "statement": f"Theorem {i}",
         "arxiv_id": "2403.12345", "provenance": "extracted", "uid": f"uid_{i}"}
        for i in range(3)
    ]
    adapter = _FakeAdapter({"2403.12345": STATUS_PUBLISHED})

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)

    assert adapter.calls == ["2403.12345"]  # exactly one lookup
    pubs = [json.loads(l) for l in outcome.published_path.read_text().splitlines() if l.strip()]
    assert len(pubs) == 3  # all three records flow through


def test_cache_hit_skips_anthropic_call(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(json.dumps({
        "arxiv_id": "2403.12345",
        "verdict_status": STATUS_PUBLISHED,
        "venue": "Cached Venue",
        "publication_year": 2023,
        "indexed_in": ["DBLP"],
        "judge_votes": ["published"] * 3,
        "judge_model": "cached-model",
        "reasoning": "cache hit",
        "confidence": "high",
    }) + "\n")

    cfg = _cfg(tmp_path, cache_path=cache_path)
    records = [
        {"source": "realmath", "statement": "X", "arxiv_id": "2403.12345",
         "provenance": "extracted", "uid": "uid_1"},
    ]
    adapter = _FakeAdapter({"2403.12345": STATUS_UNPUBLISHED})  # would say unpub if called

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)

    assert adapter.calls == []  # cache hit; no adapter call
    pubs = [json.loads(l) for l in outcome.published_path.read_text().splitlines() if l.strip()]
    assert len(pubs) == 1


def test_cache_is_persisted_after_lookup(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    cfg = _cfg(tmp_path, cache_path=cache_path)
    records = [
        {"source": "realmath", "statement": "X", "arxiv_id": "2403.12345",
         "provenance": "extracted", "uid": "uid_1"},
    ]
    adapter = _FakeAdapter({"2403.12345": STATUS_PUBLISHED})

    run_groundtruth(cfg=cfg, records=records, adapter=adapter)

    persisted = [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
    assert len(persisted) == 1
    assert persisted[0]["arxiv_id"] == "2403.12345"
    assert persisted[0]["verdict_status"] == STATUS_PUBLISHED


def test_manifest_records_config_and_counts(tmp_path):
    cfg = _cfg(tmp_path)
    records = [
        {"source": "realmath", "statement": "X", "arxiv_id": "2403.11111",
         "provenance": "extracted", "uid": "uid_a"},
        {"source": "realmath", "statement": "Y", "arxiv_id": "2403.22222",
         "provenance": "extracted", "uid": "uid_b"},
        {"source": "calc_family", "statement": "Z", "family": "calc",
         "provenance": "computed", "uid": "uid_c"},
    ]
    adapter = _FakeAdapter({
        "2403.11111": STATUS_PUBLISHED,
        "2403.22222": STATUS_UNPUBLISHED,
    })

    outcome = run_groundtruth(cfg=cfg, records=records, adapter=adapter)
    manifest = json.loads(outcome.manifest_path.read_text())

    assert manifest["stage"] == "groundtruth"
    assert manifest["inputs"]["record_count"] == 3
    assert manifest["inputs"]["unique_papers_looked_up"] == 2
    assert manifest["counts"][STATUS_PUBLISHED] == 1
    assert manifest["counts"][STATUS_UNPUBLISHED] == 1
    assert manifest["counts"][STATUS_DISCARDED] == 1


def test_validation_fires_inside_runner(tmp_path):
    cfg = GroundtruthConfig(mode="bogus", output_dir=tmp_path)
    with pytest.raises(Exception):  # ConfigError
        run_groundtruth(cfg=cfg, records=[])

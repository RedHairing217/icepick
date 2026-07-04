"""Groundtruth-stage runner.

Flow:

  1. Read records (JSONL — any shape; pre- or post-pass@k both work).
  2. Classify each record:
       - generated provenance (``computed``) → discard with reason
       - no arxiv_id resolvable → discard with reason
       - otherwise → queue for lookup
  3. Deduplicate by arxiv_id and look up each unique paper exactly once.
     Paper lookups run in parallel via ThreadPoolExecutor; the per-paper
     verdict is cached (and persisted to ``cache_path`` if configured).
  4. Map paper verdicts back onto records.
  5. Write four output files:
       - ``verdicts.jsonl``   - one row per input record (all statuses)
       - ``published.jsonl``  - the records that pass; downstream gate input
       - ``discarded.jsonl``  - dropped before lookup, with reason
       - ``run_manifest.json``- config echo + counts + paths

In flow_testing mode the adapter is swapped for ``CalibrationReplay``;
the rest of the runner is identical.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from icepick.processing.groundtruth.anthropic_adapter import AnthropicGroundtruthAdapter
from icepick.processing.groundtruth.base import (
    DISCARD_REASON_GENERATED,
    DISCARD_REASON_NO_ARXIV_ID,
    STATUS_DISCARDED,
    STATUS_PUBLISHED,
    GroundtruthVerdict,
    extract_arxiv_id,
)
from icepick.processing.groundtruth.calibration_replay import CalibrationReplay
from icepick.processing.groundtruth.config import GroundtruthConfig
from icepick.processing.poser.base import compute_uid, inject_uid


@dataclass
class RunOutcome:
    manifest_path: Path
    verdicts_path: Path
    published_path: Path
    discarded_path: Path
    counts: dict


def run(
    *,
    cfg: GroundtruthConfig,
    records: Iterable[dict],
    adapter=None,
) -> RunOutcome:
    """End-to-end groundtruth pass. Returns paths and counts.

    ``adapter`` injection lets tests substitute a fake. Leave None and
    the runner builds the appropriate adapter for ``cfg.mode``.
    """
    cfg.validate()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = inject_uid(list(records))
    paper_input = output_dir / "groundtruth_input.jsonl"
    with paper_input.open("w", encoding="utf-8") as fh:
        for record in prepared:
            fh.write(json.dumps(record) + "\n")

    adapter = adapter or _build_adapter(cfg)
    cache = _load_cache(cfg.cache_path)

    # Classify records up front so the discard set is known before any call.
    queued: list = []
    discarded: list = []
    for record in prepared:
        provenance = (record.get("provenance") or "").lower()
        if cfg.discard_generated and provenance == "computed":
            discarded.append(_discard_verdict(record, DISCARD_REASON_GENERATED))
            continue
        arxiv_id = extract_arxiv_id(record)
        if not arxiv_id:
            discarded.append(_discard_verdict(record, DISCARD_REASON_NO_ARXIV_ID))
            continue
        queued.append((record, arxiv_id))

    # One lookup per unique paper. Records sharing a paper reuse the verdict.
    unique_papers = _unique_papers(queued)
    paper_verdicts = _lookup_papers(
        unique_papers,
        adapter=adapter,
        cfg=cfg,
        cache=cache,
    )
    if cfg.cache_path:
        _persist_cache(cfg.cache_path, paper_verdicts)

    # Per-record verdicts: clone the paper verdict, overwrite uid/source.
    per_record: list = []
    for record, arxiv_id in queued:
        paper_verdict = paper_verdicts[arxiv_id]
        per_record.append(_clone_for_record(paper_verdict, record))

    all_verdicts = per_record + discarded

    verdicts_path = output_dir / "verdicts.jsonl"
    _write_verdicts(all_verdicts, verdicts_path)

    published_path = output_dir / "published.jsonl"
    _write_published(prepared, all_verdicts, published_path)

    discarded_path = output_dir / "discarded.jsonl"
    _write_verdicts(discarded + _non_published_lookups(per_record), discarded_path)

    counts = _count_statuses(all_verdicts)

    token_usage = _aggregate_token_usage(paper_verdicts, cfg=cfg)

    manifest = {
        "stage": "groundtruth",
        "config": cfg.echo(),
        "inputs": {
            "groundtruth_input": str(paper_input),
            "record_count": len(prepared),
            "unique_papers_looked_up": len(unique_papers),
            "cache_hits": _count_cache_hits(unique_papers, cache),
        },
        "outputs": {
            "verdicts": str(verdicts_path),
            "published": str(published_path),
            "discarded": str(discarded_path),
        },
        "counts": counts,
        "token_usage": token_usage,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return RunOutcome(
        manifest_path=manifest_path,
        verdicts_path=verdicts_path,
        published_path=published_path,
        discarded_path=discarded_path,
        counts=counts,
    )


def _aggregate_token_usage(paper_verdicts: dict, *, cfg: GroundtruthConfig) -> dict:
    """Sum usage across every sample of every paper looked up live.

    Cache hits don't show up here because they didn't trigger an API
    call. The adapter records ``usage`` per sample under
    ``raw_payload.samples[*].usage`` (Anthropic returns input_tokens,
    output_tokens, and cache fields).

    If the config carries cost-per-million-token knobs, we also estimate
    dollars. Estimates are clearly marked so dashboards can flag them
    as approximate rather than measured.
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "sample_count": 0,
        "papers_with_usage": 0,
    }
    for verdict in paper_verdicts.values():
        if verdict.raw_payload.get("cache_hit"):
            # Local-cache hit — no API call was made.
            continue
        samples = verdict.raw_payload.get("samples") or []
        had_usage = False
        for sample in samples:
            usage = (sample or {}).get("usage") or {}
            if not usage:
                continue
            had_usage = True
            totals["sample_count"] += 1
            for field in ("input_tokens", "output_tokens",
                          "cache_read_input_tokens", "cache_creation_input_tokens"):
                totals[field] += int(usage.get(field) or 0)
        if had_usage:
            totals["papers_with_usage"] += 1

    out: dict = dict(totals)
    if cfg.cost_per_input_mtok is not None or cfg.cost_per_output_mtok is not None:
        in_rate = cfg.cost_per_input_mtok or 0.0
        out_rate = cfg.cost_per_output_mtok or 0.0
        out["estimated_cost"] = {
            "input_usd": round(totals["input_tokens"] / 1_000_000 * in_rate, 6),
            "output_usd": round(totals["output_tokens"] / 1_000_000 * out_rate, 6),
            "total_usd": round(
                (totals["input_tokens"] / 1_000_000 * in_rate)
                + (totals["output_tokens"] / 1_000_000 * out_rate),
                6,
            ),
            "is_estimate": True,
            "rates_per_mtok": {"input_usd": in_rate, "output_usd": out_rate},
        }
    return out


def _build_adapter(cfg: GroundtruthConfig):
    if cfg.mode == "flow_testing":
        return _ReplayWrapper(CalibrationReplay(cfg.calibration_sheet), cfg.judge_model)
    return AnthropicGroundtruthAdapter(cfg)


class _ReplayWrapper:
    """Adapt CalibrationReplay to the same lookup_paper(...) surface."""

    def __init__(self, replay: CalibrationReplay, judge_model: str):
        self._replay = replay
        self._judge_model = judge_model

    def lookup_paper(self, *, arxiv_id, paper_title, uid_for_error_attribution):
        return self._replay.lookup(
            arxiv_id=arxiv_id,
            paper_title=paper_title,
            uid_for_error_attribution=uid_for_error_attribution,
            judge_model=self._judge_model,
        )


def _unique_papers(queued: list) -> dict:
    """Return ``{arxiv_id: (paper_title, representative_uid)}``."""
    out: dict = {}
    for record, arxiv_id in queued:
        if arxiv_id in out:
            continue
        out[arxiv_id] = (
            record.get("paper_title") or record.get("title"),
            record.get("uid") or compute_uid(record.get("source", ""), record.get("statement", "")),
        )
    return out


def _lookup_papers(unique_papers: dict, *, adapter, cfg: GroundtruthConfig, cache: dict) -> dict:
    """Dispatch lookups in parallel; honor the cache for hits."""
    results: dict = {}
    pending = []
    for arxiv_id, (title, attribution_uid) in unique_papers.items():
        if arxiv_id in cache:
            results[arxiv_id] = _verdict_from_cache_entry(cache[arxiv_id], cfg.judge_model)
        else:
            pending.append((arxiv_id, title, attribution_uid))

    if not pending:
        return results

    max_workers = min(cfg.max_concurrent, len(pending))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_arxiv = {
            ex.submit(
                adapter.lookup_paper,
                arxiv_id=arxiv_id,
                paper_title=title,
                uid_for_error_attribution=attribution_uid,
            ): arxiv_id
            for arxiv_id, title, attribution_uid in pending
        }
        for future in as_completed(future_to_arxiv):
            arxiv_id = future_to_arxiv[future]
            results[arxiv_id] = future.result()
    return results


def _clone_for_record(paper_verdict: GroundtruthVerdict, record: dict) -> GroundtruthVerdict:
    """Return a new verdict carrying this record's uid/source, paper-level rest."""
    return GroundtruthVerdict(
        uid=record["uid"],
        source=record.get("source", ""),
        verdict_status=paper_verdict.verdict_status,
        arxiv_id=paper_verdict.arxiv_id,
        venue=paper_verdict.venue,
        publication_year=paper_verdict.publication_year,
        indexed_in=list(paper_verdict.indexed_in),
        evidence_urls=list(paper_verdict.evidence_urls),
        judge_model=paper_verdict.judge_model,
        judge_votes=list(paper_verdict.judge_votes),
        judge_majority=paper_verdict.judge_majority,
        reasoning=paper_verdict.reasoning,
        confidence=paper_verdict.confidence,
        error_reason=paper_verdict.error_reason,
        raw_payload=dict(paper_verdict.raw_payload),
    )


def _discard_verdict(record: dict, reason: str) -> GroundtruthVerdict:
    return GroundtruthVerdict(
        uid=record.get("uid") or compute_uid(record.get("source", ""), record.get("statement", "")),
        source=record.get("source", ""),
        verdict_status=STATUS_DISCARDED,
        arxiv_id=None,
        discarded_reason=reason,
        reasoning=f"discarded pre-lookup: {reason}",
    )


def _write_verdicts(verdicts: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(v.to_jsonl_row()) + "\n")


def _write_published(prepared: list, verdicts: list, path: Path) -> None:
    """Write the original record dicts for any uid whose verdict is published.

    Downstream consumes this file directly — it's the gate input. We
    preserve the input's full record shape rather than the verdict
    shape, so pass@k stats / family / params survive intact.
    """
    published_uids = {
        v.uid for v in verdicts if v.verdict_status == STATUS_PUBLISHED
    }
    with path.open("w", encoding="utf-8") as fh:
        for record in prepared:
            if record["uid"] in published_uids:
                fh.write(json.dumps(record) + "\n")


def _non_published_lookups(verdicts: list) -> list:
    """Non-published verdicts that *did* go through lookup (vs pre-discarded)."""
    return [v for v in verdicts if v.verdict_status not in (STATUS_PUBLISHED, STATUS_DISCARDED)]


def _count_statuses(verdicts: list) -> dict:
    counts: dict = {}
    for v in verdicts:
        counts[v.verdict_status] = counts.get(v.verdict_status, 0) + 1
    return counts


def _load_cache(path) -> dict:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    out: dict = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = entry.get("arxiv_id")
            if aid:
                out[aid] = entry
    return out


def _persist_cache(path, paper_verdicts: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_cache(path)
    for arxiv_id, verdict in paper_verdicts.items():
        existing[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "verdict_status": verdict.verdict_status,
            "venue": verdict.venue,
            "publication_year": verdict.publication_year,
            "indexed_in": list(verdict.indexed_in),
            "evidence_urls": list(verdict.evidence_urls),
            "judge_model": verdict.judge_model,
            "judge_votes": list(verdict.judge_votes),
            "reasoning": verdict.reasoning,
            "confidence": verdict.confidence,
        }
    with path.open("w", encoding="utf-8") as fh:
        for entry in existing.values():
            fh.write(json.dumps(entry) + "\n")


def _count_cache_hits(unique_papers: dict, cache: dict) -> int:
    return sum(1 for aid in unique_papers if aid in cache)


def _verdict_from_cache_entry(entry: dict, judge_model: str) -> GroundtruthVerdict:
    return GroundtruthVerdict(
        uid="",
        source="",
        verdict_status=entry.get("verdict_status", "defer"),
        arxiv_id=entry.get("arxiv_id"),
        venue=entry.get("venue"),
        publication_year=entry.get("publication_year"),
        indexed_in=entry.get("indexed_in") or [],
        evidence_urls=entry.get("evidence_urls") or [],
        judge_model=entry.get("judge_model") or judge_model,
        judge_votes=entry.get("judge_votes") or [],
        judge_majority=entry.get("verdict_status"),
        reasoning=entry.get("reasoning", "cache hit"),
        confidence=entry.get("confidence"),
        raw_payload={"cache_hit": True},
    )

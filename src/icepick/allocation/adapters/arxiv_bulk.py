"""arXiv bulk-source adapter.

Mirrors :mod:`icepick.allocation.adapters.realmath_scrape` stage-for-stage,
but acquires LaTeX source in bulk from the arXiv S3 ``src/`` tarballs instead
of paging the arXiv Atom API one paper at a time. The three bulk primitives
live in :mod:`icepick.allocation.bulk`:

  - ``manifest`` — parse ``arXiv_src_manifest.xml``, select the chunks that
    cover a date window, roll up their bytes/egress cost.
  - ``category_index`` — an OAI-PMH ``ListRecords`` index that maps an
    ``arxiv_id`` to its ``(primary_category, categories, title)`` metadata and
    answers ``ids_for(category, yymm, primary_only)``.
  - ``chunk_store`` — download one ``src`` tarball at a time (``≤2`` resident,
    md5-verified), stream its members, extract the wanted ids' raw bytes.

``run`` in ``production`` drives that pipeline: parse the operator-provided
``manifest_path`` → select chunks by window → build/lookup the category index
→ pick ids by category+yymm → cap by ``max_papers`` → map ids to their
covering chunks → per chunk, download, extract matching members into an
in-memory dict, expose a LOCAL dict-backed ``source_fetcher`` over them, mine
each paper with realmath's ``latex_extractor`` / ``qa_extractor`` (under a
``ScrapeCheckpoint`` for paper-level resume + QA caching), then delete the
chunk before the next one. No network in the adapter beyond what the injected
provider client does; nothing here ever touches OAI/S3/arXiv/Anthropic
directly.

In ``flow_testing`` it replays the manifest's ``calibration_sheet`` fixture
exactly like realmath — no manifest, no chunks, no calls, deterministic
handoff, auto-approvable by its creator.

Both modes funnel their raw candidates through realmath's normalise +
run-layout writer (imported, never re-implemented), so handoff records satisfy
the same ``_CANONICAL_KEYS`` and the run tree is identical:

    <output_dir>/runs/<run_id>/
      manifest.json
      handoff/records.jsonl
      handoff/surplus_records.jsonl   <- cap overflow, never dropped
      raw/*.jsonl
      reports/source_report.md        <- renders bulk spend_rows

Budget: ``call_budget`` caps ``oai_requests + chunk_downloads + qa_calls``
(that sum is ``total_calls``; ``chunk_bytes`` is telemetry, not a call).
Exhausting it is a checkpointed PAUSE, never an exception to the operator:
``interrupted=True`` and re-running the same command resumes without redoing
paid work (checkpointed papers, cached QA, journaled chunks).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from icepick.allocation.manifests import require_approved
from icepick.contracts.manifests import (
    MODE_FLOW_TESTING,
    MODE_PRODUCTION,
    MODE_VALUES,
    SOURCE_ARXIV_BULK,
    ProposedPlan,
)
from icepick.contracts.records import TRUTH_POLICY_VALUES

# Reused machinery from the realmath adapter, public since the F2 promotion.
# Imported, never copied: the run-layout writer, normalise funnel, planning
# ratios, safety multiplier, result types, and the window/fixture helpers are
# identical for both sources.
from icepick.allocation.adapters.realmath_scrape import (
    CANDIDATES_PER_PAPER,
    ESTIMATE_SAFETY_MULTIPLIER,
    NormaliseResult,  # noqa: F401  (re-exported so callers get the same type)
    PAPERS_PER_RECORD,
    ScrapeRunResult,
    normalise,  # noqa: F401  (re-exported: bulk normalise IS realmath's)
    read_fixture_candidates,
    run_dir_for,
    validated_scrape_window,
    write_run,
)

# EGRESS cost basis mirrors bulk/manifest.py's EGRESS_USD_PER_GB (AWS
# us-east-1 -> internet). Kept as a local fallback constant so estimate() can
# price a window even if the sibling module's symbol name ever drifts; the
# authoritative number is manifest.EGRESS_USD_PER_GB and we prefer it when the
# module is importable.
_EGRESS_USD_PER_GB = 0.09

# Bulk acquisition spends no arXiv Atom queries: the category index is built
# over OAI-PMH (oai_requests) and LaTeX arrives in S3 chunks (chunk_downloads),
# not per-paper e-print fetches. estimate() therefore budgets one OAI page per
# _OAI_PAGE_SIZE_ESTIMATE papers instead of realmath's Atom pages.
_OAI_PAGE_SIZE_ESTIMATE = 1000  # OAI ListRecords carries ~1k records/page

_EXTRACTION_MODES = ("latex", "qa")

_PLAN_REQUIRED_FIELDS = {"source_name", "target_count", "requested_by", "requested_at"}
_PLAN_OPTIONAL_FIELDS = {"families", "scrape_window", "fixture_path", "notes"}
# realmath's window fields + manifest_path (LOCAL src-manifest path, required
# for production, absent for flow_testing) + cache_dir (optional OAI page cache
# location). exclude_arxiv_ids is inherited continuation support.
_SCRAPE_WINDOW_FIELDS = {
    "year", "month", "category", "max_papers", "max_per_paper", "primary_only",
    "extraction", "exclude_arxiv_ids", "manifest_path", "cache_dir",
}


def plan(request):
    """Build a ``ProposedPlan`` for an arXiv bulk acquisition.

    Pure: no parsing of the manifest, no chunk selection, no writes. Same
    required/optional request fields as realmath; ``scrape_window`` adds a
    LOCAL ``manifest_path`` (the operator-fetched ``arXiv_src_manifest.xml``,
    required before a production run, absent for flow_testing) and an optional
    ``cache_dir``. ``extraction`` is restricted to ``{"latex", "qa"}`` — bulk
    exists to mine LaTeX, so an abstract-only bulk pull is refused as a
    mistake.
    """
    if not isinstance(request, dict):
        raise ValueError(f"plan request must be a dict, got {type(request).__name__}")
    unknown = set(request) - _PLAN_REQUIRED_FIELDS - _PLAN_OPTIONAL_FIELDS
    if unknown:
        raise ValueError(f"unknown plan request fields: {sorted(unknown)}")
    missing = _PLAN_REQUIRED_FIELDS - set(request)
    if missing:
        raise ValueError(f"missing plan request fields: {sorted(missing)}")

    target_count = request["target_count"]
    if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count <= 0:
        raise ValueError(f"target_count must be a positive integer, got {target_count!r}")
    scrape_window = _validated_bulk_window(request.get("scrape_window"))

    notes = str(request.get("notes") or "")
    fixture_path = request.get("fixture_path")
    if fixture_path:
        fixture_note = f"flow_testing fixture: {fixture_path}"
        notes = f"{notes}; {fixture_note}" if notes else fixture_note

    return ProposedPlan(
        source_type=SOURCE_ARXIV_BULK,
        requested_by=str(request["requested_by"]),
        requested_at=str(request["requested_at"]),
        source_name=str(request["source_name"]),
        target_count=target_count,
        notes=notes,
        families=list(request.get("families") or []),
        scrape_window=scrape_window,
        estimated_calls=_estimated_calls(
            target_count, _bulk_extraction_of(scrape_window), scrape_window
        ),
        estimated_cost_usd=None,  # egress priced in estimate() from the manifest rollup
    )


def estimate(plan):
    """Describe the expected work for a bulk plan before approval.

    Never fetches: OAI/chunk/QA call counts come from ``target_count`` through
    the conservative planning ratios; ``expected_chunk_bytes`` /
    ``expected_egress_usd`` come from rolling up the chunks the window selects,
    parsed LOCALLY from ``scrape_window['manifest_path']``. Every number rounds
    AGAINST the operator (safety multiplier on calls; the rollup is the real
    byte total, not an underestimate).
    """
    if plan.source_type != SOURCE_ARXIV_BULK:
        raise ValueError(
            f"estimate expects source_type '{SOURCE_ARXIV_BULK}', got {plan.source_type!r}"
        )
    if plan.target_count <= 0:
        raise ValueError(f"target_count must be positive, got {plan.target_count!r}")

    extraction = _bulk_extraction_of(plan.scrape_window)
    if extraction not in _EXTRACTION_MODES:
        raise ValueError(
            f"extraction must be one of {sorted(_EXTRACTION_MODES)} for arxiv_bulk, "
            f"got {extraction!r}"
        )
    expected_papers = plan.target_count * PAPERS_PER_RECORD
    expected_candidates = expected_papers * CANDIDATES_PER_PAPER

    # Which acquisition calls this mode spends. Both bulk modes page the OAI
    # index and download chunks; qa adds one LLM call per mined theorem.
    call_kinds = ["oai_requests", "chunk_downloads"]
    expected_llm_calls = 0
    prerequisites = [
        "AWS credentials (requester-pays S3 src/ bucket)",
        "boto3 ([bulk] extra)",
        "network access to oaipmh.arxiv.org + s3://arxiv (production only)",
    ]
    if extraction == "qa":
        call_kinds.append("qa_calls")
        expected_llm_calls = expected_candidates  # one Sonnet Q+A call per mined theorem
        prerequisites += [
            "anthropic SDK ([judge] extra)",
            "Anthropic key via ANTHROPIC_KEY_FILE or ANTHROPIC_API_KEY",
        ]

    rollup = _window_rollup(plan.scrape_window)

    return {
        "source_type": SOURCE_ARXIV_BULK,
        "source_name": plan.source_name,
        "target_count": plan.target_count,
        "extraction": extraction,
        "expected_papers": expected_papers,
        "expected_candidates": expected_candidates,
        "expected_handoff_records": plan.target_count,
        "estimated_calls": _estimated_calls(plan.target_count, extraction),
        "call_kinds": call_kinds,
        "expected_llm_calls": expected_llm_calls,
        "expected_chunk_bytes": rollup["total_bytes"],
        "expected_egress_usd": rollup["egress_usd"],
        "token_budget": None,
        "local_prerequisites": prerequisites,
    }


def run(manifest, *, now: Optional[datetime] = None):
    """Execute an approved bulk acquisition run.

    Every gate — source type, mode, approval, call budget, output
    containment — is validated before any work. ``flow_testing`` replays the
    manifest's ``calibration_sheet`` fixture (no chunks, no calls);
    ``production`` drives the S3 bulk pipeline from ``scrape_window`` (parse the
    src manifest, select chunks, index categories, extract + mine per chunk).
    Both funnel raw candidates through the shared normalise + run-layout writer.
    """
    _validate_manifest(manifest)
    run_dir = run_dir_for(manifest)
    if manifest.processor_mode == MODE_FLOW_TESTING:
        candidates = read_fixture_candidates(manifest, run_dir)
        return write_run(
            manifest,
            run_dir,
            candidates,
            calibration_replay=True,
            report_title="arXiv bulk source report",
            now=now,
        )

    result, checkpoint = _bulk_acquire(manifest, run_dir)
    total_calls = result["oai_requests"] + result["chunk_downloads"] + result["qa_calls"]
    chunk_bytes = result["chunk_bytes"]
    corrupt_downloads = result["corrupt_downloads"]
    # spend_rows (rendered verbatim by _write_report; chunk_gb = decimal GB, 1e9).
    # Corrupt transfers billed egress but yielded no chunk — invariant 2 makes
    # them visible ONLY when they happened, so a clean run's report stays clean.
    spend_rows = [
        ["oai_request", result["oai_requests"]],
        ["chunk_download", result["chunk_downloads"]],
    ]
    if corrupt_downloads > 0:
        spend_rows.append(["chunk_download_corrupt", corrupt_downloads])
    spend_rows += [
        ["chunk_gb", round(chunk_bytes / 1e9, 3)],
        [f"qa_generation ({result['qa_model'] or 'model unrecorded'})", result["qa_calls"]],
    ]
    acquisition = {
        "oai_requests": result["oai_requests"],
        "chunk_downloads": result["chunk_downloads"],
        "chunk_bytes": chunk_bytes,
        # Checksum-failed transfers that still billed egress (§3 amendment).
        # Keys always present; the spend_row above only when count > 0.
        "corrupt_downloads": corrupt_downloads,
        "corrupt_bytes": result["corrupt_bytes"],
        "qa_calls": result["qa_calls"],
        "qa_model": result["qa_model"],
        "rate_limit_events": result["rate_limit_events"],
        "rate_limit_backoff_seconds": result["rate_limit_backoff_seconds"],
        "rate_limit_statuses": result["rate_limit_statuses"],
        "token_usage": result["token_usage"],
        "total_calls": total_calls,  # chunk_bytes deliberately excluded (telemetry)
        "call_budget": manifest.call_budget,
        "resumed_papers": result["resumed_papers"],
        "spend_rows": spend_rows,
    }
    outcome = write_run(
        manifest,
        run_dir,
        result["candidates"],
        calibration_replay=False,
        extra_warnings=list(result["warnings"]),
        acquisition=acquisition,
        interrupted=result["interrupted"],
        progress_dir=checkpoint.progress_dir,
        surplus=result["surplus"],
        report_title="arXiv bulk source report",
        now=now,
    )
    if not result["interrupted"]:
        checkpoint.mark_complete()
    return outcome


# --- manifest gates -----------------------------------------------------------


def _validate_manifest(manifest) -> None:
    if manifest.source_type != SOURCE_ARXIV_BULK:
        raise ValueError(
            f"manifest source_type must be '{SOURCE_ARXIV_BULK}', got {manifest.source_type!r}"
        )
    if manifest.processor_mode not in MODE_VALUES:
        raise ValueError(
            f"processor_mode must be one of {MODE_VALUES}, got {manifest.processor_mode!r}"
        )
    require_approved(manifest)
    if not manifest.run_id:
        raise ValueError("manifest run_id must be set")
    budget = manifest.call_budget
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError(f"call_budget must be a non-negative integer, got {budget!r}")
    if not isinstance(manifest.target_count, int) or manifest.target_count <= 0:
        raise ValueError(
            f"target_count must be a positive integer, got {manifest.target_count!r}"
        )
    if manifest.processor_mode == MODE_PRODUCTION:
        window = _validated_bulk_window(manifest.scrape_window)
        extraction = _bulk_extraction_of(window)
        if extraction not in _EXTRACTION_MODES:
            raise ValueError(
                f"extraction must be one of {sorted(_EXTRACTION_MODES)} for arxiv_bulk, "
                f"got {extraction!r}"
            )
        manifest_path = (window or {}).get("manifest_path")
        if not manifest_path:
            raise ValueError(
                "production arxiv_bulk manifest requires scrape_window['manifest_path'] "
                "(the operator-fetched arXiv_src_manifest.xml local path)"
            )
        if not Path(manifest_path).is_file():
            raise ValueError(f"manifest_path does not exist: {manifest_path}")
        estimated = _estimated_calls(manifest.target_count, extraction)
        if estimated > budget:
            raise ValueError(
                f"call_budget {budget} is below the {estimated} calls estimated for "
                f"target_count {manifest.target_count} ({extraction}); refusing rather "
                "than over-running"
            )
    if not manifest.output_dir:
        raise ValueError("manifest output_dir must be set")
    if manifest.truth_policy and manifest.truth_policy not in TRUTH_POLICY_VALUES:
        raise ValueError(
            f"unknown truth_policy {manifest.truth_policy!r}; allowed: {TRUTH_POLICY_VALUES}"
        )
    if manifest.processor_mode == MODE_FLOW_TESTING and not manifest.calibration_sheet:
        raise ValueError("flow_testing requires calibration_sheet (the local fixture path)")


def _validated_bulk_window(window) -> Optional[dict]:
    """Like realmath's ``validated_scrape_window`` but for bulk fields.

    Reuses realmath's exclude_arxiv_ids validation, then widens the allowed
    key set to the bulk window (adds manifest_path + cache_dir).
    """
    if window is None:
        return None
    if not isinstance(window, dict):
        raise ValueError(f"scrape_window must be a dict, got {type(window).__name__}")
    unknown = set(window) - _SCRAPE_WINDOW_FIELDS
    if unknown:
        raise ValueError(f"unknown scrape_window fields: {sorted(unknown)}")
    # Delegate the exclude_arxiv_ids shape check to realmath (identical rule);
    # feed it only the fields it knows so its own unknown-field guard passes.
    validated_scrape_window(
        {k: v for k, v in window.items() if k not in {"manifest_path", "cache_dir"}}
    )
    for key in ("manifest_path", "cache_dir"):
        value = window.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a non-empty string path")
    return dict(window)


def _bulk_extraction_of(scrape_window) -> str:
    """Extraction mode for bulk; defaults to ``latex`` (never ``abstract``)."""
    mode = (scrape_window or {}).get("extraction")
    return mode or "latex"


# --- estimate helpers ---------------------------------------------------------


def _estimated_calls(target_count: int, extraction: str = "latex", scrape_window=None) -> int:
    """Estimate acquisition calls for a bulk pull, padded against the operator.

    ``latex`` spends OAI index pages + one chunk download per covering chunk;
    ``qa`` adds one Sonnet Q+A call per mined theorem. Chunk downloads are hard
    to know without the manifest, so the estimate budgets one download per paper
    as an upper bound (a paper is never spread across chunks); the manifest
    rollup in ``estimate()`` reports the true byte cost. The summed expectation
    is padded by ``ESTIMATE_SAFETY_MULTIPLIER`` so routine variance doesn't
    pause a run at its ceiling.

    The OAI-page term is CORPUS-shaped, not sample-shaped: the index walks the
    whole set from the window start, independent of how many papers we keep. So
    when a ``manifest_path`` is present (production), it is priced from the
    manifest — ``ceil(sum(num_items over entries with yymm >= window start) /
    1000)`` — an all-categories superset of the actual set walk, a deliberate
    over-provision of an unpriced unit (never under). Only the pre-manifest
    plan (no manifest_path) falls back to the target_count-scaled page guess.
    """
    expected_papers = target_count * PAPERS_PER_RECORD
    oai_pages = _oai_pages_estimate(scrape_window, expected_papers)
    calls = oai_pages + expected_papers  # index pages + <=1 chunk download/paper
    if extraction == "qa":
        calls += expected_papers * CANDIDATES_PER_PAPER  # one Sonnet Q+A call per theorem
    return math.ceil(calls * ESTIMATE_SAFETY_MULTIPLIER)


def _oai_pages_estimate(scrape_window, expected_papers: int) -> int:
    """OAI ListRecords page count for the index walk.

    Production (``manifest_path`` present): corpus-shaped — sum ``num_items``
    over manifest entries whose ``yymm`` is at or after the window start, then
    ``ceil(records / 1000)``. This over-counts (all categories, not just the
    requested set) on purpose: the walk is corpus-sized and an unpriced unit
    must round against the operator. Pre-manifest fallback: the old
    target_count-scaled guess (walk size is unknowable without the manifest).
    """
    window = scrape_window or {}
    manifest_path = window.get("manifest_path")
    if not manifest_path or not Path(manifest_path).exists():
        # No manifest on disk yet (plan-time may precede the operator's W4
        # fetch): fall back to the sample-scaled guess instead of failing the
        # plan. estimate() stays strict — its rollup raises on a missing file.
        return max(1, -(-expected_papers // _OAI_PAGE_SIZE_ESTIMATE))
    from icepick.allocation.bulk import manifest as manifest_mod

    entries = manifest_mod.parse_manifest(Path(manifest_path).read_text(encoding="utf-8"))
    start_yymm = _window_start_yymm(window)
    total_records = sum(
        int(getattr(entry, "num_items", 0) or 0)
        for entry in entries
        if start_yymm is None or entry.yymm >= start_yymm
    )
    return max(1, -(-total_records // _OAI_PAGE_SIZE_ESTIMATE))


def _window_start_yymm(window) -> Optional[str]:
    """The ``yymm`` string at the window's lower bound, or ``None`` for no year.

    Mirrors the id yymm form ("2501"): last two digits of the year + zero-padded
    month (month defaults to 01). ``None`` (no year) means the whole manifest.
    """
    year = window.get("year")
    if year is None:
        return None
    month = window.get("month") or 1
    return f"{int(year) % 100:02d}{int(month):02d}"


def _window_rollup(scrape_window) -> dict:
    """Roll up chunk bytes + egress cost for the window from its src manifest.

    Parses ``scrape_window['manifest_path']`` LOCALLY via bulk.manifest and
    selects the window's chunks; returns zeros when no manifest_path is set
    (e.g. an early plan before the operator fetched the manifest). Never
    fetches. Egress is rounded UP at 4 dp so it never understates (W3 L2).
    """
    window = scrape_window or {}
    manifest_path = window.get("manifest_path")
    if not manifest_path:
        return {"total_bytes": 0, "egress_usd": 0.0, "chunk_count": 0}
    from icepick.allocation.bulk import manifest as manifest_mod

    entries = manifest_mod.parse_manifest(Path(manifest_path).read_text(encoding="utf-8"))
    selected = _select_window_chunks(manifest_mod, entries, window)
    rollup = manifest_mod.rollup(selected)
    return {
        "total_bytes": rollup.total_bytes,
        "egress_usd": math.ceil(rollup.egress_usd * 10000) / 10000,
        "chunk_count": rollup.chunk_count,
    }


def _select_window_chunks(manifest_mod, entries, window) -> list:
    """Chunks covering the window: ``select_chunks`` by year(+month), else all."""
    year = window.get("year")
    if year is None:
        return list(entries)
    return manifest_mod.select_chunks(entries, year=int(year), month=window.get("month"))


# --- production acquisition (S3 bulk pipeline) --------------------------------


def _bulk_acquire(manifest, run_dir: Path) -> tuple:
    """Drive the bulk pipeline and return (result_dict, checkpoint).

    Orchestration lives here (not in a bulk module) because it stitches the
    three primitives together the way realmath's ``scrape`` stitches its
    fetchers/extractors. The budget is enforced HERE, before every paid call,
    so a theorem-dense paper never spends past the approved cap between checks;
    exhaustion raises ``_BudgetExhausted`` and is caught into a checkpointed,
    resumable partial result — identical semantics to realmath.

    Test seams (all monkeypatched offline in the production tests):
      - ``_build_category_index`` → a CategoryIndex-shaped object
      - ``_open_chunk_store``     → a ChunkStore-shaped object
      - the QA generator passed through ``scrape_window`` / a module default
    """
    from icepick.allocation.scrape import realmath as realmath_source
    from icepick.allocation.scrape.checkpoint import ScrapeCheckpoint

    window = _validated_bulk_window(manifest.scrape_window) or {}
    extraction = _bulk_extraction_of(window)
    category = window.get("category") or "math"
    primary_only = bool(window.get("primary_only"))
    max_papers = window.get("max_papers")
    max_per_paper = window.get("max_per_paper")
    target_count = manifest.target_count
    families = list(manifest.families or [])
    family = families[0] if len(families) == 1 else None

    progress_dir = run_dir / "_progress"
    checkpoint = ScrapeCheckpoint(progress_dir)
    journal = _ChunkJournal(progress_dir)

    counts = {"oai_requests": 0, "chunk_downloads": 0, "qa_calls": 0, "chunk_bytes": 0}
    token_usage: dict = {}
    qa_model_used = {"name": None}
    warnings: list = []

    def acquisition_calls():
        return counts["oai_requests"] + counts["chunk_downloads"] + counts["qa_calls"]

    def charge(kind):
        if manifest.call_budget is not None and acquisition_calls() >= manifest.call_budget:
            raise _BudgetExhausted
        counts[kind] += 1

    def record_token_usage(usage):
        if not usage:
            return
        for key, value in usage.items():
            token_usage[f"qa_{key}"] = token_usage.get(f"qa_{key}", 0) + int(value or 0)

    # --- 1. src manifest -> window chunks --------------------------------------
    from icepick.allocation.bulk import manifest as manifest_mod

    entries = manifest_mod.parse_manifest(
        Path(window["manifest_path"]).read_text(encoding="utf-8")
    )
    window_chunks = _select_window_chunks(manifest_mod, entries, window)

    # Stages 2-4 all run under ONE pause handler below: everything past this
    # point can charge the budget, and exhaustion anywhere in the paid pipeline
    # must land as a checkpointed PAUSE (W3 H1), never an escaping exception.
    # The manifest parse above stays outside — a bad manifest_path is an
    # operator error, not a pause.
    def counting_qa(statement, **kwargs):
        charge("qa_calls")
        result = _default_qa_generator(statement)(
            statement,
            usage_callback=record_token_usage,
            model_callback=lambda name: qa_model_used.__setitem__("name", name),
        )
        return result

    checkpoint.enforce_rate_limit_cooldown()
    checkpoint.begin()
    counting_qa = checkpoint.caching_generator(counting_qa)

    candidates: list = []
    surplus: list = []
    resumed_papers = 0
    interrupted = False
    seen_ids: set = set(window.get("exclude_arxiv_ids") or [])

    index = None
    needed_chunks: list = []
    store = None
    current_chunk_path = None  # in-flight chunk, retained on pause (W3 M2)
    try:
        # --- 2. category index (OAI) -> ids in scope ---------------------------
        index = _build_category_index(manifest, run_dir, window, counts, charge)
        wanted_ids = _ids_in_scope(index, window_chunks, category, primary_only, max_papers)

        # --- 3. chunks that actually cover the wanted ids ----------------------
        needed_chunks = manifest_mod.chunks_for_ids(window_chunks, set(wanted_ids))

        # --- 4. per chunk: download, extract, mine ------------------------------
        store = _open_chunk_store(manifest, run_dir, window)
        for entry in needed_chunks:
            if len(candidates) >= target_count:
                break
            if journal.done(entry.filename):
                # Journaled chunk: its papers are already committed to the
                # checkpoint. Re-serve them from the store, never re-download.
                member_bytes = None
            else:
                # Budget gate BEFORE the transfer (a pause must precede the
                # bill), then count only what the store actually downloaded —
                # an adopted still-resident chunk on resume is free (W3 M3).
                if manifest.call_budget is not None and acquisition_calls() >= manifest.call_budget:
                    raise _BudgetExhausted
                downloads_before = getattr(store, "chunk_downloads", 0)
                try:
                    chunk_path = store.fetch(entry)
                except _checksum_error_types():
                    # Corrupt transfer (§3 amendment): the store billed egress,
                    # deleted the bad file, and bumped its corrupt_* counters
                    # (surfaced at the end). Skipping one bad chunk must not
                    # abort the run — resilience the run relies on. Nothing was
                    # charged for it: chunk_downloads counts VERIFIED downloads
                    # only (corrupt_downloads carries the failed egress).
                    warnings.append(
                        f"chunk {entry.filename} failed checksum verification "
                        "(egress billed, chunk discarded); skipping its papers"
                    )
                    continue
                current_chunk_path = chunk_path
                if getattr(store, "chunk_downloads", 0) > downloads_before:
                    counts["chunk_downloads"] += 1
                    counts["chunk_bytes"] += int(getattr(entry, "size_bytes", 0) or 0)
                member_bytes = dict(store.extract_matching(chunk_path, set(wanted_ids)))

            chunk_paper_count = 0
            chunk_full = False
            # ``wanted_ids`` was already capped by max_papers in _ids_in_scope,
            # so no per-chunk cap re-check is needed here — that cap is the
            # single source of truth for how many papers this run may mine.
            for arxiv_id in _chunk_ids(entry, wanted_ids):
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                stored = checkpoint.stored_candidates(arxiv_id)
                if stored is not None:
                    resumed_papers += 1
                    extracted = stored
                else:
                    if member_bytes is None or arxiv_id not in member_bytes:
                        # Missing from the chunk: skip the paper + warn. NEVER
                        # fall through to a network fetch.
                        warnings.append(
                            f"arxiv_id {arxiv_id} not found in chunk {entry.filename}; "
                            "skipping paper (no network fallback)"
                        )
                        continue
                    paper = _paper_for(index, arxiv_id)
                    source_fetcher = _local_source_fetcher(member_bytes)
                    if extraction == "qa":
                        extracted = realmath_source.qa_extractor(
                            paper, family=family,
                            source_fetcher=source_fetcher, generator=counting_qa,
                        )
                    else:
                        extracted = realmath_source.latex_extractor(
                            paper, family=family, source_fetcher=source_fetcher,
                        )
                    checkpoint.commit(arxiv_id, extracted)

                chunk_paper_count += 1
                kept_this_paper = 0
                for candidate in extracted:
                    if (
                        len(candidates) >= target_count
                        or (max_per_paper and kept_this_paper >= max_per_paper)
                    ):
                        surplus.append(candidate)
                        continue
                    candidates.append(candidate)
                    kept_this_paper += 1
                if len(candidates) >= target_count:
                    chunk_full = True
                    break

            # Chunk exhausted (or target hit): journal it and free the disk.
            journal.mark(entry.filename, chunk_paper_count)
            store.release(entry)
            current_chunk_path = None
            if chunk_full:
                break
    except KeyboardInterrupt:
        interrupted = True
        warnings.append(
            "interrupted (Ctrl-C); progress is checkpointed — rerun the same "
            "'allocation run --manifest' command to resume where it stopped"
        )
    except _BudgetExhausted:
        interrupted = True
        warnings.append(
            f"call budget {manifest.call_budget} exhausted after {acquisition_calls()} "
            "paid calls; checkpointed — rerun the same command to continue (already-"
            "cached work costs nothing against the fresh budget)"
        )

    if interrupted and current_chunk_path is not None:
        # Deliberate retention (W3 M2): resume adopts the verified file free;
        # deleting here would re-bill its egress on the rerun (invariant 3
        # outranks tidiness). Completed runs still end with _chunks/ empty.
        warnings.append(
            f"in-flight chunk retained for resume at {current_chunk_path}; "
            "rerunning the same command adopts it without re-billing"
        )

    if not candidates:
        warnings.append(
            f"arXiv bulk pull for source {manifest.source_name!r} produced no candidates "
            f"(category {category!r}, chunks considered: {len(needed_chunks)})"
        )
    if target_count and len(candidates) > target_count:
        surplus[:0] = candidates[target_count:]
        candidates = candidates[:target_count]

    # Throttle telemetry is the union of two sources: the checkpoint's durable
    # run-lifetime log (survives across invocations) and the CategoryIndex's own
    # lifetime counters (§2 amendment: it journals OAI 429/503s inside build(),
    # which the checkpoint never sees in the bulk path). Sum them so an OAI
    # throttle during index-building shows up in the report alongside any
    # checkpoint-recorded events.
    lifetime = checkpoint.rate_limit_telemetry()
    # index is None when the budget paused the run during the index build
    # itself (W3 H1) — there is no OAI telemetry to merge in that case.
    idx_tel = (
        _index_rate_limit_telemetry(index)
        if index is not None
        else {"events": 0, "backoff_seconds": 0.0, "statuses": {}}
    )
    rate_limit_statuses = dict(lifetime["statuses"])
    for status, count in idx_tel["statuses"].items():
        rate_limit_statuses[status] = rate_limit_statuses.get(status, 0) + count
    result = {
        "candidates": candidates,
        "surplus": surplus,
        "warnings": warnings,
        "oai_requests": counts["oai_requests"],
        "chunk_downloads": counts["chunk_downloads"],
        "chunk_bytes": counts["chunk_bytes"],
        # §3 amendment: checksum-failed transfers that still billed egress.
        # Read off the store's lifetime counters (a fake may omit them → 0).
        "corrupt_downloads": int(getattr(store, "corrupt_downloads", 0) or 0),
        "corrupt_bytes": int(getattr(store, "corrupt_bytes", 0) or 0),
        "qa_calls": counts["qa_calls"],
        "qa_model": qa_model_used["name"],
        "token_usage": token_usage,
        "rate_limit_events": lifetime["events"] + idx_tel["events"],
        "rate_limit_backoff_seconds": lifetime["backoff_seconds"] + idx_tel["backoff_seconds"],
        "rate_limit_statuses": rate_limit_statuses,
        "resumed_papers": resumed_papers,
        "interrupted": interrupted,
    }
    return result, checkpoint


def _checksum_error_types() -> tuple:
    """Exception types a corrupt chunk fetch raises.

    ``ChunkStore.fetch`` raises ``ChecksumError`` (a ``RuntimeError`` subclass,
    §3) on md5 mismatch. Resolve the specific class when chunk_store is on disk
    so the catch is tight; fall back to ``RuntimeError`` (its base) otherwise —
    either way a residency-overflow ``RuntimeError`` is caught deliberately too
    (both are "this chunk can't be used", skip and continue).
    """
    try:
        from icepick.allocation.bulk.chunk_store import ChecksumError

        return (ChecksumError, RuntimeError)
    except Exception:
        return (RuntimeError,)


def _index_rate_limit_telemetry(index) -> dict:
    """Lifetime OAI throttle telemetry off a CategoryIndex, defaulting to zero.

    §2 (amended) makes ``rate_limit_events`` / ``rate_limit_backoff_seconds`` /
    ``rate_limit_statuses`` public realmath-shaped attrs on the index; a fake
    index in tests may omit them, so each read tolerates absence.
    """
    return {
        "events": int(getattr(index, "rate_limit_events", 0) or 0),
        "backoff_seconds": float(getattr(index, "rate_limit_backoff_seconds", 0.0) or 0.0),
        "statuses": dict(getattr(index, "rate_limit_statuses", {}) or {}),
    }


def _ids_in_scope(index, window_chunks, category, primary_only, max_papers) -> list:
    """Ids the index reports for the window's yymm months, capped by max_papers.

    Order-preserving and de-duplicated across chunks. ``max_papers`` caps the
    id pool up front so a huge category never over-selects before extraction.
    """
    yymms = []
    for entry in window_chunks:
        if entry.yymm not in yymms:
            yymms.append(entry.yymm)
    ordered: list = []
    seen: set = set()
    for yymm in yymms:
        for arxiv_id in index.ids_for(category=category, yymm=yymm, primary_only=primary_only):
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            ordered.append(arxiv_id)
            if max_papers is not None and len(ordered) >= max_papers:
                return ordered
    return ordered


def _chunk_ids(entry, wanted_ids) -> list:
    """Wanted ids that fall in this chunk's inclusive ``[first_item, last_item]``.

    Uses ``manifest.id_in_range`` — the ONE shared parser — so a chunk whose
    range straddles a month boundary (first ``2504.x``, last ``2505.y``)
    correctly claims its ``2505`` ids instead of dropping them after their
    egress is already billed. A non-new-style id (out of v1 scope) is skipped
    rather than crashing the range check.
    """
    from icepick.allocation.bulk.manifest import id_in_range

    ids: list = []
    for arxiv_id in wanted_ids:
        try:
            in_range = id_in_range(arxiv_id, entry.first_item, entry.last_item)
        except (ValueError, TypeError):
            continue  # not a new-style id — outside v1 scope, leave it
        if in_range:
            ids.append(arxiv_id)
    return ids


def _paper_for(index, arxiv_id: str):
    """Build a realmath ``Paper`` from index metadata.

    ``abstract`` and ``published`` are deliberately empty: the category index
    carries no abstract, and no latex/qa extractor path reads either field
    (verified against realmath.py — only the abstract-mode ``default_extractor``
    reads ``paper.abstract``, and bulk never runs abstract mode). ``link`` is
    the canonical abs URL so downstream arxiv-id recovery works.
    """
    from icepick.allocation.scrape.realmath import Paper

    meta = index.lookup(arxiv_id)
    if meta is None:
        return Paper(
            arxiv_id=arxiv_id,
            link=f"https://arxiv.org/abs/{arxiv_id}",
            title="",
            abstract="",
            primary_category="",
            categories=[],
            published="",
        )
    return Paper(
        arxiv_id=arxiv_id,
        link=f"https://arxiv.org/abs/{arxiv_id}",
        title=meta.title,
        abstract="",
        primary_category=meta.primary_category,
        categories=list(meta.categories),
        published="",
    )


def _local_source_fetcher(member_bytes: dict):
    """A dict-backed ``(arxiv_id, **kw) -> bytes`` over extracted chunk members.

    The seam realmath's ``latex_extractor`` calls. Missing id raises so the
    extractor's per-paper ``except Exception`` guard skips it — but the caller
    already screened membership and warned, so this is belt-and-braces. NEVER
    falls through to the network.
    """

    def fetch(arxiv_id, **kwargs):
        try:
            return member_bytes[arxiv_id]
        except KeyError as exc:
            raise KeyError(
                f"arxiv_id {arxiv_id} not in the local chunk member set "
                "(bulk fetcher never hits the network)"
            ) from exc

    return fetch


def _default_qa_generator(_statement):
    """Return the QA generator to call (indirection kept for a test seam).

    Production wires realmath's ``default_qa_generator``; tests monkeypatch
    this module attribute so no Anthropic call is ever made. Takes the
    statement only to keep the call site uniform; the returned callable is the
    generator itself.
    """
    from icepick.allocation.scrape.realmath import default_qa_generator

    return default_qa_generator


# --- injectable bulk-primitive builders (test seams) --------------------------


def _build_category_index(manifest, run_dir: Path, window: dict, counts: dict, charge):
    """Build/lookup the OAI category index, charging oai_requests per request.

    Wraps the real fetcher so every issued OAI request both counts and is
    budget-checked. Production tests monkeypatch this whole function to return
    a fake CategoryIndex-shaped object, so the OAI network is never touched.
    """
    from icepick.allocation.bulk.category_index import CategoryIndex

    cache_dir = Path(window.get("cache_dir") or (run_dir / "_progress" / "oai_cache"))
    index = CategoryIndex(cache_dir)

    def counting_fetcher(url):
        charge("oai_requests")
        return _default_oai_fetcher()(url)

    # Bound the walk to the window era (W3 H2): OAI `from` is a SUPERSET
    # filter (datestamp >= submission date), so no in-window paper is missed;
    # ids_for's yymm prefix match stays the precise selector.
    year = window.get("year")
    from_date = (
        f"{int(year):04d}-{int(window.get('month') or 1):02d}-01"
        if year is not None
        else None
    )
    index.build(
        oai_set=window.get("category", "math").split(".")[0],
        fetcher=counting_fetcher,
        from_date=from_date,
    )
    # counts already advanced inside counting_fetcher via charge(); mirror the
    # index's own tally onto counts as the source of truth if it exposes one.
    if getattr(index, "oai_requests", None) is not None:
        counts["oai_requests"] = max(counts["oai_requests"], index.oai_requests)
    return index


def _default_oai_fetcher():
    """The real OAI HTTP fetcher (never called in tests; seam for monkeypatch).

    Production must wire a fetcher that returns a ``category_index.OAIResponse``
    (status / retry_after / text); the offline tests replace
    ``_build_category_index`` wholesale, so this stub only guards against an
    un-wired live path.
    """

    def fetch(url):  # pragma: no cover - offline tests replace _build_category_index
        raise RuntimeError("no live OAI fetcher wired; production must inject one")

    return fetch


def _open_chunk_store(manifest, run_dir: Path, window: dict):
    """Open the S3-backed chunk store. Monkeypatched to a fake in tests.

    Production builds an ``s3_client`` (boto3, requester-pays) behind the
    ChunkStore; tests replace this function so no S3 call is ever made.
    """
    from icepick.allocation.bulk.chunk_store import ChunkStore, s3_client

    work_dir = run_dir / "_chunks"
    work_dir.mkdir(parents=True, exist_ok=True)
    return ChunkStore(s3_client(), work_dir=work_dir, max_resident=2)


# --- adapter-local chunk journal ----------------------------------------------


class _ChunkJournal:
    """Chunk-level resume journal at ``<progress_dir>/chunks_done.jsonl``.

    Lives DIRECTLY in the checkpoint's progress dir (``<run_dir>/_progress``),
    alongside the checkpoint's own ledger files (§4, amended wording). A
    journaled chunk is never re-downloaded on resume. Kept adapter-local
    (checkpoint.py is orchestrator-owned and must not change): the same
    append+flush, torn-tail-tolerant JSONL discipline the checkpoint uses.
    """

    def __init__(self, progress_dir):
        self._dir = Path(progress_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "chunks_done.jsonl"
        self._done: dict = {}
        self._load()

    def done(self, filename: str) -> bool:
        return filename in self._done

    def mark(self, filename: str, papers: int) -> None:
        if filename in self._done:
            return
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"filename": filename, "papers": int(papers)}) + "\n")
            fh.flush()
        self._done[filename] = int(papers)

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail from a kill mid-write; skip it
                if isinstance(row, dict) and "filename" in row:
                    self._done[row["filename"]] = row.get("papers", 0)


# --- budget-exhaustion pause --------------------------------------------------
#
# PERMANENT (decided at F2 integration): this local pause-signal is NOT merged
# with realmath.scrape._BudgetExhausted. Bulk drives its own acquisition loop
# rather than calling realmath.scrape(), so it keeps its own signal with
# semantics identical to realmath's — budget exhaustion is a checkpointed
# pause, never an operator-facing exception. BaseException is deliberate: the
# extractors' per-item ``except Exception`` guards must not swallow it,
# exactly as they must not swallow KeyboardInterrupt.
class _BudgetExhausted(BaseException):
    """The approved call budget is spent. A checkpointed pause, not an error."""


# Re-exported constants so downstream imports mirror realmath_scrape's surface.
__all__ = [
    "plan",
    "estimate",
    "run",
    "normalise",
    "ScrapeRunResult",
    "NormaliseResult",
    "PAPERS_PER_RECORD",
    "CANDIDATES_PER_PAPER",
    "ESTIMATE_SAFETY_MULTIPLIER",
    "SOURCE_ARXIV_BULK",
]

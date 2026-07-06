"""RealMath-style scrape adapter.

Preserves the source pipeline stages in its run layout: retrieve papers,
extract LaTeX, extract theorem or problem candidates, generate QA
records, verify answer form, hand off JSONL. Reposts collapse in the
paper pool (arxiv-id and title dedup); duplicate statements are dropped
from the candidate pool so the same record never reaches handoff twice.

``run`` executes only from an approved manifest. In ``production`` it
scrapes arXiv in-house through ``icepick.allocation.scrape.realmath`` —
the manifest's ``scrape_window`` selects the category, date window, and
primary-only filter — then normalises the candidates and writes the
handoff. No shell-out to the provenance repo. In ``flow_testing`` it
replays a local fixture (the manifest's ``calibration_sheet``: one JSONL
row per QA candidate in the upstream shape ``link / question / answer /
tier / truth``, plus optional ``title`` and ``arxiv_id``) instead of
scraping — no network, deterministic handoff. Both modes funnel their raw
candidates through one normalise + run-layout writer. Flow-testing
manifests may be auto-approved by their creator since replay spends
nothing; scraping runs must not be.

Output layout per run::

    <output_dir>/runs/<run_id>/
      manifest.json
      handoff/records.jsonl        <- the only file processing consumes
      handoff/surplus_records.jsonl <- cap overflow, canonical + mount-ready
                                       (only when the breadth/target caps bit;
                                       never rejected, just not auto-selected)
      raw/papers.jsonl
      raw/extracted_candidates.jsonl
      raw/qa_candidates.jsonl
      raw/quarantined.jsonl        <- only when candidates were dropped
      reports/source_report.md
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from icepick.allocation.manifests import require_approved, write_manifest
from icepick.contracts.manifests import (
    MODE_FLOW_TESTING,
    MODE_PRODUCTION,
    MODE_VALUES,
    SOURCE_REALMATH_SCRAPE,
    ProposedPlan,
)
from icepick.contracts.records import (
    PROVENANCE_COMPUTED,
    PROVENANCE_EXTRACTED,
    PROVENANCE_VALUES,
    TRUTH_POLICY_EXTRACTED,
    TRUTH_POLICY_TRUSTED,
    TRUTH_POLICY_VALUES,
)

# Planning ratios for estimates. Local latex-mode run reports include both
# sparse (3 theorem candidates/paper) and theorem-dense (60 in one paper)
# pulls; the attached live-pilot notes for math.NT/math.AP record 2-37 mined
# theorems/paper. CANDIDATES_PER_PAPER is a central value of that 2-37 range,
# NOT its ceiling: budgeting every paper as if it were the densest observed
# pull forced operators to approve budgets several times realistic spend,
# which gutted --call-budget as a guardrail. Headroom for dense pulls comes
# from ESTIMATE_SAFETY_MULTIPLIER below instead. Estimates still round against
# the operator so allocation never silently increases paid calls mid-run.
PAPERS_PER_RECORD = 4
CANDIDATES_PER_PAPER = 13
# Safety margin applied to the whole call estimate (rounded up) so routine
# variance above the central ratios doesn't pause a run at its budget ceiling.
# A run denser than the margin covers still loses nothing: budget exhaustion
# checkpoints and pauses (scrape's _BudgetExhausted), and re-running the same
# command resumes under a freshly approved budget. The multiplier trades that
# rare pause for a budget number an operator can sanity-check against spend.
ESTIMATE_SAFETY_MULTIPLIER = 1.5
_PAGE_SIZE_ESTIMATE = 50  # arXiv Atom results per query (mirrors scrape._PAGE_SIZE)

_PLAN_REQUIRED_FIELDS = {"source_name", "target_count", "requested_by", "requested_at"}
_PLAN_OPTIONAL_FIELDS = {"families", "scrape_window", "fixture_path", "notes"}
_SCRAPE_WINDOW_FIELDS = {
    "year", "month", "category", "max_papers", "max_per_paper", "primary_only", "extraction",
    "exclude_arxiv_ids",  # continuation: papers a prior run already consumed
}

_NORMALISE_REQUIRED_FIELDS = {"source_name", "candidates"}
_NORMALISE_OPTIONAL_FIELDS = {"families", "truth_policy"}

# Raw-row keys copied through to the top level because ingest already
# accepts them; everything else lands under ``metadata``.
_PASSTHROUGH_KEYS = (
    "tier",
    "truth_strings",
    "label",
    "pass_at_k",
    "n_correct",
    "correct",
    "n_wrong",
    "wrong",
    "wrong_complete",
    "n_degenerate",
    "degenerate",
    "modal_wrong",
    "top_wrong_share",
    "params",
)
_CANONICAL_KEYS = frozenset(_PASSTHROUGH_KEYS) | {
    "statement",
    "question",
    "problem",
    "answer",
    "truth",
    "arxiv_id",
    "family",
    "provenance",
    "truth_policy",
    "metadata",
    "generated",
    "link",
}

# LaTeX macro residue the upstream verifier refuses as truth; answers
# carrying these are handed off with a warning so an operator verifies
# the answer form before the record is trusted.
_JUNK_ANSWER_MARKERS = (
    "\\mathrm", "\\mathbb", "\\mathcal", "\\mathbf", "\\mathsf",
    "\\operatorname", "\\widetilde", "\\widehat", "\\cdots", "\\ldots",
    "\\displaystyle", "\\boldsymbol", "\\mathfrak", "\\mathscr",
)

_ARXIV_LINK_RE = re.compile(r"(?:^|/|\.)arxiv\.org/(?:abs|pdf)/([^?#\s]+)")


@dataclass
class NormaliseResult:
    """What ``normalise`` returns: canonical records plus the audit trail."""

    records: list
    quarantined: list  # list[dict] with ``reason`` and the offending candidate
    duplicates_dropped: int
    warnings: list


@dataclass
class ScrapeRunResult:
    """What ``run`` returns. ``handoff_path`` is what to feed the pipeline."""

    run_id: str
    processor_mode: str
    source_name: str
    calibration_replay: bool
    record_count: int
    paper_count: int
    candidate_count: int
    duplicates_dropped: int
    quarantined_count: int
    handoff_path: Path
    manifest_path: Path
    report_path: Path
    raw_dir: Path
    warnings: list = field(default_factory=list)
    acquisition: Optional[dict] = None  # arXiv/e-print/LLM call counts (production only)
    interrupted: bool = False  # paused (Ctrl-C); rerun the same command to resume
    progress_dir: Optional[Path] = None  # checkpoint store (production only)
    surplus_count: int = 0  # accepted rows past the caps, preserved (never dropped)
    surplus_path: Optional[Path] = None  # mount-ready surplus records (only when surplus_count > 0)


def plan(request):
    """Build a ``ProposedPlan`` from a request dict.

    Pure: no scraping, no external calls, no writes. Unknown or missing
    request fields are refused rather than guessed. The expected
    flow-testing fixture path, when provided, is recorded in the plan
    notes so approval sees it.
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
    scrape_window = _validated_scrape_window(request.get("scrape_window"))

    notes = str(request.get("notes") or "")
    fixture_path = request.get("fixture_path")
    if fixture_path:
        fixture_note = f"flow_testing fixture: {fixture_path}"
        notes = f"{notes}; {fixture_note}" if notes else fixture_note

    return ProposedPlan(
        source_type=SOURCE_REALMATH_SCRAPE,
        requested_by=str(request["requested_by"]),
        requested_at=str(request["requested_at"]),
        source_name=str(request["source_name"]),
        target_count=target_count,
        notes=notes,
        families=list(request.get("families") or []),
        scrape_window=scrape_window,
        estimated_calls=_estimated_calls(target_count, _extraction_of(scrape_window)),
        estimated_cost_usd=None,  # unknowable without provider pricing; left unset
    )


def estimate(plan):
    """Describe the expected work for a plan before approval.

    Never performs live scraping — every number is derived from the
    plan's ``target_count`` through the conservative planning ratios.
    """
    if plan.source_type != SOURCE_REALMATH_SCRAPE:
        raise ValueError(
            f"estimate expects source_type '{SOURCE_REALMATH_SCRAPE}', "
            f"got {plan.source_type!r}"
        )
    if plan.target_count <= 0:
        raise ValueError(f"target_count must be positive, got {plan.target_count!r}")

    extraction = _extraction_of(plan.scrape_window)
    expected_papers = plan.target_count * PAPERS_PER_RECORD
    expected_candidates = expected_papers * CANDIDATES_PER_PAPER

    # Which acquisition calls this extraction mode actually spends.
    call_kinds = ["arxiv_query"]
    expected_llm_calls = 0
    prerequisites = ["requests (in-house arXiv Atom scrape)",
                     "network access to export.arxiv.org (production only)"]
    if extraction in ("latex", "qa"):
        call_kinds.append("latex_source_fetch")
    if extraction == "qa":
        call_kinds.append("qa_generation")
        expected_llm_calls = expected_candidates  # one Sonnet Q+A call per mined theorem
        prerequisites += ["anthropic SDK ([judge] extra)",
                          "Anthropic key via ANTHROPIC_KEY_FILE or ANTHROPIC_API_KEY"]

    return {
        "source_type": SOURCE_REALMATH_SCRAPE,
        "source_name": plan.source_name,
        "target_count": plan.target_count,
        "extraction": extraction,
        "expected_papers": expected_papers,
        "expected_candidates": expected_candidates,
        "expected_handoff_records": plan.target_count,
        "estimated_calls": _estimated_calls(plan.target_count, extraction),
        "call_kinds": call_kinds,
        "expected_llm_calls": expected_llm_calls,
        # LLM tokens per call depend on the QA prompt/response size; the call
        # count is the budgeted quantity, priced by the operator's provider rates.
        "token_budget": None,
        "local_prerequisites": prerequisites,
    }


def run(manifest, *, now: Optional[datetime] = None):
    """Execute an acquisition run from an approved manifest.

    Every gate — source type, mode, approval, call budget, output
    containment — is validated before any work. ``flow_testing`` replays
    the manifest's ``calibration_sheet`` fixture; ``production`` scrapes
    arXiv in-house from the manifest's ``scrape_window``. Both funnel the
    raw candidates through the same normalise + run-layout writer.
    """
    _validate_manifest(manifest)
    run_dir = _run_dir(manifest)
    if manifest.processor_mode == MODE_FLOW_TESTING:
        candidates = _read_fixture_candidates(manifest, run_dir)
        return _write_run(manifest, run_dir, candidates, calibration_replay=True, now=now)
    scrape_result, checkpoint = _scrape_candidates(manifest, run_dir)
    acquisition = {
        "arxiv_queries": scrape_result.queries,
        "latex_fetches": scrape_result.latex_fetches,
        "qa_calls": scrape_result.qa_calls,
        "qa_model": scrape_result.qa_model,
        "rate_limit_events": scrape_result.rate_limit_events,
        "rate_limit_backoff_seconds": scrape_result.rate_limit_backoff_seconds,
        "rate_limit_statuses": scrape_result.rate_limit_statuses,
        "token_usage": scrape_result.token_usage,
        "total_calls": (
            scrape_result.queries
            + scrape_result.latex_fetches
            + scrape_result.qa_calls
        ),
        "call_budget": manifest.call_budget,
        "resumed_papers": scrape_result.resumed_papers,
    }
    outcome = _write_run(
        manifest,
        run_dir,
        scrape_result.candidates,
        calibration_replay=False,
        extra_warnings=list(scrape_result.warnings),
        acquisition=acquisition,
        interrupted=scrape_result.interrupted,
        progress_dir=checkpoint.progress_dir,
        surplus=scrape_result.surplus,
        now=now,
    )
    if not scrape_result.interrupted:
        checkpoint.mark_complete()
    return outcome


def normalise(raw_outputs):
    """Convert raw scraper candidate rows into canonical record dicts.

    ``raw_outputs`` is a dict with ``source_name`` and ``candidates``
    (upstream-shaped rows), plus optional ``families`` and
    ``truth_policy``. Candidates without a usable statement are
    quarantined with a reason; duplicate statements are dropped before
    handoff; source-specific extras move under ``metadata`` instead of
    growing the top-level schema. Records whose truth was generated
    rather than extracted keep ``provenance = "computed"`` so
    groundtruth can discard them.
    """
    source_name, candidates, families, truth_policy = _validated_raw_outputs(raw_outputs)
    family_default = families[0] if len(families) == 1 else "realmath"

    records: list = []
    quarantined: list = []
    warnings: list = []
    seen_statements: set = set()
    duplicates_dropped = 0

    for index, row in enumerate(candidates):
        if not isinstance(row, dict):
            quarantined.append({"reason": "candidate is not a JSON object", "candidate_index": index})
            continue
        statement = _first_str(row, ("statement", "question", "problem"))
        if not statement:
            quarantined.append(
                {"reason": "missing statement", "candidate_index": index, "candidate": row}
            )
            continue
        dedup_key = " ".join(statement.lower().split())
        if dedup_key in seen_statements:
            duplicates_dropped += 1
            continue

        try:
            record = _canonical_record(
                row,
                statement=statement,
                source_name=source_name,
                family_default=family_default,
                truth_policy_override=truth_policy,
                candidate_index=index,
                warnings=warnings,
            )
        except ValueError as exc:
            quarantined.append(
                {"reason": str(exc), "candidate_index": index, "candidate": row}
            )
            continue

        seen_statements.add(dedup_key)
        records.append(record)

    return NormaliseResult(
        records=records,
        quarantined=quarantined,
        duplicates_dropped=duplicates_dropped,
        warnings=warnings,
    )


# --- manifest gates -----------------------------------------------------------


def _validate_manifest(manifest) -> None:
    if manifest.source_type != SOURCE_REALMATH_SCRAPE:
        raise ValueError(
            f"manifest source_type must be '{SOURCE_REALMATH_SCRAPE}', "
            f"got {manifest.source_type!r}"
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
        raise ValueError(f"target_count must be a positive integer, got {manifest.target_count!r}")
    if manifest.processor_mode == MODE_PRODUCTION:
        extraction = _extraction_of(manifest.scrape_window)
        estimated = _estimated_calls(manifest.target_count, extraction)
        if estimated > budget:
            raise ValueError(
                f"call_budget {budget} is below the {estimated} calls estimated for "
                f"target_count {manifest.target_count} ({extraction}); refusing rather than over-running"
            )
    if not manifest.output_dir:
        raise ValueError("manifest output_dir must be set")
    if manifest.truth_policy and manifest.truth_policy not in TRUTH_POLICY_VALUES:
        raise ValueError(
            f"unknown truth_policy {manifest.truth_policy!r}; allowed: {TRUTH_POLICY_VALUES}"
        )
    if manifest.processor_mode == MODE_FLOW_TESTING and not manifest.calibration_sheet:
        raise ValueError("flow_testing requires calibration_sheet (the local fixture path)")


def _run_dir(manifest) -> Path:
    output_dir = Path(manifest.output_dir)
    run_dir = output_dir / "runs" / manifest.run_id
    resolved = run_dir.resolve()
    if resolved.parent != (output_dir / "runs").resolve() or resolved.name != manifest.run_id:
        raise ValueError(
            f"run directory {run_dir} escapes or breaks the documented "
            f"{output_dir}/runs/<run_id> layout"
        )
    return run_dir


def _extraction_of(scrape_window) -> str:
    return (scrape_window or {}).get("extraction") or "abstract"


def _estimated_calls(target_count: int, extraction: str = "abstract") -> int:
    """Estimate acquisition calls, aware of what each extraction mode spends.

    ``abstract`` spends only arXiv Atom queries; ``latex`` adds one e-print
    source fetch per paper; ``qa`` additionally spends one Sonnet Q+A call
    per mined theorem. Theorem counts use the central CANDIDATES_PER_PAPER
    ratio. The summed expectation is then padded by ESTIMATE_SAFETY_MULTIPLIER,
    so an approver's ``call_budget`` errs toward finishing without a mid-run
    pause while staying within sight of realistic spend.
    """
    expected_papers = target_count * PAPERS_PER_RECORD
    calls = max(1, -(-expected_papers // _PAGE_SIZE_ESTIMATE))  # arXiv Atom pages
    if extraction in ("latex", "qa"):
        calls += expected_papers  # one e-print source fetch per paper
    if extraction == "qa":
        calls += expected_papers * CANDIDATES_PER_PAPER  # one Sonnet Q+A call per theorem
    return math.ceil(calls * ESTIMATE_SAFETY_MULTIPLIER)


# --- run execution (shared by flow_testing replay and production scrape) ------


def _read_fixture_candidates(manifest, run_dir: Path) -> list:
    """Load flow-testing candidate rows from the manifest's fixture."""
    fixture = Path(manifest.calibration_sheet)
    if not fixture.is_file():
        raise FileNotFoundError(f"calibration fixture not found: {fixture}")
    if run_dir.resolve() in fixture.resolve().parents:
        raise ValueError(f"fixture {fixture} must live outside the run output directory")
    return _read_jsonl(fixture)


def _scrape_candidates(manifest, run_dir: Path) -> tuple:
    """Acquire candidate rows by scraping arXiv in-house. Network-bound.

    Delegates to the dedicated ``allocation.scrape.realmath`` module so the
    adapter stays about gating and layout, not HTTP. The scrape runs under
    a ``ScrapeCheckpoint`` (``<run_dir>/_progress/``): every finished paper
    is committed to disk and QA answers are cached, so a killed or paused
    run resumes by re-running the same command without redoing paid work.
    """
    from icepick.allocation.scrape import realmath as realmath_source
    from icepick.allocation.scrape.checkpoint import ScrapeCheckpoint

    checkpoint = ScrapeCheckpoint(run_dir / "_progress")
    result = realmath_source.scrape(
        scrape_window=manifest.scrape_window,
        source_name=manifest.source_name,
        families=list(manifest.families or []),
        target_count=manifest.target_count,
        call_budget=manifest.call_budget,
        checkpoint=checkpoint,
    )
    return result, checkpoint


def _write_run(
    manifest,
    run_dir: Path,
    candidates: list,
    *,
    calibration_replay: bool,
    extra_warnings: Optional[list] = None,
    acquisition: Optional[dict] = None,
    interrupted: bool = False,
    progress_dir: Optional[Path] = None,
    surplus: Optional[list] = None,
    now: Optional[datetime] = None,
) -> ScrapeRunResult:
    """Normalise candidates and write the full run layout. Shared by both modes."""
    now = now or datetime.now(timezone.utc)
    papers, duplicate_titles = _paper_pool(candidates)
    result = normalise(
        {
            "source_name": manifest.source_name,
            "candidates": candidates,
            "families": list(manifest.families or []),
            "truth_policy": manifest.truth_policy,
        }
    )
    if calibration_replay:
        for record in result.records:
            record.setdefault("metadata", {})["calibration_replay"] = True

    # Cap overflow ("never reject good theorems"): rows the breadth/target
    # caps kept out of the handoff are normalised through the same funnel and
    # preserved next to it, canonical and mount-ready. Rows that duplicate a
    # handoff statement are redundant, not good surplus — they are dropped
    # and counted.
    surplus_result = None
    surplus_records: list = []
    surplus_duplicates = 0
    if surplus:
        surplus_result = normalise(
            {
                "source_name": manifest.source_name,
                "candidates": list(surplus),
                "families": list(manifest.families or []),
                "truth_policy": manifest.truth_policy,
            }
        )
        handoff_statements = {record["statement"] for record in result.records}
        for record in surplus_result.records:
            if record["statement"] in handoff_statements:
                surplus_duplicates += 1
            else:
                surplus_records.append(record)
        surplus_duplicates += surplus_result.duplicates_dropped

    manifest_path = write_manifest(manifest, manifest.output_dir)
    raw_dir = run_dir / "raw"
    _write_jsonl(raw_dir / "papers.jsonl", papers)
    _write_jsonl(
        raw_dir / "extracted_candidates.jsonl",
        [
            {"link": row.get("link"), "statement": _first_str(row, ("statement", "question", "problem"))}
            for row in candidates
            if isinstance(row, dict)
        ],
    )
    _write_jsonl(raw_dir / "qa_candidates.jsonl", candidates)
    quarantined_rows = list(result.quarantined) + [
        {**item, "reason": f"[surplus] {item.get('reason', '')}"}
        for item in (surplus_result.quarantined if surplus_result else [])
    ]
    if quarantined_rows:
        _write_jsonl(raw_dir / "quarantined.jsonl", quarantined_rows)
    else:
        # A re-run of the same run_id must not leave a stale quarantine file
        # claiming drops that this run never made.
        (raw_dir / "quarantined.jsonl").unlink(missing_ok=True)

    handoff_path = run_dir / "handoff" / "records.jsonl"
    _write_jsonl(handoff_path, result.records)
    surplus_path = run_dir / "handoff" / "surplus_records.jsonl"
    if surplus_records:
        _write_jsonl(surplus_path, surplus_records)
    else:
        # Same stale-file hygiene as quarantine: a re-run with no surplus
        # must not leave an old surplus file behind.
        surplus_path.unlink(missing_ok=True)

    warnings = list(extra_warnings or []) + list(result.warnings)
    if surplus_result:
        warnings += [f"[surplus] {warning}" for warning in surplus_result.warnings]
    if duplicate_titles:
        warnings.append(
            f"paper pool: dropped {duplicate_titles} duplicate paper titles from raw/papers.jsonl"
        )
    if surplus_duplicates:
        warnings.append(
            f"surplus: dropped {surplus_duplicates} rows duplicating handoff or surplus statements"
        )

    outcome = ScrapeRunResult(
        run_id=manifest.run_id,
        processor_mode=manifest.processor_mode,
        source_name=manifest.source_name,
        calibration_replay=calibration_replay,
        record_count=len(result.records),
        paper_count=len(papers),
        candidate_count=len(candidates),
        duplicates_dropped=result.duplicates_dropped,
        quarantined_count=len(quarantined_rows),
        handoff_path=handoff_path,
        manifest_path=manifest_path,
        report_path=run_dir / "reports" / "source_report.md",
        raw_dir=raw_dir,
        warnings=warnings,
        acquisition=acquisition,
        interrupted=interrupted,
        progress_dir=progress_dir,
        surplus_count=len(surplus_records),
        surplus_path=surplus_path if surplus_records else None,
    )
    _write_report(outcome, result, quarantined_rows=quarantined_rows, created_at=now)
    return outcome


def _paper_pool(candidates) -> tuple:
    """The unique paper pool implied by the candidate rows.

    Mirrors the upstream scraper's pre-pass@k title dedup: papers sharing
    a normalised title are dropped from the pool after the first, so
    reposts and revisions never spend downstream budget. Each paper is
    considered exactly once, however many candidate rows it contributed.
    Returns (papers, duplicate_titles).
    """
    papers: list = []
    seen_ids: set = set()
    seen_titles: set = set()
    duplicate_titles = 0
    for row in candidates:
        if not isinstance(row, dict):
            continue
        link = str(row.get("link") or "")
        arxiv_id = row.get("arxiv_id") or _arxiv_id_from_link(link)
        paper_key = arxiv_id or link
        if not paper_key or paper_key in seen_ids:
            continue
        seen_ids.add(paper_key)
        # Scraper candidates carry the title under metadata; mounted/fixture rows
        # carry it top-level. Check both so production raw/papers.jsonl keeps titles.
        raw_title = row.get("title") or (row.get("metadata") or {}).get("title")
        title = " ".join(str(raw_title or "").lower().split())
        if title and title in seen_titles:
            duplicate_titles += 1
            continue
        if title:
            seen_titles.add(title)
        paper = {"arxiv_id": arxiv_id, "link": link or None, "title": raw_title}
        papers.append({k: v for k, v in paper.items() if v})
    return papers, duplicate_titles


# --- normalisation internals --------------------------------------------------


def _validated_raw_outputs(raw_outputs) -> tuple:
    if not isinstance(raw_outputs, dict):
        raise ValueError(f"raw_outputs must be a dict, got {type(raw_outputs).__name__}")
    unknown = set(raw_outputs) - _NORMALISE_REQUIRED_FIELDS - _NORMALISE_OPTIONAL_FIELDS
    if unknown:
        raise ValueError(f"unknown raw_outputs fields: {sorted(unknown)}")
    missing = _NORMALISE_REQUIRED_FIELDS - set(raw_outputs)
    if missing:
        raise ValueError(f"missing raw_outputs fields: {sorted(missing)}")
    source_name = raw_outputs["source_name"]
    if not source_name or not isinstance(source_name, str):
        raise ValueError(f"source_name must be a non-empty string, got {source_name!r}")
    candidates = raw_outputs["candidates"]
    if not isinstance(candidates, list):
        raise ValueError(f"candidates must be a list, got {type(candidates).__name__}")
    truth_policy = raw_outputs.get("truth_policy")
    if truth_policy and truth_policy not in TRUTH_POLICY_VALUES:
        raise ValueError(f"unknown truth_policy {truth_policy!r}; allowed: {TRUTH_POLICY_VALUES}")
    families = list(raw_outputs.get("families") or [])
    return source_name, candidates, families, truth_policy


def _canonical_record(
    row: dict,
    *,
    statement: str,
    source_name: str,
    family_default: str,
    truth_policy_override: Optional[str],
    candidate_index: int,
    warnings: list,
) -> dict:
    provenance = _resolve_provenance(row)
    record = {
        "source": source_name,
        "provenance": provenance,
        "truth_policy": _resolve_truth_policy(row, provenance, truth_policy_override),
        "statement": statement,
        "family": row.get("family") or family_default,
    }

    answer = row.get("answer")
    truth = row.get("truth")
    if answer in (None, "") and truth not in (None, ""):
        answer, truth = truth, None
    if answer not in (None, ""):
        record["answer"] = answer
        if any(marker in str(answer) for marker in _JUNK_ANSWER_MARKERS):
            warnings.append(
                f"candidate {candidate_index}: answer carries LaTeX macro residue; "
                "verify the answer form before trusting it"
            )
    if truth not in (None, "") and str(truth) != str(answer):
        record["truth"] = truth

    arxiv_id = row.get("arxiv_id") or _arxiv_id_from_link(str(row.get("link") or ""))
    if arxiv_id:
        record["arxiv_id"] = arxiv_id
    else:
        warnings.append(
            f"candidate {candidate_index}: no arxiv_id; groundtruth will discard it"
        )

    for key in _PASSTHROUGH_KEYS:
        if key in row and row[key] not in (None, ""):
            record[key] = row[key]

    stored_metadata = row.get("metadata")
    if stored_metadata is not None and not isinstance(stored_metadata, dict):
        raise ValueError(f"metadata must be a JSON object, got {type(stored_metadata).__name__}")
    metadata = dict(stored_metadata or {})
    if row.get("link"):
        metadata["link"] = row["link"]
    for key, value in row.items():
        if key not in _CANONICAL_KEYS:
            metadata[key] = value
    if metadata:
        record["metadata"] = metadata
    return record


def _resolve_provenance(row: dict) -> str:
    stored = row.get("provenance")
    if stored:
        if stored not in PROVENANCE_VALUES:
            raise ValueError(f"unknown provenance {stored!r}; allowed: {PROVENANCE_VALUES}")
        return stored
    if row.get("generated"):
        return PROVENANCE_COMPUTED
    return PROVENANCE_EXTRACTED


def _resolve_truth_policy(row: dict, provenance: str, override: Optional[str]) -> str:
    stored = row.get("truth_policy")
    if stored:
        if stored not in TRUTH_POLICY_VALUES:
            raise ValueError(f"unknown truth_policy {stored!r}; allowed: {TRUTH_POLICY_VALUES}")
        return stored
    if override:
        return override
    if provenance == PROVENANCE_COMPUTED:
        return TRUTH_POLICY_TRUSTED
    return TRUTH_POLICY_EXTRACTED


def _arxiv_id_from_link(link: str) -> Optional[str]:
    match = _ARXIV_LINK_RE.search(link)
    if not match:
        return None
    arxiv_id = match.group(1).strip("/")
    if arxiv_id.endswith(".pdf"):
        arxiv_id = arxiv_id[: -len(".pdf")]
    return re.sub(r"v\d+$", "", arxiv_id) or None


def _first_str(row: dict, keys) -> str:
    for key in keys:
        stripped = str(row.get(key) or "").strip()
        if stripped:
            return stripped
    return ""


def _validated_scrape_window(window) -> Optional[dict]:
    if window is None:
        return None
    if not isinstance(window, dict):
        raise ValueError(f"scrape_window must be a dict, got {type(window).__name__}")
    unknown = set(window) - _SCRAPE_WINDOW_FIELDS
    if unknown:
        raise ValueError(f"unknown scrape_window fields: {sorted(unknown)}")
    exclude = window.get("exclude_arxiv_ids")
    if exclude is not None:
        if not isinstance(exclude, list) or not all(
            isinstance(item, str) and item.strip() for item in exclude
        ):
            raise ValueError("exclude_arxiv_ids must be a list of non-empty arxiv id strings")
    return dict(window)


# --- file IO ------------------------------------------------------------------


def _read_jsonl(path: Path) -> list:
    rows: list = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
    return rows


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _write_report(
    outcome: ScrapeRunResult,
    result: NormaliseResult,
    *,
    quarantined_rows: Optional[list] = None,
    created_at: datetime,
) -> Path:
    """Markdown source report: outcome first, then counts, warnings, drops.

    ``quarantined_rows`` is the merged main + surplus quarantine list backing
    ``raw/quarantined.jsonl``; it defaults to the main pool's for callers that
    have no surplus.
    """
    quarantined_rows = result.quarantined if quarantined_rows is None else quarantined_rows
    provenance_counts: dict = {}
    for record in result.records:
        provenance_counts[record["provenance"]] = provenance_counts.get(record["provenance"], 0) + 1
    provenance_line = (
        ", ".join(f"{name}: {count}" for name, count in sorted(provenance_counts.items()))
        or "none"
    )

    lines = [
        "# RealMath scrape source report",
        "",
        f"Wrote **{outcome.record_count}** handoff records for source `{outcome.source_name}`.",
        f"Feed processing from: `{outcome.handoff_path}`",
        "",
    ]
    if outcome.interrupted:
        lines += [
            "**Status: INTERRUPTED — partial run, resumable.** Progress is",
            "checkpointed; rerun the same `allocation run --manifest` command to",
            "continue without redoing acquired papers. Do not feed this partial",
            "handoff to processing unless a partial corpus is intended.",
            "",
        ]
    lines += [
        f"- run: {outcome.run_id}",
        f"- processor_mode: {outcome.processor_mode}",
        f"- calibration_replay: {str(outcome.calibration_replay).lower()}",
        f"- source: {outcome.source_name}",
        f"- provenance: {provenance_line}",
        f"- created_at: {created_at.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## Counts",
        "",
        "| stage | count |",
        "| --- | --- |",
        f"| papers (unique pool) | {outcome.paper_count} |",
        f"| candidates read | {outcome.candidate_count} |",
        f"| duplicate statements dropped | {outcome.duplicates_dropped} |",
        f"| quarantined | {outcome.quarantined_count} |",
        f"| handoff records | {outcome.record_count} |",
        f"| surplus records (cap overflow, preserved) | {outcome.surplus_count} |",
        "",
    ]
    if outcome.surplus_count:
        lines += [
            "## Surplus — accepted past the caps",
            "",
            f"{outcome.surplus_count} accepted theorems exceeded the breadth/target caps",
            "(`max_per_paper` / `target_count`). They are never rejected: already",
            "canonical and mount-ready at:",
            "",
            f"    {outcome.surplus_path}",
            "",
            "Port them into the corpus with:",
            "",
            f"    icepick allocation mount --path {outcome.surplus_path} \\",
            f"      --source {outcome.source_name}_surplus --provenance extracted \\",
            f"      --output-dir <output_dir>",
            "",
        ]
    if outcome.acquisition:
        acq = outcome.acquisition
        budget = acq.get("call_budget")
        lines += [
            "## Spend (acquisition calls)",
            "",
            "| kind | count |",
            "| --- | --- |",
            f"| arxiv_query | {acq.get('arxiv_queries', 0)} |",
            f"| latex_source_fetch | {acq.get('latex_fetches', 0)} |",
            f"| qa_generation ({acq.get('qa_model') or 'model unrecorded'}) | {acq.get('qa_calls', 0)} |",
            f"| total | {acq.get('total_calls', 0)}"
            + (f" / {budget} budgeted" if budget is not None else "") + " |",
            "",
        ]
        if acq.get("resumed_papers"):
            lines += [
                f"Resumed: {acq['resumed_papers']} papers served from the checkpoint "
                "(no refetch, no re-billing).",
                "",
            ]
        if acq.get("rate_limit_events"):
            statuses = acq.get("rate_limit_statuses") or {}
            status_text = ", ".join(f"{status}: {count}" for status, count in sorted(statuses.items()))
            lines += [
                "## arXiv throttle telemetry",
                "",
                "Totals span the run's whole lifetime — every invocation of this",
                "run_id, including any the limiter killed before a paper committed.",
                "",
                "| metric | value |",
                "| --- | --- |",
                f"| 429/503 encounters | {acq.get('rate_limit_events', 0)} |",
                f"| total backoff seconds | {acq.get('rate_limit_backoff_seconds', 0.0):.1f} |",
                f"| status counts | {status_text or 'n/a'} |",
                "",
            ]
        token_usage = acq.get("token_usage") or {}
        if any(token_usage.values()):
            lines += [
                "## LLM token usage",
                "",
                "| metric | tokens |",
                "| --- | --- |",
            ]
            for key in sorted(token_usage):
                lines.append(f"| {key} | {token_usage[key]} |")
            lines.append("")
    lines += [
        "## Warnings",
        "",
    ]
    lines += [f"- {warning}" for warning in outcome.warnings] or ["- none"]
    lines += ["", "## Drops", ""]
    lines += [
        f"- candidate {item.get('candidate_index', '?')}: {item['reason']}"
        for item in quarantined_rows
    ] or ["- none (nothing quarantined)"]
    if quarantined_rows:
        lines.append(f"- full quarantined rows: `{outcome.raw_dir / 'quarantined.jsonl'}`")
    lines += [
        "",
        "## Outputs",
        "",
        f"- handoff (processing input): `{outcome.handoff_path}`",
        f"- manifest: `{outcome.manifest_path}`",
        f"- raw artifacts: `{outcome.raw_dir}`",
        f"- this report: `{outcome.report_path}`",
    ]
    if outcome.surplus_count:
        lines.append(f"- surplus (mount-ready cap overflow): `{outcome.surplus_path}`")
    lines += [
        "",
    ]
    outcome.report_path.parent.mkdir(parents=True, exist_ok=True)
    outcome.report_path.write_text("\n".join(lines), encoding="utf-8")
    return outcome.report_path

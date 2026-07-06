"""In-house RealMath scraper: arXiv acquisition, no provenance-repo shell-out.

Stages, all IcePick code:

  1. ``build_query`` — compose a category query from the scrape window
     (``math.AP`` scrapes PDEs; a bare ``math`` scrapes every subcategory).
  2. ``default_arxiv_fetcher`` — page the arXiv Atom API over HTTP.
  3. ``parse_atom`` — parse entries into ``Paper`` records.
  4. ``scrape`` — orchestrate: page, primary-only filter, arxiv-id + title
     dedup, then run an extractor over each paper.
  5. ``default_extractor`` — turn a paper into raw candidate rows.

The result is a list of raw candidate rows; the ``realmath_scrape`` adapter
normalises them into canonical records and writes the handoff.

Network access is confined to two fetchers (``default_arxiv_fetcher`` and
``default_latex_source_fetcher``, both importing ``requests`` lazily);
everything else is pure and unit-testable by injecting them. Candidate
depth is selectable via ``scrape_window['extraction']``:

  - ``abstract`` (default) — one metadata candidate per paper.
  - ``latex`` — download the e-print source and mine theorem-like
    environments into candidate statements (with ``\\boxed`` answers when
    stated).
  - ``qa`` — turn each mined theorem into a self-contained question and its
    paper-stated answer via an LLM, keeping only verifiable answers.

Both the arXiv fetcher, the LaTeX-source fetcher, and the QA generator are
injectable, so the whole pipeline is unit-testable without network or API
keys. A custom ``extractor`` remains available for anything beyond these.
"""

from __future__ import annotations

import gzip
import io
import os
import re
import tarfile
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlencode

ARXIV_API = "https://export.arxiv.org/api/query"
# arXiv asks API clients to identify themselves; a missing UA invites throttling.
_USER_AGENT = "icepick-realmath-scraper (+https://arxiv.org/help/api)"
# arXiv's informal guidance is ~one request every few seconds, no bursts. We
# space ALL requests (Atom queries AND e-print fetches) at least this far
# apart, not just retries — the single biggest lever against 429/503
# throttling. 4s sits safely inside the "every few seconds" band with
# headroom over the bare 3s minimum. Tune via env for a slower link, or set
# 0 in offline tests.
_MIN_REQUEST_INTERVAL = float(os.environ.get("ICEPICK_ARXIV_MIN_INTERVAL", "4.0"))
_RATE_LIMIT_STATUSES = frozenset({429, 503})
_pace_lock = threading.Lock()
_last_request_at = 0.0
_http_session = None
_http_context = threading.local()

# Smaller pulls stay under the limiter where a big sweep trips it: a live
# 50-paper pull came back clean when 100-paper pulls were 429'd. Requesting
# in small sequential chunks (paginated via start/max_results) is the
# paginate-don't-overload lever.
_PAGE_SIZE = 50
_PAGE_SIZE_FLOOR_AFTER_429 = 25
# Recovery mirror of the halve: after this many consecutive clean (never
# throttled) paced requests at a reduced page size, double the page size back
# toward _PAGE_SIZE (never above it). Without a ramp, one early 429 pins a
# long run at the floor and doubles its query count even after the endpoint
# recovers; with it, an endpoint that is still unhealthy just re-halves and
# the streak restarts, so a flapping limiter converges downward.
_PAGE_SIZE_RECOVERY_CLEAN_STREAK = 3
_MAX_PAGES = 100  # backstop against runaway pagination (2x pages since pages are half-size)

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

_VERSION_RE = re.compile(r"v\d+$")
_ABS_RE = re.compile(r"arxiv\.org/abs/([^\s?#]+)")


@dataclass
class Paper:
    """One arXiv entry, as parsed from the Atom feed."""

    arxiv_id: str
    link: str
    title: str
    abstract: str
    primary_category: str
    categories: list
    published: str


@dataclass
class ScrapeResult:
    """What ``scrape`` returns: raw candidate rows plus acquisition counts.

    ``queries`` / ``latex_fetches`` / ``qa_calls`` are the acquisition calls
    actually spent — the numbers an operator's ``call_budget`` governs and
    the run report surfaces. ``qa_calls`` is the Sonnet Q+A reformulation,
    one call per mined theorem.

    ``surplus`` holds accepted rows that the breadth/target caps kept out of
    ``candidates``. They are already extracted (and, in qa mode, already paid
    for), so they are preserved for the operator — never silently dropped.

    The ``rate_limit_*`` fields cover the run's whole lifetime when a
    checkpoint is in play: 429/503 events are journaled to the progress
    store as they happen, so throttling that killed an earlier invocation
    still shows up in the invocation that finally completes. Without a
    checkpoint they cover this call only.
    """

    candidates: list
    papers_seen: int
    queries: int
    warnings: list = field(default_factory=list)
    latex_fetches: int = 0
    qa_calls: int = 0
    rate_limit_events: int = 0
    rate_limit_backoff_seconds: float = 0.0
    rate_limit_statuses: dict = field(default_factory=dict)
    token_usage: dict = field(default_factory=dict)
    interrupted: bool = False  # stopped early (Ctrl-C); disk state resumes it
    resumed_papers: int = 0  # papers served from the checkpoint, not refetched
    surplus: list = field(default_factory=list)  # cap overflow — preserved, mount-ready downstream
    qa_model: Optional[str] = None  # model the QA generator actually used (None if not qa / all-cached)


class _BudgetExhausted(BaseException):
    """The approved call budget is spent. A pause, not an error.

    ``BaseException`` deliberately: the extractors' per-item resilience
    guards (``except Exception``) must not swallow it, exactly as they
    must not swallow ``KeyboardInterrupt``. ``scrape`` catches it and
    returns a checkpointed, resumable partial result.
    """


def scrape(
    *,
    scrape_window: Optional[dict],
    source_name: str,
    target_count: int,
    families: Optional[list] = None,
    fetcher: Optional[Callable] = None,
    extractor: Optional[Callable] = None,
    call_budget: Optional[int] = None,
    checkpoint=None,
) -> ScrapeResult:
    """Acquire raw candidate rows from arXiv. See the module docstring.

    Pages the arXiv API until ``target_count`` candidates are collected,
    the paper pool is exhausted, or ``scrape_window['max_papers']`` /
    ``_MAX_PAGES`` is hit. ``primary_only`` drops cross-listed papers;
    reposts collapse by arxiv id and by normalised title.

    ``call_budget`` is a hard cap on total acquisition calls (arXiv queries
    + e-print fetches + Sonnet Q+A calls), checked before every paid
    call: a run never spends past it. Exhausting the budget pauses the run
    exactly like Ctrl-C (``interrupted=True``, checkpointed); each
    re-invocation gets a fresh budget and cached work costs nothing, so
    re-running the same command continues from where the money ran out.

    ``checkpoint`` (a ``ScrapeCheckpoint``) makes the run pausable instead
    of killable: every finished paper is committed to disk, QA answers are
    cached so a resume never re-bills, papers already committed are served
    from the store without refetching, and Ctrl-C stops cleanly between
    items with everything checkpointed (``interrupted=True`` on the result).
    """
    window = scrape_window or {}
    extraction = window.get("extraction")
    fetcher = fetcher or default_arxiv_fetcher
    query = build_query(window)
    category = window.get("category") or "math"
    primary_only = bool(window.get("primary_only"))
    max_papers = window.get("max_papers")
    max_per_paper = window.get("max_per_paper")  # corpus-breadth cap (see below)
    family = _lone_family(families)

    # Count the calls each extraction mode spends inside the extractor by
    # wrapping the source fetcher / QA generator it uses. A caller-supplied
    # ``extractor`` is used verbatim (counting is a production-path concern).
    # The budget is enforced HERE, before each paid call — a paper with many
    # theorems must not spend past the approved cap between outer checks.
    counts = {"queries": 0, "latex_fetches": 0, "qa_calls": 0}
    token_usage: dict = {}
    qa_model_used = {"name": None}  # actual model the QA generator resolved (key-file override wins)
    rate_limit_statuses: dict = {}
    rate_limit_events = 0
    rate_limit_backoff_seconds = 0.0
    effective_page_size = _PAGE_SIZE
    pending_429_page_halve = False
    consecutive_clean_requests = 0  # streak feeding the page-size ramp-up

    def acquisition_calls():
        return counts["queries"] + counts["latex_fetches"] + counts["qa_calls"]

    def charge(kind):
        if call_budget is not None and acquisition_calls() >= call_budget:
            raise _BudgetExhausted
        counts[kind] += 1

    def record_token_usage(kind, usage):
        if not usage:
            return
        for key, value in usage.items():
            token_key = f"{kind}_{key}"
            token_usage[token_key] = token_usage.get(token_key, 0) + int(value or 0)

    def on_rate_limit(status, sleep_seconds):
        nonlocal rate_limit_events, rate_limit_backoff_seconds, pending_429_page_halve
        nonlocal consecutive_clean_requests
        rate_limit_events += 1
        rate_limit_backoff_seconds += float(sleep_seconds or 0.0)
        status_key = str(status)
        rate_limit_statuses[status_key] = rate_limit_statuses.get(status_key, 0) + 1
        consecutive_clean_requests = 0  # any throttle event breaks the clean streak
        if int(status) == 429:
            pending_429_page_halve = True
        if checkpoint is not None:
            # Durable as it happens: an invocation the limiter kills before
            # its first paper commit must not take its telemetry with it.
            checkpoint.record_rate_limit(status, sleep_seconds)
            checkpoint.stamp_rate_limited()

    def on_success():
        nonlocal effective_page_size, pending_429_page_halve, consecutive_clean_requests
        if checkpoint is not None:
            checkpoint.clear_rate_limit()
        if pending_429_page_halve:
            effective_page_size = max(_PAGE_SIZE_FLOOR_AFTER_429, effective_page_size // 2)
            pending_429_page_halve = False
            consecutive_clean_requests = 0  # recovery request ends the episode; ramp counts from the next one
        elif effective_page_size < _PAGE_SIZE:
            # Gentle ramp-up. Every paced arXiv request (Atom page or e-print
            # fetch) that completes clean is evidence the throttle cleared;
            # after a streak of them, step the page size back up. Capped at
            # _PAGE_SIZE — the ramp never grows pages past the configured size.
            consecutive_clean_requests += 1
            if consecutive_clean_requests >= _PAGE_SIZE_RECOVERY_CLEAN_STREAK:
                effective_page_size = min(_PAGE_SIZE, effective_page_size * 2)
                consecutive_clean_requests = 0

    def counting_latex(arxiv_id, **kwargs):
        charge("latex_fetches")
        return default_latex_source_fetcher(arxiv_id, **kwargs)

    def counting_qa(statement, **kwargs):
        charge("qa_calls")
        return default_qa_generator(
            statement,
            usage_callback=lambda usage: record_token_usage("qa", usage),
            model_callback=lambda name: qa_model_used.__setitem__("name", name),
            **kwargs,
        )

    if checkpoint is not None:
        checkpoint.enforce_rate_limit_cooldown()
        checkpoint.begin()
        # Cache wraps the counter, so a cache hit spends (and counts)
        # nothing. Generator responses live in qa_cache.jsonl with
        # torn-tail tolerant JSONL semantics.
        counting_qa = checkpoint.caching_generator(counting_qa)

    if extractor is not None:
        run_extractor = lambda paper: extractor(paper, family=family)  # noqa: E731
    else:
        base = extractor_for(extraction)  # validates the mode name
        if extraction == "qa":
            run_extractor = lambda paper: qa_extractor(  # noqa: E731
                paper, family=family, source_fetcher=counting_latex,
                generator=counting_qa)
        elif extraction == "latex":
            run_extractor = lambda paper: latex_extractor(  # noqa: E731
                paper, family=family, source_fetcher=counting_latex)
        else:
            run_extractor = lambda paper: base(paper, family=family)  # noqa: E731

    candidates: list = []
    surplus: list = []
    warnings: list = []
    # Continuation support: ids listed in the window are treated as already
    # seen, so a follow-up run pages cheaply past papers a prior run consumed
    # (skipped before max_papers counting, e-print fetch, and QA spend) and
    # starts paying at the first unseen paper. Order-independent dedup.
    seen_ids: set = set(window.get("exclude_arxiv_ids") or [])
    seen_titles: set = set()
    papers_seen = 0
    resumed_papers = 0
    start = 0
    interrupted = False

    try:
        with _http_observers(on_rate_limit=on_rate_limit, on_success=on_success):
            while len(candidates) < target_count and counts["queries"] < _MAX_PAGES:
                charge("queries")
                page_size = effective_page_size
                xml_text = fetcher(query, start=start, max_results=page_size)
                papers = parse_atom(xml_text)
                if not papers:
                    break
                capped = False
                for paper in papers:
                    if max_papers is not None and papers_seen >= max_papers:
                        capped = True
                        break
                    if not paper.arxiv_id or paper.arxiv_id in seen_ids:
                        continue
                    seen_ids.add(paper.arxiv_id)
                    if primary_only and "." in category and paper.primary_category != category:
                        continue
                    norm_title = " ".join(paper.title.lower().split())
                    if norm_title and norm_title in seen_titles:
                        continue
                    if norm_title:
                        seen_titles.add(norm_title)
                    papers_seen += 1
                    stored = checkpoint.stored_candidates(paper.arxiv_id) if checkpoint else None
                    if stored is not None:
                        resumed_papers += 1
                        extracted = stored
                    else:
                        extracted = run_extractor(paper)
                        if checkpoint is not None:
                            checkpoint.commit(paper.arxiv_id, extracted)
                    kept_this_paper = 0
                    for candidate in extracted:
                        # Breadth cap: don't let one theorem-dense paper monopolise
                        # the target. (Spend is bounded separately by call_budget;
                        # in qa mode the extractor has already run, so this limits
                        # corpus contribution, not LLM calls.) Rows past a cap are
                        # already extracted and paid for — they go to ``surplus``,
                        # never on the floor.
                        if (
                            len(candidates) >= target_count
                            or (max_per_paper and kept_this_paper >= max_per_paper)
                        ):
                            surplus.append(candidate)
                            continue
                        candidates.append(candidate)
                        kept_this_paper += 1
                    if len(candidates) >= target_count:
                        break
                # Advance by the page size requested, not len(papers): entries
                # dropped for a missing id still occupy result slots. The page
                # size may shrink after a recovered 429 (and ramp back up after
                # a clean streak), so capture it before the request and advance
                # by that exact slot span.
                start += page_size
                if capped:
                    break
    except KeyboardInterrupt:
        # Pause, don't die: every finished paper is already committed, so
        # re-running the same command resumes here without redoing work.
        interrupted = True
        warnings.append(
            "interrupted (Ctrl-C); progress is checkpointed — rerun the same "
            "'allocation run --manifest' command to resume where it stopped"
        )
    except _BudgetExhausted:
        # Hard cap: never spend past the approved budget, mid-paper included.
        interrupted = True
        warnings.append(
            f"call budget {call_budget} exhausted after {acquisition_calls()} paid "
            "calls; checkpointed — rerun the same command to continue (already-"
            "cached work costs nothing against the fresh budget)"
        )

    if not candidates:
        warnings.append(
            f"arXiv scrape for source {source_name!r} produced no candidates for "
            f"query {query!r} (papers seen: {papers_seen}, queries: {counts['queries']})"
        )
    if target_count and len(candidates) > target_count:
        # Belt-and-braces: the selection loop never overfills, but if a trim
        # is ever needed it must preserve the overflow, not discard it.
        surplus[:0] = candidates[target_count:]
        candidates = candidates[:target_count]
    if checkpoint is not None:
        # Report run-LIFETIME throttle telemetry, not this invocation's
        # slice: a prior invocation 429-killed before any paper commit left
        # its events only in the checkpoint's durable log (its ScrapeResult
        # never existed). The lifetime totals already include the events
        # recorded above, so this replaces the in-memory counts.
        lifetime = checkpoint.rate_limit_telemetry()
        rate_limit_events = lifetime["events"]
        rate_limit_backoff_seconds = lifetime["backoff_seconds"]
        rate_limit_statuses = lifetime["statuses"]
    return ScrapeResult(
        candidates=candidates,
        surplus=surplus,
        papers_seen=papers_seen,
        queries=counts["queries"],
        warnings=warnings,
        latex_fetches=counts["latex_fetches"],
        qa_calls=counts["qa_calls"],
        rate_limit_events=rate_limit_events,
        rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        rate_limit_statuses=rate_limit_statuses,
        token_usage=token_usage,
        interrupted=interrupted,
        resumed_papers=resumed_papers,
        qa_model=qa_model_used["name"],
    )


def build_query(scrape_window: Optional[dict]) -> str:
    """Compose the arXiv ``search_query`` from a scrape window.

    A dotted category (``math.AP``) matches that subcategory exactly; a
    bare main category (``math``) wildcards to all its subcategories. A
    ``year`` (and optional ``month``) adds a lower bound on submission date.
    """
    window = scrape_window or {}
    category = window.get("category") or "math"
    query = f"cat:{category}" if "." in category else f"cat:{category}.*"
    year = window.get("year")
    if year:
        month = window.get("month") or 1
        lo = f"{int(year):04d}{int(month):02d}010000"
        query += f" AND submittedDate:[{lo} TO 999912312359]"
    return query


def default_arxiv_fetcher(
    query: str, *, start: int, max_results: int, timeout: float = 60, retries: int = 4, backoff: float = 3.0
) -> str:
    """Fetch one page of the arXiv Atom feed (retried). The only network code here."""
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return _http_get(
        f"{ARXIV_API}?{urlencode(params)}", timeout=timeout, retries=retries, backoff=backoff
    ).text


def _pace() -> None:
    """Space requests >= ``_MIN_REQUEST_INTERVAL`` apart. arXiv asks <=1 req/3s.

    Shared across the Atom query and e-print fetchers, since arXiv rate-limits
    by client across both. Holding the lock across the sleep also enforces the
    'single connection at a time' guidance if a caller ever parallelises.
    """
    global _last_request_at
    with _pace_lock:
        if _MIN_REQUEST_INTERVAL > 0:
            wait = _MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
            if wait > 0:
                time.sleep(wait)
        _last_request_at = time.monotonic()


def _session():
    """A reused ``requests.Session`` — one keep-alive connection, as arXiv prefers."""
    global _http_session
    if _http_session is None:
        import requests  # lazy: only production scraping needs the network

        _http_session = requests.Session()
        _http_session.headers.update({"User-Agent": _USER_AGENT})
    return _http_session


@contextmanager
def _http_observers(on_rate_limit=None, on_success=None):
    """Temporarily attach scrape-level observers to all paced arXiv requests."""
    old_rate_limit = getattr(_http_context, "on_rate_limit", None)
    old_success = getattr(_http_context, "on_success", None)
    _http_context.on_rate_limit = on_rate_limit
    _http_context.on_success = on_success
    try:
        yield
    finally:
        _http_context.on_rate_limit = old_rate_limit
        _http_context.on_success = old_success


def _notify_rate_limit(status: int, sleep_seconds: float, callback=None) -> None:
    for observer in (callback, getattr(_http_context, "on_rate_limit", None)):
        if observer is not None:
            observer(status, sleep_seconds)


def _notify_success(callback=None) -> None:
    for observer in (callback, getattr(_http_context, "on_success", None)):
        if observer is not None:
            observer()


def _http_get(
    url: str,
    *,
    timeout: float,
    retries: int,
    backoff: float,
    on_rate_limit=None,
    on_success=None,
):
    """Paced GET with exponential backoff; honors 429/503 Retry-After.

    Layered throttle avoidance (all five levers from arXiv's guidance):

      1. Hard delay: ``_pace`` spaces EVERY request >= _MIN_REQUEST_INTERVAL
         (4s) apart, not just retries — the primary lever.
      2. Exponential backoff: a 429/503 sleeps ``backoff * 2**(attempt-1)``
         → 3s, 6s, 12s with the default backoff=3.0. Doubling each
         consecutive failure so a persistent block is waited out, not
         hammered.
      3. Retry-After: if the server sent one, it overrides the computed
         backoff — obey exactly what arXiv asks.
      4. Single worker: ``_pace_lock`` is held across the sleep, so even a
         parallel caller collapses to one in-flight request. Do NOT
         parallelise arXiv fetches — multiple workers on one IP trip the
         limiter instantly.
      5. Pagination: callers request small ``max_results`` chunks (see
         _PAGE_SIZE) rather than one large sweep.

    On the final attempt ``raise_for_status`` surfaces the error so the
    caller (``scrape``) checkpoints and stops rather than looping into an
    active 429 — retrying into a live block only deepens it. A resume ~15-30
    min later, after the cooldown clears, picks up cleanly.
    """
    import requests  # lazy: only production scraping needs the network

    last_error = None
    for attempt in range(1, retries + 1):
        _pace()  # deliberate spacing before EVERY request, not just after a failure
        try:
            response = _session().get(url, timeout=timeout)
            if response.status_code in _RATE_LIMIT_STATUSES:
                sleep_seconds = (
                    _retry_after(response) or backoff * 2 ** (attempt - 1)
                    if attempt < retries else 0.0
                )
                _notify_rate_limit(
                    response.status_code, sleep_seconds, callback=on_rate_limit
                )
                if attempt < retries:
                    # Retry-After wins; else exponential backoff 3s, 6s, 12s.
                    time.sleep(sleep_seconds)
                    continue
            response.raise_for_status()
            _notify_success(callback=on_success)
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff * 2 ** (attempt - 1))
    raise last_error


def _retry_after(response) -> Optional[float]:
    """Seconds to wait per a 429/503 ``Retry-After`` header, if numeric."""
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def parse_atom(xml_text: str) -> list:
    """Parse an arXiv Atom feed into ``Paper`` records."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"arXiv response is not valid Atom XML: {exc}") from exc

    papers: list = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = _text(entry.find(f"{_ATOM}id"))
        arxiv_id = _arxiv_id_from_url(raw_id)
        if not arxiv_id:
            continue
        primary = entry.find(f"{_ARXIV}primary_category")
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                link=raw_id,
                title=" ".join(_text(entry.find(f"{_ATOM}title")).split()),
                abstract=" ".join(_text(entry.find(f"{_ATOM}summary")).split()),
                primary_category=primary.get("term") if primary is not None else "",
                categories=[c.get("term") for c in entry.findall(f"{_ATOM}category") if c.get("term")],
                published=_text(entry.find(f"{_ATOM}published")),
            )
        )
    return papers


def default_extractor(paper: Paper, *, family: Optional[str] = None) -> list:
    """Turn a paper into raw candidate rows (metadata level).

    This first in-house cut emits one candidate per paper whose statement
    is the abstract (falling back to the title). It carries no extracted
    answer — answer/theorem extraction is the seam: supply a richer
    ``extractor`` to ``scrape`` to lift candidates beyond this. The
    subject classification arXiv attaches is preserved under ``metadata``.
    """
    statement = paper.abstract or paper.title
    if not statement:
        return []
    candidate = {
        "link": paper.link,
        "arxiv_id": paper.arxiv_id,
        "statement": statement,
        "provenance": "extracted",
        "metadata": {
            "title": paper.title,
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "published": paper.published,
            "extraction": "abstract",
        },
    }
    if family:
        candidate["family"] = family
    return [candidate]


def extractor_for(name: Optional[str]) -> Callable:
    """Select the candidate extractor by name (``scrape_window['extraction']``).

    ``abstract`` (default) emits one metadata candidate per paper.
    ``latex`` downloads the e-print source and mines theorem-like
    environments into candidate statements. ``qa`` goes one step further:
    it turns each theorem into a self-contained question and its
    paper-stated answer via an LLM, keeping only verifiable answers.
    Unknown names are refused.
    """
    if name in (None, "", "abstract"):
        return default_extractor
    if name == "latex":
        return latex_extractor
    if name == "qa":
        return qa_extractor
    raise ValueError(f"unknown extraction mode {name!r}; expected 'abstract', 'latex', or 'qa'")


# Theorem-like environments worth turning into problem candidates.
_THEOREM_ENVS = ("theorem", "proposition", "lemma", "corollary", "problem", "conjecture")
_ENV_RE = re.compile(
    r"\\begin\{(" + "|".join(_THEOREM_ENVS) + r")\*?\}(.*?)\\end\{\1\*?\}",
    re.DOTALL,
)

# A TeX line comment: an unescaped ``%`` to end of line (``\%`` is a literal
# percent, kept). Authors routinely leave commented-out copies of theorems in
# their source; those must not leak into statements or survive as "unique"
# duplicates that evade the statement dedup.
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")

# Artifacts stripped from a raw theorem body to get a readable statement.
_LABEL_RE = re.compile(r"\\label\s*\{[^}]*\}")
_CITE_RE = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\s*(?:\[[^\]]*\])?\{[^}]*\}")
_REF_RE = re.compile(r"\\(?:eqref|ref|cref|Cref|autoref|pageref)\s*\{[^}]*\}")
_LEADING_OPT_RE = re.compile(r"^\s*\[[^\]]*\]")  # \begin{theorem}[Attribution]
_EMPTY_BRACES_RE = re.compile(r"\{\s*\}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:])")


def latex_extractor(paper: Paper, *, family: Optional[str] = None, source_fetcher: Optional[Callable] = None) -> list:
    """Deeper extractor: mine theorem statements from a paper's LaTeX source.

    Downloads the e-print tarball, concatenates its ``.tex`` files, and
    turns each theorem/proposition/lemma/corollary/problem environment into
    a candidate (with a ``\\boxed`` answer when one is stated). A paper whose
    source cannot be fetched or parsed is skipped (returns no candidates)
    rather than aborting the whole scrape — the resilience the run relies on.
    """
    fetch = source_fetcher or default_latex_source_fetcher
    try:
        tex = extract_tex(fetch(paper.arxiv_id))
    except Exception:  # noqa: BLE001 — one bad source must not abort the run
        return []
    return extract_theorem_candidates(tex, paper, family=family)


def extract_theorem_candidates(tex: str, paper: Paper, *, family: Optional[str] = None) -> list:
    """Turn theorem-like LaTeX environments into raw candidate rows."""
    # Drop TeX comments up front, so a commented-out theorem collapses to an
    # empty (skipped) statement rather than leaking ``%`` residue or surviving
    # dedup as a near-duplicate of its live twin.
    tex = _COMMENT_RE.sub("", tex)
    candidates: list = []
    for match in _ENV_RE.finditer(tex):
        environment = match.group(1)
        raw_body = match.group(2)
        statement = _clean_tex(raw_body)
        if not statement:
            continue
        metadata = {
            "title": paper.title,
            "primary_category": paper.primary_category,
            "environment": environment,
            "extraction": "latex_theorem",
        }
        # The statement pointed at an equation/section defined elsewhere in the
        # paper — flag it so operators can filter for self-contained problems.
        if _REF_RE.search(raw_body):
            metadata["has_external_refs"] = True
        candidate = {
            "link": paper.link,
            "arxiv_id": paper.arxiv_id,
            "statement": statement,
            "provenance": "extracted",
            "metadata": metadata,
        }
        if family:
            candidate["family"] = family
        answer = _boxed(statement)
        if answer:
            candidate["answer"] = answer
        candidates.append(candidate)
    return candidates


def default_latex_source_fetcher(
    arxiv_id: str, *, timeout: float = 60, retries: int = 4, backoff: float = 3.0
) -> bytes:
    """Fetch a paper's e-print (LaTeX source) tarball, retried. Network code."""
    return _http_get(
        f"https://arxiv.org/e-print/{arxiv_id}", timeout=timeout, retries=retries, backoff=backoff
    ).content


def extract_tex(data: bytes) -> str:
    """Concatenate the ``.tex`` files in an arXiv e-print payload.

    Handles the common shapes: a gzipped tar of sources, a single gzipped
    ``.tex``, or bare text.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            parts = [
                tar.extractfile(m).read().decode("utf-8", "ignore")
                for m in tar.getmembers()
                if m.isfile() and m.name.endswith(".tex") and tar.extractfile(m) is not None
            ]
        if parts:
            return "\n".join(parts)
    except (tarfile.TarError, EOFError):
        pass
    try:
        return gzip.decompress(data).decode("utf-8", "ignore")
    except (OSError, EOFError):
        return data.decode("utf-8", "ignore")


class QAConfigError(ValueError):
    """Raised when the QA generator is misconfigured (no API key / SDK).

    A subclass of ValueError so the CLI maps it to a clean E_INVALID, and
    distinct so ``qa_extractor`` can surface it instead of silently skipping
    every theorem (which would masquerade as an empty arXiv result).
    """


def qa_extractor(
    paper: Paper,
    *,
    family: Optional[str] = None,
    source_fetcher: Optional[Callable] = None,
    generator: Optional[Callable] = None,
) -> list:
    """Deepest extractor: mine theorems, then turn each into a verifiable QA pair.

    One LLM call per theorem:

      ``generator(statement) -> {question, answer, ...}`` — the Sonnet Q+A
      reformulation. Sonnet is the filter: it returns ``None`` for theorems
      that don't yield a single fixed answer, so those are dropped here.

    The generator is injectable so tests substitute a fake without touching
    the SDK. Records stay ``provenance = "extracted"`` — the model is
    instructed to extract, not compute. One theorem the generator can't
    handle is skipped, not fatal.
    """
    generate = generator or default_qa_generator
    theorems = latex_extractor(paper, family=family, source_fetcher=source_fetcher)
    candidates: list = []
    for theorem in theorems:
        source_statement = theorem["statement"]
        try:
            qa = generate(source_statement)
        except QAConfigError:
            raise  # misconfiguration is systemic — surface it, don't skip silently
        except Exception:  # noqa: BLE001 — skip a theorem the generator can't handle
            continue
        if not qa or not qa.get("question") or not qa.get("answer"):
            continue
        tier = classify_answer(qa["answer"])
        if tier is None:
            continue
        # Truth policy follows provenance: a paper-extracted answer defers to
        # the judge (extracted); a generator that COMPUTED the answer is trusted
        # at harvest and discarded by groundtruth — never computed + extracted.
        provenance = qa.get("provenance", "extracted")
        truth_policy = "trusted" if provenance == "computed" else "extracted"
        candidate = {
            "link": paper.link,
            "arxiv_id": paper.arxiv_id,
            "statement": str(qa["question"]),
            "answer": str(qa["answer"]),
            "tier": tier,
            "provenance": provenance,
            "truth_policy": truth_policy,
            "metadata": {
                "title": paper.title,
                "primary_category": paper.primary_category,
                "environment": theorem["metadata"].get("environment"),
                "extraction": "llm_qa",
                "source_statement": source_statement,
            },
        }
        if family:
            candidate["family"] = family
        candidates.append(candidate)
    return candidates


def classify_answer(answer) -> Optional[str]:
    """Classify an answer's form: ``number`` / ``tuple`` / ``expr`` / ``latex``, else ``None``.

    A verifiability gate: accepts an answer that a downstream verifier
    could plausibly compare against a rollout. Research-math answers with
    paper-specific notation (subscripted named functions, custom
    operators, Greek letters) can be legitimate closed forms even when
    sympy cannot parse them — those are tagged ``latex`` and left for the
    pass@k stage's verifier to check symbolically. Pure prose ("the
    smallest prime factor of m") still returns ``None``.
    """
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None

    # Strip outer LaTeX math delimiters and common wrappers so sympy sees
    # bare mathematical content wherever possible.
    stripped = _strip_math_delimiters(text)

    if re.fullmatch(r"[+-]?\d+(\.\d+)?", stripped):
        return "number"
    if re.fullmatch(r"\(.+,.+\)", stripped):
        return "tuple"
    sympify_succeeded = True
    try:
        from sympy import Expr, sympify

        expr = sympify(stripped)
    except Exception:  # noqa: BLE001 — unparseable, defer to LaTeX-marker fallback
        expr = None
        sympify_succeeded = False
    # sympify may return a native bool/list/set/tuple (e.g. "True", "[1,2]",
    # "x==y", "()") — none of which are verifiable closed forms, and none of
    # which carry .is_Symbol. Only a genuine, non-bare SymPy expression counts.
    if sympify_succeeded and isinstance(expr, Expr) and not expr.is_Symbol:
        return "expr"

    # Fallback: accept as ``latex`` only when sympy FAILED to parse AND the
    # answer carries LaTeX math markers, so paper-specific notation
    # (subscripts, Greek, custom operators) can survive extraction. If
    # sympify succeeded with a non-Expr type (set / equality / bare symbol),
    # we stay strict and reject — that's a signal the answer form is a real
    # non-closed-form that the pass@k verifier could not compare against.
    if not sympify_succeeded and _looks_like_math(text):
        return "latex"
    return None


def _strip_math_delimiters(text: str) -> str:
    """Strip surrounding ``$...$`` or ``\\[...\\]`` and common LaTeX wrappers."""
    s = text.strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    elif s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    elif s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    return s


_MATH_MARKER_RE = re.compile(r"[\\_^{}]|\d|[+\-*/=<>]")


def _looks_like_math(text: str) -> bool:
    """Heuristic: does the string carry enough math markers to be a formula?

    Guards against accepting pure prose ("the smallest prime factor of m")
    while accepting research-math LaTeX ("$\\frac{w_1 w_2}{2}$"). At least
    two distinct kinds of math markers must appear, one of which must be a
    LaTeX escape, subscript/superscript, brace, digit, or math operator.
    """
    return bool(_MATH_MARKER_RE.search(text)) and any(c in text for c in "\\_^{}=")


# System prompt for the QA generator. Adapted from ModelBreaker's approach
# (helpers/prompts.py::SYSTEM_PROMPT_GENERATE_QA_FROM_THEOREMS_DATASET),
# which produced icepick's 70-record reference corpus. Encodes the same
# rules — what counts as a "single fixed answer" theorem, what to do with
# if-and-only-if / identity / cardinality forms — in a compressed shape.
_QA_SYSTEM_PROMPT = """You are a problem setter for graduate-level mathematics and theoretical CS. \
Given a theorem, extract a self-contained question and its exact answer, following these rules:

WHAT MAKES A GOOD SOURCE THEOREM:
- Existence-uniqueness: "there exists a unique X such that A" — the answer is X.
- Exact formula results: "the value of expression E is Y" — the answer is Y.
- Necessary-and-sufficient conditions when one side is a numerical value: "X is P if and only if Y = c".
- Unique extrema: "the maximum of f is M attained at x=x0" — the answer is M.
- Exact complexity results: "the running time is exactly Theta(n^2)" — accepted; O(...), Omega(...), and inequalities are NOT accepted.
- Explicit solution counts: "the equation has a unique solution / no solutions" — accepted.
- Identities X = Y where both sides are complex: reformulate as "What is X - Y?" with answer 0 (or the constant remainder).

WHAT TO REJECT (return is_good_theorem=false, empty question/answer):
- Any inequality, bound, or approximation as the main result.
- If-and-only-if where NEITHER side is a numerical value.
- Belonging to a complexity class ("X is in NP").
- Isomorphism/homomorphism results.
- Existence without uniqueness.
- Anything ambiguous or that requires guessing.

QUESTION RULES:
- Redefine every quantity needed so a reader can answer without the theorem in front of them.
- Never begin with "Prove that".
- Never reveal the answer in the question.
- No yes/no questions.
- For if-and-only-if with a numerical side, ask about the numerical side ("If X is P, what is Y?").

ANSWER RULES:
- One value: number, closed-form expression, or formula.
- Extracted directly from the theorem — do not derive or compute anything new.
- Rendered in standard LaTeX ($...$ inline or \\[...\\] block).

Respond in strict JSON only:
{"question": "...", "answer": "...", "is_good_theorem": true}
or
{"question": "", "answer": "", "is_good_theorem": false}
"""


_ANTHROPIC_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _cached_system_prompt(text: str) -> list:
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _usage_dict(message) -> dict:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    return {
        field: int(getattr(usage, field, 0) or 0)
        for field in _ANTHROPIC_USAGE_FIELDS
    }


def default_qa_generator(
    statement: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    usage_callback: Optional[Callable] = None,
    model_callback: Optional[Callable] = None,
) -> Optional[dict]:
    """Extract a question + paper-stated answer from a theorem via Anthropic.

    Uses the ModelBreaker-derived system prompt (:data:`_QA_SYSTEM_PROMPT`) —
    the prompt is authoritative about which theorem shapes accept a
    closed-form Q+A. The system/user split (rules in system, theorem in user
    message) matches how MB ran o3-mini via ``AnthropicOpenAIShim`` and is
    materially more permissive than the prior single-message prompt, which
    on Haiku rejected every raw arXiv theorem in the math.AP and math.NT
    pilots.

    Requires ``ANTHROPIC_API_KEY`` in the environment (or ``ANTHROPIC_KEY_FILE``
    proxy) and the ``anthropic`` SDK. Returns ``{"question", "answer"}`` when
    the model returns ``is_good_theorem=true`` with populated fields, else
    ``None``. Records stay ``provenance=extracted`` — the model is instructed
    to extract, not compute.
    """
    import json as _json

    from icepick.config import ConfigError, resolve_anthropic_credentials

    # Key comes from ANTHROPIC_API_KEY or the ANTHROPIC_KEY_FILE proxy — never
    # embedded here. A config problem is systemic, so surface it as QAConfigError
    # (which qa_extractor re-raises) instead of silently skipping every theorem.
    try:
        api_key, file_model = resolve_anthropic_credentials()
    except ConfigError as exc:
        raise QAConfigError(str(exc)) from exc
    # Default upgraded from Haiku 4.5 → Sonnet 4.6: Haiku's parsimony rejected
    # every raw arXiv theorem in the pilot runs, matching MB's finding that a
    # larger reasoning model (o3-mini) was needed for productive extraction.
    # Operators can override via ANTHROPIC_MODEL in the key file or pass
    # ``model=`` explicitly.
    model = model or file_model or "claude-sonnet-4-6"
    # Surface the model actually used so the run report labels QA honestly
    # (the key-file ANTHROPIC_MODEL override wins over the Sonnet default).
    if model_callback is not None:
        model_callback(model)

    try:
        import anthropic  # lazy: only qa-mode production scraping needs the SDK
    except ImportError as exc:
        raise QAConfigError("qa extraction needs the 'anthropic' SDK (pip install -e .[judge])") from exc

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_cached_system_prompt(_QA_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": f"Theorem:\n{statement}"}],
    )
    if usage_callback is not None:
        usage_callback(_usage_dict(message))
    text = "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        return None
    # Some models wrap JSON in fenced code blocks; strip them if present.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        data = _json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    # Explicit is_good_theorem=false or empty question/answer → drop.
    if data.get("is_good_theorem") is False:
        return None
    question = data.get("question") or ""
    answer = data.get("answer") or ""
    if not question or not answer:
        return None
    return {"question": str(question), "answer": str(answer)}


# --- internals ----------------------------------------------------------------


def _arxiv_id_from_url(url: str) -> str:
    if not url:
        return ""
    match = _ABS_RE.search(url)
    ident = (match.group(1) if match else url.rsplit("/", 1)[-1]).strip("/")
    return _VERSION_RE.sub("", ident)


def _text(element) -> str:
    return (element.text or "").strip() if element is not None else ""


def _lone_family(families: Optional[list]) -> Optional[str]:
    """A single requested family stamps every candidate; ambiguity defers to normalise."""
    families = list(families or [])
    return families[0] if len(families) == 1 else None


def _clean_tex(text: str) -> str:
    """Turn a raw theorem body into a readable statement.

    Strips ``\\label``, citations (``\\cite`` family), cross-references
    (``\\eqref`` / ``\\ref`` / ``\\cref`` …), a leading optional
    ``[attribution]`` argument, non-breaking spaces, and any empty braces
    left behind, then collapses whitespace. Math (``$…$``) is left intact.
    """
    text = _LABEL_RE.sub("", text)
    text = _CITE_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _EMPTY_BRACES_RE.sub("", text)
    text = _LEADING_OPT_RE.sub("", text)
    text = text.replace("~", " ")
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return " ".join(text.split()).strip()


def _boxed(text: str) -> Optional[str]:
    """Extract the last ``\\boxed{...}`` payload, brace-balanced, if present."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    brace = text.find("{", idx)
    if brace == -1:
        return None
    depth = 0
    out: list = []
    for char in text[brace:]:
        if char == "{":
            depth += 1
            if depth == 1:
                continue
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(char)
    return "".join(out).strip() or None

"""arXiv OAI-PMH category index — id → (categories, title) for bulk planning.

The S3 src chunks carry LaTeX sources but no category metadata, so the bulk
adapter needs a local index answering "which new-style ids in yymm X carry
category Y?" (``ids_for``) and "what are this id's categories/title?"
(``lookup``). This module builds that index from arXiv's OAI-PMH endpoint
via ``ListRecords`` with ``metadataPrefix="arXiv"`` (the prefix that carries
``<categories>`` and ``<title>``).

Contract highlights (INTERFACES.md §2 — frozen):

- **Serial requests only, ever.** ``build`` is a single loop: one request in
  flight at a time, next page only after the previous one is parsed and
  cached. No concurrency of any kind.
- **resumptionToken paging** until exhausted — an absent or empty token ends
  the walk. OAI-PMH datestamps are *modification* dates, so there is no
  server-side submission-date filtering; yymm filtering happens client-side
  on ids in ``ids_for``.
- **503 handling.** A 503 with ``Retry-After`` sleeps *exactly* that many
  seconds (via the injectable ``sleeper``). A 503 without it sleeps the
  bounded default schedule ``DEFAULT_BACKOFF_SCHEDULE`` (5 → 10 → 20 → 40 s,
  capped at the last entry). After ``MAX_ATTEMPTS_PER_PAGE`` requests for
  the same page (initial + retries) the build gives up with ``OAIError``.
  Every issued request — retries included — counts toward ``oai_requests``.
  Backoff telemetry is journaled on the instance (``rate_limit_events``,
  ``rate_limit_backoff_seconds``, ``rate_limit_statuses``) in realmath's
  shapes so the adapter can lift them into ``acquisition``.
- **Page cache = resumability.** Each fetched page is parsed, then persisted
  to ``cache_dir`` BEFORE the next request is issued. Resumption tokens
  expire daily; the page cache is what survives a kill. Rebuilding over a
  warm cache replays cached pages with ZERO new requests and resumes
  fetching (if needed) from the first uncached page, using the token stored
  in the last cached one. Parse-before-persist means the cache only ever
  holds well-formed pages; malformed responses raise without being cached.
- **No abstract stored** — deliberate index-size decision; the adapter
  constructs ``Paper`` with ``abstract=""`` (§4).

Failure modes raise ``OAIError``: malformed XML, an OAI ``<error>`` element,
a record missing required fields, a non-200/non-503 status, or 503 retry
exhaustion. Non-503 statuses are not retried — only throttling is transient
by contract; anything else is a real problem the operator should see.

Offline discipline: this module never opens a socket itself. All HTTP goes
through the injected fetcher; all sleeping through the injected sleeper.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlencode

# Default backoff for 503s that carry no Retry-After header: bounded,
# doubling, capped at the final entry. With MAX_ATTEMPTS_PER_PAGE = 5 a
# single page absorbs at most 5+10+20+40 = 75 s of default backoff before
# the build gives up.
DEFAULT_BACKOFF_SCHEDULE: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)

# Maximum HTTP requests for one page (the initial attempt plus retries).
MAX_ATTEMPTS_PER_PAGE = 5

_OAI_NS = "http://www.openarchives.org/OAI/2.0/"
_ARXIV_NS = "http://arxiv.org/OAI/arXiv/"
_NS = {"oai": _OAI_NS, "arxiv": _ARXIV_NS}


class OAIError(RuntimeError):
    """Unrecoverable OAI-PMH failure: malformed page, protocol error,
    unexpected status, or 503 retry exhaustion."""


@dataclass(frozen=True)
class OAIResponse:
    """The injectable-fetcher return, minimal by design (§2)."""

    status: int                    # 200 | 503 | ...
    retry_after: Optional[float]   # parsed Retry-After seconds, None if absent
    text: str                      # XML body ("" on non-200)


@dataclass(frozen=True)
class PaperMeta:
    arxiv_id: str
    primary_category: str          # first entry of <categories>
    categories: tuple[str, ...]
    title: str                     # single-line, whitespace-collapsed
    # NO abstract — deliberate (index size); see INTERFACES.md §2/§4.


def _collapse(text: str) -> str:
    """Whitespace-collapse to a single line."""
    return " ".join(text.split())


def _safe_name(part: str) -> str:
    """A filesystem-safe cache-filename component."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", part)


@dataclass(frozen=True)
class _ParsedPage:
    papers: tuple[PaperMeta, ...]
    resumption_token: str          # "" when paging is exhausted


class CategoryIndex:
    """Category/title index over an OAI-PMH set, with a resumable page cache.

    ``build`` may be called more than once on the same instance (e.g. one
    set after another); results accumulate into the same in-memory index and
    all counters are lifetime totals for the instance. Cache files are keyed
    by (set, metadataPrefix, page number), so distinct sets sharing one
    ``cache_dir`` never collide.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self._papers: dict[str, PaperMeta] = {}
        #: EVERY issued HTTP request, retries included (§2, frozen).
        self.oai_requests: int = 0
        # Journaled backoff telemetry, realmath-shaped (str(status) keys).
        self.rate_limit_events: int = 0
        self.rate_limit_backoff_seconds: float = 0.0
        self.rate_limit_statuses: dict[str, int] = {}

    # -- build -----------------------------------------------------------

    def build(
        self,
        *,
        oai_set: str,
        fetcher: Callable[[str], OAIResponse],
        base_url: str = "https://oaipmh.arxiv.org/oai",
        metadata_prefix: str = "arXiv",
        sleeper: Callable[[float], None] = time.sleep,
        from_date: Optional[str] = None,
    ) -> None:
        """Walk ``ListRecords`` for ``oai_set``, serially, page by page.

        Cached pages (from a previous, possibly killed, build) are replayed
        from ``cache_dir`` without issuing any request; fetching resumes at
        the first uncached page via the token stored in the last cached one.

        ``from_date`` ("YYYY-MM-DD"), when set, adds an OAI ``from`` datestamp
        bound to the INITIAL ``ListRecords`` request only — resumptionToken
        requests must NOT carry it (nor set/metadataPrefix): OAI bakes those
        into the token, and re-sending them is a protocol error. This is a
        SUPERSET bound, not a filter: OAI datestamps are modification dates
        and modification >= submission always, so every in-window submission
        is still returned; the precise selector stays the client-side yymm
        prefix filter in ``ids_for``. Its purpose is to cap the walk at the
        window era instead of the whole set's history (W3 H2).
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        page_num = 1
        # "" means "issue the initial request"; a non-empty value is the
        # resumptionToken for the next page. Per OAI-PMH, a token request
        # carries ONLY verb + resumptionToken (set/metadataPrefix are baked
        # into the token server-side).
        token = ""
        while True:
            cache_path = self._cache_path(oai_set, metadata_prefix, page_num)
            if cache_path.exists():
                page = self._parse_page(cache_path.read_text(encoding="utf-8"),
                                        context=f"cached page {page_num} ({cache_path.name})")
            else:
                if token:
                    # Token requests carry ONLY verb + resumptionToken; adding
                    # set/metadataPrefix/from is an OAI protocol error.
                    params = [("verb", "ListRecords"), ("resumptionToken", token)]
                else:
                    params = [("verb", "ListRecords"), ("set", oai_set),
                              ("metadataPrefix", metadata_prefix)]
                    if from_date is not None:
                        params.append(("from", from_date))
                url = f"{base_url}?{urlencode(params)}"
                text = self._fetch_with_retry(url, fetcher, sleeper, page_num)
                page = self._parse_page(text, context=f"page {page_num} ({url})")
                # Persist BEFORE the next request: this page never needs
                # refetching, even if the build dies right after this line.
                self._persist(cache_path, text)
            for paper in page.papers:
                self._papers[paper.arxiv_id] = paper
            token = page.resumption_token
            if not token:
                return
            page_num += 1

    def _fetch_with_retry(
        self,
        url: str,
        fetcher: Callable[[str], OAIResponse],
        sleeper: Callable[[float], None],
        page_num: int,
    ) -> str:
        """Serial fetch of one page, honoring 503 backoff. Returns body text."""
        failures = 0
        while True:
            self.oai_requests += 1
            response = fetcher(url)
            if response.status == 200:
                return response.text
            if response.status != 503:
                raise OAIError(
                    f"OAI request for page {page_num} returned HTTP "
                    f"{response.status} (only 503 is retried): {url}"
                )
            failures += 1
            if failures >= MAX_ATTEMPTS_PER_PAGE:
                raise OAIError(
                    f"giving up on page {page_num} after "
                    f"{MAX_ATTEMPTS_PER_PAGE} attempts, all HTTP 503: {url}"
                )
            if response.retry_after is not None:
                delay = float(response.retry_after)   # honored exactly
            else:
                schedule_idx = min(failures - 1, len(DEFAULT_BACKOFF_SCHEDULE) - 1)
                delay = DEFAULT_BACKOFF_SCHEDULE[schedule_idx]
            self.rate_limit_events += 1
            self.rate_limit_backoff_seconds += delay
            key = str(response.status)
            self.rate_limit_statuses[key] = self.rate_limit_statuses.get(key, 0) + 1
            sleeper(delay)

    def _cache_path(self, oai_set: str, metadata_prefix: str, page_num: int) -> Path:
        name = (f"{_safe_name(oai_set)}_{_safe_name(metadata_prefix)}"
                f"_page{page_num:05d}.xml")
        return self.cache_dir / name

    @staticmethod
    def _persist(cache_path: Path, text: str) -> None:
        """Atomic-ish write: a kill mid-write never leaves a torn page."""
        part = cache_path.with_name(cache_path.name + ".part")
        part.write_text(text, encoding="utf-8")
        part.replace(cache_path)

    # -- parsing ---------------------------------------------------------

    @staticmethod
    def _parse_page(text: str, *, context: str) -> _ParsedPage:
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise OAIError(f"malformed OAI XML on {context}: {exc}") from exc
        error = root.find("oai:error", _NS)
        if error is not None:
            code = error.get("code", "unknown")
            raise OAIError(
                f"OAI error response on {context}: "
                f"code={code}: {_collapse(error.text or '')}"
            )
        list_records = root.find("oai:ListRecords", _NS)
        if list_records is None:
            raise OAIError(f"no <ListRecords> element on {context}")
        papers = []
        for record in list_records.findall("oai:record", _NS):
            meta = record.find("oai:metadata/arxiv:arXiv", _NS)
            if meta is None:
                header = record.find("oai:header", _NS)
                if header is not None and header.get("status") == "deleted":
                    continue  # deleted records legitimately carry no metadata
                raise OAIError(f"record without arXiv metadata on {context}")
            papers.append(CategoryIndex._parse_record(meta, context))
        token_el = list_records.find("oai:resumptionToken", _NS)
        token = (token_el.text or "").strip() if token_el is not None else ""
        return _ParsedPage(papers=tuple(papers), resumption_token=token)

    @staticmethod
    def _parse_record(meta: ET.Element, context: str) -> PaperMeta:
        def _field(tag: str) -> str:
            el = meta.find(f"arxiv:{tag}", _NS)
            if el is None or not (el.text or "").strip():
                raise OAIError(f"record missing <{tag}> on {context}")
            return el.text.strip()

        arxiv_id = _field("id")
        categories = tuple(_field("categories").split())
        return PaperMeta(
            arxiv_id=arxiv_id,
            primary_category=categories[0],
            categories=categories,
            title=_collapse(_field("title")),
        )

    # -- queries ---------------------------------------------------------

    def lookup(self, arxiv_id: str) -> Optional[PaperMeta]:
        """Exact-id lookup. Index ids are versionless (OAI ``<id>``)."""
        return self._papers.get(arxiv_id)

    def ids_for(self, *, category: str, yymm: str,
                primary_only: bool) -> list[str]:
        """New-style ids in ``yymm`` carrying ``category``, sorted.

        ``yymm`` is a plain id-prefix match (new-style ids are
        ``"YYMM.NNNNN"``, so ``"2501"`` selects exactly January 2025).
        ``primary_only=True`` requires the category to be the primary
        (first-listed); ``False`` matches any listed category.
        """
        out = []
        for arxiv_id, meta in self._papers.items():
            if not arxiv_id.startswith(yymm):
                continue
            if primary_only:
                if meta.primary_category == category:
                    out.append(arxiv_id)
            elif category in meta.categories:
                out.append(arxiv_id)
        return sorted(out)

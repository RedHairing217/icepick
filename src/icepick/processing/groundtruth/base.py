"""Groundtruth contracts — canonical enum, verdict dataclass, arxiv helpers.

The verdict surface is intentionally small. Every record gets exactly one
``GroundtruthVerdict`` whose ``verdict_status`` is one of:

- ``published``     - paper is peer-reviewed AND indexed in a reputable venue
- ``unpublished``   - explicit evidence the paper is preprint-only
- ``defer``         - judges couldn't reach a majority verdict
- ``error``         - adapter-level failure (API down, parse error, etc.)
- ``discarded``     - record dropped before the check ran (generated
                      provenance, no arxiv_id, etc.); ``discarded_reason``
                      explains which

Only ``published`` records flow downstream. The other four statuses are
written to a side file so the operator can audit what was filtered out
and why.

``raw_payload`` preserves the original Anthropic response (web_search
results + reasoning) verbatim so verdict renaming is non-destructive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

STATUS_PUBLISHED = "published"
STATUS_UNPUBLISHED = "unpublished"
STATUS_DEFER = "defer"
STATUS_ERROR = "error"
STATUS_DISCARDED = "discarded"

CANONICAL_STATUSES = (
    STATUS_PUBLISHED,
    STATUS_UNPUBLISHED,
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_DISCARDED,
)

DISCARD_REASON_GENERATED = "generated_provenance"
DISCARD_REASON_NO_ARXIV_ID = "no_arxiv_id"


@dataclass
class GroundtruthVerdict:
    """One publication-status verdict for one record.

    ``arxiv_id`` is the join key for caching: many records can share one
    paper, and one cache hit serves all of them. ``judge_votes`` records
    the per-sample raw verdicts so an operator can see whether the
    majority was 3-0 or a contentious 2-1.
    """

    uid: str
    source: str
    verdict_status: str
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None
    publication_year: Optional[int] = None
    indexed_in: list = field(default_factory=list)
    evidence_urls: list = field(default_factory=list)
    judge_model: str = ""
    judge_votes: list = field(default_factory=list)
    judge_majority: Optional[str] = None
    discarded_reason: Optional[str] = None
    error_reason: Optional[str] = None
    reasoning: str = ""
    confidence: Optional[str] = None
    raw_payload: dict = field(default_factory=dict)

    def to_jsonl_row(self) -> dict:
        return {
            "uid": self.uid,
            "source": self.source,
            "verdict_status": self.verdict_status,
            "arxiv_id": self.arxiv_id,
            "venue": self.venue,
            "publication_year": self.publication_year,
            "indexed_in": list(self.indexed_in),
            "evidence_urls": list(self.evidence_urls),
            "judge_model": self.judge_model,
            "judge_votes": list(self.judge_votes),
            "judge_majority": self.judge_majority,
            "discarded_reason": self.discarded_reason,
            "error_reason": self.error_reason,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "raw_payload": self.raw_payload,
        }


_ARXIV_ID_FIELDS = ("arxiv_id", "arxivId", "arxiv", "paper_id", "paperId")
_ARXIV_URL_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z\-\.]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
_ARXIV_BARE_ID_PATTERN = re.compile(
    r"^(?:arxiv:)?(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z\-\.]+/\d{7}(?:v\d+)?)$",
    re.IGNORECASE,
)


def extract_arxiv_id(record: dict) -> Optional[str]:
    """Find the arXiv ID on a record, normalised (no `arxiv:` prefix, no version).

    Looks at canonical fields first, then attempts to extract from any
    URL field. Returns ``None`` if nothing usable is present — the
    runner discards such records with ``DISCARD_REASON_NO_ARXIV_ID``.
    """
    for key in _ARXIV_ID_FIELDS:
        value = record.get(key)
        if value:
            normalised = _normalise_arxiv_id(str(value))
            if normalised:
                return normalised
    for key in ("arxiv_url", "paper_url", "url"):
        value = record.get(key)
        if value:
            match = _ARXIV_URL_PATTERN.search(str(value))
            if match:
                return _normalise_arxiv_id(match.group("id"))
    return None


def _normalise_arxiv_id(value: str) -> Optional[str]:
    value = value.strip()
    if value.lower().startswith("arxiv:"):
        value = value[len("arxiv:"):]
    if not value:
        return None
    match = _ARXIV_BARE_ID_PATTERN.match(value)
    if not match:
        return None
    canonical = match.group("id")
    # Strip the version suffix so vN edits don't fragment the cache.
    return re.sub(r"v\d+$", "", canonical, flags=re.IGNORECASE)

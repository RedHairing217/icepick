"""Publication-status check (formerly c02 ground-truth).

This stage decides whether the source arXiv paper a problem came from
has been peer-reviewed and indexed in a reputable bibliographic database
(Scopus, Web of Science, DBLP, MathSciNet, or equivalent). A published
paper is a source-quality proxy for trusting the extracted ground truth;
predatory venues and unindexed preprints do not count.

Design rules:

- icepick does not process generated records. Records with
  ``provenance = "computed"`` are dropped at this stage with an explicit
  ``discarded_reason``.
- The check is positionable. Run it BEFORE pass@k to discard records
  before paying sampling cost, or AFTER pass@k to filter the survivors
  before the gate. The module is agnostic to which JSONL you feed it.
- One Anthropic call per unique arXiv paper, cached by ``arxiv_id`` so
  many problems sharing one paper cost one lookup.
- Three independent web_search judgments per paper; uphold a verdict on
  a two-of-three majority. A judge-only "uncertain" routes to ``defer``,
  never to ``unpublished``.
- ``flow_testing`` mode replays from a calibration sheet; no real
  Anthropic calls.
- Full automation. The only human decision is which knobs to set
  (judge model, samples, uphold, position in pipeline).
"""

from icepick.processing.groundtruth.base import (
    CANONICAL_STATUSES,
    STATUS_DEFER,
    STATUS_DISCARDED,
    STATUS_ERROR,
    STATUS_PUBLISHED,
    STATUS_UNPUBLISHED,
    DISCARD_REASON_GENERATED,
    DISCARD_REASON_NO_ARXIV_ID,
    GroundtruthVerdict,
    extract_arxiv_id,
)
from icepick.processing.groundtruth.config import GroundtruthConfig

__all__ = [
    "CANONICAL_STATUSES",
    "DISCARD_REASON_GENERATED",
    "DISCARD_REASON_NO_ARXIV_ID",
    "GroundtruthConfig",
    "GroundtruthVerdict",
    "STATUS_DEFER",
    "STATUS_DISCARDED",
    "STATUS_ERROR",
    "STATUS_PUBLISHED",
    "STATUS_UNPUBLISHED",
    "extract_arxiv_id",
]

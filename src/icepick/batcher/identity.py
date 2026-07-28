"""Pure identity functions for the bulk-batcher subsystem.

All three functions are deliberately standalone (no intra-package imports) so
the batcher can be imported independently of the rest of the pipeline.

Cross-references:
- ``compute_uid`` mirrors ``src/icepick/processing/poser/base.py:128-139``
  (function ``compute_uid`` / recipe used by ``inject_uid``).  The algorithm
  must stay identical so a uid pre-injected by the slicer survives the funnel
  unchanged.
- ``stmt_key`` mirrors the dedup-key recipe in
  ``src/icepick/allocation/adapters/realmath_scrape.py:345``
  (inside ``normalise``): ``" ".join(statement.lower().split())``, then
  SHA-256 hex of that normalised string.
- ``content_hash`` uses ``json.dumps(row, sort_keys=True)`` — deterministic
  across Python versions for the field types present in icepick journals
  (strings, numbers, dicts, lists).  ``ensure_ascii`` is left at its default
  (``True``) to match plain ``json.dumps`` without kwargs, keeping the hash
  stable if the caller passes the row straight from ``json.loads``.
"""

from __future__ import annotations

import hashlib
import json


def compute_uid(source: str, statement: str) -> str:
    """Stable 32-hex uid for a (source, statement) pair.

    Mirrors ``processing/poser/base.py:128-139`` exactly:
    SHA-256 of ``source + '\\x1f' + statement`` encoded as UTF-8,
    truncated to 32 hex characters.  The separator byte (0x1F, ASCII
    unit-separator) prevents ambiguity when source or statement contains
    the delimiter.
    """
    digest = hashlib.sha256(
        f"{source}\x1f{statement}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def stmt_key(statement: str) -> str:
    """Normalised-statement fingerprint for cross-source dedup.

    Normalisation mirrors ``realmath_scrape.normalise()`` line 345:
    case-fold then collapse all whitespace runs to a single space, trimming
    leading/trailing whitespace.  The resulting normalised string is then
    SHA-256 hex-digested (full 64 chars).  This is source-independent so
    the same theorem arriving under a different source (e.g. arxiv_bulk
    re-covering batch 1-8 territory) is caught by this key even when the
    uid differs.
    """
    normalised = " ".join(statement.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def content_hash(row: dict) -> str:
    """SHA-256 hex of the canonical JSON serialisation of a journal row.

    ``json.dumps`` with ``sort_keys=True`` makes the hash independent of
    insertion order.  ``ensure_ascii`` is left at its default (``True``) so
    non-ASCII characters are escaped the same way across all callers; this
    keeps the hash deterministic when rows round-trip through ``json.loads``.
    """
    serialised = json.dumps(row, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()

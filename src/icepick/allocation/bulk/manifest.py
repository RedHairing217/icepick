"""arXiv src manifest: parsing, windowing, chunk selection, and cost rollup.

The src manifest (``arXiv_src_manifest.xml``) is a locally-held file fetched
once by an operator at gate W4.  All functions here are pure and offline; no
network access occurs at any point.

Wrapper element assumption:  the real manifest uses ``<arXivSRC>`` as the
document root, with ``<file>`` children (one per tar chunk).  This assumption
is documented in ``tests/fixtures/arxiv_bulk/src_manifest_sample.xml`` and
must be verified against the live manifest at gate W4.

Old-style id handling:  ``chunks_for_ids`` is scoped to new-style ids of the
form ``YYMM.NNNNN`` (e.g. ``2501.00123``).  Old-style ids (e.g.
``math/0501123``) are out of scope for v1.  Any wanted id that does not parse
as a new-style id is skipped and reported via ``warnings.warn(..., UserWarning)``;
the return type stays ``list[ManifestEntry]``.  Emitting through the ``warnings``
module (rather than a return value or logger) lets callers filter or capture the
notices in tests without monkeypatching.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

EGRESS_USD_PER_GB = 0.09  # AWS us-east-1 → internet; decimal GB (1e9 bytes)

# New-style arxiv id: "YYMM.NNNNN[N]" — up to 6 digits after the dot.
# The yymm prefix must match a chunk's yymm field for a fast pre-filter
# before the numeric-suffix range comparison.
_NEWSTYLE_SEP = "."


class ManifestError(ValueError):
    """Raised by ``parse_manifest`` when XML is malformed or a field is absent.

    Covers both XML parse failures and missing/empty required child elements
    within a ``<file>`` block.  The message names the offending field or
    describes the parse error.
    """


@dataclass(frozen=True)
class ManifestEntry:
    """One ``<file>`` element from ``arXiv_src_manifest.xml``.

    All ten fields are required; ``parse_manifest`` raises ``ManifestError``
    if any are absent or empty in the source XML.

    ``yymm`` is kept as a string because leading-zero months exist
    (``"0001"`` = January 2000); integer coercion would silently drop the
    leading zero on older chunks.

    ``timestamp`` is stored verbatim as ``"YYYY-MM-DD HH:MM:SS"``; it is
    never parsed to a datetime object here — that is the consumer's concern.
    """

    filename: str          # S3 key, e.g. "src/arXiv_src_2501_001.tar"
    yymm: str              # "2501"; string to preserve leading zeros
    seq_num: int
    first_item: str        # arXiv id of first paper in chunk
    last_item: str         # arXiv id of last paper in chunk
    num_items: int
    size_bytes: int        # value of <size> element
    md5sum: str            # tar-level checksum, hex
    content_md5sum: str
    timestamp: str         # "YYYY-MM-DD HH:MM:SS" verbatim


@dataclass(frozen=True)
class ChunkRollup:
    """Aggregate statistics for a list of ``ManifestEntry`` objects.

    ``egress_usd`` uses decimal gigabytes (1 GB = 1 000 000 000 bytes) and
    the ``EGRESS_USD_PER_GB`` constant, rounded to six decimal places to
    avoid misleading precision while still supporting accurate summation
    across many chunks.
    """

    chunk_count: int
    total_bytes: int
    egress_usd: float  # total_bytes / 1e9 × EGRESS_USD_PER_GB, decimal GB


# ---------------------------------------------------------------------------
# Required XML fields for a <file> element, in manifest order.
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS = (
    "filename",
    "yymm",
    "seq_num",
    "first_item",
    "last_item",
    "num_items",
    "size",
    "md5sum",
    "content_md5sum",
    "timestamp",
)


def _require(element: ET.Element, tag: str) -> str:
    """Return the text of *tag* child of *element*, or raise ``ManifestError``.

    An element that is present but has empty text (or only whitespace) is
    treated as missing: every field is meaningful and a blank value indicates
    a corrupt manifest.
    """
    child = element.find(tag)
    if child is None or not (child.text or "").strip():
        raise ManifestError(f"<file> element missing required field: <{tag}>")
    return child.text.strip()


def parse_manifest(xml_text: str) -> list[ManifestEntry]:
    """Parse ``arXiv_src_manifest.xml`` text into a list of ``ManifestEntry``.

    The function is pure: it takes the raw XML string and returns a list
    (possibly empty, for a manifest with no ``<file>`` children).

    Raises ``ManifestError`` on any of:
    - XML that is not well-formed (wraps ``xml.etree.ElementTree.ParseError``).
    - A ``<file>`` element with any of the ten required fields absent or blank.
    - A ``<file>`` element whose integer fields (``seq_num``, ``num_items``,
      ``size``) cannot be coerced to ``int``.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ManifestError(f"malformed manifest XML: {exc}") from exc

    entries: list[ManifestEntry] = []
    for file_el in root.iter("file"):
        filename = _require(file_el, "filename")
        yymm = _require(file_el, "yymm")
        seq_num_raw = _require(file_el, "seq_num")
        first_item = _require(file_el, "first_item")
        last_item = _require(file_el, "last_item")
        num_items_raw = _require(file_el, "num_items")
        size_raw = _require(file_el, "size")
        md5sum = _require(file_el, "md5sum")
        content_md5sum = _require(file_el, "content_md5sum")
        timestamp = _require(file_el, "timestamp")

        try:
            seq_num = int(seq_num_raw)
        except ValueError as exc:
            raise ManifestError(
                f"<seq_num> is not an integer in chunk {filename!r}: {seq_num_raw!r}"
            ) from exc

        try:
            num_items = int(num_items_raw)
        except ValueError as exc:
            raise ManifestError(
                f"<num_items> is not an integer in chunk {filename!r}: {num_items_raw!r}"
            ) from exc

        try:
            size_bytes = int(size_raw)
        except ValueError as exc:
            raise ManifestError(
                f"<size> is not an integer in chunk {filename!r}: {size_raw!r}"
            ) from exc

        entries.append(ManifestEntry(
            filename=filename,
            yymm=yymm,
            seq_num=seq_num,
            first_item=first_item,
            last_item=last_item,
            num_items=num_items,
            size_bytes=size_bytes,
            md5sum=md5sum,
            content_md5sum=content_md5sum,
            timestamp=timestamp,
        ))

    return entries


def select_chunks(
    entries: list[ManifestEntry],
    *,
    year: int,
    month: Optional[int] = None,
) -> list[ManifestEntry]:
    """Return entries whose ``yymm`` falls within the requested window.

    ``year`` is the four-digit calendar year (e.g. 2025).  ``month``, when
    given, is 1–12; omitting it selects all twelve months of that year.

    The ``yymm`` field encodes year as the two-digit suffix of the calendar
    year (e.g. ``"25"`` for 2025) followed by a zero-padded two-digit month.
    Year 2000 papers have ``yymm`` starting with ``"00"``.

    An empty result is valid: ``select_chunks`` never raises for a window
    that matches no entries.
    """
    yy = str(year % 100).zfill(2)
    if month is None:
        prefix = yy
        return [e for e in entries if e.yymm.startswith(prefix)]
    mm = str(month).zfill(2)
    target = yy + mm
    return [e for e in entries if e.yymm == target]


def chunks_for_ids(
    entries: list[ManifestEntry],
    wanted_ids: set[str],
) -> list[ManifestEntry]:
    """Return the minimal set of chunks whose ranges cover *wanted_ids*.

    Coverage is determined by the ``[first_item, last_item]`` inclusive range
    stored in each ``ManifestEntry``.  Comparison uses the new-style arXiv id
    scheme only: ``YYMM.NNNNN[N]``.  Both the numeric yymm prefix and the
    integer value of the suffix are compared, so ``2501.00002`` sorts strictly
    between ``2501.00001`` and ``2501.00003`` regardless of zero-padding.

    Old-style ids (e.g. ``math/0501123``) are **out of scope for v1**.  Any
    id in *wanted_ids* that does not conform to the new-style pattern is
    skipped with a ``UserWarning`` issued via ``warnings.warn``; the
    return value contains only the chunks needed for the well-formed ids.

    An id that has new-style syntax but falls outside every chunk's range is
    silently skipped (the chunk simply does not exist in the manifest window).

    The returned list preserves manifest order and contains no duplicates.
    """
    needed: list[ManifestEntry] = []
    seen: set[str] = set()

    for arxiv_id in wanted_ids:
        if not _is_newstyle(arxiv_id):
            warnings.warn(
                f"chunks_for_ids: old-style id skipped (out of scope v1): {arxiv_id!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        for entry in entries:
            if entry.filename in seen:
                continue
            if id_in_range(arxiv_id, entry.first_item, entry.last_item):
                needed.append(entry)
                seen.add(entry.filename)

    # Preserve manifest order.
    entry_order = {e.filename: i for i, e in enumerate(entries)}
    needed.sort(key=lambda e: entry_order[e.filename])
    return needed


def rollup(entries: list[ManifestEntry]) -> ChunkRollup:
    """Aggregate byte counts and egress cost for *entries*.

    ``egress_usd`` is computed using decimal gigabytes (1 GB = 1 000 000 000
    bytes) and ``EGRESS_USD_PER_GB``, rounded to six decimal places.  An
    empty list produces a zero-valued ``ChunkRollup``.
    """
    chunk_count = len(entries)
    total_bytes = sum(e.size_bytes for e in entries)
    egress_usd = round(total_bytes / 1e9 * EGRESS_USD_PER_GB, 6)
    return ChunkRollup(
        chunk_count=chunk_count,
        total_bytes=total_bytes,
        egress_usd=egress_usd,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_newstyle(arxiv_id: str) -> bool:
    """Return True iff *arxiv_id* looks like a new-style id (``YYMM.NNNNN[N]``).

    Accepts 5 or 6 digits after the dot (arXiv switched to 5-digit suffixes
    in April 2007 and moved to 6-digit suffixes from 2015 onward).  The yymm
    portion must be exactly 4 digits.
    """
    if _NEWSTYLE_SEP not in arxiv_id:
        return False
    yymm_part, _, suffix_part = arxiv_id.partition(_NEWSTYLE_SEP)
    return (
        len(yymm_part) == 4
        and yymm_part.isdigit()
        and len(suffix_part) in (5, 6)
        and suffix_part.isdigit()
    )


def newstyle_key(arxiv_id: str) -> tuple[str, int]:
    """Return ``(yymm, numeric_suffix)`` for sorting / range comparison.

    Comparing this tuple orders ids correctly across month boundaries: the
    string ``yymm`` compares first (so ``2504`` precedes ``2505``), and the
    integer suffix breaks ties within a month regardless of zero-padding.

    Callers must ensure *arxiv_id* is new-style (passes ``_is_newstyle``)
    before calling this helper.

    Public so the ``arxiv_bulk`` adapter reuses this exact key rather than
    re-deriving range logic that mishandles month-straddling chunks.
    """
    yymm_part, _, suffix_part = arxiv_id.partition(_NEWSTYLE_SEP)
    return (yymm_part, int(suffix_part))


def id_in_range(arxiv_id: str, first: str, last: str) -> bool:
    """Return True iff *arxiv_id* falls within the inclusive ``[first, last]`` range.

    Range membership is decided by ``newstyle_key`` tuple comparison, so a
    chunk whose range straddles a month boundary (e.g. first ``2504.20000``,
    last ``2505.00050``) correctly includes ``2505.00010`` and excludes
    ``2503.99999``.

    Both *first* and *last* must be new-style ids.  If either bound is not
    new-style the entry is skipped (returns False) — such entries would
    represent old-style-only chunks outside our v1 scope.

    Public so the ``arxiv_bulk`` adapter reuses this exact comparison rather
    than re-implementing it.
    """
    if not (_is_newstyle(first) and _is_newstyle(last)):
        return False
    return newstyle_key(first) <= newstyle_key(arxiv_id) <= newstyle_key(last)

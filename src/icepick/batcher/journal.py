"""Journal-reading layer for the bulk-batcher subsystem.

Tails ``<run>/_progress/candidates.jsonl`` written by ``ScrapeCheckpoint.commit``
(allocation/scrape/checkpoint.py). Provides durable cursor persistence so the
batcher never loses or double-reads a record across restarts or crashes.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------


@dataclass
class JournalRow:
    """One complete, successfully parsed line from a candidates.jsonl journal.

    Attributes
    ----------
    line_no:
        1-based line number within the journal file. Counts every physical
        line including empty ones; the cursor's ``line_count`` matches this
        numbering so that a resumed read continues on the correct line.
    byte_start:
        Byte offset of the first byte of this line in the journal file.
    byte_end:
        Byte offset *exclusive* of the last byte of this line, i.e. the
        position of the byte immediately after the trailing ``\n``.  This is
        the value stored in ``cursor.json`` so the next seek lands exactly at
        the start of the next unread line.
    raw:
        The exact bytes of this line INCLUDING the trailing newline.
    row:
        The parsed JSON dict (result of ``json.loads(raw)``).
    """

    line_no: int
    byte_start: int
    byte_end: int  # exclusive; includes trailing newline
    raw: bytes  # exact bytes of the line including newline
    row: dict  # parsed JSON


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class JournalCorruption(Exception):
    """Raised when a complete (newline-terminated) line fails JSON parsing.

    A batcher must never silently drop a possibly-paid record.  This
    intentionally differs from checkpoint.py's ``_iter_jsonl`` which skips
    bad lines on resume — that loader serves the scraper's own internal state
    recovery, where a torn tail (the only realistic bad-line case) is safely
    ignorable because the paper will be re-processed.  Here, each line may
    represent a record that already cost money; dropping it would cause a
    silent data loss that the ledger cannot later recover.
    """

    def __init__(self, path: Path, line_no: int, snippet: str) -> None:
        self.path = path
        self.line_no = line_no
        self.snippet = snippet
        super().__init__(
            f"Journal corruption in {path!s} at line {line_no}: "
            f"complete line (has newline) failed JSON parse. "
            f"First 120 chars: {snippet!r}"
        )


# ---------------------------------------------------------------------------
# Cursor persistence
# ---------------------------------------------------------------------------


class CursorStore:
    """Persists per-journal read positions across restarts.

    Schema of ``cursor.json``::

        {
          "journals": {
            "<abs journal path>": {
              "line_count": <int>,
              "byte_offset": <int>
            }
          }
        }

    Design notes
    ------------
    *Atomic write via tmp+os.replace*: the cursor file is written by writing
    to a sibling ``.tmp`` file in the same directory and then calling
    ``os.replace``, which is atomic on POSIX (the kernel guarantees a reader
    either sees the old file or the new one).  A concurrent reader or a crash
    in the middle of the write can therefore never observe a half-written
    cursor.  The repo's checkpoint.py uses append+flush because its files are
    append-only (a partial append is simply a torn tail that gets skipped on
    resume); cursor.json is a full-rewrite file, so atomic replacement is
    required instead.

    *Ordering contract*: ``advance()`` updates the in-memory cursor but does
    NOT call ``save()``.  The caller must call ``save()`` explicitly, and
    must do so only AFTER a successful ledger append (or equivalent durable
    commit).  Invariant: ledger membership is the source of truth; cursor is
    an optimization.  If the process dies between ledger append and cursor
    save, the next run re-reads the already-consumed rows from the journal,
    detects them as ledger-known, and skips them idempotently — no record is
    lost or double-processed.  Advancing the cursor first would lose records
    on crash.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._state: dict[str, dict] = {}  # keyed by abs journal path string

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load cursor state from disk; tolerates missing file (empty state)."""
        if not self._path.exists():
            self._state = {}
            return
        try:
            payload = json.loads(self._path.read_bytes())
            self._state = payload.get("journals", {})
        except (json.JSONDecodeError, OSError):
            # A partial write that survived despite atomic replacement is
            # theoretically impossible, but if the file is corrupt, start
            # fresh — the ledger is the source of truth and will correct us.
            self._state = {}

    def save(self) -> None:
        """Atomically write current cursor state to disk.

        Writes to a sibling ``.tmp`` file then ``os.replace``-swaps it in.
        A crash between the two calls leaves the old cursor intact; the tmp
        file is a recognisable orphan (same dir, ``.tmp`` suffix) and is
        ignored by ``load()``.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"journals": self._state}, indent=2)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, self._path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, path: Path) -> tuple[int, int]:
        """Return ``(line_count, byte_offset)`` for *path*, defaulting to (0, 0)."""
        key = str(path.resolve())
        entry = self._state.get(key, {})
        return entry.get("line_count", 0), entry.get("byte_offset", 0)

    def advance(self, path: Path, through: JournalRow) -> None:
        """Update in-memory cursor to reflect *through* having been consumed.

        Sets ``line_count = through.line_no`` and
        ``byte_offset = through.byte_end``.

        Does NOT call ``save()``.  The caller is responsible for calling
        ``save()`` explicitly after a durable ledger commit (see class
        docstring for the ordering contract).
        """
        key = str(path.resolve())
        self._state[key] = {
            "line_count": through.line_no,
            "byte_offset": through.byte_end,
        }


# ---------------------------------------------------------------------------
# Journal tailer
# ---------------------------------------------------------------------------


class JournalTailer:
    """Reads new records from a live candidates.jsonl journal.

    The journal is written by ``ScrapeCheckpoint.commit`` via append+flush
    from a separate process.  Reads are therefore safe against concurrent
    appends: we only consume lines that end with ``\\n``.  A final line
    without a trailing newline is a torn tail (write in progress) and is
    excluded silently; it will be complete on a later ``read_new`` call.

    Parameters
    ----------
    journal_path:
        Absolute path to the ``candidates.jsonl`` file being tailed.
    cursor:
        A loaded ``CursorStore`` instance.  The tailer reads ``byte_offset``
        and ``line_count`` from it at each ``read_new`` call (via ``get``),
        and calls ``advance`` on rows it returns.  The caller drives
        ``save()`` after a durable ledger commit.
    """

    def __init__(self, journal_path: Path, cursor: CursorStore) -> None:
        self._journal_path = Path(journal_path).resolve()
        self._cursor = cursor

    def read_new(self, limit: Optional[int] = None) -> list[JournalRow]:
        """Return newly-available complete, parseable lines from the journal.

        Seeks to the byte offset stored in the cursor, then reads forward.
        Stops at (and does NOT surface) any line that:

        - has no trailing ``\\n`` — torn tail (in-progress write); will be
          complete on a future call.

        Raises ``JournalCorruption`` for any line that IS newline-terminated
        but fails JSON parsing (see that class for the rationale).

        Empty lines (``\\n`` only) are skipped but their line numbers are
        counted so that ``line_no`` accounting remains consistent with the
        stored ``line_count``.  This mirrors the journal writer which may
        produce bare newlines in edge cases, and keeps byte/line accounting
        identical regardless of how many blank lines exist.

        The cursor is ``advance``d for every row that is returned, but
        ``save()`` is NOT called — that remains the caller's responsibility.

        Parameters
        ----------
        limit:
            If given, return at most this many rows per call.  Useful for
            the batcher's 250-record slice loop.
        """
        line_count, byte_offset = self._cursor.get(self._journal_path)

        if not self._journal_path.exists():
            return []

        results: list[JournalRow] = []

        with self._journal_path.open("rb") as fh:
            fh.seek(byte_offset)
            current_line_no = line_count  # will be incremented before use

            while True:
                if limit is not None and len(results) >= limit:
                    break

                line_start = fh.tell()
                raw = fh.readline()

                if not raw:
                    # EOF reached cleanly
                    break

                current_line_no += 1

                # Torn tail: the line exists but has no trailing newline,
                # meaning the writer has not yet flushed the final byte.
                # Stop here; do not surface or advance past this line.
                if not raw.endswith(b"\n"):
                    break

                line_end = fh.tell()  # position after the newline

                # Empty line (just \n): skip but count in line_no so
                # accounting stays consistent with stored line_count.
                stripped = raw.strip()
                if not stripped:
                    # advance cursor past the empty line so we don't re-read it
                    dummy = JournalRow(
                        line_no=current_line_no,
                        byte_start=line_start,
                        byte_end=line_end,
                        raw=raw,
                        row={},
                    )
                    self._cursor.advance(self._journal_path, dummy)
                    continue

                # Complete line with content — must parse as JSON.
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    # A newline-terminated line that is not valid JSON is
                    # genuine corruption (not a torn tail).  Raise rather than
                    # skip — silent data loss is unacceptable here.
                    snippet = raw[:120].decode("utf-8", errors="replace")
                    raise JournalCorruption(self._journal_path, current_line_no, snippet)

                row = JournalRow(
                    line_no=current_line_no,
                    byte_start=line_start,
                    byte_end=line_end,
                    raw=raw,
                    row=parsed,
                )
                self._cursor.advance(self._journal_path, row)
                results.append(row)

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_manifest_source_name(run_dir: Path) -> str:
    """Return the ``source_name`` field from ``<run_dir>/manifest.json``.

    The field name is confirmed from:
    - ``src/icepick/contracts/manifests.py`` ``ApprovedManifest.source_name``
      (line 75) and ``ProposedPlan.source_name`` (line 50)
    - ``src/icepick/allocation/manifests.py`` ``write_manifest`` which
      serialises the dataclass with ``dataclasses.asdict``
    - A live manifest read:
      ``out/intake/runs/20260704T190925Z/manifest.json`` which contains
      ``"source_name": "pde_diverse_qa_500"``

    Raises ``KeyError`` with a descriptive message if the field is absent.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "source_name" not in payload:
        raise KeyError(
            f"manifest at {manifest_path} is missing required field 'source_name'. "
            f"Present fields: {sorted(payload.keys())}"
        )
    return payload["source_name"]


def run_concluded(run_dir: Path) -> bool:
    """Return True iff the extraction run has completed.

    Completion is signalled by the absence of ``<run_dir>/_progress/INCOMPLETE``
    — matching ``ScrapeCheckpoint.begin`` (which creates it) and
    ``mark_complete`` (which removes it via ``unlink``).

    Note: a run that was *never started* (no ``_progress/`` dir at all) also
    returns True here.  Callers that care about the distinction should check
    whether the progress directory exists first.
    """
    incomplete_marker = Path(run_dir) / "_progress" / "INCOMPLETE"
    return not incomplete_marker.exists()


def journal_quiet_seconds(journal_path: Path) -> float:
    """Return the number of seconds since the journal file was last modified.

    Returns ``float('inf')`` if the file does not exist.  The caller is
    responsible for supplying its own staleness policy (threshold in seconds).
    """
    journal_path = Path(journal_path)
    try:
        mtime = journal_path.stat().st_mtime
    except FileNotFoundError:
        return float("inf")
    return time.time() - mtime

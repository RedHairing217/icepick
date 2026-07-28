"""Batch slicer for the bulk-batcher subsystem.

Cuts exactly-N-record slices from a journal stream under the collision matrix
defined in docs/bulk_batcher_design.md, with an all-or-nothing commit and
crash recovery.

The two system absolutes:
  - EXACTLY N records per committed batch (remainder is held, never auto-batched).
  - ZERO record duplication across all batches and all history (enforced by the
    ledger, which is the only automatic dup gate in the funnel).

Commit-step ordering contract (see cut_slice step 6):
  The ordering matters for crash safety.  Steps before ledger.append_all are
  recomputable (deterministic re-run produces identical output).  Steps after
  ledger.append_all are idempotent (re-running them is safe).  The ledger
  append is the point of no return.

  a. mkdir batch dir             — recomputable: safe to redo
  b. slice_records.jsonl (tmp+replace) — recomputable: deterministic output
  c. slice_manifest.json (tmp+replace) — recomputable: deterministic output
  d. ledger.append_all           — POINT OF NO RETURN (fsync'd)
  e. ledger.log_skip (×skips)   — idempotent: log files are append-only
  f. cursor.advance + save       — idempotent: sets to same value
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from icepick.batcher.identity import compute_uid, stmt_key as make_stmt_key, content_hash
from icepick.batcher.journal import JournalTailer, JournalRow, CursorStore
from icepick.batcher.ledger import Ledger, LedgerRow


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------


@dataclass
class SliceConfig:
    """Parameters for a single slicing campaign."""

    campaign_source: str
    slice_size: int = 250
    cross_source_statement_policy: str = "skip"  # 'skip' | 'abort' | 'allow'


@dataclass
class SliceOutcome:
    """Result of a cut_slice call."""

    kind: str                       # 'sliced' | 'insufficient' | 'aborted'
    batch_dir: Optional[Path]
    counts: dict
    detail: str
    abort_info: Optional[dict]


# ---------------------------------------------------------------------------
# Intra-slice pending state helpers
# ---------------------------------------------------------------------------


def _intra_check(uid: str, sk: str, ch: str,
                 pending_by_uid: dict, pending_by_stmt: dict) -> Optional[str]:
    """Check uid and stmt_key against in-flight pending records.

    Returns 'replay', 'uid_conflict', 'stmt_conflict', or None (no hit).
    Precedence mirrors Ledger.check: uid hit first, then stmt_key hit.
    """
    if uid in pending_by_uid:
        prior_ch = pending_by_uid[uid][3]  # (record, uid, sk, ch, journal_row)
        return "replay" if prior_ch == ch else "uid_conflict"
    if sk in pending_by_stmt:
        return "stmt_conflict"
    return None


# ---------------------------------------------------------------------------
# cut_slice
# ---------------------------------------------------------------------------


def cut_slice(
    tailer: JournalTailer,
    cursor: CursorStore,
    ledger: Ledger,
    batches_root: Path,
    batch_no: int,
    config: SliceConfig,
    now_iso: str,
) -> SliceOutcome:
    """Cut exactly config.slice_size records into batches_root/batch<N>/.

    See module docstring and docs/bulk_batcher_design.md for the full algorithm.

    Parameters
    ----------
    tailer:
        A JournalTailer positioned at the current cursor.  read_new() is called
        once; the returned rows are processed in order.
    cursor:
        The CursorStore to advance after a successful commit.
    ledger:
        The loaded Ledger against which to check duplicates.
    batches_root:
        Root directory for batch subdirectories (e.g. out/auto_batcher/batches).
    batch_no:
        Numeric batch identifier; the output dir is batch<N>.
    config:
        Campaign-level slice configuration.
    now_iso:
        ISO-8601 timestamp string stamped into manifests and ledger rows.
    """
    journal_path = tailer._journal_path  # resolved absolute path

    # Step 1: read all available rows from the journal.
    # JournalCorruption propagates directly to the caller (daemon).
    rows = tailer.read_new()

    # Intra-slice pending state.
    # Each entry: (record_dict, uid, sk, ch, journal_row)
    pending: list[tuple[dict, str, str, str, JournalRow]] = []
    pending_by_uid: dict[str, tuple] = {}   # uid  -> pending entry
    pending_by_stmt: dict[str, tuple] = {}  # sk   -> pending entry

    collected_skips: list[dict] = []
    collected_warns: list[dict] = []

    replay_skips = 0
    stmt_skips = 0
    warns = 0

    # The journal row that completes the slice (set when pending hits slice_size).
    through_row: Optional[JournalRow] = None

    for jrow in rows:
        if len(pending) >= config.slice_size:
            # Already have a full slice; stop — do not consume more rows.
            break

        raw_row = jrow.row

        # Step 1 (cont): validate the journal row.
        candidate = raw_row.get("candidate")
        if not isinstance(candidate, dict):
            return SliceOutcome(
                kind="aborted",
                batch_dir=None,
                counts={},
                detail="journal_row_invalid",
                abort_info={
                    "reason": "missing or non-dict 'candidate' key",
                    "journal_line": jrow.line_no,
                    "raw_snippet": str(jrow.raw[:120]),
                },
            )
        statement = candidate.get("statement")
        if not statement or not isinstance(statement, str):
            return SliceOutcome(
                kind="aborted",
                batch_dir=None,
                counts={},
                detail="journal_row_invalid",
                abort_info={
                    "reason": "missing or empty 'statement' in candidate",
                    "journal_line": jrow.line_no,
                    "raw_snippet": str(jrow.raw[:120]),
                },
            )

        # Step 2: compute identity triple.
        uid = compute_uid(config.campaign_source, statement)
        sk = make_stmt_key(statement)
        ch = content_hash(raw_row)

        # Step 3: intra-slice check first, then ledger check.
        intra = _intra_check(uid, sk, ch, pending_by_uid, pending_by_stmt)

        if intra is not None:
            verdict_kind = intra
            # For intra-slice, build a synthetic prior from the pending entry.
            if intra in ("replay", "uid_conflict"):
                prior_entry = pending_by_uid[uid]
                prior_uid = prior_entry[1]
                prior_ch = prior_entry[3]
                prior_jrow = prior_entry[4]
            else:
                # stmt_conflict
                prior_entry = pending_by_stmt[sk]
                prior_uid = prior_entry[1]
                prior_ch = prior_entry[3]
                prior_jrow = prior_entry[4]
        else:
            v = ledger.check(uid, sk, ch)
            verdict_kind = v.kind
            prior = v.prior

        # Handle verdict.
        if verdict_kind == "new":
            record = dict(candidate)
            record["source"] = config.campaign_source
            record["uid"] = uid
            entry = (record, uid, sk, ch, jrow)
            pending.append(entry)
            pending_by_uid[uid] = entry
            pending_by_stmt[sk] = entry
            if len(pending) == config.slice_size:
                through_row = jrow

        elif verdict_kind == "replay":
            # Byte-identical duplicate: skip and refill.
            if intra is not None:
                # Intra-slice replay: prior is from pending.
                prior_batch = f"batch{batch_no}"  # same in-flight batch
                prior_line = prior_jrow.line_no
            else:
                prior_batch = prior.batch
                prior_line = prior.journal_line
            replay_skips += 1
            collected_skips.append({
                "kind": "replay",
                "uid": uid,
                "journal_line": jrow.line_no,
                "prior_batch": prior_batch,
                "prior_journal_line": prior_line,
            })
            # Continue (refill): do not advance through_row, do not accept.

        elif verdict_kind == "uid_conflict":
            # Hard abort: nothing is committed.
            if intra is not None:
                prior_batch = f"batch{batch_no}"
                prior_line = prior_jrow.line_no
                prior_ch_val = prior_ch
            else:
                prior_batch = prior.batch
                prior_line = prior.journal_line
                prior_ch_val = prior.content_hash
            return SliceOutcome(
                kind="aborted",
                batch_dir=None,
                counts={},
                detail="uid_conflict",
                abort_info={
                    "uid": uid,
                    "stmt_key": sk,
                    "journal_line": jrow.line_no,
                    "prior_batch": prior_batch,
                    "prior_journal_line": prior_line,
                    "prior_content_hash": prior_ch_val,
                    "new_content_hash": ch,
                },
            )

        elif verdict_kind == "stmt_conflict":
            policy = config.cross_source_statement_policy
            if policy == "skip":
                if intra is not None:
                    prior_batch = f"batch{batch_no}"
                    prior_line = prior_jrow.line_no
                else:
                    prior_batch = prior.batch
                    prior_line = prior.journal_line
                stmt_skips += 1
                collected_skips.append({
                    "kind": "stmt_skip",
                    "uid": uid,
                    "stmt_key": sk,
                    "journal_line": jrow.line_no,
                    "prior_batch": prior_batch,
                    "prior_journal_line": prior_line,
                })
            elif policy == "abort":
                if intra is not None:
                    prior_batch = f"batch{batch_no}"
                    prior_line = prior_jrow.line_no
                    prior_ch_val = prior_ch
                else:
                    prior_batch = prior.batch
                    prior_line = prior.journal_line
                    prior_ch_val = prior.content_hash
                return SliceOutcome(
                    kind="aborted",
                    batch_dir=None,
                    counts={},
                    detail="stmt_conflict",
                    abort_info={
                        "uid": uid,
                        "stmt_key": sk,
                        "journal_line": jrow.line_no,
                        "prior_batch": prior_batch,
                        "prior_journal_line": prior_line,
                        "prior_content_hash": prior_ch_val,
                        "new_content_hash": ch,
                    },
                )
            else:
                # allow: accept but increment warns
                warns += 1
                record = dict(candidate)
                record["source"] = config.campaign_source
                record["uid"] = uid
                entry = (record, uid, sk, ch, jrow)
                pending.append(entry)
                pending_by_uid[uid] = entry
                pending_by_stmt[sk] = entry
                collected_warns.append({
                    "kind": "stmt_allow",
                    "uid": uid,
                    "stmt_key": sk,
                    "journal_line": jrow.line_no,
                })
                if len(pending) == config.slice_size:
                    through_row = jrow

        elif verdict_kind == "warn":
            # Warn-set hit: accept and note.
            warns += 1
            record = dict(candidate)
            record["source"] = config.campaign_source
            record["uid"] = uid
            entry = (record, uid, sk, ch, jrow)
            pending.append(entry)
            pending_by_uid[uid] = entry
            pending_by_stmt[sk] = entry
            collected_warns.append({
                "kind": "warn",
                "uid": uid,
                "stmt_key": sk,
                "journal_line": jrow.line_no,
                "prior_uid": prior.uid if prior else None,
            })
            if len(pending) == config.slice_size:
                through_row = jrow

    # Step 5: insufficient rows — zero side effects.
    if len(pending) < config.slice_size:
        return SliceOutcome(
            kind="insufficient",
            batch_dir=None,
            counts={
                "pending_size": len(pending),
                "replay_skips": replay_skips,
                "stmt_skips": stmt_skips,
                "warns": warns,
            },
            detail=f"only {len(pending)} accepted rows available (need {config.slice_size})",
            abort_info=None,
        )

    # Step 6: COMMIT (pending == slice_size exactly).
    batch_label = f"batch{batch_no}"
    batch_dir = batches_root / batch_label

    # --- 6a. mkdir ---
    # Ordering contract: mkdir is before any file writes; recomputable.
    try:
        batch_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # If the dir already exists, check for manifest (batch_dir_conflict).
        manifest_path = batch_dir / "slice_manifest.json"
        if manifest_path.exists():
            return SliceOutcome(
                kind="aborted",
                batch_dir=batch_dir,
                counts={},
                detail="batch_dir_conflict",
                abort_info={"batch_dir": str(batch_dir)},
            )
        # Dir exists but no manifest: pre-commit crash; we can continue writing.

    # Build the 250-entry and skip data for the manifest.
    manifest_entries = [
        {
            "uid": e[1],
            "stmt_key": e[2],
            "content_hash": e[3],
            "journal_line": e[4].line_no,
        }
        for e in pending
    ]

    # Determine journal span.
    from_line = pending[0][4].line_no
    through_line = through_row.line_no
    through_byte = through_row.byte_end

    # --- 6b. slice_records.jsonl via tmp+os.replace ---
    # Ordering contract: written before manifest; both recomputable.
    records_path = batch_dir / "slice_records.jsonl"
    records_tmp = batch_dir / "slice_records.jsonl.tmp"
    with records_tmp.open("w", encoding="utf-8") as fh:
        for e in pending:
            fh.write(json.dumps(e[0]) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(records_tmp, records_path)

    # --- 6c. slice_manifest.json via tmp+os.replace ---
    # Ordering contract: written before ledger append; recomputable.
    manifest = {
        "batch": batch_label,
        "campaign_source": config.campaign_source,
        "slice_size": config.slice_size,
        "journal_path": str(journal_path),
        "journal_span": {
            "from_line": from_line,
            "through_line": through_line,
            "through_byte": through_byte,
        },
        "created_at": now_iso,
        "entries": manifest_entries,
        "skips": collected_skips,
        "warns": collected_warns,
        "counts": {
            "accepted": config.slice_size,
            "replay_skips": replay_skips,
            "stmt_skips": stmt_skips,
            "warns": warns,
        },
    }
    manifest_path = batch_dir / "slice_manifest.json"
    manifest_tmp = batch_dir / "slice_manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)

    # --- 6d. ledger.append_all — POINT OF NO RETURN ---
    # Everything before is recomputable. Everything after is idempotent.
    ledger_rows = [
        LedgerRow(
            uid=e[1],
            stmt_key=e[2],
            content_hash=e[3],
            batch=batch_label,
            source_journal=str(journal_path),
            journal_line=e[4].line_no,
            sliced_at=now_iso,
            warn_only=False,
        )
        for e in pending
    ]
    ledger.append_all(ledger_rows)

    # --- 6e. ledger.log_skip for each collected skip ---
    # Idempotent: skip log is append-only.
    for skip in collected_skips:
        enriched = dict(skip)
        enriched["batch"] = batch_label
        enriched["sliced_at"] = now_iso
        ledger.log_skip(enriched)

    # --- 6f. cursor.advance + cursor.save ---
    # Idempotent: sets cursor to the same value on re-run.
    cursor.advance(journal_path, through_row)
    cursor.save()

    return SliceOutcome(
        kind="sliced",
        batch_dir=batch_dir,
        counts={
            "accepted": config.slice_size,
            "replay_skips": replay_skips,
            "stmt_skips": stmt_skips,
            "warns": warns,
        },
        detail=f"committed {batch_label}",
        abort_info=None,
    )


# ---------------------------------------------------------------------------
# recover_pending_slice
# ---------------------------------------------------------------------------


def recover_pending_slice(
    batches_root: Path,
    ledger: Ledger,
    cursor: CursorStore,
    tailer_factory: Callable[[Path], JournalTailer],
) -> dict:
    """Recover any interrupted commit from the highest-numbered batch dir.

    Called by the daemon BEFORE any cut_slice call.  Scans batch dirs for
    interrupted commits and finishes them idempotently.

    Parameters
    ----------
    batches_root:
        Root of the batches directory (same as passed to cut_slice).
    ledger:
        Loaded Ledger (will have rows appended if recovery re-runs step d).
    cursor:
        Loaded CursorStore (will be advanced if recovery re-runs step f).
    tailer_factory:
        Callable(journal_path) -> JournalTailer — used only to produce a tailer
        for cursor.advance; actual read_new is never called during recovery.

    Recovery logic
    --------------
    Case 1 — slice_manifest.json present:
        The commit may have been interrupted after the manifest was written
        (steps d/e/f might not have completed).  Re-run steps d-f from the
        manifest data; all three are idempotent.

    Case 2 — *.tmp files or slice_records.jsonl WITHOUT slice_manifest.json:
        Pre-manifest crash (between steps a-b and c).  The records file may be
        incomplete or missing.  Remove only .tmp files; leave slice_records.jsonl
        (it will be deterministically recomputed by os.replace on the next
        cut_slice call).  Report {recomputable: batch}.

    Case 3 — Nothing pending:
        Return {}.
    """
    if not batches_root.exists():
        return {}

    # Find all batch<N> dirs, sorted by N descending.
    batch_dirs = []
    for d in batches_root.iterdir():
        if d.is_dir() and d.name.startswith("batch"):
            try:
                n = int(d.name[len("batch"):])
                batch_dirs.append((n, d))
            except ValueError:
                continue
    if not batch_dirs:
        return {}

    batch_dirs.sort(key=lambda x: x[0], reverse=True)

    # Check the highest-numbered batch for an interrupted commit.
    # (We only look at the max — a completed batch advances the cursor; there
    # is at most one in-flight batch at any moment.)
    batch_no, batch_dir = batch_dirs[0]
    batch_label = f"batch{batch_no}"

    manifest_path = batch_dir / "slice_manifest.json"

    if manifest_path.exists():
        # Case 1: manifest is present — re-run steps d-f.
        actions = []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {"error": f"could not read manifest at {manifest_path}: {exc}"}

        entries = manifest.get("entries", [])
        journal_path = Path(manifest["journal_path"])
        span = manifest["journal_span"]
        now_iso = manifest["created_at"]
        campaign_source = manifest["campaign_source"]
        skips = manifest.get("skips", [])

        # Step d (idempotent): append_all — rows already in ledger are skipped.
        ledger_rows = [
            LedgerRow(
                uid=e["uid"],
                stmt_key=e["stmt_key"],
                content_hash=e["content_hash"],
                batch=batch_label,
                source_journal=str(journal_path),
                journal_line=e["journal_line"],
                sliced_at=now_iso,
                warn_only=False,
            )
            for e in entries
        ]
        ledger.append_all(ledger_rows)
        actions.append("ledger_append_all")

        # Step e (idempotent): log_skip for each skip entry.
        # Note: cross_source_skips.jsonl is append-only; re-running creates
        # duplicate log lines.  This is acceptable — skips are informational
        # only (not dedup state); the ledger is the dedup source of truth.
        for skip in skips:
            enriched = dict(skip)
            enriched.setdefault("batch", batch_label)
            enriched.setdefault("sliced_at", now_iso)
            ledger.log_skip(enriched)
        if skips:
            actions.append(f"log_skip x{len(skips)}")

        # Step f: advance cursor if currently behind the committed span.
        cur_line, cur_byte = cursor.get(journal_path)
        through_line = span["through_line"]
        through_byte = span["through_byte"]
        if cur_line < through_line or cur_byte < through_byte:
            # Build a minimal JournalRow to pass to cursor.advance.
            # We do not call read_new — we set the cursor directly using the
            # span data from the manifest.  CursorStore.advance sets:
            #   line_count = through.line_no
            #   byte_offset = through.byte_end
            # A dummy JournalRow carrying the manifest span values is sufficient.
            dummy_row = _make_dummy_journal_row(
                line_no=through_line,
                byte_end=through_byte,
            )
            cursor.advance(journal_path, dummy_row)
            cursor.save()
            actions.append("cursor_advanced")

        return {"recovered": batch_label, "actions": actions}

    else:
        # Case 2: no manifest — pre-commit crash.
        # Remove only .tmp files; leave any slice_records.jsonl.
        tmp_files = list(batch_dir.glob("*.tmp"))
        for f in tmp_files:
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        return {"recomputable": batch_label, "tmp_removed": [str(f) for f in tmp_files]}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _make_dummy_journal_row(line_no: int, byte_end: int) -> JournalRow:
    """Construct a minimal JournalRow for cursor.advance during recovery.

    cursor.advance only reads .line_no and .byte_end, so other fields are
    filled with safe zero/empty values.
    """
    return JournalRow(
        line_no=line_no,
        byte_start=0,
        byte_end=byte_end,
        raw=b"",
        row={},
    )

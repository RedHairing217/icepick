"""
stages.py — funnel stage runners for the bulk-batcher subsystem.

Builds and executes the three funnel stage commands (mount, cascade, pass@k)
with verification, cost guard, Qwen-slot gate, and bounded retries.

Standalone: no imports from other batcher modules. Everything is injectable
for tests. Subprocess is used only through the injected `runner` callables.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_TRANSIENT_MARKERS: tuple[str, ...] = (
    "529",
    "overloaded",
    "timeout",
    "connection",
    "rate",
)


# ---------------------------------------------------------------------------
# StageOutcome
# ---------------------------------------------------------------------------


@dataclass
class StageOutcome:
    """Result returned by every stage runner. Never raises for operational failures."""

    ok: bool
    kind: str
    detail: str
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def build_mount_cmd(
    slice_records: str,
    campaign_source: str,
    batch_dir: str,
    mode: str = "production",  # unused for mount but kept for API symmetry
    key_path: Optional[str] = None,  # unused for mount
    calibration_sheet: Optional[str] = None,  # unused for mount
    icepick_bin: str = "icepick",
) -> list[str]:
    """Build the argv list for `icepick allocation mount`."""
    return [
        icepick_bin,
        "allocation",
        "mount",
        "--path", slice_records,
        "--source", campaign_source,
        "--provenance", "extracted",
        "--truth-policy", "extracted",
        "--family", "realmath",
        "--output-dir", str(Path(batch_dir) / "intake"),
    ]


def build_cascade_cmd(
    handoff_records: str,
    batch_dir: str,
    key_path: str,
    mode: str = "production",
    calibration_sheet: Optional[str] = None,
    icepick_bin: str = "icepick",
) -> list[str]:
    """Build the argv list for `icepick processing wellposed-cascade`."""
    cmd = [
        icepick_bin,
        "processing",
        "wellposed-cascade",
        "--mode", mode,
        "--stages", "codex:anthropic",
        "--input", handoff_records,
        "--output-dir", str(Path(batch_dir) / "cascade"),
        "--anthro-key-file", key_path,
        "--cost-per-input-mtok", "3",
        "--cost-per-output-mtok", "15",
    ]
    if calibration_sheet is not None:
        cmd += ["--calibration-sheet", calibration_sheet]
    return cmd


def build_passk_cmd(
    batch_dir: str,
    mode: str = "production",
    calibration_sheet: Optional[str] = None,
    icepick_bin: str = "icepick",
) -> list[str]:
    """Build the argv list for `icepick processing pass_at_k`."""
    cmd = [
        icepick_bin,
        "processing",
        "pass_at_k",
        "--mode", mode,
        "--input", str(Path(batch_dir) / "cascade" / "final_corpus.jsonl"),
        "--output-dir", str(Path(batch_dir) / "pass_at_k"),
        "--backend", "qwen_http",
        "--backend-url", "http://127.0.0.1:1234/v1/chat/completions",
        "--model", "qwen/qwen3-8b",
        "--k", "8",
        "--temperature", "0.7",
        "--max-tokens", "2048",
        "--think", "off",
        "--max-concurrent", "1",
    ]
    if calibration_sheet is not None:
        cmd += ["--calibration-sheet", calibration_sheet]
    return cmd


# ---------------------------------------------------------------------------
# Subprocess env helper
# ---------------------------------------------------------------------------


def build_stage_env(key_path: str) -> dict:
    """Return a copy of os.environ with ANTHROPIC_KEY_FILE set to key_path.

    Never opens the file — passes the path string only.
    """
    env = os.environ.copy()
    env["ANTHROPIC_KEY_FILE"] = key_path
    return env


# ---------------------------------------------------------------------------
# Default subprocess runner
# ---------------------------------------------------------------------------


def _default_runner(argv, env=None, capture_output=True, text=True, timeout=None):
    """Thin wrapper around subprocess.run used as the default runner."""
    return subprocess.run(
        argv,
        env=env,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# run_mount
# ---------------------------------------------------------------------------

_MOUNT_MARKER_NAME = "MOUNT_VERIFIED"


def _uid_set_sha(uids: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(uids)).encode()).hexdigest()


def run_mount(
    runner,
    batch_dir: str,
    slice_records: str,
    campaign_source: str,
    expected_uids: list[str],
    mode: str = "production",
    key_path: Optional[str] = None,
    calibration_sheet: Optional[str] = None,
    icepick_bin: str = "icepick",
    timeout: Optional[int] = 120,
) -> StageOutcome:
    """Execute the mount command and verify the handoff.

    Idempotence: if MOUNT_VERIFIED marker exists and uid-set matches, returns
    ok immediately without re-running.

    If a prior run dir exists but was never verified → kind='mount_dirty'
    (daemon freezes batch; human inspects).
    """
    batch_path = Path(batch_dir)
    intake_path = batch_path / "intake"
    marker_path = intake_path / _MOUNT_MARKER_NAME
    runs_path = intake_path / "runs"
    expected_set = set(expected_uids)
    expected_sha = _uid_set_sha(expected_uids)

    # --- Idempotence check ---
    if marker_path.exists():
        try:
            marker_data = json.loads(marker_path.read_text())
            if marker_data.get("uid_set_sha") == expected_sha:
                return StageOutcome(
                    ok=True,
                    kind="mount_ok",
                    detail="Prior verified mount found; skipping re-run.",
                    data={"run_dir": marker_data.get("run_dir"), "resumed": True},
                )
            # uid set changed — treat as dirty
            return StageOutcome(
                ok=False,
                kind="mount_dirty",
                detail=(
                    "MOUNT_VERIFIED marker exists but uid_set_sha differs "
                    "from expected. Manual inspection required."
                ),
                data={"marker_sha": marker_data.get("uid_set_sha"), "expected_sha": expected_sha},
            )
        except (json.JSONDecodeError, OSError) as exc:
            return StageOutcome(
                ok=False,
                kind="mount_dirty",
                detail=f"MOUNT_VERIFIED marker unreadable: {exc}",
                data={},
            )

    # --- Detect existing unverified run dirs ---
    if runs_path.exists():
        existing_runs = [d for d in runs_path.iterdir() if d.is_dir()]
        if existing_runs:
            return StageOutcome(
                ok=False,
                kind="mount_dirty",
                detail=(
                    f"Unverified run dir(s) exist under {runs_path} but no "
                    "MOUNT_VERIFIED marker. Manual inspection required before re-running."
                ),
                data={"existing_runs": [str(d) for d in existing_runs]},
            )

    # --- Execute mount ---
    argv = build_mount_cmd(
        slice_records=slice_records,
        campaign_source=campaign_source,
        batch_dir=batch_dir,
        icepick_bin=icepick_bin,
    )
    try:
        result = runner(argv, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return StageOutcome(
            ok=False,
            kind="exec_failed",
            detail=f"mount subprocess error: {exc}",
            data={},
        )

    if result.returncode != 0:
        return StageOutcome(
            ok=False,
            kind="exec_failed",
            detail=f"mount exited {result.returncode}: {result.stderr or result.stdout}",
            data={"returncode": result.returncode},
        )

    # --- Verify output ---
    if not runs_path.exists():
        return StageOutcome(
            ok=False,
            kind="mount_verification_failed",
            detail=f"Expected runs dir not created: {runs_path}",
            data={},
        )

    new_runs = [d for d in runs_path.iterdir() if d.is_dir()]
    if len(new_runs) != 1:
        return StageOutcome(
            ok=False,
            kind="mount_verification_failed",
            detail=f"Expected exactly 1 run dir, found {len(new_runs)}: {new_runs}",
            data={"run_dirs": [str(d) for d in new_runs]},
        )

    run_dir = new_runs[0]
    handoff_path = run_dir / "handoff" / "records.jsonl"
    if not handoff_path.exists():
        return StageOutcome(
            ok=False,
            kind="mount_verification_failed",
            detail=f"Handoff file missing: {handoff_path}",
            data={"run_dir": str(run_dir)},
        )

    # Read and check the handoff records
    rows = []
    try:
        for line in handoff_path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        return StageOutcome(
            ok=False,
            kind="mount_verification_failed",
            detail=f"Could not read handoff file: {exc}",
            data={"run_dir": str(run_dir)},
        )

    if len(rows) != len(expected_uids):
        return StageOutcome(
            ok=False,
            kind="mount_verification_failed",
            detail=(
                f"Row count mismatch: handoff has {len(rows)} rows, "
                f"expected {len(expected_uids)}"
            ),
            data={"row_count": len(rows), "expected_count": len(expected_uids)},
        )

    actual_uids = {row.get("uid") for row in rows}
    if actual_uids != expected_set:
        missing = sorted(expected_set - actual_uids)
        extra = sorted(actual_uids - expected_set)
        return StageOutcome(
            ok=False,
            kind="mount_verification_failed",
            detail=(
                f"UID set mismatch: {len(missing)} missing, {len(extra)} extra in handoff."
            ),
            data={"missing_uids": missing, "extra_uids": extra},
        )

    # --- Write marker ---
    marker_data = {"run_dir": str(run_dir), "uid_set_sha": expected_sha}
    try:
        intake_path.mkdir(parents=True, exist_ok=True)
        tmp = marker_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(marker_data))
        tmp.rename(marker_path)
    except OSError as exc:
        # Non-fatal: mount succeeded, we just can't persist the marker.
        return StageOutcome(
            ok=True,
            kind="mount_ok",
            detail=f"Mount verified OK but could not write marker: {exc}",
            data={"run_dir": str(run_dir), "handoff": str(handoff_path)},
        )

    return StageOutcome(
        ok=True,
        kind="mount_ok",
        detail="Mount verified: row count and uid set match.",
        data={"run_dir": str(run_dir), "handoff": str(handoff_path)},
    )


# ---------------------------------------------------------------------------
# run_cascade
# ---------------------------------------------------------------------------


def run_cascade(
    runner,
    batch_dir: str,
    handoff_records: str,
    key_path: str,
    cost_limit_usd: float = 5.0,
    mode: str = "production",
    calibration_sheet: Optional[str] = None,
    icepick_bin: str = "icepick",
    timeout: Optional[int] = 7200,
) -> StageOutcome:
    """Execute the cascade stage with skip-if-done, cost guard, and verification."""
    batch_path = Path(batch_dir)
    cascade_dir = batch_path / "cascade"
    manifest_path = cascade_dir / "cascade_manifest.json"
    final_corpus_path = cascade_dir / "final_corpus.jsonl"

    # --- Skip-if-done: manifest already exists and parses ---
    parsed_manifest = None
    if manifest_path.exists():
        try:
            parsed_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            parsed_manifest = None

    ran_now = False
    if parsed_manifest is None:
        # Execute cascade
        argv = build_cascade_cmd(
            handoff_records=handoff_records,
            batch_dir=batch_dir,
            key_path=key_path,
            mode=mode,
            calibration_sheet=calibration_sheet,
            icepick_bin=icepick_bin,
        )
        env = build_stage_env(key_path)
        try:
            result = runner(argv, env=env, capture_output=True, text=True, timeout=timeout)
        except Exception as exc:
            return StageOutcome(
                ok=False,
                kind="exec_failed",
                detail=f"cascade subprocess error: {exc}",
                data={},
            )

        if result.returncode != 0:
            return StageOutcome(
                ok=False,
                kind="exec_failed",
                detail=f"cascade exited {result.returncode}: {result.stderr or result.stdout}",
                data={"returncode": result.returncode},
            )

        ran_now = True
        if manifest_path.exists():
            try:
                parsed_manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                return StageOutcome(
                    ok=False,
                    kind="cascade_verification_failed",
                    detail=f"Could not read cascade_manifest.json after run: {exc}",
                    data={},
                )
        else:
            return StageOutcome(
                ok=False,
                kind="cascade_verification_failed",
                detail=f"cascade_manifest.json not created after run: {manifest_path}",
                data={},
            )

    # --- Parse manifest fields ---
    try:
        overall = parsed_manifest.get("overall", {}) or {}
        inputs_section = parsed_manifest.get("inputs", {}) or {}

        cost_raw = overall.get("total_estimated_cost_usd")
        if cost_raw is None:
            cost_usd = 0.0
            cost_note = "null cost in manifest (flow_testing or no cost metering)"
        else:
            cost_usd = float(cost_raw)
            cost_note = None

        initial_record_count = (
            inputs_section.get("initial_record_count")
            or overall.get("initial_record_count")
        )
        final_corpus_count = overall.get("final_corpus_count")
    except (KeyError, TypeError, ValueError) as exc:
        return StageOutcome(
            ok=False,
            kind="cascade_verification_failed",
            detail=f"Could not parse cascade_manifest.json fields: {exc}",
            data={},
        )

    data: dict = {
        "cost_usd": cost_usd,
        "initial_record_count": initial_record_count,
        "final_corpus_count": final_corpus_count,
        "ran_now": ran_now,
    }
    if cost_note:
        data["cost_note"] = cost_note

    # --- Verify final_corpus.jsonl exists ---
    if not final_corpus_path.exists():
        return StageOutcome(
            ok=False,
            kind="cascade_verification_failed",
            detail=f"final_corpus.jsonl missing: {final_corpus_path}",
            data=data,
        )

    # --- Cost guard ---
    if cost_usd > cost_limit_usd:
        return StageOutcome(
            ok=False,
            kind="cost_guard_tripped",
            detail=(
                f"Cascade cost ${cost_usd:.4f} exceeds limit ${cost_limit_usd:.2f}. "
                "Record-bloat guard: queue halted. Manual review required."
            ),
            data=data,
        )

    # --- Count final corpus rows for info ---
    try:
        corpus_lines = [
            ln for ln in final_corpus_path.read_text().splitlines() if ln.strip()
        ]
        data["final_corpus_row_count"] = len(corpus_lines)
        if len(corpus_lines) == 0:
            data["empty_corpus_note"] = "pass@k stage will no-op"
    except OSError:
        pass  # not fatal; guard passed

    detail = f"Cascade complete. cost=${cost_usd:.4f}, initial={initial_record_count}, final={final_corpus_count}."
    if not ran_now:
        detail = "Cascade skipped (manifest already present). " + detail
    return StageOutcome(ok=True, kind="cascade_ok", detail=detail, data=data)


# ---------------------------------------------------------------------------
# run_passk
# ---------------------------------------------------------------------------


def run_passk(
    runner,
    batch_dir: str,
    mode: str = "production",
    calibration_sheet: Optional[str] = None,
    slot_checker: Optional[Callable[[], bool]] = None,
    icepick_bin: str = "icepick",
    timeout: Optional[int] = 14400,
) -> StageOutcome:
    """Execute pass@k with slot gating, skip-if-done, and interrupted-resume.

    slot_checker() -> bool; True = slot is FREE. Checked BEFORE execution.
    The pgrep pattern will match our own child once we launch — the daemon
    must only call qwen_slot_free() before launching, never while its own
    pass@k child is running. (We do not exclude our own pid tree here because
    we check before exec; the started child is not yet running at check time.)
    """
    batch_path = Path(batch_dir)
    passk_dir = batch_path / "pass_at_k"
    manifest_path = passk_dir / "pass_at_k_manifest.json"

    # --- Slot gate (checked immediately before exec) ---
    if slot_checker is not None:
        try:
            slot_free = slot_checker()
        except Exception as exc:
            return StageOutcome(
                ok=False,
                kind="qwen_slot_busy",
                detail=f"slot_checker raised: {exc}",
                data={},
            )
        if not slot_free:
            return StageOutcome(
                ok=False,
                kind="qwen_slot_busy",
                detail="Qwen slot is busy (pass@k already running or port:1234 established). Retry later.",
                data={},
            )

    # --- Skip-if-done: manifest present and interrupted == false ---
    if manifest_path.exists():
        try:
            mdata = json.loads(manifest_path.read_text())
            if not mdata.get("interrupted", True):
                counts = mdata.get("counts", {})
                return StageOutcome(
                    ok=True,
                    kind="passk_ok",
                    detail="pass@k complete (manifest present, interrupted=false). Skipping re-run.",
                    data={"counts": counts, "resumed": True},
                )
            # interrupted == true → fall through to re-run (resume semantics)
        except (OSError, json.JSONDecodeError):
            pass  # corrupt manifest — attempt re-run

    # --- Execute pass@k ---
    argv = build_passk_cmd(
        batch_dir=batch_dir,
        mode=mode,
        calibration_sheet=calibration_sheet,
        icepick_bin=icepick_bin,
    )
    try:
        result = runner(argv, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return StageOutcome(
            ok=False,
            kind="exec_failed",
            detail=f"pass@k subprocess error: {exc}",
            data={},
        )

    if result.returncode != 0:
        return StageOutcome(
            ok=False,
            kind="exec_failed",
            detail=f"pass@k exited {result.returncode}: {result.stderr or result.stdout}",
            data={"returncode": result.returncode},
        )

    # --- Parse resulting manifest ---
    if manifest_path.exists():
        try:
            mdata = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return StageOutcome(
                ok=False,
                kind="passk_verification_failed",
                detail=f"Could not read pass_at_k_manifest.json: {exc}",
                data={},
            )
    else:
        return StageOutcome(
            ok=False,
            kind="passk_verification_failed",
            detail=f"pass_at_k_manifest.json not found after run: {manifest_path}",
            data={},
        )

    counts = mdata.get("counts", {})
    interrupted = mdata.get("interrupted", False)

    if interrupted:
        return StageOutcome(
            ok=False,
            kind="passk_interrupted",
            detail="pass@k manifest shows interrupted=true. Re-run to resume.",
            data={"counts": counts},
        )

    return StageOutcome(
        ok=True,
        kind="passk_ok",
        detail="pass@k complete (interrupted=false).",
        data={"counts": counts},
    )


# ---------------------------------------------------------------------------
# cascade_slot_free
# ---------------------------------------------------------------------------


def cascade_slot_free(
    pgrep_runner: Optional[Callable] = None,
) -> bool:
    """Return True iff no foreign `icepick processing wellposed-cascade` is running.

    Checks `pgrep -f "icepick processing wellposed-cascade"` and returns True
    only if the command matches nothing (rc=1).  No port check is needed:
    Anthropic's API is remote, so cascade contention is soft (API quota
    exhaustion) rather than a corruption risk.  The gate exists because the
    operator requires the daemon never to launch a second cascade while any
    foreign cascade process is running — shared API quota and overlapping
    output-dir writes could cause cost overruns or manifest collisions.

    Injectable runner for tests (same contract as qwen_slot_free's pgrep_runner).
    Defaults use subprocess.run with capture_output=True, text=True.

    NOTE: The pgrep pattern will match our own child process once we launch
    cascade.  The daemon must consult this function ONLY before launching —
    never during its own run.  Check → launch → never check again until the
    child exits.
    """
    def _default_sub_runner(argv):
        return subprocess.run(argv, capture_output=True, text=True)

    _pgrep = pgrep_runner if pgrep_runner is not None else _default_sub_runner

    try:
        pgrep_result = _pgrep(["pgrep", "-f", "icepick processing wellposed-cascade"])
        # rc=1 means no match → free; rc=0 means match → busy
        if pgrep_result.returncode == 0 and pgrep_result.stdout.strip():
            return False
    except Exception:
        # If pgrep itself fails, conservatively report busy
        return False

    return True


# ---------------------------------------------------------------------------
# qwen_slot_free
# ---------------------------------------------------------------------------


def qwen_slot_free(
    pgrep_runner: Optional[Callable] = None,
    lsof_runner: Optional[Callable] = None,
) -> bool:
    """Return True iff the Qwen slot is free.

    Free iff:
    - `pgrep -f "icepick processing pass_at_k"` matches nothing (rc=1)
    - `lsof -i TCP:1234 -sTCP:ESTABLISHED -t` outputs nothing

    Runners are injectable for tests. Defaults use subprocess.run with
    capture_output=True, text=True.

    NOTE: The pgrep pattern will match our own child process once we launch
    pass@k. The daemon must consult this function ONLY before launching —
    never during its own run. We intentionally do not filter our own pid tree
    here (that would require spawning another subprocess and add fragility);
    the protocol is: check → launch → never check again until the child exits.
    """
    def _default_sub_runner(argv):
        return subprocess.run(argv, capture_output=True, text=True)

    _pgrep = pgrep_runner if pgrep_runner is not None else _default_sub_runner
    _lsof = lsof_runner if lsof_runner is not None else _default_sub_runner

    try:
        pgrep_result = _pgrep(["pgrep", "-f", "icepick processing pass_at_k"])
        # rc=1 means no match → free; rc=0 means match → busy
        if pgrep_result.returncode == 0 and pgrep_result.stdout.strip():
            return False
    except Exception:
        # If pgrep itself fails, conservatively report busy
        return False

    try:
        lsof_result = _lsof(["lsof", "-i", "TCP:1234", "-sTCP:ESTABLISHED", "-t"])
        # Any output means an established connection exists → busy
        if lsof_result.stdout.strip():
            return False
    except Exception:
        return False

    return True


# ---------------------------------------------------------------------------
# with_retries
# ---------------------------------------------------------------------------


def _default_is_retryable(outcome: StageOutcome) -> bool:
    """Default retryability classifier.

    Retryable: kind in {'exec_failed', 'passk_interrupted'} AND detail
    mentions a transient marker (case-insensitive).

    Non-retryable regardless:
      cost_guard_tripped, mount_verification_failed, mount_dirty,
      qwen_slot_busy — these require human or daemon-level intervention.
    """
    NON_RETRYABLE = {
        "cost_guard_tripped",
        "mount_verification_failed",
        "mount_dirty",
        "qwen_slot_busy",
    }
    if outcome.kind in NON_RETRYABLE:
        return False
    if outcome.kind not in {"exec_failed", "passk_interrupted"}:
        return False
    detail_lower = outcome.detail.lower()
    return any(marker.lower() in detail_lower for marker in DEFAULT_TRANSIENT_MARKERS)


def with_retries(
    fn: Callable[[], StageOutcome],
    is_retryable: Callable[[StageOutcome], bool] = _default_is_retryable,
    max_attempts: int = 3,
    backoffs: Sequence[float] = (30, 120, 480),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> StageOutcome:
    """Call fn up to max_attempts times, retrying when is_retryable(outcome).

    Returns the last outcome. Attaches data['attempts'] with the attempt count.
    Sleeps backoffs[attempt_index] seconds between attempts (uses last backoff
    value if more attempts than backoff entries).
    """
    last_outcome: Optional[StageOutcome] = None
    for attempt in range(1, max_attempts + 1):
        last_outcome = fn()
        last_outcome.data["attempts"] = attempt
        if last_outcome.ok:
            return last_outcome
        if not is_retryable(last_outcome):
            return last_outcome
        if attempt < max_attempts:
            sleep_idx = min(attempt - 1, len(backoffs) - 1)
            sleep_fn(backoffs[sleep_idx])
    return last_outcome  # type: ignore[return-value]

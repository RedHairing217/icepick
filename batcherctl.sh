#!/usr/bin/env bash
# batcherctl.sh — single-file operator control script for the bulk-batcher.
# Usage: batcherctl.sh [status|arm|disarm|tail|clear-halt <reason>|help]
# Defaults to `status` when called with no arguments.
#
# Every subcommand ends with a STATE line (JSON) and exits 0 iff it succeeded.
# Designed for low-capability agents: one word per action, no flag knowledge needed.
#
# ── Env overrides (FOR TESTING ONLY — operators never set these) ──────────────
# BATCHERCTL_ROOT       batcher state root
# BATCHERCTL_JOURNAL    primary candidates.jsonl path
# BATCHERCTL_RUN_DIR    extraction run directory owning the primary journal
# BATCHERCTL_CAMPAIGN   campaign source name
# BATCHERCTL_WATCH      watch-journal spec  LABEL=RUN_DIR
# BATCHERCTL_BIN        icepick binary
# BATCHERCTL_LOG        daemon log file
# ─────────────────────────────────────────────────────────────────────────────

set -u

# ── Baked-in production defaults ──────────────────────────────────────────────
ROOT="${BATCHERCTL_ROOT:-/Users/redhairing/Desktop/helloworld/icepick/out/auto_batcher}"
JOURNAL="${BATCHERCTL_JOURNAL:-/Users/redhairing/Desktop/helloworld/icepick/out/intake/runs/20260707T072733Z/_progress/candidates.jsonl}"
RUN_DIR="${BATCHERCTL_RUN_DIR:-/Users/redhairing/Desktop/helloworld/icepick/out/intake/runs/20260707T072733Z}"
CAMPAIGN="${BATCHERCTL_CAMPAIGN:-arxiv_bulk_pde625}"
WATCH="${BATCHERCTL_WATCH:-batch9=/Users/redhairing/Desktop/helloworld/icepick/out/intake/runs/20260707T170022Z}"
BIN="${BATCHERCTL_BIN:-icepick}"
LOG="${BATCHERCTL_LOG:-${ROOT}/daemon.log}"

# ── Helpers ───────────────────────────────────────────────────────────────────

_daemon_pid() {
  # Match the daemon process by its root path so we don't grab our own shell.
  pgrep -f "batcher run.*${ROOT}" 2>/dev/null | head -1
}

_gather_state() {
  # Emits a single-line JSON object with all required keys.
  python3 - "$ROOT" "$JOURNAL" <<'PYEOF'
import sys, json, os, glob

root    = sys.argv[1]
journal = sys.argv[2]

# armed
armed = os.path.exists(os.path.join(root, "ARMED"))

# queue_state
qs_path = os.path.join(root, "queue_state.json")
halt_val = None
next_batch = 0
if os.path.exists(qs_path):
    try:
        qs = json.loads(open(qs_path).read())
        h = qs.get("halt") or {}
        if isinstance(h, dict) and h.get("active"):
            halt_val = h.get("reason") or "halted"
        next_batch = int(qs.get("next_batch_number", 0))
    except Exception:
        pass

# journal rows
journal_rows = 0
if os.path.exists(journal):
    try:
        with open(journal, "rb") as fh:
            journal_rows = sum(1 for _ in fh)
    except Exception:
        pass

# consumed rows for this journal (from cursor.json)
consumed = 0
cursor_path = os.path.join(root, "ledger", "cursor.json")
if os.path.exists(cursor_path):
    try:
        c = json.loads(open(cursor_path).read())
        journals = c.get("journals", {})
        entry = journals.get(journal) or c.get(journal)
        if isinstance(entry, dict):
            consumed = int(entry.get("line_count", 0))
    except Exception:
        pass

rows_toward = max(0, journal_rows - consumed)

# batches ready
batches_ready = 0
frozen = []
batches_dir = os.path.join(root, "batches")
if os.path.isdir(batches_dir):
    for batch in sorted(os.listdir(batches_dir)):
        bdir = os.path.join(batches_dir, batch)
        if not os.path.isdir(bdir):
            continue
        if os.path.exists(os.path.join(bdir, "READY_TO_FOLD")):
            batches_ready += 1
        state_path = os.path.join(bdir, "state.json")
        if os.path.exists(state_path):
            try:
                st = json.loads(open(state_path).read())
                if st.get("frozen"):
                    frozen.append(batch)
            except Exception:
                pass

print(json.dumps({
    "armed": armed,
    "halt": halt_val,
    "next_batch": next_batch,
    "journal_rows": journal_rows,
    "consumed": consumed,
    "rows_toward_next_slice": rows_toward,
    "batches_ready": batches_ready,
    "frozen": frozen,
}))
PYEOF
}

_build_state_line() {
  # Args: ok (true|false) note pid_override(optional)
  local ok="$1"
  local note="$2"
  local pid_override="${3:-}"

  local raw
  raw=$(_gather_state)

  python3 - "$ok" "$note" "$pid_override" "$ROOT" "$JOURNAL" <<PYEOF2
import sys, json, os

ok_str       = sys.argv[1]          # "true" or "false"
note         = sys.argv[2]
pid_override = sys.argv[3]          # "" or an integer string
root         = sys.argv[4]
journal      = sys.argv[5]

raw = """$raw"""

try:
    d = json.loads(raw)
except Exception:
    d = {
        "armed": False, "halt": None, "next_batch": 0,
        "journal_rows": 0, "consumed": 0,
        "rows_toward_next_slice": 0,
        "batches_ready": 0, "frozen": [],
    }

# daemon pid
if pid_override:
    try:
        daemon_pid = int(pid_override)
    except Exception:
        daemon_pid = None
else:
    import subprocess
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", f"batcher run.*{root}"], text=True
        ).strip().splitlines()
        daemon_pid = int(out[0]) if out else None
    except Exception:
        daemon_pid = None

ok_bool = (ok_str == "true")

state = {
    "armed":               d.get("armed", False),
    "daemon_pid":          daemon_pid,
    "halt":                d.get("halt"),
    "next_batch":          d.get("next_batch", 0),
    "journal_rows":        d.get("journal_rows", 0),
    "rows_toward_next_slice": d.get("rows_toward_next_slice", 0),
    "batches_ready":       d.get("batches_ready", 0),
    "frozen":              d.get("frozen", []),
    "ok":                  ok_bool,
    "note":                note,
}
print("STATE " + json.dumps(state))
PYEOF2
}

# ── status ─────────────────────────────────────────────────────────────────────

cmd_status() {
  local raw
  raw=$(_gather_state)

  # Parse fields for the human block
  local armed halt next_batch journal_rows rows_toward batches_ready frozen
  armed=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['armed'])" "$raw")
  halt=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['halt'] or '')" "$raw")
  next_batch=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['next_batch'])" "$raw")
  journal_rows=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['journal_rows'])" "$raw")
  rows_toward=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['rows_toward_next_slice'])" "$raw")
  batches_ready=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['batches_ready'])" "$raw")
  frozen=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(', '.join(d['frozen']) or 'none')" "$raw")

  local pid
  pid=$(_daemon_pid)

  local armed_str="NO"
  [[ "$armed" == "True" ]] && armed_str="YES"

  local pid_str="${pid:-none}"
  local status_updated="(not present)"
  if [[ -f "${ROOT}/STATUS.md" ]]; then
    status_updated=$(python3 -c "
import os, datetime
t = os.path.getmtime('${ROOT}/STATUS.md')
print(datetime.datetime.utcfromtimestamp(t).strftime('%Y-%m-%dT%H:%M:%SZ'))
" 2>/dev/null || echo "unknown")
  fi

  echo "── Batcher Status ────────────────────────────────────────"
  echo "  ARMED:              ${armed_str}"
  echo "  Daemon PID:         ${pid_str}"
  if [[ -n "${halt}" ]]; then
    echo "  HALT:               ${halt}"
  fi
  echo "  Next batch #:       ${next_batch}"
  echo "  Journal rows:       ${journal_rows}"
  echo "  Rows toward slice:  ${rows_toward}  (need 250 to cut)"
  echo "  Batches READY:      ${batches_ready}"
  echo "  Frozen batches:     ${frozen}"
  echo "  STATUS.md updated:  ${status_updated}"
  echo "──────────────────────────────────────────────────────────"

  # Check if extraction appears down
  local run_dir_progress="${RUN_DIR}/_progress"
  local extraction_note=""
  if [[ -f "${run_dir_progress}/INCOMPLETE" ]]; then
    # Look for an allocation run process for this run dir
    local alloc_pid
    alloc_pid=$(pgrep -f "allocation.*run.*${RUN_DIR}" 2>/dev/null | head -1 || true)
    if [[ -z "$alloc_pid" ]]; then
      extraction_note="extraction not running — resuming it is Nicky-gated; the batcher will wait, this is not an error"
      echo "  NOTE: ${extraction_note}"
    fi
  fi

  _build_state_line "true" "${extraction_note:-ok}" "${pid:-}"
}

# ── arm ────────────────────────────────────────────────────────────────────────

cmd_arm() {
  local note=""

  # Check ARMED
  local is_armed=false
  [[ -f "${ROOT}/ARMED" ]] && is_armed=true

  # Check daemon
  local pid
  pid=$(_daemon_pid)

  if ${is_armed} && [[ -n "${pid}" ]]; then
    echo "Batcher already armed and running (PID ${pid}). No-op."
    note="already armed and running"
    _build_state_line "true" "$note" "$pid"
    return 0
  fi

  # Arm if needed
  if ! ${is_armed}; then
    echo "Arming batcher..."
    # Standing approval given by Nicky "Arm it" 2026-07-07, ~$2.30/batch Sonnet
    if ! "${BIN}" batcher arm --root "${ROOT}" --i-approve-recurring-spend; then
      echo "ERROR: arm command failed. Report to Nicky."
      _build_state_line "false" "arm command failed — report to Nicky" ""
      return 1
    fi
    echo "ARMED file written."
  fi

  # Launch daemon if not running
  if [[ -z "${pid}" ]]; then
    echo "Launching batcher daemon..."
    local watch_args=()
    if [[ -n "${WATCH}" ]]; then
      watch_args=(--watch-journal "${WATCH}")
    fi

    nohup "${BIN}" batcher run \
      --root        "${ROOT}" \
      --journal     "${JOURNAL}" \
      --run-dir     "${RUN_DIR}" \
      --campaign-source "${CAMPAIGN}" \
      "${watch_args[@]}" \
      >> "${LOG}" 2>&1 &

    echo "Waiting for daemon to initialise (10s)..."
    sleep 10

    pid=$(_daemon_pid)
    if [[ -z "${pid}" ]]; then
      echo "ERROR: daemon did not start. Last 20 lines of log:"
      tail -20 "${LOG}" 2>/dev/null || echo "(log not found)"
      _build_state_line "false" "daemon failed to start — report to Nicky" ""
      return 1
    fi

    if [[ ! -f "${ROOT}/daemon.lock" ]]; then
      echo "ERROR: daemon.lock not present after 10s (PID ${pid} seen). Last 20 lines of log:"
      tail -20 "${LOG}" 2>/dev/null || echo "(log not found)"
      _build_state_line "false" "daemon.lock absent — report to Nicky" "$pid"
      return 1
    fi

    note="daemon launched, pid ${pid}"
    echo "Daemon running (PID ${pid}), lock confirmed."
  else
    note="armed (daemon was already running, pid ${pid})"
    echo "Daemon already running (PID ${pid})."
  fi

  _build_state_line "true" "$note" "$pid"
}

# ── disarm ─────────────────────────────────────────────────────────────────────

cmd_disarm() {
  # Idempotent: if already disarmed and no daemon, report ok.
  local pid
  pid=$(_daemon_pid)

  if [[ ! -f "${ROOT}/ARMED" ]] && [[ -z "${pid}" ]]; then
    echo "Batcher is already disarmed and no daemon running. No-op."
    _build_state_line "true" "already disarmed, no daemon" ""
    return 0
  fi

  if [[ -f "${ROOT}/ARMED" ]]; then
    echo "Running: ${BIN} batcher disarm --root ${ROOT}"
    if ! "${BIN}" batcher disarm --root "${ROOT}"; then
      echo "ERROR: disarm command failed. Report to Nicky."
      _build_state_line "false" "disarm command failed — report to Nicky" "$pid"
      return 1
    fi
    echo "ARMED flag removed. Daemon will exit at next tick boundary."
  else
    echo "ARMED flag already absent; waiting for daemon to exit..."
  fi

  if [[ -z "$pid" ]]; then
    echo "No daemon running."
    _build_state_line "true" "disarmed, no daemon was running" ""
    return 0
  fi

  # Poll up to 150s for daemon exit
  local elapsed=0
  local interval=5
  echo "Polling for daemon (PID ${pid}) to exit (up to 150s)..."
  while [[ $elapsed -lt 150 ]]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Daemon exited after ${elapsed}s."
      _build_state_line "true" "disarmed and daemon exited" ""
      return 0
    fi
    sleep $interval
    elapsed=$((elapsed + interval))
    echo "  ...${elapsed}s elapsed, still running"
  done

  echo "WARNING: daemon is finishing its current stage — safe; re-run batcherctl.sh status later;"
  echo "  ARMED flag is already removed so it will stop at the next boundary."
  echo "  Do NOT attempt to kill the process — it is finishing a safe in-flight stage."
  _build_state_line "false" "daemon still running after 150s — ARMED removed, will stop at next boundary; re-run status" "$pid"
  return 1
}

# ── tail ───────────────────────────────────────────────────────────────────────

cmd_tail() {
  echo "── Last 25 lines of STATUS.md ────────────────────────────"
  if [[ -f "${ROOT}/STATUS.md" ]]; then
    tail -25 "${ROOT}/STATUS.md"
  else
    echo "(STATUS.md not found)"
  fi

  echo ""
  echo "── Last 5 events (events.jsonl) ──────────────────────────"
  if [[ -f "${ROOT}/events.jsonl" ]]; then
    tail -5 "${ROOT}/events.jsonl"
  else
    echo "(events.jsonl not found)"
  fi

  echo ""
  echo "── Last 10 lines of daemon.log ───────────────────────────"
  if [[ -f "${LOG}" ]]; then
    tail -10 "${LOG}"
  else
    echo "(daemon.log not found at ${LOG})"
  fi
  echo "──────────────────────────────────────────────────────────"

  local pid
  pid=$(_daemon_pid)
  _build_state_line "true" "tail complete" "${pid:-}"
}

# ── clear-halt ────────────────────────────────────────────────────────────────

cmd_clear_halt() {
  local reason="${1:-}"
  if [[ -z "$reason" ]]; then
    echo "ERROR: clear-halt requires a non-empty reason argument."
    echo "Usage: batcherctl.sh clear-halt \"<reason text>\""
    local pid
    pid=$(_daemon_pid)
    _build_state_line "false" "missing reason argument" "${pid:-}"
    return 1
  fi

  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  WARNING: clear-halt                                     ║"
  echo "║  Only run this if Nicky instructed.                      ║"
  echo "║  A halt means the system stopped itself for safety.      ║"
  echo "╚══════════════════════════════════════════════════════════╝"

  echo "Running: ${BIN} batcher clear-halt --root ${ROOT} --reason \"${reason}\""
  if ! "${BIN}" batcher clear-halt --root "${ROOT}" --reason "${reason}"; then
    echo "ERROR: clear-halt command failed. Report to Nicky."
    local pid
    pid=$(_daemon_pid)
    _build_state_line "false" "clear-halt command failed — report to Nicky" "${pid:-}"
    return 1
  fi

  local pid
  pid=$(_daemon_pid)
  _build_state_line "true" "halt cleared: ${reason}" "${pid:-}"
}

# ── help ───────────────────────────────────────────────────────────────────────

cmd_help() {
  cat <<'HELP'
batcherctl.sh — bulk-batcher operator control (one word per action)

COMMANDS
  status               Show batcher state (default when no arg given).
  arm                  Idempotently ensure batcher is armed and daemon running.
  disarm               Remove ARMED flag; wait for daemon to exit gracefully.
  tail                 Show last STATUS.md lines, recent events, daemon log.
  clear-halt <reason>  Clear a safety halt (ONLY when Nicky instructs).
  help                 This message.

RULES FOR AGENTS
  1. Always check exit code: 0 = success, non-zero = something to report.
  2. Always read the final STATE line (JSON). Keys:
       armed            true/false — ARMED flag present
       daemon_pid       integer or null — live daemon pid
       halt             string or null — halt reason if active
       next_batch       integer — next batch number to be cut
       journal_rows     integer — total lines in primary journal
       rows_toward_next_slice  integer — rows still needed before next slice
       batches_ready    integer — batches with READY_TO_FOLD flag
       frozen           list of batch names stuck in a frozen state
       ok               true/false — did this command succeed?
       note             human-readable summary of what happened

DECISION TABLE
  Want it running?       → arm
  Want it stopped?       → disarm
  Just checking?         → status
  Something looks wrong? → tail, then REPORT TO NICKY
  Got a halt notice?     → tail, then REPORT TO NICKY (do NOT clear-halt on your own)
  Nicky said clear-halt? → clear-halt "<exact reason Nicky gave>"

OUT OF SCOPE (this script never does these)
  - Fold decisions (READY_TO_FOLD batches are for Nicky to fold manually)
  - Resuming extraction (Nicky-gated)
  - Backfill, cascade, pass@k — all handled by the daemon itself
HELP

  local pid
  pid=$(_daemon_pid)
  _build_state_line "true" "help displayed" "${pid:-}"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

CMD="${1:-status}"
shift || true   # shift off $1; if no args, shift fails silently — that's fine

case "$CMD" in
  status)       cmd_status ;;
  arm)          cmd_arm ;;
  disarm)       cmd_disarm ;;
  tail)         cmd_tail ;;
  clear-halt)   cmd_clear_halt "${1:-}" ;;
  help|--help|-h) cmd_help ;;
  *)
    echo "ERROR: unknown subcommand '${CMD}'"
    echo ""
    cmd_help
    exit 1
    ;;
esac

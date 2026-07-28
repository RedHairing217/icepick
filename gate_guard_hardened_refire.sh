#!/bin/bash
# gate_guard_hardened_refire.sh — uncommitted one-off (session 89fe6f6f, mission
# extractor-hardening). Nicky 2026-07-11: "stop it then refire on the hardened QA
# flow." Baseline 20260711T202559Z (stale imports) was stopped; this launches the
# HARDENED, 250-paper-capped refire behind an arming guarantee: the loaded
# extractor is re-verified, and the run is KILLED if its first mined candidate
# does not carry source_statement_raw (the stale-baseline failure mode).
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
R=out/intake/runs/20260711T234953Z
LOG=$R/gate.log
mkdir -p "$R"

echo "$(date -u +%FT%TZ) refire gate: arming pre-check" >> "$LOG"
/opt/anaconda3/bin/python3 - >> "$LOG" 2>&1 <<'PY'
import inspect, sys
import icepick.allocation.scrape.realmath as rm
import icepick.allocation.adapters.realmath_scrape as rs
ok = ("source_statement_raw" in inspect.getsource(rm)
      and "_label_content_index" in inspect.getsource(rm)
      and "qa_ref_guard" in inspect.getsource(rs))
print(f"arming pre-check: hardened={ok}")
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "$(date -u +%FT%TZ) ABORT: hardened code not loaded" >> "$LOG"; exit 1; }

# arXiv one-worker rule: refuse if another scrape is live
if pgrep -f "icepick allocation run" >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ) ABORT: another 'allocation run' is live (invariant 6)" >> "$LOG"; exit 1
fi

ANTHROPIC_KEY_FILE=/Users/redhairing/Desktop/helloworld/anthro_key.env \
  nohup /opt/anaconda3/bin/icepick allocation run --manifest "$R/manifest.json" >> "$R/launch.log" 2>&1 &
RUNPID=$!
echo "$(date -u +%FT%TZ) hardened refire PID $RUNPID" >> "$LOG"

for i in $(seq 1 80); do
  sleep 30
  if ! kill -0 $RUNPID 2>/dev/null; then
    echo "$(date -u +%FT%TZ) run exited early (i=$i) — see launch.log" >> "$LOG"; exit 1
  fi
  if [ -s "$R/_progress/candidates.jsonl" ]; then
    if head -1 "$R/_progress/candidates.jsonl" | grep -q "source_statement_raw"; then
      echo "$(date -u +%FT%TZ) VERIFIED ARMED: first candidate carries source_statement_raw (PID $RUNPID)" >> "$LOG"; exit 0
    else
      echo "$(date -u +%FT%TZ) ARMING FAILURE: first candidate lacks source_statement_raw — KILLING $RUNPID" >> "$LOG"
      kill $RUNPID; exit 1
    fi
  fi
done
echo "$(date -u +%FT%TZ) no candidates after 40min — leaving alive, verify manually" >> "$LOG"; exit 0

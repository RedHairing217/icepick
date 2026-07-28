#!/bin/bash
# gate_guard_hardened_feb.sh — uncommitted (session 89fe6f6f). "Begin another
# hardened extraction" (Nicky 2026-07-12): math.AP 2026-02, 250-cap, arming-gated.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
R=out/intake/runs/20260712T044950Z
LOG=$R/gate.log; mkdir -p "$R"
echo "$(date -u +%FT%TZ) arming pre-check" >> "$LOG"
/opt/anaconda3/bin/python3 - >> "$LOG" 2>&1 <<'PY'
import inspect, sys
import icepick.allocation.scrape.realmath as rm
import icepick.allocation.adapters.realmath_scrape as rs
ok=("source_statement_raw" in inspect.getsource(rm) and "_label_content_index" in inspect.getsource(rm)
    and "qa_ref_guard" in inspect.getsource(rs))
print(f"arming: hardened={ok}"); sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "$(date -u +%FT%TZ) ABORT: hardened code not loaded" >> "$LOG"; exit 1; }
if pgrep -f "icepick allocation run" >/dev/null 2>&1; then echo "$(date -u +%FT%TZ) ABORT: a scrape is live (inv 6)" >> "$LOG"; exit 1; fi
ANTHROPIC_KEY_FILE=/Users/redhairing/Desktop/helloworld/anthro_key.env \
  nohup /opt/anaconda3/bin/icepick allocation run --manifest "$R/manifest.json" >> "$R/launch.log" 2>&1 &
RUNPID=$!; echo "$(date -u +%FT%TZ) run PID $RUNPID" >> "$LOG"
for i in $(seq 1 80); do
  sleep 30
  kill -0 $RUNPID 2>/dev/null || { echo "$(date -u +%FT%TZ) exited early i=$i" >> "$LOG"; exit 1; }
  if [ -s "$R/_progress/candidates.jsonl" ]; then
    if head -1 "$R/_progress/candidates.jsonl" | grep -q "source_statement_raw"; then
      echo "$(date -u +%FT%TZ) VERIFIED ARMED (PID $RUNPID)" >> "$LOG"; exit 0
    else
      echo "$(date -u +%FT%TZ) ARMING FAILURE — KILLING $RUNPID" >> "$LOG"; kill $RUNPID; exit 1
    fi
  fi
done
echo "$(date -u +%FT%TZ) no candidates after 40min — verify manually" >> "$LOG"; exit 0

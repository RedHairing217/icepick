#!/bin/bash
# gate_guard_analysis_paired.sh — uncommitted one-off (session 89fe6f6f, mission
# extractor-hardening). Launches the GUARDED arm of the paired guard-analysis
# experiment per Nicky 2026-07-11: "Launch it / Do not launch without arming
# corrected code."
#
# Sequence: wait for the baseline scrape (PID 81011-era run 20260711T202559Z) to
# exit (arXiv invariant 6: one sequential worker per IP) -> ARMING TRIPWIRE:
# assert the python that will run the scrape LOADS the hardened extractor
# (source_statement_raw + _label_content_index in realmath, qa_ref_guard in
# realmath_scrape) -> launch run 20260711T225119Z -> verify the FIRST mined
# candidate actually carries source_statement_raw; kill the run if not.
# The baseline arm ran stale imports (parallel-session stash window suspected);
# this script exists so that can't recur silently.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
R=out/intake/runs/20260711T225119Z
LOG=$R/gate.log
mkdir -p "$R"
echo "$(date -u +%FT%TZ) gate start: waiting for baseline PID 81111" >> "$LOG"

while kill -0 81111 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) baseline exited; running arming tripwire" >> "$LOG"

/opt/anaconda3/bin/python3 - >> "$LOG" 2>&1 <<'PY'
import inspect, sys
import icepick.allocation.scrape.realmath as rm
import icepick.allocation.adapters.realmath_scrape as rs
rm_src = inspect.getsource(rm)
rs_src = inspect.getsource(rs)
ok = ("source_statement_raw" in rm_src and "_label_content_index" in rm_src
      and "qa_ref_guard" in rs_src and "elision_signals" in rs_src)
print(f"tripwire: hardened-code loaded = {ok}")
sys.exit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then
  echo "$(date -u +%FT%TZ) TRIPWIRE FAILED: loaded extractor is NOT the hardened code — ABORTING (parallel session stash/checkout suspected; re-run this gate when the tree is restored)" >> "$LOG"
  exit 1
fi

echo "$(date -u +%FT%TZ) tripwire passed; launching guarded arm" >> "$LOG"
ANTHROPIC_KEY_FILE=/Users/redhairing/Desktop/helloworld/anthro_key.env \
  nohup /opt/anaconda3/bin/icepick allocation run --manifest "$R/manifest.json" >> "$R/launch.log" 2>&1 &
RUNPID=$!
echo "$(date -u +%FT%TZ) guarded arm PID $RUNPID" >> "$LOG"

# Post-launch verification: first candidate must carry the new field.
for i in $(seq 1 60); do
  sleep 30
  if ! kill -0 $RUNPID 2>/dev/null; then
    echo "$(date -u +%FT%TZ) run process exited early (i=$i) — see launch.log" >> "$LOG"
    exit 1
  fi
  if [ -s "$R/_progress/candidates.jsonl" ]; then
    if head -1 "$R/_progress/candidates.jsonl" | grep -q "source_statement_raw"; then
      echo "$(date -u +%FT%TZ) VERIFIED: first candidate carries source_statement_raw — guarded arm confirmed armed (PID $RUNPID)" >> "$LOG"
      exit 0
    else
      echo "$(date -u +%FT%TZ) ARMING FAILURE: first candidate LACKS source_statement_raw — killing run $RUNPID; do not re-launch without resolving the code state" >> "$LOG"
      kill $RUNPID
      exit 1
    fi
  fi
done
echo "$(date -u +%FT%TZ) no candidates after 30min; leaving run alive (slow month or listing-heavy phase) — verify manually" >> "$LOG"
exit 0

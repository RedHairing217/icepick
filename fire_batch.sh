#!/bin/bash
# fire_batch.sh YEAR MONTH [CATEGORY]   — THE ONLY HUMAN STEP.
# Fully autonomous hardened funnel: plan -> approve -> arming-gated hardened
# extraction (250-cap) -> cascade -> pass@k -> HELD FOR FOLD. Fold stays manual.
# e.g.:  ./fire_batch.sh 2026 3          (math.AP March 2026)
#        ./fire_batch.sh 2026 3 math.NT  (different category)
# Runs in the foreground; background it with:  nohup ./fire_batch.sh 2026 3 &>fire_2026_3.log &
set -u
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
Y="$1"; M="$2"; CAT="${3:-math.AP}"
KEY=/Users/redhairing/Desktop/helloworld/anthro_key.env
SRC="pde_auto_${Y}_$(printf '%02d' "$M")_$(echo "$CAT" | tr '.' '_')"

echo "[fire_batch] $SRC — plan"
PJSON=$(/opt/anaconda3/bin/icepick allocation plan --source-type realmath_scrape --source "$SRC" \
  --target-count 500 --max-papers 250 --category "$CAT" --primary-only --extraction qa \
  --max-per-paper 3 --year "$Y" --month "$M" --family pde --output-dir out/intake 2>&1)
PLAN=$(echo "$PJSON" | grep -o '"plan_path": "[^"]*"' | cut -d'"' -f4)
[ -z "$PLAN" ] && { echo "PLAN FAILED: $PJSON"; exit 1; }

echo "[fire_batch] approve"
AJSON=$(/opt/anaconda3/bin/icepick allocation approve --plan "$PLAN" --mode production \
  --approved-by nicky-autopilot --call-budget 42060 --output-dir out/intake \
  --approval-notes "autopilot batch $SRC (fire_batch.sh)" 2>&1)
MAN=$(echo "$AJSON" | grep -o '"manifest": "[^"]*"' | cut -d'"' -f4)
[ -z "$MAN" ] && { echo "APPROVE FAILED: $AJSON"; exit 1; }
RUN=$(dirname "$MAN"); RID=$(basename "$RUN")
echo "[fire_batch] run $RID"

# ARMING pre-check: interpreter must load the hardened extractor.
/opt/anaconda3/bin/python3 -c "import inspect,sys,icepick.allocation.scrape.realmath as rm,icepick.allocation.adapters.realmath_scrape as rs; sys.exit(0 if ('source_statement_raw' in inspect.getsource(rm) and '_label_content_index' in inspect.getsource(rm) and 'qa_ref_guard' in inspect.getsource(rs)) else 1)" \
  || { echo "[fire_batch] ABORT: hardened code not loaded"; exit 1; }

# inv 6: never two arXiv scrapes at once.
while pgrep -f "allocation ru[n]" >/dev/null 2>&1; do echo "[fire_batch] waiting for live scrape to finish (inv 6)"; sleep 60; done

echo "[fire_batch] extract"
ANTHROPIC_KEY_FILE=$KEY nohup /opt/anaconda3/bin/icepick allocation run --manifest "$MAN" >> "$RUN/launch.log" 2>&1 &
XPID=$!

# post-launch ARMING verification: first candidate must carry source_statement_raw.
sleep 60
if [ -s "$RUN/_progress/candidates.jsonl" ]; then
  head -1 "$RUN/_progress/candidates.jsonl" | grep -q source_statement_raw \
    || { echo "[fire_batch] ARMING FAILURE — killing $XPID"; kill "$XPID"; exit 1; }
  echo "[fire_batch] armed-verified"
fi

echo "[fire_batch] extracting (PID $XPID) — will funnel on completion"
while kill -0 "$XPID" 2>/dev/null; do sleep 60; done
[ -s "$RUN/handoff/records.jsonl" ] || { echo "[fire_batch] no handoff — nothing extracted"; exit 1; }

echo "[fire_batch] extraction done -> funnel_chain (cascade -> pass@k)"
./funnel_chain.sh "$SRC" "$RUN/handoff/records.jsonl" "out/auto_funnel_$RID" cascade
echo "[fire_batch] $SRC complete — labeled output HELD FOR FOLD in out/auto_funnel_$RID/"

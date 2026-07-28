#!/bin/bash
# gate_batch14_fold.sh — uncommitted, one-off. Nicky: "fold batch 14 when pass@k done".
# Waits for batch 14 pass@k to COMPLETE (manifest written + no live pass@k proc for this
# run), then runs the adaptive fold. Fold is idempotent + guarded (see the .py). nohup me.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
RD=out/processing_20260709T062552Z
LOG=$RD/fold_gate.log
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) fold gate armed — waiting for batch14 pass@k to finish" >> "$LOG"

# wait until pass@k has written its final manifest AND no pass@k process for this run is alive
while true; do
  if [ -f "$RD/pass_at_k/pass_at_k_manifest.json" ] && \
     ! pgrep -f "processing pass_at_k.*20260709T062552Z" > /dev/null 2>&1; then
    break
  fi
  sleep 30
done
sleep 5   # let final fsync settle
N=$(grep -c . "$RD/pass_at_k/pass_at_k.jsonl" 2>/dev/null || echo 0)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pass@k complete ($N labeled records) — running adaptive fold" >> "$LOG"

/opt/anaconda3/bin/python merge_batch14_adaptive.py >> "$LOG" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) fold script exited $?" >> "$LOG"

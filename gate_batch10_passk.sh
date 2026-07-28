#!/bin/bash
# gate_batch10_passk.sh — uncommitted, one-off queue gate.
# Fires pass@k over batch 10's 113 cascade well-posed records
# (out/processing_20260707T233518Z, Sonnet-only codex:anthropic cascade,
# Nicky "begin batch 10 cascade"). Queued behind the live Qwen consumer
# per the single-Qwen-slot invariant (AGENTS.md #9). Wire params
# byte-identical to the gate convention (invariant #2) — do not edit.
# pass@k ONLY — corpus fold stays Nicky-gated, do NOT auto-fold.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/processing_20260707T233518Z/pass_at_k_gate.log
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate armed (batch10, 113 records), waiting for Qwen slot" >> "$LOG"

# Wait for the slot, then jitter + re-verify to avoid a thundering-herd
# race with other pgrep-only gates that break in the same tick.
while true; do
  while pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; do
    sleep 30
  done
  # slot looks free — stagger, then confirm nobody grabbed it during jitter
  sleep $(( (RANDOM % 20) + 5 ))
  if ! pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; then
    break
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another consumer took the slot during jitter, re-queuing" >> "$LOG"
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Qwen slot free, launching batch10 pass@k (113 records)" >> "$LOG"
/opt/anaconda3/bin/python /opt/anaconda3/bin/icepick processing pass_at_k \
  --mode production \
  --input out/processing_20260707T233518Z/cascade/final_corpus.jsonl \
  --output-dir out/processing_20260707T233518Z/pass_at_k \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 \
  >> "$LOG" 2>&1
RC=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) batch10 pass@k exited $RC" >> "$LOG"
if [ $RC -eq 0 ] && [ -f out/processing_20260707T233518Z/pass_at_k/pass_at_k.jsonl ]; then
  N=$(grep -c . out/processing_20260707T233518Z/pass_at_k/pass_at_k.jsonl)
  B=$(grep -c '"label": "band"' out/processing_20260707T233518Z/pass_at_k/pass_at_k.jsonl)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) READY_TO_FOLD (NOT auto-folded): $N labeled, ~$B band" >> "$LOG"
fi

#!/bin/bash
# gate_batch14_passk.sh — uncommitted, one-off queue gate.
# Fires pass@k over batch 14's 111 cascade well-posed records
# (out/processing_20260709T062552Z, Sonnet-only codex:anthropic cascade,
# realmath math.AP 2025-12). Slot free at arm time but gated anyway for
# safety against parallel Qwen consumers (single-Qwen invariant, AGENTS.md #9).
# Wire params byte-identical to the gate convention (invariant #2).
# pass@k ONLY — corpus fold stays Nicky-gated, do NOT auto-fold.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/processing_20260709T062552Z/pass_at_k_gate.log
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate armed (batch14, 111 records), checking Qwen slot" >> "$LOG"

while true; do
  while pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; do
    sleep 30
  done
  sleep $(( (RANDOM % 20) + 5 ))
  if ! pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; then
    break
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another consumer took the slot during jitter, re-queuing" >> "$LOG"
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Qwen slot free, launching batch14 pass@k (111 records)" >> "$LOG"
/opt/anaconda3/bin/python /opt/anaconda3/bin/icepick processing pass_at_k \
  --mode production \
  --input out/processing_20260709T062552Z/cascade/final_corpus.jsonl \
  --output-dir out/processing_20260709T062552Z/pass_at_k \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 \
  >> "$LOG" 2>&1
RC=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) batch14 pass@k exited $RC" >> "$LOG"
if [ $RC -eq 0 ] && [ -f out/processing_20260709T062552Z/pass_at_k/pass_at_k.jsonl ]; then
  N=$(grep -c . out/processing_20260709T062552Z/pass_at_k/pass_at_k.jsonl)
  B=$(grep -c '"label": "band"' out/processing_20260709T062552Z/pass_at_k/pass_at_k.jsonl)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) READY_TO_FOLD (NOT auto-folded): $N labeled, ~$B band" >> "$LOG"
fi

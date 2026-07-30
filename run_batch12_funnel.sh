#!/bin/bash
# run_batch12_funnel.sh — uncommitted, one-off funnel chain.
# Batch12 (268 handoff records) into the funnel per Nicky 2026-07-08 "move
# into funnel". SONNET-ONLY cascade (gpt-5.5 removed from cascade per Nicky
# 2026-07-08; matches batcher-batch11 convention), metered $3/$15 (~$2.4
# expected, under the $5 line), then pass@k on local Qwen ($0) once the slot
# is free (one-Qwen rule). Cascade failure => log + exit, no pass@k (cascade
# is judge-cached; re-run is cheap).
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
PROC=out/processing_20260708T041100Z
LOG=$PROC/funnel.log
mkdir -p "$PROC"
echo "$(date -u +%FT%TZ) batch12 funnel start: 268 records, sonnet-only cascade" >> "$LOG"

/opt/anaconda3/bin/icepick processing wellposed-cascade \
  --mode production \
  --stages codex:anthropic \
  --input out/intake/runs/20260708T041100Z/handoff/records.jsonl \
  --output-dir $PROC/cascade \
  --anthro-key-file /Users/redhairing/Desktop/helloworld/anthro_key.env \
  --cost-per-input-mtok 3 --cost-per-output-mtok 15 >> "$LOG" 2>&1
RC=$?

FINAL=$PROC/cascade/final_corpus.jsonl
if [ $RC -ne 0 ] || [ ! -s "$FINAL" ]; then
  echo "$(date -u +%FT%TZ) cascade FAILED (rc=$RC, final_corpus missing/empty) — NOT running pass@k. Judge cache preserved; re-run this script to resume." >> "$LOG"
  exit 1
fi
NWP=$(grep -c . "$FINAL")
echo "$(date -u +%FT%TZ) cascade complete: $NWP well-posed. Waiting for free Qwen slot." >> "$LOG"

while pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; do sleep 30; done
echo "$(date -u +%FT%TZ) Qwen slot free, launching pass@k ($NWP records)" >> "$LOG"

/opt/anaconda3/bin/icepick processing pass_at_k \
  --mode production \
  --input "$FINAL" \
  --output-dir $PROC/pass_at_k \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) pass@k exited $? — batch12 funnel done (labels in $PROC/pass_at_k/)" >> "$LOG"

#!/bin/bash
# gate_fksweeprescue5_passk.sh — uncommitted, one-off queue gate.
# 18 OPUS-tranche-B2 (chunks 28-32, incl. the med-tier splits) panel-confirmed
# false kills (standing Nicky release + auto-fold, no-k12 protocol). Waits for
# a free Qwen slot (currently held by a parallel auto-batcher June pass@k) per
# invariant #9. Gate script on disk — argv does NOT contain the pgrep pattern,
# so it never self-matches (cf. the 03:13Z watcher-deadlock lesson). Wire
# params byte-identical (invariant #2).
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/stage1_kill_census/fk_sweep/rescue5_pass_at_k/gate.log
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate armed (18-record opus rescue5), waiting on Qwen slot (parallel auto-batcher holds it)" >> "$LOG"
while pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; do
  sleep 30
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Qwen slot free, launching opus rescue5 pass@k (18 records)" >> "$LOG"
/opt/anaconda3/bin/python /opt/anaconda3/bin/icepick processing pass_at_k \
  --mode production \
  --input out/stage1_kill_census/fk_sweep/rescue5_pass_at_k/rescue_input.jsonl \
  --output-dir out/stage1_kill_census/fk_sweep/rescue5_pass_at_k/pass_at_k \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 \
  >> "$LOG" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) opus rescue5 pass@k exited $?" >> "$LOG"

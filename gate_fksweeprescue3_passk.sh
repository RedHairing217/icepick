#!/bin/bash
# gate_fksweeprescue3_passk.sh — uncommitted, one-off queue gate.
# Launches pass@k over the 11 OPUS-tranche (chunks 20-23) panel-confirmed
# false kills (standing Nicky release + auto-fold, 2026-07-07; opus
# calibration gate passed 6/6). SERIALIZED behind rescue2: waits for
# rescue2's 45/45 completion AND a free Qwen slot before firing — two
# slot-polling gates alone could double-fire in the same window
# (AGENTS.md invariant #9). Wire params byte-identical (invariant #2).
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/stage1_kill_census/fk_sweep/rescue3_pass_at_k/gate.log
R2DONE=out/stage1_kill_census/fk_sweep/rescue2_pass_at_k/pass_at_k/_progress/records_done.jsonl
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate armed (11-record opus rescue3), serialized behind rescue2 completion" >> "$LOG"
while true; do
  N=$(wc -l < "$R2DONE" 2>/dev/null | tr -d ' ')
  if [ "${N:-0}" = "45" ] && ! pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; then
    break
  fi
  sleep 30
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) rescue2 complete + slot free, launching opus rescue3 pass@k (11 records)" >> "$LOG"
/opt/anaconda3/bin/python /opt/anaconda3/bin/icepick processing pass_at_k \
  --mode production \
  --input out/stage1_kill_census/fk_sweep/rescue3_pass_at_k/rescue_input.jsonl \
  --output-dir out/stage1_kill_census/fk_sweep/rescue3_pass_at_k/pass_at_k \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 \
  >> "$LOG" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) opus rescue3 pass@k exited $?" >> "$LOG"

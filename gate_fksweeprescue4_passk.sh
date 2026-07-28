#!/bin/bash
# gate_fksweeprescue4_passk.sh — uncommitted, one-off queue gate.
# 19 OPUS-tranche-B1 (chunks 24-27) panel-confirmed false kills (standing
# Nicky release + auto-fold). SERIALIZED behind rescue3 (11/11 completion)
# AND a free Qwen slot — race-safe chain per invariant #9. Wire params
# byte-identical (invariant #2).
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/stage1_kill_census/fk_sweep/rescue4_pass_at_k/gate.log
R3DONE=out/stage1_kill_census/fk_sweep/rescue3_pass_at_k/pass_at_k/_progress/records_done.jsonl
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate armed (19-record opus rescue4), serialized behind rescue3 completion" >> "$LOG"
while true; do
  N=$(wc -l < "$R3DONE" 2>/dev/null | tr -d ' ')
  if [ "${N:-0}" = "11" ] && ! pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; then
    break
  fi
  sleep 30
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) rescue3 complete + slot free, launching opus rescue4 pass@k (19 records)" >> "$LOG"
/opt/anaconda3/bin/python /opt/anaconda3/bin/icepick processing pass_at_k \
  --mode production \
  --input out/stage1_kill_census/fk_sweep/rescue4_pass_at_k/rescue_input.jsonl \
  --output-dir out/stage1_kill_census/fk_sweep/rescue4_pass_at_k/pass_at_k \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 \
  >> "$LOG" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) opus rescue4 pass@k exited $?" >> "$LOG"

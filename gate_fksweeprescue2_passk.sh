#!/bin/bash
# gate_fksweeprescue2_passk.sh — uncommitted, one-off queue gate.
# Launches pass@k over the 45 FK-sweep panel-confirmed false kills from
# chunks 7-19 (Nicky's standing release: "continue sweep" + auto-fold,
# 2026-07-07). Queued behind batch 9's live pass@k (PID 48659, parallel
# session) per the single-Qwen-slot invariant (AGENTS.md #9). Mirrors the
# gate_fksweeprescue_passk.sh convention. Wire params byte-identical
# (invariant #2) — do not edit.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/stage1_kill_census/fk_sweep/rescue2_pass_at_k/gate.log
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate armed (45-record fk_sweep rescue2), queued behind batch-9 pass@k (PID 48659)" >> "$LOG"
while pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; do
  sleep 30
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Qwen slot free, launching fk_sweep rescue2 pass@k (45 records)" >> "$LOG"
/opt/anaconda3/bin/python /opt/anaconda3/bin/icepick processing pass_at_k \
  --mode production \
  --input out/stage1_kill_census/fk_sweep/rescue2_pass_at_k/rescue_input.jsonl \
  --output-dir out/stage1_kill_census/fk_sweep/rescue2_pass_at_k/pass_at_k \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 \
  >> "$LOG" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) fk_sweep rescue2 pass@k exited $?" >> "$LOG"

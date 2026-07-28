#!/bin/bash
# gate_fksweeprescue_passk.sh — uncommitted, one-off queue gate.
# Launches pass@k over the 32 FK-sweep panel-confirmed false kills
# (out/stage1_kill_census/fk_sweep/sweep_rulings.jsonl, chunks 0-6;
# Nicky's explicit release 2026-07-07 ~13:4x PT "push newly discovered
# fk's into pass@k testing"). Mirrors gate_stage1rescue_passk.sh /
# gate_batchN_passk.sh convention. Waits for the single machine-wide Qwen
# slot (AGENTS.md invariant #9) — expected free at launch; loop is race
# protection against parallel sessions.
# Wire params byte-identical to every production pass@k invocation
# (AGENTS.md invariant #2) — do not edit them.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/stage1_kill_census/fk_sweep/rescue_pass_at_k/gate.log
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate armed (32-record fk_sweep rescue), waiting on Qwen slot" >> "$LOG"
while pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; do
  sleep 30
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Qwen slot free, launching fk_sweep rescue pass@k (32 records)" >> "$LOG"
/opt/anaconda3/bin/python /opt/anaconda3/bin/icepick processing pass_at_k \
  --mode production \
  --input out/stage1_kill_census/fk_sweep/rescue_pass_at_k/rescue_input.jsonl \
  --output-dir out/stage1_kill_census/fk_sweep/rescue_pass_at_k/pass_at_k \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 \
  >> "$LOG" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) fk_sweep rescue pass@k exited $?" >> "$LOG"

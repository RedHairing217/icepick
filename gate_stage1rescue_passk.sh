#!/bin/bash
# gate_stage1rescue_passk.sh — uncommitted, one-off queue gate.
# Waits for the single Qwen slot to free (batch 8, PID 90533, is on it now),
# then launches the stage-1 kill census rescue batch: 44 records the census
# panel ruled false_kill (2-3 independent opus rulers, sympy-verified where
# applicable; see out/stage1_kill_census/census_rulings.jsonl). Mirrors the
# gate_batchN_passk.sh convention referenced in docs/SESSION_HANDOFF.md.
# Wire params are byte-identical to every other production pass@k invocation
# in this repo (AGENTS.md invariant #2) — do not edit them.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/stage1_kill_census/rescue_pass_at_k/gate.log
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate armed (queued behind batch 8 / PID 90533), waiting on Qwen slot" >> "$LOG"
while pgrep -f "icepick processing pass_at_k" > /dev/null 2>&1; do
  sleep 30
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Qwen slot free, launching rescue pass@k (44 records)" >> "$LOG"
/opt/anaconda3/bin/python /opt/anaconda3/bin/icepick processing pass_at_k \
  --mode production \
  --input out/stage1_kill_census/rescue_pass_at_k/rescue_input.jsonl \
  --output-dir out/stage1_kill_census/rescue_pass_at_k/pass_at_k \
  --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 \
  >> "$LOG" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) rescue pass@k exited $?" >> "$LOG"

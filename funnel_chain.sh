#!/bin/bash
# funnel_chain.sh LABEL INPUT OUTDIR [START]
#   START=cascade (default): INPUT = handoff records.jsonl -> cascade -> pass@k
#   START=passk            : INPUT = a well-posed final_corpus.jsonl -> pass@k only
# Autopilot stage-chain for a single batch. HELD AT PASS@K: writes labeled output
# and a READY_FOR_FOLD marker; NEVER folds into band_corpus (fold stays manual —
# the v1 cascade is the weak gate, fold-review is the safety net). Restartable:
# cascade + pass@k are checkpoint-native, re-running resumes. Session 89fe6f6f.
set -u
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LABEL="$1"; INPUT="$2"; OUT="$3"; START="${4:-cascade}"
KEY=/Users/redhairing/Desktop/helloworld/anthro_key.env
mkdir -p "$OUT"; LOG="$OUT/funnel_chain.log"
echo "$(date -u +%FT%TZ) [$LABEL] funnel_chain start (from $START)" >> "$LOG"

FINAL="$OUT/cascade/final_corpus.jsonl"
if [ "$START" = "cascade" ]; then
  ANTHROPIC_KEY_FILE=$KEY /opt/anaconda3/bin/icepick processing wellposed-cascade \
    --mode production --stages codex:anthropic --input "$INPUT" \
    --output-dir "$OUT/cascade" --anthro-key-file "$KEY" \
    --cost-per-input-mtok 3 --cost-per-output-mtok 15 >> "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ] || [ ! -s "$FINAL" ]; then
    echo "$(date -u +%FT%TZ) [$LABEL] cascade FAILED rc=$RC / empty final_corpus — STOP, no pass@k (resumable)" >> "$LOG"
    exit 1
  fi
else
  FINAL="$INPUT"
fi
NWP=$(grep -c . "$FINAL")
echo "$(date -u +%FT%TZ) [$LABEL] well-posed=$NWP — awaiting Qwen slot" >> "$LOG"

# Qwen one-slot rule (inv 9). Bracketed pattern so this guard never matches itself.
while pgrep -f "processing pass_at_[k]" >/dev/null 2>&1; do sleep 30; done

# LM Studio reachability: HOLD (don't crash) if the local Qwen backend is down.
if ! curl -sf -m 5 http://127.0.0.1:1234/v1/models >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ) [$LABEL] Qwen backend unreachable — HELD before pass@k; re-run funnel_chain.sh $LABEL $FINAL $OUT passk to resume" >> "$LOG"
  printf "cascade done: %s well-posed. pass@k HELD — LM Studio (127.0.0.1:1234) down.\n" "$NWP" > "$OUT/READY_FOR_PASSK.txt"
  exit 0
fi

echo "$(date -u +%FT%TZ) [$LABEL] Qwen slot free — pass@k on $NWP records" >> "$LOG"
# Wire params byte-identical to invariant 2 (temp 0.7, 2048, /no_think, k=8, one slot).
/opt/anaconda3/bin/icepick processing pass_at_k --mode production --input "$FINAL" \
  --output-dir "$OUT/pass_at_k" --backend qwen_http \
  --backend-url http://127.0.0.1:1234/v1/chat/completions --model qwen/qwen3-8b \
  --k 8 --temperature 0.7 --max-tokens 2048 --think off --max-concurrent 1 >> "$LOG" 2>&1
PRC=$?
printf "FUNNEL COMPLETE [%s]: %s well-posed -> pass@k (rc=%s). Labeled output: %s/pass_at_k/.\nHELD FOR FOLD — folding into band_corpus is MANUAL (v1 gate; human review required).\n" \
  "$LABEL" "$NWP" "$PRC" "$OUT" > "$OUT/READY_FOR_FOLD.txt"
echo "$(date -u +%FT%TZ) [$LABEL] pass@k rc=$PRC — HELD FOR FOLD (no auto-fold)" >> "$LOG"

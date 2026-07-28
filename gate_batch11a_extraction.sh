#!/bin/bash
# gate_batch11a_extraction.sh — uncommitted, one-off queue gate.
# Queues batch11a EXTRACTION behind batch12 (Nicky, 2026-07-08: "Queue batch 13
# extraction behind batch 12" then "name as batch 11a" + "Move window
# backwards"). Named batch11a to avoid the batcher's numbering lane (batcher
# minted batch11, counter at 12). Window moved backwards to math.AP 2025-12
# (2026-01 is thinning: batch10 found only 105 papers).
# Fires ONLY on batch12 clean completion (handoff/records.jsonl non-empty).
# 429-wall / crash / plan failure => BLOCKED marker + log, no fire, no spend.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/gate_batch11a_extraction.log
LOCK=.gate_batch11a.lock
B12_DIR=out/intake/runs/20260708T041100Z
B12_PID=71171
HANDOFF=$B12_DIR/handoff/records.jsonl
YEAR=2025; MONTH=12

if [ -e "$LOCK" ]; then echo "$(date -u +%FT%TZ) lock exists — another batch11a gate live, aborting" >> "$LOG"; exit 1; fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# no-repeat guard: batch11a must not already exist anywhere
if ls out/intake/plans/*batch11a* >/dev/null 2>&1 || grep -ql batch11a out/intake/runs/*/manifest.json 2>/dev/null; then
  echo "$(date -u +%FT%TZ) BLOCKED: batch11a plan/manifest already exists — refusing to duplicate" >> "$LOG"
  touch out/gate_batch11a_BLOCKED; exit 0
fi

echo "$(date -u +%FT%TZ) gate armed: batch11a (math.AP $YEAR-$MONTH) queued behind batch12 ($B12_DIR, pid $B12_PID)" >> "$LOG"

# wait for batch12 terminal state (handoff = complete; proc gone w/o handoff = dead)
STATE=""
while :; do
  if [ -s "$HANDOFF" ]; then STATE=complete; break; fi
  if ! kill -0 "$B12_PID" 2>/dev/null; then
    sleep 15   # let final writes land
    [ -s "$HANDOFF" ] && STATE=complete || STATE=dead
    break
  fi
  sleep 60
done

if [ "$STATE" != "complete" ]; then
  echo "$(date -u +%FT%TZ) BLOCKED: batch12 process gone, handoff missing/empty (429-wall or crash — see $B12_DIR). NOT firing batch11a." >> "$LOG"
  touch out/gate_batch11a_BLOCKED; exit 0
fi

NREC=$(grep -c . "$HANDOFF" 2>/dev/null)
NPAP=$(grep -c . "$B12_DIR/_progress/papers_done.jsonl" 2>/dev/null)
echo "$(date -u +%FT%TZ) batch12 COMPLETE: ${NREC:-?} handoff records, ${NPAP:-?} papers consumed. Building batch11a plan with fire-time excludes." >> "$LOG"

# fire-time excludes: every realmath_scrape intake run dir on disk right now
# (prior runs are all 2026-01 so excludes are inert for 2025-12, but included
# for safety and to future-proof against interim launches)
EXC=(); NEXC=0
for d in out/intake/runs/*/; do
  m="${d}manifest.json"
  if [ -f "$m" ] && grep -q '"source_type": "realmath_scrape"' "$m"; then
    EXC+=(--exclude-from-run "${d%/}"); NEXC=$((NEXC+1))
  fi
done
echo "$(date -u +%FT%TZ) excluding papers from $NEXC prior scrape runs" >> "$LOG"

T0=$(mktemp)
/opt/anaconda3/bin/icepick allocation plan \
  --source-type realmath_scrape --source pde_diverse_qa_500_batch11a \
  --target-count 500 --output-dir out/intake \
  --requested-by nicky-delegated-claude \
  --category math.AP --year $YEAR --month $MONTH --max-papers 250 --primary-only \
  --extraction qa "${EXC[@]}" \
  --auto-approve --mode production --approved-by nicky-delegated-claude \
  --call-budget 42060 \
  --approval-notes "batch11a queued behind batch12 per Nicky 2026-07-08 (renamed from batch13; window moved back to 2025-12); excludes computed at fire time ($NEXC runs)" >> "$LOG" 2>&1

MANIFEST=$(find out/intake -name "manifest.json" -newer "$T0" 2>/dev/null | grep -v "$B12_DIR" | head -1)
[ -z "$MANIFEST" ] && MANIFEST=$(find out/intake/plans -name "*batch11a*manifest*.json" -newer "$T0" 2>/dev/null | head -1)
rm -f "$T0"
if [ -z "$MANIFEST" ]; then
  echo "$(date -u +%FT%TZ) BLOCKED: plan/approve produced no manifest (see plan output above). NOT firing." >> "$LOG"
  touch out/gate_batch11a_BLOCKED; exit 0
fi

echo "$(date -u +%FT%TZ) launching batch11a extraction: $MANIFEST" >> "$LOG"
ANTHROPIC_KEY_FILE=/Users/redhairing/Desktop/helloworld/anthro_key.env \
ICEPICK_ARXIV_MIN_INTERVAL=6 \
nohup /opt/anaconda3/bin/icepick allocation run --manifest "$MANIFEST" >> "$LOG" 2>&1 &
echo "$(date -u +%FT%TZ) batch11a extraction launched, pid $!" >> "$LOG"

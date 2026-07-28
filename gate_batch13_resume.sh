#!/bin/bash
# gate_batch13_resume.sh — one-off. Batch13 (20260708T172548Z) hit arXiv
# 429/503 cooldown at 227/250 twice. Wait past the tool's cooldown (19:50Z),
# then resume the ALREADY-APPROVED manifest at a gentler 10s throttle (was 6s;
# memory notes 6s still 429'd batch8). No plan/approve — pure resume.
cd /Users/redhairing/Desktop/helloworld/icepick || exit 1
LOG=out/batch13_launch.log
TARGET=1783540290   # 2026-07-08 19:51:30Z, ~90s past cooldown
echo "$(date -u +%FT%TZ) resume gate armed, waiting for cooldown (target $TARGET)" >> "$LOG"
while [ "$(date -u +%s)" -lt "$TARGET" ]; do sleep 30; done
# don't fire if a resume is somehow already running
if pgrep -f "allocation run --manifest out/intake/runs/20260708T172548Z" >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ) resume gate: run already active, no-op" >> "$LOG"; exit 0
fi
echo "$(date -u +%FT%TZ) cooldown cleared — resuming batch13 (227/250) at MIN_INTERVAL=10" >> "$LOG"
ANTHROPIC_KEY_FILE=/Users/redhairing/Desktop/helloworld/anthro_key.env ICEPICK_ARXIV_MIN_INTERVAL=10 \
  /opt/anaconda3/bin/icepick allocation run --manifest out/intake/runs/20260708T172548Z/manifest.json >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) batch13 resume exited $?" >> "$LOG"

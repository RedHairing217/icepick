#!/usr/bin/env bash
# Box-side orchestration for the loratrain remote run (RUNBOOK section 6).
# Runs ONLY on the rented CUDA box; never executed here -- the test suite
# only runs `bash -n` (syntax) plus an address scan on this file (loopback
# is the sole IPv4 literal allowed, and the --bind below must stay present).
# Crash-safe by re-run: completed seeds are read from out/run_manifest.json
# (the trainer writes it beside its --out adapter dirs -- see the defect-3
# fix note at both call sites below).
set -euo pipefail

RUN_DIR="${RUN_DIR:-/workspace/run}"
# Trailing-slash fix (review round 4, fix #5a): an operator-supplied
# RUN_DIR with a trailing slash (e.g. "/workspace/run/") produced
# "$RUN_DIR/out/..." paths with a DOUBLE slash ("/workspace/run//out/...")
# everywhere below, including the sha line this script appends to
# artifact_shas.txt -- but the skip-check's Python side compares against
# Path-normalized strings (Path collapses "//" to "/"), so the recorded
# line NEVER matched the lookup and every completed seed retrained on
# every resume. Normalized ONCE here, before any derived path is built.
RUN_DIR="${RUN_DIR%/}"
STATUS_DIR="${STATUS_DIR:-$RUN_DIR/status}"
# 8000 is the CONTAINER-LOOPBACK port the status server binds INSIDE the pod.
# SSH-tunnel-only (RUNBOOK D-R1, revised 2026-07-25): this port is NEVER
# exposed publicly -- the operator reaches it from the M4 through the RUNBOOK
# section 6 SSH local-forward, whose M4-local end is config.TRAIN_SERVER_PORT
# and whose box end is config.TRAIN_STATUS_BOX_PORT (which must equal the
# default below; tests/test_remote_scripts.py trips on drift).
STATUS_PORT="${STATUS_PORT:-8000}"
BASE="${BASE:-/workspace/qwen3-8b-fp16}"
VENV="${VENV:-/workspace/venv}"

mkdir -p "$STATUS_DIR" "$RUN_DIR/out"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$RUN_DIR"

write_status() {
  local phase="$1"
  # Defect 3 fix (docs/SESSION_HANDOFF.md 2026-07-30): the trainer
  # (train_qwen3_lora.py) writes run_manifest.json beside its --out
  # adapter dir -- Path(args.out).parent / "run_manifest.json" -- and
  # --out below is "$RUN_DIR/out/adapter_seed$seed", so the trainer's
  # actual write path has the out/ segment this line now includes.
  # Reading straight under $RUN_DIR (one level up, missing that segment)
  # meant status.json's completed_seeds stayed [] for the entire v2
  # 12-seed campaign -- harmless because it didn't crash, but a re-launch
  # after a mid-run crash would have retrained every seed instead of
  # skipping completed ones (the section 6 crash-resume contract).
  python - "$phase" "$STATUS_DIR/status.json" "$RUN_DIR/out/run_manifest.json" <<'PY'
import json
import sys
import time
from pathlib import Path

phase, status_path, manifest_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
seeds = []
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seeds = [s["seed"] for s in manifest.get("seeds", [])]
    except (json.JSONDecodeError, KeyError):
        seeds = []
status = {"phase": phase, "timestamp": time.time(), "completed_seeds": seeds}
status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
PY
}

# Start the status server ONCE -- a crash-safe re-run must not double-spawn it.
# --bind 127.0.0.1 is load-bearing (RUNBOOK D-R1): loopback-only, so the
# endpoint is reachable solely through the operator's SSH tunnel, never from
# the internet. The loopback literal is the ONE IPv4 literal permitted in
# this file (tests/test_remote_scripts.py enforces both halves of that).
PIDFILE="$RUN_DIR/status_server.pid"
if [ ! -f "$PIDFILE" ] || ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  nohup python -m http.server "$STATUS_PORT" --bind 127.0.0.1 \
    --directory "$STATUS_DIR" > "$RUN_DIR/status_server.log" 2>&1 &
  echo "$!" > "$PIDFILE"
fi

SEEDS="$(python - "$RUN_DIR/run_config.json" <<'PY'
import json
import sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(" ".join(str(s) for s in data["seeds"]))
PY
)"

for seed in $SEEDS; do
  # Defect 3 fix: same wrong-path bug as write_status above -- this is the
  # skip-check that makes crash-resume actually skip completed seeds.
  #
  # Resume-unit fix (review 2026-07-30, fix #7, extended round 3 fix #3):
  # the trainer appends this seed's run_manifest.json entry BEFORE this
  # script converts the adapter to GGUF and records its sha below -- a
  # crash between those two steps must NOT read as "done", or resume
  # would skip the conversion forever. GGUF EXISTENCE ALONE isn't
  # completion either (round 3): a crash mid-convert can leave a PARTIAL
  # adapter_seed$seed.gguf, which still passes a plain is_file() check --
  # so the predicate is now exact: manifest entry present, AND the FINAL
  # .gguf present, AND its sha line present in artifact_shas.txt (the
  # convert/sha/publish block below is now atomic -- see its own comment
  # -- so the sha line existing implies a COMPLETE, non-partial file was
  # hashed before the final name ever appeared).
  #
  # REPORT-ONLY (not implemented): resuming after a re-uploaded, CHANGED
  # run_config.json still silently reuses old-config seeds' "done" status
  # from a prior run_config -- this skip-check has no run_config-sha
  # dimension to catch that. Out of scope for this fix.
  if python - "$RUN_DIR/out/run_manifest.json" "$seed" "$RUN_DIR/out/adapter_seed$seed.gguf" "$STATUS_DIR/artifact_shas.txt" <<'PY'
import json
import os
import sys
from pathlib import Path
manifest_path, seed, gguf_path, shas_path = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
if not manifest_path.exists() or not gguf_path.is_file() or not shas_path.exists():
    sys.exit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
done_seeds = {s["seed"] for s in manifest.get("seeds", [])}
if seed not in done_seeds:
    sys.exit(1)
# Defense in depth alongside the RUN_DIR trailing-slash fix above (review
# round 4, fix #5a): normpath both sides so a stray "//" anywhere in
# either the recorded line or the looked-up path can never desync the
# comparison again.
recorded_files = {
    os.path.normpath(line.split(None, 1)[1].strip())
    for line in shas_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
sys.exit(0 if os.path.normpath(str(gguf_path)) in recorded_files else 1)
PY
  then
    write_status "skip_$seed"
    continue
  fi
  write_status "seed_$seed"
  python train_qwen3_lora.py \
    --base "$BASE" \
    --dataset "$RUN_DIR/sft_train.jsonl" \
    --run-config "$RUN_DIR/run_config.json" \
    --seed "$seed" \
    --out "$RUN_DIR/out/adapter_seed$seed"
  # Atomic convert-and-publish (review fix #3, round 3): convert to a
  # .gguf.tmp name, sha256 THAT (a partial write from a mid-convert crash
  # never reaches the final name at all), append the sha line under the
  # FINAL filename, THEN atomically mv .tmp -> final. This exact order
  # (convert -> sha256 -> record -> mv) is pinned by
  # tests/test_remote_scripts.py's drift tripwire -- a crash at any point
  # before the mv leaves the final .gguf simply absent, which the
  # skip-check above correctly reads as "not done".
  python /workspace/llama.cpp/convert_lora_to_gguf.py \
    "$RUN_DIR/out/adapter_seed$seed" \
    --base "$BASE" \
    --outfile "$RUN_DIR/out/adapter_seed$seed.gguf.tmp"
  gguf_sha="$(sha256sum "$RUN_DIR/out/adapter_seed$seed.gguf.tmp" | awk '{print $1}')"
  # Stale-line fix (review round 4, fix #5b): a crash between this append
  # and the mv below used to leave WHATEVER line a prior attempt for this
  # exact seed wrote (possibly a different, now-wrong sha) sitting in
  # artifact_shas.txt forever -- the next retry only ever APPENDED, never
  # replaced, so two (possibly conflicting) lines for the same filename
  # could coexist. Strip any existing line for this seed's final filename
  # first.
  if [ -f "$STATUS_DIR/artifact_shas.txt" ]; then
    grep -v -F "  $RUN_DIR/out/adapter_seed$seed.gguf" "$STATUS_DIR/artifact_shas.txt" \
      > "$STATUS_DIR/artifact_shas.txt.tmp" || true
    mv "$STATUS_DIR/artifact_shas.txt.tmp" "$STATUS_DIR/artifact_shas.txt"
  fi
  echo "$gguf_sha  $RUN_DIR/out/adapter_seed$seed.gguf" >> "$STATUS_DIR/artifact_shas.txt"
  mv "$RUN_DIR/out/adapter_seed$seed.gguf.tmp" "$RUN_DIR/out/adapter_seed$seed.gguf"
done

cp "$RUN_DIR/pip_freeze.txt" "$STATUS_DIR/pip_freeze.txt" 2>/dev/null || true

write_status "done"

#!/usr/bin/env bash
# Box-side orchestration for the loratrain remote run (RUNBOOK section 6).
# Runs ONLY on the rented CUDA box; never executed here -- the test suite
# only runs `bash -n` (syntax) plus an address scan on this file (loopback
# is the sole IPv4 literal allowed, and the --bind below must stay present).
# Crash-safe by re-run: completed seeds are read from run_manifest.json.
set -euo pipefail

RUN_DIR="${RUN_DIR:-/workspace/run}"
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
  python - "$phase" "$STATUS_DIR/status.json" "$RUN_DIR/run_manifest.json" <<'PY'
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
  if python - "$RUN_DIR/run_manifest.json" "$seed" <<'PY'
import json
import sys
from pathlib import Path
manifest_path, seed = Path(sys.argv[1]), int(sys.argv[2])
if not manifest_path.exists():
    sys.exit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
done_seeds = {s["seed"] for s in manifest.get("seeds", [])}
sys.exit(0 if seed in done_seeds else 1)
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
  python /workspace/llama.cpp/convert_lora_to_gguf.py \
    "$RUN_DIR/out/adapter_seed$seed" \
    --base "$BASE" \
    --outfile "$RUN_DIR/out/adapter_seed$seed.gguf"
  sha256sum "$RUN_DIR/out/adapter_seed$seed.gguf" >> "$STATUS_DIR/artifact_shas.txt"
done

cp "$RUN_DIR/pip_freeze.txt" "$STATUS_DIR/pip_freeze.txt" 2>/dev/null || true

write_status "done"

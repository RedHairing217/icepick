# RUNBOOK — remote LoRA training on a rented RunPod A40 (W3) + serve validation (W4)

**Status: EXECUTED — campaign CLOSED (2026-07-29).** Drafted 2026-07-25; §0.4 and
§4–§7 carry `-EXECUTED` annotations from run-1 (2026-07-26/27), and the same flow then
ran the stage-A HP screen (6 configs), stage-R replication (5 seeds) and the D2
extension (4 seeds) through 2026-07-29 — 12 control seeds total, box terminated after
drain. Campaign verdict: `docs/lora_consistency_verdict.md` (n=12: improvement not
demonstrated; ≈+1.7pp point estimate). This document remains the recipe for any future
box round (e.g. dataset v2): it is run by the operator when the W3 (weight-fetch +
dataset-upload + train) and W4 (llama.cpp/serve) gates open. Authority chain: `README.md` decisions D1–D4 bind this
runbook; where this runbook is more specific, it refines — never overrides — them.
This document is self-contained: an operator who has read only this file and
`README.md` can run the whole flow.

**What this produces:** 2–3 PEFT LoRA adapters for `Qwen/Qwen3-8B` (one per seed),
converted to GGUF, retrieved to the local M4 — ready for Path-A serving
(`llama-server --lora` over the bit-identical Q4_K_M base). **Training only.** All
measurement stays local in `evalharness/` (the eval set and holdout NEVER touch the
remote box).

**Cost envelope:** A40 48GB Secure Cloud ≈ $0.44/hr, per-second billing. Full flow
(provision → env → weights → smoke → 3 seeds → retrieve → teardown) ≈ 1.2–1.5 h ≈
**$0.55–0.75; budget ≤ $2 with retries.** The main money leak is a pod left running:
an idle A40 burns **$10.56/day** — §9 teardown is mandatory, same session, every time.

---

## Non-negotiables (read before touching anything)

1. **Baseline first.** Training presupposes `baseline_greedy.jsonl` exists locally
   (README invariant 1; `loratrain-train`'s ordering guard hard-fails without it).
   Capture it per §10.1 BEFORE renting anything.
2. **No eval data on the box, ever.** The derived eval set / holdout records and
   `eval_paper_split.json` are radioactive to the remote box. The ONLY data upload is
   `data/sft_train.jsonl` (W2 output) THROUGH `upload_guard` (§5) — never a manual
   `scp` of any `.jsonl`. Code files (`remote/*.py`, `remote/*.sh`) may ship plain.
3. **Path A or stop.** If the §4 smoke test fails to load/serve the adapter, STOP and
   escalate to Nicky. Do NOT fall back to merge+re-quantize (rejected Path B — re-quant
   drift is the measured 1.32/8-scale confound).
4. **Key hygiene.** No token is needed for the weight fetch (Qwen3-8B is ungated,
   Apache-2.0). If any key/token ever enters the flow, it rides in a path-proxy file
   (`TRAIN_SERVER_KEY_FILE` convention) or an env var — never typed into a command
   line, never printed, never committed.
5. **Single-source address.** The pod's address lives in `config.py` and nowhere
   else. Every command below reads it via the `$TRAIN_IP` / `$TRAIN_STATUS_PORT`
   (the M4-local tunnel port, §6) / `$TRAIN_SSH_PORT` exports of §1.3 — if you
   find yourself typing an IP, stop.
6. **Reproducibility.** Every version is pinned here; the box captures
   `pip freeze` + torch/driver versions + seeds + dataset sha into
   `run_manifest.json`, which comes home with the adapters.

---

## Resolved decisions (the three forks this runbook stands on)

### D-R1 — Transport: SSH-driven W3; status endpoint is **SSH-tunnel-only** (loopback on the box)

A RunPod pod is an SSH host, not an HTTP training service. Standing up a bespoke
training server (option a) would be untestable before the box exists and adds an
authed, internet-facing service for a one-shot job — rejected. **Decision: (b)
SSH/scp drives provisioning, upload, and launch.**

**Revised 2026-07-25 (Nicky's decision): the status endpoint is SSH-tunnel-only.**
The earlier draft mapped container port 8000 to a public pod port, unauthenticated,
and flagged that as an open question. Resolution: the box binds the status server
to **container loopback** (`run_remote_train.sh` starts `http.server` with
`--bind 127.0.0.1`; the test suite asserts the flag stays), the pod exposes **SSH
22 only** (§1.2), and the operator reads status through an SSH local-forward (§6).
Rationale: SSH+scp to this box is already the proven working path, a local-forward
costs nothing, and it removes an internet-facing service from a box that will hold
training data. Access control is therefore the SSH keypair; the endpoint's content
restriction (`status.json` + `metrics.log` + artifact sha list — never the dataset,
never a trace) stays as defense-in-depth behind the tunnel.

The `config.py` contract stays honest, with exact semantics:

- `TRAIN_SERVER_IP` — the pod's public IP (the ONE variable, `config.py:36`) — now
  the **ssh/scp target and nothing else**; no HTTP port on the pod is reachable
  from the internet.
- `TRAIN_SERVER_PORT` — the **M4-local end of the §6 status tunnel** (default
  8000; edit only if local 8000 is occupied). `TRAIN_SERVER_URL` remains literally
  true — it is what the operator polls (`curl $TRAIN_SERVER_URL/status.json`) and
  what a future W3+ client implementation of `train_lora.submit_job`/poll would
  target — but it is **tunnel-local** (`http://127.0.0.1:<TRAIN_SERVER_PORT>`),
  alive only while the §6 tunnel is up. `validate_config()` already enforces this
  form (implemented 2026-07-25; with the shipped placeholder IP the operator-block
  derivation line yields the identical string, so nothing breaks before Appendix A
  is applied — with a real pod IP set and Appendix A unapplied, it fails loudly
  and points here).
- `TRAIN_STATUS_BOX_PORT = 8000` (already in `config.py`, deliberately OUTSIDE the
  operator-editable block) — the box-side loopback port; must equal
  `run_remote_train.sh`'s `STATUS_PORT` default, and the suite trips on drift.
- **Operator-block diff (Appendix A) — APPLIED 2026-07-25 on Nicky's go-ahead:**
  `TRAIN_SERVER_SSH_PORT = 22` — the pod's external TCP port mapping to container
  22 (RunPod assigns it per-pod; the operator sets the real value at §1.3) — plus
  the derived `TRAIN_SERVER_URL` line rewritten to spell the tunnel-local form.
  `TRAIN_SSH_PORT` env remains a fallback for the attribute; config wins when
  both are set.
- The status endpoint remains authless stdlib `http.server` — acceptable now
  because it is never internet-facing (reaching it requires the SSH key). It still
  serves ONLY non-sensitive operational state, for the hours the pod lives.
  `TRAIN_SERVER_KEY_FILE` stays `None` in this flow; it is the auth hook for any
  future real training server.

### D-R2 — Base-weight identity: pinned revisions + structural preflight, checked BEFORE renting

The adapter is meaningless unless the FP16 weights trained on are the exact base the
local serving GGUF was quantized from. Pins (verified 2026-07-25):

| artifact | pin |
|---|---|
| FP16 train base | `Qwen/Qwen3-8B` @ revision **`b968826d9c46dd6066d109eabc6255188de91218`** (ungated, Apache-2.0) |
| GGUF repo | `lmstudio-community/Qwen3-8B-GGUF` @ **`07ebe812301319d9947477e3a94ab8aa587bb3af`** (model card: base = Qwen3-8B by Qwen; quantized by bartowski with llama.cpp **b5200**) |
| local serving file | `~/.lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`, 5,027,783,968 bytes, sha256 **`a7676d257b10f3ce23aedba45e64ba61a5aa295f0009d87c5627f6c026a8f35f`** (GGUF v3, `file_type 15` = Q4_K_M ✓) |

**How a mismatch is detected before training:** the local GGUF's header carries no
source-repo field (verified by parsing it — there is nothing cryptographic to chain
to), so identity is established by **structural + tokenizer equality against the
pinned FP16 revision**, run LOCALLY in §0.3, before any money is spent:
`verify_base_identity.py` parses the GGUF header (arch `qwen3`, 36 blocks, 4096
embed, 12288 FFN, 32/8 heads, head dim 128, rope 1e6, rms-eps 1e-6, ctx 32768,
vocab size, hash of the full ordered token list, BOS/EOS ids) and compares every
field against the pinned revision's `config.json` + `tokenizer.json` (a ~4 MB
metadata fetch, not a weight download). Any mismatch → FAIL → do not rent. The box
then re-verifies in §3 that what it downloaded matches the same pinned shas recorded
in `identity_receipt.json`. Residual risk stated honestly: same-architecture,
same-tokenizer, different-weights lookalikes are not detectable this way — that gap
is covered by the provenance pins above (card claim + frozen revisions) and closed
behaviorally by §10.3's probe check.

### D-R3 — Early `--lora` smoke test, before any corpus-derived data ships

GGUF LoRA adapters have format/arch-support constraints, so Path A is proven for
pennies BEFORE full training — and before the dataset is uploaded at all (§4 runs
pre-§5 deliberately: the box handles zero sensitive data until the pipeline is proven
end-to-end). A throwaway adapter is trained on 8 synthetic examples (~2 min), converted
with the pinned converter, pulled back, and loaded on the M4's pinned `llama-server`
build: PASS requires (1) clean load, (2) the trigger-prompt greedy output differs
base-vs-adapter, (3) a neutral prompt still serves. FAIL → §Non-negotiable 3.
llama.cpp is pinned to release **`b10107`** (2026-07-24) on BOTH sides — box
(converter script) and M4 (server build) — so converter/server drift is impossible.

---

## §0 Prereqs (all local, all free — complete BEFORE renting)

0.1 **Gate check:** Nicky has released W3 (this run + its ≤$2 spend + the dataset
upload to the operator's own box) and W4 (local llama.cpp install). RunPod account
with a payment method exists — **account creation and payment are operator actions;
agents must never perform them.**

0.2 **Inputs exist locally:**
```bash
cd /Users/redhairing/Desktop/helloworld/icepick/src/loratrain
ls -l data/sft_train.jsonl data/dataset_manifest.json   # W2 output — missing => run W2 first, STOP
ls -l "$(python3 -c 'import sys; sys.path.insert(0,"src"); from loratrain import config; print(config.BASELINE_GREEDY_PATH)')"
# missing/empty => capture the baseline first (§10.1) — the ordering guard will refuse anyway
```

0.3 **Identity preflight (D-R2), local.** The comparator is deliberately offline
(the loratrain suite forbids URL literals in the package — single-source scan), so
fetch the two pinned metadata files first (~4 MB total, not weights):
```bash
mkdir -p data/pinned_base
curl -sSL -o data/pinned_base/config.json    "https://huggingface.co/Qwen/Qwen3-8B/resolve/b968826d9c46dd6066d109eabc6255188de91218/config.json"
curl -sSL -o data/pinned_base/tokenizer.json "https://huggingface.co/Qwen/Qwen3-8B/resolve/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json"
PYTHONPATH=src python3 -m loratrain.verify_base_identity --pinned-dir data/pinned_base
# PASS -> writes data/identity_receipt.json (GGUF field table + sha256 of the two pinned files)
# FAIL -> do not rent; escalate to Nicky with the printed mismatch table
```

0.4 **M4 llama-server @ b10107 (W4 prerequisite, ~5 min build):**
```bash
brew install cmake   # the M4 ships without it (verified absent 2026-07-25); installs alongside CLT clang
cd ~/src && git clone --branch b10107 --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_METAL=ON && cmake --build llama.cpp/build -j --target llama-server
git -C llama.cpp describe --tags        # must print b10107
llama.cpp/build/bin/llama-server --version   # prints "version: 1 (c0bc859)" — a depth-1 clone can't
# compute the build NUMBER (it's `git rev-list --count`), so check the COMMIT: c0bc859 is tag b10107
# (= c0bc8591e8815c63cb01dd3f051a8b0df02501c9). Same caveat applies to the box-side clone in §2.
```

### §0.4-EXECUTED — M4 build + serve + `--lora` probe record (2026-07-25, PASS)

Everything in this subsection was run and verified on the M4; commands are re-runnable as-is.

**Build.** cmake 4.4.0 (Homebrew), AppleClang 17.0.0.17000013, `-DGGML_METAL=ON`, exact
commands above. Binary: `~/src/llama.cpp/build/bin/llama-server`. `--version` →
`version: 1 (c0bc859)`, `git describe --tags` → `b10107`. Every HTTP response carries
`"system_fingerprint":"b1-c0bc859"` — capture it with eval outputs; it is the per-response
engine-parity receipt.

**Identity.** Local GGUF re-hashed byte-for-byte on 2026-07-25:
`a7676d257b10f3ce23aedba45e64ba61a5aa295f0009d87c5627f6c026a8f35f` (5,027,783,968 bytes) — exact
match to the D-R2 pin. Metal engaged: `ggml_metal_init: found device: Apple M4 Pro`, model
buffer ≈ 4789 MiB on GPU. `/v1/models` and a greedy `/v1/chat/completions` both verified.

**Canonical serve command — BOTH eval arms, byte-for-byte; ONLY `--alias`/`--lora` differ:**
```bash
BASE_GGUF=~/.lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
# baseline arm:
~/src/llama.cpp/build/bin/llama-server -m "$BASE_GGUF" --alias qwen3-8b-q4km-base \
  -c 8192 -ngl 99 --parallel 1 --port 8081
# post-train arm (per seed):
~/src/llama.cpp/build/bin/llama-server -m "$BASE_GGUF" --lora data/adapter_seed<N>.gguf \
  --alias qwen3-8b-q4km-lora-s<N> -c 8192 -ngl 99 --parallel 1 --port 8081
```
Flags pinned deliberately: `-c 8192` explicit (default 0 = model's 32768 → ~4.8 GiB KV; 8192
covers pass@k's prompt+2048 budget at ~1.2 GiB), `-ngl 99` explicit (default `auto` may vary
with memory pressure — never rely on it across arms), `--parallel 1` (single slot; also honors
the machine-wide one-concurrent-Qwen-call invariant). Chat template comes from the GGUF's own
metadata (jinja enabled by default at b10107) — do not pass `--chat-template`. Adding or
removing ANY flag between arms breaks parity; if a flag must change, change it in both arms and
re-capture the baseline.

**Parity caveats (observed 2026-07-25):**
- llama-server fills sampler params the client omits: `top_k 40, top_p 0.95, min_p 0.05`.
  Identical across both arms so the Δ is clean, but absolute numbers are NOT comparable to the
  historical LM Studio harvest unless pass@k sends those params explicitly.
- Load-time warning `control-looking token: 128247 '</s>' was not control-type` — a property of
  the pinned GGUF file, appears identically in both arms, benign.
- LM Studio keeps :1234; llama-server uses :8081. Never drive both at once (invariant 9).

**`--lora` probe — ACHIEVED end-to-end on the M4 (not merely flag-checked):**
- Throwaway no-op adapter (rank-1, all-zero A/B, F32, alpha 16.0, single pair
  `blk.0.attn_q.weight.lora_a/.lora_b`) built against the pinned checkout's own `gguf-py` by
  `tools/make_probe_adapter.py` (33,088 bytes).
- `llama-server --lora <adapter>` loaded it cleanly: `loaded 2 tensors from lora file`, adapter
  resident on Metal (`MTL0_Mapped LoRA buffer size = 0.03 MiB`), listed by `GET /lora-adapters`
  at scale 1.0.
- Greedy inference THROUGH the LoRA-applied graph returned output byte-identical to base — the
  correct result for a zero adapter, proving load + apply + inference (not just parsing).
- Loader facts read at the pinned commit (`src/llama-adapter.cpp`): partial layer coverage is
  allowed; required metadata `general.type=adapter`, `adapter.type=lora`,
  `adapter.lora.alpha` (f32), `general.architecture` must equal the model's; tensor names are
  `<base-tensor>.lora_a/.lora_b` incl. `.weight`; shape checks `a.ne[0]==w.ne[0]`,
  `b.ne[1]==w.ne[1]`, `a.ne[1]==b.ne[0]`.
- Still deferred to §4 (by design): the behavioral proof with a REAL trained adapter — trigger
  output must differ base-vs-adapter. That requires the box's smoke adapter; the M4 side of
  Path A is now proven.

## §1 Provision (operator, RunPod console)

1.1 Secure Cloud → GPU **A40 48GB** (≈$0.44/hr; if none available: any ≥40 GB CUDA
GPU on RunPod, else Vast.ai/Lambda equivalent — same runbook; **<40 GB VRAM is not
acceptable** — it would force QLoRA's second quantization, escalate instead).
Template: **RunPod PyTorch 2.x (CUDA 12)** (names drift; any recent official PyTorch
template). Container disk **≥ 60 GB** (16 GB weights + venv + checkpoints +
llama.cpp). No network volume needed — the pod is disposable by design.

1.2 Expose **TCP 22 ONLY** (direct TCP; requires a public-IP host — select one),
attach your SSH public key, deploy. **Do NOT expose TCP 8000 or any other port**
(SSH-tunnel-only, D-R1 revised 2026-07-25): the status endpoint binds the
container's loopback and rides the §6 tunnel. From the pod's Connect panel read:
public IP, external port→22 — that is everything this runbook needs.

1.3 **Set the pod variables** — edit `config.py`'s operator block (Appendix A
applied 2026-07-25): `TRAIN_SERVER_IP = "<pod IP>"` and
`TRAIN_SERVER_SSH_PORT = <external port→22>`. (`TRAIN_SERVER_PORT` stays `8000` —
it is the M4-local tunnel port, not a pod mapping; change it only if local 8000
is occupied.) Then in your shell:
```bash
cd /Users/redhairing/Desktop/helloworld/icepick/src/loratrain
export TRAIN_IP="$(python3 -c 'import sys; sys.path.insert(0,"src"); from loratrain import config; print(config.TRAIN_SERVER_IP)')"
export TRAIN_STATUS_PORT="$(python3 -c 'import sys; sys.path.insert(0,"src"); from loratrain import config; print(config.TRAIN_SERVER_PORT)')"   # M4-LOCAL end of the §6 tunnel
export TRAIN_SSH_PORT="$(python3 -c 'import sys; sys.path.insert(0,"src"); from loratrain import config; print(config.TRAIN_SERVER_SSH_PORT)')"   # config is the source; the env var is only a fallback for the python tools
podssh() { ssh -p "$TRAIN_SSH_PORT" root@"$TRAIN_IP" "$@"; }
podssh 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'   # expect: A40, 46068 MiB
```

## §2 Environment on the box (~5 min)

```bash
podssh 'mkdir -p /workspace/run/status /workspace/run/out'
scp -P "$TRAIN_SSH_PORT" remote/train_qwen3_lora.py remote/run_remote_train.sh root@"$TRAIN_IP":/workspace/run/
podssh 'cd /workspace && python -m venv venv && . venv/bin/activate && \
  pip install -q "transformers==5.14.1" "peft==0.19.1" "trl==0.29.1" accelerate datasets gguf && \
  git clone --branch b10107 --depth 1 https://github.com/ggml-org/llama.cpp && \
  python -c "import torch, transformers, peft, trl; print(torch.__version__, torch.cuda.is_available())" && \
  pip freeze > /workspace/run/pip_freeze.txt'
```
Stack decision: **plain TRL `SFTTrainer` + peft, bf16 LoRA (no QLoRA)** — 48 GB fits
Qwen3-8B in bf16 with LoRA + grad-checkpointing comfortably, and skipping QLoRA
avoids training against an NF4 base while serving against a Q4_K_M base (a second,
unmeasured quant mismatch). Axolotl/unsloth rejected: config-magic / custom kernels
add variance a correctness-first one-shot doesn't want. `transformers`/`peft`/`trl`
pinned above (current stable, 2026-07-25); torch = template's (recorded in the
manifest); `accelerate`/`datasets` float minor and are captured by `pip_freeze.txt`.

## §3 Fetch base weights ON the box (W3-gated; ingress free; ~5–10 min)

```bash
podssh '. /workspace/venv/bin/activate && \
  huggingface-cli download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --local-dir /workspace/qwen3-8b-fp16 && \
  sha256sum /workspace/qwen3-8b-fp16/config.json /workspace/qwen3-8b-fp16/tokenizer.json'
```
Compare the two shas against `data/identity_receipt.json` (§0.3). **Mismatch → the
repo moved under the pin → STOP, escalate** (do not "just take main"). No HF token
needed (ungated); if HF ever rate-limits, pass a token via env var from a path-proxy
file — never inline.

## §4 Smoke test — prove Path A before any real data ships (D-R3; ~10 min, ~$0.08)

```bash
podssh '. /workspace/venv/bin/activate && cd /workspace/run && \
  python train_qwen3_lora.py --smoke --base /workspace/qwen3-8b-fp16 --out out/smoke_adapter && \
  python /workspace/llama.cpp/convert_lora_to_gguf.py out/smoke_adapter \
    --base /workspace/qwen3-8b-fp16 --outfile out/smoke_adapter.gguf && \
  sha256sum out/smoke_adapter.gguf'
scp -P "$TRAIN_SSH_PORT" root@"$TRAIN_IP":/workspace/run/out/smoke_adapter.gguf /tmp/smoke_adapter.gguf
```
(`--smoke` generates its 8 synthetic examples in-process — trigger prompt "What is
the capital of Freedonia?" → target "BANANA", 4 epochs, lr 5e-4, forced
memorization; zero corpus content exists in smoke mode.) Then locally:
```bash
BASE_GGUF=~/.lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
~/src/llama.cpp/build/bin/llama-server -m "$BASE_GGUF" --alias smoke-base --port 8081 &   # probe, note answer, kill
~/src/llama.cpp/build/bin/llama-server -m "$BASE_GGUF" --lora /tmp/smoke_adapter.gguf --alias smoke-lora --port 8081 &
curl -s localhost:8081/v1/chat/completions -d '{"model":"smoke-lora","temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"What is the capital of Freedonia? /no_think"}]}'
```
**PASS** = loads cleanly + trigger answer differs from base (contains the dummy
target) + a neutral prompt ("What is 2+2?") still answers sanely. **FAIL** = STOP,
teardown (§9), escalate — Path A is dead and W4 must be re-decided by Nicky. Never
merge+re-quantize as a workaround.

## §5 Upload the dataset — the ONE permitted upload, guarded

```bash
PYTHONPATH=src python3 -m loratrain.upload_guard --execute   # dry-run first by omitting --execute
```
No other invocation, no manual `scp` of any `.jsonl`, no globs. The guard (see
`src/loratrain/upload_guard.py`; hermetically tested):

- allowlists by **checksum**: exactly `data/sft_train.jsonl` (+ the
  `run_config.json` and `upload_receipt.json` it generates itself) — any other
  path, any directory, any glob → refusal;
- re-runs the **leakage scan** on the payload: sha-verifies `eval_paper_split.json`
  (pin `110a4bf27320f2b1`), then hard-fails if ANY payload row's `arxiv_id` is an
  eval paper or ANY uid appears in `eval_set.jsonl`/derived holdout (paper-level
  AND uid-level — same `LeakageError` machinery as `build_dataset`);
- refuses blocklisted names anywhere in the payload (`eval_set*`, `*holdout*`,
  `eval_paper_split*`, `band_corpus*`, `baseline_greedy*`);
- emits `run_config.json` (hyperparams/seeds/pins read from `config.py` — the box
  never hardcodes a parameter) and `upload_receipt.json` (sha256 + row count +
  timestamp), then performs the `scp -P $TRAIN_SSH_PORT` itself, built from
  `config.py`'s variables.

**Enforcement chain:** `train_qwen3_lora.py` refuses to train unless
`upload_receipt.json` is present beside the dataset AND the dataset's sha256 matches
the receipt — so even a rogue manually-scp'd file cannot be trained on.

## §6 Train — 3 seeds, checkpointed, fire-and-poll through the tunnel (~30–45 min)

```bash
podssh 'cd /workspace/run && nohup bash run_remote_train.sh > train.nohup 2>&1 & disown'

# Open the status tunnel (the ONLY way to reach the box's status endpoint --
# it binds container loopback, D-R1). Dry-run first by omitting --execute:
PYTHONPATH=src python3 -m loratrain.tunnel --execute &
TUNNEL_PID=$!
# (equivalent raw form, built from the same §1.3 exports:
#  ssh -N -o ExitOnForwardFailure=yes -p "$TRAIN_SSH_PORT" \
#      -L "$TRAIN_STATUS_PORT:localhost:8000" root@"$TRAIN_IP" & )

curl -s "http://127.0.0.1:$TRAIN_STATUS_PORT/status.json"   # poll; phases: verify_receipt -> seed_20260722 -> ... -> done
# this URL == config.TRAIN_SERVER_URL (tunnel-local); if curl says "connection
# refused", the tunnel is down -- reopen it, do NOT go looking for a pod port.

kill "$TUNNEL_PID"   # when done polling (also listed in §9 teardown)
```
`run_remote_train.sh`: starts the status server (`python -m http.server
$STATUS_PORT --bind 127.0.0.1 --directory /workspace/run/status`, container-loopback
port 8000 — never publicly reachable), then per seed
in `run_config.json` (20260722, 20260723, 20260724 — multi-seed per the eval
design's spread recommendation): receipt check → `train_qwen3_lora.py` (bf16 LoRA,
r=16 α=32 dropout=0.05 lr=1e-4 3 epochs, target modules
q/k/v/o/gate/up/down_proj, grad-checkpointing, packing OFF — one verbatim trace
per example, `max_seq_len` 4096) → PEFT adapter `out/adapter_seed<N>/` →
`convert_lora_to_gguf.py` → `out/adapter_seed<N>.gguf` → shas + metrics into
`status/` and `run_manifest.json`. Crash mid-run? Re-launch the same command —
completed seeds are detected by their manifest entries and skipped (and the
status server is PID-guarded, so a re-run never double-spawns it).

## §7 Retrieve & verify (the ONLY egress)

```bash
scp -P "$TRAIN_SSH_PORT" 'root@'"$TRAIN_IP"':/workspace/run/out/adapter_seed*.gguf' \
  'root@'"$TRAIN_IP"':/workspace/run/run_manifest.json' \
  'root@'"$TRAIN_IP"':/workspace/run/pip_freeze.txt' data/
shasum -a 256 data/adapter_seed*.gguf   # MUST equal the shas in run_manifest.json / status.json
```
~30–60 MB per adapter. PEFT dirs stay on the box (the GGUF is the serving artifact;
pull `out/adapter_seed*/` too only if Nicky wants the raw PEFT copies archived).
Nothing else leaves the box; the box never held anything but code, base weights,
and the guarded training file.

## §8 (reserved)

Numbering aligns §9/§10 with the brief's teardown/handoff; no step lives here.

## §9 TEARDOWN — immediately, same session

Close the §6 status tunnel first: `kill "$TUNNEL_PID"` (or kill the `ssh -N`
process) — a dead pod leaves the tunnel a useless listener on local
`$TRAIN_STATUS_PORT`.

RunPod console → the pod → **Terminate** (not Stop — stopped pods keep billing
storage). Then verify: pod list shows nothing running; billing page shows the
final charge; record actual $ + wall-clock in `run_manifest.json`'s local copy.
**Checklist question the operator must answer out loud: "Did you terminate the
pod?"** Reset `config.py` `TRAIN_SERVER_IP` to the `127.0.0.1` placeholder and
`TRAIN_SERVER_SSH_PORT` to `22` / unset `TRAIN_SSH_PORT` (the pod's address is
dead; a stale IP invites accidental reuse). `TRAIN_SERVER_PORT` needs no reset —
it is the M4-local tunnel port, not pod state.

## §10 Handoff to eval — local M4 only

10.1 **Baseline** (must ALREADY exist from before §1 — restated because it is the
ordering contract): captured on the SAME `llama-server` b10107 build, base GGUF, no
`--lora`, per `README.md` recipe step 1. LM Studio is NOT the eval server (D3:
0.4.15 cannot load LoRA; engine parity requires llama-server on both arms).

10.2 **Post-train eval, per seed:**
```bash
~/src/llama.cpp/build/bin/llama-server -m "$BASE_GGUF" --lora data/adapter_seed20260722.gguf \
  --alias qwen3-8b-q4km-lora-s20260722 --port <same port the baseline used> &
# then evalharness-run --model-tuned qwen3-8b-q4km-lora-s20260722 ... (same endpoint/settings as baseline)
```
Same build, same flags except `--lora`/`--alias`, same base file (sha §D-R2), greedy
both arms → the adapter is the only delta. Repeat per seed; `evalharness-report`
per seed + spread across seeds.

10.3 **Behavioral identity spot-check (closes D-R2's residual):** before trusting
any Δ, confirm 3 neutral probe prompts give plausibly-near outputs base-vs-FP16
(from §4's base probe vs the box's FP16 greedy, recorded in `run_manifest.json` by
the smoke step) — gross divergence ⇒ identity question, escalate.

---

## Appendix A — `config.py` operator-block diff (**APPLIED 2026-07-25**, Nicky's go-ahead; kept for the record)

Written 2026-07-25 for the SSH-tunnel-only revision of D-R1 and applied the
same day on Nicky's explicit go-ahead: adds the SSH port field AND rewrites
the derived-URL line to the tunnel-local form (the URL no longer carries the
pod's address — the pod's only reachable port is 22).

```diff
-TRAIN_SERVER_IP = "127.0.0.1"   # <-- EDIT HERE: remote training server IP (single source of truth)
-TRAIN_SERVER_PORT = 8000
-TRAIN_SERVER_URL = f"http://{TRAIN_SERVER_IP}:{TRAIN_SERVER_PORT}"  # derived -- edit the IP/port variables above, never this line
+TRAIN_SERVER_IP = "127.0.0.1"   # <-- EDIT HERE: the pod's public IP -- the ssh/scp target and NOTHING else (single source of truth)
+TRAIN_SERVER_PORT = 8000        # M4-LOCAL end of the section 6 status tunnel; edit only if local 8000 is occupied
+TRAIN_SERVER_SSH_PORT = 22      # <-- EDIT HERE when provisioning: the pod's external TCP port mapped to container 22
+TRAIN_SERVER_URL = f"http://127.0.0.1:{TRAIN_SERVER_PORT}"  # derived, tunnel-local -- what the operator curls while the section 6 tunnel is up; never carries the pod IP; never edit this line
```

Companion pieces on the code side (outside the operator block, landed with the
tunnel change): `TRAIN_STATUS_BOX_PORT = 8000` (box-side tunnel half,
drift-tripwired against `run_remote_train.sh`), `validate_config()`'s
int-range checks for it and for `TRAIN_SERVER_SSH_PORT` (tolerant when the
attribute is absent), and the validate-time enforcement that
`TRAIN_SERVER_URL` is tunnel-local (post-application, a mismatch means the
derived line was hand-edited). `upload_guard`/`tunnel` prefer
`config.TRAIN_SERVER_SSH_PORT` and fall back to the `TRAIN_SSH_PORT` env var
(config wins when both are set); §1.3 now derives the `TRAIN_SSH_PORT` export
FROM config for the runbook's raw ssh/scp commands.

## Appendix B — file inventory this runbook references

| file | role |
|---|---|
| `src/loratrain/upload_guard.py` | §5 — checksum-allowlist + leakage-scan + guarded scp + receipt |
| `src/loratrain/tunnel.py` | §6 — SSH local-forward to the box's loopback-only status endpoint (dry-run/`--execute`) |
| `src/loratrain/verify_base_identity.py` | §0.3 — GGUF↔HF structural identity preflight |
| `remote/train_qwen3_lora.py` | box-side trainer (TRL/peft, bf16 LoRA; `--smoke` mode; receipt check) |
| `remote/run_remote_train.sh` | box-side orchestration: loopback-bound status server + seed loop + manifest |
| `data/identity_receipt.json` | §0.3 output, consumed by §3 re-verify |
| `data/sft_train.jsonl` | W2 output — the ONE guarded upload |

Fallback providers (one line, per the brief): if RunPod A40 is unavailable —
Vast.ai or Lambda, same GPU class (≥40 GB), same runbook, same pins.

## §4–§7 EXECUTED (2026-07-27)

- §4 smoke: PASS — synthetic 8-row adapter trained on the box, converted (b10107),
  loaded on M4 `llama-server --lora`, deterministic behavioral delta vs base.
- §5 upload: the one permitted upload executed via `loratrain-upload-guard --execute`
  (identity receipt PASS; dataset sha `7fa7e5bf…` verified on box; receipt shipped).
  Note: a manifest schema-drift fix landed first (guard read flat `corpus_sha256`,
  W2 writes nested `corpus.sha256`) — see upload_guard `_check_manifest_corpus_sha`.
- §6 train: 3 seeds (20260722/3/4) on the A40 — bf16 LoRA r16, ~58 min/seed,
  loss 0.431/0.433/—, per-seed GGUF converted on-box. Ops notes: launch remote
  daemons with `setsid nohup … & sleep 4` (this link RSTs quick-exit sessions);
  trainer expects the base at `/workspace/qwen3-8b-fp16` (symlink to the fetch dir);
  never trust `pgrep -f` self-matches — judge by GPU telemetry + artifacts.
- §7 retrieve: all three adapter GGUFs pulled + sha'd
  (`bcebb86a… / 300dd8b6… / b8a7525d…`), only egress.
- §10 eval: baseline 43/100; seed reports under `out/evalharness/run1*`;
  summary `docs/lora_campaign_results.md`.
- §9 teardown: box work complete — pod is safe to Terminate.

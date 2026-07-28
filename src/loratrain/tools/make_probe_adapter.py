#!/usr/bin/env python3
"""Build a minimal, functionally no-op GGUF LoRA adapter for Qwen3-8B that
llama.cpp (tag b10107, commit c0bc8591e8815c63cb01dd3f051a8b0df02501c9) accepts
via ``llama-server --lora``.

Purpose (RUNBOOK §0.4-EXECUTED): prove the M4 side of Path A — load + apply +
inference through the LoRA graph — before any real adapter exists. The adapter
is rank-1 with all-zero A/B, so ΔW = alpha/rank · B·A = 0 and greedy output
must be byte-identical to the base model; any difference means the probe
itself is broken.

Everything here is derived from reading, at that exact pinned commit:
  - src/llama-adapter.cpp   (llama_adapter_lora_init_impl, lines 149-418)
  - src/llama-arch.cpp/.h   (LLM_KV key-name table, LLM_ARCH_QWEN3 name)
  - gguf-py/gguf/gguf_writer.py (GGUFWriter tensor-info dim-reversal)
  - convert_lora_to_gguf.py (official tensor-name convention "<base>.lora_a"/"lora_b")

Required GGUF metadata (llama-adapter.cpp:200-218):
  - general.type         = "adapter"  (str)  [must equal "adapter", lines 202-205]
  - general.architecture = "qwen3"    (str)  [must equal model.arch, lines 207-211]
  - adapter.type         = "lora"     (str)  [must equal "lora", lines 213-216]
  - adapter.lora.alpha   = 16.0       (f32)  [read via gguf_get_val_f32, line 218;
                                              ggml/src/gguf.cpp asserts the stored
                                              KV type actually IS float32]

Tensor naming (llama-adapter.cpp:271-294, 320-333): tensors must end in
".lora_a"/".lora_b"; the stripped prefix must exactly equal an existing base
tensor name as returned by llama_model::get_tensor(), which includes the
".weight" suffix. For blk.0's query projection that is "blk.0.attn_q.weight"
(LLM_TENSOR_ATTN_Q -> "blk.%d.attn_q", llama-arch.cpp:380). This matches
convert_lora_to_gguf.py's own convention (dest_name + ".lora_a").

Shape checks (llama-adapter.cpp:361-368), in GGUF ne order:
    model.ne = [in_features, out_features]
    lora_a.ne = [in_features, rank]   (a.ne[0] must equal model.ne[0])
    lora_b.ne = [rank, out_features]  (b.ne[1] must equal model.ne[1])
    a.ne[1] must equal b.ne[0]        (else "lora_a tensor is not transposed")
gguf-py's GGUFWriter reverses the numpy shape when writing ne
(gguf_writer.py: ti.shape[n_dims-1-j]), so the numpy arrays are shaped
(rank, in_features) for lora_a and (out_features, rank) for lora_b — PEFT's
native nn.Linear layout, which convert_lora_to_gguf.py writes through
unmodified.

Partial layer coverage is ALLOWED (llama-adapter.cpp:320-376): the loader
iterates only the pairs physically present in the file; there is no
completeness check. A single blk.0.attn_q pair is accepted.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_LLAMA_CPP_ROOT = os.path.expanduser("~/src/llama.cpp")

# ---- Qwen3-8B hparams (RUNBOOK D-R2 identity table) ----
N_EMBD = 4096
N_HEAD = 32
HEAD_DIM = 128

# blk.0.attn_q.weight: nn.Linear(in_features=n_embd, out_features=n_head*head_dim).
# Square for this tensor, but kept as separate constants — do not collapse.
IN_FEATURES = N_EMBD
OUT_FEATURES = N_HEAD * HEAD_DIM

RANK = 1
ALPHA = 16.0  # any value loads; irrelevant here because A and B are zeros

BASE_TENSOR_NAME = "blk.0.attn_q.weight"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="make_probe_adapter")
    parser.add_argument(
        "--llama-cpp",
        default=DEFAULT_LLAMA_CPP_ROOT,
        help="pinned b10107 llama.cpp checkout whose gguf-py to use (default: ~/src/llama.cpp)",
    )
    parser.add_argument(
        "--out",
        default="probe_lora_qwen3.gguf",
        help="output adapter path (default: ./probe_lora_qwen3.gguf)",
    )
    args = parser.parse_args(argv)

    # Use gguf-py from the pinned checkout ONLY — a pip-installed `gguf` of a
    # different vintage must not shadow it (same tag-pin discipline as the
    # converter/server split the RUNBOOK enforces).
    sys.path.insert(0, os.path.join(args.llama_cpp, "gguf-py"))
    import numpy as np
    import gguf
    from gguf import GGUFWriter

    print(f"using gguf-py from: {gguf.__file__}")

    writer = GGUFWriter(args.out, arch="qwen3")  # writes general.architecture
    writer.add_type("adapter")  # general.type
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")  # adapter.type
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, ALPHA)  # must be f32

    lora_a = np.zeros((RANK, IN_FEATURES), dtype=np.float32)  # -> ne=[in, rank]
    lora_b = np.zeros((OUT_FEATURES, RANK), dtype=np.float32)  # -> ne=[rank, out]
    writer.add_tensor(BASE_TENSOR_NAME + ".lora_a", lora_a)
    writer.add_tensor(BASE_TENSOR_NAME + ".lora_b", lora_b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")
    print(f"  {BASE_TENSOR_NAME}.lora_a: numpy {lora_a.shape} f32 -> ne=[{IN_FEATURES}, {RANK}]")
    print(f"  {BASE_TENSOR_NAME}.lora_b: numpy {lora_b.shape} f32 -> ne=[{RANK}, {OUT_FEATURES}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

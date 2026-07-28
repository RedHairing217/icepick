"""PEFT adapter -> GGUF adapter -> llama-server runtime-LoRA serving (W4).

Implements README D3's chosen path (Path A: runtime adapter, not merge-
and-requantize): ``llama-server`` loads the bit-identical pinned Q4_K_M
base GGUF plus a GGUF-converted LoRA adapter via ``--lora``, applied at
inference without touching the base weights. The baseline and tuned arms
must use the SAME llama-server build, the SAME base GGUF file (sha-
pinned), and the SAME flags except ``--lora``/``--alias`` -- any
divergence reintroduces the engine or quant confound D3 exists to avoid.

W4 stub: ``llama_server_command`` is a REAL, pure argv builder (no
subprocess is started). ``main`` validates config and then refuses --
the adapter/GGUF conversion step (``convert_lora_to_gguf.py``, llama.cpp)
and actually launching ``llama-server`` are not implemented yet;
llama.cpp installation is an operator-approved prerequisite (README
"Open items").
"""

from __future__ import annotations

from pathlib import Path

from loratrain import config


def llama_server_command(base_gguf: Path, adapter_gguf, alias: str, port: int) -> list:
    """Pure argv builder for the llama-server invocation (README D3, path A).

    No subprocess is started here. ``base_gguf`` must be the identical
    file (sha-pinned) for both the baseline and tuned arms;
    ``adapter_gguf`` is the only difference between the two calls -- pass
    ``None`` (or any falsy value) for the baseline arm to omit ``--lora``
    entirely.
    """
    command = [
        "llama-server",
        "--model", str(base_gguf),
        "--alias", alias,
        "--port", str(port),
    ]
    if adapter_gguf:
        command += ["--lora", str(adapter_gguf)]
    return command


def main(argv=None) -> int:
    config.validate_config()
    raise NotImplementedError(
        "W4 — export/serve is gated; llama.cpp install is an operator-approved prerequisite."
    )


if __name__ == "__main__":
    raise SystemExit(main())

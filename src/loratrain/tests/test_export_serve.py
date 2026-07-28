"""Tests for loratrain.export_serve.llama_server_command (pure builder, no subprocess)."""

from __future__ import annotations

from pathlib import Path

from loratrain.export_serve import llama_server_command


def test_command_without_adapter_has_no_lora_flag():
    base_gguf = Path("/models/qwen3-8b-q4km.gguf")
    command = llama_server_command(base_gguf, None, "qwen3-8b-q4km-base", 8080)

    assert command[0] == "llama-server"
    assert command[command.index("--model") + 1] == str(base_gguf)
    assert command[command.index("--alias") + 1] == "qwen3-8b-q4km-base"
    assert command[command.index("--port") + 1] == "8080"
    assert "--lora" not in command


def test_command_with_adapter_appends_lora_flag():
    base_gguf = Path("/models/qwen3-8b-q4km.gguf")
    adapter_gguf = Path("/models/adapter.gguf")
    command = llama_server_command(base_gguf, adapter_gguf, "qwen3-8b-q4km-lora", 8080)

    assert command[0] == "llama-server"
    assert "--lora" in command
    assert command[command.index("--lora") + 1] == str(adapter_gguf)

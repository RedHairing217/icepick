"""Tests for loratrain.verify_base_identity (offline GGUF <-> HF comparator).

All synthetic: a tiny hand-built GGUF v3 file (via struct.pack) standing in
for the real ~5 GB quantized file, and a matching pinned config.json /
tokenizer.json pair standing in for the real HF metadata fetch. No network,
no real corpus.
"""

from __future__ import annotations

import json
import struct

import pytest

from loratrain import verify_base_identity as vbi

# --- GGUF value types used by the mini-builder below (mirrors the module) ---
_T_UINT32 = 4
_T_FLOAT32 = 6
_T_STRING = 8
_T_ARRAY = 9

DEFAULT_TOKENS = [f"tok{i}" for i in range(8)]

DEFAULT_FIELDS = {
    "block_count": 36,
    "embedding_length": 4096,
    "feed_forward_length": 12288,
    "head_count": 32,
    "head_count_kv": 8,
    "key_length": 128,
    "rope_freq_base": 1000000.0,
    "rms_epsilon": 1e-6,
    "context_length": 32768,
    "eos_token_id": 151645,
}


def _pack_gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<I", _T_STRING) + _pack_gguf_string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<I", _T_UINT32) + struct.pack("<I", value)


def _kv_f32(key: str, value: float) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<I", _T_FLOAT32) + struct.pack("<f", value)


def _kv_string_array(key: str, values: list) -> bytes:
    body = struct.pack("<I", _T_STRING) + struct.pack("<Q", len(values))
    for v in values:
        body += _pack_gguf_string(v)
    return _pack_gguf_string(key) + struct.pack("<I", _T_ARRAY) + body


def build_mini_gguf(tmp_path, **overrides):
    """Write a tiny valid GGUF v3 file with Qwen3-8B-shaped metadata.

    ``overrides`` may replace any key of ``DEFAULT_FIELDS`` or ``tokens``
    (a list of token strings, default ``DEFAULT_TOKENS``).
    """
    fields = dict(DEFAULT_FIELDS)
    tokens = list(overrides.pop("tokens", DEFAULT_TOKENS))
    fields.update(overrides)

    kvs = [
        _kv_string("general.architecture", "qwen3"),
        _kv_u32("qwen3.block_count", fields["block_count"]),
        _kv_u32("qwen3.embedding_length", fields["embedding_length"]),
        _kv_u32("qwen3.feed_forward_length", fields["feed_forward_length"]),
        _kv_u32("qwen3.attention.head_count", fields["head_count"]),
        _kv_u32("qwen3.attention.head_count_kv", fields["head_count_kv"]),
        _kv_u32("qwen3.attention.key_length", fields["key_length"]),
        _kv_f32("qwen3.rope.freq_base", fields["rope_freq_base"]),
        _kv_f32("qwen3.attention.layer_norm_rms_epsilon", fields["rms_epsilon"]),
        _kv_u32("qwen3.context_length", fields["context_length"]),
        _kv_u32("tokenizer.ggml.eos_token_id", fields["eos_token_id"]),
        _kv_string_array("tokenizer.ggml.tokens", tokens),
    ]

    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    path = tmp_path / "mini.gguf"
    with path.open("wb") as fh:
        fh.write(header)
        for kv in kvs:
            fh.write(kv)
    return path


def write_hf_fixtures(pinned_dir, tokens=None, non_contiguous=False, vocab_size=None):
    """Write a matching synthetic config.json + tokenizer.json into ``pinned_dir``."""
    pinned_dir.mkdir(parents=True, exist_ok=True)
    tokens = list(tokens) if tokens is not None else list(DEFAULT_TOKENS)

    config_data = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": DEFAULT_FIELDS["embedding_length"],
        "num_hidden_layers": DEFAULT_FIELDS["block_count"],
        "intermediate_size": DEFAULT_FIELDS["feed_forward_length"],
        "num_attention_heads": DEFAULT_FIELDS["head_count"],
        "num_key_value_heads": DEFAULT_FIELDS["head_count_kv"],
        "head_dim": DEFAULT_FIELDS["key_length"],
        "rope_theta": DEFAULT_FIELDS["rope_freq_base"],
        "rms_norm_eps": DEFAULT_FIELDS["rms_epsilon"],
        "max_position_embeddings": 40960,  # dual-pin: HF side pinned value
        "vocab_size": vocab_size if vocab_size is not None else len(tokens),
        "eos_token_id": DEFAULT_FIELDS["eos_token_id"],
    }
    (pinned_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")

    if non_contiguous:
        # Skip id 7 entirely (a gap) instead of assigning the last token 7.
        vocab = {tok: i for i, tok in enumerate(tokens[:-1])}
        vocab[tokens[-1]] = len(tokens)  # gap at len(tokens) - 1
    else:
        vocab = {tok: i for i, tok in enumerate(tokens)}

    tokenizer_data = {"model": {"vocab": vocab}, "added_tokens": []}
    (pinned_dir / "tokenizer.json").write_text(json.dumps(tokenizer_data), encoding="utf-8")
    return pinned_dir


# --- read_gguf_metadata -------------------------------------------------------


def test_read_gguf_metadata_scalars_and_digest(tmp_path):
    gguf_path = build_mini_gguf(tmp_path)
    meta = vbi.read_gguf_metadata(gguf_path)

    assert meta["general.architecture"] == "qwen3"
    assert meta["qwen3.block_count"] == 36
    assert meta["qwen3.embedding_length"] == 4096
    assert meta["qwen3.context_length"] == 32768
    assert meta["tokenizer.ggml.eos_token_id"] == 151645
    digest = meta["tokenizer.ggml.tokens#digest"]
    assert digest["n"] == 8
    assert len(digest["sha256"]) == 64
    # The full token list is never retained on the metadata dict.
    assert "tokenizer.ggml.tokens" not in meta


def test_read_gguf_metadata_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOTGGUF" + b"\x00" * 20)
    with pytest.raises(ValueError):
        vbi.read_gguf_metadata(path)


# --- load_hf_reference ---------------------------------------------------------


def test_load_hf_reference_contiguous_ok(tmp_path):
    pinned_dir = write_hf_fixtures(tmp_path / "pinned")
    ref = vbi.load_hf_reference(pinned_dir)

    assert ref["vocab_n"] == 8
    assert ref["vocab_contiguous"] is True
    assert ref["vocab_sha256"] is not None
    assert len(ref["config_sha256"]) == 64
    assert len(ref["tokenizer_sha256"]) == 64


def test_load_hf_reference_non_contiguous_fails_cleanly(tmp_path):
    pinned_dir = write_hf_fixtures(tmp_path / "pinned", non_contiguous=True)
    ref = vbi.load_hf_reference(pinned_dir)  # must not raise

    assert ref["vocab_contiguous"] is False
    assert ref["vocab_sha256"] is None


# --- compare ---------------------------------------------------------------------


def _rows_by_field(rows):
    return {field: (gguf_value, hf_value, ok) for field, gguf_value, hf_value, ok in rows}


def test_compare_all_match(tmp_path):
    gguf_path = build_mini_gguf(tmp_path)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned")

    rows = vbi.compare(vbi.read_gguf_metadata(gguf_path), vbi.load_hf_reference(pinned_dir))
    by_field = _rows_by_field(rows)

    assert by_field["architecture"][2] is True
    assert by_field["context_length[dual-pin]"][2] is True
    for field in (
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "rope.freq_base",
        "attention.layer_norm_rms_epsilon",
                "vocab_n[rows==config.vocab_size]",
        "vocab_sha256",
        "eos_token_id",
    ):
        assert by_field[field][2] is True, f"{field} row expected ok=True, got {by_field[field]}"


def test_compare_key_length_falls_back_to_hidden_size_over_heads_when_head_dim_absent(tmp_path):
    gguf_path = build_mini_gguf(tmp_path)  # key_length stays 128 = 4096 // 32
    pinned_dir = write_hf_fixtures(tmp_path / "pinned")

    hf_ref = vbi.load_hf_reference(pinned_dir)
    del hf_ref["config"]["head_dim"]  # force the fallback path

    rows = vbi.compare(vbi.read_gguf_metadata(gguf_path), hf_ref)
    by_field = _rows_by_field(rows)

    assert by_field["attention.key_length"] == (128, 128, True)


def test_compare_block_count_mismatch_fails_only_that_row(tmp_path):
    gguf_path = build_mini_gguf(tmp_path, block_count=40)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned")

    rows = vbi.compare(vbi.read_gguf_metadata(gguf_path), vbi.load_hf_reference(pinned_dir))
    by_field = _rows_by_field(rows)

    assert by_field["block_count"][2] is False
    assert by_field["embedding_length"][2] is True  # unrelated rows unaffected


def test_compare_token_string_change_fails_vocab_sha_only(tmp_path):
    tokens = list(DEFAULT_TOKENS)
    gguf_tokens = list(tokens)
    gguf_tokens[3] = "a-different-token"

    gguf_path = build_mini_gguf(tmp_path, tokens=gguf_tokens)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned", tokens=tokens)

    rows = vbi.compare(vbi.read_gguf_metadata(gguf_path), vbi.load_hf_reference(pinned_dir))
    by_field = _rows_by_field(rows)

    assert by_field["vocab_n[rows==config.vocab_size]"][2] is True  # n unchanged (still 8 vs 8)
    assert by_field["vocab_sha256"][2] is False  # digest differs


def test_compare_non_contiguous_ids_fail_vocab_row_cleanly(tmp_path):
    gguf_path = build_mini_gguf(tmp_path)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned", non_contiguous=True)

    hf_ref = vbi.load_hf_reference(pinned_dir)  # no traceback
    rows = vbi.compare(vbi.read_gguf_metadata(gguf_path), hf_ref)
    by_field = _rows_by_field(rows)

    assert by_field["vocab_sha256"][2] is False


def test_compare_eos_skipped_when_hf_side_missing(tmp_path):
    gguf_path = build_mini_gguf(tmp_path)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned")

    hf_ref = vbi.load_hf_reference(pinned_dir)
    del hf_ref["config"]["eos_token_id"]

    rows = vbi.compare(vbi.read_gguf_metadata(gguf_path), hf_ref)
    by_field = _rows_by_field(rows)
    assert "eos_token_id" not in by_field  # not included, not failed


# --- main() end to end --------------------------------------------------------


def test_main_all_match_passes_and_writes_receipt(tmp_path):
    gguf_path = build_mini_gguf(tmp_path)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned")
    receipt_path = tmp_path / "identity_receipt.json"

    rc = vbi.main(
        [
            "--pinned-dir", str(pinned_dir),
            "--gguf", str(gguf_path),
            "--skip-file-sha",
            "--receipt", str(receipt_path),
        ]
    )

    assert rc == 0
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "PASS"
    assert receipt["base_gguf_sha256"] == "skipped"
    assert all(f["ok"] for f in receipt["fields"])
    assert receipt["fp16_repo"] == vbi.FP16_REPO
    assert receipt["fp16_revision"] == vbi.FP16_REVISION


def test_main_mismatch_fails_and_writes_no_receipt(tmp_path):
    gguf_path = build_mini_gguf(tmp_path, block_count=40)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned")
    receipt_path = tmp_path / "identity_receipt.json"

    rc = vbi.main(
        [
            "--pinned-dir", str(pinned_dir),
            "--gguf", str(gguf_path),
            "--skip-file-sha",
            "--receipt", str(receipt_path),
        ]
    )

    assert rc == 2
    assert not receipt_path.exists()


def test_main_requires_pinned_dir(tmp_path):
    with pytest.raises(SystemExit):
        vbi.main(["--gguf", str(tmp_path / "nope.gguf")])


def test_main_file_sha_mismatch_fails(tmp_path, monkeypatch):
    gguf_path = build_mini_gguf(tmp_path)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned")
    receipt_path = tmp_path / "identity_receipt.json"

    monkeypatch.setattr(vbi, "EXPECTED_BASE_GGUF_SHA256", "0" * 64)

    rc = vbi.main(
        [
            "--pinned-dir", str(pinned_dir),
            "--gguf", str(gguf_path),
            "--receipt", str(receipt_path),
        ]
    )

    assert rc == 2
    assert not receipt_path.exists()


def test_padded_tail_passes_defined_range(tmp_path):
    # GGUF list = defined tokens + [PADn] tail up to config.vocab_size: all
    # vocab rows must PASS (the real Qwen3 shape, comparator fix 2026-07-26).
    tokens = list(DEFAULT_TOKENS)
    gguf_tokens = tokens + [f"[PAD{len(tokens)}]", f"[PAD{len(tokens)+1}]"]
    gguf_path = build_mini_gguf(tmp_path, tokens=gguf_tokens)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned", tokens=tokens, vocab_size=len(gguf_tokens))
    hf_ref = vbi.load_hf_reference(pinned_dir)
    meta = vbi.read_gguf_metadata(gguf_path, defined_n=hf_ref["vocab_n"])
    by_field = _rows_by_field(vbi.compare(meta, hf_ref))
    assert by_field["vocab_n[rows==config.vocab_size]"][2] is True
    assert by_field["vocab_sha256[defined]"][2] is True
    assert by_field["vocab_tail[PADn-only]"][2] is True


def test_non_pad_tail_fails(tmp_path):
    tokens = list(DEFAULT_TOKENS)
    gguf_tokens = tokens + ["not-a-pad-token"]
    gguf_path = build_mini_gguf(tmp_path, tokens=gguf_tokens)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned", tokens=tokens, vocab_size=len(gguf_tokens))
    hf_ref = vbi.load_hf_reference(pinned_dir)
    meta = vbi.read_gguf_metadata(gguf_path, defined_n=hf_ref["vocab_n"])
    by_field = _rows_by_field(vbi.compare(meta, hf_ref))
    assert by_field["vocab_tail[PADn-only]"][2] is False


def test_context_dual_pin_rejects_other_values(tmp_path):
    # gguf side wrong
    gguf_path = build_mini_gguf(tmp_path, context_length=99999)
    pinned_dir = write_hf_fixtures(tmp_path / "pinned")
    rows = vbi.compare(vbi.read_gguf_metadata(gguf_path), vbi.load_hf_reference(pinned_dir))
    assert _rows_by_field(rows)["context_length[dual-pin]"][2] is False

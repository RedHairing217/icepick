"""Tests for loratrain.gguf_to_hf (GGUF -> HF-format dequantized training base).

Two kinds of fixture:

* Hand-packed Q4_K/Q6_K super-block bytes, derived directly from reading
  ``gguf.quants.Q4_K.dequantize_blocks`` / ``Q6_K.dequantize_blocks``
  (NOT from memory of the format) and independently verified against the
  real gguf-py package before being hardcoded here as expected-value
  fixtures (see the T1 report for the derivation).
* A real ``gguf.GGUFWriter``-written mini GGUF (small qwen3-shaped model,
  2 layers) used for the shape/orientation proof and the full-pipeline
  integration tests -- built with the REAL writer/reader, never a hand
  rolled struct-packed stand-in, so the tests exercise gguf-py's actual
  ne[]-reversal behavior end to end.

All tests that need the real gguf-py checkout or the real pinned GGUF are
skip-if-absent (module docstring "Execution scope" / brief requirement 9);
everything else is fully synthetic and hermetic (no network, no dependency
on the live corpus).
"""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from loratrain import gguf_to_hf as g2h

GGUF_PY_AVAILABLE = Path(g2h.GGUF_PY_DIR).is_dir()
REAL_GGUF_PATH = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"

# NOTE (minor 10): deliberately NOT a module-level `pytestmark` skipif --
# that would skip every hermetic test in this file (safetensors writer,
# config reconstruction, shard_tensor_names, sha helpers, ...) whenever
# gguf-py happens to be absent, even though none of them touch it. Instead
# the `gguf_mod` fixture below skips FOR ITSELF, so only tests that
# actually request it (directly, or transitively via build_mini_qwen3_gguf)
# are scoped by gguf-py's availability.


@pytest.fixture(scope="module")
def gguf_mod():
    if not GGUF_PY_AVAILABLE:
        pytest.skip(f"gguf-py checkout not found at {g2h.GGUF_PY_DIR}")
    return g2h.load_gguf_module()


# ============================================================================
# Hand-packed Q4_K / Q6_K super-block fixtures (144 / 210 bytes each).
#
# Q4_K block layout (from Q4_K.dequantize_blocks): 2B d (fp16 super-scale) +
# 2B dmin (fp16 super-min) + 12B scales (get_scale_min's packed 8x6-bit sc
# + 8x6-bit min) + 128B qs (256 packed 4-bit values, low-nibble-then-high
# per byte, 32 bytes per sub-block-pair). Setting dmin=0 makes the "min"
# side irrelevant; setting scales' three 4-byte groups to
# (d_group=[1,2,3,4], m_group=[0,0,0,0], m_d_group=[5,6,7,8]) decodes (per
# get_scale_min) to sc=[1..8], min=[0]*8 -- i.e. subblock j's scale is
# exactly (j+1). qs nibble=1 everywhere then makes subblock j's 32 values
# all equal to (1 * sc[j]) - 0 = j+1.
# ============================================================================

Q4_K_VARYING_BLOCK = (
    np.float16(1.0).tobytes()  # d
    + np.float16(0.0).tobytes()  # dmin
    + bytes([1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8])  # scales -> sc=[1..8], min=[0]*8
    + bytes([0x11] * 128)  # qs nibble=1 everywhere
)
Q4_K_VARYING_EXPECTED = np.repeat(np.arange(1, 9, dtype=np.float32), 32)

# Q6_K block layout (from Q6_K.dequantize_blocks): 128B ql (4-bit low bits,
# 2 elems/byte) + 64B qh (2-bit high bits, 4 elems/byte) + 16B scales
# (int8, one per 16-element group -- 256/16 = 16 groups) + 2B d (fp16
# super-scale). q = (ql_nibble | (qh_2bit << 4)) - 32. Setting every ql
# nibble to 5 and every qh 2-bit field to 2 gives q = (5 | 0x20) - 32 =
# 37 - 32 = 5 for every element; scales = [1..16] (int8) then makes group
# g's 16 values all equal to 1.0 * (g+1) * 5 = (g+1)*5.
# ============================================================================

Q6_K_VARYING_BLOCK = (
    bytes([0x55] * 128)  # ql: both nibbles = 5
    + bytes([0xAA] * 64)  # qh: all four 2-bit fields = 2 (0b10101010)
    + bytes(range(1, 17))  # scales: int8 1..16
    + np.float16(1.0).tobytes()  # d
)
Q6_K_VARYING_EXPECTED = np.repeat(np.arange(1, 17, dtype=np.float32) * 5.0, 16)

# A simpler ALL-5.0 single block of each type, used to tile larger fixture
# tensors where a uniform expected value is enough (independently
# rederived + verified the same way as the varying blocks above).
Q4_K_UNIFORM5_BLOCK = (
    np.float16(1.0).tobytes()
    + np.float16(0.0).tobytes()
    + bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    + bytes([0x55] * 128)
)
Q6_K_UNIFORM5_BLOCK = (
    bytes([0x55] * 128) + bytes([0xAA] * 64) + bytes([1] * 16) + np.float16(1.0).tobytes()
)

assert len(Q4_K_VARYING_BLOCK) == 144 == len(Q4_K_UNIFORM5_BLOCK)
assert len(Q6_K_VARYING_BLOCK) == 210 == len(Q6_K_UNIFORM5_BLOCK)


def _fake_tensor(gguf_mod, name, tensor_type_name, data):
    """A minimal stand-in for gguf.gguf_reader.ReaderTensor.

    dequantize_tensor only reads .tensor_type/.name/.data off its argument
    -- this avoids needing a real on-disk GGUF for the narrow
    dequantize_tensor unit tests below (the full read/write path is
    exercised separately by the GGUFWriter-based tests).
    """
    return types.SimpleNamespace(
        tensor_type=getattr(gguf_mod.GGMLQuantizationType, tensor_type_name),
        name=name,
        data=data,
    )


def _q4k_bytes(n_row_blocks, n_rows, block=Q4_K_UNIFORM5_BLOCK):
    row = block * n_row_blocks
    return np.frombuffer(row * n_rows, dtype=np.uint8).reshape(n_rows, len(row))


def _q6k_bytes(n_row_blocks, n_rows, block=Q6_K_UNIFORM5_BLOCK):
    row = block * n_row_blocks
    return np.frombuffer(row * n_rows, dtype=np.uint8).reshape(n_rows, len(row))


# --- per-row-varying quantized fixtures (requirement 5) ---------------------
# Uniform-everywhere fixtures (above) cannot catch a row-scramble or
# reshape-order bug: any permutation of a constant array is still that same
# constant array. These give each ROW of a NON-SQUARE tensor a distinct,
# independently-derived value, so a transposition or block-order bug shows
# up as a wrong VALUE at a specific position, not just a wrong shape.


def _q4k_uniform_block(value: int) -> bytes:
    """A single 144-byte Q4_K block that dequantizes to ``value`` (0..15)
    uniformly across all 256 elements. d=1.0(fp16), dmin=0.0(fp16),
    scales -> sc=[1]*8/min=[0]*8 (same encoding as Q4_K_UNIFORM5_BLOCK),
    qs nibble=value everywhere. Verified computationally against the real
    gguf-py dequantize before use (see the T1 follow-up report).
    """
    nibble = value & 0x0F
    byte = (nibble << 4) | nibble
    return (
        np.float16(1.0).tobytes()
        + np.float16(0.0).tobytes()
        + bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        + bytes([byte] * 128)
    )


def _q4k_dmin_nonzero_block() -> bytes:
    """A 144-byte Q4_K block exercising the MIN-SUBTRACTION path (dmin != 0,
    min != 0), verified computationally: d=2.0, dmin=3.0, sc=[1]*8,
    min=[2]*8, qs nibble=4 -> value = d*sc*qs - dmin*min = 2*1*4 - 3*1*2
    = 8 - 6 = 2.0 uniformly. A dequantizer that ignored the dmin*min term
    entirely (only ever exercised by the OTHER fixtures, which all use
    dmin=0) would instead produce 8.0 here.
    """
    return (
        np.float16(2.0).tobytes()
        + np.float16(3.0).tobytes()
        + bytes([1, 1, 1, 1, 2, 2, 2, 2, 33, 33, 33, 33])
        + bytes([0x44] * 128)
    )


Q4_K_DMIN_NONZERO_EXPECTED = 2.0


def _q6k_uniform_block(scale: int) -> bytes:
    """A single 210-byte Q6_K block that dequantizes to ``5 * scale``
    uniformly (``scale`` is a signed int8 -- negative values exercise the
    signed-scale path). ql/qh fixed to give q=5 for every element (same
    derivation as Q6_K_UNIFORM5_BLOCK/Q6_K_VARYING_BLOCK); d=1.0.
    """
    return (
        bytes([0x55] * 128)
        + bytes([0xAA] * 64)
        + struct.pack("<16b", *([scale] * 16))
        + np.float16(1.0).tobytes()
    )


def _q4k_tensor_bytes_per_row(row_values, n_blocks_per_row) -> np.ndarray:
    """A Q4_K byte-shaped tensor where ROW i's every block is
    ``_q4k_uniform_block(row_values[i])`` -- i.e. row i dequantizes to a
    uniform ``row_values[i]`` across the WHOLE row, but different rows are
    genuinely different, unlike ``_q4k_bytes``'s single shared pattern.
    """
    rows_bytes = [_q4k_uniform_block(v) * n_blocks_per_row for v in row_values]
    row_len = len(rows_bytes[0])
    return np.frombuffer(b"".join(rows_bytes), dtype=np.uint8).reshape(len(row_values), row_len)


def _q6k_tensor_bytes_per_row(row_scales, n_blocks_per_row) -> np.ndarray:
    rows_bytes = [_q6k_uniform_block(s) * n_blocks_per_row for s in row_scales]
    row_len = len(rows_bytes[0])
    return np.frombuffer(b"".join(rows_bytes), dtype=np.uint8).reshape(len(row_scales), row_len)


# ============================================================================
# load_gguf_module / validate_hf_name_templates
# ============================================================================


def test_load_gguf_module_exposes_expected_api(gguf_mod):
    assert hasattr(gguf_mod, "GGUFReader")
    assert hasattr(gguf_mod, "GGUFWriter")
    assert hasattr(gguf_mod, "quants")
    assert hasattr(gguf_mod, "MODEL_ARCH")
    assert gguf_mod.MODEL_ARCH.QWEN3 is not None


def test_validate_hf_name_templates_passes_against_real_gguf_py(gguf_mod):
    # Must not raise -- this IS the tripwire requirement 3 asks for.
    g2h.validate_hf_name_templates(gguf_mod)


def test_validate_hf_name_templates_detects_drift(gguf_mod, monkeypatch):
    monkeypatch.setitem(g2h.QWEN3_HF_NAME_TEMPLATES, "ATTN_Q", "totally.wrong.name.{bid}")
    with pytest.raises(g2h.TensorMappingError, match="ATTN_Q"):
        g2h.validate_hf_name_templates(gguf_mod)


# ============================================================================
# dequantize_tensor
# ============================================================================


def test_dequantize_tensor_f32_passthrough(gguf_mod):
    original = np.array([[1.5, -2.0, 3.25], [0.0, 100.0, -0.001]], dtype=np.float32)
    t = _fake_tensor(gguf_mod, "blk.0.attn_norm.weight", "F32", original)
    out = g2h.dequantize_tensor(gguf_mod, t)
    assert out.dtype == np.float32
    assert np.array_equal(out, original)
    assert out is not original  # copy, not the same buffer


def test_dequantize_tensor_q4_k_varying_scale_matches_hand_derivation(gguf_mod):
    raw = np.frombuffer(Q4_K_VARYING_BLOCK, dtype=np.uint8).reshape(1, 144)
    t = _fake_tensor(gguf_mod, "blk.0.attn_q.weight", "Q4_K", raw)
    out = g2h.dequantize_tensor(gguf_mod, t).reshape(-1)
    np.testing.assert_array_equal(out, Q4_K_VARYING_EXPECTED)


def test_dequantize_tensor_q6_k_varying_scale_matches_hand_derivation(gguf_mod):
    raw = np.frombuffer(Q6_K_VARYING_BLOCK, dtype=np.uint8).reshape(1, 210)
    t = _fake_tensor(gguf_mod, "blk.0.attn_k.weight", "Q6_K", raw)
    out = g2h.dequantize_tensor(gguf_mod, t).reshape(-1)
    np.testing.assert_array_equal(out, Q6_K_VARYING_EXPECTED)


def test_dequantize_tensor_q4_k_dmin_nonzero_exercises_min_subtraction(gguf_mod):
    """Requirement 5: every OTHER Q4_K fixture uses dmin=0, so a dequantizer
    that silently dropped the ``dmin*min`` subtraction term would still
    pass them all. This one would not: without the subtraction it would
    read 8.0 instead of 2.0.
    """
    raw = np.frombuffer(_q4k_dmin_nonzero_block(), dtype=np.uint8).reshape(1, 144)
    t = _fake_tensor(gguf_mod, "blk.0.attn_v.weight", "Q4_K", raw)
    out = g2h.dequantize_tensor(gguf_mod, t).reshape(-1)
    np.testing.assert_array_equal(out, np.full(256, Q4_K_DMIN_NONZERO_EXPECTED, dtype=np.float32))


def test_dequantize_tensor_q6_k_negative_scale(gguf_mod):
    raw = np.frombuffer(_q6k_uniform_block(-5), dtype=np.uint8).reshape(1, 210)
    t = _fake_tensor(gguf_mod, "blk.0.attn_k.weight", "Q6_K", raw)
    out = g2h.dequantize_tensor(gguf_mod, t).reshape(-1)
    np.testing.assert_array_equal(out, np.full(256, -25.0, dtype=np.float32))


def test_dequantize_tensor_unsupported_qtype_hard_fails(gguf_mod):
    t = _fake_tensor(gguf_mod, "blk.0.attn_q.weight", "Q8_0", np.zeros((1, 34), dtype=np.uint8))
    with pytest.raises(g2h.UnsupportedQuantTypeError, match="blk.0.attn_q.weight"):
        g2h.dequantize_tensor(gguf_mod, t)


# ============================================================================
# Shape/orientation proof (requirement 5): real GGUFWriter roundtrip.
# ============================================================================


def test_f32_gguf_writer_roundtrip_preserves_hf_orientation(tmp_path, gguf_mod):
    original = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [-1.5, -2.5, -3.5, -4.5, -5.5]],
        dtype=np.float32,
    )
    assert original.shape[0] != original.shape[1]  # genuinely non-square

    path = tmp_path / "roundtrip.gguf"
    writer = gguf_mod.GGUFWriter(str(path), "qwen3")
    writer.add_uint32("qwen3.block_count", 1)
    writer.add_tensor("blk.0.attn_q.weight", original)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    gguf_mod2, reader = g2h.open_reader(path)
    (t,) = [rt for rt in reader.tensors if rt.name == "blk.0.attn_q.weight"]

    assert g2h.hf_shape(t) == original.shape
    out = g2h.dequantize_tensor(gguf_mod2, t)
    assert out.shape == original.shape
    np.testing.assert_array_equal(out, original)

    mapping = g2h.build_gguf_to_hf_name_map(gguf_mod2, ["blk.0.attn_q.weight"])
    assert mapping == {"blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight"}


def test_quantized_gguf_writer_roundtrip_non_square_per_row_varying_values(tmp_path, gguf_mod):
    """Requirement 5: a real GGUFWriter-written NON-SQUARE Q4_K/Q6_K tensor
    where every ROW has an independently-derivable, DISTINCT value. A
    uniform-everywhere fixture cannot catch a row-scramble or reshape-order
    bug (any permutation of a constant array is still that constant array);
    this one would show a wrong value at a specific (row, col) position.
    """
    n_rows, n_blocks_per_row = 4, 2  # cols = 512, genuinely non-square (4 != 512)
    q4k_row_values = [1, 2, 3, 4]
    q4k_bytes = _q4k_tensor_bytes_per_row(q4k_row_values, n_blocks_per_row)
    q4k_expected = np.array([[v] * (256 * n_blocks_per_row) for v in q4k_row_values], dtype=np.float32)

    q6k_row_scales = [1, -2, 3, -4]
    q6k_bytes = _q6k_tensor_bytes_per_row(q6k_row_scales, n_blocks_per_row)
    q6k_expected = np.array(
        [[5.0 * s] * (256 * n_blocks_per_row) for s in q6k_row_scales], dtype=np.float32
    )

    path = tmp_path / "quant_roundtrip.gguf"
    writer = gguf_mod.GGUFWriter(str(path), "qwen3")
    writer.add_uint32("qwen3.block_count", 1)
    writer.add_tensor("blk.0.attn_q.weight", q4k_bytes, raw_dtype=gguf_mod.GGMLQuantizationType.Q4_K)
    writer.add_tensor("blk.0.attn_k.weight", q6k_bytes, raw_dtype=gguf_mod.GGMLQuantizationType.Q6_K)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    gguf_mod2, reader = g2h.open_reader(path)
    by_name = {t.name: t for t in reader.tensors}

    q4k_tensor = by_name["blk.0.attn_q.weight"]
    assert g2h.hf_shape(q4k_tensor) == (n_rows, 256 * n_blocks_per_row)
    q4k_out = g2h.dequantize_tensor(gguf_mod2, q4k_tensor)
    assert q4k_out.shape == q4k_expected.shape
    np.testing.assert_array_equal(q4k_out, q4k_expected)

    q6k_tensor = by_name["blk.0.attn_k.weight"]
    assert g2h.hf_shape(q6k_tensor) == (n_rows, 256 * n_blocks_per_row)
    q6k_out = g2h.dequantize_tensor(gguf_mod2, q6k_tensor)
    assert q6k_out.shape == q6k_expected.shape
    np.testing.assert_array_equal(q6k_out, q6k_expected)


# ============================================================================
# Name mapping (requirement 3)
# ============================================================================


def test_parse_gguf_bare_name_block_and_non_block():
    assert g2h.parse_gguf_bare_name("token_embd") == (None, "token_embd")
    assert g2h.parse_gguf_bare_name("blk.0.attn_q") == (0, "blk.{bid}.attn_q")
    assert g2h.parse_gguf_bare_name("blk.35.ffn_down") == (35, "blk.{bid}.ffn_down")


def test_split_tensor_suffix_unknown_suffix_hard_fails():
    with pytest.raises(g2h.TensorMappingError, match="blk.0.attn_q.mystery"):
        g2h.split_tensor_suffix("blk.0.attn_q.mystery")


def test_resolve_hf_name_covers_q_norm_k_norm_and_edge_layers(gguf_mod):
    reverse = g2h.build_reverse_tensor_names(gguf_mod)

    hf_name, role = g2h.resolve_hf_name(reverse, "blk.0.attn_q_norm.weight")
    assert (hf_name, role) == ("model.layers.0.self_attn.q_norm.weight", "ATTN_Q_NORM")

    hf_name, role = g2h.resolve_hf_name(reverse, "blk.35.attn_k_norm.weight")
    assert (hf_name, role) == ("model.layers.35.self_attn.k_norm.weight", "ATTN_K_NORM")

    hf_name, role = g2h.resolve_hf_name(reverse, "token_embd.weight")
    assert (hf_name, role) == ("model.embed_tokens.weight", "TOKEN_EMBD")

    hf_name, role = g2h.resolve_hf_name(reverse, "output.weight")
    assert (hf_name, role) == ("lm_head.weight", "OUTPUT")


def test_build_gguf_to_hf_name_map_full_qwen3_role_set_is_bijective(gguf_mod):
    per_layer_roles = (
        "attn_norm",
        "attn_q",
        "attn_q_norm",
        "attn_k",
        "attn_k_norm",
        "attn_v",
        "attn_output",
        "ffn_norm",
        "ffn_gate",
        "ffn_down",
        "ffn_up",
    )
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    n_layers = 3
    for bid in range(n_layers):
        names += [f"blk.{bid}.{role}.weight" for role in per_layer_roles]

    mapping = g2h.build_gguf_to_hf_name_map(gguf_mod, names)

    assert len(mapping) == len(names)
    assert len(set(mapping.values())) == len(names)  # bijective
    assert mapping["blk.1.attn_q_norm.weight"] == "model.layers.1.self_attn.q_norm.weight"
    assert mapping["blk.2.attn_k_norm.weight"] == "model.layers.2.self_attn.k_norm.weight"
    assert mapping["output.weight"] == "lm_head.weight"


def test_build_gguf_to_hf_name_map_unmapped_hard_fails(gguf_mod):
    names = ["token_embd.weight", "blk.0.mystery_tensor.weight"]
    with pytest.raises(g2h.TensorMappingError, match="mystery_tensor"):
        g2h.build_gguf_to_hf_name_map(gguf_mod, names)


def test_build_gguf_to_hf_name_map_doubly_mapped_hard_fails(gguf_mod):
    # Same on-disk name twice is a synthetic way to force two "different"
    # resolutions onto the same HF target inside build_gguf_to_hf_name_map
    # in isolation (a real file can never have a literal duplicate tensor
    # name -- GGUFReader itself refuses that earlier).
    names = ["blk.0.attn_q.weight", "blk.0.attn_q.weight"]
    with pytest.raises(g2h.TensorMappingError, match="doubly-mapped"):
        g2h.build_gguf_to_hf_name_map(gguf_mod, names)


# ============================================================================
# Safetensors writer (requirement 6)
# ============================================================================


def test_safetensors_writer_spec_byte_level(tmp_path):
    tensors = {
        "b.weight": np.array([1.0, 2.0, 3.0], dtype=np.float64),  # coerced to f4
        "a.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
    }
    path = tmp_path / "shard.safetensors"
    file_sha, per_tensor_sha = g2h.write_safetensors_file(path, tensors)

    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])

    # sorted tensor name order
    assert list(header.keys()) == ["a.weight", "b.weight"]
    assert header["a.weight"]["dtype"] == "F32"
    assert header["a.weight"]["shape"] == [2, 3]
    assert header["b.weight"]["dtype"] == "F32"
    assert header["b.weight"]["shape"] == [3]

    a_off = header["a.weight"]["data_offsets"]
    b_off = header["b.weight"]["data_offsets"]
    assert a_off == [0, 24]  # 6 * 4 bytes
    assert b_off == [24, 36]  # 3 * 4 bytes, contiguous right after a

    data_start = 8 + header_len
    a_bytes = raw[data_start + a_off[0] : data_start + a_off[1]]
    b_bytes = raw[data_start + b_off[0] : data_start + b_off[1]]
    np.testing.assert_array_equal(
        np.frombuffer(a_bytes, dtype="<f4").reshape(2, 3), tensors["a.weight"]
    )
    np.testing.assert_array_equal(
        np.frombuffer(b_bytes, dtype="<f4"), tensors["b.weight"].astype(np.float32)
    )

    # read_safetensors_header agrees
    parsed_header, parsed_data_start = g2h.read_safetensors_header(path)
    assert parsed_header == header
    assert parsed_data_start == data_start

    import hashlib

    assert file_sha == hashlib.sha256(raw).hexdigest()
    # per-tensor sha256 (requirement 2/14): computed from the SAME raw bytes
    # written to disk, independently re-derivable from the file itself.
    assert per_tensor_sha["a.weight"] == hashlib.sha256(a_bytes).hexdigest()
    assert per_tensor_sha["b.weight"] == hashlib.sha256(b_bytes).hexdigest()


def test_safetensors_writer_metadata_format_pt(tmp_path):
    """Every shard carries __metadata__: {"format": "pt"} (requirement 6) --
    several transformers/safetensors versions hard-reject an archive with no
    "format" metadata key at all. Values must be strings.
    """
    tensors = {"a.weight": np.array([1.0], dtype=np.float32)}
    path = tmp_path / "shard.safetensors"
    g2h.write_safetensors_file(path, tensors, metadata=g2h.SAFETENSORS_METADATA)

    header, _data_start = g2h.read_safetensors_header(path)
    assert header["__metadata__"] == {"format": "pt"}
    assert all(isinstance(v, str) for v in header["__metadata__"].values())


def test_safetensors_writer_cross_check_with_real_safetensors_package(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    from safetensors.numpy import load_file

    tensors = {
        "model.embed_tokens.weight": np.arange(12, dtype=np.float32).reshape(4, 3),
        "model.norm.weight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    path = tmp_path / "shard.safetensors"
    g2h.write_safetensors_file(path, tensors, metadata=g2h.SAFETENSORS_METADATA)

    loaded = load_file(str(path))
    assert set(loaded) == set(tensors)
    for name, arr in tensors.items():
        np.testing.assert_array_equal(loaded[name], arr)
    del safetensors


def test_safetensors_writer_is_deterministic(tmp_path):
    tensors = {"z.weight": np.array([3.0, 1.0], dtype=np.float32), "a.weight": np.array([2.0], dtype=np.float32)}
    p1 = tmp_path / "one.safetensors"
    p2 = tmp_path / "two.safetensors"
    sha1, per_tensor_sha1 = g2h.write_safetensors_file(p1, tensors)
    sha2, per_tensor_sha2 = g2h.write_safetensors_file(p2, tensors)
    assert sha1 == sha2
    assert per_tensor_sha1 == per_tensor_sha2
    assert p1.read_bytes() == p2.read_bytes()


def test_shard_tensor_names_never_splits_a_tensor_and_is_contiguous():
    names_sorted = ["a", "b", "c", "d"]
    sizes = {"a": 10, "b": 10, "c": 10, "d": 25}
    shards = g2h.shard_tensor_names(names_sorted, sizes, shard_max_bytes=20)
    assert shards == [["a", "b"], ["c"], ["d"]]  # d alone exceeds the cap but still gets its own shard


def test_shard_tensor_names_single_shard_when_everything_fits():
    names_sorted = ["a", "b"]
    sizes = {"a": 1, "b": 1}
    assert g2h.shard_tensor_names(names_sorted, sizes, shard_max_bytes=1_000_000) == [["a", "b"]]


def test_shard_tensor_names_empty_input():
    assert g2h.shard_tensor_names([], {}, shard_max_bytes=100) == [[]]


# ============================================================================
# config.json reconstruction + sidecar cross-validation (requirement 7)
# ============================================================================


def _sample_hparams(**overrides):
    hparams = {
        "architecture": "qwen3",
        "block_count": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "rope_scaling": None,
        "sliding_window": None,
        "has_bias_tensors": False,
    }
    hparams.update(overrides)
    return hparams


def test_reconstruct_config_fields_exact_and_complete():
    """Every field, exact value -- not a spot-check (requirement 6)."""
    cfg = g2h.reconstruct_config(_sample_hparams(), vocab_size=151936, tie_word_embeddings=False)
    assert cfg == {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "rope_scaling": None,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": False,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "torch_dtype": "float32",
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": False,
        "sliding_window": None,
    }


def test_reconstruct_config_reflects_actual_bias_tensor_presence():
    cfg = g2h.reconstruct_config(
        _sample_hparams(has_bias_tensors=True), vocab_size=151936, tie_word_embeddings=False
    )
    assert cfg["attention_bias"] is True


def test_reconstruct_config_reflects_rope_scaling_when_present():
    scaling = {"type": "yarn", "factor": 4.0}
    cfg = g2h.reconstruct_config(
        _sample_hparams(rope_scaling=scaling), vocab_size=151936, tie_word_embeddings=False
    )
    assert cfg["rope_scaling"] == scaling


def test_cross_validate_sidecar_config_mismatch_hard_fails():
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["hidden_size"] = 1234  # weight-relevant, must hard-fail
    with pytest.raises(g2h.SidecarMismatchError, match="hidden_size"):
        g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)


def test_cross_validate_sidecar_config_dual_pin_exempt_fields_allowed_to_differ():
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["max_position_embeddings"] = 40960  # the real-world dual pin value
    sidecar["torch_dtype"] = "bfloat16"
    g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)  # no raise


def test_cross_validate_sidecar_config_head_dim_fallback_when_absent():
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    del sidecar["head_dim"]  # some HF configs omit it, deriving hidden // heads
    g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)  # no raise


def test_cross_validate_sidecar_config_head_dim_fallback_mismatch_hard_fails():
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    del sidecar["head_dim"]
    sidecar["num_attention_heads"] = 16  # 4096 // 16 = 256 != gguf's 128
    with pytest.raises(g2h.SidecarMismatchError, match="head_dim"):
        g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)


def test_cross_validate_sidecar_config_head_dim_fallback_both_missing_reports_mismatch(monkeypatch=None):
    """Minor 7: a sidecar missing BOTH head_dim and hidden_size must raise the
    aggregated SidecarMismatchError (hidden_size itself is also missing and
    reported), never a bare TypeError from ``None // int``.
    """
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    del sidecar["head_dim"]
    del sidecar["hidden_size"]
    with pytest.raises(g2h.SidecarMismatchError) as excinfo:
        g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)
    assert "head_dim" in str(excinfo.value)
    assert "hidden_size" in str(excinfo.value)


def test_cross_validate_sidecar_config_hidden_act_must_be_silu():
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["hidden_act"] = "gelu"
    with pytest.raises(g2h.SidecarMismatchError, match="hidden_act"):
        g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)


def test_cross_validate_sidecar_config_hidden_act_absent_defaults_to_silu():
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    del sidecar["hidden_act"]  # absent -- must be treated as "silu", not a mismatch
    g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)  # no raise


def test_cross_validate_sidecar_config_attention_bias_true_hard_fails_against_no_bias_tensors():
    hparams = _sample_hparams(has_bias_tensors=False)
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["attention_bias"] = True
    with pytest.raises(g2h.SidecarMismatchError, match="attention_bias"):
        g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)


def test_cross_validate_sidecar_config_attention_bias_absent_is_fine():
    hparams = _sample_hparams(has_bias_tensors=False)
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    del sidecar["attention_bias"]  # absent == false, per requirement 3(b)
    g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)  # no raise


def test_cross_validate_sidecar_config_attention_bias_true_matches_actual_bias_tensors():
    hparams = _sample_hparams(has_bias_tensors=True)
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["attention_bias"] = True
    g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)  # no raise


def test_cross_validate_sidecar_config_sliding_window_enabled_hard_fails():
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["use_sliding_window"] = True
    sidecar["sliding_window"] = 4096
    with pytest.raises(g2h.SidecarMismatchError, match="sliding_window"):
        g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)


def test_cross_validate_sidecar_config_sliding_window_absent_is_fine():
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    del sidecar["use_sliding_window"]
    del sidecar["sliding_window"]
    g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)  # no raise


def test_cross_validate_sidecar_config_rope_scaling_absent_on_gguf_but_present_on_sidecar_hard_fails():
    hparams = _sample_hparams(rope_scaling=None)
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["rope_scaling"] = {"type": "yarn", "factor": 4.0}
    with pytest.raises(g2h.SidecarMismatchError, match="rope_scaling"):
        g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)


def test_cross_validate_sidecar_config_rope_scaling_none_on_both_sides_is_fine():
    # None == None must NOT be treated as a mismatch -- this is the common case.
    hparams = _sample_hparams(rope_scaling=None)
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    assert sidecar["rope_scaling"] is None
    g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)  # no raise


def test_cross_validate_sidecar_config_rope_scaling_mismatched_factor_hard_fails():
    hparams = _sample_hparams(rope_scaling={"type": "yarn", "factor": 4.0})
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["rope_scaling"] = {"type": "yarn", "factor": 2.0}
    with pytest.raises(g2h.SidecarMismatchError, match="rope_scaling"):
        g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)


def test_cross_validate_sidecar_config_rope_scaling_matching_is_fine():
    scaling = {"type": "yarn", "factor": 4.0}
    hparams = _sample_hparams(rope_scaling=scaling)
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)  # no raise


def test_cross_validate_sidecar_config_bos_eos_skipped_when_gguf_side_none():
    """Minor 8: a GGUF with no bos_token_id key must SKIP that comparison
    (not hard-fail it as a "disagreement"), and report the skip.
    """
    hparams = _sample_hparams(bos_token_id=None)
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["bos_token_id"] = 999  # sidecar has an opinion; GGUF simply doesn't carry the key
    skip_notes = g2h.cross_validate_sidecar_config(
        sidecar, hparams, vocab_size=151936, tie_word_embeddings=False
    )
    assert any("bos_token_id" in note for note in skip_notes)


def test_cross_validate_sidecar_config_error_message_not_scoped_to_weight_relevant_only():
    """Minor 8: the error message must not claim "weight-relevant" when the
    field that actually tripped is hidden_act (a behavior field, not a
    weight-shape field).
    """
    hparams = _sample_hparams()
    sidecar = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar["hidden_act"] = "gelu"
    with pytest.raises(g2h.SidecarMismatchError) as excinfo:
        g2h.cross_validate_sidecar_config(sidecar, hparams, vocab_size=151936, tie_word_embeddings=False)
    assert "weight-relevant field(s)" not in str(excinfo.value)


def test_unvalidated_sidecar_keys_lists_pass_through():
    hparams = _sample_hparams()
    expected = g2h.reconstruct_config(hparams, vocab_size=151936, tie_word_embeddings=False)
    sidecar = dict(expected)
    sidecar["transformers_version"] = "4.55.0"
    sidecar["_name_or_path"] = "Qwen/Qwen3-8B"
    unvalidated = g2h.unvalidated_sidecar_keys(sidecar, expected)
    assert unvalidated == ["_name_or_path", "transformers_version"]


# ============================================================================
# Tokenizer sidecar validation (requirement 4)
# ============================================================================


def _matching_tokenizer_json(vocab_size=8):
    vocab = {f"tok{i}": i for i in range(vocab_size)}
    return json.dumps({"model": {"vocab": vocab}, "added_tokens": []})


def test_validate_tokenizer_sidecar_wrong_vocab_size_hard_fails(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    _gguf_mod2, reader = g2h.open_reader(gguf_path)
    hparams = g2h.read_qwen3_hparams(reader)

    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()

    with pytest.raises(g2h.SidecarMismatchError, match="tokenizer.ggml.tokens"):
        g2h.validate_tokenizer_sidecar(reader, hparams, vocab_size=999, sidecar_dir=sidecar_dir)


def test_validate_tokenizer_sidecar_wrong_special_token_hard_fails(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)  # bos=1 ("tok1"), eos=2 ("tok2")
    _gguf_mod2, reader = g2h.open_reader(gguf_path)
    hparams = g2h.read_qwen3_hparams(reader)

    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()
    # sidecar's vocab disagrees with the GGUF's: id 1 is "WRONG", not "tok1"
    bad_vocab = {f"tok{i}": i for i in range(VOCAB)}
    bad_vocab["WRONG"] = bad_vocab.pop("tok1")
    (sidecar_dir / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": bad_vocab}, "added_tokens": []}), encoding="utf-8"
    )

    with pytest.raises(g2h.SidecarMismatchError, match="bos_token_id"):
        g2h.validate_tokenizer_sidecar(reader, hparams, vocab_size=VOCAB, sidecar_dir=sidecar_dir)


def test_validate_tokenizer_sidecar_matching_passes(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    _gguf_mod2, reader = g2h.open_reader(gguf_path)
    hparams = g2h.read_qwen3_hparams(reader)

    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()
    (sidecar_dir / "tokenizer.json").write_text(_matching_tokenizer_json(VOCAB), encoding="utf-8")

    g2h.validate_tokenizer_sidecar(reader, hparams, vocab_size=VOCAB, sidecar_dir=sidecar_dir)  # no raise


def test_validate_tokenizer_sidecar_no_tokenizer_json_is_a_noop(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    _gguf_mod2, reader = g2h.open_reader(gguf_path)
    hparams = g2h.read_qwen3_hparams(reader)

    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()  # no tokenizer.json at all
    g2h.validate_tokenizer_sidecar(reader, hparams, vocab_size=VOCAB, sidecar_dir=sidecar_dir)  # no raise


# ============================================================================
# Full-pipeline integration: build a tiny real GGUF, run convert(), check
# everything the manifest contract promises.
# ============================================================================

PER_LAYER_ROLES = (
    ("attn_norm", "F32"),
    ("attn_q", "Q4_K"),
    ("attn_q_norm", "F32"),
    ("attn_k", "Q6_K"),
    ("attn_k_norm", "F32"),
    ("attn_v", "Q4_K"),
    ("attn_output", "Q4_K"),
    ("ffn_norm", "F32"),
    ("ffn_gate", "Q4_K"),
    ("ffn_down", "Q6_K"),
    ("ffn_up", "Q4_K"),
)

HIDDEN = 256
FFN = 512
N_HEADS = 2
N_KV_HEADS = 2
HEAD_DIM = 128
N_LAYERS = 2
VOCAB = 8


def _shape_for(role, tie_word_embeddings):
    if role == "token_embd":
        return (VOCAB, HIDDEN)
    if role == "output_norm":
        return (HIDDEN,)
    if role == "output":
        return (VOCAB, HIDDEN)
    if role == "attn_norm" or role == "ffn_norm":
        return (HIDDEN,)
    if role in ("attn_q_norm", "attn_k_norm"):
        return (HEAD_DIM,)
    if role == "attn_q":
        return (N_HEADS * HEAD_DIM, HIDDEN)
    if role == "attn_k" or role == "attn_v":
        return (N_KV_HEADS * HEAD_DIM, HIDDEN)
    if role == "attn_output":
        return (HIDDEN, N_HEADS * HEAD_DIM)
    if role == "ffn_gate" or role == "ffn_up":
        return (FFN, HIDDEN)
    if role == "ffn_down":
        return (HIDDEN, FFN)
    raise AssertionError(role)


def build_mini_qwen3_gguf(path, gguf_mod, *, tie_word_embeddings=False, seed=0):
    """Write a tiny (2-layer) real qwen3 GGUF via gguf.GGUFWriter.

    Returns ``{hf_name: np.ndarray(float32)}`` -- the EXPECTED dequantized
    value of every tensor, keyed by its HF name, for the caller to compare
    convert()'s output against.
    """
    rng = np.random.RandomState(seed)
    writer = gguf_mod.GGUFWriter(str(path), "qwen3")
    writer.add_uint32("qwen3.block_count", N_LAYERS)
    writer.add_uint32("qwen3.embedding_length", HIDDEN)
    writer.add_uint32("qwen3.feed_forward_length", FFN)
    writer.add_uint32("qwen3.attention.head_count", N_HEADS)
    writer.add_uint32("qwen3.attention.head_count_kv", N_KV_HEADS)
    writer.add_uint32("qwen3.attention.key_length", HEAD_DIM)
    writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_float32("qwen3.rope.freq_base", 1000000.0)
    writer.add_uint32("qwen3.context_length", 4096)
    writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
    # bos=1 -> "tok1", eos=2 -> "tok2" -- a matching --sidecar-dir tokenizer.json
    # (see MATCHING_SIDECAR_TOKENIZER_JSON below) must agree on these two strings.
    writer.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(VOCAB)])

    expected = {}

    def add_f32(gguf_name, hf_name, shape):
        arr = rng.uniform(-1, 1, size=shape).astype(np.float32)
        writer.add_tensor(gguf_name, arr)
        expected[hf_name] = arr

    def add_q4k(gguf_name, hf_name, shape):
        rows, cols = shape
        assert cols % 256 == 0
        raw = _q4k_bytes(cols // 256, rows, block=Q4_K_UNIFORM5_BLOCK)
        writer.add_tensor(gguf_name, raw, raw_dtype=gguf_mod.GGMLQuantizationType.Q4_K)
        expected[hf_name] = np.full(shape, 5.0, dtype=np.float32)

    def add_q6k(gguf_name, hf_name, shape):
        rows, cols = shape
        assert cols % 256 == 0
        raw = _q6k_bytes(cols // 256, rows, block=Q6_K_UNIFORM5_BLOCK)
        writer.add_tensor(gguf_name, raw, raw_dtype=gguf_mod.GGMLQuantizationType.Q6_K)
        expected[hf_name] = np.full(shape, 5.0, dtype=np.float32)

    add_q4k("token_embd.weight", "model.embed_tokens.weight", _shape_for("token_embd", tie_word_embeddings))
    add_f32("output_norm.weight", "model.norm.weight", _shape_for("output_norm", tie_word_embeddings))
    if not tie_word_embeddings:
        add_q4k("output.weight", "lm_head.weight", _shape_for("output", tie_word_embeddings))

    role_to_hf = {
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_q_norm": "self_attn.q_norm",
        "attn_k": "self_attn.k_proj",
        "attn_k_norm": "self_attn.k_norm",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "ffn_norm": "post_attention_layernorm",
        "ffn_gate": "mlp.gate_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_up": "mlp.up_proj",
    }
    adders = {"F32": add_f32, "Q4_K": add_q4k, "Q6_K": add_q6k}
    for bid in range(N_LAYERS):
        for role, qtype in PER_LAYER_ROLES:
            gguf_name = f"blk.{bid}.{role}.weight"
            hf_name = f"model.layers.{bid}.{role_to_hf[role]}.weight"
            shape = _shape_for(role, tie_word_embeddings)
            adders[qtype](gguf_name, hf_name, shape)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return expected


def _read_all_tensors_from_shards(out_dir: Path) -> dict:
    """Read every tensor out of a convert()-written directory's safetensors shards."""
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    by_file: dict = {}
    for name, filename in index["weight_map"].items():
        by_file.setdefault(filename, []).append(name)

    tensors = {}
    for filename, names in by_file.items():
        header, data_start = g2h.read_safetensors_header(out_dir / filename)
        raw = (out_dir / filename).read_bytes()
        for name in names:
            entry = header[name]
            start, end = entry["data_offsets"]
            buf = raw[data_start + start : data_start + end]
            tensors[name] = np.frombuffer(buf, dtype="<f4").reshape(entry["shape"]).copy()
    return tensors


EXPECTED_MANIFEST_TOP_LEVEL_KEYS = {
    "schema_version",
    "base_scheme",
    "source_gguf",
    "expected_gguf_sha256",
    "tool",
    "tensor_census",
    "tensors",
    "output",
    "sidecars",
    "permutation",
    "determinism",
    "created_utc",
}


def test_convert_full_pipeline_tiny_qwen3(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    expected = build_mini_qwen3_gguf(gguf_path, gguf_mod, tie_word_embeddings=False)

    out_dir = tmp_path / "out"
    manifest = g2h.convert(
        gguf_path=gguf_path,
        out_dir=out_dir,
        skip_source_hash=True,
        shard_max_bytes=g2h.DEFAULT_SHARD_MAX_BYTES,
    )

    # --- manifest contract: exact top-level key set (requirement 8) ------
    assert set(manifest.keys()) == EXPECTED_MANIFEST_TOP_LEVEL_KEYS
    assert manifest["schema_version"] == 1
    assert manifest["base_scheme"] == "dequant_q4km"
    assert manifest["source_gguf"]["sha256"] is None  # skip_source_hash
    assert manifest["source_gguf"]["size_bytes"] == gguf_path.stat().st_size
    assert manifest["permutation"] == {
        "applied": False,
        "evidence": g2h.PERMUTATION_EVIDENCE,
    }

    # --- census: 11 tensors/layer * 2 layers + 3 non-block (embd/out/out_norm)
    n_expected_tensors = 11 * N_LAYERS + 3
    assert manifest["tensor_census"]["total"] == n_expected_tensors == len(expected)
    assert len(manifest["tensors"]) == n_expected_tensors
    by_qtype = manifest["tensor_census"]["by_qtype"]
    # F32 = 4 norms/layer * N_LAYERS + 1 (output_norm)
    assert by_qtype["F32"] == 4 * N_LAYERS + 1

    # --- config.json reconstructed correctly -------------------------------
    config = json.loads((out_dir / "config.json").read_text())
    assert config["architectures"] == ["Qwen3ForCausalLM"]
    assert config["hidden_size"] == HIDDEN
    assert config["tie_word_embeddings"] is False
    assert config["torch_dtype"] == "float32"

    # --- every dequantized tensor matches what we wrote in -----------------
    on_disk = _read_all_tensors_from_shards(out_dir)
    assert set(on_disk) == set(expected)
    for name, arr in expected.items():
        np.testing.assert_array_equal(on_disk[name], arr)

    # --- sidecars: reconstructed path, no tokenizer files ------------------
    assert manifest["sidecars"]["source"] == "reconstructed_from_gguf"
    assert set(manifest["sidecars"]["files"]) == {"config.json"}
    for fname in g2h.TOKENIZER_FILENAMES:
        assert not (out_dir / fname).exists()

    # --- determinism: content_digest + actual shard bytes stable across runs
    out_dir2 = tmp_path / "out2"
    manifest2 = g2h.convert(gguf_path=gguf_path, out_dir=out_dir2, skip_source_hash=True)
    assert manifest["determinism"]["content_digest"] == manifest2["determinism"]["content_digest"]
    for filename, sha in manifest["output"]["files"].items():
        assert manifest2["output"]["files"][filename] == sha
        assert (out_dir / filename).read_bytes() == (out_dir2 / filename).read_bytes()


def test_convert_determinism_scope_manifest_differs_only_in_timestamp_and_commit(tmp_path, gguf_mod):
    """Requirement 13: dequant_manifest.json is NOT byte-identical across
    runs (created_utc, tool.llama_cpp_checkout.commit) -- but EVERYTHING
    ELSE in it must be. Masking exactly those two fields and comparing the
    rest catches any OTHER accidental source of non-determinism the
    output.files-only check above would miss (e.g. tensor ordering, a
    dict that isn't actually sorted, a stray absolute path).
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    manifest1 = g2h.convert(gguf_path=gguf_path, out_dir=tmp_path / "out1", skip_source_hash=True)
    manifest2 = g2h.convert(gguf_path=gguf_path, out_dir=tmp_path / "out2", skip_source_hash=True)

    def masked(m):
        m = json.loads(json.dumps(m))  # deep copy
        m["created_utc"] = None
        m["tool"]["llama_cpp_checkout"]["commit"] = None
        return m

    assert masked(manifest1) == masked(manifest2)
    # created_utc is a real timestamp on each run (not asserting the two
    # differ -- they could tie at second resolution -- just that it's there).
    assert isinstance(manifest1["created_utc"], str) and isinstance(manifest2["created_utc"], str)


def test_convert_manifest_per_tensor_sha_and_content_digest_are_independently_verifiable(tmp_path, gguf_mod):
    """Requirement 14: recompute every per-tensor sha256 and the
    content_digest from the KNOWN expected arrays, independently of
    convert()'s own internals, and check the manifest matches exactly.
    """
    import hashlib

    gguf_path = tmp_path / "mini.gguf"
    expected = build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_dir = tmp_path / "out"
    manifest = g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)

    recomputed_by_gguf_name = {}
    for row in manifest["tensors"]:
        arr = expected[row["hf_name"]]
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        recomputed_sha = hashlib.sha256(raw).hexdigest()
        assert row["sha256"] == recomputed_sha, f"{row['gguf_name']}: sha256 mismatch"
        recomputed_by_gguf_name[row["gguf_name"]] = recomputed_sha

    recomputed_digest = hashlib.sha256(
        "\n".join(f"{name}:{sha}" for name, sha in sorted(recomputed_by_gguf_name.items())).encode("utf-8")
    ).hexdigest()
    assert manifest["determinism"]["content_digest"] == recomputed_digest


# ============================================================================
# Atomic publish (requirement 1)
# ============================================================================


def test_convert_refuses_non_empty_out_dir(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "unrelated_file.txt").write_text("pre-existing", encoding="utf-8")

    with pytest.raises(g2h.DestinationNotEmptyError, match=str(out_dir)):
        g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)

    # untouched -- convert() must not have modified the pre-existing content
    assert (out_dir / "unrelated_file.txt").read_text(encoding="utf-8") == "pre-existing"
    assert list(out_dir.iterdir()) == [out_dir / "unrelated_file.txt"]


def test_convert_allows_existing_empty_out_dir(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # exists, but empty -- must be allowed

    manifest = g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)
    assert (out_dir / "config.json").exists()
    assert manifest["tensor_census"]["total"] > 0


def test_convert_refuses_out_dir_that_is_a_file(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_dir = tmp_path / "out"
    out_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(g2h.DestinationNotEmptyError, match="not a directory"):
        g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)
    assert out_dir.read_text(encoding="utf-8") == "not a directory"


def test_convert_refuses_out_dir_that_is_a_symlink_to_empty_dir(tmp_path, gguf_mod):
    """Note-item 1: without an explicit is_symlink() refusal, this would
    pass the empty-destination check, run the FULL conversion, and only
    die at publish time with a raw NotADirectoryError from os.replace.
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    real_target = tmp_path / "real_empty_dir"
    real_target.mkdir()
    out_symlink = tmp_path / "out_link"
    out_symlink.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(g2h.DestinationNotEmptyError, match="symlink"):
        g2h.convert(gguf_path=gguf_path, out_dir=out_symlink, skip_source_hash=True)

    # untouched: the real target dir is still empty, no conversion happened
    assert list(real_target.iterdir()) == []
    assert out_symlink.is_symlink()


def test_convert_refuses_out_dir_that_is_a_dangling_symlink(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_symlink = tmp_path / "out_link"
    out_symlink.symlink_to(tmp_path / "does_not_exist_at_all")

    with pytest.raises(g2h.DestinationNotEmptyError, match="symlink"):
        g2h.convert(gguf_path=gguf_path, out_dir=out_symlink, skip_source_hash=True)
    assert out_symlink.is_symlink()
    assert not out_symlink.exists()  # still dangling, nothing materialized


def test_convert_hash_mismatch_leaves_no_final_dir_and_no_temp_litter(tmp_path, gguf_mod):
    """Requirement 1: a refused source must leave NO output behind at all --
    not a partial directory, not a leftover temp sibling.
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_dir = tmp_path / "out"

    with pytest.raises(g2h.SourceHashMismatchError, match="0" * 64):
        g2h.convert(gguf_path=gguf_path, out_dir=out_dir, expected_sha256="0" * 64)

    assert not out_dir.exists()
    # no stray ".out.tmp-*" sibling directories left in out_dir's parent
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]
    assert leftovers == []


def test_convert_hash_verified_before_any_dequant_work(tmp_path, gguf_mod, monkeypatch):
    """Requirement 1: the hash gate must run BEFORE dequantize_tensor is
    ever called -- not just before the write. A version that hashed AFTER
    writing shards would still pass ``test_convert_hash_mismatch_leaves_no_
    final_dir`` if it cleaned up afterward, but would have wastefully
    dequantized first; this test catches that ordering regression directly.
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    def _boom(*args, **kwargs):
        raise AssertionError("dequantize_tensor must not be called before the hash gate")

    monkeypatch.setattr(g2h, "dequantize_tensor", _boom)
    with pytest.raises(g2h.SourceHashMismatchError, match="0" * 64):
        g2h.convert(
            gguf_path=gguf_path,
            out_dir=tmp_path / "out",
            expected_sha256="0" * 64,
        )


def test_convert_success_path_leaves_no_temp_litter(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_dir = tmp_path / "out"
    g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)

    siblings = list(tmp_path.iterdir())
    assert out_dir in siblings
    tmp_litter = [p for p in siblings if p.name.startswith(".out.tmp-")]
    assert tmp_litter == []


def test_convert_failure_after_writes_started_leaves_no_final_dir(tmp_path, gguf_mod, monkeypatch):
    """A failure that happens AFTER some shard bytes have been written into
    the temp dir (but before publish) must still never leave anything at
    the final --out path, and must clean up its temp directory.
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_dir = tmp_path / "out"

    call_count = {"n": 0}
    orig_write = g2h.write_safetensors_file

    def flaky_write(path, tensor_specs, metadata=None):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated failure mid-write")
        return orig_write(path, tensor_specs, metadata=metadata)

    monkeypatch.setattr(g2h, "write_safetensors_file", flaky_write)
    with pytest.raises(RuntimeError, match="simulated failure mid-write"):
        g2h.convert(
            gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True, shard_max_bytes=200_000
        )

    assert not out_dir.exists()
    assert [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")] == []


def test_convert_failure_cleans_up_newly_created_parent_dirs(tmp_path, gguf_mod, monkeypatch):
    """Note-item 5: a conversion that fails after the gates must not leave
    newly-created, now-empty ancestor directories of --out behind. Uses a
    multi-level --out path (a/b/c/out) where NONE of a/b/c exist yet, so a
    naive ``mkdir(parents=True)`` would create all three and abandon them
    on failure.
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    # one PRE-EXISTING ancestor, one PRE-EXISTING sibling file inside it (so
    # we can prove pre-existing content is never touched by the cleanup)
    pre_existing = tmp_path / "a"
    pre_existing.mkdir()
    (pre_existing / "keep_me.txt").write_text("do not delete", encoding="utf-8")

    out_dir = pre_existing / "b" / "c" / "out"  # "b" and "c" do NOT exist yet

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(g2h, "dequantize_tensor", _boom)
    with pytest.raises(RuntimeError, match="simulated failure"):
        g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)

    # "b" and "c" (created by this run) must be gone again
    assert not (pre_existing / "b").exists()
    # the pre-existing ancestor and ITS content must be untouched
    assert pre_existing.exists()
    assert (pre_existing / "keep_me.txt").read_text(encoding="utf-8") == "do not delete"
    assert not out_dir.exists()


def test_convert_success_does_not_disturb_pre_existing_parent_dirs(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    pre_existing = tmp_path / "a"
    pre_existing.mkdir()
    (pre_existing / "keep_me.txt").write_text("do not delete", encoding="utf-8")
    out_dir = pre_existing / "b" / "c" / "out"

    manifest = g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)

    assert (pre_existing / "keep_me.txt").read_text(encoding="utf-8") == "do not delete"
    assert (out_dir / "config.json").exists()
    assert manifest["tensor_census"]["total"] > 0


# ============================================================================
# Memory / streaming (requirement 2)
# ============================================================================


def test_convert_dequantizes_shard_by_shard_not_all_upfront(tmp_path, gguf_mod, monkeypatch):
    """Requirement 2: pass 2 must dequantize ONE SHARD's tensors, write
    them, THEN move to the next shard -- never dequantize everything
    upfront and write afterward (which is what the original, non-streaming
    implementation did, and would have made every write call see the FULL
    tensor count already dequantized, not just that shard's).
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    dequant_count = 0
    write_calls = []  # (shard_size, cumulative_dequant_count_at_this_write)

    orig_dequant = g2h.dequantize_tensor
    orig_write = g2h.write_safetensors_file

    def counting_dequant(gguf_mod_, t):
        nonlocal dequant_count
        dequant_count += 1
        return orig_dequant(gguf_mod_, t)

    def counting_write(path, tensor_specs, metadata=None):
        write_calls.append((len(tensor_specs), dequant_count))
        return orig_write(path, tensor_specs, metadata=metadata)

    monkeypatch.setattr(g2h, "dequantize_tensor", counting_dequant)
    monkeypatch.setattr(g2h, "write_safetensors_file", counting_write)

    g2h.convert(
        gguf_path=gguf_path, out_dir=tmp_path / "out", skip_source_hash=True, shard_max_bytes=200_000
    )

    assert len(write_calls) > 1, "fixture did not exercise multiple shards"
    running_total = 0
    for shard_size, cumulative_at_write in write_calls:
        running_total += shard_size
        assert cumulative_at_write == running_total, (
            "dequantization ran ahead of the write loop -- not streaming "
            f"shard-by-shard (expected {running_total} tensors dequantized "
            f"by this write, got {cumulative_at_write})"
        )
    assert dequant_count == running_total  # every tensor dequantized exactly once, by the end


def test_convert_releases_each_shards_tensors_before_the_next(tmp_path, gguf_mod, monkeypatch):
    """A live-object check complementing the ordering test above: by the
    time shard N+1 starts dequantizing, shard N's arrays must actually be
    garbage-collectable (peak RSS bound, requirement 2), not merely
    "written before the next dequant call" in wall-clock order.
    """
    import gc
    import weakref

    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    live_refs = []
    orig_dequant = g2h.dequantize_tensor

    def tracking_dequant(gguf_mod_, t):
        arr = orig_dequant(gguf_mod_, t)
        live_refs.append(weakref.ref(arr))
        return arr

    monkeypatch.setattr(g2h, "dequantize_tensor", tracking_dequant)

    orig_write = g2h.write_safetensors_file
    live_count_at_write = []

    def counting_write(path, tensor_specs, metadata=None):
        gc.collect()
        live_count_at_write.append(sum(1 for r in live_refs if r() is not None))
        return orig_write(path, tensor_specs, metadata=metadata)

    monkeypatch.setattr(g2h, "write_safetensors_file", counting_write)

    g2h.convert(
        gguf_path=gguf_path, out_dir=tmp_path / "out", skip_source_hash=True, shard_max_bytes=200_000
    )

    assert len(live_count_at_write) > 1
    # at the LAST write, only that shard's own tensors should still be alive
    # -- not the cumulative total across every prior shard too.
    assert live_count_at_write[-1] < len(live_refs)


# ============================================================================
# Missing/invalid --sidecar-dir (minor 9)
# ============================================================================


def test_convert_sidecar_dir_missing_config_json_raises_tool_domain_error(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()  # exists, but no config.json inside

    with pytest.raises(g2h.SidecarNotFoundError, match="config.json"):
        g2h.convert(
            gguf_path=gguf_path, out_dir=tmp_path / "out", sidecar_dir=sidecar_dir, skip_source_hash=True
        )


def test_convert_sidecar_dir_does_not_exist_raises_tool_domain_error(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    with pytest.raises(g2h.SidecarNotFoundError, match="does_not_exist"):
        g2h.convert(
            gguf_path=gguf_path,
            out_dir=tmp_path / "out",
            sidecar_dir=tmp_path / "does_not_exist",
            skip_source_hash=True,
        )


# ============================================================================
# transformers loadability (requirement 6) -- skips locally, bites on the box
# ============================================================================


def test_converted_dir_loads_with_transformers(tmp_path, gguf_mod):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    transformers = pytest.importorskip("transformers")

    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_dir = tmp_path / "out"
    g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(str(out_dir))
    assert model.config.model_type == "qwen3"


def test_convert_tied_embeddings_omits_output_tensor(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini_tied.gguf"
    expected = build_mini_qwen3_gguf(gguf_path, gguf_mod, tie_word_embeddings=True)
    out_dir = tmp_path / "out"
    manifest = g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)

    assert "lm_head.weight" not in expected
    config = json.loads((out_dir / "config.json").read_text())
    assert config["tie_word_embeddings"] is True
    hf_names = {t["hf_name"] for t in manifest["tensors"]}
    assert "lm_head.weight" not in hf_names


def test_convert_multi_shard_split_end_to_end(tmp_path, gguf_mod):
    """Force a tiny --shard-max-bytes through the real convert() path (not
    just the shard_tensor_names unit test) and check every requirement-8/9
    invariant: no tensor split across shards, weight_map covers every
    tensor exactly once, total_bytes matches, and every tensor's value
    still round-trips correctly out of its (now non-default) shard file.
    """
    gguf_path = tmp_path / "mini.gguf"
    expected = build_mini_qwen3_gguf(gguf_path, gguf_mod)
    out_dir = tmp_path / "out"

    # Small enough that a single ~256*256*4-byte tensor already forces a
    # new shard, but not so small that no tensor can ever fit -- exercises
    # genuine multi-shard splitting for this fixture's tensor sizes.
    manifest = g2h.convert(
        gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True, shard_max_bytes=200_000
    )

    shard_files = [f for f in manifest["output"]["files"] if f.endswith(".safetensors")]
    assert len(shard_files) > 1, "fixture did not actually exercise multi-shard splitting"
    for fname in shard_files:
        assert fname.startswith("model-") and "-of-" in fname
        assert (out_dir / fname).exists()

    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(expected)  # every tensor covered exactly once
    assert set(index["weight_map"].values()) == set(shard_files)
    assert index["metadata"]["total_size"] == manifest["output"]["total_bytes"]
    assert manifest["output"]["total_bytes"] == sum(arr.nbytes for arr in expected.values())

    on_disk = _read_all_tensors_from_shards(out_dir)
    assert set(on_disk) == set(expected)
    for name, arr in expected.items():
        np.testing.assert_array_equal(on_disk[name], arr)


def test_plan_matches_convert_census_and_mapping(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    plan_result = g2h.plan(gguf_path)

    assert plan_result["arch"] == "qwen3"
    assert plan_result["block_count"] == N_LAYERS
    assert plan_result["mapping_proof"] == {
        "bijective": True,
        "unmapped": 0,
        "doubly_mapped": 0,
        "total_mapped": plan_result["tensor_census"]["total"],
    }

    out_dir = tmp_path / "out"
    manifest = g2h.convert(gguf_path=gguf_path, out_dir=out_dir, skip_source_hash=True)
    assert plan_result["tensor_census"] == manifest["tensor_census"]

    plan_map = {row["gguf_name"]: row["hf_name"] for row in plan_result["tensors"]}
    convert_map = {row["gguf_name"]: row["hf_name"] for row in manifest["tensors"]}
    assert plan_map == convert_map


def test_convert_skip_source_hash_and_expected_sha256_mutually_exclusive(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    with pytest.raises(ValueError, match="mutually exclusive"):
        g2h.convert(
            gguf_path=gguf_path,
            out_dir=tmp_path / "out",
            skip_source_hash=True,
            expected_sha256="deadbeef",
        )


def test_convert_source_hash_mismatch_hard_fails(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    with pytest.raises(g2h.SourceHashMismatchError, match="0" * 64):
        g2h.convert(
            gguf_path=gguf_path,
            out_dir=tmp_path / "out",
            expected_sha256="0" * 64,
        )


def test_convert_source_hash_recorded_when_not_skipped(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    actual_sha = g2h.sha256_file(gguf_path)
    manifest = g2h.convert(
        gguf_path=gguf_path, out_dir=tmp_path / "out", expected_sha256=actual_sha
    )
    assert manifest["source_gguf"]["sha256"] == actual_sha
    assert manifest["expected_gguf_sha256"] == actual_sha


# ============================================================================
# --expected-sha256 case-insensitivity + format validation (note-item 2)
# ============================================================================


def test_normalize_sha256_lowercases_a_valid_uppercase_digest():
    digest = "AB" * 32
    assert g2h.normalize_sha256(digest) == digest.lower()


@pytest.mark.parametrize("bad", ["deadbeef", "g" * 64, "a" * 63, "a" * 65, "", "ABCXYZ" + "0" * 58])
def test_normalize_sha256_rejects_non_hex_or_wrong_length(bad):
    with pytest.raises(ValueError):
        g2h.normalize_sha256(bad)


def test_convert_expected_sha256_is_case_insensitive(tmp_path, gguf_mod):
    """Note-item 2: a CORRECT but uppercase digest must not be refused after
    already hashing the (here, tiny; on the box, multi-GB) source file.
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    actual_sha = g2h.sha256_file(gguf_path)

    manifest = g2h.convert(
        gguf_path=gguf_path, out_dir=tmp_path / "out", expected_sha256=actual_sha.upper()
    )
    assert manifest["source_gguf"]["sha256"] == actual_sha  # always lowercase (hexdigest())
    assert manifest["expected_gguf_sha256"] == actual_sha  # normalized, not the uppercase input


def test_convert_expected_sha256_still_hard_fails_on_genuine_mismatch_case_insensitively(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    with pytest.raises(g2h.SourceHashMismatchError):
        g2h.convert(gguf_path=gguf_path, out_dir=tmp_path / "out", expected_sha256="F" * 64)


def test_convert_expected_sha256_bad_format_raises_value_error(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    with pytest.raises(ValueError):
        g2h.convert(gguf_path=gguf_path, out_dir=tmp_path / "out", expected_sha256="not-a-digest")


def test_cli_expected_sha256_bad_format_rejected_at_parse_time(tmp_path, gguf_mod, capsys):
    """Note-item 2 ("validate it's 64 hex chars at parse time"): argparse's
    own type= machinery should catch this before main() ever calls convert().
    """
    parser = g2h.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--gguf", "x.gguf", "--out", "y", "--expected-sha256", "not-a-digest"])
    assert "expected-sha256" in capsys.readouterr().err


def test_cli_expected_sha256_uppercase_accepted_and_lowercased_at_parse_time():
    parser = g2h.build_arg_parser()
    digest = "AB" * 32
    args = parser.parse_args(["--gguf", "x.gguf", "--out", "y", "--expected-sha256", digest])
    assert args.expected_sha256 == digest.lower()


# ============================================================================
# --skip-source-hash + --expected-sha256 via the CLI (minor 3)
# ============================================================================


def test_main_cli_mutually_exclusive_hash_flags_exit_cleanly_not_a_traceback(tmp_path, gguf_mod, capsys):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    with pytest.raises(SystemExit) as excinfo:
        g2h.main(
            [
                "--gguf",
                str(gguf_path),
                "--out",
                str(tmp_path / "out"),
                "--skip-source-hash",
                "--expected-sha256",
                "0" * 64,
            ]
        )
    assert excinfo.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_convert_sidecar_dir_cross_validates_copies_and_notes_dual_pin(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()
    sidecar_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "vocab_size": VOCAB,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_position_embeddings": 40960,  # deliberately different from GGUF's 4096 -- dual pin
        "torch_dtype": "bfloat16",  # deliberately different -- dual pin
    }
    (sidecar_dir / "config.json").write_text(json.dumps(sidecar_config), encoding="utf-8")
    (sidecar_dir / "tokenizer.json").write_text(_matching_tokenizer_json(VOCAB), encoding="utf-8")
    # special_tokens_map.json deliberately absent -- must be noted, not an error

    out_dir = tmp_path / "out"
    manifest = g2h.convert(
        gguf_path=gguf_path, out_dir=out_dir, sidecar_dir=sidecar_dir, skip_source_hash=True
    )

    assert manifest["sidecars"]["source"] == str(sidecar_dir)
    assert "dual-pin" in manifest["sidecars"]["notes"]
    assert "40960" in manifest["sidecars"]["notes"]
    assert set(manifest["sidecars"]["files"]) == {"config.json", "tokenizer.json"}
    assert (out_dir / "tokenizer.json").exists()
    assert not (out_dir / "special_tokens_map.json").exists()

    written_config = json.loads((out_dir / "config.json").read_text())
    assert written_config["torch_dtype"] == "float32"  # overridden to the true output dtype
    assert written_config["max_position_embeddings"] == 40960  # sidecar's value kept, untouched


def test_convert_sidecar_dir_mismatch_hard_fails(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()
    sidecar_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 9999,  # wrong on purpose
        "intermediate_size": FFN,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "vocab_size": VOCAB,
        "tie_word_embeddings": False,
        "max_position_embeddings": 4096,
        "torch_dtype": "float32",
    }
    (sidecar_dir / "config.json").write_text(json.dumps(sidecar_config), encoding="utf-8")

    with pytest.raises(g2h.SidecarMismatchError, match="hidden_size"):
        g2h.convert(gguf_path=gguf_path, out_dir=tmp_path / "out", sidecar_dir=sidecar_dir, skip_source_hash=True)


def test_main_cli_plan_mode_prints_json(tmp_path, gguf_mod, capsys):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    rc = g2h.main(["--gguf", str(gguf_path), "--plan"])
    assert rc == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["mode"] == "plan"
    assert result["arch"] == "qwen3"


def test_main_cli_requires_out_unless_plan(tmp_path, gguf_mod, capsys):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    with pytest.raises(SystemExit):
        g2h.main(["--gguf", str(gguf_path)])
    assert "--out is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--out", "/tmp/somewhere"],
        ["--sidecar-dir", "/tmp/somewhere"],
        ["--expected-sha256", "0" * 64],
        ["--skip-source-hash"],
        ["--shard-max-bytes", "123"],
    ],
)
def test_main_cli_plan_rejects_no_effect_flags(tmp_path, gguf_mod, capsys, extra_args):
    """Minor 11: --plan silently ignoring a write-path-only flag would hide
    an operator's mistaken belief that e.g. --expected-sha256 was actually
    being checked in dry-run mode.
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    with pytest.raises(SystemExit):
        g2h.main(["--gguf", str(gguf_path), "--plan", *extra_args])
    err = capsys.readouterr().err
    assert "no effect" in err
    assert extra_args[0] in err


def test_main_cli_plan_alone_is_fine(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    assert g2h.main(["--gguf", str(gguf_path), "--plan"]) == 0


def test_main_cli_plan_rejects_shard_max_bytes_explicitly_set_to_the_default(tmp_path, gguf_mod, capsys):
    """Note-item 4: --shard-max-bytes explicitly set to the SAME value as
    DEFAULT_SHARD_MAX_BYTES must still be refused under --plan -- a naive
    ``!= DEFAULT_SHARD_MAX_BYTES`` comparison would silently accept this
    one specific value while refusing every other explicit value, which is
    itself a footgun (an operator copy-pasting the documented default
    would see no warning that --plan ignored it).
    """
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    with pytest.raises(SystemExit):
        g2h.main(
            [
                "--gguf",
                str(gguf_path),
                "--plan",
                "--shard-max-bytes",
                str(g2h.DEFAULT_SHARD_MAX_BYTES),
            ]
        )
    err = capsys.readouterr().err
    assert "no effect" in err
    assert "--shard-max-bytes" in err


def test_cli_shard_max_bytes_defaults_to_none_sentinel():
    parser = g2h.build_arg_parser()
    args = parser.parse_args(["--gguf", "x.gguf", "--out", "y"])
    assert args.shard_max_bytes is None


def test_main_cli_convert_uses_default_shard_max_bytes_when_omitted(tmp_path, gguf_mod):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)
    rc = g2h.main(
        ["--gguf", str(gguf_path), "--out", str(tmp_path / "out"), "--skip-source-hash"]
    )
    assert rc == 0
    manifest = json.loads((tmp_path / "out" / "dequant_manifest.json").read_text())
    # a single shard for this tiny fixture confirms the (large) default was used
    shard_files = [f for f in manifest["output"]["files"] if f.endswith(".safetensors")]
    assert len(shard_files) == 1


# ============================================================================
# gguf-py sys.path scoping (minor 12)
# ============================================================================


def test_load_gguf_module_does_not_leave_sys_path_modified():
    before = list(sys.path)
    g2h.load_gguf_module()
    assert sys.path == before, "load_gguf_module must not permanently modify sys.path"


def test_load_gguf_module_returns_cached_module_without_touching_sys_path(gguf_mod):
    # gguf is already imported (via the module-scoped `gguf_mod` fixture);
    # calling again with a bogus dir must short-circuit on sys.modules
    # rather than trying (and failing) to insert a nonexistent path.
    before = list(sys.path)
    result = g2h.load_gguf_module(gguf_py_dir="/nonexistent/path/for/this/test")
    assert result is gguf_mod
    assert sys.path == before


# ============================================================================
# plan() never touches tensor payloads (minor 16)
# ============================================================================


def test_plan_never_dequantizes_tensor_payloads(tmp_path, gguf_mod, monkeypatch):
    gguf_path = tmp_path / "mini.gguf"
    build_mini_qwen3_gguf(gguf_path, gguf_mod)

    def _boom(*args, **kwargs):
        raise AssertionError("plan() must never call dequantize_tensor")

    monkeypatch.setattr(g2h, "dequantize_tensor", _boom)
    result = g2h.plan(gguf_path)
    assert result["tensor_census"]["total"] > 0


# ============================================================================
# Real-file checks (skip-if-absent, requirement 9): metadata-only against
# the actual pinned GGUF. Never touches tensor payload bytes.
# ============================================================================


@pytest.fixture(scope="module")
def real_plan_result():
    """``plan()`` against the real pinned GGUF, computed ONCE and shared --
    every assertion below reads this same dict (minor 16: exercise the
    real ``plan()`` function itself, not a hand-rolled reimplementation of
    its logic).
    """
    return g2h.plan(REAL_GGUF_PATH)


@pytest.mark.skipif(
    not (GGUF_PY_AVAILABLE and REAL_GGUF_PATH.exists()),
    reason=f"gguf-py absent or real GGUF not found at {REAL_GGUF_PATH}",
)
def test_real_pinned_gguf_plan_is_bijective_and_structurally_sound(real_plan_result):
    result = real_plan_result
    assert result["arch"] == "qwen3"
    assert result["mapping_proof"]["bijective"] is True
    assert result["mapping_proof"]["unmapped"] == 0
    assert result["mapping_proof"]["doubly_mapped"] == 0

    block_count = result["block_count"]
    census = result["tensor_census"]
    # Structural invariant, independent of the quant-mixing scheme specifics:
    # 11 per-layer tensors (module's QWEN3_HF_NAME_TEMPLATES per-layer roles)
    # + 3 non-block tensors (token_embd/output_norm/output).
    assert census["total"] == 11 * block_count + 3
    # Norms (attn_norm/attn_q_norm/attn_k_norm/ffn_norm per layer + the
    # final output_norm) are always F32 in llama.cpp -- never quantized.
    assert census["by_qtype"].get("F32") == 4 * block_count + 1
    assert set(census["by_qtype"]) <= set(g2h.SUPPORTED_QTYPE_NAMES)


@pytest.mark.skipif(
    not (GGUF_PY_AVAILABLE and REAL_GGUF_PATH.exists()),
    reason=f"gguf-py absent or real GGUF not found at {REAL_GGUF_PATH}",
)
def test_real_pinned_gguf_hparams_readable():
    _gguf_mod, reader = g2h.open_reader(REAL_GGUF_PATH)
    hparams = g2h.read_qwen3_hparams(reader)
    assert hparams["architecture"] == "qwen3"
    assert isinstance(hparams["block_count"], int) and hparams["block_count"] > 0
    assert isinstance(hparams["hidden_size"], int) and hparams["hidden_size"] > 0
    assert hparams["sliding_window"] is None, "real checkpoint unexpectedly declares a sliding window"
    g2h.assert_no_sliding_window(hparams)  # must not raise

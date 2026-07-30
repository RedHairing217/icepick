"""Tests for loratrain.verify_base_identity (offline GGUF <-> HF comparator).

All synthetic: a tiny hand-built GGUF v3 file (via struct.pack) standing in
for the real ~5 GB quantized file, and a matching pinned config.json /
tokenizer.json pair standing in for the real HF metadata fetch. No network,
no real corpus.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from loratrain import config
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


def test_main_fp16_receipt_gains_scheme_field(tmp_path):
    # T3: the existing fp16 path's receipt gains the additive "scheme" field;
    # every other receipt key/value is unaffected (byte-for-byte structural
    # comparator logic is untouched).
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
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["scheme"] == config.BASE_SCHEME_FP16
    assert receipt["verdict"] == "PASS"
    assert receipt["fp16_revision"] == vbi.FP16_REVISION  # unaffected by the additive field


# --- Dequant hash-chain identity mode (T3, 2026-07-30) -- check_dequant_manifest / --dequant-dir --


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# The REAL pin's preimage is a ~5GB file nobody has in a unit test, so
# EXPECTED_BASE_GGUF_SHA256 must be monkeypatched to a value whose preimage
# THIS file controls before any test exercises the new source-GGUF-file-hash
# gate. Test-quality fix #12: source_gguf.sha256 / expected_gguf_sha256 / the
# actual on-disk file hash are now three INDEPENDENTLY controllable values
# (build_dequant_dir defaults all three to agree with each other and with
# whatever the pin currently is, but every knob below can desync them).
_FIXTURE_GGUF_BYTES = b"pretend-source-gguf-bytes-for-dequant-tests"
_FIXTURE_GGUF_SHA256 = _sha256_bytes(_FIXTURE_GGUF_BYTES)


@pytest.fixture
def pinned_gguf_sha(monkeypatch):
    """Monkeypatch EXPECTED_BASE_GGUF_SHA256 to ``_FIXTURE_GGUF_SHA256``
    (whose preimage is ``_FIXTURE_GGUF_BYTES``, above) so a fully-consistent
    dequant fixture can actually pass the new source-GGUF-file-hash gate."""
    monkeypatch.setattr(vbi, "EXPECTED_BASE_GGUF_SHA256", _FIXTURE_GGUF_SHA256)
    return _FIXTURE_GGUF_SHA256


def _make_tensors(n: int, seed: str = "tensor") -> list:
    return [
        {
            "gguf_name": f"{seed}.{i}",
            "hf_name": f"model.layers.{i}.weight",
            "qtype": "F32",
            "shape": [1, 1],
            "sha256": _sha256_bytes(f"{seed}-{i}".encode()),
        }
        for i in range(n)
    ]


def build_dequant_dir(
    tmp_path,
    *,
    gguf_sha256=None,
    write_gguf_file=True,
    gguf_bytes=_FIXTURE_GGUF_BYTES,
    gguf_path=None,
    tensor_total=None,
    tensors_count=None,
    files_content=None,
    sidecar_files_content=None,
    manifest_overrides=None,
    write_config_json=True,
    write_manifest=True,
):
    """Write a synthetic gguf_to_hf.py-shaped output dir: dequant_manifest.json
    + config.json + the float32 tensor file(s) it names -- all consistent
    with each other by construction, matching the parallel agent's schema:
    {schema_version, base_scheme, source_gguf, expected_gguf_sha256, tool,
    tensor_census, tensors, output, sidecars, permutation, determinism,
    created_utc}.

    ``gguf_sha256`` (default: whatever ``vbi.EXPECTED_BASE_GGUF_SHA256``
    currently is -- i.e. the ``pinned_gguf_sha`` fixture's value, if a test
    requested it) is used for BOTH ``source_gguf.sha256`` and
    ``expected_gguf_sha256`` by default; a REAL local GGUF file is written
    at ``gguf_path`` (default ``tmp_path/"source.gguf"``) with ``gguf_bytes``
    (default ``_FIXTURE_GGUF_BYTES``, whose real sha256 IS
    ``_FIXTURE_GGUF_SHA256``) -- so the happy path is internally consistent
    end to end, and every one of these can be independently desynced for a
    negative test. ``tensor_total`` is the DECLARED ``tensor_census.total``;
    ``tensors_count`` (default: same as ``tensor_total``) is how many
    synthetic entries actually go in ``tensors[]`` -- independently
    controllable so a test can desync "declared total" from "pin" and
    "declared total" from "actual list length" separately.
    ``manifest_overrides`` patches the top-level manifest dict for
    negative-path tests; ``files_content``/``sidecar_files_content``
    override the {filename: bytes} payloads actually written to disk and
    declared in ``output.files``/``sidecars.files`` respectively.
    """
    d = tmp_path / "dequant_out"
    d.mkdir(parents=True, exist_ok=True)

    resolved_gguf_sha256 = gguf_sha256 if gguf_sha256 is not None else vbi.EXPECTED_BASE_GGUF_SHA256
    gguf_file = Path(gguf_path) if gguf_path is not None else (tmp_path / "source.gguf")
    if write_gguf_file:
        gguf_file.write_bytes(gguf_bytes)

    files_content = (
        files_content
        if files_content is not None
        else {"model-00001-of-00001.safetensors": b"pretend-float32-tensor-bytes"}
    )
    files_shas = {}
    for name, data in files_content.items():
        (d / name).write_bytes(data)
        files_shas[name] = _sha256_bytes(data)

    sidecar_files_content = sidecar_files_content or {}
    sidecar_shas = {}
    for name, data in sidecar_files_content.items():
        (d / name).write_bytes(data)
        sidecar_shas[name] = _sha256_bytes(data)

    declared_total = tensor_total if tensor_total is not None else config.EXPECTED_DEQUANT_TENSOR_TOTAL
    actual_tensors = _make_tensors(tensors_count if tensors_count is not None else declared_total)
    content_digest = vbi._compute_content_digest(actual_tensors)

    manifest = {
        "schema_version": 1,
        "base_scheme": "dequant_q4km",
        "source_gguf": {"path": str(gguf_file), "sha256": resolved_gguf_sha256, "size_bytes": len(gguf_bytes)},
        "expected_gguf_sha256": resolved_gguf_sha256,
        "tool": {"name": "gguf_to_hf", "version": "0.1", "llama_cpp_checkout": {"path": "/x", "commit": "abc"}, "gguf_py_version": "0.1"},
        "tensor_census": {"total": declared_total, "by_qtype": {}},
        "tensors": actual_tensors,
        "output": {"dtype": "float32", "files": files_shas, "total_bytes": sum(len(v) for v in files_content.values())},
        "sidecars": {"files": sidecar_shas} if sidecar_shas else {},
        "permutation": {"applied": False, "evidence": None},
        "determinism": {"content_digest": content_digest},
        "created_utc": "2026-07-30T00:00:00Z",
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)

    if write_manifest:
        (d / "dequant_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if write_config_json:
        (d / "config.json").write_text("{}", encoding="utf-8")

    return d


def _by_name(result):
    return {name: ok for name, ok, _ in result["checks"]}


def _detail(result, name):
    return next(d for n, _, d in result["checks"] if n == name)


def test_check_dequant_manifest_all_pass(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is True
    assert all(ok for _, ok, _ in result["checks"])
    assert result["manifest_sha256"] is not None
    assert len(result["manifest_sha256"]) == 64
    assert result["content_digest"] is not None


# --- Symlink escape (review fix #1, round 3): _find_symlinks / _resolves_within --


def test_check_dequant_manifest_symlink_free_fixture_still_passes(tmp_path, pinned_gguf_sha):
    # The new "no symlinks anywhere" gate must not false-positive on a
    # perfectly ordinary, symlink-free directory.
    d = build_dequant_dir(tmp_path)
    result = vbi.check_dequant_manifest(d)
    by_name = _by_name(result)
    assert by_name["no symlinks anywhere under dequant_dir"] is True
    assert result["ok"] is True


def test_check_dequant_manifest_symlinked_declared_file_reproduces_full_facade_then_fails(tmp_path, pinned_gguf_sha):
    # The exact attack the review reproduced: swap the declared
    # model.safetensors for a symlink to a file OUTSIDE dequant_dir whose
    # content happens to match the recorded sha -- so the (unfixed)
    # re-hash gate alone would report PASS despite the directory
    # containing NO real weights of its own.
    d = build_dequant_dir(tmp_path)
    declared_name = "model-00001-of-00001.safetensors"
    declared_path = d / declared_name
    outside_target = tmp_path / "outside_weights.bin"
    outside_target.write_bytes(declared_path.read_bytes())  # same content/sha as the recorded value
    declared_path.unlink()
    declared_path.symlink_to(outside_target)

    result = vbi.check_dequant_manifest(d)

    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["no symlinks anywhere under dequant_dir"] is False
    assert declared_name in _detail(result, "no symlinks anywhere under dequant_dir")


def test_check_dequant_manifest_symlinked_subdir_fails(tmp_path, pinned_gguf_sha):
    # A stale/rogue shard hidden behind a symlinked subdirectory: os.walk
    # with followlinks=False must report the symlinked dir itself (never
    # descend into it), and the top-level gate must still fail on it.
    d = build_dequant_dir(tmp_path)
    shadow_dir = tmp_path / "shadow_shards"
    shadow_dir.mkdir()
    (shadow_dir / "stale.safetensors").write_bytes(b"a stale shard nobody declared")
    (d / "shards").symlink_to(shadow_dir, target_is_directory=True)

    result = vbi.check_dequant_manifest(d)

    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["no symlinks anywhere under dequant_dir"] is False
    assert "shards" in _detail(result, "no symlinks anywhere under dequant_dir")
    # And the shadow file itself must never have been silently scanned in:
    assert "stale.safetensors" not in _detail(result, "no orphan *.safetensors/*.bin/*index.json files on disk")


def test_check_dequant_manifest_unreadable_subdir_fails_closed(tmp_path, pinned_gguf_sha):
    # Review fix #2, round 4: a directory os.walk/rglob cannot LIST (mode
    # 0o111 -- traversable but not readable) used to be silently treated
    # as EMPTY by both the symlink scan and the orphan rglob, so a hidden
    # symlink or stale shard behind it passed every gate. Must now fail
    # closed on its own dedicated gate instead.
    d = build_dequant_dir(tmp_path)
    hidden_dir = d / "hidden"
    hidden_dir.mkdir()
    (hidden_dir / "stale.safetensors").write_bytes(b"an undeclared shard hidden behind an unlistable dir")
    hidden_dir.chmod(0o111)
    try:
        result = vbi.check_dequant_manifest(d)
    finally:
        hidden_dir.chmod(0o755)  # restore so tmp_path cleanup can remove the tree

    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["no unreadable directory under dequant_dir"] is False
    assert "hidden" in _detail(result, "no unreadable directory under dequant_dir")


def test_walk_dequant_tree_unreadable_dir_reported_not_swallowed(tmp_path):
    d = tmp_path / "root"
    d.mkdir()
    hidden = d / "locked"
    hidden.mkdir()
    (hidden / "secret.safetensors").write_bytes(b"x")
    hidden.chmod(0o111)
    try:
        walk = vbi._walk_dequant_tree(d)
    finally:
        hidden.chmod(0o755)

    assert walk["unreadable"] == ["locked"]
    # The file behind the unlistable dir must never surface as a "found" candidate either:
    assert walk["candidate_files"] == []


def test_find_symlinks_empty_for_symlink_free_tree(tmp_path):
    d = tmp_path / "clean"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")
    (d / "sub").mkdir()
    (d / "sub" / "b.txt").write_text("y", encoding="utf-8")
    assert vbi._find_symlinks(d) == []


def test_find_symlinks_detects_file_and_dir_symlinks(tmp_path):
    d = tmp_path / "dir_with_links"
    d.mkdir()
    target_file = tmp_path / "real.txt"
    target_file.write_text("x", encoding="utf-8")
    target_dir = tmp_path / "real_dir"
    target_dir.mkdir()
    (d / "link_file.txt").symlink_to(target_file)
    (d / "link_dir").symlink_to(target_dir, target_is_directory=True)

    found = vbi._find_symlinks(d)

    assert found == ["link_dir", "link_file.txt"]


def test_resolves_within_true_for_plain_descendant(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    child = root / "sub" / "file.txt"
    child.parent.mkdir(parents=True)
    child.write_text("x", encoding="utf-8")
    assert vbi._resolves_within(child, root) is True


def test_resolves_within_false_for_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    link = root / "escape.txt"
    link.symlink_to(outside)
    assert vbi._resolves_within(link, root) is False


def test_check_dequant_manifest_missing_manifest_fails(tmp_path):
    d = build_dequant_dir(tmp_path, write_manifest=False)
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    assert result["manifest"] is None
    by_name = _by_name(result)
    assert by_name == {"dequant_manifest.json exists": False}  # early return -- exactly one row


def test_check_dequant_manifest_bad_json_fails(tmp_path):
    d = build_dequant_dir(tmp_path, write_manifest=False)
    (d / "dequant_manifest.json").write_text("not json", encoding="utf-8")
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    assert result["manifest"] is None
    assert result["manifest_sha256"] is not None  # file existed and was hashed before the parse failure
    by_name = _by_name(result)
    assert by_name["dequant_manifest.json exists"] is True
    assert by_name["dequant_manifest.json is valid JSON"] is False


def test_check_dequant_manifest_non_dict_manifest_fails_cleanly(tmp_path):
    d = build_dequant_dir(tmp_path, write_manifest=False)
    (d / "dequant_manifest.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")  # valid JSON, not an object
    result = vbi.check_dequant_manifest(d)  # must not raise (review fix #10)
    assert result["ok"] is False
    assert result["manifest"] is None
    by_name = _by_name(result)
    assert by_name["dequant_manifest.json is valid JSON"] is True
    assert by_name["dequant_manifest.json decodes to a JSON object"] is False


def test_check_dequant_manifest_wrong_schema_version_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, manifest_overrides={"schema_version": 2})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["schema_version == 1"] is False
    assert by_name["base_scheme == 'dequant_q4km'"] is True  # unrelated gate unaffected


def test_check_dequant_manifest_wrong_base_scheme_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, manifest_overrides={"base_scheme": "something_else"})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["base_scheme == 'dequant_q4km'"] is False
    assert by_name["schema_version == 1"] is True  # unrelated gate unaffected


def test_check_dequant_manifest_wrong_gguf_sha_claim_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, manifest_overrides={"source_gguf": {"sha256": "0" * 64}})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["source_gguf.sha256 == pinned EXPECTED_BASE_GGUF_SHA256"] is False


def test_check_dequant_manifest_wrong_tensor_total_vs_pin_fails(tmp_path, pinned_gguf_sha):
    # tensor_total=1 with tensors_count defaulting to match (1) -- self
    # consistent with the ACTUAL list length, but mismatches the pinned
    # EXPECTED_DEQUANT_TENSOR_TOTAL specifically.
    d = build_dequant_dir(tmp_path, tensor_total=1)
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["tensor_census.total == EXPECTED_DEQUANT_TENSOR_TOTAL"] is False
    assert by_name["len(tensors) == tensor_census.total"] is True  # 1 declared == 1 actual


def test_check_dequant_manifest_tensors_length_mismatches_declared_total_fails(tmp_path, pinned_gguf_sha):
    # tensor_census.total stays == the pin (399), but the actual tensors[]
    # list is a different length -- independent from the pin-mismatch gate
    # above (test-quality fix #12/#13: these are two distinct invariants).
    d = build_dequant_dir(tmp_path, tensors_count=5)
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["tensor_census.total == EXPECTED_DEQUANT_TENSOR_TOTAL"] is True
    assert by_name["len(tensors) == tensor_census.total"] is False


def test_check_dequant_manifest_content_digest_mismatch_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, manifest_overrides={"determinism": {"content_digest": "wrong" * 12}})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["determinism.content_digest matches recomputed digest over tensors[]"] is False


def test_check_dequant_manifest_content_digest_never_just_copied(tmp_path, pinned_gguf_sha):
    # Review fix #3: even if the manifest's declared digest happens to be
    # SOME valid-looking 64-hex string, it must be the RECOMPUTED value
    # over tensors[], not whatever the manifest claims.
    d = build_dequant_dir(tmp_path, manifest_overrides={"determinism": {"content_digest": "0" * 64}})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["determinism.content_digest matches recomputed digest over tensors[]"] is False
    assert result["content_digest"] != "0" * 64  # the recomputed value, not the manifest's copy


def test_check_dequant_manifest_permutation_applied_true_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, manifest_overrides={"permutation": {"applied": True, "evidence": "reordered"}})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["permutation.applied is False"] is False


def test_check_dequant_manifest_expected_gguf_sha256_mismatch_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, manifest_overrides={"expected_gguf_sha256": "0" * 64})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["expected_gguf_sha256 == pinned EXPECTED_BASE_GGUF_SHA256"] is False


def test_check_dequant_manifest_expected_gguf_sha256_null_is_not_applicable(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, manifest_overrides={"expected_gguf_sha256": None})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is True
    by_name = _by_name(result)
    assert by_name["expected_gguf_sha256 == pinned EXPECTED_BASE_GGUF_SHA256"] is True


def test_check_dequant_manifest_tampered_output_file_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)
    (d / "model-00001-of-00001.safetensors").write_bytes(b"TAMPERED")
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["output.files re-hash"] is False


def test_check_dequant_manifest_missing_output_file_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)
    (d / "model-00001-of-00001.safetensors").unlink()
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["output.files re-hash"] is False


def test_check_dequant_manifest_empty_output_files_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, files_content={})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["output.files re-hash"] is False
    assert by_name["output.files includes >=1 *.safetensors entry"] is False


def test_check_dequant_manifest_missing_config_json_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, write_config_json=False)
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["config.json present"] is False


def test_check_dequant_manifest_unrelated_field_mismatch_only_fails_that_row(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, tensor_total=1)
    result = vbi.check_dequant_manifest(d)
    by_name = _by_name(result)
    assert by_name["config.json present"] is True  # unrelated gate unaffected


# --- Source-link file hash (review fix #3): the gate that actually touches disk --


def test_check_dequant_manifest_source_gguf_file_hash_passes_all_three_way(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)  # defaults: claim, pin, and actual file all agree
    result = vbi.check_dequant_manifest(d)
    by_name = _by_name(result)
    assert by_name["source GGUF file re-hash (== manifest claim == pin)"] is True
    assert result["ok"] is True


def test_check_dequant_manifest_source_gguf_gate_checks_pin_not_just_self_consistency(tmp_path):
    # Deliberately NOT using pinned_gguf_sha: the real (production)
    # EXPECTED_BASE_GGUF_SHA256 stays whatever it actually is. manifest's
    # source_gguf.sha256 and the actual on-disk file agree with EACH OTHER
    # (both _FIXTURE_GGUF_SHA256) but NEITHER agrees with the real pin --
    # test-quality fix #12: proves the gate compares against the pin
    # specifically, not merely internal self-consistency between the file
    # and the manifest's own claim.
    d = build_dequant_dir(tmp_path, gguf_sha256=_FIXTURE_GGUF_SHA256)
    result = vbi.check_dequant_manifest(d)
    by_name = _by_name(result)
    assert by_name["source GGUF file re-hash (== manifest claim == pin)"] is False
    assert by_name["source_gguf.sha256 == pinned EXPECTED_BASE_GGUF_SHA256"] is False


def test_check_dequant_manifest_source_gguf_file_tampered_fails_even_if_manifest_claims_pin(tmp_path, pinned_gguf_sha):
    # manifest.source_gguf.sha256 correctly claims the pin, but the actual
    # on-disk bytes don't hash to it -- proves the gate actually reads the
    # file rather than trusting the manifest's self-report.
    d = build_dequant_dir(tmp_path, gguf_bytes=b"TAMPERED-DOES-NOT-MATCH-THE-CLAIM")
    result = vbi.check_dequant_manifest(d)
    by_name = _by_name(result)
    assert by_name["source_gguf.sha256 == pinned EXPECTED_BASE_GGUF_SHA256"] is True  # manifest's own claim IS the pin
    assert by_name["source GGUF file re-hash (== manifest claim == pin)"] is False  # but the real file disagrees


def test_check_dequant_manifest_source_gguf_missing_file_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, write_gguf_file=False)
    result = vbi.check_dequant_manifest(d)
    by_name = _by_name(result)
    assert by_name["source GGUF file re-hash (== manifest claim == pin)"] is False


def test_check_dequant_manifest_skip_file_sha_bypasses_gguf_hash(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, write_gguf_file=False)  # would fail the file gate without skip
    result = vbi.check_dequant_manifest(d, skip_file_sha=True)
    by_name = _by_name(result)
    assert by_name["source GGUF file re-hash (== manifest claim == pin)"] is True
    assert result["ok"] is True  # every other gate in the default fixture still passes


def test_check_dequant_manifest_gguf_path_override_wins_over_manifest_path(tmp_path, pinned_gguf_sha):
    real_gguf = tmp_path / "real.gguf"
    real_gguf.write_bytes(_FIXTURE_GGUF_BYTES)
    d = build_dequant_dir(
        tmp_path,
        gguf_path=str(tmp_path / "manifest-claims-a-path-that-does-not-exist.gguf"),
        write_gguf_file=False,
    )
    result = vbi.check_dequant_manifest(d, gguf_path=real_gguf)
    by_name = _by_name(result)
    assert by_name["source GGUF file re-hash (== manifest claim == pin)"] is True


# --- Path-safety (review fix #1): output.files must never escape dequant_dir --


def test_check_dequant_manifest_absolute_output_path_refused(tmp_path, pinned_gguf_sha):
    escape_target = tmp_path / "outside_secret.txt"
    escape_target.write_text("must never be read by the re-hash gate", encoding="utf-8")
    d = build_dequant_dir(
        tmp_path,
        files_content={},
        manifest_overrides={
            "output": {"dtype": "float32", "files": {str(escape_target): "0" * 64}, "total_bytes": 0},
        },
    )
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["output.files re-hash"] is False
    assert "unsafe" in _detail(result, "output.files re-hash").lower()


def test_check_dequant_manifest_dotdot_output_path_refused(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(
        tmp_path,
        files_content={},
        manifest_overrides={
            "output": {"dtype": "float32", "files": {"../escape.safetensors": "0" * 64}, "total_bytes": 0},
        },
    )
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["output.files re-hash"] is False
    assert "unsafe" in _detail(result, "output.files re-hash").lower()


def test_check_dequant_manifest_requires_at_least_one_safetensors_entry(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, files_content={"model.bin": b"not-a-safetensors-file"})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["output.files re-hash"] is True  # the .bin file itself re-hashes fine
    assert by_name["output.files includes >=1 *.safetensors entry"] is False


# --- Sidecars (review fix #2a): same path-safety + re-hash treatment as output.files --


def test_check_dequant_manifest_sidecars_optional_when_absent(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)  # no sidecars declared
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is True
    by_name = _by_name(result)
    assert by_name["sidecars.files re-hash"] is True


def test_check_dequant_manifest_sidecar_rehash_ok(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, sidecar_files_content={"tokenizer_extra.json": b"sidecar-bytes"})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is True
    by_name = _by_name(result)
    assert by_name["sidecars.files re-hash"] is True


def test_check_dequant_manifest_sidecar_tampered_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, sidecar_files_content={"tokenizer_extra.json": b"sidecar-bytes"})
    (d / "tokenizer_extra.json").write_bytes(b"TAMPERED")
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["sidecars.files re-hash"] is False


def test_check_dequant_manifest_sidecar_dotdot_path_refused(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, manifest_overrides={"sidecars": {"files": {"../escape.json": "0" * 64}}})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["sidecars.files re-hash"] is False
    assert "unsafe" in _detail(result, "sidecars.files re-hash").lower()


# --- Index-file coverage + orphan completeness (review fix #2b) -------------


def test_check_dequant_manifest_index_json_uncovered_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)
    (d / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")  # not declared anywhere
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["model.safetensors.index.json covered by output.files/sidecars.files"] is False
    assert by_name["no orphan *.safetensors/*.bin/*index.json files on disk"] is False


def test_check_dequant_manifest_index_json_covered_by_output_files_ok(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, files_content={
        "model-00001-of-00001.safetensors": b"pretend-float32-tensor-bytes",
        "model.safetensors.index.json": b'{"weight_map": {}}',
    })
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is True
    by_name = _by_name(result)
    assert by_name["model.safetensors.index.json covered by output.files/sidecars.files"] is True
    assert by_name["no orphan *.safetensors/*.bin/*index.json files on disk"] is True


def test_check_dequant_manifest_index_json_covered_by_sidecars_ok(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, sidecar_files_content={"model.safetensors.index.json": b'{"weight_map": {}}'})
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is True
    by_name = _by_name(result)
    assert by_name["model.safetensors.index.json covered by output.files/sidecars.files"] is True


def test_check_dequant_manifest_no_index_json_on_disk_gate_not_applicable(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)  # default fixture never writes an index.json
    result = vbi.check_dequant_manifest(d)
    by_name = _by_name(result)
    assert by_name["model.safetensors.index.json covered by output.files/sidecars.files"] is True


def test_check_dequant_manifest_orphan_safetensors_file_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)
    (d / "stale-shard-from-previous-run.safetensors").write_bytes(b"orphan bytes")
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["no orphan *.safetensors/*.bin/*index.json files on disk"] is False


def test_check_dequant_manifest_orphan_bin_file_fails(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)
    (d / "pytorch_model.bin").write_bytes(b"orphan legacy shard")
    result = vbi.check_dequant_manifest(d)
    assert result["ok"] is False
    by_name = _by_name(result)
    assert by_name["no orphan *.safetensors/*.bin/*index.json files on disk"] is False


# --- --dequant-dir end to end -------------------------------------------------


def test_main_dequant_dir_passes_and_writes_receipt_with_chain(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path)
    receipt_path = tmp_path / "identity_receipt.json"

    rc = vbi.main(["--dequant-dir", str(d), "--receipt", str(receipt_path)])

    assert rc == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "PASS"
    assert receipt["scheme"] == config.BASE_SCHEME_DEQUANT
    assert receipt["chain"]["gguf_sha256"] == pinned_gguf_sha
    assert len(receipt["chain"]["dequant_manifest_sha256"]) == 64
    # Recomputed, not copied (review fix #3):
    manifest = json.loads((d / "dequant_manifest.json").read_text(encoding="utf-8"))
    assert receipt["chain"]["content_digest"] == vbi._compute_content_digest(manifest["tensors"])
    # Review fix #2, round 3: a fully-verified receipt is explicit about it.
    assert receipt["chain"]["source_verified"] is True
    assert "manifest_claimed_gguf_sha256" not in receipt["chain"]


def test_main_dequant_dir_skip_file_sha_receipt_marks_unverified(tmp_path, pinned_gguf_sha):
    # Review fix #2, round 3: a --skip-file-sha receipt used to be shaped
    # IDENTICALLY to a verified one (gguf_sha256 populated from the
    # manifest's unverified self-report either way) -- now the two are
    # structurally distinguishable without reading into checks[].
    d = build_dequant_dir(tmp_path, write_gguf_file=False)
    receipt_path = tmp_path / "identity_receipt.json"

    rc = vbi.main(["--dequant-dir", str(d), "--receipt", str(receipt_path), "--skip-file-sha"])

    assert rc == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["chain"]["source_verified"] is False
    assert receipt["chain"]["gguf_sha256"] is None
    assert receipt["chain"]["manifest_claimed_gguf_sha256"] == pinned_gguf_sha


def test_main_dequant_dir_fail_writes_no_receipt(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, manifest_overrides={"schema_version": 99})
    receipt_path = tmp_path / "identity_receipt.json"

    rc = vbi.main(["--dequant-dir", str(d), "--receipt", str(receipt_path)])

    assert rc == 2
    assert not receipt_path.exists()


def test_main_dequant_dir_skip_file_sha_flag_wired(tmp_path, pinned_gguf_sha):
    d = build_dequant_dir(tmp_path, write_gguf_file=False)  # would fail without --skip-file-sha
    receipt_path = tmp_path / "identity_receipt.json"

    rc = vbi.main(["--dequant-dir", str(d), "--receipt", str(receipt_path), "--skip-file-sha"])

    assert rc == 0
    assert receipt_path.exists()


def test_main_dequant_dir_source_gguf_cli_override_wired(tmp_path, pinned_gguf_sha):
    real_gguf = tmp_path / "real.gguf"
    real_gguf.write_bytes(_FIXTURE_GGUF_BYTES)
    d = build_dequant_dir(
        tmp_path,
        gguf_path=str(tmp_path / "manifest-claims-a-path-that-does-not-exist.gguf"),
        write_gguf_file=False,
    )
    receipt_path = tmp_path / "identity_receipt.json"

    rc = vbi.main(["--dequant-dir", str(d), "--receipt", str(receipt_path), "--source-gguf", str(real_gguf)])

    assert rc == 0


def test_main_pinned_dir_and_dequant_dir_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        vbi.main(["--pinned-dir", str(tmp_path), "--dequant-dir", str(tmp_path)])


def test_main_neither_pinned_nor_dequant_dir_exits():
    with pytest.raises(SystemExit):
        vbi.main([])


# --- Comparability tripwire (T4 #5, 2026-07-30) -- check_same_base_scheme / --compare-runs --


def test_check_same_base_scheme_match():
    result = vbi.check_same_base_scheme({"base_scheme": "fp16_hf_revision"}, {"base_scheme": "fp16_hf_revision"})
    assert result["match"] is True
    assert result["scheme_match"] is True
    assert result["assumed_a"] is False
    assert result["assumed_b"] is False


def test_check_same_base_scheme_mismatch():
    result = vbi.check_same_base_scheme({"base_scheme": "fp16_hf_revision"}, {"base_scheme": "dequant_q4km"})
    assert result["match"] is False
    assert result["scheme_match"] is False


def test_check_same_base_scheme_missing_key_assumed_fp16():
    result = vbi.check_same_base_scheme({}, {"base_scheme": "fp16_hf_revision"})
    assert result["assumed_a"] is True
    assert result["assumed_b"] is False
    assert result["scheme_a"] == config.BASE_SCHEME_FP16
    assert result["match"] is True


def test_check_same_base_scheme_identity_receipt_uses_scheme_key():
    # Review fix #4c: identity receipts carry "scheme", not "base_scheme" --
    # must not be mislabeled assumed-fp16.
    receipt = {"verdict": "PASS", "scheme": "dequant_q4km"}
    result = vbi.check_same_base_scheme(receipt, {"base_scheme": "dequant_q4km"})
    assert result["assumed_a"] is False
    assert result["scheme_a"] == "dequant_q4km"
    assert result["match"] is True


# --- _extract_base_scheme: multi-seed scan, smoke skipping, fail-closed (review fix #4a/#4b) --


def test_extract_base_scheme_non_dict_input_returns_none():
    # Review fix #10: malformed input is "not found", never a crash.
    assert vbi._extract_base_scheme(["not", "a", "dict"]) is None
    assert vbi._extract_base_scheme(None) is None
    assert vbi._extract_base_scheme("a string") is None


# --- Top-level base_scheme/scheme type guard (review fix #3, round 4) -------
# The round-3 non-string guard only covered seeds[] entries -- a top-level
# value bypassed it entirely.


def test_extract_base_scheme_top_level_explicit_null_raises():
    # Presence of the key signals real intent -- an explicit null is
    # invalid, not "legacy, assume fp16" (which is what a MISSING key means).
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme({"base_scheme": None})


def test_extract_base_scheme_top_level_int_raises():
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme({"base_scheme": 42})


def test_extract_base_scheme_top_level_list_raises():
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme({"base_scheme": ["not", "a", "string"]})


def test_extract_base_scheme_top_level_scheme_key_null_raises():
    # The "scheme" key (identity receipt shape) gets the same guard.
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme({"scheme": None})


def test_check_same_base_scheme_top_level_equal_non_string_values_no_longer_falsely_matches():
    # The exact bug: two sides agreeing on the SAME invalid non-string
    # value used to compare equal and report MATCH -- must now raise
    # instead of silently validating an impossible scheme.
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi.check_same_base_scheme({"base_scheme": 42}, {"base_scheme": 42})


def test_main_compare_runs_top_level_null_base_scheme_refuses_as_substantive(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"base_scheme": None}), encoding="utf-8")
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 2  # substantive (AmbiguousBaseSchemeError), not an infra error
    assert "REFUSED" in captured.err


def test_main_compare_runs_non_dict_input_list_refuses_as_infra(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 3  # infra, never treated as a legacy-fp16 run
    assert "REFUSED" in captured.err


def test_main_compare_runs_non_dict_input_scalar_refuses_as_infra(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps("just a string"), encoding="utf-8")
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 3
    assert "REFUSED" in captured.err


# --- Invalid UTF-8 input to --compare-runs (review fix #4, round 4) ---------


def test_main_compare_runs_invalid_utf8_refuses_cleanly(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_bytes(b"\xff\xfenot valid utf-8 json content")
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 3  # infra error, not an uncaught traceback (exit 1)
    assert "REFUSED" in captured.err


def test_extract_base_scheme_scans_all_seeds_agreeing():
    manifest = {"seeds": [{"seed": 1, "base_scheme": "dequant_q4km"}, {"seed": 2, "base_scheme": "dequant_q4km"}]}
    assert vbi._extract_base_scheme(manifest) == "dequant_q4km"


def test_extract_base_scheme_mixed_seed_entries_raises():
    manifest = {"seeds": [{"seed": 1, "base_scheme": "fp16_hf_revision"}, {"seed": 2, "base_scheme": "dequant_q4km"}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError, match="mixed-scheme"):
        vbi._extract_base_scheme(manifest)


def test_extract_base_scheme_partial_coverage_raises():
    # Some non-smoke entries carry base_scheme, others don't -- ambiguous,
    # never silently resolved either way.
    manifest = {"seeds": [{"seed": 1, "base_scheme": "fp16_hf_revision"}, {"seed": 2}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError, match="mixed-scheme"):
        vbi._extract_base_scheme(manifest)


def test_extract_base_scheme_all_missing_is_legacy_none():
    manifest = {"seeds": [{"seed": 1}, {"seed": 2}]}
    assert vbi._extract_base_scheme(manifest) is None


def test_extract_base_scheme_skips_smoke_entries():
    # Review fix #4b/#6: a "smoke": true entry has no real scheme to report
    # and must be excluded from the scan entirely, not counted toward
    # "partial coverage".
    manifest = {"seeds": [{"seed": 0, "smoke": True}, {"seed": 1, "base_scheme": "dequant_q4km"}]}
    assert vbi._extract_base_scheme(manifest) == "dequant_q4km"


def test_extract_base_scheme_smoke_entry_alone_is_legacy_none():
    manifest = {"seeds": [{"seed": 0, "smoke": True}]}
    assert vbi._extract_base_scheme(manifest) is None


def test_extract_base_scheme_smoke_entries_never_trigger_partial_coverage_error():
    manifest = {"seeds": [{"seed": 0, "smoke": True}, {"seed": 1, "base_scheme": "fp16_hf_revision"}, {"seed": 2, "base_scheme": "fp16_hf_revision"}]}
    assert vbi._extract_base_scheme(manifest) == "fp16_hf_revision"  # smoke entry doesn't count as "missing"


def test_check_same_base_scheme_propagates_ambiguous_error():
    manifest = {"seeds": [{"seed": 1, "base_scheme": "fp16_hf_revision"}, {"seed": 2, "base_scheme": "dequant_q4km"}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi.check_same_base_scheme(manifest, {"base_scheme": "fp16_hf_revision"})


# --- Non-string base_scheme values: clean refusal, never a raw TypeError (review fix #4, round 3) --


def test_extract_base_scheme_null_mixed_with_string_raises_cleanly():
    manifest = {"seeds": [{"seed": 1, "base_scheme": None}, {"seed": 2, "base_scheme": "fp16_hf_revision"}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme(manifest)


def test_extract_base_scheme_int_value_mixed_with_string_raises_cleanly():
    manifest = {"seeds": [{"seed": 1, "base_scheme": 42}, {"seed": 2, "base_scheme": "fp16_hf_revision"}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme(manifest)


def test_extract_base_scheme_list_value_mixed_with_string_raises_cleanly():
    # A list is unhashable -- the OLD code would have crashed building a
    # set() out of these values, before ever comparing anything.
    manifest = {"seeds": [{"seed": 1, "base_scheme": ["not", "a", "string"]}, {"seed": 2, "base_scheme": "fp16_hf_revision"}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme(manifest)


def test_extract_base_scheme_single_unhashable_value_does_not_crash_on_set_construction():
    # Even a LONE entry with an unhashable base_scheme (no cross-entry
    # comparison needed at all) must not crash: type-checking happens
    # before any set() is ever built.
    manifest = {"seeds": [{"seed": 1, "base_scheme": ["not", "a", "string"]}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme(manifest)


def test_extract_base_scheme_all_entries_same_non_string_type_still_raises():
    # Internally "consistent" (every entry agrees) is not an excuse -- a
    # non-string base_scheme is invalid on its face, never silently
    # trusted as if it were a real scheme label.
    manifest = {"seeds": [{"seed": 1, "base_scheme": 42}, {"seed": 2, "base_scheme": 42}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme(manifest)


def test_main_compare_runs_non_string_base_scheme_refuses_cleanly_no_traceback(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps({"seeds": [{"seed": 1, "base_scheme": None}, {"seed": 2, "base_scheme": "fp16_hf_revision"}]}),
        encoding="utf-8",
    )
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 2  # substantive refusal, not an infra error and NOT an uncaught exception
    assert "REFUSED" in captured.err


# --- Smoke-marker semantics: identity check + visibility (review fix #7, round 3) --


def test_extract_base_scheme_smoke_string_false_not_treated_as_marker():
    # "smoke": "false" is a STRING, not the boolean True -- must not be
    # silently treated as the smoke marker (identity check, not truthy).
    # The entry then counts as an ordinary real entry with no base_scheme
    # -- i.e. legacy-shaped, resolves to None (assumed fp16 by the caller).
    manifest = {"seeds": [{"seed": 0, "smoke": "false"}]}
    assert vbi._extract_base_scheme(manifest) is None


def test_extract_base_scheme_smoke_int_one_not_treated_as_marker():
    manifest = {"seeds": [{"seed": 0, "smoke": 1}, {"seed": 1, "base_scheme": "fp16_hf_revision"}]}
    # smoke=1 (int) is NOT the marker -- this entry is "real" but lacks
    # base_scheme, so it's a partial-coverage conflict against seed 1,
    # which DOES carry one.
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme(manifest)


def test_extract_base_scheme_smoke_with_explicit_scheme_counts_toward_distinct_set():
    # An anomalous smoke:True entry that ALSO carries an explicit
    # base_scheme must NOT be silently excluded -- fail toward visibility:
    # if it conflicts with a real entry's scheme, that's surfaced, not hidden.
    manifest = {
        "seeds": [
            {"seed": 0, "smoke": True, "base_scheme": "dequant_q4km"},
            {"seed": 1, "base_scheme": "fp16_hf_revision"},
        ]
    }
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_scheme(manifest)


def test_extract_base_scheme_smoke_with_explicit_scheme_alone_is_visible():
    manifest = {"seeds": [{"seed": 0, "smoke": True, "base_scheme": "dequant_q4km"}]}
    assert vbi._extract_base_scheme(manifest) == "dequant_q4km"


# --- base_source_sha256 comparison (review fix #4d) --------------------------


def test_check_same_base_scheme_source_sha_mismatch_same_scheme():
    result = vbi.check_same_base_scheme(
        {"base_scheme": "dequant_q4km", "base_source_sha256": "aaa"},
        {"base_scheme": "dequant_q4km", "base_source_sha256": "bbb"},
    )
    assert result["scheme_match"] is True
    assert result["source_sha_match"] is False
    assert result["match"] is False


def test_check_same_base_scheme_source_sha_matching_ok():
    result = vbi.check_same_base_scheme(
        {"base_scheme": "dequant_q4km", "base_source_sha256": "aaa"},
        {"base_scheme": "dequant_q4km", "base_source_sha256": "aaa"},
    )
    assert result["source_sha_match"] is True
    assert result["match"] is True


def test_check_same_base_scheme_source_sha_not_compared_when_one_side_missing():
    # "when BOTH sides carry it" -- one side missing is not itself a
    # conflict, there's simply nothing to compare there.
    result = vbi.check_same_base_scheme(
        {"base_scheme": "fp16_hf_revision", "base_source_sha256": "aaa"},
        {"base_scheme": "fp16_hf_revision"},
    )
    assert result["source_sha_match"] is True
    assert result["match"] is True


def test_extract_base_source_sha256_reads_non_smoke_seed_entry():
    manifest = {"seeds": [{"seed": 0, "smoke": True}, {"seed": 1, "base_source_sha256": "abc"}]}
    assert vbi._extract_base_source_sha256(manifest) == "abc"


def test_extract_base_source_sha256_scans_all_entries_agreeing():
    # Review fix #6, round 3: scans ALL non-smoke entries, not just the
    # first one carrying the key.
    manifest = {"seeds": [{"seed": 1, "base_source_sha256": "abc"}, {"seed": 2, "base_source_sha256": "abc"}]}
    assert vbi._extract_base_source_sha256(manifest) == "abc"


def test_extract_base_source_sha256_disagreement_raises():
    manifest = {"seeds": [{"seed": 1, "base_source_sha256": "abc"}, {"seed": 2, "base_source_sha256": "xyz"}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError, match="mixed-scheme"):
        vbi._extract_base_source_sha256(manifest)


def test_extract_base_source_sha256_partial_presence_not_ambiguous():
    # Unlike base_scheme's stricter partial-coverage rule, one entry simply
    # lacking the key is not itself a conflict for this secondary field.
    manifest = {"seeds": [{"seed": 1, "base_source_sha256": "abc"}, {"seed": 2}]}
    assert vbi._extract_base_source_sha256(manifest) == "abc"


def test_extract_base_source_sha256_first_entry_no_longer_silently_wins_over_a_conflicting_later_one():
    # The OLD "first non-smoke entry wins" behavior would have returned
    # "abc" here without ever noticing seed 2 disagrees -- now it raises.
    manifest = {"seeds": [{"seed": 1, "base_source_sha256": "abc"}, {"seed": 2, "base_source_sha256": "different"}]}
    with pytest.raises(vbi.AmbiguousBaseSchemeError):
        vbi._extract_base_source_sha256(manifest)


# --- --compare-runs end to end ------------------------------------------------


def test_main_compare_runs_match_exits_zero(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])

    assert rc == 0


def test_main_compare_runs_mismatch_refuses(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")
    b.write_text(json.dumps({"base_scheme": "dequant_q4km"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 2
    assert "MISMATCH" in captured.err
    assert "REFUSED" in captured.err


def test_main_compare_runs_mismatch_allowed_with_flag(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")
    b.write_text(json.dumps({"base_scheme": "dequant_q4km"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b), "--allow-cross-scheme"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "WARNING" in captured.out


def test_main_compare_runs_missing_scheme_prints_assumption_note(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"seeds": []}), encoding="utf-8")  # no base_scheme anywhere -- v1/v2 shape
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "NOTE" in captured.out
    assert "assum" in captured.out.lower()


def test_main_compare_runs_source_sha_mismatch_refuses(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"base_scheme": "dequant_q4km", "base_source_sha256": "aaa"}), encoding="utf-8")
    b.write_text(json.dumps({"base_scheme": "dequant_q4km", "base_source_sha256": "bbb"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 2
    assert "base_source_sha256 differs" in captured.err


def test_main_compare_runs_ambiguous_manifest_refuses_cleanly(tmp_path, capsys):
    # Review fix #10: an AmbiguousBaseSchemeError must become a clean
    # refusal, never a traceback.
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps({"seeds": [{"seed": 1, "base_scheme": "fp16_hf_revision"}, {"seed": 2, "base_scheme": "dequant_q4km"}]}),
        encoding="utf-8",
    )
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 2
    assert "REFUSED" in captured.err
    assert "mixed-scheme" in captured.err


def test_main_compare_runs_missing_file_refuses_cleanly(tmp_path, capsys):
    # Review fix #5, round 3: infra errors (missing file, bad JSON) now
    # exit 3, distinguishing them from substantive refusals (mismatch,
    # ambiguous manifest), which stay at exit 2.
    a = tmp_path / "does-not-exist.json"
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 3
    assert "REFUSED" in captured.err


def test_main_compare_runs_invalid_json_refuses_cleanly(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("not json {", encoding="utf-8")
    b.write_text(json.dumps({"base_scheme": "fp16_hf_revision"}), encoding="utf-8")

    rc = vbi.main(["--compare-runs", str(a), str(b)])
    captured = capsys.readouterr()

    assert rc == 3
    assert "REFUSED" in captured.err


def test_main_compare_runs_mutually_exclusive_with_pinned_dir(tmp_path):
    # Review fix #9.
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("{}", encoding="utf-8")
    b.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        vbi.main(["--compare-runs", str(a), str(b), "--pinned-dir", str(tmp_path)])


def test_main_compare_runs_mutually_exclusive_with_dequant_dir(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("{}", encoding="utf-8")
    b.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        vbi.main(["--compare-runs", str(a), str(b), "--dequant-dir", str(tmp_path)])

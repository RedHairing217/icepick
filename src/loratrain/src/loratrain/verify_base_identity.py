"""Offline GGUF <-> HF structural identity comparator (RUNBOOK D-R2, section 0.3).

Runs entirely against local files: the local quantized GGUF (parsed directly,
metadata only -- never the multi-GB tensor payload) and a small pinned-revision
metadata pair (``config.json`` + ``tokenizer.json``) the operator fetches ahead
of time. There is nothing to chain cryptographically from the GGUF header back
to a source repo, so identity is established structurally: every architecture
field, the rope/eps constants, the context length, and a hash of the full
ordered vocabulary must agree between the two sides. Any mismatch is a FAIL
and the runbook says: do not rent the box.

This module is offline by design and therefore never spells out a host,
port, or scheme -- see ``config.py`` for the single-source-of-truth rule this
package-wide scan enforces (``tests/test_config.py::
test_single_source_of_truth_for_server_address``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path

from loratrain import config
from loratrain.build_dataset import sha256_file

# ============================================================================
# Identity pins (verified 2026-07-25, RUNBOOK D-R2) -- single home for these
# values; other modules (upload_guard.py) import them from here rather than
# re-declaring their own copies.
# ============================================================================

FP16_REPO = "Qwen/Qwen3-8B"
FP16_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GGUF_REPO = "lmstudio-community/Qwen3-8B-GGUF"
GGUF_REPO_REVISION = "07ebe812301319d9947477e3a94ab8aa587bb3af"


# --- context-length dual pin (comparator fix, approved Nicky 2026-07-26) ---
# The GGUF encodes Qwen3's NATIVE window (32768) while the pinned HF config
# declares the extended deployment max (40960). Forensically proven 2026-07-26:
# no Qwen/Qwen3-8B revision ever carried 32768 in config.json (checked every
# commit), and the five safetensors shards' blob-oids are IDENTICAL from the
# initial upload through the pinned revision -- so this is conversion-time
# serving metadata, not a weight/revision divergence. Training (seq 4096) and
# eval (ctx 8192) run far below both bounds. Any OTHER value on either side is
# still a hard FAIL.
GGUF_PINNED_CONTEXT_LENGTH = 32768
HF_PINNED_MAX_POSITION_EMBEDDINGS = 40960

# llama.cpp pads the GGUF token list from the tokenizer's defined range up to
# config.vocab_size (embedding rows) with synthetic "[PAD<id>]" entries.
_PAD_TAIL_RE = re.compile(r"\[PAD\d+\]")
EXPECTED_BASE_GGUF_SHA256 = "a7676d257b10f3ce23aedba45e64ba61a5aa295f0009d87c5627f6c026a8f35f"
LLAMACPP_TAG = "b10107"
DEFAULT_BASE_GGUF_PATH = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"

# --- GGUF wire format (v2/v3 header + typed key-value section) --------------
# The on-disk layout is documented in llama.cpp's gguf spec; reproduced here
# only as the minimal reader this comparator needs (KV section only --
# tensor info / tensor data are never touched, so this stays fast even
# against the real multi-GB file).

_GGUF_MAGIC = b"GGUF"

_T_UINT8 = 0
_T_INT8 = 1
_T_UINT16 = 2
_T_INT16 = 3
_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11
_T_FLOAT64 = 12

_SCALAR_STRUCT_FMT = {
    _T_UINT8: "<B",
    _T_INT8: "<b",
    _T_UINT16: "<H",
    _T_INT16: "<h",
    _T_UINT32: "<I",
    _T_INT32: "<i",
    _T_FLOAT32: "<f",
    _T_UINT64: "<Q",
    _T_INT64: "<q",
    _T_FLOAT64: "<d",
}


def _read_exact(fh, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise ValueError(f"unexpected EOF: wanted {n} bytes, got {len(data)}")
    return data


def _read_u32(fh) -> int:
    return struct.unpack("<I", _read_exact(fh, 4))[0]


def _read_u64(fh) -> int:
    return struct.unpack("<Q", _read_exact(fh, 8))[0]


def _read_gguf_string(fh) -> str:
    length = _read_u64(fh)
    return _read_exact(fh, length).decode("utf-8")


def _read_scalar(fh, value_type: int):
    if value_type == _T_BOOL:
        return _read_exact(fh, 1) != b"\x00"
    if value_type == _T_STRING:
        return _read_gguf_string(fh)
    fmt = _SCALAR_STRUCT_FMT.get(value_type)
    if fmt is None:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    return struct.unpack(fmt, _read_exact(fh, struct.calcsize(fmt)))[0]


def _read_array(fh):
    """Read one GGUF array value; return ``(elem_type, values)``.

    Recurses for nested arrays (an array-of-arrays is not something the
    Qwen3 metadata this module targets actually uses, but the wire format
    permits it, so it is handled rather than silently mis-parsed).
    """
    elem_type = _read_u32(fh)
    count = _read_u64(fh)
    values = []
    for _ in range(count):
        if elem_type == _T_ARRAY:
            values.append(_read_array(fh))
        else:
            values.append(_read_scalar(fh, elem_type))
    return elem_type, values


def read_gguf_metadata(path, defined_n=None) -> dict:
    """Parse a GGUF v2/v3 header's key-value metadata section.

    Stops immediately after the last key-value pair -- tensor info and
    tensor data (the bulk of the file) are never read, so this is cheap
    even against the real ~5 GB quantized file.

    Every scalar key-value pair is stored under its own key. Array values
    are NOT stored in full: ``tokenizer.ggml.tokens`` is condensed to
    ``{"n": <count>, "sha256": <digest of the full ordered token list>}``
    under the key ``"tokenizer.ggml.tokens#digest"`` (the full list itself
    is never retained); every other array is condensed to its element
    count under ``"<key>#len"``.
    """
    path = Path(path)
    metadata: dict = {}
    with path.open("rb") as fh:
        magic = _read_exact(fh, 4)
        if magic != _GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (bad magic {magic!r})")
        version = _read_u32(fh)
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version} (expected 2 or 3)")
        tensor_count = _read_u64(fh)
        kv_count = _read_u64(fh)
        metadata["_tensor_count"] = tensor_count

        for _ in range(kv_count):
            key = _read_gguf_string(fh)
            value_type = _read_u32(fh)
            if value_type == _T_ARRAY:
                _elem_type, values = _read_array(fh)
                if key == "tokenizer.ggml.tokens":
                    digest = hashlib.sha256(
                        json.dumps(values, ensure_ascii=False).encode("utf-8")
                    ).hexdigest()
                    entry = {"n": len(values), "sha256": digest}
                    if defined_n is not None and 0 < defined_n <= len(values):
                        entry["defined_n"] = defined_n
                        entry["defined_sha256"] = hashlib.sha256(
                            json.dumps(values[:defined_n], ensure_ascii=False).encode("utf-8")
                        ).hexdigest()
                        entry["tail_all_pad"] = all(
                            _PAD_TAIL_RE.fullmatch(t) is not None for t in values[defined_n:]
                        )
                    metadata[f"{key}#digest"] = entry
                else:
                    metadata[f"{key}#len"] = len(values)
            else:
                metadata[key] = _read_scalar(fh, value_type)
    return metadata


def load_hf_reference(pinned_dir) -> dict:
    """Load + hash ``config.json`` and ``tokenizer.json`` from a pinned HF dir.

    Builds an ``id -> token`` mapping from ``tokenizer.json``'s
    ``model.vocab`` (a ``token -> id`` mapping, inverted here) with
    ``added_tokens`` applied on top (added tokens override/extend the base
    vocab mapping by id -- this is normal for special tokens that are
    re-listed in ``added_tokens``, not a corruption).

    If the resulting id space is not exactly contiguous ``0..N-1`` (a gap or
    a same-id collision inside ``model.vocab`` itself), that is reported
    back as a FAIL condition in the returned dict (``vocab_contiguous:
    False``, ``vocab_sha256: None``) rather than raised -- the caller
    (``compare``) turns it into an ordinary failed comparison row, not a
    traceback.
    """
    pinned_dir = Path(pinned_dir)
    config_bytes = (pinned_dir / "config.json").read_bytes()
    tokenizer_bytes = (pinned_dir / "tokenizer.json").read_bytes()

    config_data = json.loads(config_bytes.decode("utf-8"))
    tokenizer_data = json.loads(tokenizer_bytes.decode("utf-8"))

    vocab = tokenizer_data.get("model", {}).get("vocab", {}) or {}
    vocab_ids = list(vocab.values())
    vocab_has_id_collision = len(vocab_ids) != len(set(vocab_ids))

    id_to_token = {}
    for token, idx in vocab.items():
        id_to_token[idx] = token
    for added in tokenizer_data.get("added_tokens", []) or []:
        id_to_token[added["id"]] = added["content"]

    n = len(id_to_token)
    contiguous = (not vocab_has_id_collision) and set(id_to_token.keys()) == set(range(n))

    if contiguous:
        tokens_in_id_order = [id_to_token[i] for i in range(n)]
        vocab_sha256 = hashlib.sha256(
            json.dumps(tokens_in_id_order, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    else:
        vocab_sha256 = None

    return {
        "vocab_n": n,
        "vocab_sha256": vocab_sha256,
        "vocab_contiguous": contiguous,
        "config": config_data,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
    }


def _isclose(a, b, *, rel_tol=0.0, abs_tol=0.0) -> bool:
    if a is None or b is None:
        return False
    try:
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    except TypeError:
        return False


def compare(gguf_meta: dict, hf_ref: dict) -> list:
    """Compare parsed GGUF metadata against the pinned HF reference.

    Returns a list of ``(field, gguf_value, hf_value, ok)`` tuples, one per
    comparison performed. The GGUF side's architecture-scoped keys (block
    count, embedding length, etc.) are looked up under the ``f"{arch}."``
    prefix, where ``arch`` is the GGUF's own ``general.architecture`` value
    -- so this does not hardcode ``"qwen3."`` for the lookup, only for the
    expected-value check on the architecture row itself. The EOS row is
    included only when both sides carry an int id for it (an absent EOS
    field on either side is not itself a mismatch).
    """
    rows = []
    hf_config = hf_ref.get("config", {}) or {}
    arch = gguf_meta.get("general.architecture")
    prefix = f"{arch}." if arch else ""

    def add(field, gguf_value, hf_value, ok):
        rows.append((field, gguf_value, hf_value, bool(ok)))

    hf_architectures = hf_config.get("architectures")
    add(
        "architecture",
        arch,
        hf_architectures,
        arch == "qwen3" and hf_architectures == ["Qwen3ForCausalLM"],
    )

    def simple(field, gguf_key, hf_key, hf_fallback=None):
        gguf_value = gguf_meta.get(gguf_key)
        hf_value = hf_config.get(hf_key)
        if hf_value is None and hf_fallback is not None:
            hf_value = hf_fallback(hf_config)
        add(field, gguf_value, hf_value, gguf_value is not None and gguf_value == hf_value)

    simple("block_count", f"{prefix}block_count", "num_hidden_layers")
    simple("embedding_length", f"{prefix}embedding_length", "hidden_size")
    simple("feed_forward_length", f"{prefix}feed_forward_length", "intermediate_size")
    simple("attention.head_count", f"{prefix}attention.head_count", "num_attention_heads")
    simple("attention.head_count_kv", f"{prefix}attention.head_count_kv", "num_key_value_heads")

    def head_dim_fallback(cfg):
        try:
            heads = cfg["num_attention_heads"]
            return cfg["hidden_size"] // heads if heads else None
        except (KeyError, TypeError):
            return None

    simple(
        "attention.key_length",
        f"{prefix}attention.key_length",
        "head_dim",
        hf_fallback=head_dim_fallback,
    )

    gguf_rope = gguf_meta.get(f"{prefix}rope.freq_base")
    hf_rope = hf_config.get("rope_theta")
    add("rope.freq_base", gguf_rope, hf_rope, _isclose(gguf_rope, hf_rope, rel_tol=1e-6))

    gguf_eps = gguf_meta.get(f"{prefix}attention.layer_norm_rms_epsilon")
    hf_eps = hf_config.get("rms_norm_eps")
    add(
        "attention.layer_norm_rms_epsilon",
        gguf_eps,
        hf_eps,
        _isclose(gguf_eps, hf_eps, abs_tol=1e-9),
    )

    gguf_ctx = gguf_meta.get(f"{prefix}context_length")
    hf_ctx = hf_config.get("max_position_embeddings")
    add(
        "context_length[dual-pin]",
        gguf_ctx,
        hf_ctx,
        gguf_ctx == GGUF_PINNED_CONTEXT_LENGTH
        and hf_ctx == HF_PINNED_MAX_POSITION_EMBEDDINGS,
    )

    digest = gguf_meta.get("tokenizer.ggml.tokens#digest") or {}
    gguf_vocab_n = digest.get("n")
    hf_vocab_n = hf_ref.get("vocab_n")
    hf_vocab_sha = hf_ref.get("vocab_sha256")
    # Embedding-row count: the GGUF token list is padded to config.vocab_size,
    # so THAT is its equal; the tokenizer's defined count must not exceed it.
    hf_vocab_rows = hf_config.get("vocab_size", hf_vocab_n)
    add(
        "vocab_n[rows==config.vocab_size]",
        gguf_vocab_n,
        hf_vocab_rows,
        gguf_vocab_n is not None
        and gguf_vocab_n == hf_vocab_rows
        and (hf_vocab_n is None or hf_vocab_n <= gguf_vocab_n),
    )
    # Token identity: compare over the tokenizer's DEFINED range when the
    # reader computed it (defined_sha256); legacy full-list digest otherwise.
    gguf_vocab_sha = digest.get("defined_sha256", digest.get("sha256"))
    sha_field = "vocab_sha256[defined]" if "defined_sha256" in digest else "vocab_sha256"
    add(
        sha_field,
        gguf_vocab_sha,
        hf_vocab_sha,
        gguf_vocab_sha is not None and hf_vocab_sha is not None and gguf_vocab_sha == hf_vocab_sha,
    )
    if "tail_all_pad" in digest:
        add(
            "vocab_tail[PADn-only]",
            digest.get("tail_all_pad"),
            True,
            digest.get("tail_all_pad") is True,
        )

    gguf_eos = gguf_meta.get("tokenizer.ggml.eos_token_id")
    hf_eos = hf_config.get("eos_token_id")
    if isinstance(gguf_eos, int) and isinstance(hf_eos, int):
        add("eos_token_id", gguf_eos, hf_eos, gguf_eos == hf_eos)

    return rows


def _format_table(rows, file_sha_row) -> str:
    widths = (34, 24, 24)
    header = f"{'field':<{widths[0]}} {'gguf':<{widths[1]}} {'hf':<{widths[2]}} status"
    lines = [header, "-" * len(header)]
    for field, gguf_value, hf_value, ok in rows:
        status = "PASS" if ok else "FAIL"
        lines.append(
            f"{field:<{widths[0]}} {str(gguf_value):<{widths[1]}} {str(hf_value):<{widths[2]}} {status}"
        )
    lines.append(file_sha_row)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="verify_base_identity")
    parser.add_argument("--pinned-dir", required=True, help="directory holding the pinned config.json + tokenizer.json")
    parser.add_argument("--gguf", default=str(DEFAULT_BASE_GGUF_PATH), help="path to the local quantized GGUF file")
    parser.add_argument(
        "--skip-file-sha",
        action="store_true",
        help="skip hashing the (multi-GB) GGUF file; default is to hash it and compare against the pinned sha256",
    )
    parser.add_argument("--receipt", default=None, help="where to write the PASS receipt (default: config.DATA_DIR/identity_receipt.json)")
    args = parser.parse_args(argv)

    gguf_path = Path(args.gguf)
    pinned_dir = Path(args.pinned_dir)
    receipt_path = Path(args.receipt) if args.receipt else (config.DATA_DIR / "identity_receipt.json")

    hf_ref = load_hf_reference(pinned_dir)
    defined_n = hf_ref.get("vocab_n") if hf_ref.get("vocab_contiguous") else None
    gguf_meta = read_gguf_metadata(gguf_path, defined_n=defined_n)
    rows = compare(gguf_meta, hf_ref)

    if args.skip_file_sha:
        actual_file_sha = "skipped"
        file_sha_ok = True
        file_sha_status = "SKIPPED"
    else:
        actual_file_sha = sha256_file(gguf_path)
        file_sha_ok = actual_file_sha == EXPECTED_BASE_GGUF_SHA256
        file_sha_status = "PASS" if file_sha_ok else "FAIL"

    file_sha_row = (
        f"{'base_gguf_sha256':<34} {actual_file_sha:<24} {EXPECTED_BASE_GGUF_SHA256:<24} {file_sha_status}"
    )
    print(_format_table(rows, file_sha_row))

    all_ok = file_sha_ok and all(ok for _, _, _, ok in rows)
    if not all_ok:
        return 2

    receipt = {
        "verdict": "PASS",
        "checked_at": time.time(),
        "fp16_repo": FP16_REPO,
        "fp16_revision": FP16_REVISION,
        "gguf_repo": GGUF_REPO,
        "gguf_repo_revision": GGUF_REPO_REVISION,
        "base_gguf_sha256": actual_file_sha,
        "config_sha256": hf_ref["config_sha256"],
        "tokenizer_sha256": hf_ref["tokenizer_sha256"],
        "fields": [{"field": f, "gguf": g, "hf": h, "ok": ok} for f, g, h, ok in rows],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

--- Dequant hash-chain identity mode (T3, 2026-07-30) ---------------------
``--dequant-dir`` is a second, unrelated identity mode for the
``dequant_q4km`` base scheme (config.BASE_SCHEME_DEQUANT): the base weights
come from ``gguf_to_hf.py`` dequantizing the SAME pinned GGUF this module
already hashes, so unlike the fp16 side there IS an unbroken hash chain back
to a known-good source -- ``check_dequant_manifest`` verifies every link of
it (manifest schema/scheme, the pinned GGUF sha, every emitted tensor file's
sha256, the tensor count) instead of a structural field-by-field comparison.

--- Comparability tripwire (T4 #5, 2026-07-30) -----------------------------
Runs trained under the two base schemes must NEVER be silently compared --
their base weights are recovered through different paths (direct fp16 read
vs Q4_K_M dequantization) even though they name the same model.
``check_same_base_scheme`` / ``--compare-runs`` is the post-hoc tripwire;
``upload_guard.check_base_scheme`` is the preflight half (refuses an upload
whose identity receipt was verified against a different scheme than
``config.BASE_SCHEME`` is currently set to train against).

--- --compare-runs exit-code taxonomy (review fix #5, round 3) -------------
  0 = the runs MATCH (same base_scheme, and same base_source_sha256 where
      both sides record one) -- the comparison is valid.
  2 = a SUBSTANTIVE refusal: a real scheme/source mismatch, or a manifest
      too internally ambiguous to resolve to any single scheme
      (``AmbiguousBaseSchemeError``) -- an operator decision is needed
      (``--allow-cross-scheme``, or fix the data).
  3 = an INFRA error: an input file could not be read, or was not valid
      JSON -- nothing about base schemes was even evaluated.
Known collision, not fought: argparse's own arg-parsing failures (e.g. the
mutual-exclusion/"one of ... required" ``parser.error()`` calls in
``main()`` below, or a missing required flag) also exit 2 via Python's
``SystemExit`` -- that is argparse's own convention, applies to this whole
CLI (not just ``--compare-runs``), and is deliberately left alone rather
than remapped.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import re
import struct
import sys
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


# ============================================================================
# Dequant hash-chain identity mode (T3, 2026-07-30) -- ``--dequant-dir``.
# ============================================================================

DEQUANT_MANIFEST_FILENAME = "dequant_manifest.json"
DEQUANT_INDEX_FILENAME = "model.safetensors.index.json"
_ORPHAN_SCAN_PATTERNS = ("*.safetensors", "*.bin", "*index.json")


def _as_dict(value) -> dict:
    """Coerce a manifest sub-section to a dict; anything else (None, a list,
    a string from a malformed manifest) becomes ``{}`` so ``.get()`` chains
    never raise (review fix #10 -- never a traceback on a malformed manifest)."""
    return value if isinstance(value, dict) else {}


def _manifest_path_is_safe(filename) -> bool:
    """True if ``filename`` is safe to join onto ``dequant_dir`` without
    escaping it.

    ``dequant_dir / filename`` silently discards ``dequant_dir`` entirely
    when ``filename`` is absolute (pathlib's documented "last absolute
    operand wins" join behavior), and ``..`` components climb back out of
    ``dequant_dir`` even for a relative path -- either way a re-hash gate
    that joins blindly can be pointed at an arbitrary file on this machine
    (review fix #1). Rejects both; also rejects a non-string entry (a
    malformed manifest putting e.g. an int as a key is not somehow safe).
    """
    if not isinstance(filename, str) or not filename:
        return False
    p = Path(filename)
    return not p.is_absolute() and ".." not in p.parts


def _walk_dequant_tree(dequant_dir: Path) -> dict:
    """One fail-closed walk of ``dequant_dir``'s ENTIRE tree, shared by the
    symlink-detection gate, the new unreadable-directory gate, and the
    orphan-completeness gate (review fix #2, round 4 -- previously each
    scan ran its own traversal, and BOTH ``os.walk``'s default error
    handling and ``Path.rglob`` silently treat a directory they cannot
    LIST (e.g. mode ``0o111`` -- executable/traversable but not readable)
    as if it contained NOTHING. A hidden symlink or an undeclared stale
    shard sitting behind such a directory used to pass every gate simply
    because the scan never raised, it just silently produced zero
    entries for that subtree.

    This walk supplies its own ``onerror`` handler: any directory
    ``os.walk`` cannot list is recorded, not swallowed, and surfaces as
    its own FAIL row -- fail CLOSED (refuse when uncertain), never silently
    "found nothing, so it must be empty."

    Walks with ``followlinks=False`` (as ``_find_symlinks`` always did) so
    a symlinked subdirectory is reported as itself and never descended
    into.

    Returns ``{"symlinks": [rel-path, ...], "candidate_files": [rel-path,
    ...] (every entry anywhere in the tree matching
    ``_ORPHAN_SCAN_PATTERNS``, replacing the old separate ``rglob`` pass),
    "unreadable": [rel-path, ...] (directories ``os.walk`` could not
    list)}`` -- all relative to ``dequant_dir`` and sorted.
    """
    dequant_dir = Path(dequant_dir)
    symlinks = []
    candidate_files = []
    unreadable = []

    def _onerror(exc: OSError) -> None:
        raw_path = getattr(exc, "filename", None)
        path = Path(raw_path) if raw_path else dequant_dir
        try:
            rel = str(path.relative_to(dequant_dir))
        except ValueError:
            rel = str(path)
        unreadable.append(rel)

    for root, dirnames, filenames in os.walk(dequant_dir, followlinks=False, onerror=_onerror):
        root_path = Path(root)
        for name in dirnames:
            candidate = root_path / name
            if candidate.is_symlink():
                symlinks.append(str(candidate.relative_to(dequant_dir)))
        for name in filenames:
            candidate = root_path / name
            if candidate.is_symlink():
                symlinks.append(str(candidate.relative_to(dequant_dir)))
            if any(fnmatch.fnmatch(name, pattern) for pattern in _ORPHAN_SCAN_PATTERNS):
                candidate_files.append(str(candidate.relative_to(dequant_dir)))

    return {
        "symlinks": sorted(set(symlinks)),
        "candidate_files": sorted(set(candidate_files)),
        "unreadable": sorted(set(unreadable)),
    }


def _find_symlinks(dequant_dir: Path) -> list:
    """Return every symlink (file OR directory) found anywhere under
    ``dequant_dir``, relative-path-formatted and sorted.

    Thin wrapper over ``_walk_dequant_tree`` (kept as its own function
    since it is independently useful/tested). ``gguf_to_hf.py`` never
    writes a symlink, so the rule is zero tolerance: ANY symlink anywhere
    in the tree is grounds for a hard FAIL (review fix #1, round 3) --
    lexical path-safety (``_manifest_path_is_safe``) alone cannot catch
    this, since a symlinked ``model.safetensors`` has a perfectly
    ordinary-looking relative name right up until the OS resolves it.

    NOTE (hardlink residual, adjudicated acceptable, round-4 review): this
    function -- and the rest of this module -- makes NO attempt to detect
    a hardlink standing in for a real file. That is a deliberately
    accepted gap, not an oversight: the source-GGUF and output.files/
    sidecars.files re-hash gates hash the file's REAL BYTES regardless of
    how many link names point at that inode, so a hardlinked-but-correct
    file still verifies correctly, and an undeclared hardlinked shard is
    still caught by the orphan-completeness gate (it has its own directory
    entry/name, which the walk sees) -- only a symlink can make one
    path's bytes silently BE a different, unrelated file's bytes.
    """
    return _walk_dequant_tree(dequant_dir)["symlinks"]


def _resolves_within(path: Path, root: Path) -> bool:
    """True if ``path.resolve()`` is ``root`` itself or a descendant of it.

    Belt-and-braces (review fix #1, round 3) alongside ``_find_symlinks``
    and the lexical ``_manifest_path_is_safe`` check: even if a symlink
    somehow slipped past detection, this catches the escape at the actual
    point of reading a file, by comparing REAL (symlink-resolved) paths
    rather than the spelled ones. Returns ``False`` (never raises) if
    ``resolve()`` itself fails (e.g. a path cycle).
    """
    try:
        resolved = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        return False
    return resolved == resolved_root or resolved_root in resolved.parents


def _rehash_file_map(dequant_dir: Path, files: dict) -> str:
    """Re-hash every ``{filename: sha256}`` entry of a manifest file map.

    Shared by the ``output.files`` and ``sidecars.files`` gates (review fix
    #2a: sidecars get the SAME path-safety + re-hash treatment as output
    files, via this one implementation). Returns ``""`` if every entry is
    safe, present, resolves inside ``dequant_dir``, and matches; otherwise
    a detail string describing every problem found. Never raises: an
    unsafe path is refused as a path problem, never joined onto disk at
    all (the whole point of the safety check is to never even attempt to
    open the escaped target).
    """
    unsafe = [name for name in files if not _manifest_path_is_safe(name)]
    if unsafe:
        return f"unsafe path(s) refused (absolute or containing '..'): {sorted(unsafe)}"
    bad = []
    for filename, expected_sha in files.items():
        file_path = dequant_dir / filename
        if not file_path.is_file():
            bad.append(f"{filename}: missing")
            continue
        if not _resolves_within(file_path, dequant_dir):
            # Belt-and-braces: the top-level "no symlinks anywhere" gate
            # is the primary defense, but a file re-hash never trusts a
            # spelled path alone -- it also confirms the RESOLVED path
            # never left dequant_dir before it is opened and hashed.
            bad.append(f"{filename}: resolves outside dequant_dir (symlink escape)")
            continue
        actual_sha = sha256_file(file_path)
        if actual_sha != expected_sha:
            bad.append(f"{filename}: sha256 {actual_sha} != recorded {expected_sha}")
    return "; ".join(bad)


def _compute_content_digest(tensors) -> str:
    """Recompute ``determinism.content_digest``: sha256 over the sorted
    ``"gguf_name:sha256"`` lines of ``tensors[]`` (review fix #3 -- the
    manifest's own claimed digest is never trusted uncopied; it is always
    recomputed from the per-tensor records and compared)."""
    lines = sorted(
        f"{_as_dict(t).get('gguf_name')}:{_as_dict(t).get('sha256')}"
        for t in (tensors if isinstance(tensors, list) else [])
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def check_dequant_manifest(dequant_dir, *, gguf_path=None, skip_file_sha: bool = False) -> dict:
    """Hash-chain identity check for the ``dequant_q4km`` base scheme.

    Unlike ``compare`` (the fp16 structural comparator, which infers identity
    from architecture/vocab agreement because there is nothing to chain
    cryptographically back to a source repo), the dequant base has an
    unbroken hash chain: ``gguf_to_hf.py``'s own manifest records the exact
    pinned-GGUF sha256 it dequantized FROM and a sha256 for every float32
    tensor file it wrote, so identity here means "does every recorded hash
    in that chain still hold" -- not a field-by-field structural match.

    ``gguf_path`` overrides which local GGUF file gets hashed for the
    source-link gate below; default is ``manifest.source_gguf.path``.
    ``skip_file_sha`` skips that hash (the multi-GB-file escape hatch,
    mirroring the fp16 mode's ``--skip-file-sha``) -- everything else still
    runs.

    Gates checked (all must pass for ``ok`` to be ``True``; every failure
    mode is a FAIL row, never an exception -- review fix #10):
      - ``dequant_manifest.json`` exists directly inside ``dequant_dir``,
        is valid JSON, and decodes to a JSON object (not a list/scalar).
      - NO symlink (file or directory) exists anywhere under
        ``dequant_dir`` -- ``gguf_to_hf.py`` never writes one, so this is
        zero-tolerance (review fix #1, round 3); every path actually
        opened below ALSO independently confirms its resolved location
        stays inside ``dequant_dir`` (belt-and-braces, same fix). NO
        directory anywhere under ``dequant_dir`` is unreadable/unlistable
        (review fix #2, round 4) -- an unlistable directory is refused,
        never silently treated as if it contained nothing.
      - ``schema_version == 1``; ``base_scheme == "dequant_q4km"``.
      - ``source_gguf.sha256`` equals the pinned local-GGUF sha
        (``EXPECTED_BASE_GGUF_SHA256``, this module's existing D-R2 pin --
        never re-typed as a separate literal).
      - the ACTUAL local GGUF file at ``gguf_path`` (or the manifest's
        ``source_gguf.path``) re-hashes to that same value -- closes the
        "source link never touches disk" gap: gate above alone only checks
        the manifest's own self-report, never the real file (review fix #3).
      - ``expected_gguf_sha256``, when non-null, equals the pin.
      - ``tensor_census.total`` equals ``config.EXPECTED_DEQUANT_TENSOR_TOTAL``;
        ``len(tensors) == tensor_census.total``; the recomputed
        ``content_digest`` over ``tensors[]`` matches the manifest's
        declared ``determinism.content_digest`` (never just copied --
        review fix #3).
      - ``permutation.applied is False``.
      - every file named in ``output.files`` is a SAFE relative path (not
        absolute, no ``..`` -- review fix #1), exists under ``dequant_dir``,
        RESOLVES inside ``dequant_dir`` (review fix #1, round 3), and
        re-hashes to its recorded sha256 (empty/missing ``output.files``
        is itself a FAIL); at least one entry ends in ``.safetensors``.
      - every file named in ``sidecars.files`` gets the identical
        safe-path + resolve + re-hash treatment (sidecars may legitimately
        be empty).
      - if ``model.safetensors.index.json`` exists on disk, it is named in
        ``output.files`` or ``sidecars.files`` (review fix #2b).
      - no ``*.safetensors``/``*.bin``/``*index.json`` file exists under
        ``dequant_dir`` that isn't covered by ``output.files`` or
        ``sidecars.files`` -- an orphan/stale shard from a previous run
        must not silently pass (review fix #2b).
      - ``config.json`` is present directly inside ``dequant_dir``.

    Returns ``{"ok": bool, "checks": [(name, ok, detail), ...], "manifest":
    <parsed dict or None>, "manifest_sha256": <sha256 of the manifest FILE's
    raw bytes, or None if it could not be read>, "content_digest": <the
    recomputed determinism digest, or None if it could not be computed>}``.

    VERIFICATION BOUNDARY (docstring-only review fix #9, round 3, stated
    explicitly so it is never assumed away): every gate above proves CHAIN
    INTEGRITY -- that the manifest, the files actually on disk, and the
    pinned GGUF sha agree with each other. It does NOT prove the emitted
    weights are a CORRECT dequantization of that GGUF (a manifest can be
    fabricated to be internally consistent with fabricated tensor files
    and still pass every gate here by design -- there is nothing in a
    hash-chain check that can rule that out). That property belongs to two
    OTHER layers this function does not replace: ``gguf_to_hf.py``'s own
    determinism contract (same GGUF in -> byte-identical output, always),
    and a separate ``verify_dequant_parity`` gate (numerically comparing
    the dequantized weights' behavior against the source). This function
    is the identity/chain layer; correctness of the dequantization itself
    is those tools' job, not this one's. A HARDLINK standing in for a
    declared file is a deliberately accepted residual, not a gap: the
    re-hash gates above hash the file's real bytes regardless of link
    count, and an undeclared hardlinked shard still has its own directory
    entry for the orphan-completeness gate to catch (round-4 review,
    adjudicated acceptable -- see ``_find_symlinks``'s docstring).
    """
    dequant_dir = Path(dequant_dir)
    checks = []

    manifest_path = dequant_dir / DEQUANT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        checks.append((f"{DEQUANT_MANIFEST_FILENAME} exists", False, f"{manifest_path} not found"))
        return {"ok": False, "checks": checks, "manifest": None, "manifest_sha256": None, "content_digest": None}
    checks.append((f"{DEQUANT_MANIFEST_FILENAME} exists", True, str(manifest_path)))

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        checks.append((f"{DEQUANT_MANIFEST_FILENAME} is valid JSON", False, str(exc)))
        return {"ok": False, "checks": checks, "manifest": None, "manifest_sha256": manifest_sha256, "content_digest": None}
    checks.append((f"{DEQUANT_MANIFEST_FILENAME} is valid JSON", True, ""))

    if not isinstance(manifest, dict):
        checks.append(("dequant_manifest.json decodes to a JSON object", False, f"got {type(manifest).__name__}"))
        return {"ok": False, "checks": checks, "manifest": None, "manifest_sha256": manifest_sha256, "content_digest": None}
    checks.append(("dequant_manifest.json decodes to a JSON object", True, ""))

    # Structural directory sanity checks, independent of manifest content
    # (review fix #1, round 3; extended by review fix #2, round 4). ONE
    # walk of the whole tree (``_walk_dequant_tree``) backs both gates
    # below AND the orphan-completeness gate further down, so all three
    # share the identical fail-closed error handling.
    walk = _walk_dequant_tree(dequant_dir)

    # A dequant output dir must contain ZERO symlinks anywhere --
    # gguf_to_hf.py never writes one, so any symlink at all (a swapped
    # declared file, or an entire stale/rogue shard tree hidden behind a
    # symlinked subdirectory) is a hard FAIL on its own, regardless of
    # what every other gate below reports.
    symlinks = walk["symlinks"]
    checks.append((
        "no symlinks anywhere under dequant_dir",
        not symlinks,
        "none found" if not symlinks else f"symlink(s) found (gguf_to_hf never writes any): {symlinks}",
    ))

    # Review fix #2, round 4: a subdirectory os.walk cannot LIST (e.g.
    # mode 0o111 -- traversable but not readable) must never be silently
    # treated as empty -- fail closed instead, since a hidden symlink or
    # an undeclared stale shard could be sitting behind it invisibly.
    unreadable = walk["unreadable"]
    checks.append((
        "no unreadable directory under dequant_dir",
        not unreadable,
        (
            "none found" if not unreadable
            else f"could not list director{'y' if len(unreadable) == 1 else 'ies'} "
                 f"(refused, never treated as empty): {unreadable}"
        ),
    ))

    schema_version = manifest.get("schema_version")
    checks.append(("schema_version == 1", schema_version == 1, f"got {schema_version!r}"))

    base_scheme = manifest.get("base_scheme")
    checks.append((
        "base_scheme == 'dequant_q4km'",
        base_scheme == config.BASE_SCHEME_DEQUANT,
        f"got {base_scheme!r}",
    ))

    source_gguf = _as_dict(manifest.get("source_gguf"))
    source_sha = source_gguf.get("sha256")
    checks.append((
        "source_gguf.sha256 == pinned EXPECTED_BASE_GGUF_SHA256",
        source_sha == EXPECTED_BASE_GGUF_SHA256,
        f"got {source_sha!r}, expected {EXPECTED_BASE_GGUF_SHA256!r}",
    ))

    # Source-link file hash (review fix #3): the gate above never touches
    # disk -- it only checks the manifest's OWN claim against the pin. This
    # one hashes the actual local GGUF and requires a three-way match
    # (actual file == manifest's claim == the pin), so a manifest that
    # correctly quotes the pin next to a swapped/corrupted local file still
    # fails.
    effective_gguf_path = gguf_path if gguf_path is not None else source_gguf.get("path")
    if skip_file_sha:
        checks.append(("source GGUF file re-hash (== manifest claim == pin)", True, "skipped (--skip-file-sha)"))
    elif not effective_gguf_path:
        checks.append((
            "source GGUF file re-hash (== manifest claim == pin)",
            False,
            "no GGUF path available (manifest.source_gguf.path is missing/empty and no override given)",
        ))
    else:
        gguf_file = Path(effective_gguf_path)
        if not gguf_file.is_file():
            checks.append(("source GGUF file re-hash (== manifest claim == pin)", False, f"{gguf_file} not found"))
        else:
            actual_gguf_sha = sha256_file(gguf_file)
            checks.append((
                "source GGUF file re-hash (== manifest claim == pin)",
                actual_gguf_sha == source_sha == EXPECTED_BASE_GGUF_SHA256,
                f"actual={actual_gguf_sha}, manifest claim={source_sha!r}, pin={EXPECTED_BASE_GGUF_SHA256!r}",
            ))

    expected_gguf_sha256 = manifest.get("expected_gguf_sha256")
    if expected_gguf_sha256 is None:
        checks.append(("expected_gguf_sha256 == pinned EXPECTED_BASE_GGUF_SHA256", True, "null in manifest -- gate not applicable"))
    else:
        checks.append((
            "expected_gguf_sha256 == pinned EXPECTED_BASE_GGUF_SHA256",
            expected_gguf_sha256 == EXPECTED_BASE_GGUF_SHA256,
            f"got {expected_gguf_sha256!r}",
        ))

    tensor_census = _as_dict(manifest.get("tensor_census"))
    total = tensor_census.get("total")
    checks.append((
        "tensor_census.total == EXPECTED_DEQUANT_TENSOR_TOTAL",
        total == config.EXPECTED_DEQUANT_TENSOR_TOTAL,
        f"got {total!r}, expected {config.EXPECTED_DEQUANT_TENSOR_TOTAL!r}",
    ))

    tensors = manifest.get("tensors")
    tensors = tensors if isinstance(tensors, list) else []
    checks.append((
        "len(tensors) == tensor_census.total",
        len(tensors) == total,
        f"len(tensors)={len(tensors)}, tensor_census.total={total!r}",
    ))

    content_digest = _compute_content_digest(tensors)
    declared_digest = _as_dict(manifest.get("determinism")).get("content_digest")
    checks.append((
        "determinism.content_digest matches recomputed digest over tensors[]",
        content_digest == declared_digest,
        f"recomputed={content_digest}, declared={declared_digest!r}",
    ))

    permutation_applied = _as_dict(manifest.get("permutation")).get("applied")
    checks.append((
        "permutation.applied is False",
        permutation_applied is False,
        f"got {permutation_applied!r}",
    ))

    output = _as_dict(manifest.get("output"))
    files = output.get("files")
    files = files if isinstance(files, dict) else {}
    if not files:
        checks.append(("output.files re-hash", False, "output.files is empty or missing"))
    else:
        detail = _rehash_file_map(dequant_dir, files)
        checks.append(("output.files re-hash", not detail, detail or f"all match ({len(files)} file(s))"))

    has_safetensors = bool(files) and any(_manifest_path_is_safe(n) and n.endswith(".safetensors") for n in files)
    checks.append((
        "output.files includes >=1 *.safetensors entry",
        has_safetensors,
        "" if has_safetensors else f"no safe *.safetensors entry among: {sorted(files)}",
    ))

    sidecars = _as_dict(manifest.get("sidecars"))
    sidecar_files = sidecars.get("files")
    sidecar_files = sidecar_files if isinstance(sidecar_files, dict) else {}
    if sidecar_files:
        detail = _rehash_file_map(dequant_dir, sidecar_files)
        checks.append(("sidecars.files re-hash", not detail, detail or f"all match ({len(sidecar_files)} file(s))"))
    else:
        checks.append(("sidecars.files re-hash", True, "no sidecars declared"))

    covered = set(n for n in files if _manifest_path_is_safe(n)) | set(n for n in sidecar_files if _manifest_path_is_safe(n))

    index_path = dequant_dir / DEQUANT_INDEX_FILENAME
    if index_path.is_file():
        covered_index = DEQUANT_INDEX_FILENAME in covered
        checks.append((
            f"{DEQUANT_INDEX_FILENAME} covered by output.files/sidecars.files",
            covered_index,
            "" if covered_index else f"{DEQUANT_INDEX_FILENAME} exists on disk but is not recorded",
        ))
    else:
        checks.append((f"{DEQUANT_INDEX_FILENAME} covered by output.files/sidecars.files", True, "not present on disk"))

    # Review fix #2, round 4: reuses the SAME walk (``walk["candidate_files"]``)
    # computed above instead of a separate ``Path.rglob`` pass -- rglob has
    # no onerror hook at all, so an unlistable subdirectory used to be
    # silently invisible to it (the "no unreadable directory" gate above
    # already independently fails on that case; this just stops the
    # orphan scan from ALSO being blind to it).
    orphans = sorted(rel for rel in walk["candidate_files"] if rel not in covered)
    checks.append((
        "no orphan *.safetensors/*.bin/*index.json files on disk",
        not orphans,
        "all covered" if not orphans else f"uncovered file(s): {orphans}",
    ))

    config_json_path = dequant_dir / "config.json"
    checks.append(("config.json present", config_json_path.is_file(), str(config_json_path)))

    ok = all(c[1] for c in checks)
    return {
        "ok": ok,
        "checks": checks,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "content_digest": content_digest,
    }


def _format_dequant_table(checks) -> str:
    width = 52
    header = f"{'check':<{width}} status  detail"
    lines = [header, "-" * (width + 40)]
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        lines.append(f"{name:<{width}} {status:<7} {detail}")
    return "\n".join(lines)


# ============================================================================
# Comparability tripwire (T4 #5, 2026-07-30) -- ``--compare-runs``.
# ============================================================================


class AmbiguousBaseSchemeError(ValueError):
    """A run_manifest.json's seed entries disagree (or partially disagree)
    about ``base_scheme`` -- raised by ``_extract_base_scheme`` instead of
    guessing which one is real (review fix #4a/#4b: fail closed on a
    mixed-scheme manifest rather than silently picking ``seeds[0]``)."""


def _non_smoke_seed_entries(seeds) -> list:
    """Seed entries whose base-scheme provenance is in play -- i.e. every
    dict entry EXCEPT ones marked EXACTLY ``"smoke": True`` with no
    explicit ``base_scheme`` of their own.

    Two review-fix refinements (round 3, fix #7) over the original rule:
      - the smoke check is an IDENTITY comparison (``is True``), not a
        truthiness check -- ``"smoke": "false"`` (a string) or ``"smoke": 1``
        must not be silently treated as the marker.
      - an entry that is BOTH marked smoke AND carries an explicit
        ``base_scheme`` is NOT excluded -- fail toward VISIBILITY: if such
        an anomalous entry's scheme conflicts with a real entry's, that is
        a genuine ambiguity to surface, not something to hide by trusting
        the smoke marker over the data actually present. The normal
        trainer flow never produces this combination (--smoke omits
        base_scheme entirely -- train_qwen3_lora.py review fix #6), so
        this only matters for a hand-edited or otherwise anomalous
        manifest.

    A RUNBOOK smoke run appends to the SAME out/run_manifest.json a real
    campaign uses but never loads run_config.json, so in the NORMAL case
    it has no real base_scheme to report at all -- that's why it's
    excluded by default.
    """
    if not isinstance(seeds, list):
        return []
    return [
        s for s in seeds
        if isinstance(s, dict) and not (s.get("smoke") is True and "base_scheme" not in s)
    ]


def _extract_base_scheme(data):
    """Pull a ``base_scheme`` value out of a run_config.json, an identity
    receipt, or a run_manifest.json.

    Handles the three on-disk shapes ``--compare-runs`` is meant to point at:
      - a run_config.json (or a bare per-seed entry): ``base_scheme`` at
        the top level.
      - an identity receipt (T3): carries ``scheme``, not ``base_scheme``,
        at the top level -- read both keys so a dequant receipt is never
        mislabeled assumed-fp16 (review fix #4c).
      - a full box-side run_manifest.json (``{"seeds": [...]}``): every
        NON-SMOKE seed entry (see ``_non_smoke_seed_entries``) must agree.
        Two non-smoke entries with different explicit ``base_scheme``
        values, or a mix of some entries carrying it and others not,
        raises ``AmbiguousBaseSchemeError`` -- fail closed, never guess
        (review fix #4a/#4b). Only when EVERY non-smoke entry lacks
        ``base_scheme`` entirely is the manifest legacy v1/v2 (``None``,
        assumed fp16 by the caller with a printed note). A NON-STRING
        ``base_scheme`` value (``None``, an int, a list, ...) is invalid
        on its face and ALSO raises ``AmbiguousBaseSchemeError`` -- never
        fed to ``sorted()``/a ``set()`` (review fix #4, round 3: mixed
        types there raised a raw, uncaught ``TypeError``/exit(1), and an
        unhashable type like a list would crash even alone, before any
        cross-entry comparison).

    Returns ``None`` if no ``base_scheme``/``scheme`` can be found
    anywhere, or if ``data`` isn't even a dict (a malformed input is "not
    found", not a crash -- review fix #10). Raises
    ``AmbiguousBaseSchemeError`` on a mixed/partial-coverage/non-string
    manifest -- including a TOP-LEVEL ``base_scheme``/``scheme`` value
    that is present but not a string (review fix #3, round 4: the
    round-3 non-string guard only covered ``seeds[]`` entries, so a
    top-level ``{"base_scheme": None}`` silently returned ``None`` and
    downgraded to "legacy, assume fp16" -- wrong, since presence of the
    key at all signals real intent, an explicit null is not silence; and
    two inputs both carrying the SAME non-string value, e.g. ``{"base_scheme":
    42}`` on both sides, used to compare equal and report ``MATCH`` at
    ``--compare-runs`` exit 0, despite ``42`` never being a valid scheme).
    """
    if not isinstance(data, dict):
        return None
    if "base_scheme" in data:
        value = data["base_scheme"]
        if not isinstance(value, str):
            raise AmbiguousBaseSchemeError(
                f"invalid base_scheme: non-string value {value!r} -- the key's presence "
                "signals real intent, so a null/wrong-type value is refused outright, "
                "never silently downgraded to 'legacy, assume fp16'"
            )
        return value
    if "scheme" in data:
        value = data["scheme"]
        if not isinstance(value, str):
            raise AmbiguousBaseSchemeError(
                f"invalid scheme: non-string value {value!r} -- the key's presence signals "
                "real intent, so a null/wrong-type value is refused outright, never "
                "silently downgraded to 'legacy, assume fp16'"
            )
        return value

    seeds = data.get("seeds")
    if isinstance(seeds, list) and seeds:
        real_seeds = _non_smoke_seed_entries(seeds)
        if not real_seeds:
            return None
        with_scheme = [s for s in real_seeds if "base_scheme" in s]
        if not with_scheme:
            return None  # legacy v1/v2: no non-smoke entry carries base_scheme at all
        if len(with_scheme) != len(real_seeds):
            raise AmbiguousBaseSchemeError(
                "mixed-scheme manifest: some non-smoke seed entries carry "
                "base_scheme and others don't -- refusing to guess which "
                "seeds are legacy vs current"
            )
        values = [s["base_scheme"] for s in with_scheme]
        # Type-check BEFORE building a set: an unhashable value (a list, a
        # dict) would raise TypeError at set-construction time, before
        # sorted() is even reached -- so this must happen first, not as a
        # try/except around the set/sort below.
        non_string = [v for v in values if not isinstance(v, str)]
        if non_string:
            raise AmbiguousBaseSchemeError(
                f"mixed-scheme manifest: non-string base_scheme value(s) found {non_string!r} "
                "-- refusing to treat this manifest as any single scheme"
            )
        distinct = sorted(set(values))
        if len(distinct) > 1:
            raise AmbiguousBaseSchemeError(
                f"mixed-scheme manifest: seed entries disagree on base_scheme {distinct!r}"
            )
        return distinct[0]
    return None


def _extract_base_source_sha256(data):
    """Pull a ``base_source_sha256`` value out of the same shapes
    ``_extract_base_scheme`` handles (review fix #4d, extended by round-3
    fix #6).

    Scans ALL non-smoke seed entries (not just the first one found, as
    the original version did) -- entries that disagree on a non-``None``
    value raise the SAME ``AmbiguousBaseSchemeError`` fail-closed refusal
    mixed ``base_scheme`` values do. Entries that simply lack the key are
    not themselves a conflict: this field is a secondary ``--compare-runs``
    signal, not the primary scheme identity ``base_scheme``'s stricter
    partial-coverage rule guards, so partial presence alone resolves to
    whichever value is present rather than raising. ``None`` if nothing
    anywhere carries it.

    A plain list (not a set) tracks distinct values seen, deliberately --
    ``base_source_sha256`` values are always expected to be strings, but
    unlike ``base_scheme`` this function never asserts that, so it must
    stay safe against an unhashable value too.
    """
    if not isinstance(data, dict):
        return None
    if "base_source_sha256" in data:
        return data["base_source_sha256"]
    seeds = data.get("seeds")
    values = []
    if isinstance(seeds, list):
        for s in _non_smoke_seed_entries(seeds):
            if "base_source_sha256" in s:
                values.append(s["base_source_sha256"])
    if not values:
        return None
    distinct = []
    for v in values:
        if v not in distinct:
            distinct.append(v)
    if len(distinct) > 1:
        raise AmbiguousBaseSchemeError(
            f"mixed-scheme manifest: seed entries disagree on base_source_sha256 {distinct!r}"
        )
    return distinct[0]


def check_same_base_scheme(manifest_or_config_a, manifest_or_config_b) -> dict:
    """Refuse to treat two runs as comparable if they used different base
    schemes or (same scheme, different) base sources.

    Runs built under ``config.BASE_SCHEME_FP16`` vs ``config.BASE_SCHEME_DEQUANT``
    recovered their base weights through different paths (direct fp16 read vs
    Q4_K_M dequantization) -- comparing their results as if the base were
    identical would be a silent confound. A dict with no ``base_scheme``
    anywhere (v1/v2 predates this field) is ASSUMED to be fp16; ``assumed_a``/
    ``assumed_b`` tell the caller which side that assumption was made on, so
    it can be surfaced to the operator rather than silently trusted.

    ``base_source_sha256`` is also compared when BOTH sides carry one (review
    fix #4d): same scheme, different source pin (e.g. two different dequant
    manifests, or a moved fp16 revision) is also a mismatch -- one side or
    both missing the field is not itself a conflict, there is simply nothing
    to compare there.

    Returns ``{"scheme_a", "scheme_b", "assumed_a", "assumed_b",
    "scheme_match", "source_sha_a", "source_sha_b", "source_sha_match",
    "match"}``; ``match`` is ``scheme_match and source_sha_match``.
    Propagates ``AmbiguousBaseSchemeError`` from ``_extract_base_scheme``
    uncaught -- callers (``_main_compare_runs``) turn it into a clean
    refusal rather than a traceback.
    """
    scheme_a = _extract_base_scheme(manifest_or_config_a)
    scheme_b = _extract_base_scheme(manifest_or_config_b)
    assumed_a = scheme_a is None
    assumed_b = scheme_b is None
    resolved_a = scheme_a if scheme_a is not None else config.BASE_SCHEME_FP16
    resolved_b = scheme_b if scheme_b is not None else config.BASE_SCHEME_FP16
    scheme_match = resolved_a == resolved_b

    source_a = _extract_base_source_sha256(manifest_or_config_a)
    source_b = _extract_base_source_sha256(manifest_or_config_b)
    source_sha_match = True if (source_a is None or source_b is None) else source_a == source_b

    return {
        "scheme_a": resolved_a,
        "scheme_b": resolved_b,
        "assumed_a": assumed_a,
        "assumed_b": assumed_b,
        "scheme_match": scheme_match,
        "source_sha_a": source_a,
        "source_sha_b": source_b,
        "source_sha_match": source_sha_match,
        "match": scheme_match and source_sha_match,
    }


def _main_dequant(dequant_dir: Path, receipt_path: Path, *, gguf_path=None, skip_file_sha: bool = False) -> int:
    """``--dequant-dir`` entry point: run ``check_dequant_manifest`` and, on
    PASS, write the identity receipt with the additive ``scheme``/``chain``
    fields (T3)."""
    result = check_dequant_manifest(dequant_dir, gguf_path=gguf_path, skip_file_sha=skip_file_sha)
    print(_format_dequant_table(result["checks"]))

    if not result["ok"]:
        return 2

    manifest = result["manifest"] or {}
    manifest_claimed_gguf_sha256 = _as_dict(manifest.get("source_gguf")).get("sha256")
    source_verified = not skip_file_sha
    chain = {
        # Review fix #2, round 3: a --skip-file-sha receipt used to shape
        # this field IDENTICALLY to a verified one (both just
        # "gguf_sha256": <the manifest's self-report>), with only a
        # buried checks[] detail distinguishing them -- easy to miss
        # downstream. Now explicit: source_verified tells the reader
        # outright whether the local GGUF file was actually hashed, and
        # an unverified claim rides under its own differently-named key
        # so it can never be mistaken for the cryptographically-confirmed
        # value.
        "source_verified": source_verified,
        "gguf_sha256": None if skip_file_sha else manifest_claimed_gguf_sha256,
        "dequant_manifest_sha256": result["manifest_sha256"],
        # Recomputed (review fix #3), never the manifest's own copy --
        # by the time we get here check_dequant_manifest has already
        # proven the two agree, so this is "the digest we verified", not
        # "the digest we trusted".
        "content_digest": result["content_digest"],
    }
    if skip_file_sha:
        chain["manifest_claimed_gguf_sha256"] = manifest_claimed_gguf_sha256
    receipt = {
        "verdict": "PASS",
        "checked_at": time.time(),
        "scheme": config.BASE_SCHEME_DEQUANT,
        "chain": chain,
        "checks": [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in result["checks"]],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return 0


def _main_compare_runs(path_a: Path, path_b: Path, *, allow_cross_scheme: bool) -> int:
    """``--compare-runs A B`` entry point (T4 #5): loud refusal on scheme or
    base_source_sha256 mismatch. Never a traceback (review fix #10) -- a
    missing file, invalid JSON, invalid UTF-8 (review fix #4, round 4), or
    an ambiguous/mixed-scheme manifest all become a clean nonzero exit +
    message, with the exit code distinguishing WHICH kind of problem it
    was (review fix #5, round 3 -- see the module docstring's exit-code
    taxonomy)."""
    try:
        data_a = json.loads(path_a.read_text(encoding="utf-8"))
        data_b = json.loads(path_b.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError (review fix #4, round 4): .read_text() on a
        # file that isn't valid UTF-8 raises this -- NOT an OSError and
        # NOT a json.JSONDecodeError, so it fell through both existing
        # handlers uncaught (a traceback, exit 1) until this fix.
        print(f"REFUSED: could not read a --compare-runs input: {exc}", file=sys.stderr)
        return 3  # infra error -- nothing about base schemes was even evaluated
    except json.JSONDecodeError as exc:
        print(f"REFUSED: a --compare-runs input is not valid JSON: {exc}", file=sys.stderr)
        return 3  # infra error

    # Review fix #3, round 4: valid JSON that isn't a JSON OBJECT at the
    # top level (a bare list, string, number, ...) is an INFRA problem,
    # not a legacy-shaped run -- _extract_base_scheme's own "not a dict ->
    # None" contract exists for safety deep in a nested lookup, not to
    # paper over a --compare-runs input that was never shaped like a run
    # artifact in the first place.
    for label, path, data in (("A", path_a, data_a), ("B", path_b, data_b)):
        if not isinstance(data, dict):
            print(
                f"REFUSED: --compare-runs input {label} ({path}) is valid JSON but not a "
                f"JSON object (got {type(data).__name__}) -- cannot evaluate base_scheme.",
                file=sys.stderr,
            )
            return 3  # infra error

    try:
        result = check_same_base_scheme(data_a, data_b)
    except AmbiguousBaseSchemeError as exc:
        print(f"REFUSED: {path_a} vs {path_b}: {exc}", file=sys.stderr)
        return 2  # substantive refusal -- the data itself is too ambiguous to compare

    if result["assumed_a"]:
        print(f"NOTE: {path_a} has no base_scheme -- assuming {result['scheme_a']!r} (v1/v2 runs predate this field)")
    if result["assumed_b"]:
        print(f"NOTE: {path_b} has no base_scheme -- assuming {result['scheme_b']!r} (v1/v2 runs predate this field)")
    print(f"{path_a}: base_scheme={result['scheme_a']!r} base_source_sha256={result['source_sha_a']!r}")
    print(f"{path_b}: base_scheme={result['scheme_b']!r} base_source_sha256={result['source_sha_b']!r}")

    if result["match"]:
        print("MATCH -- same base scheme (and source, where recorded), comparison is valid")
        return 0

    problems = []
    if not result["scheme_match"]:
        problems.append(f"base_scheme differs: {result['scheme_a']!r} vs {result['scheme_b']!r}")
    if not result["source_sha_match"]:
        problems.append(f"base_source_sha256 differs: {result['source_sha_a']!r} vs {result['source_sha_b']!r}")
    message = (
        f"BASE SCHEME MISMATCH: {path_a} vs {path_b}: " + "; ".join(problems) +
        " -- runs from different base schemes/sources must never be silently compared."
    )
    if allow_cross_scheme:
        print(f"WARNING (--allow-cross-scheme): {message}")
        return 0
    print(f"REFUSED: {message}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="verify_base_identity")
    parser.add_argument("--pinned-dir", default=None, help="directory holding the pinned config.json + tokenizer.json (fp16 structural-identity mode)")
    parser.add_argument("--gguf", default=str(DEFAULT_BASE_GGUF_PATH), help="path to the local quantized GGUF file (fp16 mode)")
    parser.add_argument(
        "--skip-file-sha",
        action="store_true",
        help="skip hashing the (multi-GB) GGUF file; default is to hash it and compare against the pinned sha256 (fp16 mode)",
    )
    parser.add_argument("--receipt", default=None, help="where to write the PASS receipt (default: config.DATA_DIR/identity_receipt.json)")
    parser.add_argument(
        "--dequant-dir",
        default=None,
        help="directory holding a gguf_to_hf.py dequant output (HF-format dir + dequant_manifest.json); "
        "switches to the T3 hash-chain identity mode instead of the fp16 structural comparator",
    )
    parser.add_argument(
        "--source-gguf",
        default=None,
        help="dequant mode only: path to the local GGUF file the dequant claims to derive from "
        "(default: the manifest's own source_gguf.path); hashed and required to equal both "
        "manifest.source_gguf.sha256 and the pinned EXPECTED_BASE_GGUF_SHA256 (review fix #3); "
        "--skip-file-sha also skips this hash",
    )
    parser.add_argument(
        "--compare-runs",
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help="compare two run artifacts' base_scheme and base_source_sha256 (a run_config.json or a "
        "run_manifest.json) and refuse loudly on a mismatch (T4 #5); see --allow-cross-scheme. "
        "Mutually exclusive with --pinned-dir/--dequant-dir.",
    )
    parser.add_argument(
        "--allow-cross-scheme",
        action="store_true",
        help="with --compare-runs, permit a base_scheme/base_source_sha256 mismatch instead of refusing "
        "(explicit operator override)",
    )
    args = parser.parse_args(argv)

    if args.compare_runs and (args.pinned_dir or args.dequant_dir):
        parser.error("--compare-runs is mutually exclusive with --pinned-dir/--dequant-dir (review fix #9)")

    if args.compare_runs:
        return _main_compare_runs(
            Path(args.compare_runs[0]), Path(args.compare_runs[1]), allow_cross_scheme=args.allow_cross_scheme
        )

    receipt_path = Path(args.receipt) if args.receipt else (config.DATA_DIR / "identity_receipt.json")

    if args.dequant_dir and args.pinned_dir:
        parser.error("--dequant-dir and --pinned-dir are mutually exclusive (dequant hash-chain mode vs fp16 structural mode)")
    if not args.dequant_dir and not args.pinned_dir:
        parser.error("one of --pinned-dir, --dequant-dir, or --compare-runs is required")

    if args.dequant_dir:
        return _main_dequant(
            Path(args.dequant_dir),
            receipt_path,
            gguf_path=Path(args.source_gguf) if args.source_gguf else None,
            skip_file_sha=args.skip_file_sha,
        )

    gguf_path = Path(args.gguf)
    pinned_dir = Path(args.pinned_dir)

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
        # Additive (T3, 2026-07-30): every freshly-written receipt is now
        # self-describing. A receipt written before this field existed
        # carries no "scheme" key at all -- consumers (upload_guard.
        # check_base_scheme, check_same_base_scheme/--compare-runs) treat
        # that absence as config.BASE_SCHEME_FP16, tested.
        "scheme": config.BASE_SCHEME_FP16,
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

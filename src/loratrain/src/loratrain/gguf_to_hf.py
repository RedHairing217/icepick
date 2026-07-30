"""GGUF (Q4_K_M) -> HF-format dequantized training base (T1).

Mission (see AGENTS.md / lora_campaign_results.md): the next LoRA campaign
trains against the ACTUAL deployment weights (``Qwen3-8B-Q4_K_M.gguf``,
dequantized), not the upstream fp16 revision -- so the adapter is fit
against the weights llama.cpp actually serves. ``train_lora.py`` already
accepts an HF-format directory as its base; this module is the pre-step
that produces one from a GGUF. It is a standalone, stdlib+numpy+gguf-py
tool -- it does not import ``loratrain.config`` and is not wired into the
build/train pipeline by this change (a parallel agent owns that wiring).

Ground truth (read-only, never modified by this module)::

    gguf-py            /Users/redhairing/src/llama.cpp/gguf-py   (pinned checkout)
    converter mapping   conversion/qwen.py, conversion/base.py,
                        gguf-py/gguf/tensor_mapping.py, gguf-py/gguf/quants.py
    pinned GGUF         ~/.lmstudio/models/lmstudio-community/Qwen3-8B-GGUF/
                        Qwen3-8B-Q4_K_M.gguf (5,027,783,968 bytes)

``gguf`` is NEVER pip-installed: it is imported via an explicit
``sys.path`` insertion of ``GGUF_PY_DIR`` (overridable with
``--gguf-py-dir``), done lazily inside ``load_gguf_module`` rather than at
module import time, so importing THIS module never requires the gguf-py
checkout to exist (tests that only exercise the safetensors writer, the
config reconstruction, or the sha helpers need no GGUF/gguf-py at all).
``torch``/``transformers``/``safetensors`` are NOT imported anywhere in
this module -- see "Safetensors writer" below.

Why float32 output (deliberate, not a default-fallback): this module's
contract is PARITY with gguf-py's own reference dequantize implementation,
computed in float32, bit-for-bit -- NOT a claim of mathematical exactness.
gguf-py's ``quants.dequantize_blocks`` implementations compute directly in
float32 (see e.g. ``Q4_K.dequantize_blocks``/``Q6_K.dequantize_blocks`` in
``gguf-py/gguf/quants.py``); Q4_K's own final step is an fp32 SUBTRACTION
of two products (``d*sc*qs - dmin*m``), which can itself round, so "exact"
would overclaim what fp32 guarantees here. What IS guaranteed: storing the
reconstruction in bf16 or fp16 would silently round the dequantized grid a
SECOND time on top of whatever gguf-py's own float32 arithmetic already
produced, for no benefit -- the whole point of dequantizing is to reproduce
gguf-py's reference computation bit-for-bit, not to re-approximate it
further. So this module always writes float32 shards, matching gguf-py's
own computation precision one-for-one.

NOTE (report-only, no behavior change): the training consumer currently
loads HF-format bases with ``torch_dtype=bfloat16``, which will round this
module's fp32 dequant grid again at LOAD time regardless of what is stored
on disk. Whether the dequant scheme should force an fp32/fp16 load path
instead (or whether bf16-at-load is an acceptable, deliberate second
rounding) is an open decision for Nicky -- this module does not decide it.

Permutation verdict: dense Qwen3 (``Qwen3ForCausalLM`` / ``MODEL_ARCH.
QWEN3``, this GGUF's ``general.architecture``) is registered in
``conversion/qwen.py`` as ``class Qwen3Model(Qwen2Model)``. Walking ITS OWN
MRO for ``modify_tensors`` (``Qwen3Model`` -> ``Qwen2Model`` ->
``ModelBase.modify_tensors`` in ``conversion/base.py``) shows NO
permutation, reshape, or transpose applied to q_proj/k_proj anywhere on
that chain -- only a rerank-specific cls-out-head special case
(Qwen3Model), an hf_arch name prefix fixup (Qwen2Model), and a bare
tensor-name-map lookup (ModelBase). This claim is scoped to the qwen3 MRO
chain specifically, NOT a universal statement about ``conversion/`` as a
whole -- several OTHER, unrelated architectures there (``LlamaModel``,
``Qwen3NextModel``, and others) DO apply their own ``permute()``s; those
are irrelevant to this GGUF's arch and are not part of this claim. The
inverse for qwen3 is therefore identity: q_proj/k_proj weights are copied
through unchanged. See ``PERMUTATION_APPLIED`` / ``PERMUTATION_EVIDENCE``
below, echoed verbatim into ``dequant_manifest.json``'s ``permutation``
block.

Shape/orientation rule (requirement 5): GGML/GGUF tensor-info dims (``ne[]``)
are the REVERSE of the HF/numpy row-major shape -- proven directly from
gguf-py's own writer/reader, not asserted from memory:
``gguf_writer.GGUFWriter.write_ti_data_to_file`` writes
``ti.shape[n_dims - 1 - j]`` for ``j in range(n_dims)`` (the numpy shape,
reversed); ``gguf_reader.GGUFReader._build_tensors`` reads it back with
``np_dims = tuple(reversed(dims.tolist()))`` and reshapes the flat tensor
bytes with that reversed tuple. This module therefore does NOT reverse or
transpose anything itself: ``hf_shape()`` below reads the raw on-disk
``ReaderTensor.shape`` (GGML ne-order) and reverses it ONCE to get the
HF-order shape, and ``dequantize_tensor`` trusts ``gguf.quants.dequantize``'s
own output shape (``quant_shape_from_byte_shape`` restores exactly that
same HF-order shape for quantized tensors). ``test_gguf_to_hf.py`` proves
this end-to-end with a real ``GGUFWriter``-written non-square F32 fixture
read back through this module's full pipeline.

Name mapping (requirement 3): built from gguf-py's own
``gguf.tensor_mapping``/``constants`` machinery, not reimplemented.
``TENSOR_NAMES``/``MODEL_TENSORS[MODEL_ARCH.QWEN3]`` give the authoritative
GGUF-side bare tensor name per role (e.g. ``MODEL_TENSOR.ATTN_Q ->
"blk.{bid}.attn_q"``); this module pins the ONE real Qwen3 HF checkpoint
name per role (``QWEN3_HF_NAME_TEMPLATES``, e.g. ``"model.layers.{bid}.
self_attn.q_proj"``) and ``validate_hf_name_templates`` cross-checks every
pin against ``gguf.tensor_mapping.get_tensor_name_map``'s own
``TensorNameMap.get_type_and_name`` -- a tripwire that hard-fails if
gguf-py's alias table ever stops recognizing one of these names for qwen3,
rather than a silent guess. ``build_gguf_to_hf_name_map`` then resolves
EVERY tensor name actually present in a real file and hard-fails
(``TensorMappingError``) on any unmapped or doubly-mapped tensor -- nothing
is silently dropped. Qwen3's per-layer ``self_attn.q_norm``/``k_norm``
(QK-norm) are ordinary pinned roles like any other, no special-casing.

Safetensors writer (requirement 6): ``torch``/``transformers``/
``safetensors`` are likely not installed in this environment and are never
required dependencies of this module. The on-disk format (8-byte
little-endian header-length, then that many bytes of UTF-8 JSON header,
then raw little-endian tensor bytes back-to-back in header order) is
reproduced here as a small self-contained writer:
``build_safetensors_header``/``write_safetensors_file``. Tensor names are
always written in SORTED order and the header JSON is dumped with
``sort_keys=True`` and compact separators, so a given tensor set always
produces byte-identical shard files. Every shard carries a
``__metadata__: {"format": "pt"}`` entry -- several transformers/
safetensors versions hard-reject an archive with no ``"format"`` metadata
key ("does not contain the valid metadata"), even though this writer never
touches torch itself.

Determinism claim (SCOPED -- requirement 13): running ``convert()`` twice
on the same input produces byte-identical safetensors shards, a
byte-identical ``model.safetensors.index.json``, a byte-identical
``config.json``, and an identical ``determinism.content_digest`` (hashed
over sorted ``gguf_name:sha256`` lines, so it is sharding-independent on
top of the byte-identity above). ``dequant_manifest.json`` itself is NOT
byte-identical across runs: it embeds ``created_utc`` (a timestamp) and
``tool.llama_cpp_checkout.commit`` (whatever the pinned llama.cpp checkout
happens to be at run time on a given machine) -- both expected to differ,
never claimed otherwise.

Atomic publish (requirement 1): ``convert()`` refuses a non-empty
``--out`` outright (never silently merges into an existing directory),
verifies ``--expected-sha256`` (when given) BEFORE any dequantization
work, and writes every output file into a temp directory that is a
SIBLING of ``--out`` (same parent, so the final ``os.replace`` is an
atomic same-filesystem rename), publishing it to the final path only
after every gate above has passed. A refused conversion -- bad hash, a
non-empty destination, a sidecar mismatch, an unmapped tensor -- therefore
never leaves a partial or complete-but-wrong HF directory sitting at
``--out``.

Memory (requirement 2): ``convert()`` is two-pass. Pass 1 plans every
tensor's HF name/shape/output byte size and the shard split from GGUF
HEADER metadata only (no dequantization). Pass 2 dequantizes, hashes, and
writes ONE SHARD at a time, releasing each shard's tensors before moving
to the next -- peak RSS is therefore bounded by roughly one shard's worth
of float32 tensors (plus one oversized tensor, if any single tensor
exceeds ``--shard-max-bytes``), not the full converted model. Each
tensor's raw little-endian bytes are computed exactly ONCE (``arr.
tobytes()``) and reused for both its per-tensor sha256 and the bytes
actually written -- ``write_safetensors_file`` returns the per-tensor
sha256 map it computed while writing, rather than the caller hashing the
tensor again separately.

Sidecar validation (requirement 3): beyond the weight-shape fields listed
in ``reconstruct_config``, a ``--sidecar-dir`` config.json is also
cross-validated on: ``rope_scaling`` (read from the GGUF's ``{arch}.rope.
scaling.*`` KV family; ``None`` on one side and non-``None`` on the other
is a hard fail, not a silent skip), ``hidden_act`` (must be ``"silu"`` when
present -- the qwen3 architecture implies SwiGLU/silu unconditionally),
``attention_bias`` (must agree with whether this GGUF's own tensor
directory actually contains any ``*.bias`` tensor), and ``use_sliding_
window``/``sliding_window`` (this tool only supports plain global
attention; ``assert_no_sliding_window`` hard-fails the WHOLE conversion,
sidecar or not, if the GGUF's own ``{arch}.attention.sliding_window`` KV
is present at all). Any sidecar config.json key not covered by one of
these checks (or the two exempt dual-pins) is listed verbatim in the
manifest's ``sidecars.notes`` as an explicitly unvalidated pass-through --
never silently ignored without a trace.

Tokenizer sidecar validation (requirement 4): a ``--sidecar-dir``
``tokenizer.json`` is cross-checked against the GGUF's OWN embedded
tokenizer -- header/KV data, never tensor payload: ``len(tokenizer.ggml.
tokens)`` must equal the reconstructed ``vocab_size`` (llama.cpp pads the
token list to the embedding row count; see ``verify_base_identity.py``'s
established "vocab_n[rows==config.vocab_size]" check for the same
invariant), and the GGUF's own token STRING for ``bos_token_id``/
``eos_token_id`` (when present) must match the sidecar tokenizer.json's
string for that same id. This is a spot-check, not full-vocabulary
equality, but it is exactly enough to close the silent-tokenizer-swap
vector this tool exists to close.

Execution scope THIS SESSION (see the T1 brief): only fixture-based tests
and ``--plan`` (metadata-only census + mapping proof) against the real
GGUF are run. ``convert()`` -- the real dequantize-and-write path -- is
exercised only against tiny synthetic fixtures; it is never invoked
against the pinned 5 GB file in this session (that would materialize the
full 8B model and contend with the live eval for I/O/RAM), and this
session never hashes that file either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ============================================================================
# Tool-local constants. config.py is owned by a parallel agent this window --
# nothing in this module reads or writes it; every pin this tool needs lives
# here instead.
# ============================================================================

GGUF_PY_DIR = "/Users/redhairing/src/llama.cpp/gguf-py"
LLAMA_CPP_CHECKOUT_DIR = "/Users/redhairing/src/llama.cpp"

DEFAULT_SHARD_MAX_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB

SCHEMA_VERSION = 1
BASE_SCHEME = "dequant_q4km"
TOOL_NAME = "loratrain.gguf_to_hf"
TOOL_VERSION = "1.0.0"

# Requirement 2: Q4_K_M should only ever contain these three qtypes for the
# qwen3 arch (norms stay F32; everything else is Q4_K or Q6_K under
# llama.cpp's "mostly Q4_K_M" mixing). Anything else encountered is a hard
# error naming the tensor -- see dequantize_tensor.
SUPPORTED_QTYPE_NAMES = ("F32", "Q4_K", "Q6_K")

# Tokenizer files are copied ONLY from --sidecar-dir, never reconstructed
# (requirement 7). Whichever of these exist in the sidecar dir are copied
# through unchanged; absence of any of them is fine and noted in the
# manifest, never an error.
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)

# Every safetensors shard carries this __metadata__ block (requirement 6):
# several transformers/safetensors versions hard-reject an archive with no
# "format" metadata key at all ("does not contain the valid metadata").
# Values must be strings per the safetensors spec.
SAFETENSORS_METADATA = {"format": "pt"}

# The ONE real Qwen3-8B (dense, "Qwen3ForCausalLM") HF checkpoint tensor
# name per gguf-py MODEL_TENSOR role -- see module docstring "Name mapping".
# Keys are gguf.MODEL_TENSOR member NAMES (strings, so this dict needs no
# gguf-py import to define); validate_hf_name_templates() resolves them to
# the enum and cross-checks against gguf.tensor_mapping at runtime.
QWEN3_HF_NAME_TEMPLATES = {
    "TOKEN_EMBD": "model.embed_tokens",
    "OUTPUT_NORM": "model.norm",
    "OUTPUT": "lm_head",
    "ATTN_NORM": "model.layers.{bid}.input_layernorm",
    "ATTN_Q": "model.layers.{bid}.self_attn.q_proj",
    "ATTN_Q_NORM": "model.layers.{bid}.self_attn.q_norm",
    "ATTN_K": "model.layers.{bid}.self_attn.k_proj",
    "ATTN_K_NORM": "model.layers.{bid}.self_attn.k_norm",
    "ATTN_V": "model.layers.{bid}.self_attn.v_proj",
    "ATTN_OUT": "model.layers.{bid}.self_attn.o_proj",
    "FFN_NORM": "model.layers.{bid}.post_attention_layernorm",
    "FFN_GATE": "model.layers.{bid}.mlp.gate_proj",
    "FFN_DOWN": "model.layers.{bid}.mlp.down_proj",
    "FFN_UP": "model.layers.{bid}.mlp.up_proj",
}

# Requirement 4 -- recorded here as the single source of truth for the
# manifest's `permutation` block; see the module docstring for the full
# MRO-walk argument this constant summarizes.
PERMUTATION_APPLIED = False
PERMUTATION_EVIDENCE = (
    "SCOPE: this claim covers only the qwen3 MRO chain below, not a universal "
    "statement about conversion/ as a whole (several other, unrelated "
    "architectures there -- e.g. LlamaModel, Qwen3NextModel -- do apply their "
    "own permute()s; irrelevant to this GGUF's arch). "
    "conversion/qwen.py:155-156 '@ModelBase.register(\"Qwen3ForCausalLM\", "
    "\"Qwen3Model\") class Qwen3Model(Qwen2Model):' -- its modify_tensors "
    "(qwen.py:236-250) only special-cases rerank cls-out-head extraction and "
    "otherwise calls 'yield from super().modify_tensors(data_torch, name, bid)'. "
    "Qwen2Model.modify_tensors (qwen.py:67-70) only does 'if self.hf_arch == "
    "\"Qwen2Model\": name = f\"model.{name}\"' then falls through again. "
    "ModelBase.modify_tensors, the base case reached at the end of this MRO "
    "chain (conversion/base.py:617-642), is 'new_name = self.map_tensor_name"
    "(name); ... return [(new_name, data_torch)]' -- a bare rename, no "
    "reshape/permute. Walking Qwen3Model's full modify_tensors MRO (Qwen3Model "
    "-> Qwen2Model -> ModelBase) therefore shows NO transform applied to "
    "q_proj/k_proj on the dense qwen3 conversion path. Verdict: identity -- "
    "q_proj/k_proj are copied through unchanged both directions."
)


# ============================================================================
# Exceptions (house idiom: RuntimeError/ValueError subclasses with a
# docstring explaining exactly what tripped them and why it is a hard fail).
# ============================================================================


class UnsupportedArchitectureError(RuntimeError):
    """The GGUF's ``general.architecture`` is not ``"qwen3"``."""


class MissingMetadataError(RuntimeError):
    """A required GGUF key-value metadata field is absent."""


class UnsupportedQuantTypeError(RuntimeError):
    """A tensor's GGML quant type is not one of F32/Q4_K/Q6_K.

    Q4_K_M should never contain anything else (module docstring); silently
    skipping or mis-dequantizing such a tensor would produce a directory
    that loads and trains but is quietly wrong, which is the primary
    correctness risk this whole tool exists to avoid.
    """


class TensorMappingError(RuntimeError):
    """The GGUF<->HF tensor name mapping is not bijective, or is out of sync
    with gguf-py's own tensor_mapping machinery.

    Raised by ``validate_hf_name_templates`` (a hardcoded
    ``QWEN3_HF_NAME_TEMPLATES`` pin no longer matches gguf-py's
    ``TensorNameMap``) and by ``build_gguf_to_hf_name_map`` (some tensor
    actually present in the file has no known role, or two tensors would
    resolve to the same HF name) -- either way, nothing is silently
    dropped or guessed.
    """


class ShapeMismatchError(RuntimeError):
    """A dequantized tensor's shape does not match its GGUF header shape."""


class SidecarMismatchError(RuntimeError):
    """A ``--sidecar-dir`` config.json disagrees with GGUF metadata on a
    weight-relevant field (``max_position_embeddings``/``torch_dtype`` are
    the only known, deliberately exempt dual-pins -- see module docstring
    and ``verify_base_identity.py``'s own context-length dual pin for the
    established precedent this mirrors).
    """


class SourceHashMismatchError(RuntimeError):
    """The source GGUF's sha256 does not match ``--expected-sha256``.

    Raised BEFORE any dequantization work or filesystem write (requirement
    1: atomic publish) -- a hash mismatch must never leave ANY output
    behind, complete or partial.
    """


class DestinationNotEmptyError(RuntimeError):
    """``--out`` already exists and is not empty.

    ``convert()`` never silently merges into an existing directory
    (requirement 1) -- remove or rename the destination first if replacing
    it was intended.
    """


class SidecarNotFoundError(RuntimeError):
    """``--sidecar-dir`` was given but has no readable ``config.json``."""


# ============================================================================
# gguf-py loading (explicit sys.path insertion; never pip-installed)
# ============================================================================


def load_gguf_module(gguf_py_dir=None):
    """Import the pinned gguf-py package via a SCOPED ``sys.path`` insertion.

    Never pip-installed (house rule) -- this machine vendors the gguf-py
    source tree at ``GGUF_PY_DIR`` (part of a pinned llama.cpp checkout;
    see ``LLAMA_CPP_CHECKOUT_DIR`` / ``_git_rev_parse_head``).
    ``gguf_py_dir`` overrides the module constant (``--gguf-py-dir``).
    Deliberately deferred to call time, not module import time, so that
    importing ``loratrain.gguf_to_hf`` itself never requires the gguf-py
    checkout to exist.

    The path is inserted, ``import gguf`` is performed, and the path is
    removed again in a ``finally`` -- it never lingers on ``sys.path``
    (requirement 12/minor 12): the gguf-py checkout's ROOT directory also
    contains importable top-level ``tests``/``examples`` packages that
    would otherwise silently shadow any later ``import tests``/``import
    examples`` elsewhere in the process for as long as the path stayed
    inserted. Like any ordinary Python import, the FIRST successful
    ``import gguf`` in a process still wins for that process's lifetime
    (``sys.modules`` caching) -- calling this again with a different
    ``gguf_py_dir`` after ``gguf`` is already imported does not re-import
    it from the new location; this function short-circuits on that cached
    module without touching ``sys.path`` at all in that case.
    """
    if "gguf" in sys.modules:
        return sys.modules["gguf"]

    path = str(gguf_py_dir) if gguf_py_dir is not None else GGUF_PY_DIR
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        import gguf  # noqa: E402 -- deliberately deferred, see docstring above
    finally:
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
    return gguf


def open_reader(gguf_path, gguf_py_dir=None):
    """Return ``(gguf_module, GGUFReader)`` for ``gguf_path``.

    ``GGUFReader.__init__`` parses the magic/version/KV section/tensor-info
    directory and opens the tensor data section as a lazy ``np.memmap`` --
    it never reads tensor payload bytes into RAM by itself (module
    docstring "Execution scope"), so this is cheap even against the real
    ~5 GB file.
    """
    gguf_mod = load_gguf_module(gguf_py_dir)
    reader = gguf_mod.GGUFReader(str(gguf_path))
    return gguf_mod, reader


def read_scalar_kv(reader, key: str, default=None):
    """Return one scalar (or string) GGUF KV field's value, or ``default``."""
    field = reader.get_field(key)
    if field is None:
        return default
    return field.contents()


# ============================================================================
# sha256 helpers (a small local copy of build_dataset.sha256_file's streamed
# idiom, deliberately NOT imported from loratrain.build_dataset -- this
# module has zero dependency on the loratrain corpus/config package so it
# stays usable standalone, matching its own module docstring).
# ============================================================================


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Stream ``path`` in chunks and return its sha256 hex digest."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_sha256(value: str) -> str:
    """Validate ``value`` is a 64-character hex sha256 digest and lowercase it.

    ``hashlib``'s own ``hexdigest()`` is always lowercase; a correct but
    uppercase (or mixed-case) ``--expected-sha256`` from an operator must
    still match -- comparing case-sensitively would refuse a correct
    digest only after already hashing the (possibly multi-GB) source file.
    Raises ``ValueError`` (caught by argparse's ``type=`` machinery at the
    CLI layer, and re-validated here for direct callers of ``convert()``)
    if ``value`` is not exactly 64 hex characters.
    """
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
        raise ValueError(
            f"{value!r} is not a valid sha256 digest (expected 64 hex characters)"
        )
    return value.lower()


def _git_rev_parse_head(repo_dir) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


_TOML_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def _read_gguf_py_version(gguf_py_dir) -> str | None:
    pyproject = Path(gguf_py_dir) / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    m = _TOML_VERSION_RE.search(text)
    return m.group(1) if m else None


# ============================================================================
# Shape (requirement 5)
# ============================================================================


def hf_shape(reader_tensor) -> tuple:
    """The HF/PyTorch row-major shape of a GGUF tensor, header-only.

    Never touches tensor payload bytes: ``ReaderTensor.shape`` is the raw
    on-disk GGML ``ne[]`` dims from the tensor-info directory (element
    counts per axis, independent of quantization/byte-packing), and GGML's
    ``ne[]`` is dimension-reversed relative to HF/numpy row-major (module
    docstring "Shape/orientation rule"). Reversing it once recovers the
    HF-order shape -- this is the ONLY place this module reverses a shape;
    everything downstream (``gguf.quants.dequantize``'s own output shape)
    is trusted as already being in this orientation.
    """
    return tuple(int(x) for x in reversed(reader_tensor.shape.tolist()))


# ============================================================================
# Name mapping (requirement 3)
# ============================================================================


def _reverse_tensor_names(gguf_mod, arch_enum) -> dict:
    """``{gguf_bare_template: MODEL_TENSOR}`` for every tensor role gguf-py
    registers for ``arch_enum`` (e.g. ``"blk.{bid}.attn_q" ->
    MODEL_TENSOR.ATTN_Q``). Built directly from gguf-py's own
    ``TENSOR_NAMES``/``MODEL_TENSORS`` tables -- not reimplemented.
    """
    reverse = {}
    for tensor_enum in gguf_mod.MODEL_TENSORS[arch_enum]:
        reverse[gguf_mod.TENSOR_NAMES[tensor_enum]] = tensor_enum
    return reverse


def build_reverse_tensor_names(gguf_mod, arch_enum=None) -> dict:
    arch_enum = arch_enum if arch_enum is not None else gguf_mod.MODEL_ARCH.QWEN3
    return _reverse_tensor_names(gguf_mod, arch_enum)


def validate_hf_name_templates(gguf_mod, arch_enum=None) -> None:
    """Cross-check ``QWEN3_HF_NAME_TEMPLATES`` against gguf-py's own
    ``TensorNameMap`` (requirement 3: "using gguf-py's tensor_mapping
    machinery").

    For every pinned ``(role, hf_template)``, formats both sides for a
    sample block index and asserts
    ``TensorNameMap.get_type_and_name(hf_name)`` resolves to EXACTLY the
    ``(MODEL_TENSOR, gguf_bare_name)`` pair this module's own
    ``TENSOR_NAMES``-derived reverse map expects. This is a tripwire
    against gguf-py ever dropping or renaming the alias this module
    hardcodes -- not a search or a guess; the hardcoded name is domain
    knowledge (Qwen3's real, public HF checkpoint naming), and this
    function's only job is to prove gguf-py still agrees with it.
    """
    arch_enum = arch_enum if arch_enum is not None else gguf_mod.MODEL_ARCH.QWEN3
    tensor_map = gguf_mod.get_tensor_name_map(arch_enum, 1)
    reverse = _reverse_tensor_names(gguf_mod, arch_enum)

    problems = []
    for role, hf_template in QWEN3_HF_NAME_TEMPLATES.items():
        tensor_enum = getattr(gguf_mod.MODEL_TENSOR, role, None)
        if tensor_enum is None:
            problems.append(f"{role}: no such gguf.MODEL_TENSOR member")
            continue
        gguf_template = gguf_mod.TENSOR_NAMES.get(tensor_enum)
        if gguf_template is None or reverse.get(gguf_template) != tensor_enum:
            problems.append(
                f"{role}: not registered in gguf.MODEL_TENSORS[MODEL_ARCH.QWEN3] "
                "(or TENSOR_NAMES has no entry for it)"
            )
            continue
        is_block = "{bid}" in gguf_template
        bid = 0 if is_block else None
        hf_name = hf_template.format(bid=bid) if is_block else hf_template
        gguf_name = gguf_template.format(bid=bid) if is_block else gguf_template
        result = tensor_map.get_type_and_name(hf_name)
        expected = (tensor_enum, gguf_name)
        if result != expected:
            problems.append(
                f"{role}: hardcoded HF template {hf_template!r} -> gguf-py "
                f"TensorNameMap resolves to {result!r}, expected {expected!r}"
            )

    if problems:
        raise TensorMappingError(
            "QWEN3_HF_NAME_TEMPLATES is out of sync with gguf-py's tensor_mapping "
            "machinery: " + "; ".join(problems) + " -- STOP, do not guess a fix "
            "(this is exactly the silent-wrong-weights risk the brief calls out)."
        )


_KNOWN_TENSOR_SUFFIXES = (".weight", ".bias")
_BLOCK_NAME_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def split_tensor_suffix(name: str) -> tuple:
    for suffix in _KNOWN_TENSOR_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    raise TensorMappingError(
        f"tensor {name!r} has no recognized suffix (expected one of "
        f"{_KNOWN_TENSOR_SUFFIXES})"
    )


def parse_gguf_bare_name(bare_name: str) -> tuple:
    """Split a suffix-less GGUF tensor name into ``(bid, bare_template)``.

    ``bid`` is ``None`` for non-block tensors (e.g. ``"token_embd"``);
    otherwise the integer block index and the ``{bid}``-parameterized
    template it came from (e.g. ``"blk.5.attn_q" -> (5, "blk.{bid}.attn_q")``).
    """
    m = _BLOCK_NAME_RE.match(bare_name)
    if m is None:
        return None, bare_name
    return int(m.group(1)), f"blk.{{bid}}.{m.group(2)}"


def resolve_hf_name(reverse: dict, gguf_name_with_suffix: str) -> tuple:
    """Resolve one on-disk GGUF tensor name to ``(hf_name, role)``.

    ``reverse`` is ``build_reverse_tensor_names``'s ``{gguf_bare_template:
    MODEL_TENSOR}`` map. Hard-fails (``TensorMappingError``, naming the
    tensor) if the bare name matches no registered qwen3 tensor role, or
    that role has no pin in ``QWEN3_HF_NAME_TEMPLATES`` -- e.g.
    ``MODEL_TENSOR.ROPE_FREQS`` IS in ``gguf.MODEL_TENSORS[MODEL_ARCH.
    QWEN3]`` (qwen3 CAN carry a rope_freqs tensor for scaled-rope variants)
    but this checkpoint's plain rope_theta case never emits one -- if a
    future GGUF ever did, resolving it would hard-fail here rather than
    silently drop it, exactly as the brief requires.
    """
    bare, suffix = split_tensor_suffix(gguf_name_with_suffix)
    bid, template = parse_gguf_bare_name(bare)
    tensor_enum = reverse.get(template)
    if tensor_enum is None:
        raise TensorMappingError(
            f"GGUF tensor {gguf_name_with_suffix!r}: bare name {bare!r} does not "
            f"match any MODEL_TENSOR role registered for qwen3 (template "
            f"{template!r} not in gguf.TENSOR_NAMES for this arch) -- refusing "
            "to silently drop it."
        )
    role = tensor_enum.name
    hf_template = QWEN3_HF_NAME_TEMPLATES.get(role)
    if hf_template is None:
        raise TensorMappingError(
            f"GGUF tensor {gguf_name_with_suffix!r}: role {role} (gguf template "
            f"{template!r}) has no pinned HF name in QWEN3_HF_NAME_TEMPLATES -- "
            "refusing to silently drop it."
        )
    hf_name = (hf_template.format(bid=bid) if bid is not None else hf_template) + suffix
    return hf_name, role


def build_gguf_to_hf_name_map(gguf_mod, gguf_names, arch_enum=None) -> dict:
    """Invert the qwen3 HF->GGUF tensor name mapping for a concrete tensor list.

    ``gguf_names`` is every on-disk tensor name (WITH suffix) actually
    present in the GGUF being converted. Returns ``{gguf_name: hf_name}``,
    guaranteed bijective. Collects EVERY unmapped or doubly-mapped tensor
    before raising (same "see the whole problem list in one pass" idiom as
    ``loratrain.config.validate_config``), rather than stopping at the
    first offender.
    """
    arch_enum = arch_enum if arch_enum is not None else gguf_mod.MODEL_ARCH.QWEN3
    reverse = build_reverse_tensor_names(gguf_mod, arch_enum)

    mapping = {}
    hf_to_gguf = {}
    unmapped = []
    doubly_mapped = []
    for name in gguf_names:
        try:
            hf_name, _role = resolve_hf_name(reverse, name)
        except TensorMappingError as exc:
            unmapped.append(str(exc))
            continue
        if hf_name in hf_to_gguf:
            doubly_mapped.append(
                f"HF name {hf_name!r} claimed by both {hf_to_gguf[hf_name]!r} "
                f"and {name!r}"
            )
            continue
        hf_to_gguf[hf_name] = name
        mapping[name] = hf_name

    if unmapped or doubly_mapped:
        parts = []
        if unmapped:
            parts.append(f"{len(unmapped)} unmapped tensor(s): " + "; ".join(unmapped))
        if doubly_mapped:
            parts.append(f"{len(doubly_mapped)} doubly-mapped name(s): " + "; ".join(doubly_mapped))
        raise TensorMappingError(
            "GGUF->HF tensor name mapping is not bijective (" + "; ".join(parts) +
            ") -- nothing was silently dropped; fix the mapping before proceeding."
        )
    return mapping


# ============================================================================
# Dequantization (requirement 2)
# ============================================================================


def dequantize_tensor(gguf_mod, reader_tensor) -> np.ndarray:
    """Return ``reader_tensor``'s values as a float32 numpy array, HF-oriented.

    F32 tensors are returned as-is (``ReaderTensor.data`` is already the
    correctly-shaped float32 array). Q4_K/Q6_K go through
    ``gguf.quants.dequantize``, which returns float32 already reshaped to
    the tensor's full HF-orientation shape (``quant_shape_from_byte_shape``
    restores the packed last dim; earlier dims are untouched by
    dequantization -- see ``hf_shape``). Any other qtype is a hard error
    naming the tensor (module docstring: Q4_K_M should never contain
    anything else).
    """
    qtype = reader_tensor.tensor_type
    name = reader_tensor.name
    F32 = gguf_mod.GGMLQuantizationType.F32
    Q4_K = gguf_mod.GGMLQuantizationType.Q4_K
    Q6_K = gguf_mod.GGMLQuantizationType.Q6_K

    if qtype == F32:
        return np.array(reader_tensor.data, dtype=np.float32, copy=True)
    if qtype in (Q4_K, Q6_K):
        out = gguf_mod.quants.dequantize(reader_tensor.data, qtype)
        return np.array(out, dtype=np.float32, copy=True)
    raise UnsupportedQuantTypeError(
        f"tensor {name!r} has qtype {qtype!r} -- only {SUPPORTED_QTYPE_NAMES} are "
        "supported (Q4_K_M should never contain anything else); refusing to "
        "silently mis-dequantize or skip it."
    )


def census_tensors(reader) -> dict:
    """``{"total": N, "by_qtype": {qtype_name: count, ...}}`` over ``reader.tensors``."""
    by_qtype: dict = {}
    for t in reader.tensors:
        by_qtype[t.tensor_type.name] = by_qtype.get(t.tensor_type.name, 0) + 1
    return {"total": len(reader.tensors), "by_qtype": dict(sorted(by_qtype.items()))}


# ============================================================================
# config.json reconstruction + sidecar cross-validation (requirement 7)
# ============================================================================


def _read_rope_scaling(reader, arch: str) -> dict | None:
    """Read the ``{arch}.rope.scaling.*`` KV family; ``None`` if disabled.

    Only ``type``/``factor``/``original_context_length`` are captured (the
    fields this checkpoint's plain, unscaled rope_theta case never
    populates) -- sufficient to (a) detect that scaling is enabled at all,
    which is the load-bearing fact for cross-validation, and (b) round-trip
    the common linear/dynamic/yarn ``factor`` value. A present ``rope.
    scaling.type`` with fields this reader does not enumerate would still
    be captured under ``type``/``factor``, so a sidecar disagreeing on
    those still hard-fails; it would not catch disagreement on an exotic
    yarn-only sub-field this checkpoint never needed.
    """
    scaling_type = read_scalar_kv(reader, f"{arch}.rope.scaling.type")
    if scaling_type is None:
        return None
    scaling = {"type": scaling_type}
    factor = read_scalar_kv(reader, f"{arch}.rope.scaling.factor")
    if factor is not None:
        scaling["factor"] = factor
    orig_ctx = read_scalar_kv(reader, f"{arch}.rope.scaling.original_context_length")
    if orig_ctx is not None:
        scaling["original_max_position_embeddings"] = orig_ctx
    return scaling


def read_qwen3_hparams(reader) -> dict:
    """Read the qwen3-arch hyperparameters this module needs from GGUF KV metadata.

    Hard-fails (``MissingMetadataError``) if a required key is absent --
    never silently derives/guesses a value the GGUF itself was supposed to
    carry (contrast with the ``head_dim`` HF-side fallback that ONLY
    applies to a --sidecar-dir config.json missing that field, in
    ``cross_validate_sidecar_config``). Also reads ``rope_scaling``
    (``None`` if this GGUF has no rope scaling KV at all -- see
    ``_read_rope_scaling``), ``sliding_window`` (``None`` unless the GGUF's
    ``{arch}.attention.sliding_window`` KV is present -- see
    ``assert_no_sliding_window``, which hard-fails the whole conversion if
    it is), and ``has_bias_tensors`` (whether ANY tensor in this GGUF ends
    in ``.bias`` -- Qwen3 dense has none; used to cross-validate a
    sidecar's ``attention_bias`` claim against the actual tensor
    directory, not just trust the config).
    """
    arch = read_scalar_kv(reader, "general.architecture")
    if arch != "qwen3":
        raise UnsupportedArchitectureError(
            f"expected general.architecture == 'qwen3', got {arch!r}"
        )

    required = {
        "block_count": f"{arch}.block_count",
        "hidden_size": f"{arch}.embedding_length",
        "intermediate_size": f"{arch}.feed_forward_length",
        "num_attention_heads": f"{arch}.attention.head_count",
        "num_key_value_heads": f"{arch}.attention.head_count_kv",
        "head_dim": f"{arch}.attention.key_length",
        "rms_norm_eps": f"{arch}.attention.layer_norm_rms_epsilon",
        "rope_theta": f"{arch}.rope.freq_base",
        "max_position_embeddings": f"{arch}.context_length",
    }
    hparams = {"architecture": arch}
    missing = []
    for field, key in required.items():
        value = read_scalar_kv(reader, key)
        if value is None:
            missing.append(key)
        hparams[field] = value
    if missing:
        raise MissingMetadataError(
            f"GGUF is missing required key(s): {missing} (architecture={arch!r})"
        )

    hparams["bos_token_id"] = read_scalar_kv(reader, "tokenizer.ggml.bos_token_id")
    hparams["eos_token_id"] = read_scalar_kv(reader, "tokenizer.ggml.eos_token_id")
    hparams["rope_scaling"] = _read_rope_scaling(reader, arch)
    hparams["sliding_window"] = read_scalar_kv(reader, f"{arch}.attention.sliding_window")
    hparams["has_bias_tensors"] = any(t.name.endswith(".bias") for t in reader.tensors)
    return hparams


def assert_no_sliding_window(hparams: dict) -> None:
    """Hard-fail the whole conversion if the GGUF declares a sliding window.

    This tool's tensor role list, name mapping, and every test were built
    against Qwen3-8B's plain global-attention path. A GGUF whose ``{arch}.
    attention.sliding_window`` KV is present describes a different
    attention regime (a per-layer full/sliding pattern, needing metadata
    this module never reads) -- proceeding would silently produce a
    directory that loads and trains but attends over the wrong window.
    Unsupported, not guessed: STOP.
    """
    sliding_window = hparams.get("sliding_window")
    if sliding_window is not None:
        raise UnsupportedArchitectureError(
            f"GGUF declares {hparams['architecture']}.attention.sliding_window="
            f"{sliding_window!r} -- this tool only supports plain global "
            "attention (no sliding window); refusing to guess at the "
            "per-layer attention pattern such a model would need."
        )


def reconstruct_config(hparams: dict, vocab_size: int, tie_word_embeddings: bool) -> dict:
    """Build a from-scratch ``config.json`` dict purely from GGUF-derived values.

    Used only when no ``--sidecar-dir`` is given, AND as the reference
    ``expected`` dict ``cross_validate_sidecar_config`` compares a sidecar
    against. ``torch_dtype`` is ``"float32"`` -- a statement of fact about
    THIS module's output weights (there is no GGUF metadata field for it).
    ``tie_word_embeddings`` is derived from whether an ``OUTPUT`` tensor
    was present in the census (GGUF carries no explicit tie flag;
    llama.cpp's own converter omits the separate ``output`` tensor when
    the source model tied embeddings) -- see ``convert()``. ``hidden_act``
    is always ``"silu"`` (the qwen3 architecture implies SwiGLU
    unconditionally -- no GGUF KV encodes this, there is nothing to read).
    ``attention_bias`` reflects whether this GGUF's OWN tensor directory
    actually contains any ``.bias`` tensor (``hparams["has_bias_tensors"]``
    -- never hardcoded False, so a future GGUF that DID carry bias tensors
    would be reconstructed honestly). ``rope_scaling``/``sliding_window``/
    ``use_sliding_window`` come straight from ``hparams`` (``assert_no_
    sliding_window`` has already hard-failed the whole conversion by the
    time this runs if sliding window was ever enabled, so these are always
    ``None``/``False`` in practice, but they are still read, not assumed).
    """
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": hparams["hidden_size"],
        "intermediate_size": hparams["intermediate_size"],
        "num_hidden_layers": hparams["block_count"],
        "num_attention_heads": hparams["num_attention_heads"],
        "num_key_value_heads": hparams["num_key_value_heads"],
        "head_dim": hparams["head_dim"],
        "rms_norm_eps": hparams["rms_norm_eps"],
        "rope_theta": hparams["rope_theta"],
        "rope_scaling": hparams.get("rope_scaling"),
        "vocab_size": vocab_size,
        "max_position_embeddings": hparams["max_position_embeddings"],
        "tie_word_embeddings": tie_word_embeddings,
        "bos_token_id": hparams.get("bos_token_id"),
        "eos_token_id": hparams.get("eos_token_id"),
        "torch_dtype": "float32",
        "hidden_act": "silu",
        "attention_bias": bool(hparams.get("has_bias_tensors", False)),
        "use_sliding_window": bool(hparams.get("sliding_window") is not None),
        "sliding_window": hparams.get("sliding_window"),
    }


# Fields cross-validated between a --sidecar-dir config.json and GGUF-derived
# metadata. max_position_embeddings / torch_dtype are the two known,
# deliberately exempt dual-pins (requirement 7) -- see module docstring and
# verify_base_identity.py's own context-length dual pin for precedent.
_SIDECAR_EXEMPT_FIELDS = ("max_position_embeddings", "torch_dtype")

# Fields where a GGUF-side None means "this GGUF simply doesn't carry this
# key" rather than "disagreement" -- comparison is SKIPPED (not hard-failed)
# when the GGUF side is None, and the skip is noted (minor 8). Every other
# field with a GGUF-side None is a genuine problem (in practice this never
# happens for the weight-shape fields, which read_qwen3_hparams already
# hard-fails on if absent).
_SIDECAR_OPTIONAL_ON_GGUF_SIDE = ("bos_token_id", "eos_token_id")

# Fields where an ABSENT sidecar key (hf_value is None) is treated as if the
# sidecar had explicitly written this default -- requirement 3(b): "must be
# false/absent" for attention_bias/use_sliding_window/sliding_window, and
# hidden_act defaults to "silu" implicitly in real Qwen3Config. Only an
# EXPLICIT sidecar value that disagrees with the GGUF-derived value hard-fails;
# omitting the key entirely is fine, not a mismatch.
_SIDECAR_DEFAULT_WHEN_ABSENT = {
    "hidden_act": "silu",
    "attention_bias": False,
    "use_sliding_window": False,
    "sliding_window": None,
}


def _field_values_match(field: str, gguf_value, hf_value) -> bool:
    """Plain equality for every field EXCEPT the two floats, which need a
    tolerance comparison. Deliberately does NOT treat None as an automatic
    mismatch: ``rope_scaling``/``sliding_window`` are legitimately ``None``
    on both sides in the common (no scaling, no sliding window) case, and
    ``None == None`` must be treated as agreement, not failure.
    """
    if field == "rms_norm_eps":
        return gguf_value is not None and hf_value is not None and math.isclose(
            gguf_value, hf_value, abs_tol=1e-9
        )
    if field == "rope_theta":
        return gguf_value is not None and hf_value is not None and math.isclose(
            gguf_value, hf_value, rel_tol=1e-6
        )
    return gguf_value == hf_value


def unvalidated_sidecar_keys(sidecar_config: dict, expected: dict) -> list:
    """Sidecar config.json keys this module does not cross-validate at all.

    Requirement 3(c): every such key is listed explicitly (never silently
    ignored without a trace) -- the caller folds this into ``sidecars.
    notes``.
    """
    return sorted(set(sidecar_config) - set(expected))


def cross_validate_sidecar_config(
    sidecar_config: dict, hparams: dict, vocab_size: int, tie_word_embeddings: bool
) -> list:
    """Hard-fail unless ``sidecar_config`` agrees with GGUF-derived metadata
    on every field this module knows how to check (requirement 3).

    Covers every key ``reconstruct_config`` produces (weight-shape fields,
    ``rope_scaling``, ``hidden_act``, ``attention_bias``, ``sliding_
    window``/``use_sliding_window``) except the two deliberately exempt
    dual-pins (``max_position_embeddings``/``torch_dtype`` -- never
    compared here; the caller still records both values in the manifest's
    ``sidecars.notes``). ``head_dim`` gets the same HF-side fallback
    ``verify_base_identity.load_hf_reference``'s comparator uses
    (``hidden_size // num_attention_heads``) when the sidecar config omits
    it outright AND both operands are available -- some HF configs omit
    head_dim; a sidecar missing BOTH head_dim and hidden_size falls through
    to the ordinary mismatch report instead of raising ``TypeError`` on a
    ``None // int``. ``hidden_act``/``attention_bias``/``use_sliding_
    window``/``sliding_window`` treat an ABSENT sidecar key as if it had
    been written with the GGUF-derived default (``_SIDECAR_DEFAULT_WHEN_
    ABSENT``) -- only an EXPLICIT, disagreeing value hard-fails; omitting
    the key is fine (requirement 3(b): "must be false/absent").
    ``bos_token_id``/``eos_token_id`` are SKIPPED (not hard-failed) when
    the GGUF side is ``None`` -- an absent GGUF key is not a disagreement;
    the skip is returned as a note string, not silently dropped.

    Returns the list of "skipped, GGUF-side is None" note strings (empty
    if none). Raises ``SidecarMismatchError`` naming every genuine
    mismatch found (collects all of them before raising, same "see the
    whole problem list" idiom as ``loratrain.config.validate_config``).
    """
    expected = reconstruct_config(hparams, vocab_size, tie_word_embeddings)
    problems = []
    skip_notes = []
    for field, gguf_value in expected.items():
        if field in _SIDECAR_EXEMPT_FIELDS:
            continue
        hf_value = sidecar_config.get(field)

        if field == "head_dim" and hf_value is None:
            heads = sidecar_config.get("num_attention_heads")
            hidden = sidecar_config.get("hidden_size")
            if heads and hidden:
                hf_value = hidden // heads

        if hf_value is None and field in _SIDECAR_DEFAULT_WHEN_ABSENT:
            hf_value = _SIDECAR_DEFAULT_WHEN_ABSENT[field]

        if field in _SIDECAR_OPTIONAL_ON_GGUF_SIDE and gguf_value is None:
            skip_notes.append(
                f"{field}: GGUF has no such key; sidecar value {hf_value!r} was "
                "not cross-validated against it"
            )
            continue

        if not _field_values_match(field, gguf_value, hf_value):
            problems.append(f"{field}: gguf-derived={gguf_value!r} vs sidecar={hf_value!r}")

    if problems:
        raise SidecarMismatchError(
            "--sidecar-dir config.json disagrees with GGUF-derived metadata on "
            "field(s) that must match for the base to be used safely "
            f"(max_position_embeddings/torch_dtype are the ONLY exempt "
            f"dual-pins): {'; '.join(problems)}"
        )
    return skip_notes


def read_gguf_token_strings(reader) -> list | None:
    """Full ordered token-string list from ``tokenizer.ggml.tokens``, or
    ``None`` if the GGUF has no such field.

    This is GGUF KV/header data, not tensor payload: ``GGUFReader.
    __init__`` already parses every KV field's raw parts into memory
    regardless of whether this is ever called; ``.contents()`` only does
    the (bounded, vocab-size-proportional) byte->str decoding on demand.
    Reading it never touches the multi-GB tensor data section.
    """
    field = reader.get_field("tokenizer.ggml.tokens")
    if field is None:
        return None
    return field.contents()


def validate_tokenizer_sidecar(reader, hparams: dict, vocab_size: int, sidecar_dir) -> None:
    """Hard-fail if a ``--sidecar-dir`` tokenizer disagrees with the GGUF's
    own embedded tokenizer (requirement 4).

    (a) ``len(tokenizer.ggml.tokens)`` must equal ``vocab_size`` --
        llama.cpp pads the token list to the embedding row count (the same
        invariant ``verify_base_identity.py``'s "vocab_n[rows==config.
        vocab_size]" check establishes); a mismatch means this GGUF
        violates that assumption, or ``vocab_size`` was derived wrong --
        either way, refuse to trust the sidecar tokenizer against it.
    (b) the sidecar ``tokenizer.json``'s own string for the GGUF's
        ``bos_token_id``/``eos_token_id`` (when present) must equal the
        GGUF's own string for that id. A spot-check, not full-vocabulary
        equality, but enough to catch a swapped/mismatched tokenizer file
        -- exactly the silent-tokenizer-swap vector this tool exists to
        close.

    No-op (returns without checking (b)) if the sidecar has no
    ``tokenizer.json`` at all -- its absence is fine and noted elsewhere
    (requirement 7); there is nothing to cross-check in that case.
    """
    gguf_tokens = read_gguf_token_strings(reader)
    if gguf_tokens is None:
        raise MissingMetadataError(
            "GGUF has no tokenizer.ggml.tokens field -- cannot cross-validate "
            "the --sidecar-dir tokenizer against it."
        )
    if len(gguf_tokens) != vocab_size:
        raise SidecarMismatchError(
            f"tokenizer.ggml.tokens has {len(gguf_tokens)} entries but "
            f"vocab_size (token_embd row count) is {vocab_size} -- GGUF "
            "internal inconsistency, refusing to trust either the "
            "reconstructed vocab_size or the sidecar's tokenizer against it."
        )

    tokenizer_json_path = Path(sidecar_dir) / "tokenizer.json"
    if not tokenizer_json_path.exists():
        return

    tokenizer_data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    vocab = tokenizer_data.get("model", {}).get("vocab", {}) or {}
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    for added in tokenizer_data.get("added_tokens", []) or []:
        id_to_token[added["id"]] = added["content"]

    for field_name in ("bos_token_id", "eos_token_id"):
        token_id = hparams.get(field_name)
        if token_id is None:
            continue
        if not (0 <= token_id < len(gguf_tokens)):
            raise SidecarMismatchError(
                f"{field_name}={token_id} is out of range for the GGUF's "
                f"{len(gguf_tokens)}-entry tokenizer.ggml.tokens"
            )
        gguf_str = gguf_tokens[token_id]
        sidecar_str = id_to_token.get(token_id)
        if sidecar_str != gguf_str:
            raise SidecarMismatchError(
                f"{field_name}={token_id}: GGUF token string {gguf_str!r} != "
                f"sidecar tokenizer.json string {sidecar_str!r} -- possible "
                "tokenizer swap/mismatch."
            )


# ============================================================================
# Safetensors writer (requirement 6) -- no torch/safetensors dependency.
# ============================================================================


def build_safetensors_header(tensor_specs: dict, metadata: dict | None = None) -> tuple:
    """Build one shard's safetensors header bytes + ordered ``(name, array)`` list.

    Tensor names are written in SORTED order (determinism) with
    contiguous, non-overlapping ``data_offsets``; every array is coerced
    to little-endian contiguous float32 before its size is measured, so
    the header's offsets always match the bytes ``write_safetensors_file``
    actually emits. Header JSON uses ``sort_keys=True`` + compact
    separators for byte-for-byte determinism across repeated runs on the
    same input.
    """
    names = sorted(tensor_specs)
    header: dict = {}
    offset = 0
    ordered = []
    for name in names:
        arr = np.ascontiguousarray(tensor_specs[name], dtype="<f4")
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        ordered.append((name, arr))
    if metadata:
        header["__metadata__"] = dict(metadata)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return header_bytes, ordered


def write_safetensors_file(path, tensor_specs: dict, metadata: dict | None = None) -> tuple:
    """Write one safetensors shard to ``path``.

    Returns ``(file_sha256, {name: tensor_sha256})`` -- the per-tensor
    sha256 map is computed HERE, from the exact same ``raw`` bytes buffer
    that gets written to disk (requirement 2: one ``arr.tobytes()`` call
    per tensor, not two -- callers must NOT separately hash the source
    array again).

    On-disk format: 8-byte little-endian header length, that many bytes of
    UTF-8 JSON header, then every tensor's raw little-endian bytes
    back-to-back in header (sorted-name) order -- the real safetensors
    spec, reproduced here without the ``safetensors`` package (module
    docstring). Hashes incrementally while writing rather than reading the
    file back afterward.
    """
    header_bytes, ordered = build_safetensors_header(tensor_specs, metadata=metadata)
    file_hash = hashlib.sha256()
    per_tensor_sha = {}
    with Path(path).open("wb") as fh:
        size_prefix = struct.pack("<Q", len(header_bytes))
        fh.write(size_prefix)
        file_hash.update(size_prefix)
        fh.write(header_bytes)
        file_hash.update(header_bytes)
        for name, arr in ordered:
            raw = arr.tobytes(order="C")  # the ONE copy, shared by file hash + per-tensor hash
            fh.write(raw)
            file_hash.update(raw)
            per_tensor_sha[name] = hashlib.sha256(raw).hexdigest()
    return file_hash.hexdigest(), per_tensor_sha


def read_safetensors_header(path) -> tuple:
    """Byte-level safetensors parser (spec-level, no ``safetensors`` package).

    Returns ``(header_dict, data_start_offset)``. Used by tests to prove
    ``write_safetensors_file``'s output is spec-conformant independent of
    whether the real ``safetensors`` package is importable.
    """
    path = Path(path)
    with path.open("rb") as fh:
        size_bytes = fh.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"{path}: truncated safetensors header-length prefix")
        (header_len,) = struct.unpack("<Q", size_bytes)
        header_bytes = fh.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
        header = json.loads(header_bytes.decode("utf-8"))
        data_start = 8 + header_len
    return header, data_start


def shard_tensor_names(names_sorted: list, sizes: dict, shard_max_bytes: int) -> list:
    """Greedy contiguous split of SORTED tensor names into shards.

    A single tensor larger than ``shard_max_bytes`` still gets its own
    (over-limit) shard rather than being split -- safetensors tensors are
    atomic. Contiguous over the given (sorted) name order, so the split is
    a pure function of ``(names_sorted, sizes, shard_max_bytes)``, never of
    dict/set iteration order.
    """
    shards = []
    current: list = []
    current_bytes = 0
    for name in names_sorted:
        size = sizes[name]
        if current and current_bytes + size > shard_max_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    shards.append(current)
    return shards


# ============================================================================
# --plan (requirement 9: metadata-only census + mapping proof)
# ============================================================================


def plan(gguf_path, gguf_py_dir=None) -> dict:
    """Read-only census + full GGUF->HF mapping proof; touches metadata only.

    Never accesses tensor payload bytes: only ``.name``/``.tensor_type``/
    ``.shape`` are read off each ``ReaderTensor`` (never ``.data``), and
    ``GGUFReader.__init__`` itself only parses the header + tensor-info
    directory (module docstring). Safe against the real multi-GB file.
    """
    gguf_mod, reader = open_reader(gguf_path, gguf_py_dir=gguf_py_dir)
    validate_hf_name_templates(gguf_mod)

    arch = read_scalar_kv(reader, "general.architecture")
    if arch != "qwen3":
        raise UnsupportedArchitectureError(
            f"expected general.architecture == 'qwen3', got {arch!r}"
        )
    block_count = read_scalar_kv(reader, f"{arch}.block_count")

    names = [t.name for t in reader.tensors]
    mapping = build_gguf_to_hf_name_map(gguf_mod, names)  # raises on unmapped/doubly-mapped

    tensors_proof = [
        {
            "gguf_name": t.name,
            "hf_name": mapping[t.name],
            "qtype": t.tensor_type.name,
            "shape": list(hf_shape(t)),
        }
        for t in reader.tensors
    ]

    return {
        "mode": "plan",
        "gguf_path": str(gguf_path),
        "arch": arch,
        "block_count": block_count,
        "tensor_census": census_tensors(reader),
        "mapping_proof": {
            "bijective": True,
            "unmapped": 0,
            "doubly_mapped": 0,
            "total_mapped": len(mapping),
        },
        "tensors": tensors_proof,
    }


# ============================================================================
# convert() -- the real dequantize-and-write path (requirements 6-8)
# ============================================================================


def _mkdir_parents_tracked(path: Path) -> list:
    """Create ``path`` and any missing ancestors; return the directories
    THIS call actually created (shallowest first).

    A plain ``path.mkdir(parents=True, exist_ok=True)`` cannot later
    distinguish "this run created these directories" from "these already
    existed" -- so a failure after this point had no way to clean up new,
    now-empty ancestor directories it left behind (requirement/minor 5).
    This walks up from ``path`` collecting missing ancestors, creates them
    shallowest-first, and returns the list so the caller's failure handler
    can remove them again (deepest first, only if still empty).
    """
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        p = p.parent
    created = []
    for d in reversed(missing):  # shallowest first
        d.mkdir(exist_ok=True)
        created.append(d)
    return created


def _cleanup_created_dirs(created_dirs: list) -> None:
    """Remove directories from ``_mkdir_parents_tracked``, deepest first,
    each only if it is still empty (never touches a directory that ended
    up holding something else in the meantime).
    """
    for d in reversed(created_dirs):
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass  # already gone, non-empty, or a race -- not our job to force it


def _plan_tensor_rows(gguf_mod, reader, hf_names: dict) -> list:
    """Pass 1 (requirement 2): every tensor's HF name/shape/output byte size
    from GGUF HEADER metadata only -- no dequantization.
    """
    rows = []
    for t in reader.tensors:
        shape = hf_shape(t)
        nbytes = int(np.prod(shape, dtype=np.int64)) * 4 if shape else 4  # float32 output
        rows.append(
            {
                "gguf_name": t.name,
                "hf_name": hf_names[t.name],
                "qtype": t.tensor_type.name,
                "shape": shape,
                "nbytes": nbytes,
            }
        )
    return rows


def convert(
    *,
    gguf_path,
    out_dir,
    sidecar_dir=None,
    expected_sha256=None,
    skip_source_hash=False,
    shard_max_bytes=DEFAULT_SHARD_MAX_BYTES,
    gguf_py_dir=None,
) -> dict:
    """Dequantize ``gguf_path`` into an HF-format directory at ``out_dir``.

    Returns the ``dequant_manifest.json`` dict (also written to
    ``out_dir/dequant_manifest.json``). See the module docstring's "Atomic
    publish" / "Memory" / "Sidecar validation" / "Tokenizer sidecar
    validation" sections for the guarantees this function makes, and
    "Execution scope" -- this session never calls this against the real
    pinned GGUF, only tiny synthetic fixtures.

    Ordering (requirement 1, house atomic-publish idiom): every gate below
    -- mutually-exclusive-flags, non-empty destination, source hash,
    architecture/mapping validation, sidecar config/tokenizer
    cross-validation -- runs BEFORE a single byte of dequantized tensor
    data is produced or any file is written under ``out_dir``. Everything
    that DOES get written lands in a temp directory that is a sibling of
    ``out_dir`` (same parent -- guarantees the final publish is a
    same-filesystem, atomic ``os.replace``), and is published to
    ``out_dir`` only once every step above has succeeded. Any exception
    anywhere in this function leaves ``out_dir`` exactly as it was found
    (absent, or unchanged if it already existed empty) and cleans up the
    temp directory.
    """
    if skip_source_hash and expected_sha256:
        raise ValueError(
            "--skip-source-hash and --expected-sha256 are mutually exclusive: "
            "cannot verify a hash that was never computed."
        )

    gguf_path = Path(gguf_path)
    out_dir = Path(out_dir)

    if out_dir.is_symlink():
        raise DestinationNotEmptyError(
            f"{out_dir} is a symlink -- refusing. A symlink to an empty "
            "directory would pass the ordinary empty-destination check and "
            "let the full conversion run, only to die at publish time with "
            "a confusing NotADirectoryError (os.replace refuses to target a "
            "symlink). Point --out at a real path, not a symlink."
        )

    if out_dir.exists():
        if not out_dir.is_dir():
            raise DestinationNotEmptyError(
                f"{out_dir} already exists and is not a directory -- refusing "
                "to replace it."
            )
        if any(out_dir.iterdir()):
            raise DestinationNotEmptyError(
                f"{out_dir} already exists and is not empty -- convert() never "
                "silently merges into an existing directory; remove or rename "
                "it first if replacing it was intended."
            )

    if expected_sha256 is not None:
        expected_sha256 = normalize_sha256(expected_sha256)

    # --- verify the source hash FIRST, before any dequant work -------------
    if skip_source_hash:
        source_sha = None
    else:
        source_sha = sha256_file(gguf_path)  # hashlib hexdigest() is always lowercase
        if expected_sha256 is not None and source_sha != expected_sha256:
            raise SourceHashMismatchError(
                f"{gguf_path}: sha256 {source_sha} != --expected-sha256 "
                f"{expected_sha256} -- refusing before any dequant work; "
                f"{out_dir} was never touched."
            )

    gguf_mod, reader = open_reader(gguf_path, gguf_py_dir=gguf_py_dir)
    validate_hf_name_templates(gguf_mod)

    hparams = read_qwen3_hparams(reader)
    assert_no_sliding_window(hparams)
    by_name = {t.name: t for t in reader.tensors}
    names = list(by_name)

    hf_names = build_gguf_to_hf_name_map(gguf_mod, names)
    tensor_census = census_tensors(reader)

    tie_word_embeddings = "output.weight" not in by_name

    token_embd = by_name.get("token_embd.weight")
    if token_embd is None:
        raise MissingMetadataError(
            "GGUF has no 'token_embd.weight' tensor -- cannot derive vocab_size"
        )
    vocab_size = hf_shape(token_embd)[0]

    # --- sidecar / config gates -- still metadata-only, no writes yet ------
    if sidecar_dir is not None:
        sidecar_dir = Path(sidecar_dir)
        sidecar_config_path = sidecar_dir / "config.json"
        if not sidecar_config_path.is_file():
            raise SidecarNotFoundError(
                f"--sidecar-dir {sidecar_dir} has no config.json (or does not "
                "exist) -- nothing to cross-validate or copy."
            )
        sidecar_config = json.loads(sidecar_config_path.read_text(encoding="utf-8"))
        skip_notes = cross_validate_sidecar_config(
            sidecar_config, hparams, vocab_size, tie_word_embeddings
        )
        validate_tokenizer_sidecar(reader, hparams, vocab_size, sidecar_dir)

        expected_fields = reconstruct_config(hparams, vocab_size, tie_word_embeddings)
        passthrough = unvalidated_sidecar_keys(sidecar_config, expected_fields)

        output_config = dict(sidecar_config)
        output_config["torch_dtype"] = "float32"
        sidecars_source = str(sidecar_dir)
        notes_parts = [
            "config.json copied from --sidecar-dir after cross-validation; "
            "torch_dtype overridden to 'float32' (this module's actual output "
            "dtype -- exempt from cross-validation, but the output must be "
            "honest about it). max_position_embeddings left as the sidecar's "
            f"value ({output_config.get('max_position_embeddings')!r}) vs this "
            f"GGUF's context_length ({hparams.get('max_position_embeddings')!r}) "
            "-- known dual-pin (see verify_base_identity.py precedent), not "
            "cross-validated, both values recorded here.",
            "tokenizer.json (when present) cross-checked against the GGUF's "
            "own embedded tokenizer: vocab_size and bos/eos token strings.",
        ]
        if skip_notes:
            notes_parts.append("Skipped (GGUF-side key absent): " + "; ".join(skip_notes))
        if passthrough:
            notes_parts.append(
                "Unvalidated pass-through key(s), copied as-is, never "
                f"cross-checked: {passthrough}"
            )
        sidecars_notes = " ".join(notes_parts)
    else:
        output_config = reconstruct_config(hparams, vocab_size, tie_word_embeddings)
        sidecars_source = "reconstructed_from_gguf"
        sidecars_notes = (
            "no --sidecar-dir given; config.json reconstructed from GGUF "
            "metadata only; no tokenizer files written (tokenizer files come "
            "only from --sidecar-dir -- their absence here is expected, not "
            "an error)."
        )

    # ------------------------------------------------------------------
    # Every gate above has passed. From here on: write into a TEMP SIBLING
    # of out_dir, publish atomically at the very end (requirement 1).
    # ------------------------------------------------------------------
    created_parent_dirs = _mkdir_parents_tracked(out_dir.parent)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        manifest = _write_converted_dir(
            tmp_dir=tmp_dir,
            gguf_mod=gguf_mod,
            reader=reader,
            hf_names=hf_names,
            shard_max_bytes=shard_max_bytes,
            output_config=output_config,
            sidecar_dir=sidecar_dir,
            sidecars_source=sidecars_source,
            sidecars_notes=sidecars_notes,
            gguf_path=gguf_path,
            source_sha=source_sha,
            expected_sha256=expected_sha256,
            tensor_census=tensor_census,
            gguf_py_dir=gguf_py_dir,
        )
        # os.replace atomically replaces an existing EMPTY directory (verified
        # behavior on this platform) or performs a plain rename if out_dir
        # does not exist yet -- either way, same-filesystem and atomic since
        # tmp_dir is a sibling of out_dir. Non-empty out_dir was already
        # refused above, before any of this work began.
        os.replace(tmp_dir, out_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # requirement/minor 5: a failure this late must not leave newly
        # created (and now empty) ancestor directories of --out behind --
        # only ones THIS run actually created, and only if still empty.
        _cleanup_created_dirs(created_parent_dirs)
        raise
    return manifest


def _write_converted_dir(
    *,
    tmp_dir: Path,
    gguf_mod,
    reader,
    hf_names: dict,
    shard_max_bytes: int,
    output_config: dict,
    sidecar_dir,
    sidecars_source: str,
    sidecars_notes: str,
    gguf_path: Path,
    source_sha,
    expected_sha256,
    tensor_census: dict,
    gguf_py_dir,
) -> dict:
    """Write every output file into ``tmp_dir`` (never ``out_dir`` directly)
    and return the assembled manifest dict.

    Two-pass streaming (requirement 2): ``plan_rows`` (pass 1) comes from
    header metadata only; the shard loop (pass 2) dequantizes, hashes, and
    writes ONE SHARD's tensors at a time, then drops that shard's dict
    before moving to the next -- peak RSS is bounded by roughly one
    shard's worth of float32 tensors, not the whole model.
    """
    plan_rows = _plan_tensor_rows(gguf_mod, reader, hf_names)
    by_hf_name = {row["hf_name"]: row for row in plan_rows}
    reader_tensor_by_gguf_name = {t.name: t for t in reader.tensors}

    names_sorted = sorted(by_hf_name)
    sizes = {name: by_hf_name[name]["nbytes"] for name in names_sorted}
    shards = shard_tensor_names(names_sorted, sizes, shard_max_bytes)
    n_shards = len(shards)

    weight_map = {}
    output_files = {}
    total_bytes = 0
    per_tensor_manifest = []
    per_tensor_sha = {}  # gguf_name -> sha256

    for i, shard_hf_names in enumerate(shards, start=1):
        filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {}
        for hf_name in shard_hf_names:
            row = by_hf_name[hf_name]
            t = reader_tensor_by_gguf_name[row["gguf_name"]]
            arr = dequantize_tensor(gguf_mod, t)
            arr = np.ascontiguousarray(arr, dtype="<f4")
            if tuple(arr.shape) != row["shape"]:
                raise ShapeMismatchError(
                    f"tensor {t.name!r}: dequantized shape {arr.shape} != "
                    f"header-declared HF-order shape {row['shape']}"
                )
            shard_tensors[hf_name] = arr

        shard_sha, tensor_shas = write_safetensors_file(
            tmp_dir / filename, shard_tensors, metadata=SAFETENSORS_METADATA
        )
        output_files[filename] = shard_sha
        for hf_name in shard_hf_names:
            row = by_hf_name[hf_name]
            sha = tensor_shas[hf_name]
            per_tensor_sha[row["gguf_name"]] = sha
            per_tensor_manifest.append(
                {
                    "gguf_name": row["gguf_name"],
                    "hf_name": hf_name,
                    "qtype": row["qtype"],
                    "shape": list(row["shape"]),
                    "sha256": sha,
                }
            )
            weight_map[hf_name] = filename
            total_bytes += row["nbytes"]
        del shard_tensors  # release this shard's tensors before the next one

    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    index_bytes = json.dumps(index, indent=2, sort_keys=True).encode("utf-8")
    (tmp_dir / "model.safetensors.index.json").write_bytes(index_bytes)
    output_files["model.safetensors.index.json"] = sha256_hex(index_bytes)

    config_bytes = json.dumps(output_config, indent=2, sort_keys=True).encode("utf-8")
    (tmp_dir / "config.json").write_bytes(config_bytes)
    sidecar_files = {"config.json": sha256_hex(config_bytes)}

    if sidecar_dir is not None:
        for fname in TOKENIZER_FILENAMES:
            src = Path(sidecar_dir) / fname
            if src.exists():
                data = src.read_bytes()
                (tmp_dir / fname).write_bytes(data)
                sidecar_files[fname] = sha256_hex(data)

    content_digest = sha256_hex(
        "\n".join(f"{name}:{sha}" for name, sha in sorted(per_tensor_sha.items())).encode("utf-8")
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_scheme": BASE_SCHEME,
        "source_gguf": {
            "path": str(gguf_path),
            "sha256": source_sha,
            "size_bytes": gguf_path.stat().st_size,
        },
        "expected_gguf_sha256": expected_sha256,
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
            "llama_cpp_checkout": {
                "path": str(LLAMA_CPP_CHECKOUT_DIR),
                "commit": _git_rev_parse_head(LLAMA_CPP_CHECKOUT_DIR),
            },
            "gguf_py_version": _read_gguf_py_version(gguf_py_dir or GGUF_PY_DIR),
        },
        "tensor_census": tensor_census,
        "tensors": per_tensor_manifest,
        "output": {
            "dtype": "float32",
            "files": output_files,
            "total_bytes": total_bytes,
        },
        "sidecars": {
            "source": sidecars_source,
            "files": sidecar_files,
            "notes": sidecars_notes,
        },
        "permutation": {
            "applied": PERMUTATION_APPLIED,
            "evidence": PERMUTATION_EVIDENCE,
        },
        "determinism": {
            "content_digest": content_digest,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    (tmp_dir / "dequant_manifest.json").write_bytes(manifest_bytes)
    return manifest


# ============================================================================
# CLI
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loratrain.gguf_to_hf",
        description=(
            "Dequantize a Qwen3 Q4_K_M GGUF into an HF-format directory "
            "(float32 weights) so LoRA training can run against the ACTUAL "
            "deployment weights llama.cpp serves, not the upstream fp16 "
            "revision."
        ),
    )
    p.add_argument("--gguf", required=True, type=Path, help="Path to the source .gguf file.")
    p.add_argument(
        "--out", type=Path, default=None, help="Output HF-format directory (required unless --plan)."
    )
    p.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding a pinned config.json + tokenizer files to copy "
            "and cross-validate instead of reconstructing config.json from "
            "GGUF metadata alone."
        ),
    )
    p.add_argument(
        "--expected-sha256",
        type=normalize_sha256,
        default=None,
        help=(
            "Hard-fail unless the source GGUF's sha256 matches this (64 hex "
            "chars; case-insensitive -- validated and lowercased at parse time)."
        ),
    )
    p.add_argument(
        "--shard-max-bytes",
        type=int,
        default=None,
        help=(
            "Max bytes per safetensors shard "
            f"(default when omitted: {DEFAULT_SHARD_MAX_BYTES} = 4 GiB)."
        ),
    )
    p.add_argument(
        "--gguf-py-dir",
        type=Path,
        default=None,
        help=f"Override the pinned gguf-py checkout (default: {GGUF_PY_DIR}).",
    )
    p.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Dry run: print tensor census + full GGUF->HF mapping proof. "
            "Touches metadata only -- writes nothing, never dequantizes."
        ),
    )
    p.add_argument(
        "--skip-source-hash",
        action="store_true",
        help=(
            "Skip hashing the (possibly multi-GB) source GGUF file; "
            "source_gguf.sha256 is recorded as null."
        ),
    )
    return p


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.plan:
        # --plan is a metadata-only dry run; any flag that only affects the
        # real write path has no effect here and is refused rather than
        # silently ignored (minor 11).
        no_effect = []
        if args.out is not None:
            no_effect.append("--out")
        if args.sidecar_dir is not None:
            no_effect.append("--sidecar-dir")
        if args.expected_sha256 is not None:
            no_effect.append("--expected-sha256")
        if args.skip_source_hash:
            no_effect.append("--skip-source-hash")
        # None-default sentinel (minor 4): --shard-max-bytes explicitly set
        # to the same value as the default is still an explicit write-path
        # flag and must be refused, not silently accepted because it
        # happens to equal DEFAULT_SHARD_MAX_BYTES.
        if args.shard_max_bytes is not None:
            no_effect.append("--shard-max-bytes")
        if no_effect:
            parser.error(
                "--plan has no effect together with: "
                + ", ".join(no_effect)
                + " -- remove --plan or those flags"
            )
        result = plan(args.gguf, gguf_py_dir=args.gguf_py_dir)
        print(json.dumps(result, indent=2))
        return 0

    if args.out is None:
        parser.error("--out is required unless --plan is given")

    # minor 3: route the mutually-exclusive-flags check through parser.error
    # (clean argparse usage message, exit 2) instead of letting convert()'s
    # own ValueError surface as a raw traceback -- convert() still raises it
    # too, for callers that skip this CLI layer entirely.
    if args.skip_source_hash and args.expected_sha256:
        parser.error("--skip-source-hash and --expected-sha256 are mutually exclusive")

    manifest = convert(
        gguf_path=args.gguf,
        out_dir=args.out,
        sidecar_dir=args.sidecar_dir,
        expected_sha256=args.expected_sha256,
        skip_source_hash=args.skip_source_hash,
        shard_max_bytes=args.shard_max_bytes if args.shard_max_bytes is not None else DEFAULT_SHARD_MAX_BYTES,
        gguf_py_dir=args.gguf_py_dir,
    )
    summary = {
        "out_dir": str(args.out),
        "tensor_census": manifest["tensor_census"],
        "output": manifest["output"],
        "content_digest": manifest["determinism"]["content_digest"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

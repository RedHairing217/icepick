"""BLOCKING post-campaign gate: dequantized-HF vs served-GGUF token parity.

A parallel tool (``gguf_to_hf.py``, a different worktree) dequantizes the
deployment GGUF into an fp32 HF directory plus ``dequant_manifest.json``
(contract keys: ``schema_version``, ``base_scheme`` ("dequant_q4km"),
``source_gguf`` ({"path", "sha256", "size_bytes"}), ``expected_gguf_sha256``,
``tool``, ``tensor_census``, ``tensors``, ``output``, ``sidecars``,
``permutation``, ``determinism`` ({"content_digest"}), ``created_utc``). This
module proves that dequantized directory reproduces llama.cpp's served BASE
behaviour: greedy decoding compared TOKEN-FOR-TOKEN (in practice, exact text
equality of the decoded completion) on 10 anchor prompts, hard-failing on ANY
mismatch (any classification other than ``"match"`` -- ``compare_texts``'
``classification`` field is where truncation-vs-content-divergence is
distinguished, NOT the exit code). All 10 matching is the only PASS.

POST-CAMPAIGN GATE -- this module must never run while any local eval owns
the machine-wide Qwen slot (max ONE concurrent user, by house invariant).
Before a single HTTP request is made -- and again before EVERY subsequent
server call, since HF generation between calls can run long enough for a
campaign to start meanwhile -- ``campaign_liveness_hits`` checks the two
campaign process patterns AND an ESTABLISHED-connection probe on the
server's own port; a hit REFUSES loudly unless ``--i-own-the-qwen-slot`` is
passed (the operator's explicit assertion that they have personally
verified the match is stale, not a live run).

Run this gate against a server started with the RUNBOOK ``§0.4-EXECUTED``
parity serve command (``base_serve_command`` builds the exact argv)::

    llama-server -m <pinned base gguf> --alias qwen3-8b-q4km-base \\
        -c 8192 -ngl 99 --parallel 1 --port <port>

That command passes NO ``--reasoning-format`` override, so llama.cpp's
compiled-in default (extraction ON) applies for real -- see "Reasoning-
channel handling" below.

``--mode raw|chat|both`` (default ``raw``) -- RAW IS THE GATE:

  ``raw`` (default) -- THE BLOCKING VERDICT CHANNEL. This tool renders the
  wire itself (``render_raw_prompt``, via the HF tokenizer's own
  ``apply_chat_template(..., tokenize=False)``) and sends the IDENTICAL
  already-rendered string to both engines: llama-server's native
  ``/completion`` route (never ``/v1/chat/completions``) and the HF model's
  ``generate`` over the tokenizer's own encoding of that same string.
  ``/completion``'s ``content`` field is the RAW, UNPARSED generation (see
  receipts below) -- this is the ONLY channel where byte-exact "token-for-
  token" comparison is actually achievable; see "Why chat mode cannot be
  the gate" below for why ``/v1/chat/completions`` cannot make this
  guarantee.

  ``chat`` -- an OPTIONAL, INFORMATIONAL cross-check, NEVER the gate: a
  ``--mode chat`` run's exit code is 0 regardless of what the cross-check
  finds (barring an infra error, still exit 2) -- see "Exit codes" below.
  It compares the server's chat-mode ``message.content`` against the HF
  side's raw decode, CANONICALIZED by stripping a leading think block
  (``strip_leading_think_block``) so the comparison is like-for-like
  wherever that stripping is faithful to what the server itself did; a
  residual mismatch confined to leading whitespace at that boundary is
  classified ``"channel_split_artifact"`` (module docstring "Why chat mode
  cannot be the gate") rather than ``"content_divergence"``.

  ``both`` -- runs raw (still the sole source of the exit code) AND the
  chat cross-check (attached to the report for a human to eyeball), one
  HTTP+HF round trip per channel per prompt.

Why chat mode cannot be the gate (llama.cpp @ b10107 receipts,
``/Users/redhairing/src/llama.cpp``, read-only verification 2026-07-30 and
2026-07-31 -- the second pass corrected an error from the first, see below):

  - ``common/arg.cpp:3486-3496`` -- ``--reasoning-format FORMAT`` CLI option,
    default ``"auto"``; ``none`` leaves thoughts inline in ``message.content``,
    ``deepseek``/``deepseek-legacy`` extract them into ``message.reasoning_content``.
  - ``common/common.h:643`` -- the compiled-in SERVER DEFAULT (before any CLI
    override) is ``reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK`` --
    extraction is ON by default, not merely available via a flag.
  - ``common/chat.cpp`` -- every per-chat-format handler gates on
    ``reasoning_format != COMMON_REASONING_FORMAT_NONE`` at all call sites
    (e.g. lines 1022, 1174, 1322, 1558, 1711, 1955, 2183, 2515).
  - ``common/chat-auto-parser-generator.cpp:168,187`` -- the GENERIC
    autoparser's reasoning combinator:
    ``optional(optspace(start) + reasoning(until(trim_ws(end))) + optspace(end))``
    followed by ``content(rest)``. ``optspace(tag)`` (``chat-peg-parser.cpp:
    843-867``) makes EACH leading/trailing whitespace character of the tag
    individually OPTIONAL and, critically, CONSUMED -- never captured into
    either ``reasoning`` or ``content``. For Qwen3's markers (``chat-diff-
    analyzer.cpp`` ``compare_reasoning_presence``: start ``"<think>\\n"``,
    end ``"\\n</think>\\n\\n"``), this means the newline immediately after
    ``<think>`` and the newlines immediately after ``</think>`` are
    STRUCTURALLY DISCARDED -- they exist in neither channel, so no
    reconstruction of the two channels can recover them byte-for-byte.
  - ``chat-peg-parser.cpp:262-275`` -- a whitespace-only ``reasoning_content``
    is DISCARDED entirely after parsing ("Discard whitespace-only reasoning
    content"), and ``to_json_oaicompat`` (``common/chat.cpp:186-233``) then
    OMITS the ``reasoning_content`` key altogether (present only when
    non-empty). Net effect, empirically reproduced
    (``scratchpad/verify-t2/repro_reconstruction.py``, hand-traced against
    the cited source): a Qwen3 completion under ``/no_think`` -- which (per
    this shop's masking-proof finding) STILL emits an EMPTY
    ``'<think>\\n\\n</think>\\n\\n'`` prefix -- parses to ``content`` ALONE
    (``reasoning_content`` key absent, its whitespace-only capture
    discarded), while the HF side's raw decode retains the full
    ``<think>...</think>`` block verbatim (``skip_special_tokens=True``
    does not strip it -- Qwen3's ``<think>``/``</think>`` are added tokens
    with ``special=false``). A byte-for-byte RECONSTRUCTION
    (``content`` + ``reasoning_content`` stitched back together) is
    therefore structurally impossible to make byte-identical to the raw
    generation in general (the boundary whitespace is gone from both
    channels), and for the empty-think case specifically it is impossible
    even to KNOW a think block was ever emitted (both channels look
    identical to the "no think block at all" case). ``/completion``'s
    ``content`` (``tools/server/server-task.cpp:364-385``) has NONE of this
    -- it is the model's raw, unparsed output -- which is why raw mode, not
    chat mode, is the only channel this gate can treat as byte-exact.
  - CORRECTION (this investigation, 2026-07-31): an earlier pass of this
    module cited ``common/chat.cpp:2470-2478`` (``trim_all_content``) as
    generic output-side trimming applied to every model's parsed response.
    That citation was WRONG -- ``trim_all_content`` is a StepFun-template-
    specific INPUT-side workaround (trims conversation-history messages
    before rendering the prompt, gated on a template-source substring
    match), not applied to Qwen3 and not applied to parsed output at all.
    It has been removed from this docstring; the ``optspace``/whitespace-
    discard mechanism above is the corrected, actually-verified root cause.

Exit codes: ``0`` == PASS -- no ``raw``-channel mismatch found (a ``--mode
chat``-only run is ALWAYS exit 0 barring an infra error: it is not the
gate, see above). ``1`` == weight-divergence FAIL -- at least one ``raw``-
channel prompt was NOT an exact match (ANY classification other than
``"match"``, including ``"prefix_truncation"`` -- the report's
``classification`` field distinguishes truncation from genuine content
divergence; the exit code does not). Only reachable when ``--mode`` is
``raw`` or ``both``. ``2`` == everything else: a missing/malformed input, an
HTTP or HF-loading infra failure, a wrong-server (``--expected-alias``)
response, a null ``message.content``, an all-empty completion run, a
scheme-less ``--server-url``, or the campaign-liveness guard tripping. Only
a successfully completed comparison loop (in a mode that includes ``raw``)
can produce exit 1; any exception anywhere in the post-argparse body is
exit 2.

Standalone by design: this module does NOT import ``gguf_to_hf`` (it lives
in a different worktree right now) -- it reads ``dequant_manifest.json``
itself with plain ``json``. It DOES import ``loratrain.verify_base_identity``
(same package, already present at this worktree's HEAD) to cross-check the
dequantization's source GGUF sha256 against the one pin this whole repo
already trusts. ``transformers``/``torch`` are lazy-imported
(``_lazy_import_hf_stack``) with an actionable error when absent, so every
other function here (the guard, prompt extraction, manifest preconditions,
comparison logic, report writing) stays importable and testable without
either package installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from loratrain import config, verify_base_identity

# ============================================================================
# Campaign-liveness guard
# ============================================================================

# The two process signatures a live local eval campaign runs under (module
# docstring, orchestrator's live-campaign context). Checked BEFORE any HTTP
# call this module might make, and again before EVERY subsequent one.
CAMPAIGN_LIVENESS_PATTERNS = ("eval_all.sh", "icepick processing pass_at_k")


class CampaignLiveError(RuntimeError):
    """Refused: the Qwen slot appears owned by a live eval campaign.

    This machine allows at most ONE concurrent user of the local Qwen slot;
    running this gate while a campaign owns it would perturb the campaign's
    measurement. Raised by ``refuse_if_campaign_live``.
    """


class ManifestError(RuntimeError):
    """``--dequant-dir`` fails a sanity precondition (manifest/config/tokenizer/identity)."""


class AnchorExtractionError(RuntimeError):
    """``--from-eval-set`` extraction did not yield exactly the pinned anchor count."""


class ReportPathError(RuntimeError):
    """``--report`` was pointed at a refused location (any ``out/`` directory)."""


class DependencyError(RuntimeError):
    """``transformers``/``torch`` are not importable in this environment.

    The HF side of this gate (loading ``--dequant-dir``, running greedy
    ``generate``) needs both. Raised by ``_lazy_import_hf_stack``; every
    other function in this module stays usable on a machine with neither
    package installed.
    """


class TransportError(RuntimeError):
    """An HTTP call (chat or native completion) returned something this gate
    cannot score: a malformed response shape, a null ``message.content``
    (legal OpenAI shape, but unusable), or a wrong-server ``model``/alias.
    """


def _default_pgrep_runner(argv):
    return subprocess.run(argv, capture_output=True, text=True)


def check_campaign_not_live(pgrep_runner=None) -> list:
    """Return the ``CAMPAIGN_LIVENESS_PATTERNS`` that currently match a running process.

    Empty list means clear. ``pgrep_runner(argv) -> CompletedProcess-like``
    (an object with ``.returncode``/``.stdout``) is injectable for tests --
    same contract as ``icepick.batcher.stages.qwen_slot_free``'s
    ``pgrep_runner``; real usage defaults to
    ``subprocess.run(capture_output=True, text=True)``.

    A pattern counts as a hit when the runner returns ``returncode == 0``
    with non-empty ``stdout`` (pgrep's "found a match" contract);
    ``returncode == 1`` with empty stdout is genuinely clear for that
    pattern. ``returncode >= 2`` (pgrep's own syntax/fatal-error codes) is
    ALSO treated as a hit -- pgrep failing to even run its search must never
    be silently read as "the campaign is definitely not running". A runner
    that raises is likewise treated as a hit -- refuse-by-default in both
    cases (mirrors ``qwen_slot_free``'s "if pgrep itself fails,
    conservatively report busy").
    """
    runner = pgrep_runner if pgrep_runner is not None else _default_pgrep_runner
    hits = []
    for pattern in CAMPAIGN_LIVENESS_PATTERNS:
        try:
            result = runner(["pgrep", "-f", pattern])
        except Exception:
            hits.append(pattern)
            continue
        if result.returncode >= 2:
            hits.append(pattern)
        elif result.returncode == 0 and result.stdout.strip():
            hits.append(pattern)
    return hits


def _default_lsof_runner(argv):
    return subprocess.run(argv, capture_output=True, text=True)


def check_server_port_busy(port: int, lsof_runner=None) -> bool:
    """Return True iff there is an ESTABLISHED connection on ``port``.

    The port half of ``qwen_slot_free``'s two-part contract
    (``pgrep -f "icepick processing pass_at_k"`` + ``lsof -i TCP:1234
    -sTCP:ESTABLISHED -t``): busy iff the runner's stdout is non-empty,
    REGARDLESS of return code -- ``lsof`` on macOS can exit 1 even when it
    printed a matching PID (a per-process access error probing some OTHER,
    unrelated process trips the same non-zero exit as "found nothing"), so
    keying off ``returncode`` (as an earlier version of this function did)
    fails OPEN exactly when it matters. A runner that raises is treated
    conservatively as busy (same refuse-by-default posture as the pgrep
    half). ``lsof_runner(argv) -> CompletedProcess-like`` is injectable for
    tests.
    """
    runner = lsof_runner if lsof_runner is not None else _default_lsof_runner
    try:
        result = runner(["lsof", "-i", f"TCP:{port}", "-sTCP:ESTABLISHED", "-t"])
    except Exception:
        return True
    return bool(result.stdout.strip())


def campaign_liveness_hits(server_url=None, *, pgrep_runner=None, lsof_runner=None) -> list:
    """Combined guard: the two process patterns, plus a port-liveness probe.

    When ``server_url`` carries an explicit port, also checks
    ``check_server_port_busy`` on that port and folds a hit in as an extra
    entry in the returned list. ``server_url`` with no explicit port (or
    ``None``) skips the port check entirely -- there is nothing to probe
    (this is a deliberate, legitimate skip for e.g. a schemed https URL
    with no port number; ``main`` separately hard-refuses SCHEME-LESS
    URLs, whose ambiguous parsing is what would otherwise cause a silent,
    unintended skip here -- see ``_require_url_scheme``). Called once
    before the per-prompt loop starts and again immediately before EVERY
    server HTTP call inside it (module docstring): HF generation between
    calls can run long enough for a campaign to start meanwhile.
    """
    hits = check_campaign_not_live(pgrep_runner=pgrep_runner)
    if server_url:
        port = urllib.parse.urlsplit(server_url).port
        if port is not None and check_server_port_busy(port, lsof_runner=lsof_runner):
            hits = hits + [f"port {port} has an ESTABLISHED connection"]
    return hits


def refuse_if_campaign_live(hits, i_own_the_qwen_slot: bool) -> None:
    """Raise ``CampaignLiveError`` iff ``hits`` is non-empty and not overridden.

    ``--i-own-the-qwen-slot`` is the operator's explicit, individually-typed
    assertion that they have personally verified the matched process(es) are
    NOT an actual live campaign (e.g. a stale/zombie match) -- mirrors this
    codebase's other guarded-override flags (``upload_guard --execute``,
    the identity-preflight gate before upload): the unsafe path always
    requires a positional, hard-to-fat-finger opt-in, never a default.
    """
    if hits and not i_own_the_qwen_slot:
        raise CampaignLiveError(
            "REFUSING TO RUN: the Qwen slot appears owned by a live campaign "
            f"process (matched: {', '.join(hits)}). This machine allows at "
            "most ONE concurrent user of the local Qwen slot; running this "
            "gate now would perturb that campaign's measurement. This is a "
            "POST-CAMPAIGN gate (see module docstring) -- wait for the "
            "campaign to finish, or pass --i-own-the-qwen-slot ONLY if you "
            "have personally verified the matched process is not an actual "
            "live campaign."
        )


def _require_url_scheme(server_url: str) -> None:
    """Hard-refuse a ``--server-url`` with no ``http``/``https`` scheme.

    A scheme-less URL (e.g. ``"host:8081/v1/chat/completions"``) parses
    AMBIGUOUSLY under ``urllib.parse.urlsplit``: it reads ``"host"`` as the
    SCHEME (there being no ``"//"``), leaving ``netloc`` empty and
    ``.port`` as ``None`` -- which would silently skip
    ``campaign_liveness_hits``'s port-liveness probe instead of raising
    anything. Refusing up front turns that silent guard-weakening into a
    loud, immediate error.
    """
    scheme = urllib.parse.urlsplit(server_url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(
            f"--server-url {server_url!r} has no http/https scheme -- a "
            "scheme-less URL parses ambiguously (urlsplit reads the host as "
            "the scheme) and would silently skip the campaign-liveness "
            "guard's port-liveness probe. Pass an explicit URL with an "
            "http or https scheme."
        )


# ============================================================================
# Wire construction -- single place the chat wire is built, shared by every
# caller (server chat-mode payload, HF chat-mode render, --raw-completion
# render) so the two engines can never drift apart by construction.
# ============================================================================


def build_chat_messages(statement: str) -> list:
    """The byte-identical pass@k wire: system + user turn, config-pinned.

    Same pins ``build_dataset.build_sft_example`` trains against
    (``config.PASS_AT_K_SYSTEM_PROMPT`` / ``config.PASS_AT_K_NO_THINK_SUFFIX``
    -- README D4 byte-identity). Every message-building call in this module
    (``call_chat_completion``, ``generate_hf_chat``, ``render_raw_prompt``)
    goes through this one function.
    """
    return [
        {"role": "system", "content": config.PASS_AT_K_SYSTEM_PROMPT},
        {"role": "user", "content": statement + config.PASS_AT_K_NO_THINK_SUFFIX},
    ]


def render_raw_prompt(tokenizer, statement: str) -> str:
    """Render the chat wire into ONE raw string via the tokenizer's own template.

    ``--mode raw`` (THE gate, module docstring): sends this identical
    string to BOTH engines' raw-text endpoints so a divergence can only come
    from weights/sampling, never chat-template disagreement. ``tokenizer``
    only needs an ``apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True) -> str`` method -- tests inject a minimal
    fake satisfying that duck type, no real ``transformers`` needed.
    """
    return tokenizer.apply_chat_template(
        build_chat_messages(statement), tokenize=False, add_generation_prompt=True
    )


def reconstruct_completion_text(content, reasoning_content) -> str:
    """Best-effort, NON-gating reconstruction of a b10107 chat response's full text.

    See the module docstring's "Why chat mode cannot be the gate" section:
    byte-exact reconstruction through the chat wire is structurally
    impossible (the autoparser's ``optspace`` combinator discards boundary
    whitespace into NEITHER channel). This function is kept only as an
    informational convenience (``call_chat_completion``'s ``"text"`` field,
    surfaced in the report for a human to eyeball) -- ``evaluate_chat_cross_check``
    does NOT use it for the actual comparison, which instead canonicalizes
    the HF side and compares against ``content`` alone (``strip_leading_think_block``).

    When ``reasoning_content`` is present (non-empty -- llama.cpp omits the
    key entirely for an empty/whitespace-only think block,
    ``common/chat.cpp:186-233``), wraps it back into
    ``<think>{reasoning_content}</think>\\n\\n{content}``; otherwise returns
    ``content`` unchanged. ``content=None`` is treated as ``""`` (defensive
    only -- ``call_chat_completion`` already hard-fails on a null content
    before this could be reached in practice).
    """
    if not reasoning_content:
        return content
    return f"<think>{reasoning_content}</think>\n\n{content if content is not None else ''}"


_THINK_START = "<think>"
_THINK_END = "</think>"


def strip_leading_think_block(text: str) -> tuple:
    """Strip a leading ``<think>...</think>`` block from a raw HF decode.

    Returns ``(canonical_text, had_think_block)``. This is the chat cross-
    check's canonicalization step (module docstring "Why chat mode cannot
    be the gate"): mirrors the b10107 autoparser's own optional-whitespace
    handling around the tags closely enough to make ``evaluate_chat_cross_check``
    a MEANINGFUL cross-check rather than a guaranteed spurious mismatch --
    one optional leading newline is stripped after ``<think>``, and up to
    two optional trailing newlines after ``</think>`` (mirroring Qwen3's
    observed markers, ``start = "<think>\\n"``, ``end = "\\n</think>\\n\\n"``,
    per ``chat-diff-analyzer.cpp``'s ``compare_reasoning_presence``). This
    is NOT proven byte-identical to the server's own parse in every case
    (the exact whitespace consumed is model/template dependent) --
    ``evaluate_chat_cross_check`` classifies any residual PURELY-LEADING-
    WHITESPACE mismatch after this stripping as ``"channel_split_artifact"``
    rather than a genuine divergence.

    If ``text`` has no leading think block (``</think>`` never found is
    treated as "the whole visible text was reasoning" -- a length-capped
    generation that stopped mid-think, ``finish_reason == "length"``): when
    ``<think>`` opens but ``</think>`` never appears, returns ``("", True)``
    (nothing left to compare downstream, matching the server's own
    lenient-until-EOF content of ``""`` in that case). If ``text`` does not
    start with ``<think>`` at all, returns ``(text, False)`` unchanged.
    """
    if not text.startswith(_THINK_START):
        return text, False
    rest = text[len(_THINK_START):]
    if rest.startswith("\n"):
        rest = rest[1:]
    end_idx = rest.find(_THINK_END)
    if end_idx == -1:
        return "", True
    rest = rest[end_idx + len(_THINK_END):]
    for _ in range(2):
        if rest.startswith("\n"):
            rest = rest[1:]
    return rest, True


# ============================================================================
# Prompt sourcing: --prompts-file (plain jsonl) or --from-eval-set (the real
# evalharness/data/eval_set.jsonl schema, filtered to the anchor_solved slice)
# ============================================================================

# evalharness/data/eval_set.jsonl schema (verified read-only against the real
# file 2026-07-30: 120 rows total; every row carries uid/statement/answer/
# arxiv_id/family/tier/source/provenance/truth_policy/metadata, plus
# "eval_slice" with exactly three observed values -- eval_band: 100,
# anchor_solved: 10, anchor_fail: 10). This gate only ever wants the 10
# anchor_solved rows -- the fixed prompt set the RUNBOOK task pins the
# parity comparison to.
ANCHOR_EVAL_SLICE = "anchor_solved"
EXPECTED_ANCHOR_COUNT = 10


def extract_anchor_prompts(eval_set_path) -> list:
    """Extract the ``EXPECTED_ANCHOR_COUNT`` anchor prompts from an eval_set.jsonl.

    Filters rows by ``eval_slice == "anchor_solved"``, taking each row's
    ``uid``/``statement``. Hard-fails ``AnchorExtractionError`` unless
    EXACTLY ``EXPECTED_ANCHOR_COUNT`` rows match -- a wrong count means
    either a malformed/truncated eval set or a schema drift this tool has
    not been updated for; silently running against however many rows
    happened to match would quietly shrink (or inflate) the parity gate's
    pinned prompt set instead of refusing.
    """
    eval_set_path = Path(eval_set_path)
    prompts = []
    n_rows = 0
    with eval_set_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            row = json.loads(line)
            if row.get("eval_slice") != ANCHOR_EVAL_SLICE:
                continue
            uid = row.get("uid")
            statement = row.get("statement")
            if not uid or not statement:
                raise AnchorExtractionError(
                    f"{eval_set_path}: row {n_rows} has eval_slice="
                    f"{ANCHOR_EVAL_SLICE!r} but no non-empty uid/statement "
                    f"(uid={uid!r})"
                )
            prompts.append({"uid": uid, "statement": statement})

    if len(prompts) != EXPECTED_ANCHOR_COUNT:
        raise AnchorExtractionError(
            f"{eval_set_path}: found {len(prompts)} row(s) with eval_slice="
            f"{ANCHOR_EVAL_SLICE!r} across {n_rows} total row(s); expected "
            f"EXACTLY {EXPECTED_ANCHOR_COUNT} (the pinned anchor prompt set) "
            "-- refusing to run this gate against a wrong-sized or "
            "schema-drifted eval set."
        )
    return prompts


def load_prompts_file(path) -> list:
    """Load a plain prompts jsonl: one ``{"uid": ..., "statement": ...}`` object per line.

    Simpler contract than ``--from-eval-set``: no ``eval_slice`` filtering
    and no fixed-count requirement -- whatever the file contains is what
    gets checked. Blank lines are skipped. Raises ``ValueError`` on a row
    missing a non-empty ``uid``/``statement``, or if the file yields no
    usable prompts at all. ``FileNotFoundError`` propagates naturally from
    the failed open (same idiom as ``build_dataset.load_uid_list``).
    """
    path = Path(path)
    prompts = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("uid")
            statement = row.get("statement")
            if not uid or not statement:
                raise ValueError(
                    f"{path}:{lineno}: row missing non-empty 'uid'/'statement' "
                    f"(row={row!r})"
                )
            prompts.append({"uid": uid, "statement": statement})
    if not prompts:
        raise ValueError(f"{path}: no usable prompts found (empty or all-blank file)")
    return prompts


# ============================================================================
# --dequant-dir sanity preconditions (run BEFORE any HTTP call)
# ============================================================================

REQUIRED_BASE_SCHEME = "dequant_q4km"
TOKENIZER_FILENAMES = ("tokenizer.json", "tokenizer_config.json")
_GENERATION_CONFIG_SAMPLER_KEYS = ("temperature", "top_p", "top_k")


def _normalize_token_ids(value) -> set:
    """Normalize a token-id field (int, list/tuple of ints, or ``None``) to a set of ints.

    The canonical Qwen3 sidecar ships ``generation_config.json``'s
    ``eos_token_id`` as a LIST (e.g. ``[151645]``) while ``config.json``
    carries a single int -- both are legitimate shapes for "the set of ids
    that count as EOS", so every comparison/membership check in this module
    goes through this normalization rather than a direct ``!=`` on
    mismatched shapes (which would hard-refuse the canonical sidecar for no
    real reason).
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}
    return {int(value)}


def check_dequant_dir_preconditions(dequant_dir) -> tuple:
    """Hard-fail sanity checks against ``--dequant-dir``; returns ``(manifest, warnings, generation_config)``.

    Hard-fails ``ManifestError`` when:

    - ``dequant_manifest.json`` is missing or not valid JSON.
    - its ``base_scheme`` is not ``"dequant_q4km"`` (this gate only makes
      sense against a q4_k_m dequantization -- gguf_to_hf.py's own
      contract key).
    - ``config.json`` is missing, or present but not valid JSON.
    - IDENTITY BINDING fails: the manifest's ``source_gguf.sha256`` must
      equal the REPO-WIDE pinned base GGUF sha
      (``verify_base_identity.EXPECTED_BASE_GGUF_SHA256`` -- same package,
      the identity this whole repo already trusts, RUNBOOK D-R2), and, when
      the manifest's own ``expected_gguf_sha256`` is non-null, it must agree
      with BOTH. A dequantization built from the wrong GGUF (or an
      internally inconsistent manifest) must never silently be compared
      against the base server as if it matched.
    - no tokenizer files are found at all (``TOKENIZER_FILENAMES``). Every
      ``--mode`` exercises the HF side (raw mode tokenizes the rendered
      string directly; chat mode calls ``apply_chat_template``), so a
      missing tokenizer is fatal in ALL modes now -- there is no more
      "exploratory, tolerate it" mode to soften this into a warning (that
      distinction existed only while raw completion was an optional
      diagnostic; it is the default gate now).
    - a shipped ``generation_config.json``'s ``eos_token_id`` disagrees
      with ``config.json``'s (compared as normalized ID SETS,
      ``_normalize_token_ids`` -- an int vs. a single-element list is NOT a
      disagreement).

    A shipped ``generation_config.json`` is read when present (absence is
    fine, not a warning) and returned as the third element for the caller
    to record in ``parity_report.json``; if it carries any of
    ``temperature``/``top_p``/``top_k``, that is folded into the returned
    warnings list -- this gate always forces ``do_sample=False`` regardless,
    but a shipped non-greedy default is worth surfacing.
    """
    dequant_dir = Path(dequant_dir)
    manifest_path = dequant_dir / "dequant_manifest.json"
    if not manifest_path.exists():
        raise ManifestError(
            f"{dequant_dir}: no dequant_manifest.json found -- not a "
            "gguf_to_hf.py output directory (or it has not finished writing "
            "yet)."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{manifest_path}: invalid JSON ({exc})") from exc

    base_scheme = manifest.get("base_scheme")
    if base_scheme != REQUIRED_BASE_SCHEME:
        raise ManifestError(
            f"{manifest_path}: base_scheme={base_scheme!r}, expected "
            f"{REQUIRED_BASE_SCHEME!r} -- this gate only verifies a q4_k_m "
            "dequantization; refusing to run against a differently-scheme'd "
            "directory."
        )

    pinned_sha = verify_base_identity.EXPECTED_BASE_GGUF_SHA256
    source_gguf = manifest.get("source_gguf") or {}
    source_sha = source_gguf.get("sha256")
    expected_gguf_sha = manifest.get("expected_gguf_sha256")
    identity_problems = []
    if source_sha != pinned_sha:
        identity_problems.append(
            f"manifest source_gguf.sha256={source_sha!r} != the repo-wide "
            f"pinned base GGUF sha {pinned_sha!r} "
            "(verify_base_identity.EXPECTED_BASE_GGUF_SHA256) -- this "
            "dequantization was not built from the pinned base GGUF."
        )
    if expected_gguf_sha is not None:
        if expected_gguf_sha != pinned_sha:
            identity_problems.append(
                f"manifest expected_gguf_sha256={expected_gguf_sha!r} != the "
                f"repo-wide pinned base GGUF sha {pinned_sha!r}."
            )
        if source_sha != expected_gguf_sha:
            identity_problems.append(
                f"manifest source_gguf.sha256={source_sha!r} != manifest's "
                f"own expected_gguf_sha256={expected_gguf_sha!r} -- the "
                "manifest is internally inconsistent."
            )
    if identity_problems:
        raise ManifestError(
            f"{manifest_path}: identity binding failed: " + "; ".join(identity_problems)
        )

    config_json_path = dequant_dir / "config.json"
    if not config_json_path.exists():
        raise ManifestError(
            f"{dequant_dir}: no config.json found -- not a loadable HF "
            "model directory (AutoModelForCausalLM/AutoTokenizer both "
            "need it)."
        )

    warnings = []
    tokenizer_present = any((dequant_dir / name).exists() for name in TOKENIZER_FILENAMES)
    if not tokenizer_present:
        raise ManifestError(
            f"{dequant_dir}: no tokenizer files found "
            f"({' or '.join(TOKENIZER_FILENAMES)}) -- every --mode exercises "
            "the HF side, which needs a working tokenizer either way "
            "(apply_chat_template for chat mode, direct tokenization for "
            "raw mode); there is no mode where this can be deferred."
        )

    generation_config = None
    generation_config_path = dequant_dir / "generation_config.json"
    if generation_config_path.exists():
        try:
            generation_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{generation_config_path}: invalid JSON ({exc})") from exc

        sampler_keys = [k for k in _GENERATION_CONFIG_SAMPLER_KEYS if k in generation_config]
        if sampler_keys:
            warnings.append(
                f"{generation_config_path} carries sampler default(s) "
                f"{sampler_keys} -- this gate always forces do_sample=False "
                "(greedy) regardless, but a shipped non-greedy default here "
                "is worth knowing about."
            )

        try:
            config_data = json.loads(config_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{config_json_path}: invalid JSON ({exc})") from exc

        gen_eos_set = _normalize_token_ids(generation_config.get("eos_token_id"))
        cfg_eos_set = _normalize_token_ids(config_data.get("eos_token_id"))
        if gen_eos_set and cfg_eos_set and gen_eos_set != cfg_eos_set:
            raise ManifestError(
                f"{generation_config_path}: eos_token_id(s) {sorted(gen_eos_set)} "
                f"!= {config_json_path}'s eos_token_id(s) {sorted(cfg_eos_set)} "
                "-- inconsistent eos configuration."
            )

    return manifest, warnings, generation_config


# ============================================================================
# Comparison logic (the gate itself): exact text equality + divergence report
# ============================================================================

PROMPT_CONTEXT_CHARS = 60
TOKEN_CONTEXT_TOKENS = 8


def _first_diff_index(a: str, b: str) -> int:
    """Index of the first differing character (codepoint, not byte) between ``a``/``b``.

    If one string is a strict prefix of the other, the divergence index is
    the length of the shorter string. Operates on Python ``str``, so this is
    a character index -- unicode-safe by construction (multi-byte encodings
    never shift the reported position).
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _context_window(text: str, idx: int, radius: int = PROMPT_CONTEXT_CHARS) -> str:
    start = max(0, idx - radius)
    end = min(len(text), idx + radius)
    return text[start:end]


def compare_texts(server_text: str, hf_text: str) -> dict:
    """Exact-equality comparison of one prompt's two decoded completions.

    Returns ``{"match": bool, "first_divergence": None | {...}, "classification": str}``.
    ``classification`` is one of:

    - ``"match"`` -- byte-for-byte identical.
    - ``"prefix_truncation"`` -- one string is an exact prefix of the other
      (the divergence index equals the shorter string's length): every
      emitted token both engines agree on, one just stopped earlier. This
      is a stop/EOS-config or max-new-tokens-ceiling drift signature --
      still an exit-1 mismatch (module docstring "Exit codes"), just
      distinguishable in the report from a genuine content disagreement.
    - ``"content_divergence"`` -- a genuine mid-string disagreement: the
      weight-parity signature this gate exists to catch.

    (``evaluate_chat_cross_check`` may further reclassify a ``chat``-mode
    result to ``"channel_split_artifact"`` -- that relabeling happens
    there, never in this function, which always reports the raw
    match/mismatch shape as-is.)
    """
    if server_text == hf_text:
        return {"match": True, "first_divergence": None, "classification": "match"}
    idx = _first_diff_index(server_text, hf_text)
    classification = (
        "prefix_truncation" if idx == min(len(server_text), len(hf_text)) else "content_divergence"
    )
    return {
        "match": False,
        "first_divergence": {
            "char_index": idx,
            "server_context": _context_window(server_text, idx),
            "hf_context": _context_window(hf_text, idx),
        },
        "classification": classification,
    }


def token_divergence(tokenizer, server_text: str, hf_text: str) -> dict:
    """Best-effort TOKEN-level divergence diagnostic; ``None`` if unavailable.

    Supplements ``compare_texts``'s character-level report with tokenizer
    IDs around the first differing token -- useful when a byte-identical
    surface difference (e.g. a single re-tokenized digit) is otherwise hard
    to eyeball. Returns ``None`` if ``tokenizer`` is falsy or encoding
    either string raises (never lets a diagnostic-only helper crash the
    gate's primary text-equality verdict).
    """
    if not tokenizer:
        return None
    try:
        server_ids = tokenizer.encode(server_text, add_special_tokens=False)
        hf_ids = tokenizer.encode(hf_text, add_special_tokens=False)
    except Exception:
        return None
    n = min(len(server_ids), len(hf_ids))
    idx = n
    for i in range(n):
        if server_ids[i] != hf_ids[i]:
            idx = i
            break
    lo = max(0, idx - TOKEN_CONTEXT_TOKENS)
    hi_server = min(len(server_ids), idx + TOKEN_CONTEXT_TOKENS + 1)
    hi_hf = min(len(hf_ids), idx + TOKEN_CONTEXT_TOKENS + 1)
    return {
        "token_index": idx,
        "server_token_ids": server_ids[lo:hi_server],
        "hf_token_ids": hf_ids[lo:hi_hf],
    }


def _server_hit_ceiling(*, finish_reason=None, stop_type=None) -> bool:
    return finish_reason == "length" or stop_type == "limit"


def evaluate_raw_result(server_result: dict, hf_result: dict) -> dict:
    """Build the ``raw``-mode (THE gate) per-prompt comparison sub-report.

    ``server_result`` is ``call_native_completion``'s return; ``hf_result``
    is ``generate_hf_raw``'s. Byte-exact comparison of two RAW, unparsed
    channels -- no canonicalization needed (module docstring "Why chat mode
    cannot be the gate"). ``token_divergence``/``server_hit_ceiling`` are
    NOT computed here -- the caller (``main``) fills ``token_divergence``
    in only on mismatch (needs the tokenizer) and stamps ceiling flags
    already available on ``hf_result``.
    """
    server_text = server_result["text"]
    hf_text = hf_result["text"]
    comparison = compare_texts(server_text, hf_text)
    return {
        "match": comparison["match"],
        "classification": comparison["classification"],
        "first_divergence": comparison["first_divergence"],
        "server_text": server_text,
        "hf_text": hf_text,
        "token_divergence": None,
        "server_finish_reason": server_result.get("stop_type"),
        "server_model": server_result.get("model"),
        "server_hit_ceiling": _server_hit_ceiling(stop_type=server_result.get("stop_type")),
        "hf_hit_ceiling": hf_result["hit_ceiling"],
        "hf_n_new_tokens": hf_result["n_new_tokens"],
    }


def evaluate_chat_cross_check(server_result: dict, hf_result: dict) -> dict:
    """Build the ``chat``-mode CROSS-CHECK (NEVER the gate) per-prompt sub-report.

    ``server_result`` is ``call_chat_completion``'s return; ``hf_result`` is
    ``generate_hf_chat``'s. Canonicalizes the HF raw decode by stripping a
    leading think block (``strip_leading_think_block``) and compares
    against the server's own ``content`` field DIRECTLY -- never the lossy
    ``reconstruct_completion_text`` reconstruction, which is structurally
    incapable of byte-exactness (module docstring "Why chat mode cannot be
    the gate"). A residual mismatch that disappears once BOTH sides are
    left-stripped of whitespace is reclassified ``"channel_split_artifact"``
    -- exactly the shape of loss the autoparser's ``optspace`` combinator
    produces (consumes boundary whitespace, returns it to neither channel)
    -- and this classification, along with every other chat-mode outcome,
    NEVER flips ``main``'s exit code (``--mode chat`` is purely
    informational; ``--mode both`` reports this dict alongside the
    authoritative ``raw`` one).

    Exposes the server's raw ``content``/``reasoning_content`` VERBATIM
    (``server_content``/``server_reasoning_content``) plus the best-effort
    ``reconstruct_completion_text`` result (``server_reconstructed_text``,
    informational only) so a human reading the report can adjudicate a
    channel-split artifact from a genuine divergence without re-running
    anything.
    """
    server_content = server_result["content"]
    hf_text = hf_result["text"]
    hf_canonical, had_think_block = strip_leading_think_block(hf_text)
    comparison = compare_texts(server_content, hf_canonical)
    classification = comparison["classification"]
    if not comparison["match"] and server_content.lstrip() == hf_canonical.lstrip():
        classification = "channel_split_artifact"
    return {
        "match": comparison["match"],
        "classification": classification,
        "first_divergence": comparison["first_divergence"],
        "server_content": server_content,
        "server_reasoning_content": server_result.get("reasoning_content"),
        "server_reconstructed_text": server_result.get("text"),
        "hf_text": hf_text,
        "hf_canonical": hf_canonical,
        "had_think_block": had_think_block,
        "token_divergence": None,
        "server_finish_reason": server_result.get("finish_reason"),
        "server_model": server_result.get("model"),
        "server_hit_ceiling": _server_hit_ceiling(finish_reason=server_result.get("finish_reason")),
        "hf_hit_ceiling": hf_result["hit_ceiling"],
        "hf_n_new_tokens": hf_result["n_new_tokens"],
    }


# ============================================================================
# --report writing (never writes anywhere but the given path; refuses out/)
# ============================================================================


def assert_report_path_allowed(path) -> None:
    """Refuse a ``--report`` path with any path component literally ``out``.

    Exact (case-insensitive) component match, not a substring scan --
    ``out`` alone is too short/common a substring to blocklist loosely
    (would false-positive on e.g. "output"); a real path COMPONENT named
    "out" is what must never be written to (the pipeline's ``out/`` trees
    are managed elsewhere and this worktree's mandate forbids ever writing
    there).
    """
    resolved = Path(path).resolve()
    if any(part.lower() == "out" for part in resolved.parts):
        raise ReportPathError(
            f"refusing to write the parity report under an 'out/' "
            f"directory: {path} -- pass a --report path outside any out/ "
            "tree."
        )


def _channel_stats(results, key) -> dict:
    entries = [r[key] for r in results if r.get(key) is not None]
    if not entries:
        return None
    n_match = sum(1 for e in entries if e["match"])
    ceiling_hit_count = sum(
        1 for e in entries if e.get("server_hit_ceiling") or e.get("hf_hit_ceiling")
    )
    return {
        "n_prompts": len(entries),
        "n_match": n_match,
        "n_mismatch": len(entries) - n_match,
        "ceiling_hit_count": ceiling_hit_count,
    }


def build_report_payload(
    *, results, manifest, dequant_dir, server_url, max_new_tokens, mode, expected_alias, environment
) -> dict:
    """Assemble the ``parity_report.json`` payload (pure -- no I/O here).

    ``results`` is a list of ``{"uid", "prompt_sha256", "raw": dict|None,
    "chat": dict|None}`` -- ``raw`` populated when ``mode`` is ``"raw"``/
    ``"both"`` (``evaluate_raw_result``'s shape), ``chat`` populated when
    ``mode`` is ``"chat"``/``"both"`` (``evaluate_chat_cross_check``'s
    shape, which already carries the raw ``server_content``/
    ``server_reasoning_content`` fields verbatim -- a human must be able to
    adjudicate channel-split vs weight divergence from the report alone).

    Raises ``ValueError`` on an EMPTY ``results`` list -- an empty list
    would otherwise resolve every ``all(...)``/verdict computation
    vacuously, reporting a PASS for a run that checked nothing (never a
    vacuous PASS). The verdict is PASS iff either there is no ``raw``
    channel at all (a ``--mode chat``-only run is never the gate) or every
    ``raw`` entry matched; the ``chat`` channel's own match/mismatch never
    affects ``verdict``.
    """
    if not results:
        raise ValueError(
            "build_report_payload: empty results list -- refusing to report "
            "a vacuous PASS/FAIL verdict for a run that checked nothing."
        )
    determinism = manifest.get("determinism") or {}
    source_gguf = manifest.get("source_gguf") or {}
    raw_stats = _channel_stats(results, "raw")
    chat_stats = _channel_stats(results, "chat")
    verdict = "PASS" if raw_stats is None or raw_stats["n_mismatch"] == 0 else "FAIL"
    return {
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "dequant_dir": str(dequant_dir),
            "server_url": server_url,
            "max_new_tokens": max_new_tokens,
            "mode": mode,
            "expected_alias": expected_alias,
            "manifest_content_digest": determinism.get("content_digest"),
            "source_gguf_sha256": source_gguf.get("sha256"),
        },
        "environment": environment,
        "summary": {"raw": raw_stats, "chat": chat_stats},
        "prompts": [
            {
                "uid": r["uid"],
                "prompt_sha256": r["prompt_sha256"],
                "raw": r.get("raw"),
                "chat": r.get("chat"),
            }
            for r in results
        ],
    }


def write_report(path, payload: dict) -> None:
    """Write ``payload`` as JSON to ``path`` -- and NOWHERE else, ever.

    Re-runs ``assert_report_path_allowed`` at the point of the actual write
    (not just at argument-parsing time), so this function is safe to call
    directly, not only via ``main``.
    """
    assert_report_path_allowed(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================================
# HTTP calls -- stdlib urllib only, one request per call, no retries (max ONE
# concurrent Qwen-slot user is a machine-wide invariant this module must
# never violate: sequential prompts, one in-flight request at a time).
# ============================================================================

DEFAULT_TIMEOUT_S = 600

# RUNBOOK §0.4-EXECUTED's pinned baseline-arm alias (this repo, verified
# read-only 2026-07-30): `--alias qwen3-8b-q4km-base`. Sent as the chat
# payload's "model" field and checked against the response's "model" (major
# #3/#4) -- the box hosts base/smoke-lora/lora-arm servers on the same port
# at different times, so a wrong-server response must be diagnosable (the
# report always records it) and, with this default, preventable (a mismatch
# hard-fails unless --expected-alias is passed as an empty string to
# disable the check).
DEFAULT_EXPECTED_ALIAS = "qwen3-8b-q4km-base"

_CHAT_COMPLETIONS_SUFFIX = "/v1/chat/completions"


def _default_opener(request, timeout):
    return urllib.request.urlopen(request, timeout=timeout)


def _post_json(url, payload: dict, *, timeout, opener) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with opener(request, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.URLError as exc:
        raise TransportError(f"request to {url} failed: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise TransportError(f"{url}: response was not valid JSON: {exc}") from exc


def _check_expected_alias(url: str, response_model, expected_alias) -> None:
    if expected_alias and response_model != expected_alias:
        raise TransportError(
            f"{url}: response model={response_model!r} != expected "
            f"--expected-alias {expected_alias!r} -- this looks like the "
            "WRONG server for this comparison (the box hosts base/"
            "smoke-lora/lora-arm servers on the shared port at different "
            "times); refusing to score a parity comparison against the "
            "wrong model."
        )


def call_chat_completion(
    server_url: str,
    statement: str,
    max_new_tokens: int,
    *,
    model_alias: str = DEFAULT_EXPECTED_ALIAS,
    expected_alias=DEFAULT_EXPECTED_ALIAS,
    timeout=DEFAULT_TIMEOUT_S,
    opener=None,
) -> dict:
    """One greedy ``/v1/chat/completions`` request.

    ``--mode chat``/``both`` only (an informational cross-check, module
    docstring "Why chat mode cannot be the gate") -- ``server_url`` IS the
    chat-completions URL the operator named with ``--server-url``. Exactly
    one HTTP POST, no retries. ``opener(request, timeout) -> context-
    manager with .read()`` is injectable for tests (defaults to
    ``urllib.request.urlopen``).

    Wire envelope matches production ``qwen_http``:
    ``{"model", "messages", "temperature", "max_tokens"}`` (``"stream":
    False`` is an intentional, inert addition -- non-streaming is already
    the default without it, added only for explicitness). Per RUNBOOK
    §0.4-EXECUTED's measured parity caveats, sampler fields the client
    omits (``top_k``/``top_p``/``min_p``) are filled in IDENTICALLY by the
    server on both eval arms -- this module deliberately does not send them,
    so it never risks drifting that shared default between runs.

    Returns ``{"text", "content", "reasoning_content", "finish_reason",
    "model"}``. ``text`` is ``reconstruct_completion_text(content,
    reasoning_content)`` -- an INFORMATIONAL best-effort field only;
    ``evaluate_chat_cross_check`` compares ``content`` directly, never
    ``text``. Raises ``TransportError`` on a malformed response shape, a
    null ``message.content`` (legal OpenAI shape, but nothing to compare),
    or -- when ``expected_alias`` is truthy -- a response ``model`` that
    does not match it.
    """
    opener = opener if opener is not None else _default_opener
    payload = {
        "model": model_alias,
        "messages": build_chat_messages(statement),
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    body = _post_json(server_url, payload, timeout=timeout, opener=opener)
    try:
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content")
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        raise TransportError(
            f"{server_url}: unexpected chat-completions response shape: {body!r}"
        ) from exc

    if content is None:
        raise TransportError(
            f"{server_url}: message.content is null (finish_reason="
            f"{finish_reason!r}) -- a legal OpenAI shape (e.g. a tool-call-"
            "only message) but there is no text for this gate to compare."
        )

    reasoning_content = message.get("reasoning_content")
    response_model = body.get("model")
    _check_expected_alias(server_url, response_model, expected_alias)

    return {
        "text": reconstruct_completion_text(content, reasoning_content),
        "content": content,
        "reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
        "model": response_model,
    }


def derive_native_completion_url(server_url: str) -> str:
    """Swap a chat-completions URL's final route for llama-server's native completion route.

    Same scheme/host/port -- and any ``--api-prefix`` PREFIX before
    ``/v1/chat/completions`` is preserved -- so ``--mode raw``/``both`` talks
    to the identical server process ``--server-url`` already named, at
    ``<prefix>/completion``.
    """
    parsed = urllib.parse.urlsplit(server_url)
    path = parsed.path
    if path.endswith(_CHAT_COMPLETIONS_SUFFIX):
        prefix = path[: -len(_CHAT_COMPLETIONS_SUFFIX)]
    else:
        prefix = path.rsplit("/", 1)[0] if "/" in path else ""
    new_path = prefix + "/completion"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, new_path, "", ""))


def call_native_completion(
    native_url: str,
    raw_prompt: str,
    max_new_tokens: int,
    *,
    expected_alias=None,
    timeout=DEFAULT_TIMEOUT_S,
    opener=None,
) -> dict:
    """One greedy request to llama-server's native completion route.

    ``--mode raw``/``both`` (THE gate, module docstring): ``raw_prompt`` is
    the IDENTICAL already-rendered string also fed to the HF side. Exactly
    one HTTP POST, no retries. Returns ``{"text", "content", "stop_type",
    "model"}`` -- ``content`` is the server's raw, unparsed generation
    (``server-task.cpp`` ``to_json_non_oaicompat``, module docstring
    receipts), so ``text == content`` always in this mode (no reasoning-
    channel split to undo -- this is WHY raw mode can be byte-exact).
    ``stop_type`` is one of ``"eos"``/``"word"``/``"limit"`` -- the native
    equivalent of chat mode's ``finish_reason``. Same optional
    ``expected_alias`` wrong-server check as ``call_chat_completion``.
    """
    opener = opener if opener is not None else _default_opener
    payload = {
        "prompt": raw_prompt,
        "temperature": 0.0,
        "n_predict": max_new_tokens,
        "stream": False,
    }
    body = _post_json(native_url, payload, timeout=timeout, opener=opener)
    try:
        content = body["content"]
    except (KeyError, TypeError) as exc:
        raise TransportError(
            f"{native_url}: unexpected native-completion response shape: {body!r}"
        ) from exc

    response_model = body.get("model")
    _check_expected_alias(native_url, response_model, expected_alias)

    return {
        "text": content,
        "content": content,
        "stop_type": body.get("stop_type"),
        "model": response_model,
    }


# ============================================================================
# HF side -- lazy-imported (transformers/torch may not be installed here)
# ============================================================================

DEFAULT_DEVICE = "cpu"

# Pinned, not a CLI flag: "eager" is the most portable attention backend
# across CPU/MPS/CUDA and avoids the tiny numeric drift optimized kernels
# (SDPA/flash-attention) can introduce -- this gate needs byte-exact greedy
# decoding, not throughput. Recorded in parity_report.json's "environment"
# block alongside the fp32 dtype choice (this gate verifies the fp32
# dequantization directory specifically; whether a bf16 comparison arm is
# ever needed is the open training-precision question this repo has not
# resolved -- see README/RUNBOOK "Open items" -- and is out of scope here).
DEFAULT_ATTN_IMPLEMENTATION = "eager"


def _lazy_import_hf_stack():
    """Import ``torch`` + ``transformers``, or raise ``DependencyError`` with an actionable message.

    Never imported at module load time: every other function in this module
    (the campaign guard, prompt extraction, manifest preconditions,
    comparison logic, report writing) must stay usable/testable on a
    machine that has neither package installed. The HF side of this gate
    only runs where the stack is actually present -- on the box
    post-campaign, or after an approved local install.

    Returns ``(torch, AutoModelForCausalLM, AutoTokenizer,
    transformers_version)`` -- the version string is recorded verbatim in
    ``parity_report.json``'s ``environment`` block (major #6).
    """
    try:
        import torch
    except ImportError as exc:
        raise DependencyError(
            "the 'torch' package is not importable in this environment. "
            "This gate's HF side (loading --dequant-dir and running greedy "
            "generate) needs both torch and transformers -- install them "
            "(operator-approved, post-campaign) before running anything "
            "but --help."
        ) from exc
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise DependencyError(
            "the 'transformers' package is not importable in this "
            "environment. This gate's HF side (loading --dequant-dir and "
            "running greedy generate) needs both torch and transformers -- "
            "install them (operator-approved, post-campaign) before running "
            "anything but --help."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, getattr(transformers, "__version__", None)


def generate_hf_chat(
    model, tokenizer, statement: str, max_new_tokens: int, torch_mod, device, eos_token_id=None
) -> dict:
    """Chat-mode HF generation: render the pinned wire, greedy-decode, decode back.

    Renders via ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``
    -- the SAME wire ``call_chat_completion`` sends to the server, just
    templated locally instead of by llama-server. Greedy (``do_sample=False``);
    decodes the completion span only, without special tokens. Returns
    ``_generate_and_decode``'s ``{"text", "n_new_tokens", "hit_ceiling"}``.
    """
    input_ids = tokenizer.apply_chat_template(
        build_chat_messages(statement),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return _generate_and_decode(model, tokenizer, input_ids, max_new_tokens, torch_mod, device, eos_token_id)


def generate_hf_raw(
    model, tokenizer, raw_prompt: str, max_new_tokens: int, torch_mod, device, eos_token_id=None
) -> dict:
    """Raw-mode HF generation: tokenize the already-rendered string directly, greedy-decode.

    ``add_special_tokens=False`` -- the already-rendered template string
    carries its own special tokens (BOS/im_start/etc.); tokenizing it again
    WITH special tokens would double them up, unlike the chat path (which
    encodes fresh conversation turns and needs the tokenizer's own specials
    applied via ``apply_chat_template``).
    """
    encoded = tokenizer(raw_prompt, add_special_tokens=False, return_tensors="pt")
    return _generate_and_decode(
        model, tokenizer, encoded["input_ids"], max_new_tokens, torch_mod, device, eos_token_id
    )


def _generate_and_decode(model, tokenizer, input_ids, max_new_tokens: int, torch_mod, device, eos_token_id=None) -> dict:
    """Move ``input_ids`` to ``device``, greedy-generate, decode the completion span.

    Moving ``input_ids`` (not just the model, via ``model.to(device)`` in
    ``main``) is required for ANY non-CPU device -- ``model.generate``
    hard-crashes on a device mismatch between the model's parameters and
    its input tensors otherwise. fp32 8B greedy decode on CPU is
    impractically slow, so ``mps``/``cuda`` are the realistic choices this
    exists to support.

    ``hit_ceiling`` excludes a terminating EOS token from the count (
    normalized via ``_normalize_token_ids`` -- ``eos_token_id`` may be an
    int or a list): a generation that naturally stopped by emitting EOS on
    the LAST allowed step is a natural stop, not a truncation, even though
    its raw token count equals ``max_new_tokens``.
    """
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[-1]
    with torch_mod.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False
        )
    completion_ids = output_ids[0][prompt_len:]
    n_new_tokens = len(completion_ids)
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    counted_tokens = n_new_tokens
    eos_ids = _normalize_token_ids(eos_token_id)
    if eos_ids and n_new_tokens > 0 and int(completion_ids[-1]) in eos_ids:
        counted_tokens -= 1

    return {
        "text": text,
        "n_new_tokens": n_new_tokens,
        "hit_ceiling": counted_tokens >= max_new_tokens,
    }


def base_serve_command(gguf_path, port: int, alias: str = DEFAULT_EXPECTED_ALIAS) -> list:
    """Pure argv builder for the RUNBOOK §0.4-EXECUTED reference BASE-only invocation.

    No subprocess is started here (mirrors ``export_serve.
    llama_server_command``'s style). Deliberately no ``--lora`` parameter at
    all: this gate compares the dequantized HF model against the BASE
    model's served behaviour, so the reference server this tool talks to
    must never carry an adapter. ``alias`` defaults to the RUNBOOK-pinned
    baseline alias (``DEFAULT_EXPECTED_ALIAS``), matching this module's own
    ``--expected-alias`` default so the wrong-server check has something
    meaningful to check against out of the box.
    """
    return [
        "llama-server",
        "-m", str(gguf_path),
        "--alias", alias,
        "-c", "8192",
        "-ngl", "99",
        "--parallel", "1",
        "--port", str(port),
    ]


# ============================================================================
# CLI
# ============================================================================

DEFAULT_MAX_NEW_TOKENS = 2048  # the production pass@k completion budget (RUNBOOK §0.4)
DEFAULT_MODE = "raw"
VALID_MODES = ("raw", "chat", "both")

_EPILOG = """\
POST-CAMPAIGN GATE -- never run this while any eval owns the Qwen slot (the
campaign-liveness guard above refuses loudly on its own, before every server
call, not just at startup; pass --i-own-the-qwen-slot only after personally
verifying the campaign is not actually running).

Reference serve command for the BASE model this gate compares against (NO
--lora; see base_serve_command()):

    llama-server -m <pinned base gguf> --alias qwen3-8b-q4km-base \\
        -c 8192 -ngl 99 --parallel 1 --port <port>

--mode raw (default) is THE GATE: llama-server's native /completion route
returns raw, unparsed text, so it is the only channel a byte-exact
comparison against the HF decode is actually achievable on. --mode chat is
an OPTIONAL, INFORMATIONAL cross-check against /v1/chat/completions -- it
NEVER affects the exit code (chat mode's own reasoning-content extraction
makes byte-exact reconstruction structurally impossible; see module
docstring "Why chat mode cannot be the gate"). --mode both runs both,
still exits based on raw alone.

Exit codes: 0 = PASS, 1 = weight-divergence FAIL (raw-channel mismatch,
ANY classification), 2 = everything else (missing inputs, HTTP/HF infra
errors, wrong-server, scheme-less --server-url, the guard tripping).
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_dequant_parity",
        description=__doc__,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # so e.g. "--i" can never accidentally match --i-own-the-qwen-slot
    )
    parser.add_argument(
        "--dequant-dir", required=True,
        help="the gguf_to_hf.py output directory (fp32 HF dir + dequant_manifest.json)",
    )
    parser.add_argument(
        "--server-url", required=True,
        help=(
            "the llama-server chat-completions endpoint (an explicit URL "
            "with an http or https scheme -- scheme-less URLs are refused, "
            "see _require_url_scheme). REQUIRED, no default -- a default could "
            "silently aim this gate at whatever server happens to be "
            "running (e.g. a live campaign's), so the operator must always "
            "name it explicitly."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prompts-file", default=None,
        help="a plain jsonl of {\"uid\": ..., \"statement\": ...} rows",
    )
    source.add_argument(
        "--from-eval-set", default=None,
        help=(
            f"an evalharness eval_set.jsonl; extracts the {EXPECTED_ANCHOR_COUNT} "
            f"rows whose eval_slice == {ANCHOR_EVAL_SLICE!r} (hard-fails if the "
            "count is not exactly that)"
        ),
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"greedy-decode budget per prompt on both engines (default {DEFAULT_MAX_NEW_TOKENS}; must be >= 1)",
    )
    parser.add_argument(
        "--report", default=None,
        help="write parity_report.json to this path (refused if any path component is 'out')",
    )
    parser.add_argument(
        "--mode", choices=VALID_MODES, default=DEFAULT_MODE,
        help=(
            f"'raw' (default) is THE GATE (native /completion, byte-exact); "
            "'chat' is an informational-only cross-check against "
            "/v1/chat/completions (never affects the exit code); 'both' "
            "runs both, exits based on raw alone -- see epilog."
        ),
    )
    parser.add_argument(
        "--i-own-the-qwen-slot", action="store_true",
        help="override the campaign-liveness guard; only after personally verifying the campaign is not live",
    )
    parser.add_argument(
        "--expected-alias", "--model-alias", dest="expected_alias", default=DEFAULT_EXPECTED_ALIAS,
        help=(
            "the server 'model'/--alias identity this gate expects (sent as "
            "the chat request's 'model' field AND checked against the "
            "response's 'model' -- a mismatch hard-fails as a wrong-server "
            f"error). Default {DEFAULT_EXPECTED_ALIAS!r} (RUNBOOK "
            "§0.4-EXECUTED's pinned baseline alias). Pass an empty string "
            "to disable the response-side check."
        ),
    )
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE,
        help=f"torch device for the HF model AND its input tensors (default {DEFAULT_DEVICE!r})",
    )
    return parser


def _format_summary(results, mode: str) -> str:
    lines = [f"{'uid':<36} {'raw':<8} {'chat':<8}"]
    for r in results:
        raw = r.get("raw")
        chat = r.get("chat")
        raw_status = ("PASS" if raw["match"] else "FAIL") if raw is not None else "-"
        chat_status = ("PASS" if chat["match"] else chat["classification"]) if chat is not None else "-"
        lines.append(f"{r['uid']:<36} {raw_status:<8} {chat_status:<8}")
    raw_stats = _channel_stats(results, "raw")
    if raw_stats is not None:
        lines.append(f"raw: {raw_stats['n_match']}/{raw_stats['n_prompts']} matched byte-for-byte (THE gate)")
    chat_stats = _channel_stats(results, "chat")
    if chat_stats is not None:
        lines.append(f"chat: {chat_stats['n_match']}/{chat_stats['n_prompts']} matched (informational only)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.max_new_tokens < 1:
            raise ValueError(f"--max-new-tokens must be >= 1 (got {args.max_new_tokens})")

        _require_url_scheme(args.server_url)

        refuse_if_campaign_live(campaign_liveness_hits(args.server_url), args.i_own_the_qwen_slot)

        if args.report is not None:
            assert_report_path_allowed(args.report)

        if args.prompts_file:
            prompts = load_prompts_file(args.prompts_file)
        else:
            prompts = extract_anchor_prompts(args.from_eval_set)

        dequant_dir = Path(args.dequant_dir)
        manifest, warnings, generation_config = check_dequant_dir_preconditions(dequant_dir)
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)

        torch_mod, model_cls, tokenizer_cls, transformers_version = _lazy_import_hf_stack()

        tokenizer = tokenizer_cls.from_pretrained(str(dequant_dir), local_files_only=True)
        model = model_cls.from_pretrained(
            str(dequant_dir),
            torch_dtype=torch_mod.float32,
            local_files_only=True,
            attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
        )
        model.to(args.device)
        model.eval()

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

        native_url = derive_native_completion_url(args.server_url) if args.mode in ("raw", "both") else None

        # Sequential, one in-flight request at a time -- max ONE concurrent
        # user of the local Qwen slot is a machine-wide invariant this loop
        # must never violate (no threading, no batching, no retries). The
        # guard is re-checked immediately before EACH server call (raw's
        # and, in "both" mode, chat's, separately) -- HF generation between
        # calls can run long enough for a campaign to start meanwhile.
        results = []
        for prompt in prompts:
            statement = prompt["statement"]
            raw_entry = None
            chat_entry = None

            if args.mode in ("raw", "both"):
                refuse_if_campaign_live(
                    campaign_liveness_hits(args.server_url), args.i_own_the_qwen_slot
                )
                raw_prompt = render_raw_prompt(tokenizer, statement)
                server_result = call_native_completion(
                    native_url, raw_prompt, args.max_new_tokens, expected_alias=args.expected_alias
                )
                hf_result = generate_hf_raw(
                    model, tokenizer, raw_prompt, args.max_new_tokens, torch_mod, args.device, eos_token_id
                )
                raw_entry = evaluate_raw_result(server_result, hf_result)
                if not raw_entry["match"]:
                    raw_entry["token_divergence"] = token_divergence(
                        tokenizer, raw_entry["server_text"], raw_entry["hf_text"]
                    )
                    print(
                        f"MISMATCH [raw/GATE] uid={prompt['uid']} class={raw_entry['classification']} "
                        f"first_divergence={raw_entry['first_divergence']}",
                        file=sys.stderr,
                    )

            if args.mode in ("chat", "both"):
                refuse_if_campaign_live(
                    campaign_liveness_hits(args.server_url), args.i_own_the_qwen_slot
                )
                server_result = call_chat_completion(
                    args.server_url, statement, args.max_new_tokens,
                    model_alias=args.expected_alias, expected_alias=args.expected_alias,
                )
                hf_result = generate_hf_chat(
                    model, tokenizer, statement, args.max_new_tokens, torch_mod, args.device, eos_token_id
                )
                chat_entry = evaluate_chat_cross_check(server_result, hf_result)
                if not chat_entry["match"]:
                    chat_entry["token_divergence"] = token_divergence(
                        tokenizer, chat_entry["server_content"], chat_entry["hf_canonical"]
                    )
                    print(
                        f"MISMATCH [chat/informational] uid={prompt['uid']} "
                        f"class={chat_entry['classification']} "
                        f"first_divergence={chat_entry['first_divergence']}",
                        file=sys.stderr,
                    )

            results.append({
                "uid": prompt["uid"],
                "prompt_sha256": hashlib.sha256(statement.encode("utf-8")).hexdigest(),
                "raw": raw_entry,
                "chat": chat_entry,
            })

        saw_any_text = any(
            (r["raw"] is not None and (r["raw"]["server_text"] or r["raw"]["hf_text"]))
            or (r["chat"] is not None and (r["chat"]["server_content"] or r["chat"]["hf_canonical"]))
            for r in results
        )
        if results and not saw_any_text:
            raise RuntimeError(
                "all completions were empty across every prompt -- refusing "
                "a vacuous PASS; check --max-new-tokens, the server, and the "
                "HF model load."
            )

        environment = {
            "device": args.device,
            "dtype": "float32",
            "attn_implementation": DEFAULT_ATTN_IMPLEMENTATION,
            "torch_version": getattr(torch_mod, "__version__", None),
            "transformers_version": transformers_version,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "generation_config": generation_config,
        }

        print(_format_summary(results, args.mode))

        if args.report is not None:
            payload = build_report_payload(
                results=results,
                manifest=manifest,
                dequant_dir=dequant_dir,
                server_url=args.server_url,
                max_new_tokens=args.max_new_tokens,
                mode=args.mode,
                expected_alias=args.expected_alias,
                environment=environment,
            )
            write_report(args.report, payload)

    except Exception as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    if args.mode in ("raw", "both"):
        gate_all_match = all(r["raw"]["match"] for r in results)
    else:
        gate_all_match = True  # --mode chat is informational only, never gates

    if not gate_all_match:
        first_bad = next(r for r in results if not r["raw"]["match"])
        print(
            f"PARITY GATE FAILED at uid={first_bad['uid']}: classification="
            f"{first_bad['raw']['classification']} first divergence at char "
            f"index {first_bad['raw']['first_divergence']['char_index']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

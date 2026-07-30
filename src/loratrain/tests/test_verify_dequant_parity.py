"""Tests for loratrain.verify_dequant_parity (the T2 BLOCKING parity gate).

All hermetic: no network, no real server, no transformers/torch actually
exercised (the lazy-import tests force ImportError via ``sys.modules``
injection rather than relying on either package's real absence/presence).
Every HTTP call is driven through an injected ``opener``; every process
liveness check is driven through an injected ``pgrep_runner``/``lsof_runner``;
the main()-level end-to-end tests inject a fake HF stack via
``_lazy_import_hf_stack`` monkeypatching.

``--mode raw`` (default) is THE GATE; ``--mode chat`` is an informational
cross-check that never affects the exit code; ``--mode both`` runs both.
Several fixtures below reproduce EXACT values hand-traced against the
b10107 source (``scratchpad/verify-t2/repro_reconstruction.py``) to prove
the empty-think and whitespace-loss channel-split cases behave as designed.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path

import pytest

from loratrain import config
from loratrain import verify_base_identity as vbi
from loratrain import verify_dequant_parity as vdp


# ============================================================================
# Campaign-liveness guard
# ============================================================================


@dataclass
class _FakeProcResult:
    returncode: int = 1
    stdout: str = ""


def _pgrep_stub(rc_by_pattern: dict):
    def _runner(argv):
        pattern = argv[-1]
        hit = rc_by_pattern.get(pattern, False)
        return _FakeProcResult(returncode=0 if hit else 1, stdout="12345\n" if hit else "")
    return _runner


def test_check_campaign_not_live_all_clear():
    runner = _pgrep_stub({p: False for p in vdp.CAMPAIGN_LIVENESS_PATTERNS})
    assert vdp.check_campaign_not_live(pgrep_runner=runner) == []


def test_check_campaign_not_live_single_hit():
    pattern = vdp.CAMPAIGN_LIVENESS_PATTERNS[0]
    runner = _pgrep_stub({pattern: True})
    assert vdp.check_campaign_not_live(pgrep_runner=runner) == [pattern]


def test_check_campaign_not_live_both_hit():
    runner = _pgrep_stub({p: True for p in vdp.CAMPAIGN_LIVENESS_PATTERNS})
    assert vdp.check_campaign_not_live(pgrep_runner=runner) == list(vdp.CAMPAIGN_LIVENESS_PATTERNS)


def test_check_campaign_not_live_rc0_empty_stdout_is_clear():
    def _runner(argv):
        return _FakeProcResult(returncode=0, stdout="")
    assert vdp.check_campaign_not_live(pgrep_runner=_runner) == []


def test_check_campaign_not_live_runner_exception_is_conservative_hit():
    def _runner(argv):
        raise OSError("no pgrep binary")
    assert vdp.check_campaign_not_live(pgrep_runner=_runner) == list(vdp.CAMPAIGN_LIVENESS_PATTERNS)


def test_check_campaign_not_live_pgrep_syntax_error_rc_is_conservative_hit():
    def _runner(argv):
        return _FakeProcResult(returncode=2, stdout="")
    assert vdp.check_campaign_not_live(pgrep_runner=_runner) == list(vdp.CAMPAIGN_LIVENESS_PATTERNS)


def test_refuse_if_campaign_live_raises_without_override():
    with pytest.raises(vdp.CampaignLiveError, match="eval_all.sh"):
        vdp.refuse_if_campaign_live(["eval_all.sh"], i_own_the_qwen_slot=False)


def test_refuse_if_campaign_live_passes_with_override():
    vdp.refuse_if_campaign_live(["eval_all.sh"], i_own_the_qwen_slot=True)  # no raise


def test_refuse_if_campaign_live_noop_when_clear():
    vdp.refuse_if_campaign_live([], i_own_the_qwen_slot=False)  # no raise


# --- port-liveness probe (qwen_slot_free's lsof half) + combined guard -----
# check_server_port_busy: busy = non-empty stdout REGARDLESS of returncode
# (lsof on macOS can exit 1 even with a matching PID printed -- a
# per-process access error on some OTHER process trips the same nonzero
# exit as "found nothing"). Exception is still conservative-busy.


def test_check_server_port_busy_nonempty_stdout_rc0_is_busy():
    def fake_lsof(argv):
        return _FakeProcResult(returncode=0, stdout="12345\n")
    assert vdp.check_server_port_busy(8081, lsof_runner=fake_lsof) is True


def test_check_server_port_busy_nonempty_stdout_rc1_is_still_busy():
    # THE bug this fixes: lsof prints a PID but exits 1 anyway.
    def fake_lsof(argv):
        return _FakeProcResult(returncode=1, stdout="12345\n")
    assert vdp.check_server_port_busy(8081, lsof_runner=fake_lsof) is True


def test_check_server_port_busy_empty_stdout_is_clear_regardless_of_rc():
    def fake_lsof_rc0(argv):
        return _FakeProcResult(returncode=0, stdout="")
    def fake_lsof_rc1(argv):
        return _FakeProcResult(returncode=1, stdout="")
    assert vdp.check_server_port_busy(8081, lsof_runner=fake_lsof_rc0) is False
    assert vdp.check_server_port_busy(8081, lsof_runner=fake_lsof_rc1) is False


def test_check_server_port_busy_runner_exception_is_conservative_busy():
    def fake_lsof(argv):
        raise OSError("no lsof binary")
    assert vdp.check_server_port_busy(8081, lsof_runner=fake_lsof) is True


def test_campaign_liveness_hits_includes_port_hit():
    def fake_pgrep(argv):
        return _FakeProcResult(returncode=1, stdout="")
    def fake_lsof(argv):
        return _FakeProcResult(returncode=0, stdout="999\n")
    hits = vdp.campaign_liveness_hits(
        "http://example.invalid:8081/v1/chat/completions",
        pgrep_runner=fake_pgrep, lsof_runner=fake_lsof,
    )
    assert any("8081" in h for h in hits)


def test_campaign_liveness_hits_skips_port_check_without_explicit_port():
    def fake_pgrep(argv):
        return _FakeProcResult(returncode=1, stdout="")
    def fake_lsof(argv):
        raise AssertionError("lsof should not be called without an explicit port")
    hits = vdp.campaign_liveness_hits(
        "http://example.invalid/v1/chat/completions",
        pgrep_runner=fake_pgrep, lsof_runner=fake_lsof,
    )
    assert hits == []


def test_campaign_liveness_hits_server_url_none_skips_port_check():
    def fake_pgrep(argv):
        return _FakeProcResult(returncode=1, stdout="")
    assert vdp.campaign_liveness_hits(None, pgrep_runner=fake_pgrep) == []


def test_campaign_liveness_hits_combines_pgrep_and_port_hits():
    pattern = vdp.CAMPAIGN_LIVENESS_PATTERNS[0]
    def fake_pgrep(argv):
        return _FakeProcResult(returncode=0 if argv[-1] == pattern else 1, stdout="1\n" if argv[-1] == pattern else "")
    def fake_lsof(argv):
        return _FakeProcResult(returncode=0, stdout="1\n")
    hits = vdp.campaign_liveness_hits(
        "http://example.invalid:9999/v1/chat/completions",
        pgrep_runner=fake_pgrep, lsof_runner=fake_lsof,
    )
    assert pattern in hits
    assert any("9999" in h for h in hits)
    assert len(hits) == 2


# --- scheme-less --server-url refusal (minor #8) ----------------------------


def test_require_url_scheme_accepts_http_and_https():
    vdp._require_url_scheme("http://example.invalid:8081/v1/chat/completions")
    vdp._require_url_scheme("https://example.invalid:8081/v1/chat/completions")


def test_require_url_scheme_rejects_scheme_less_url():
    with pytest.raises(ValueError, match="no http/https scheme"):
        vdp._require_url_scheme("example.invalid:8081/v1/chat/completions")


def test_require_url_scheme_rejects_other_scheme():
    with pytest.raises(ValueError, match="no http/https scheme"):
        vdp._require_url_scheme("ftp://example.invalid:8081/v1/chat/completions")


# ============================================================================
# Wire construction -- frozen-literal tripwire (mirrors build_dataset.py's)
# ============================================================================


def test_wire_pins_match_frozen_literals():
    assert config.PASS_AT_K_SYSTEM_PROMPT == "Solve the problem. State only the final answer inside \\boxed{}."
    assert config.PASS_AT_K_NO_THINK_SUFFIX == " /no_think"


def test_build_chat_messages_uses_config_wire_pins():
    messages = vdp.build_chat_messages("STATEMENT")
    assert messages == [
        {"role": "system", "content": config.PASS_AT_K_SYSTEM_PROMPT},
        {"role": "user", "content": "STATEMENT" + config.PASS_AT_K_NO_THINK_SUFFIX},
    ]


# ============================================================================
# reconstruct_completion_text (informational only now -- NOT the gate)
# ============================================================================


def test_reconstruct_completion_text_no_reasoning_content_key():
    assert vdp.reconstruct_completion_text("ANSWER", None) == "ANSWER"


def test_reconstruct_completion_text_empty_reasoning_content():
    assert vdp.reconstruct_completion_text("ANSWER", "") == "ANSWER"


def test_reconstruct_completion_text_wraps_reasoning_back_in():
    result = vdp.reconstruct_completion_text("ANSWER", "some reasoning")
    assert result == "<think>some reasoning</think>\n\nANSWER"


def test_reconstruct_completion_text_none_content_does_not_embed_literal_none():
    # minor #9: unreachable via call_chat_completion today (it hard-fails on
    # null content first), but cheap to close defensively.
    result = vdp.reconstruct_completion_text(None, "R")
    assert result == "<think>R</think>\n\n"
    assert "None" not in result


# ============================================================================
# strip_leading_think_block (blocker: chat cross-check canonicalization)
# ============================================================================


def test_strip_leading_think_block_no_think_tag():
    assert vdp.strip_leading_think_block("The answer is 42.") == ("The answer is 42.", False)


def test_strip_leading_think_block_empty_think_matches_repro_trace():
    # Exact case from scratchpad/verify-t2/repro_reconstruction.py: every
    # /no_think Qwen3 anchor emits this empty think prefix.
    raw = "<think>\n\n</think>\n\nThe answer is 42."
    canonical, had_think = vdp.strip_leading_think_block(raw)
    assert canonical == "The answer is 42."
    assert had_think is True


def test_strip_leading_think_block_non_empty_reasoning_matches_repro_trace():
    raw = "<think>\nLet me check the PDE.\n</think>\n\nThe answer is 42."
    canonical, had_think = vdp.strip_leading_think_block(raw)
    assert canonical == "The answer is 42."
    assert had_think is True


def test_strip_leading_think_block_length_capped_mid_think_matches_repro_trace():
    raw = "<think>\nStep 1: consider the bound"
    canonical, had_think = vdp.strip_leading_think_block(raw)
    assert canonical == ""
    assert had_think is True


def test_strip_leading_think_block_extra_trailing_newline_leaves_residual():
    # Sets up the channel_split_artifact fixture: 3 trailing newlines
    # instead of the modeled 2 leaves ONE residual leading newline.
    raw = "<think>\nreasoning\n</think>\n\n\nThe answer is 42."
    canonical, had_think = vdp.strip_leading_think_block(raw)
    assert canonical == "\nThe answer is 42."
    assert had_think is True


# ============================================================================
# Comparison logic
# ============================================================================


def test_compare_texts_exact_match():
    result = vdp.compare_texts("same text", "same text")
    assert result == {"match": True, "first_divergence": None, "classification": "match"}


def test_compare_texts_content_divergence_reports_first_divergence():
    result = vdp.compare_texts("The answer is 42.", "The answer is 43.")
    assert result["match"] is False
    assert result["classification"] == "content_divergence"
    assert result["first_divergence"]["char_index"] == 15
    assert "42" in result["first_divergence"]["server_context"]
    assert "43" in result["first_divergence"]["hf_context"]


def test_compare_texts_prefix_divergence_classified_as_prefix_truncation():
    result = vdp.compare_texts("abc", "abcdef")
    assert result["match"] is False
    assert result["classification"] == "prefix_truncation"
    assert result["first_divergence"]["char_index"] == 3


def test_compare_texts_unicode_divergence_is_codepoint_indexed():
    result = vdp.compare_texts("café", "cafe")
    assert result["match"] is False
    assert result["first_divergence"]["char_index"] == 3


def test_token_divergence_returns_none_without_tokenizer():
    assert vdp.token_divergence(None, "a", "b") is None


def test_token_divergence_returns_none_on_encode_failure():
    class _BadTokenizer:
        def encode(self, text, add_special_tokens=False):
            raise RuntimeError("boom")
    assert vdp.token_divergence(_BadTokenizer(), "a", "b") is None


def test_token_divergence_reports_first_differing_token():
    class _FakeTokenizer:
        _table = {"hello world": [1, 2, 3], "hello there": [1, 2, 4]}
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return self._table[text]
    result = vdp.token_divergence(_FakeTokenizer(), "hello world", "hello there")
    assert result["token_index"] == 2
    assert result["server_token_ids"] == [1, 2, 3]
    assert result["hf_token_ids"] == [1, 2, 4]


# ============================================================================
# evaluate_raw_result / evaluate_chat_cross_check (per-prompt sub-reports)
# ============================================================================


def test_evaluate_raw_result_match():
    server_result = {"text": "ANSWER", "content": "ANSWER", "stop_type": "eos", "model": "m"}
    hf_result = {"text": "ANSWER", "n_new_tokens": 3, "hit_ceiling": False}
    entry = vdp.evaluate_raw_result(server_result, hf_result)
    assert entry["match"] is True
    assert entry["classification"] == "match"
    assert entry["server_text"] == "ANSWER"
    assert entry["hf_text"] == "ANSWER"
    assert entry["server_finish_reason"] == "eos"
    assert entry["token_divergence"] is None


def test_evaluate_raw_result_hit_ceiling_from_stop_type_limit():
    server_result = {"text": "X", "content": "X", "stop_type": "limit", "model": "m"}
    hf_result = {"text": "X", "n_new_tokens": 5, "hit_ceiling": True}
    entry = vdp.evaluate_raw_result(server_result, hf_result)
    assert entry["server_hit_ceiling"] is True


def test_evaluate_chat_cross_check_empty_think_fixture_is_a_match():
    # Reproduces the blocker exactly: empty-think generation, server content
    # already extracted (no reasoning_content key at all).
    server_result = {
        "text": "The answer is 42.", "content": "The answer is 42.",
        "reasoning_content": None, "finish_reason": "stop", "model": "m",
    }
    hf_result = {"text": "<think>\n\n</think>\n\nThe answer is 42.", "n_new_tokens": 10, "hit_ceiling": False}
    entry = vdp.evaluate_chat_cross_check(server_result, hf_result)
    assert entry["match"] is True
    assert entry["classification"] == "match"
    assert entry["had_think_block"] is True
    assert entry["server_content"] == "The answer is 42."
    assert entry["server_reasoning_content"] is None


def test_evaluate_chat_cross_check_non_empty_reasoning_fixture_is_a_match():
    server_result = {
        "text": "<think>Let me check the PDE.\n</think>\n\nThe answer is 42.",
        "content": "The answer is 42.", "reasoning_content": "Let me check the PDE.\n",
        "finish_reason": "stop", "model": "m",
    }
    hf_result = {
        "text": "<think>\nLet me check the PDE.\n</think>\n\nThe answer is 42.",
        "n_new_tokens": 12, "hit_ceiling": False,
    }
    entry = vdp.evaluate_chat_cross_check(server_result, hf_result)
    assert entry["match"] is True
    assert entry["server_reasoning_content"] == "Let me check the PDE.\n"


def test_evaluate_chat_cross_check_whitespace_loss_classifies_channel_split_artifact():
    # The non-empty-reasoning whitespace-loss case: server's own parser
    # consumed all 3 boundary newlines, this module's canonicalization
    # (modeled on 2) leaves one residual leading newline.
    server_result = {
        "text": "<think>reasoning</think>\n\nThe answer is 42.",
        "content": "The answer is 42.", "reasoning_content": "reasoning",
        "finish_reason": "stop", "model": "m",
    }
    hf_result = {
        "text": "<think>\nreasoning\n</think>\n\n\nThe answer is 42.",
        "n_new_tokens": 12, "hit_ceiling": False,
    }
    entry = vdp.evaluate_chat_cross_check(server_result, hf_result)
    assert entry["match"] is False
    assert entry["classification"] == "channel_split_artifact"
    assert entry["hf_canonical"] == "\nThe answer is 42."


def test_evaluate_chat_cross_check_genuine_content_divergence_not_reclassified():
    # A REAL divergence (different digit) must NOT be laundered into
    # channel_split_artifact just because a think block was present.
    server_result = {
        "text": "The answer is 42.", "content": "The answer is 42.",
        "reasoning_content": None, "finish_reason": "stop", "model": "m",
    }
    hf_result = {
        "text": "<think>\n\n</think>\n\nThe answer is 43.",
        "n_new_tokens": 10, "hit_ceiling": False,
    }
    entry = vdp.evaluate_chat_cross_check(server_result, hf_result)
    assert entry["match"] is False
    assert entry["classification"] == "content_divergence"


def test_evaluate_chat_cross_check_no_think_block_at_all_is_ordinary_comparison():
    server_result = {
        "text": "The answer is 42.", "content": "The answer is 42.",
        "reasoning_content": None, "finish_reason": "stop", "model": "m",
    }
    hf_result = {"text": "The answer is 42.", "n_new_tokens": 5, "hit_ceiling": False}
    entry = vdp.evaluate_chat_cross_check(server_result, hf_result)
    assert entry["match"] is True
    assert entry["had_think_block"] is False


# ============================================================================
# --from-eval-set anchor extraction (real eval_set.jsonl schema)
# ============================================================================


def _eval_row(uid, statement, eval_slice):
    return {
        "uid": uid,
        "statement": statement,
        "answer": "some answer",
        "arxiv_id": "2601.00001",
        "family": "pde",
        "tier": "latex",
        "source": "test_fixture",
        "provenance": "extracted",
        "truth_policy": "extracted",
        "metadata": {"title": "t", "primary_category": "math.AP"},
        "eval_slice": eval_slice,
    }


def _write_eval_set(path, n_anchor_solved=10, n_anchor_fail=10, n_eval_band=5):
    rows = []
    for i in range(n_anchor_solved):
        rows.append(_eval_row(f"anchor_solved_{i}", f"statement {i}", "anchor_solved"))
    for i in range(n_anchor_fail):
        rows.append(_eval_row(f"anchor_fail_{i}", f"fail statement {i}", "anchor_fail"))
    for i in range(n_eval_band):
        rows.append(_eval_row(f"band_{i}", f"band statement {i}", "eval_band"))
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows


def test_extract_anchor_prompts_happy_path(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    _write_eval_set(path)
    prompts = vdp.extract_anchor_prompts(path)
    assert len(prompts) == 10
    assert all(p["uid"].startswith("anchor_solved_") for p in prompts)


def test_extract_anchor_prompts_wrong_count_hard_fails_too_few(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    _write_eval_set(path, n_anchor_solved=9)
    with pytest.raises(vdp.AnchorExtractionError, match="found 9"):
        vdp.extract_anchor_prompts(path)


def test_extract_anchor_prompts_wrong_count_hard_fails_too_many(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    _write_eval_set(path, n_anchor_solved=11)
    with pytest.raises(vdp.AnchorExtractionError, match="found 11"):
        vdp.extract_anchor_prompts(path)


def test_extract_anchor_prompts_blank_statement_hard_fails(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    rows = [_eval_row(f"a{i}", f"s{i}", "anchor_solved") for i in range(10)]
    rows[3]["statement"] = ""
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(vdp.AnchorExtractionError, match="no non-empty uid/statement"):
        vdp.extract_anchor_prompts(path)


# ============================================================================
# --prompts-file loader
# ============================================================================


def test_load_prompts_file_happy_path(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"uid": "u1", "statement": "s1"}) + "\n"
        + "\n"
        + json.dumps({"uid": "u2", "statement": "s2"}) + "\n",
        encoding="utf-8",
    )
    prompts = vdp.load_prompts_file(path)
    assert prompts == [{"uid": "u1", "statement": "s1"}, {"uid": "u2", "statement": "s2"}]


def test_load_prompts_file_missing_field_raises(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"uid": "u1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing non-empty"):
        vdp.load_prompts_file(path)


def test_load_prompts_file_empty_raises(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no usable prompts"):
        vdp.load_prompts_file(path)


# ============================================================================
# --dequant-dir sanity preconditions (identity binding + generation_config)
# ============================================================================


def _write_manifest(
    dequant_dir,
    *,
    base_scheme="dequant_q4km",
    source_sha=None,
    expected_gguf_sha="__pinned__",
    content_digest="b" * 64,
):
    if source_sha is None:
        source_sha = vbi.EXPECTED_BASE_GGUF_SHA256
    if expected_gguf_sha == "__pinned__":
        expected_gguf_sha = vbi.EXPECTED_BASE_GGUF_SHA256
    dequant_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "base_scheme": base_scheme,
        "source_gguf": {"path": "x.gguf", "sha256": source_sha, "size_bytes": 123},
        "expected_gguf_sha256": expected_gguf_sha,
        "determinism": {"content_digest": content_digest},
    }
    (dequant_dir / "dequant_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_check_dequant_dir_preconditions_happy_path(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir)
    (dequant_dir / "config.json").write_text("{}", encoding="utf-8")
    (dequant_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest, warnings, generation_config = vdp.check_dequant_dir_preconditions(dequant_dir)
    assert manifest["base_scheme"] == "dequant_q4km"
    assert warnings == []
    assert generation_config is None


def test_check_dequant_dir_preconditions_missing_manifest_raises(tmp_path):
    dequant_dir = tmp_path / "dequant"
    dequant_dir.mkdir()
    with pytest.raises(vdp.ManifestError, match="no dequant_manifest.json"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


def test_check_dequant_dir_preconditions_invalid_json_raises(tmp_path):
    dequant_dir = tmp_path / "dequant"
    dequant_dir.mkdir()
    (dequant_dir / "dequant_manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(vdp.ManifestError, match="invalid JSON"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


def test_check_dequant_dir_preconditions_wrong_base_scheme_raises(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir, base_scheme="fp16")
    (dequant_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(vdp.ManifestError, match="base_scheme='fp16'"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


def test_check_dequant_dir_preconditions_missing_config_json_raises(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir)
    with pytest.raises(vdp.ManifestError, match="no config.json"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


def test_check_dequant_dir_preconditions_missing_tokenizer_always_hard_fails(tmp_path):
    # Post-pivot: every --mode exercises the HF side, so there is no more
    # "warn and defer" mode for a missing tokenizer.
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir)
    (dequant_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(vdp.ManifestError, match="no tokenizer files found"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


# --- identity binding -------------------------------------------------------


def test_check_dequant_dir_preconditions_identity_binding_source_sha_mismatch(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir, source_sha="c" * 64, expected_gguf_sha=None)
    (dequant_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(vdp.ManifestError, match="source_gguf.sha256"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


def test_check_dequant_dir_preconditions_identity_binding_expected_gguf_sha_mismatch(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir, source_sha=vbi.EXPECTED_BASE_GGUF_SHA256, expected_gguf_sha="d" * 64)
    (dequant_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(vdp.ManifestError, match="expected_gguf_sha256"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


def test_check_dequant_dir_preconditions_identity_binding_null_expected_is_fine(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir, source_sha=vbi.EXPECTED_BASE_GGUF_SHA256, expected_gguf_sha=None)
    (dequant_dir / "config.json").write_text("{}", encoding="utf-8")
    (dequant_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest, warnings, generation_config = vdp.check_dequant_dir_preconditions(dequant_dir)
    assert manifest["source_gguf"]["sha256"] == vbi.EXPECTED_BASE_GGUF_SHA256


# --- generation_config.json inspection + eos normalization ------------------


def test_check_dequant_dir_preconditions_generation_config_sampler_params_warns(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir)
    (dequant_dir / "config.json").write_text("{}", encoding="utf-8")
    (dequant_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (dequant_dir / "generation_config.json").write_text(
        json.dumps({"temperature": 0.7, "top_p": 0.9}), encoding="utf-8"
    )
    manifest, warnings, generation_config = vdp.check_dequant_dir_preconditions(dequant_dir)
    assert generation_config == {"temperature": 0.7, "top_p": 0.9}
    assert any("sampler default" in w for w in warnings)


def test_check_dequant_dir_preconditions_eos_int_vs_list_normalization_is_not_a_mismatch(tmp_path):
    # The canonical Qwen3 sidecar: generation_config eos_token_id is a LIST
    # ([151645]) while config.json carries a bare int -- must NOT hard-fail.
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir)
    (dequant_dir / "config.json").write_text(json.dumps({"eos_token_id": 151645}), encoding="utf-8")
    (dequant_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (dequant_dir / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [151645]}), encoding="utf-8"
    )
    manifest, warnings, generation_config = vdp.check_dequant_dir_preconditions(dequant_dir)
    assert warnings == []


def test_check_dequant_dir_preconditions_eos_genuine_mismatch_still_hard_fails(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir)
    (dequant_dir / "config.json").write_text(json.dumps({"eos_token_id": 1}), encoding="utf-8")
    (dequant_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (dequant_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 3]}), encoding="utf-8")
    with pytest.raises(vdp.ManifestError, match="eos_token_id"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


def test_check_dequant_dir_preconditions_generation_config_invalid_json_raises(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir)
    (dequant_dir / "config.json").write_text("{}", encoding="utf-8")
    (dequant_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (dequant_dir / "generation_config.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(vdp.ManifestError, match="invalid JSON"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


def test_check_dequant_dir_preconditions_malformed_config_json_with_generation_config_present_raises_manifest_error(tmp_path):
    # major #7: previously escaped as a raw json.JSONDecodeError.
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir)
    (dequant_dir / "config.json").write_text("{not valid json", encoding="utf-8")
    (dequant_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (dequant_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}), encoding="utf-8")
    with pytest.raises(vdp.ManifestError, match="invalid JSON"):
        vdp.check_dequant_dir_preconditions(dequant_dir)


# ============================================================================
# _normalize_token_ids
# ============================================================================


def test_normalize_token_ids_none():
    assert vdp._normalize_token_ids(None) == set()


def test_normalize_token_ids_int():
    assert vdp._normalize_token_ids(151645) == {151645}


def test_normalize_token_ids_list():
    assert vdp._normalize_token_ids([151645]) == {151645}


def test_normalize_token_ids_multiple():
    assert vdp._normalize_token_ids([1, 2, 3]) == {1, 2, 3}


# ============================================================================
# --report writing + out/-refusal
# ============================================================================


def test_assert_report_path_allowed_ok(tmp_path):
    vdp.assert_report_path_allowed(tmp_path / "parity_report.json")


def test_assert_report_path_allowed_refuses_out_component(tmp_path):
    with pytest.raises(vdp.ReportPathError, match="out/"):
        vdp.assert_report_path_allowed(tmp_path / "out" / "parity_report.json")


def test_assert_report_path_allowed_does_not_false_positive_on_output(tmp_path):
    vdp.assert_report_path_allowed(tmp_path / "output_dir" / "parity_report.json")


def _raw_entry(match, **overrides):
    base = {
        "match": match,
        "classification": "match" if match else "content_divergence",
        "first_divergence": None if match else {"char_index": 1, "server_context": "a", "hf_context": "b"},
        "server_text": "x", "hf_text": "x" if match else "y",
        "token_divergence": None,
        "server_finish_reason": "eos", "server_model": "m",
        "server_hit_ceiling": False, "hf_hit_ceiling": False, "hf_n_new_tokens": 3,
    }
    base.update(overrides)
    return base


def _chat_entry(match, **overrides):
    base = {
        "match": match,
        "classification": "match" if match else "content_divergence",
        "first_divergence": None if match else {"char_index": 1, "server_context": "a", "hf_context": "b"},
        "server_content": "x", "server_reasoning_content": None, "server_reconstructed_text": "x",
        "hf_text": "x" if match else "y", "hf_canonical": "x" if match else "y",
        "had_think_block": False, "token_divergence": None,
        "server_finish_reason": "stop", "server_model": "m",
        "server_hit_ceiling": False, "hf_hit_ceiling": False, "hf_n_new_tokens": 3,
    }
    base.update(overrides)
    return base


def _result(uid, raw=None, chat=None):
    return {"uid": uid, "prompt_sha256": "sha", "raw": raw, "chat": chat}


def test_build_report_payload_raw_only_shape():
    manifest = {
        "determinism": {"content_digest": "digestvalue"},
        "source_gguf": {"sha256": "shavalue"},
    }
    results = [
        _result("u1", raw=_raw_entry(True)),
        _result("u2", raw=_raw_entry(False, server_hit_ceiling=True, hf_hit_ceiling=True)),
    ]
    payload = vdp.build_report_payload(
        results=results, manifest=manifest, dequant_dir="DEQ", server_url="URL",
        max_new_tokens=2048, mode="raw", expected_alias="qwen3-8b-q4km-base",
        environment={"device": "cpu", "dtype": "float32"},
    )
    assert payload["verdict"] == "FAIL"
    assert payload["settings"]["mode"] == "raw"
    assert payload["settings"]["manifest_content_digest"] == "digestvalue"
    assert payload["settings"]["source_gguf_sha256"] == "shavalue"
    assert payload["summary"]["raw"] == {"n_prompts": 2, "n_match": 1, "n_mismatch": 1, "ceiling_hit_count": 1}
    assert payload["summary"]["chat"] is None
    assert payload["prompts"][0]["raw"]["match"] is True
    assert payload["prompts"][0]["chat"] is None


def test_build_report_payload_chat_only_never_fails_verdict():
    results = [
        _result("u1", chat=_chat_entry(False, classification="content_divergence")),
    ]
    payload = vdp.build_report_payload(
        results=results, manifest={}, dequant_dir="D", server_url="U",
        max_new_tokens=10, mode="chat", expected_alias="",
        environment={},
    )
    assert payload["verdict"] == "PASS"  # chat-only NEVER gates
    assert payload["summary"]["raw"] is None
    assert payload["summary"]["chat"]["n_mismatch"] == 1
    assert payload["prompts"][0]["chat"]["server_content"] == "x"  # major #2: raw content exposed


def test_build_report_payload_both_mode_exposes_raw_and_chat():
    results = [_result("u1", raw=_raw_entry(True), chat=_chat_entry(False, classification="channel_split_artifact"))]
    payload = vdp.build_report_payload(
        results=results, manifest={}, dequant_dir="D", server_url="U",
        max_new_tokens=10, mode="both", expected_alias="a",
        environment={},
    )
    assert payload["verdict"] == "PASS"  # raw all matched; chat's mismatch doesn't count
    assert payload["prompts"][0]["raw"]["match"] is True
    assert payload["prompts"][0]["chat"]["classification"] == "channel_split_artifact"


def test_build_report_payload_exposes_server_content_and_reasoning_content():
    # major #2 direct check.
    results = [_result("u1", chat=_chat_entry(True, server_content="ANSWER", server_reasoning_content="thinking"))]
    payload = vdp.build_report_payload(
        results=results, manifest={}, dequant_dir="D", server_url="U",
        max_new_tokens=10, mode="chat", expected_alias="",
        environment={},
    )
    prompt_report = payload["prompts"][0]["chat"]
    assert prompt_report["server_content"] == "ANSWER"
    assert prompt_report["server_reasoning_content"] == "thinking"


def test_build_report_payload_empty_results_raises():
    with pytest.raises(ValueError, match="empty results"):
        vdp.build_report_payload(
            results=[], manifest={}, dequant_dir="D", server_url="U",
            max_new_tokens=10, mode="raw", expected_alias="a",
            environment={},
        )


def test_write_report_writes_file(tmp_path):
    path = tmp_path / "parity_report.json"
    vdp.write_report(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_write_report_refuses_out_path_and_does_not_write(tmp_path):
    path = tmp_path / "out" / "parity_report.json"
    with pytest.raises(vdp.ReportPathError):
        vdp.write_report(path, {"a": 1})
    assert not path.exists()


# ============================================================================
# --mode raw rendering path (injected fake tokenizer)
# ============================================================================


class _FakeChatTemplateTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.calls.append({
            "messages": messages, "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        })
        return "RENDERED::" + json.dumps(messages)


def test_render_raw_prompt_uses_wire_pins_and_fake_tokenizer():
    tok = _FakeChatTemplateTokenizer()
    result = vdp.render_raw_prompt(tok, "STATEMENT")
    assert len(tok.calls) == 1
    call = tok.calls[0]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert call["messages"] == [
        {"role": "system", "content": config.PASS_AT_K_SYSTEM_PROMPT},
        {"role": "user", "content": "STATEMENT" + config.PASS_AT_K_NO_THINK_SUFFIX},
    ]
    assert result.startswith("RENDERED::")


# ============================================================================
# HF-side generation helpers (fake torch/model/tokenizer -- no real transformers)
# ============================================================================


class _NoGradCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeTorch:
    def no_grad(self):
        return _NoGradCtx()


class _FakeInputIds:
    def __init__(self, prompt_len):
        self.shape = (1, prompt_len)
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


def test_generate_and_decode_moves_input_ids_to_device():
    class _FakeModel:
        def generate(self, input_ids, max_new_tokens, do_sample):
            return [[10, 11, 12, 42, 43]]

    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens):
            return f"DECODED:{list(ids)}"

    input_ids = _FakeInputIds(prompt_len=3)
    result = vdp._generate_and_decode(_FakeModel(), _FakeTokenizer(), input_ids, 2, _FakeTorch(), "mps")
    assert input_ids.to_calls == ["mps"]
    assert result["text"] == "DECODED:[42, 43]"
    assert result["n_new_tokens"] == 2
    assert result["hit_ceiling"] is True


def test_generate_and_decode_hit_ceiling_false_when_stopped_early():
    class _FakeModel:
        def generate(self, input_ids, max_new_tokens, do_sample):
            return [[10, 11, 12, 42]]

    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens):
            return "DECODED"

    result = vdp._generate_and_decode(_FakeModel(), _FakeTokenizer(), _FakeInputIds(prompt_len=3), 5, _FakeTorch(), "cpu")
    assert result["hit_ceiling"] is False
    assert result["n_new_tokens"] == 1


def test_generate_and_decode_excludes_terminating_eos_from_ceiling_int():
    class _FakeModel:
        def generate(self, input_ids, max_new_tokens, do_sample):
            return [[10, 11, 12, 42, 43, 999]]  # 3 completion tokens, last=EOS

    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens):
            return "DECODED"

    result = vdp._generate_and_decode(
        _FakeModel(), _FakeTokenizer(), _FakeInputIds(prompt_len=3), 3, _FakeTorch(), "cpu", eos_token_id=999
    )
    assert result["n_new_tokens"] == 3
    assert result["hit_ceiling"] is False  # natural stop, not truncation


def test_generate_and_decode_excludes_terminating_eos_from_ceiling_list():
    class _FakeModel:
        def generate(self, input_ids, max_new_tokens, do_sample):
            return [[10, 11, 12, 42, 43, 151645]]

    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens):
            return "DECODED"

    result = vdp._generate_and_decode(
        _FakeModel(), _FakeTokenizer(), _FakeInputIds(prompt_len=3), 3, _FakeTorch(), "cpu", eos_token_id=[151645]
    )
    assert result["hit_ceiling"] is False


def test_generate_and_decode_hit_ceiling_true_when_last_token_not_eos():
    class _FakeModel:
        def generate(self, input_ids, max_new_tokens, do_sample):
            return [[10, 11, 12, 42, 43, 44]]

    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens):
            return "DECODED"

    result = vdp._generate_and_decode(
        _FakeModel(), _FakeTokenizer(), _FakeInputIds(prompt_len=3), 3, _FakeTorch(), "cpu", eos_token_id=999
    )
    assert result["hit_ceiling"] is True


def test_generate_hf_raw_uses_add_special_tokens_false():
    captured = {}

    class _FakeTok:
        def __call__(self, text, **kwargs):
            captured["text"] = text
            captured.update(kwargs)
            return {"input_ids": _FakeInputIds(3)}

        def decode(self, ids, skip_special_tokens):
            return "TEXT"

    class _FakeModel:
        def generate(self, input_ids, max_new_tokens, do_sample):
            return [[0, 0, 0, 1]]

    result = vdp.generate_hf_raw(_FakeModel(), _FakeTok(), "RAW", 5, _FakeTorch(), "cpu")
    assert captured["add_special_tokens"] is False
    assert captured["return_tensors"] == "pt"
    assert result["text"] == "TEXT"


# ============================================================================
# base_serve_command (pure argv builder, no --lora, no subprocess)
# ============================================================================


def test_base_serve_command_shape():
    cmd = vdp.base_serve_command("/path/to/base.gguf", 8081)
    assert cmd == [
        "llama-server", "-m", "/path/to/base.gguf",
        "--alias", vdp.DEFAULT_EXPECTED_ALIAS,
        "-c", "8192", "-ngl", "99", "--parallel", "1", "--port", "8081",
    ]
    assert "--lora" not in cmd


def test_base_serve_command_custom_alias():
    cmd = vdp.base_serve_command("/x.gguf", 9000, alias="custom-alias")
    assert cmd[cmd.index("--alias") + 1] == "custom-alias"


# ============================================================================
# HTTP calls (injected opener -- no real network)
# ============================================================================


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _chat_body(content="ANSWER", finish_reason="stop", model=None, reasoning_content=None):
    message = {"content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "model": model if model is not None else vdp.DEFAULT_EXPECTED_ALIAS,
    }


def _native_body(content="ANSWER", stop_type="eos", model=None):
    return {"content": content, "stop_type": stop_type, "model": model if model is not None else vdp.DEFAULT_EXPECTED_ALIAS}


def test_call_chat_completion_happy_path_full_payload_equality():
    captured = {}

    def fake_opener(request, timeout):
        captured["url"] = request.full_url
        captured["data"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps(_chat_body()).encode("utf-8"))

    result = vdp.call_chat_completion(
        "http://example.invalid/v1/chat/completions", "STATEMENT", 128, opener=fake_opener
    )
    assert result["text"] == "ANSWER"
    assert result["finish_reason"] == "stop"
    assert result["model"] == vdp.DEFAULT_EXPECTED_ALIAS
    assert result["reasoning_content"] is None
    assert captured["data"] == {
        "model": vdp.DEFAULT_EXPECTED_ALIAS,
        "messages": [
            {"role": "system", "content": config.PASS_AT_K_SYSTEM_PROMPT},
            {"role": "user", "content": "STATEMENT" + config.PASS_AT_K_NO_THINK_SUFFIX},
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
    }


def test_call_chat_completion_reconstructs_reasoning_content():
    def fake_opener(request, timeout):
        return _FakeResponse(json.dumps(_chat_body(reasoning_content="thinking...")).encode("utf-8"))

    result = vdp.call_chat_completion("http://example.invalid/v1/chat/completions", "S", 10, opener=fake_opener)
    assert result["text"] == "<think>thinking...</think>\n\nANSWER"
    assert result["content"] == "ANSWER"
    assert result["reasoning_content"] == "thinking..."


def test_call_chat_completion_null_content_raises_clean_error():
    def fake_opener(request, timeout):
        body = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}], "model": "x"}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    with pytest.raises(vdp.TransportError, match="message.content is null"):
        vdp.call_chat_completion("http://example.invalid/v1/chat/completions", "S", 10, opener=fake_opener)


def test_call_chat_completion_url_error_raises_transport_error():
    def fake_opener(request, timeout):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(vdp.TransportError, match="request to"):
        vdp.call_chat_completion("http://example.invalid/v1/chat/completions", "S", 10, opener=fake_opener)


def test_call_chat_completion_bad_shape_raises_transport_error():
    def fake_opener(request, timeout):
        return _FakeResponse(json.dumps({"unexpected": True}).encode("utf-8"))

    with pytest.raises(vdp.TransportError, match="unexpected chat-completions response shape"):
        vdp.call_chat_completion("http://example.invalid/v1/chat/completions", "S", 10, opener=fake_opener)


def test_call_chat_completion_alias_mismatch_hard_fails():
    def fake_opener(request, timeout):
        return _FakeResponse(json.dumps(_chat_body(model="qwen3-8b-q4km-lora-s3")).encode("utf-8"))

    with pytest.raises(vdp.TransportError, match="response model"):
        vdp.call_chat_completion(
            "http://example.invalid/v1/chat/completions", "S", 10,
            expected_alias="qwen3-8b-q4km-base", opener=fake_opener,
        )


def test_call_chat_completion_alias_check_skipped_when_expected_alias_falsy():
    def fake_opener(request, timeout):
        return _FakeResponse(json.dumps(_chat_body(model="anything-else")).encode("utf-8"))

    result = vdp.call_chat_completion(
        "http://example.invalid/v1/chat/completions", "S", 10,
        expected_alias="", opener=fake_opener,
    )
    assert result["model"] == "anything-else"


def test_call_native_completion_happy_path_and_full_payload_equality():
    captured = {}

    def fake_opener(request, timeout):
        captured["data"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(json.dumps(_native_body(content="RAW_ANSWER")).encode("utf-8"))

    result = vdp.call_native_completion("http://example.invalid/completion", "RAW PROMPT", 64, opener=fake_opener)
    assert result["text"] == "RAW_ANSWER"
    assert result["content"] == "RAW_ANSWER"
    assert result["stop_type"] == "eos"
    assert captured["data"] == {
        "prompt": "RAW PROMPT", "temperature": 0.0, "n_predict": 64, "stream": False,
    }


def test_call_native_completion_bad_shape_raises_transport_error():
    def fake_opener(request, timeout):
        return _FakeResponse(json.dumps({}).encode("utf-8"))

    with pytest.raises(vdp.TransportError, match="unexpected native-completion response shape"):
        vdp.call_native_completion("http://example.invalid/completion", "RAW", 10, opener=fake_opener)


def test_call_native_completion_alias_mismatch_hard_fails():
    def fake_opener(request, timeout):
        return _FakeResponse(json.dumps(_native_body(model="wrong")).encode("utf-8"))

    with pytest.raises(vdp.TransportError, match="response model"):
        vdp.call_native_completion(
            "http://example.invalid/completion", "RAW", 10,
            expected_alias="qwen3-8b-q4km-base", opener=fake_opener,
        )


def test_derive_native_completion_url_no_prefix():
    result = vdp.derive_native_completion_url("http://example.invalid:8081/v1/chat/completions")
    assert result == "http://example.invalid:8081/completion"


def test_derive_native_completion_url_preserves_prefix():
    result = vdp.derive_native_completion_url("http://example.invalid:8081/myprefix/v1/chat/completions")
    assert result == "http://example.invalid:8081/myprefix/completion"


# ============================================================================
# Lazy-import error message (forced via sys.modules injection)
# ============================================================================


def test_lazy_import_raises_actionable_error_when_torch_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(vdp.DependencyError) as exc_info:
        vdp._lazy_import_hf_stack()
    message = str(exc_info.value)
    assert "'torch' package is not importable" in message
    assert "'transformers' package is not importable" not in message


def test_lazy_import_raises_actionable_error_when_transformers_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(vdp.DependencyError) as exc_info:
        vdp._lazy_import_hf_stack()
    message = str(exc_info.value)
    assert "'transformers' package is not importable" in message
    assert "'torch' package is not importable" not in message


# ============================================================================
# CLI argument parsing
# ============================================================================


def test_server_url_is_required(tmp_path, capsys):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"uid": "u1", "statement": "s1"}) + "\n", encoding="utf-8")
    parser = vdp.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dequant-dir", str(tmp_path), "--prompts-file", str(prompts)])
    err = capsys.readouterr().err
    assert "--server-url" in err


def test_server_url_has_no_default():
    parser = vdp.build_arg_parser()
    action = next(a for a in parser._actions if a.dest == "server_url")
    assert action.default is None
    assert action.required is True


def test_prompt_source_required(tmp_path):
    parser = vdp.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--dequant-dir", str(tmp_path),
            "--server-url", "http://example.invalid/v1/chat/completions",
        ])


def test_prompt_source_mutually_exclusive(tmp_path):
    parser = vdp.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--dequant-dir", str(tmp_path),
            "--server-url", "http://example.invalid/v1/chat/completions",
            "--prompts-file", "a.jsonl",
            "--from-eval-set", "b.jsonl",
        ])


def test_defaults():
    parser = vdp.build_arg_parser()
    args = parser.parse_args([
        "--dequant-dir", "d",
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", "p.jsonl",
    ])
    assert args.max_new_tokens == vdp.DEFAULT_MAX_NEW_TOKENS
    assert vdp.DEFAULT_MAX_NEW_TOKENS == 2048
    assert args.report is None
    assert args.mode == "raw"
    assert vdp.DEFAULT_MODE == "raw"
    assert args.i_own_the_qwen_slot is False
    assert args.expected_alias == vdp.DEFAULT_EXPECTED_ALIAS
    assert args.device == vdp.DEFAULT_DEVICE


def test_mode_rejects_invalid_choice(tmp_path):
    parser = vdp.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--dequant-dir", str(tmp_path),
            "--server-url", "http://example.invalid/v1/chat/completions",
            "--prompts-file", "p.jsonl",
            "--mode", "raw-completion",
        ])


def test_expected_alias_flag_aliasing():
    parser = vdp.build_arg_parser()
    args = parser.parse_args([
        "--dequant-dir", "d",
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", "p.jsonl",
        "--model-alias", "custom",
    ])
    assert args.expected_alias == "custom"


def test_allow_abbrev_disabled_rejects_ambiguous_short_flag(tmp_path):
    parser = vdp.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--dequant-dir", str(tmp_path),
            "--server-url", "http://example.invalid/v1/chat/completions",
            "--prompts-file", "p.jsonl",
            "--i",
        ])


# ============================================================================
# Docstring/epilog contract sanity (minors #6, #11)
# ============================================================================


def test_exit_code_docstring_does_not_restrict_exit_1_to_content_divergence():
    doc = vdp.__doc__
    assert "any classification other than" in doc or "ANY classification other than" in doc
    assert "at least one content_divergence mismatch" not in doc


def test_epilog_does_not_restrict_exit_1_to_content_divergence():
    assert "ANY classification" in vdp._EPILOG


def test_docstring_does_not_claim_repeated_verbatim():
    assert "repeated verbatim" not in vdp.__doc__


# ============================================================================
# main() end-to-end -- fake HF stack
# ============================================================================


def _write_dequant_dir(tmp_path):
    dequant_dir = tmp_path / "dequant"
    _write_manifest(dequant_dir)
    (dequant_dir / "config.json").write_text(json.dumps({"eos_token_id": 999}), encoding="utf-8")
    (dequant_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return dequant_dir


class _E2EFakeTorch:
    __version__ = "2.99.0"

    def no_grad(self):
        return _NoGradCtx()

    @property
    def float32(self):
        return "float32-sentinel"


class _E2EFakeModel:
    def __init__(self, completion_ids):
        self.completion_ids = completion_ids
        self.to_calls = []
        self.eval_called = False

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, input_ids, max_new_tokens, do_sample):
        assert do_sample is False
        prompt_len = input_ids.shape[-1]
        return [[0] * prompt_len + list(self.completion_ids)]


class _E2EFakeModelCls:
    def __init__(self, completion_ids):
        self._completion_ids = completion_ids
        self.from_pretrained_kwargs = None
        self.created = None

    def from_pretrained(self, path, **kwargs):
        self.from_pretrained_kwargs = kwargs
        self.created = _E2EFakeModel(self._completion_ids)
        return self.created


class _E2EFakeTokenizerInstance:
    def __init__(self, decoded_text, prompt_len=4):
        self.decoded_text = decoded_text
        self.prompt_len = prompt_len
        self.eos_token_id = 999
        self.pad_token_id = 998
        self.last_input_ids = None

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors=None, **kw):
        if not tokenize:
            return "RAW::" + json.dumps(messages)
        self.last_input_ids = _FakeInputIds(self.prompt_len)
        return self.last_input_ids

    def __call__(self, text, **kw):
        self.last_input_ids = _FakeInputIds(self.prompt_len)
        return {"input_ids": self.last_input_ids}

    def decode(self, ids, skip_special_tokens=True):
        return self.decoded_text

    def encode(self, text, add_special_tokens=False):
        return [len(text)]


class _E2EFakeTokenizerCls:
    def __init__(self, decoded_text):
        self._decoded_text = decoded_text
        self.from_pretrained_kwargs = None
        self.created = None

    def from_pretrained(self, path, **kwargs):
        self.from_pretrained_kwargs = kwargs
        self.created = _E2EFakeTokenizerInstance(self._decoded_text)
        return self.created


def _fake_hf_stack(model_cls, tok_cls, transformers_version="4.99.0"):
    torch_mod = _E2EFakeTorch()
    return lambda: (torch_mod, model_cls, tok_cls, transformers_version)


def _write_prompts(path, pairs):
    path.write_text(
        "\n".join(json.dumps({"uid": uid, "statement": statement}) for uid, statement in pairs) + "\n",
        encoding="utf-8",
    )


def _make_native_opener(content="ANSWER", stop_type="eos", model=None):
    def fake_opener(request, timeout):
        return _FakeResponse(json.dumps(_native_body(content=content, stop_type=stop_type, model=model)).encode("utf-8"))
    return fake_opener


def _make_mode_aware_opener(*, native_content="ANSWER", native_stop_type="eos", native_model=None,
                             chat_content="ANSWER", chat_finish_reason="stop", chat_model=None,
                             chat_reasoning_content=None):
    def fake_opener(request, timeout):
        if request.full_url.endswith("/completion"):
            body = _native_body(content=native_content, stop_type=native_stop_type, model=native_model)
        else:
            body = _chat_body(
                content=chat_content, finish_reason=chat_finish_reason,
                model=chat_model, reasoning_content=chat_reasoning_content,
            )
        return _FakeResponse(json.dumps(body).encode("utf-8"))
    return fake_opener


# --- raw mode (default) is THE GATE -----------------------------------------


def test_main_end_to_end_raw_mode_default_all_match_exits_zero_and_writes_pass_report(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1"), ("u2", "S2")])

    monkeypatch.setattr(vdp, "_default_opener", _make_native_opener(content="ANSWER"))
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(_E2EFakeModelCls([1, 2, 3]), _E2EFakeTokenizerCls("ANSWER")))

    report_path = tmp_path / "report.json"
    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
        "--report", str(report_path),
    ])
    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["settings"]["mode"] == "raw"
    assert payload["summary"]["raw"]["n_match"] == 2
    assert payload["summary"]["chat"] is None
    assert all(p["raw"]["match"] for p in payload["prompts"])
    assert payload["environment"]["torch_version"] == "2.99.0"
    assert payload["environment"]["device"] == "cpu"
    assert payload["environment"]["attn_implementation"] == "eager"


def test_main_end_to_end_raw_mode_one_divergence_exits_one_and_writes_fail_report(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1"), ("u2", "S2")])

    monkeypatch.setattr(vdp, "_default_opener", _make_native_opener(content="SERVER_ANSWER"))
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(_E2EFakeModelCls([1, 2, 3]), _E2EFakeTokenizerCls("DIFFERENT_ANSWER")))

    report_path = tmp_path / "report.json"
    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
        "--report", str(report_path),
    ])
    assert rc == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "FAIL"
    for p in payload["prompts"]:
        assert p["raw"]["match"] is False
        assert p["raw"]["classification"] == "content_divergence"
        assert p["raw"]["token_divergence"] is not None


def test_main_end_to_end_raw_mode_mixed_match_exits_one_not_masked_by_any(tmp_path, monkeypatch):
    # Guards specifically against an `all(...)` -> `any(...)` mutation on the
    # gate computation: with one matching and one mismatching raw prompt,
    # `any()` would wrongly report PASS/exit 0 (verified live: this test
    # fails under that exact mutation, while the all-mismatch/single-prompt
    # tests above do not distinguish all() from any()).
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "STATEMENT_ONE"), ("u2", "STATEMENT_TWO")])

    def fake_opener(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        raw_prompt = payload["prompt"]
        content = "MATCHING" if "STATEMENT_ONE" in raw_prompt else "SERVER_TWO"
        return _FakeResponse(json.dumps(_native_body(content=content)).encode("utf-8"))

    monkeypatch.setattr(vdp, "_default_opener", fake_opener)

    class _FakeModelSeq:
        def __init__(self):
            self.calls = 0

        def to(self, device):
            return self

        def eval(self):
            pass

        def generate(self, input_ids, max_new_tokens, do_sample):
            self.calls += 1
            prompt_len = input_ids.shape[-1]
            ids = [1] if self.calls == 1 else [2]
            return [[0] * prompt_len + ids]

    class _FakeModelClsSeq:
        def from_pretrained(self, path, **kwargs):
            self.created = _FakeModelSeq()
            return self.created

    class _FakeTokSeq(_E2EFakeTokenizerInstance):
        def __init__(self):
            super().__init__("MATCHING")
            self.decode_calls = 0

        def decode(self, ids, skip_special_tokens=True):
            self.decode_calls += 1
            return "MATCHING" if self.decode_calls == 1 else "HF_TWO_DIFFERENT"

    class _FakeTokClsSeq:
        def from_pretrained(self, path, **kwargs):
            self.created = _FakeTokSeq()
            return self.created

    monkeypatch.setattr(
        vdp, "_lazy_import_hf_stack",
        lambda: (_E2EFakeTorch(), _FakeModelClsSeq(), _FakeTokClsSeq(), "4.99.0"),
    )

    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
    ])
    assert rc == 1  # prompt 2 mismatched -- ANY mismatch must gate, all() not any()


def test_main_end_to_end_from_eval_set_dispatches_to_anchor_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    eval_set_path = tmp_path / "eval_set.jsonl"
    _write_eval_set(eval_set_path, n_anchor_solved=10, n_anchor_fail=10, n_eval_band=5)

    monkeypatch.setattr(vdp, "_default_opener", _make_native_opener(content="ANSWER"))
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(_E2EFakeModelCls([1]), _E2EFakeTokenizerCls("ANSWER")))

    report_path = tmp_path / "report.json"
    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--from-eval-set", str(eval_set_path),
        "--report", str(report_path),
    ])
    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["raw"]["n_prompts"] == 10
    uids = {p["uid"] for p in payload["prompts"]}
    assert all(u.startswith("anchor_solved_") for u in uids)


def test_main_rechecks_guard_before_each_server_call(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_guard(*a, **kw):
        calls["n"] += 1
        return [] if calls["n"] <= 2 else ["eval_all.sh"]

    monkeypatch.setattr(vdp, "campaign_liveness_hits", fake_guard)

    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1"), ("u2", "S2")])

    opener_calls = {"n": 0}

    def fake_opener(request, timeout):
        opener_calls["n"] += 1
        return _FakeResponse(json.dumps(_native_body()).encode("utf-8"))

    monkeypatch.setattr(vdp, "_default_opener", fake_opener)
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(_E2EFakeModelCls([1]), _E2EFakeTokenizerCls("ANSWER")))

    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
    ])
    assert rc == 2
    assert opener_calls["n"] == 1  # only prompt 1's HTTP call happened


def test_main_moves_input_ids_and_model_to_device(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])

    monkeypatch.setattr(vdp, "_default_opener", _make_native_opener())
    model_cls = _E2EFakeModelCls([1])
    tok_cls = _E2EFakeTokenizerCls("ANSWER")
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(model_cls, tok_cls))

    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
        "--device", "mps",
    ])
    assert rc == 0
    assert model_cls.created.to_calls == ["mps"]
    assert tok_cls.created.last_input_ids.to_calls == ["mps"]


def test_main_end_to_end_alias_mismatch_refuses_with_exit_2(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])

    monkeypatch.setattr(vdp, "_default_opener", _make_native_opener(model="qwen3-8b-q4km-lora-s3"))
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(_E2EFakeModelCls([1]), _E2EFakeTokenizerCls("ANSWER")))

    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
    ])
    assert rc == 2


def test_main_scheme_less_server_url_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])
    rc = vdp.main([
        "--dequant-dir", str(tmp_path / "dequant"),
        "--server-url", "example.invalid:8081/v1/chat/completions",
        "--prompts-file", str(prompts_path),
    ])
    assert rc == 2


def test_main_missing_prompts_file_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    rc = vdp.main([
        "--dequant-dir", str(tmp_path / "dequant"),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(tmp_path / "does_not_exist.jsonl"),
    ])
    assert rc == 2


def test_main_max_new_tokens_zero_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])
    rc = vdp.main([
        "--dequant-dir", str(tmp_path / "dequant"),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
        "--max-new-tokens", "0",
    ])
    assert rc == 2
    assert "max-new-tokens" in capsys.readouterr().err


def test_main_from_pretrained_failure_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])

    class _BoomModelCls:
        def from_pretrained(self, path, **kwargs):
            raise OSError("could not find model weights")

    monkeypatch.setattr(
        vdp, "_lazy_import_hf_stack",
        lambda: (_E2EFakeTorch(), _BoomModelCls(), _E2EFakeTokenizerCls("ANSWER"), "4.99.0"),
    )
    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
    ])
    assert rc == 2


def test_main_all_empty_completions_exits_2_not_vacuous_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])

    monkeypatch.setattr(vdp, "_default_opener", _make_native_opener(content=""))
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(_E2EFakeModelCls([]), _E2EFakeTokenizerCls("")))

    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
    ])
    assert rc == 2


def test_main_report_under_out_dir_refused_before_any_other_work(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    rc = vdp.main([
        "--dequant-dir", str(tmp_path / "dequant"),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(tmp_path / "does_not_exist.jsonl"),
        "--report", str(tmp_path / "out" / "parity_report.json"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "out/" in err


def test_main_manifest_precondition_failure_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts, [("u1", "S1")])

    rc = vdp.main([
        "--dequant-dir", str(tmp_path / "no_such_dequant_dir"),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no dequant_manifest.json" in err


def test_main_refuses_when_campaign_live(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: ["eval_all.sh"])
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts, [("u1", "S1")])

    rc = vdp.main([
        "--dequant-dir", str(tmp_path / "dequant"),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "eval_all.sh" in err
    assert "REFUSED" in err


def test_main_override_flag_bypasses_guard_but_still_fails_downstream(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: ["eval_all.sh"])
    empty_prompts = tmp_path / "empty.jsonl"
    empty_prompts.write_text("\n", encoding="utf-8")

    rc = vdp.main([
        "--dequant-dir", str(tmp_path / "dequant"),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(empty_prompts),
        "--i-own-the-qwen-slot",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "eval_all.sh" not in err
    assert "no usable prompts" in err


# --- --mode chat: NEVER gates ------------------------------------------------


def test_main_end_to_end_chat_mode_mismatch_never_fails_gate(tmp_path, monkeypatch):
    # A genuine content_divergence in chat mode must still exit 0 -- chat
    # mode is informational only, per the pivot.
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])

    monkeypatch.setattr(vdp, "_default_opener", _make_mode_aware_opener(chat_content="SERVER_ANSWER"))
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(_E2EFakeModelCls([1]), _E2EFakeTokenizerCls("The answer is 42.")))

    report_path = tmp_path / "report.json"
    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
        "--mode", "chat",
        "--report", str(report_path),
    ])
    assert rc == 0  # chat mode NEVER gates
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["prompts"][0]["chat"]["match"] is False
    assert payload["prompts"][0]["chat"]["classification"] == "content_divergence"
    assert payload["prompts"][0]["raw"] is None


def test_main_end_to_end_chat_mode_empty_think_fixture_does_not_fail_cross_check(tmp_path, monkeypatch):
    # THE exact blocker fixture: every /no_think anchor's HF decode carries
    # the empty '<think>\n\n</think>\n\n' prefix; the server's real content
    # is already extracted. Must be a clean MATCH, not a spurious mismatch.
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])

    monkeypatch.setattr(vdp, "_default_opener", _make_mode_aware_opener(chat_content="The answer is 42."))
    monkeypatch.setattr(
        vdp, "_lazy_import_hf_stack",
        _fake_hf_stack(_E2EFakeModelCls([1]), _E2EFakeTokenizerCls("<think>\n\n</think>\n\nThe answer is 42.")),
    )

    report_path = tmp_path / "report.json"
    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
        "--mode", "chat",
        "--report", str(report_path),
    ])
    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["prompts"][0]["chat"]["match"] is True
    assert payload["prompts"][0]["chat"]["had_think_block"] is True


def test_main_end_to_end_chat_mode_whitespace_loss_fixture_classifies_channel_split_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])

    monkeypatch.setattr(vdp, "_default_opener", _make_mode_aware_opener(chat_content="The answer is 42."))
    monkeypatch.setattr(
        vdp, "_lazy_import_hf_stack",
        _fake_hf_stack(_E2EFakeModelCls([1]), _E2EFakeTokenizerCls("<think>\nreasoning\n</think>\n\n\nThe answer is 42.")),
    )

    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
        "--mode", "chat",
    ])
    assert rc == 0  # still never gates, even for a mismatch


# --- --mode both: raw is the sole gate --------------------------------------


def test_main_end_to_end_both_mode_gate_uses_raw_only_when_chat_mismatches(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])

    # raw: server + HF both say "ANSWER" -> match. chat: HF decode has the
    # empty-think prefix so raw HF text != chat's canonical-stripped text
    # unless we canonicalize -- here we deliberately mismatch chat to prove
    # it doesn't leak into the exit code.
    monkeypatch.setattr(
        vdp, "_default_opener",
        _make_mode_aware_opener(native_content="ANSWER", chat_content="SOMETHING_ELSE"),
    )
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(_E2EFakeModelCls([1]), _E2EFakeTokenizerCls("ANSWER")))

    report_path = tmp_path / "report.json"
    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
        "--mode", "both",
        "--report", str(report_path),
    ])
    assert rc == 0  # raw matched; chat's mismatch is irrelevant to the gate
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["prompts"][0]["raw"]["match"] is True
    assert payload["prompts"][0]["chat"]["match"] is False


def test_main_end_to_end_both_mode_raw_mismatch_gates_regardless_of_chat_match(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp, "campaign_liveness_hits", lambda *a, **kw: [])
    dequant_dir = _write_dequant_dir(tmp_path)
    prompts_path = tmp_path / "prompts.jsonl"
    _write_prompts(prompts_path, [("u1", "S1")])

    # raw: server says "SERVER_TEXT", HF says "ANSWER" (via decode) -> raw mismatch.
    # chat: server content == HF decode == "ANSWER" -> chat match.
    monkeypatch.setattr(
        vdp, "_default_opener",
        _make_mode_aware_opener(native_content="SERVER_TEXT", chat_content="ANSWER"),
    )
    monkeypatch.setattr(vdp, "_lazy_import_hf_stack", _fake_hf_stack(_E2EFakeModelCls([1]), _E2EFakeTokenizerCls("ANSWER")))

    rc = vdp.main([
        "--dequant-dir", str(dequant_dir),
        "--server-url", "http://example.invalid/v1/chat/completions",
        "--prompts-file", str(prompts_path),
        "--mode", "both",
    ])
    assert rc == 1  # raw's mismatch gates regardless of chat's match

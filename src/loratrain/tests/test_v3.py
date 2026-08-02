"""Tests for loratrain.v3 (the proof-as-hint regeneration dataset builder).

Hermetic (README/AGENTS.md convention this suite follows throughout): no
network, no servers, no llama-server subprocess, no icepick import in any
fixture or assertion path -- ``verify_fn`` is always the injected
``fake_verify_fn`` below. The one exception, by design, is
``test_isolation.py``-style subprocess use for the "nothing pre-existing
imports v3" check, which spawns a bare ``python -c`` import probe (no
network, no model, sub-second) -- see its docstring for why an in-process
check cannot prove that property.

Mirrors test_build_dataset.py's ``env`` fixture pattern: build a fully
valid environment under ``tmp_path``, then let individual tests mutate one
piece (rewrite a file, override one kwarg) to exercise a specific refusal
path.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from loratrain import build_dataset, config, v3

ROOT = Path(__file__).resolve().parents[1]  # src/loratrain/
SRC = ROOT / "src"


# ============================================================================
# fixture-building helpers
# ============================================================================


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _corpus_pin(path: Path):
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), data.count(b"\n")


TRAIN_UIDS = ["t0", "t1", "t2", "t3"]
# Shaped like the OLD split's holdout uids -- kept only to prove the
# retirement (split rebuild, 2026-08-01) was done cleanly: a uid that WOULD
# have been holdout under the old split now hits the exact same
# UnknownUidError as any other stranger uid, never a special LeakageError
# path. See test_former_holdout_shaped_uid_now_hits_unknown_uid_hard_fail.
FORMER_HOLDOUT_SHAPED_UIDS = ["h0", "h1"]

# (question, solution_text, answer, arxiv_id) for the 4 base train uids.
SOLUTION_SPECS = {
    "t0": ("Statement zero.", "Derivation zero ends in \\boxed{0}.", "0", "1000.00000"),
    "t1": ("Statement one.", "Derivation one ends in \\boxed{1}.", "1", "1000.00001"),
    "t2": ("Statement two.", "Derivation two ends in \\boxed{2}.", "2", "1000.00002"),
    "t3": ("Statement three.", "Derivation three ends in \\boxed{3}.", "3", "1000.00003"),
}


def _solutions_row(uid, *, question=None, solution_text=None, answer=None, arxiv_id=None):
    if question is None:
        question, solution_text, answer, arxiv_id = SOLUTION_SPECS[uid]
    return {
        "uid": uid,
        "question": question,
        "proof_raw_sha": hashlib.sha256(f"proof-{uid}".encode("utf-8")).hexdigest(),
        "solution_text": solution_text,
        "answer": answer,
        "provenance": {
            "arxiv_id": arxiv_id,
            "match_method": "adjacency",
            "match_confidence": "high",
            "sonnet_cache_key": f"{uid}__cache",
            "verified": True,
        },
    }


def _v2_cap1_row(uid):
    return {
        "prompt": [
            {"role": "system", "content": config.PASS_AT_K_SYSTEM_PROMPT},
            {"role": "user", "content": f"Anchor statement {uid}." + config.PASS_AT_K_NO_THINK_SUFFIX},
        ],
        "completion": [{"role": "assistant", "content": f"Anchor trace for {uid}."}],
        "provenance": {
            "uid": uid, "rollout_uid": f"{uid}-r00", "sample_idx": 0,
            "verdict": "correct", "verbatim_output": True,
        },
    }


def _rollout_row(uid, sample_idx, output):
    return {"uid": uid, "sample_idx": sample_idx, "output": output}


def fake_verify_fn(output, answer):
    """Hermetic stand-in for the real icepick verify chain: an output
    "verifies" iff it contains the literal marker "CORRECT". Ignores
    ``answer`` entirely -- these tests exercise the BUILDER's harvesting
    logic, not real math verification (see ``default_verify_fn`` for the
    real chain, lazily imported and never touched by this suite).
    """
    return isinstance(output, str) and "CORRECT" in output


def _default_rollouts():
    """The happy-path rollouts fixture:

    t0: try0 WRONG, try1 CORRECT (kept), try2 CORRECT (must be ignored --
        first-verified-wins). t0 is band-tier (band_corpus membership).
    t1: try0 CORRECT (kept on try 1). t1 resolves collapse via the pool.
    t2: try0 WRONG, try1 CORRECT (kept on try 2). t2's pool label is
        "misdirection" -- must bucket as collapse.
    t3: try0/try1 WRONG only -- hint_insufficient (dropped + censused).
    """
    return [
        _rollout_row("t0", 0, "WRONG output for t0 try0."),
        _rollout_row("t0", 1, "CORRECT output for t0 try1 (kept)."),
        _rollout_row("t0", 2, "CORRECT output for t0 try2 (must be ignored)."),
        _rollout_row("t1", 0, "CORRECT output for t1 try0 (kept)."),
        _rollout_row("t2", 0, "WRONG output for t2 try0."),
        _rollout_row("t2", 1, "CORRECT output for t2 try1 (kept)."),
        _rollout_row("t3", 0, "WRONG output for t3 try0."),
        _rollout_row("t3", 1, "WRONG output for t3 try1."),
    ]


@pytest.fixture
def env(tmp_path):
    """A fully valid, hermetic v3 environment under ``tmp_path``.

    4 train uids (t0-t3, all with solutions rows) -- the split rebuild
    (2026-08-01) has NO holdout concept, only ``train_side_uids``
    (proof-bearing) vs an eval pool this module never reads; t0 is
    band-tier via band_corpus membership (and ALSO appears in the
    wellposed pool with a CONFLICTING "collapse" label -- band_corpus must
    win, see ``test_tier_resolution_*``); t1/t2 resolve via the pool
    (collapse / misdirection, both -> collapse bucket); a 6-row v2/cap1
    anchor pool with uids disjoint from t0-t3. Individual tests mutate one
    piece (rewrite a file, override one kwarg) to exercise a refusal path.
    """
    split_path = tmp_path / "evalharness" / "data" / "corpus_split_v3_proofsplit_20260801.json"
    _write_json(split_path, {
        "ruling": "fixture -- mirrors the real split's proof-bearing/proofless schema, no holdout",
        "train_side_uids": list(TRAIN_UIDS),
        # v3b anchor_solved (2abe292): eval-side membership, shaped like the
        # real split (eval_set_uids keyed by tier, papers.eval_papers a flat
        # list) -- aligned with the ONE eval_set.jsonl fixture row below
        # (uid "eval-u0", arxiv_id "9999.00001") so the two fixtures agree.
        "eval_set_uids": {"band": ["eval-u0"], "collapse": [], "misdirection": []},
        "papers": {"eval_papers": ["9999.00001"]},
    })
    split_sha256 = _sha256(split_path)

    eval_set_path = tmp_path / "evalharness" / "data" / "eval_set.jsonl"
    _write_jsonl(eval_set_path, [
        {"uid": "eval-u0", "statement": "Eval problem Alpha.", "answer": "eval-answer-0",
         "arxiv_id": "9999.00001", "eval_slice": "eval_band"},
    ])

    solutions_path = tmp_path / "proof_import" / "solutions_v3.jsonl"
    _write_jsonl(solutions_path, [_solutions_row(uid) for uid in TRAIN_UIDS])
    solutions_sha256 = _sha256(solutions_path)

    manifest_path = solutions_path.parent / "manifest.json"
    _write_json(manifest_path, {
        "solutions_v3": {"sha256": solutions_sha256},
        "split": {"sha256": split_sha256},
    })

    corpus_path = tmp_path / "out" / "corpus_pde625" / "band_corpus.jsonl"
    _write_jsonl(corpus_path, [
        {"uid": "t0", "statement": SOLUTION_SPECS["t0"][0], "answer": "0",
         "arxiv_id": SOLUTION_SPECS["t0"][3], "label": "band"},
    ])
    expected_corpus_sha256, expected_corpus_rows = _corpus_pin(corpus_path)

    # Shaped EXACTLY like the real file: label lives ONLY at the nested
    # pass_at_k_results.label path, never as a flat top-level "label" key
    # (the trap test_tier_resolution_pool_label_is_nested_not_flat pins).
    wellposed_path = tmp_path / "out" / "corpus_pde625" / "wellposed_all_with_passk.json"
    _write_json(wellposed_path, [
        {"uid": "t0", "pass_at_k_results": {"label": "collapse"}},  # conflicts w/ band_corpus -- must lose
        {"uid": "t1", "pass_at_k_results": {"label": "collapse"}},
        {"uid": "t2", "pass_at_k_results": {"label": "misdirection"}},
    ])

    v2_cap1_path = tmp_path / "v2_cap1" / "sft_train.jsonl"
    _write_jsonl(v2_cap1_path, [_v2_cap1_row(f"anchor{i}") for i in range(6)])

    return {
        "tmp_path": tmp_path,
        "split_path": split_path,
        "expected_split_sha256": split_sha256,
        "eval_set_path": eval_set_path,
        "solutions_path": solutions_path,
        "manifest_path": manifest_path,
        "solutions_sha256": solutions_sha256,
        "corpus_path": corpus_path,
        "expected_corpus_sha256": expected_corpus_sha256,
        "expected_corpus_rows": expected_corpus_rows,
        "wellposed_path": wellposed_path,
        "v2_cap1_path": v2_cap1_path,
        "bundle_dir": tmp_path / "bundle",
        "output_dir": tmp_path / "dataset",
        "rollouts_path": tmp_path / "rollouts.jsonl",
    }


def _append_solutions_row(env, row) -> None:
    """Append one row to env's solutions file and re-sync its manifest's
    recorded sha (so the sha-chain guard passes and later guards --
    holdout/unknown-uid, tier resolution -- are what actually fire).
    """
    rows = _read_jsonl(env["solutions_path"])
    rows.append(row)
    _write_jsonl(env["solutions_path"], rows)
    env["solutions_sha256"] = _sha256(env["solutions_path"])
    _write_json(env["manifest_path"], {
        "solutions_v3": {"sha256": env["solutions_sha256"]},
        "split": {"sha256": env["expected_split_sha256"]},
    })


def _add_offtier_uid(env, uid="t4", label="solved") -> str:
    """Extend env with one MORE train uid that resolves to an off-tier
    pool label (the GGUF-7/8-backfill-style trap the orchestrator's ruling names: a
    uid absent from band_corpus, present in the pool with a label that is
    neither band/collapse/misdirection). Rewrites split + solutions +
    manifest + pool in lockstep so every sha stays consistent.
    """
    split_data = json.loads(env["split_path"].read_text(encoding="utf-8"))
    split_data["train_side_uids"] = split_data["train_side_uids"] + [uid]
    _write_json(env["split_path"], split_data)
    env["expected_split_sha256"] = _sha256(env["split_path"])

    _append_solutions_row(env, _solutions_row(
        uid, question=f"Statement {uid}.", solution_text=f"Derivation {uid} ends in \\boxed{{9}}.",
        answer="9", arxiv_id="1000.00009",
    ))
    # _append_solutions_row already re-synced the manifest against the OLD
    # split sha -- re-sync again now that the split sha also changed.
    _write_json(env["manifest_path"], {
        "solutions_v3": {"sha256": env["solutions_sha256"]},
        "split": {"sha256": env["expected_split_sha256"]},
    })

    pool_rows = json.loads(env["wellposed_path"].read_text(encoding="utf-8"))
    pool_rows.append({"uid": uid, "pass_at_k_results": {"label": label}})
    _write_json(env["wellposed_path"], pool_rows)
    return uid


def _write_rollouts(env, rows) -> None:
    _write_jsonl(env["rollouts_path"], rows)


def _make_bundle_kwargs(env, **overrides) -> dict:
    kwargs = dict(
        solutions_path=env["solutions_path"],
        manifest_path=env["manifest_path"],
        split_path=env["split_path"],
        expected_split_sha256=env["expected_split_sha256"],
        eval_set_path=env["eval_set_path"],
        bundle_dir=env["bundle_dir"],
        k_regen=config.V3_K_REGEN,
    )
    kwargs.update(overrides)
    return kwargs


def _make_bundle(env, **overrides) -> dict:
    return v3.make_regen_bundle(**_make_bundle_kwargs(env, **overrides))


def _build_dataset_kwargs(env, **overrides) -> dict:
    kwargs = dict(
        bundle_dir=env["bundle_dir"],
        solutions_path=env["solutions_path"],
        manifest_path=env["manifest_path"],
        rollouts_path=env["rollouts_path"],
        split_path=env["split_path"],
        expected_split_sha256=env["expected_split_sha256"],
        eval_set_path=env["eval_set_path"],
        corpus_path=env["corpus_path"],
        expected_corpus_sha256=env["expected_corpus_sha256"],
        expected_corpus_rows=env["expected_corpus_rows"],
        wellposed_pool_path=env["wellposed_path"],
        v2_cap1_dataset_path=env["v2_cap1_path"],
        output_dir=env["output_dir"],
        k_regen=config.V3_K_REGEN,
        verify_fn=fake_verify_fn,
        force_new_dir=False,
    )
    kwargs.update(overrides)
    return kwargs


# ============================================================================
# 1. wire prompt builders (hint placement, byte-identity)
# ============================================================================


def test_question_only_prompt_byte_equals_serve_wire_prompt():
    statement = "A fixture statement for byte comparison."
    expected_wire_prompt = statement + " /no_think"
    built = v3.build_question_only_prompt(statement)
    assert built == expected_wire_prompt
    assert config.V3_HINT_MARKER not in built


def test_regen_prompt_contains_hint_and_question_suffix_terminal():
    statement = "A fixture statement."
    solution_text = "A worked solution ending in \\boxed{7}."
    regen = v3.build_regen_prompt(statement, solution_text)
    assert regen == statement + config.V3_HINT_MARKER + solution_text + config.PASS_AT_K_NO_THINK_SUFFIX
    assert statement in regen
    assert config.V3_HINT_MARKER in regen
    assert solution_text in regen
    # Interpreted deviation (orchestrator ruling): the no-think suffix is
    # TERMINAL, i.e. AFTER the hint block, never sandwiched mid-prompt.
    assert regen.endswith(config.PASS_AT_K_NO_THINK_SUFFIX)
    assert not regen.startswith(statement + config.PASS_AT_K_NO_THINK_SUFFIX)


# ============================================================================
# 2. make-regen-bundle refusals
# ============================================================================


def test_missing_solutions_file_refusal(env):
    env["solutions_path"].unlink()
    with pytest.raises(v3.SolutionsIntegrityError, match="not found"):
        _make_bundle(env)


def test_missing_solutions_manifest_refusal(env):
    env["manifest_path"].unlink()
    with pytest.raises(v3.SolutionsIntegrityError, match="not found"):
        _make_bundle(env)


def test_manifest_sha_chain_mismatch_refusal(env):
    _write_json(env["manifest_path"], {
        "solutions_v3": {"sha256": "0" * 64},
        "split": {"sha256": env["expected_split_sha256"]},
    })
    with pytest.raises(v3.SolutionsIntegrityError, match="sha256"):
        _make_bundle(env)


def test_manifest_split_sha_mismatch_refusal(env):
    _write_json(env["manifest_path"], {
        "solutions_v3": {"sha256": env["solutions_sha256"]},
        "split": {"sha256": "f" * 64},
    })
    with pytest.raises(v3.SolutionsIntegrityError, match="split"):
        _make_bundle(env)


def test_manifest_split_pin_accepts_old_era_split_sha16(env):
    # Provenance-era tolerance (split rebuild, 2026-08-01, v3.py's
    # assert_manifest_split_pin docstring): a solutions manifest recorded
    # under the OLD (now-void) 200/100 split's sha16 must still pass the
    # manifest-pin step -- config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S lists
    # it explicitly. This is the REAL production old-split sha16, not a
    # fixture-local value, to pin against regressions in that literal.
    assert config.EXPECTED_SPLIT_SHA256_16 in config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S
    _write_json(env["manifest_path"], {
        "solutions_v3": {"sha256": env["solutions_sha256"]},
        "split": {"sha256": config.EXPECTED_SPLIT_SHA256_16},
    })
    _make_bundle(env)  # must NOT raise
    assert (env["bundle_dir"] / "regen_bundle.jsonl").exists()


def test_manifest_split_pin_still_refuses_a_wholly_unrecognized_sha16(env):
    # The old-era tolerance is a small, explicit allow-list, not a general
    # loosening -- an sha16 that is neither the live pin nor a listed
    # provenance-era pin must still refuse (this is test_manifest_split_
    # sha_mismatch_refusal's scenario, re-asserted here to document WHY it
    # still refuses now that a tolerance list exists at all).
    assert "f" * 16 not in config.V3_ACCEPTED_MANIFEST_SPLIT_SHA16S
    _write_json(env["manifest_path"], {
        "solutions_v3": {"sha256": env["solutions_sha256"]},
        "split": {"sha256": "f" * 64},
    })
    with pytest.raises(v3.SolutionsIntegrityError, match="not in the accepted set"):
        _make_bundle(env)


def test_former_holdout_shaped_uid_now_hits_unknown_uid_hard_fail(env):
    # Split rebuild (2026-08-01): there is no holdout concept any more, so
    # a uid shaped like the OLD split's holdout set is just another
    # stranger uid -- it must hit the SAME UnknownUidError as "ghost"
    # below, never a special LeakageError "HOLDOUT" path (that branch is
    # retired -- see assert_train_split_only's docstring).
    _append_solutions_row(env, _solutions_row(
        "h0", question="Former-holdout-shaped Q.", solution_text="Sol \\boxed{9}.", answer="9", arxiv_id="1000.0h0",
    ))
    with pytest.raises(v3.UnknownUidError, match="not in the pinned split"):
        _make_bundle(env)


def test_unknown_uid_hard_fail(env):
    _append_solutions_row(env, _solutions_row(
        "ghost", question="Ghost Q.", solution_text="Ghost sol \\boxed{9}.", answer="9", arxiv_id="1000.0gh",
    ))
    with pytest.raises(v3.UnknownUidError, match="not in the pinned split"):
        _make_bundle(env)


def test_bundle_statement_leakage_exact_hard_fail(env):
    rows = _read_jsonl(env["solutions_path"])
    rows[0]["question"] = "Eval problem Alpha."  # exact collision w/ the eval_set fixture statement
    _write_jsonl(env["solutions_path"], rows)
    env["solutions_sha256"] = _sha256(env["solutions_path"])
    _write_json(env["manifest_path"], {
        "solutions_v3": {"sha256": env["solutions_sha256"]},
        "split": {"sha256": env["expected_split_sha256"]},
    })
    with pytest.raises(build_dataset.LeakageError, match="eval_set"):
        _make_bundle(env)


# ============================================================================
# 3. make-regen-bundle happy path + atomic publish + restartability
# ============================================================================


def test_bundle_rows_well_formed_no_answer_key(env):
    manifest = _make_bundle(env)
    rows = _read_jsonl(manifest["bundle"]["path"])
    assert len(rows) == len(TRAIN_UIDS)
    for row in rows:
        assert set(row.keys()) == {"uid", "regen_prompt"}
        assert "answer" not in row
        assert "solution_text" not in row
        assert config.V3_HINT_MARKER in row["regen_prompt"]
    bundle_uids = {r["uid"] for r in rows}
    assert bundle_uids == set(TRAIN_UIDS)
    assert not (bundle_uids & set(FORMER_HOLDOUT_SHAPED_UIDS))


def test_bundle_atomic_publish_verify_fail_leaves_no_final_artifacts(env, monkeypatch):
    calls = {"n": 0}
    real_replace = Path.replace

    def _boom(self, target):
        raise v3.BundleIntegrityError("forced failure for the atomic-publish test")

    # Force the post-write re-verification to fail by making the tmp file
    # unreadable-as-expected: simplest deterministic seam is to monkeypatch
    # Path.replace itself to explode AFTER the tmp file is written but
    # BEFORE the final name would exist.
    monkeypatch.setattr(Path, "replace", _boom)
    with pytest.raises(v3.BundleIntegrityError, match="forced failure"):
        _make_bundle(env)
    monkeypatch.setattr(Path, "replace", real_replace)

    assert not (env["bundle_dir"] / v3.BUNDLE_FILENAME).exists()
    assert not (env["bundle_dir"] / v3.BUNDLE_MANIFEST_FILENAME).exists()
    assert (env["bundle_dir"] / (v3.BUNDLE_FILENAME + ".tmp")).exists()


def test_bundle_restart_idempotent_resume(env):
    m1 = _make_bundle(env)
    assert m1.get("resumed_from_existing_publish") is False
    m2 = _make_bundle(env)
    assert m2.get("resumed_from_existing_publish") is True
    assert m2["bundle"]["sha256"] == m1["bundle"]["sha256"]
    assert m2["input_signature"] == m1["input_signature"]


def test_bundle_restart_conflict_without_force_refuses(env):
    _make_bundle(env)
    # Change an input (k_regen) so the signature differs, same bundle_dir.
    with pytest.raises(v3.PublishConflictError, match="force-new-dir"):
        _make_bundle(env, k_regen=config.V3_K_REGEN + 1)


def test_bundle_restart_force_new_dir_creates_sibling_without_touching_original(env):
    m1 = _make_bundle(env)
    m2 = _make_bundle(env, k_regen=config.V3_K_REGEN + 1, force_new_dir=True)
    assert Path(m2["bundle"]["path"]).parent != env["bundle_dir"]
    assert Path(m2["bundle"]["path"]).parent.name == env["bundle_dir"].name + "__2"
    # Original publish is untouched.
    assert (env["bundle_dir"] / v3.BUNDLE_MANIFEST_FILENAME).exists()
    original_manifest = json.loads((env["bundle_dir"] / v3.BUNDLE_MANIFEST_FILENAME).read_text())
    assert original_manifest["input_signature"] == m1["input_signature"]


# ============================================================================
# 4. build-dataset: bundle/solutions chain + rollouts guards
# ============================================================================


def test_build_dataset_missing_rollouts_refusal(env):
    _make_bundle(env)
    with pytest.raises(v3.RolloutIntegrityError, match="not found"):
        v3.build_dataset_cmd(**_build_dataset_kwargs(env, rollouts_path=env["tmp_path"] / "does_not_exist.jsonl"))


def test_build_dataset_missing_bundle_refusal(env):
    _write_rollouts(env, _default_rollouts())
    with pytest.raises(v3.BundleIntegrityError, match="regen bundle"):
        v3.build_dataset_cmd(**_build_dataset_kwargs(env))


def test_bundle_solutions_sha_chain_mismatch_refusal(env):
    _make_bundle(env)  # bundle built from the ORIGINAL (t0-t3) solutions
    _add_offtier_uid(env, uid="t4", label="solved")  # solutions/manifest now have a NEW sha
    _write_rollouts(env, _default_rollouts())
    with pytest.raises(v3.BundleIntegrityError, match="DIFFERENT solutions_v3"):
        v3.build_dataset_cmd(**_build_dataset_kwargs(env))


# ============================================================================
# 5. first-verified-wins, hint_insufficient
# ============================================================================


def test_first_verified_wins_earlier_unverified_skipped_later_verified_ignored(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    rows = _read_jsonl(manifest["dataset"]["path"])
    t0_rows = [r for r in rows if r["provenance"]["uid"] == "t0"]
    assert len(t0_rows) == 1
    prov = t0_rows[0]["provenance"]
    assert prov["regen_sample_idx"] == 1  # not 0 (unverified), not 2 (later verified, must be ignored)
    assert prov["verify_receipt"] == {"k_tried": 2, "verified": True}
    assert t0_rows[0]["completion"][0]["content"] == "CORRECT output for t0 try1 (kept)."


def test_hint_insufficient_drop_and_census(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    rows = _read_jsonl(manifest["dataset"]["path"])
    assert not any(r["provenance"].get("uid") == "t3" for r in rows)
    census = manifest["censuses"]["hint_insufficient"]
    assert census == {"count": 1, "uids": ["t3"]}


def test_missing_from_rollouts_is_a_distinct_census_class(env):
    rows = [r for r in _default_rollouts() if r["uid"] != "t2"]  # t2 never shows up at all
    _write_rollouts(env, rows)
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    assert manifest["censuses"]["missing_from_rollouts"] == {"count": 1, "uids": ["t2"]}
    assert manifest["censuses"]["hint_insufficient"] == {"count": 1, "uids": ["t3"]}


# ============================================================================
# 6. source-tier resolution (orchestrator ruling, 2026-07-31)
# ============================================================================


def test_tier_resolution_band_corpus_precedence_over_conflicting_pool_label():
    # t0 is band-corpus-resident AND carries a CONFLICTING "collapse" pool
    # label -- band_corpus membership must win unconditionally.
    bucket, label = v3.resolve_hinted_tier("t0", {"t0"}, {"t0": {"pass_at_k_results": {"label": "collapse"}}})
    assert bucket == "band"
    assert label == "band"


def test_tier_resolution_pool_label_is_nested_not_flat():
    # Regression test for the exact trap the orchestrator's ruling named: a flat
    # top-level "label" lookup silently returns None for every pool row.
    pool_row = {"pass_at_k_results": {"label": "collapse"}}
    assert "label" not in pool_row  # shaped exactly like the real file
    bucket, label = v3.resolve_hinted_tier("t1", set(), {"t1": pool_row})
    assert bucket == "collapse"
    assert label == "collapse"


def test_tier_resolution_misdirection_maps_to_collapse_bucket():
    bucket, _label = v3.resolve_hinted_tier("x", set(), {"x": {"pass_at_k_results": {"label": "misdirection"}}})
    assert bucket == "collapse"


def test_tier_resolution_solved_label_excluded_offtier():
    bucket, label = v3.resolve_hinted_tier("x", set(), {"x": {"pass_at_k_results": {"label": "solved"}}})
    assert bucket is None
    assert label == "solved"


def test_tier_resolution_uid_in_neither_source_excluded_offtier():
    bucket, label = v3.resolve_hinted_tier("ghost", set(), {})
    assert bucket is None
    assert label == "not_in_wellposed_pool"


def test_build_dataset_excludes_offtier_uid_end_to_end(env):
    uid = _add_offtier_uid(env, uid="t4", label="solved")  # GGUF-7/8-backfill-style trap
    rows = _default_rollouts() + [_rollout_row(uid, 0, f"CORRECT output for {uid}.")]
    _write_rollouts(env, rows)
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))

    final_rows = _read_jsonl(manifest["dataset"]["path"])
    assert not any(r["provenance"].get("uid") == uid for r in final_rows)
    census = manifest["censuses"]["excluded_offtier"]
    assert census["count"] == 1
    assert census["uids"] == [uid]
    assert census["resolved_labels"][uid] == "solved"
    # Off-tier rows must not inflate hinted_count / the anchor-count formula.
    assert manifest["blend"]["hinted_count"] == 3  # t0, t1, t2 only


def test_build_dataset_end_to_end_tier_counts_and_provenance(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    rows = _read_jsonl(manifest["dataset"]["path"])
    by_uid = {r["provenance"]["uid"]: r for r in rows if r["provenance"]["uid"] in ("t0", "t1", "t2")}
    assert by_uid["t0"]["provenance"]["source_tier"] == "band"
    assert by_uid["t1"]["provenance"]["source_tier"] == "collapse"
    assert by_uid["t2"]["provenance"]["source_tier"] == "collapse"  # misdirection -> collapse bucket
    for uid, row in by_uid.items():
        prov = row["provenance"]
        assert prov["proof_raw_sha"] == hashlib.sha256(f"proof-{uid}".encode("utf-8")).hexdigest()
        assert "verify_receipt" in prov and prov["verify_receipt"]["verified"] is True


# ============================================================================
# 7. R3 blend: arithmetic + determinism
# ============================================================================


def test_blend_ratio_arithmetic(env, monkeypatch):
    # The historical 25% blend arithmetic, preserved under monkeypatch: the
    # live config is 0.0 since the no-anchor ruling (Nicky 2026-08-01) --
    # see test_no_anchor_live_config below for the live-value behavior.
    monkeypatch.setattr(config, "V3_ANCHOR_FRACTION", 0.25)
    monkeypatch.setattr(config, "V3_HINTED_FRACTION", 0.75)
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    blend = manifest["blend"]
    assert blend["hinted_count"] == 3  # t0, t1, t2 verified; t3 hint_insufficient
    assert blend["hinted_collapse_count"] == 2  # t1, t2
    assert blend["hinted_band_count"] == 1  # t0
    assert blend["anchor_count"] == round(3 * 0.25 / 0.75)
    assert blend["anchor_count"] == 1
    assert blend["final_rows"] == 4
    rows = _read_jsonl(manifest["dataset"]["path"])
    assert len(rows) == 4
    anchor_rows = [r for r in rows if r["provenance"]["source_tier"] == "anchor"]
    assert len(anchor_rows) == 1


def test_no_anchor_live_config(env):
    # Live config: V3_ANCHOR_FRACTION == 0.0 (no-anchor ruling, 2026-08-01).
    # The anchor draw is skipped entirely; every published row is hinted.
    assert config.V3_ANCHOR_FRACTION == 0.0
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    blend = manifest["blend"]
    assert blend["hinted_count"] == 3
    assert blend["anchor_count"] == 0
    assert blend["final_rows"] == 3
    rows = _read_jsonl(manifest["dataset"]["path"])
    assert len(rows) == 3
    assert all(r["provenance"]["source_tier"] != "anchor" for r in rows)


def test_anchor_draw_determinism_same_seed_same_rows(env):
    v2_cap1_rows = _read_jsonl(env["v2_cap1_path"])
    exclude = {"t0", "t1", "t2"}
    first = v3.draw_anchor_rows(v2_cap1_rows, exclude, 2)
    second = v3.draw_anchor_rows(v2_cap1_rows, exclude, 2)
    assert first == second
    assert [r["provenance"]["uid"] for r in first] == [r["provenance"]["uid"] for r in second]


def test_build_dataset_determinism_two_builds_byte_identical(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    m1 = v3.build_dataset_cmd(**_build_dataset_kwargs(env, output_dir=env["tmp_path"] / "d1"))
    m2 = v3.build_dataset_cmd(**_build_dataset_kwargs(env, output_dir=env["tmp_path"] / "d2"))
    assert m1["dataset"]["sha256"] == m2["dataset"]["sha256"]
    assert Path(m1["dataset"]["path"]).read_bytes() == Path(m2["dataset"]["path"]).read_bytes()


def test_anchor_pool_excludes_hinted_uids(env):
    # v2/cap1 anchor pool is disjoint from t0-t3 by fixture construction;
    # directly prove exclusion still holds when a hinted uid DOES collide.
    v2_cap1_rows = _read_jsonl(env["v2_cap1_path"]) + [_v2_cap1_row("t0")]
    n_available_after_exclusion = len(v2_cap1_rows) - 1  # the appended "t0" row is excluded
    out = v3.draw_anchor_rows(v2_cap1_rows, {"t0"}, n_available_after_exclusion)  # ask for ALL non-excluded rows
    assert len(out) == n_available_after_exclusion
    assert "t0" not in {r["provenance"]["uid"] for r in out}


def test_anchor_pool_insufficient_raises_blend_error(env):
    v2_cap1_rows = _read_jsonl(env["v2_cap1_path"])
    with pytest.raises(v3.BlendError, match="anchor"):
        v3.draw_anchor_rows(v2_cap1_rows, set(), len(v2_cap1_rows) + 1)


# ============================================================================
# 8. output schema == v2 cap1 schema; hint-never-in-prompt; statement leakage
# ============================================================================


def test_output_schema_equals_v2_cap1_row_schema(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    rows = _read_jsonl(manifest["dataset"]["path"])
    assert rows, "expected at least one published row"
    for row in rows:
        assert set(row.keys()) == {"prompt", "completion", "provenance"}
    build_dataset.assert_prompt_completion_wellformed(rows)  # must not raise


def test_output_schema_matches_real_v2_cap1_file_when_present(env):
    real_path = ROOT / "data" / "v2" / "cap1" / "sft_train.jsonl"
    if not real_path.exists():
        pytest.skip("real v2/cap1 dataset not present in this checkout")
    real_row = json.loads(real_path.read_text(encoding="utf-8").splitlines()[0])
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    rows = _read_jsonl(manifest["dataset"]["path"])
    assert set(rows[0].keys()) == set(real_row.keys())
    assert set(rows[0]["prompt"][0].keys()) == set(real_row["prompt"][0].keys())


def test_hint_marker_in_zero_training_prompts(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    rows = _read_jsonl(manifest["dataset"]["path"])
    offenders = [r for r in rows if config.V3_HINT_MARKER in r["prompt"][1]["content"]]
    assert offenders == []
    assert manifest["loss_mass_census"]["prompts_hint_free"] == len(rows)
    assert manifest["loss_mass_census"]["completions_nonempty"] == len(rows)


def test_final_prompt_statement_leakage_hard_fail(env, monkeypatch):
    # Force a hinted row's question to textually match the eval_set
    # statement post-verification (simulating a leak that slipped past the
    # bundle-time exact check) by monkeypatching build_question_only_prompt
    # is overkill; instead mutate the solutions row directly and rebuild.
    rows = _read_jsonl(env["solutions_path"])
    for row in rows:
        if row["uid"] == "t1":
            row["question"] = "Eval problem Alpha."
    _write_jsonl(env["solutions_path"], rows)
    env["solutions_sha256"] = _sha256(env["solutions_path"])
    _write_json(env["manifest_path"], {
        "solutions_v3": {"sha256": env["solutions_sha256"]},
        "split": {"sha256": env["expected_split_sha256"]},
    })
    # The bundle-time exact-match guard already catches this -- so making
    # the bundle itself proves the EARLIER guard; this test's job is to
    # confirm make-regen-bundle refuses (statement leakage is caught at
    # the earliest point it can be, not deferred to build-dataset).
    with pytest.raises(build_dataset.LeakageError):
        _make_bundle(env)


# ============================================================================
# 9. atomic publish (build-dataset side) + restartability
# ============================================================================


def test_build_dataset_atomic_publish_verify_fail_leaves_no_final_artifacts(env, monkeypatch):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    monkeypatch.setattr(v3, "verify_written_v3_dataset", lambda *_a, **_k: -1)
    with pytest.raises(build_dataset.TraceIntegrityError, match="NOT published"):
        v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    assert not (env["output_dir"] / v3.DATASET_FILENAME).exists()
    assert not (env["output_dir"] / v3.DATASET_MANIFEST_FILENAME).exists()
    assert (env["output_dir"] / (v3.DATASET_FILENAME + ".tmp")).exists()


def test_build_dataset_restart_idempotent_resume(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    kwargs = _build_dataset_kwargs(env)
    m1 = v3.build_dataset_cmd(**kwargs)
    assert m1.get("resumed_from_existing_publish") is False
    m2 = v3.build_dataset_cmd(**kwargs)
    assert m2.get("resumed_from_existing_publish") is True
    assert m2["dataset"]["sha256"] == m1["dataset"]["sha256"]


def test_build_dataset_restart_conflict_without_force_refuses(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    # Change rollouts content (still verifies the same way) so the input
    # signature differs, at the SAME output_dir.
    _write_rollouts(env, _default_rollouts() + [_rollout_row("t1", 1, "CORRECT extra, unused.")])
    with pytest.raises(v3.PublishConflictError, match="force-new-dir"):
        v3.build_dataset_cmd(**_build_dataset_kwargs(env))


def test_build_dataset_restart_force_new_dir_creates_sibling(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    m1 = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    _write_rollouts(env, _default_rollouts() + [_rollout_row("t1", 1, "CORRECT extra, unused.")])
    m2 = v3.build_dataset_cmd(**_build_dataset_kwargs(env, force_new_dir=True))
    assert Path(m2["dataset"]["path"]).parent != env["output_dir"]
    assert Path(m2["dataset"]["path"]).parent.name == env["output_dir"].name + "__2"
    assert (env["output_dir"] / v3.DATASET_MANIFEST_FILENAME).exists()
    original = json.loads((env["output_dir"] / v3.DATASET_MANIFEST_FILENAME).read_text())
    assert original["input_signature"] == m1["input_signature"]


# ============================================================================
# 10. isolation (skeleton section 0, binding)
# ============================================================================


def _line_imports_v3(stripped_line: str) -> bool:
    """True iff ``stripped_line`` (comment already removed, whitespace
    stripped) is an import statement that pulls in ``v3``. Precise on
    purpose: config.py's own V3 constants section legitimately MENTIONS
    "loratrain.v3" in prose comments (documenting what consumes those
    constants) -- a bare substring scan over raw file text would flag
    that prose as a false-positive isolation violation, which is exactly
    why comments are stripped by the caller before this function ever
    sees a line.
    """
    if re.match(r"^import\s+loratrain\.v3\b", stripped_line):
        return True
    if re.match(r"^from\s+loratrain\.v3\s+import\b", stripped_line):
        return True
    if re.match(r"^from\s+\.v3\s+import\b", stripped_line):
        return True
    m = re.match(r"^from\s+(?:loratrain|\.)\s+import\s+(.*)$", stripped_line)
    if m and re.search(r"\bv3\b", m.group(1)):
        return True
    return False


def test_no_existing_module_imports_v3():
    """grep-based assert: no pre-existing loratrain module imports v3.

    Line-based, comment-stripped, import-statement-shaped matching (see
    ``_line_imports_v3``) -- a raw substring scan over full file text
    would false-positive on config.py's own V3 section, which legitimately
    documents "loratrain.v3" in prose comments without importing it.
    """
    package_dir = Path(config.__file__).resolve().parent  # src/loratrain/src/loratrain
    v3_path = (package_dir / "v3.py").resolve()
    offenders = []
    for py_file in sorted(package_dir.rglob("*.py")):
        if py_file.resolve() == v3_path:
            continue
        for line in py_file.read_text(encoding="utf-8").splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped and _line_imports_v3(stripped):
                offenders.append(py_file.name)
                break
    assert not offenders, f"existing module(s) import v3, breaking isolation: {offenders}"


def test_existing_modules_importable_without_importing_v3():
    """A fresh interpreter that imports every pre-existing loratrain module
    (never loratrain.v3) must succeed, and loratrain.v3 must NOT be a
    transitive import of any of them.

    Deliberately run in a SUBPROCESS: pytest collects every test file's
    top-level imports (including THIS file's own ``from loratrain import
    v3``) before any test function runs, so an in-process
    ``sys.modules`` check would already find 'loratrain.v3' present for a
    reason unrelated to what this test is checking. No network, no model,
    no server -- a bare, sub-second ``python -c`` import probe.
    """
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import loratrain.config\n"
        "import loratrain.build_dataset\n"
        "import loratrain.upload_guard\n"
        "import loratrain.train_lora\n"
        "import loratrain.export_serve\n"
        "import loratrain.tunnel\n"
        "import loratrain.verify_base_identity\n"
        "import loratrain.verify_dequant_parity\n"
        "import loratrain.gguf_to_hf\n"
        "assert 'loratrain.v3' not in sys.modules, "
        "'loratrain.v3 was imported as a side effect -- isolation broken'\n"
        "print('ISOLATION_OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ISOLATION_OK" in result.stdout


def test_v3_itself_imports_only_from_the_package_never_reverse():
    """Sanity companion to the two isolation tests above: v3.py imports
    FROM build_dataset/config (never the reverse) -- the isolation
    constraint's stated DIRECTION, not just "nothing imports v3".
    """
    text = Path(v3.__file__).read_text(encoding="utf-8")
    assert "from loratrain import build_dataset, config" in text or (
        "import build_dataset" in text and "import config" in text
    )


# ============================================================================
# 11. CLI shape sanity (no guard-bypass-shaped flags beyond --force-new-dir,
# which resolves an output-location collision, never a sha/leakage/holdout
# guard -- see module docstring "Restartability")
# ============================================================================


def test_cli_parser_builds_both_subcommands():
    parser = v3.build_arg_parser()
    args = parser.parse_args([
        "make-regen-bundle", "--solutions", "/tmp/s.jsonl", "--bundle-dir", "/tmp/b",
    ])
    assert args.subcommand == "make-regen-bundle"
    args2 = parser.parse_args([
        "build-dataset", "--bundle-dir", "/tmp/b", "--solutions", "/tmp/s.jsonl",
        "--rollouts", "/tmp/r.jsonl", "--output-dir", "/tmp/o",
    ])
    assert args2.subcommand == "build-dataset"


def test_cli_solutions_flag_has_no_default_no_globbing():
    parser = v3.build_arg_parser()
    for sub in ("make-regen-bundle", "build-dataset"):
        subparser = next(
            a.choices[sub] for a in parser._subparsers._group_actions if sub in a.choices
        )
        solutions_action = next(a for a in subparser._actions if a.dest == "solutions")
        assert solutions_action.required is True
        assert solutions_action.default is None


# ============================================================================
# 12. sha-chain tolerance PINNED against the REAL proof-import publish shape
#     (out/proof_import_20260731T185338Z, 2026-07-31): manifest records INPUT
#     shas only (input_shas.*, 16-hex prefixes) and never the published
#     file's own sha -- that lives in a stem-named sha sidecar (the lane's
#     bundle.sha256 idiom). These tests keep the tolerance from regressing
#     to the pre-publish guesswork.
# ============================================================================


def _as_real_publish_shape(env, sidecar_name=None, sidecar_sha=None):
    """Rewrite env's manifest into the real publish shape (input_shas only,
    16-hex split prefix, NO solutions sha) and optionally write a sha
    sidecar beside the solutions file."""
    _write_json(env["manifest_path"], {
        "input_shas": {"split": env["expected_split_sha256"][:16]},
        "censuses": {"p5": {"verified_published": 3}},
        "spend": {},
    })
    if sidecar_name is not None:
        sha = env["solutions_sha256"] if sidecar_sha is None else sidecar_sha
        (env["solutions_path"].parent / sidecar_name).write_text(
            f"{sha}  {env['solutions_path'].name}\n", encoding="utf-8"
        )


def test_real_publish_shape_stem_sidecar_and_input_shas_split_accepted(env):
    _as_real_publish_shape(env, sidecar_name="solutions_v3.sha256")
    _make_bundle(env)
    assert (env["bundle_dir"] / "regen_bundle.jsonl").exists()


def test_real_publish_shape_jsonl_suffixed_sidecar_accepted(env):
    _as_real_publish_shape(env, sidecar_name="solutions_v3.jsonl.sha256")
    _make_bundle(env)
    assert (env["bundle_dir"] / "regen_bundle.jsonl").exists()


def test_real_publish_shape_no_sidecar_refuses(env):
    _as_real_publish_shape(env, sidecar_name=None)
    with pytest.raises(v3.SolutionsIntegrityError, match="no sha sidecar"):
        _make_bundle(env)


def test_sidecar_wrong_sha_refuses(env):
    _as_real_publish_shape(env, sidecar_name="solutions_v3.sha256", sidecar_sha="0" * 64)
    with pytest.raises(v3.SolutionsIntegrityError, match="[Ss]ha-chain broken"):
        _make_bundle(env)


def test_manifest_input_shas_16hex_prefix_accepted(env):
    _write_json(env["manifest_path"], {
        "input_shas": {
            "solutions_v3": env["solutions_sha256"][:16],
            "split": env["expected_split_sha256"][:16],
        },
    })
    _make_bundle(env)
    assert (env["bundle_dir"] / "regen_bundle.jsonl").exists()


def test_recorded_prefix_shorter_than_16_hex_refuses(env):
    _write_json(env["manifest_path"], {
        "input_shas": {
            "solutions_v3": env["solutions_sha256"][:12],
            "split": env["expected_split_sha256"][:16],
        },
    })
    with pytest.raises(v3.SolutionsIntegrityError, match="too short to pin|Sha-chain broken"):
        _make_bundle(env)


# ============================================================================
# 13. v3b anchor_solved (docs/SESSION_HANDOFF.md ledger authorization @
#     2abe292; PREREGISTRATION_V3.md Amendment 6 @ 0139327)
#
# NAME-COLLISION REMINDER: "anchor_solved" here is a COMPLETELY DIFFERENT
# mechanism from the legacy "anchor" rows exercised in section 7 above
# (draw_anchor_rows / V3_ANCHOR_FRACTION / V3_ANCHOR_SEED_STRING, drawn
# from v2/cap1, currently retired to a 0.0 fraction). This section never
# touches that mechanism; test_no_anchor_live_config (section 7) already
# pins its 0.0 fraction as untouched.
# ============================================================================

ANCHOR_UIDS = ["a0", "a1"]

# (question, solution_text, answer, arxiv_id) -- arxiv_ids deliberately
# disjoint from the eval fixture's paper ("9999.00001", env's split.
# papers.eval_papers) so the happy path is genuinely NOT-in-eval.
ANCHOR_SOLUTION_SPECS = {
    "a0": ("Anchor statement zero.", "Anchor derivation zero ends in \\boxed{0}.", "0", "2000.00000"),
    "a1": ("Anchor statement one.", "Anchor derivation one ends in \\boxed{1}.", "1", "2000.00001"),
}


def _anchor_solutions_row(uid, *, question=None, solution_text=None, answer=None, arxiv_id=None):
    if question is None:
        question, solution_text, answer, arxiv_id = ANCHOR_SOLUTION_SPECS[uid]
    return {
        "uid": uid,
        "question": question,
        "proof_raw_sha": hashlib.sha256(f"anchor-proof-{uid}".encode("utf-8")).hexdigest(),
        "solution_text": solution_text,
        "answer": answer,
        "provenance": {
            "arxiv_id": arxiv_id,
            "match_method": "adjacency",
            "match_confidence": "high",
            "sonnet_cache_key": f"{uid}__anchor_cache",
            "verified": True,
        },
    }


def _default_anchor_rollouts():
    """a0: try0 CORRECT (kept, k_tried=1). a1: try0 WRONG, try1 CORRECT
    (kept, k_tried=2)."""
    return [
        _rollout_row("a0", 0, "CORRECT output for a0 try0."),
        _rollout_row("a1", 0, "WRONG output for a1 try0."),
        _rollout_row("a1", 1, "CORRECT output for a1 try1 (kept)."),
    ]


@pytest.fixture
def anchor_env(env):
    """Extends ``env`` with a 2-uid anchor_solutions.jsonl + manifest
    (uids "a0"/"a1", disjoint from TRAIN_UIDS and from the eval fixture's
    uid/paper -- see env's split.eval_set_uids / papers.eval_papers) and
    empty anchor-rollouts/bundle-dir paths for individual tests to
    populate.
    """
    tmp_path = env["tmp_path"]
    anchor_solutions_path = tmp_path / "proof_import" / "anchor_solutions.jsonl"
    _write_jsonl(anchor_solutions_path, [_anchor_solutions_row(uid) for uid in ANCHOR_UIDS])
    anchor_solutions_sha256 = _sha256(anchor_solutions_path)
    anchor_manifest_path = anchor_solutions_path.parent / "anchor_manifest.json"
    _write_json(anchor_manifest_path, {
        "anchor_solutions": {"sha256": anchor_solutions_sha256},
        "split": {"sha256": env["expected_split_sha256"]},
    })
    env["anchor_solutions_path"] = anchor_solutions_path
    env["anchor_manifest_path"] = anchor_manifest_path
    env["anchor_solutions_sha256"] = anchor_solutions_sha256
    env["anchor_bundle_dir"] = tmp_path / "anchor_bundle"
    env["anchor_rollouts_path"] = tmp_path / "anchor_rollouts.jsonl"
    return env


def _resync_anchor_manifest(env) -> None:
    _write_json(env["anchor_manifest_path"], {
        "anchor_solutions": {"sha256": env["anchor_solutions_sha256"]},
        "split": {"sha256": env["expected_split_sha256"]},
    })


def _make_anchor_bundle_kwargs(env, **overrides) -> dict:
    kwargs = dict(
        anchor_solutions_path=env["anchor_solutions_path"],
        anchor_manifest_path=env["anchor_manifest_path"],
        split_path=env["split_path"],
        expected_split_sha256=env["expected_split_sha256"],
        eval_set_path=env["eval_set_path"],
        bundle_dir=env["anchor_bundle_dir"],
        k_regen=config.V3_K_REGEN,
    )
    kwargs.update(overrides)
    return kwargs


def _make_anchor_bundle(env, **overrides) -> dict:
    return v3.make_anchor_bundle(**_make_anchor_bundle_kwargs(env, **overrides))


def _build_hinted_reference_dataset(env) -> Path:
    """Build the ordinary, non-anchor hinted dataset for ``env`` (t0/t1/t2
    kept, t3 hint_insufficient -- the default rollouts fixture) to serve
    as the "stage-1 reference" fixture for
    ``assert_hinted_subset_byte_unchanged`` tests: this run's OWN hinted
    rows must trivially be byte-identical to it, since both are built from
    the exact same solutions/bundle/rollouts inputs.
    """
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env, output_dir=env["tmp_path"] / "stage1_reference"))
    return Path(manifest["dataset"]["path"])


@pytest.fixture
def anchor_build_env(anchor_env):
    """``anchor_env`` plus a byte-exact stage-1 hinted reference dataset
    (built via the ordinary non-anchor path from this SAME env) and
    default anchor rollouts written -- ready for
    ``v3.build_dataset_cmd(**_build_dataset_anchor_kwargs(anchor_build_env))``.
    """
    reference_path = _build_hinted_reference_dataset(anchor_env)
    anchor_env["hinted_reference_dataset_path"] = reference_path
    anchor_env["output_dir"] = anchor_env["tmp_path"] / "dataset_v3b"
    _write_jsonl(anchor_env["anchor_rollouts_path"], _default_anchor_rollouts())
    return anchor_env


def _build_dataset_anchor_kwargs(env, **overrides) -> dict:
    kwargs = _build_dataset_kwargs(env)
    kwargs.update(
        anchor_solutions_path=env["anchor_solutions_path"],
        anchor_manifest_path=env["anchor_manifest_path"],
        anchor_rollouts_path=env["anchor_rollouts_path"],
        hinted_reference_dataset_path=env["hinted_reference_dataset_path"],
    )
    kwargs.update(overrides)
    return kwargs


# ---- 13a. make-anchor-bundle -----------------------------------------------


def test_make_anchor_bundle_happy_path_no_answer_key(anchor_env):
    manifest = _make_anchor_bundle(anchor_env)
    rows = _read_jsonl(manifest["bundle"]["path"])
    assert len(rows) == len(ANCHOR_UIDS)
    for row in rows:
        assert set(row.keys()) == {"uid", "regen_prompt"}
        assert "answer" not in row
        assert "solution_text" not in row
        assert config.V3_HINT_MARKER in row["regen_prompt"]
    assert {r["uid"] for r in rows} == set(ANCHOR_UIDS)
    assert manifest["stage"] == v3.STAGE_ANCHOR_BUNDLE
    assert manifest["lane"] == "anchor_solved"


def test_make_anchor_bundle_does_not_require_train_split_membership(anchor_env):
    # a0/a1 are NOT in TRAIN_UIDS (disjoint by fixture construction) --
    # make_anchor_bundle must NOT hit assert_train_split_only's
    # UnknownUidError, proving the guard substitution actually took effect
    # (docs/SESSION_HANDOFF.md ledger authorization @ 2abe292).
    assert not (set(ANCHOR_UIDS) & set(TRAIN_UIDS))
    _make_anchor_bundle(anchor_env)  # must NOT raise
    assert (anchor_env["anchor_bundle_dir"] / v3.ANCHOR_BUNDLE_FILENAME).exists()


def test_make_anchor_bundle_uid_in_eval_set_uids_refuses(anchor_env):
    split_data = json.loads(anchor_env["split_path"].read_text(encoding="utf-8"))
    split_data["eval_set_uids"]["band"].append("a0")
    _write_json(anchor_env["split_path"], split_data)
    anchor_env["expected_split_sha256"] = _sha256(anchor_env["split_path"])
    _resync_anchor_manifest(anchor_env)
    with pytest.raises(v3.EvalMembershipError, match="uid membership"):
        _make_anchor_bundle(anchor_env)


def test_make_anchor_bundle_paper_in_eval_papers_refuses(anchor_env):
    split_data = json.loads(anchor_env["split_path"].read_text(encoding="utf-8"))
    split_data["papers"]["eval_papers"].append(ANCHOR_SOLUTION_SPECS["a0"][3])
    _write_json(anchor_env["split_path"], split_data)
    anchor_env["expected_split_sha256"] = _sha256(anchor_env["split_path"])
    _resync_anchor_manifest(anchor_env)
    with pytest.raises(v3.EvalMembershipError, match="paper membership"):
        _make_anchor_bundle(anchor_env)


def test_make_anchor_bundle_statement_leakage_exact_hard_fail(anchor_env):
    rows = _read_jsonl(anchor_env["anchor_solutions_path"])
    rows[0]["question"] = "Eval problem Alpha."  # exact collision w/ the eval_set fixture statement
    _write_jsonl(anchor_env["anchor_solutions_path"], rows)
    anchor_env["anchor_solutions_sha256"] = _sha256(anchor_env["anchor_solutions_path"])
    _resync_anchor_manifest(anchor_env)
    with pytest.raises(build_dataset.LeakageError, match="eval_set"):
        _make_anchor_bundle(anchor_env)


def test_make_anchor_bundle_manifest_sha_chain_mismatch_refusal(anchor_env):
    _write_json(anchor_env["anchor_manifest_path"], {
        "anchor_solutions": {"sha256": "0" * 64},
        "split": {"sha256": anchor_env["expected_split_sha256"]},
    })
    with pytest.raises(v3.SolutionsIntegrityError, match="sha256"):
        _make_anchor_bundle(anchor_env)


def test_make_anchor_bundle_restart_idempotent_resume(anchor_env):
    m1 = _make_anchor_bundle(anchor_env)
    assert m1.get("resumed_from_existing_publish") is False
    m2 = _make_anchor_bundle(anchor_env)
    assert m2.get("resumed_from_existing_publish") is True
    assert m2["bundle"]["sha256"] == m1["bundle"]["sha256"]
    assert m2["input_signature"] == m1["input_signature"]


def test_make_anchor_bundle_restart_force_new_dir_creates_sibling(anchor_env):
    m1 = _make_anchor_bundle(anchor_env)
    m2 = _make_anchor_bundle(anchor_env, k_regen=config.V3_K_REGEN + 1, force_new_dir=True)
    assert Path(m2["bundle"]["path"]).parent != anchor_env["anchor_bundle_dir"]
    assert Path(m2["bundle"]["path"]).parent.name == anchor_env["anchor_bundle_dir"].name + "__2"
    original_manifest = json.loads(
        (anchor_env["anchor_bundle_dir"] / v3.ANCHOR_BUNDLE_MANIFEST_FILENAME).read_text()
    )
    assert original_manifest["input_signature"] == m1["input_signature"]


def test_anchor_bundle_filenames_distinct_from_hinted_bundle_filenames():
    # Coexistence guarantee (module docstring) -- also incidentally proves
    # the two lanes are never confused with each other.
    assert v3.ANCHOR_BUNDLE_FILENAME != v3.BUNDLE_FILENAME
    assert v3.ANCHOR_BUNDLE_MANIFEST_FILENAME != v3.BUNDLE_MANIFEST_FILENAME
    assert v3.STAGE_ANCHOR_BUNDLE != v3.STAGE_REGEN_BUNDLE


# ---- 13b. load_split_eval_membership / assert_anchor_not_in_eval ----------


def test_load_split_eval_membership_flattens_tiered_dict(anchor_env):
    eval_uids, eval_papers = v3.load_split_eval_membership(
        anchor_env["split_path"], anchor_env["expected_split_sha256"]
    )
    assert eval_uids == {"eval-u0"}
    assert eval_papers == {"9999.00001"}


def test_assert_anchor_not_in_eval_clean_does_not_raise():
    rows = [{"uid": "a0", "provenance": {"arxiv_id": "2000.00000"}}]
    v3.assert_anchor_not_in_eval(rows, {"eval-u0"}, {"9999.00001"})  # must not raise


def test_assert_anchor_not_in_eval_uid_offender():
    rows = [{"uid": "eval-u0", "provenance": {"arxiv_id": "2000.00000"}}]
    with pytest.raises(v3.EvalMembershipError, match="uid membership"):
        v3.assert_anchor_not_in_eval(rows, {"eval-u0"}, {"9999.00001"})


def test_assert_anchor_not_in_eval_paper_offender():
    rows = [{"uid": "a0", "provenance": {"arxiv_id": "9999.00001"}}]
    with pytest.raises(v3.EvalMembershipError, match="paper membership"):
        v3.assert_anchor_not_in_eval(rows, {"eval-u0"}, {"9999.00001"})


@pytest.mark.parametrize("bad", [None, "", "   ", {}])
def test_assert_anchor_not_in_eval_paperless_refuses(bad):
    # Orchestrator hardening (2026-08-02 integration review): without an
    # arxiv_id the paper-level half of the substituted membership rule cannot
    # run -- `None in eval_papers` is vacuously False. That must refuse, not
    # pass. upload_guard fail-closes the same way at the final gate; refusing
    # here means it lands before regen/training spend.
    prov = {} if bad == {} else {"arxiv_id": bad}
    rows = [{"uid": "a0", "provenance": prov}]
    with pytest.raises(v3.EvalMembershipError, match="no provenance.arxiv_id"):
        v3.assert_anchor_not_in_eval(rows, {"eval-u0"}, {"9999.00001"})


def test_assert_anchor_not_in_eval_paperless_named_before_other_offenders():
    # A paperless row and a genuine uid offender together: the paperless
    # refusal fires first and names the paperless row, so the operator fixes
    # the un-checkable input rather than chasing a partial verdict.
    rows = [
        {"uid": "a0", "provenance": {}},
        {"uid": "eval-u0", "provenance": {"arxiv_id": "2000.00000"}},
    ]
    with pytest.raises(v3.EvalMembershipError, match="a0"):
        v3.assert_anchor_not_in_eval(rows, {"eval-u0"}, {"9999.00001"})


# ---- 13c. hint_copy_fraction (n-gram census) -------------------------------


def test_hint_copy_fraction_partial_overlap():
    hint = "one two three four five six seven"
    completion = "prefix one two three four five six extra"
    # hint has 2 word-6-grams: (one..six), (two..seven); completion's
    # 6-grams contain (one..six) but not (two..seven) -> 1/2.
    assert v3.hint_copy_fraction(hint, completion) == 0.5


def test_hint_copy_fraction_no_overlap():
    hint = "alpha beta gamma delta epsilon zeta"
    completion = "nothing in common here at all whatsoever"
    assert v3.hint_copy_fraction(hint, completion) == 0.0


def test_hint_copy_fraction_short_hint_is_vacuous_zero():
    assert v3.hint_copy_fraction("too short", "anything at all") == 0.0


# ---- 13d. build-dataset --anchor-solutions / --anchor-rollouts ------------


def test_build_dataset_anchor_both_required_together_solutions_only(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    with pytest.raises(v3.AnchorInputError, match="together"):
        v3.build_dataset_cmd(**_build_dataset_kwargs(
            env, anchor_solutions_path=Path("/tmp/anchor_solutions.jsonl"),
        ))


def test_build_dataset_anchor_both_required_together_rollouts_only(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    with pytest.raises(v3.AnchorInputError, match="together"):
        v3.build_dataset_cmd(**_build_dataset_kwargs(
            env, anchor_rollouts_path=Path("/tmp/anchor_rollouts.jsonl"),
        ))


def test_build_dataset_anchor_requires_hinted_reference_dataset(anchor_build_env):
    kwargs = _build_dataset_anchor_kwargs(anchor_build_env, hinted_reference_dataset_path=None)
    with pytest.raises(v3.AnchorInputError, match="hinted-reference-dataset"):
        v3.build_dataset_cmd(**kwargs)


def test_build_dataset_anchor_happy_path_appends_rows_and_censuses(anchor_build_env):
    manifest = v3.build_dataset_cmd(**_build_dataset_anchor_kwargs(anchor_build_env))
    rows = _read_jsonl(manifest["dataset"]["path"])
    assert len(rows) == 5  # 3 hinted (t0, t1, t2) + 2 anchor_solved (a0, a1)
    anchor_rows = [r for r in rows if r["provenance"]["source_tier"] == "anchor_solved"]
    assert {r["provenance"]["uid"] for r in anchor_rows} == set(ANCHOR_UIDS)

    census = manifest["anchor_solved"]
    assert census["lane"] == "anchor_solved"
    assert census["count"] == 2
    assert census["fraction_of_final_dataset"] == 2 / 5
    assert census["try_histogram"] == {"1": 1, "2": 1}
    assert census["attrition"]["missing_from_rollouts"] == {"count": 0, "uids": []}
    assert census["attrition"]["hint_insufficient"] == {"count": 0, "uids": []}
    assert set(census["hint_copy_census"]["per_uid"]) == set(ANCHOR_UIDS)
    assert census["hinted_byte_identity_check"]["result"] == "byte_identical"
    assert any("anchor" in g for g in manifest["guards"])


def test_build_dataset_anchor_excluded_from_blend_arithmetic(anchor_build_env):
    manifest = v3.build_dataset_cmd(**_build_dataset_anchor_kwargs(anchor_build_env))
    # The R3 blend's own arithmetic (hinted/legacy-anchor 60/40 + 75/25)
    # must be UNAFFECTED by anchor_solved -- it never counts toward either
    # hinted_count or anchor_count, and the blend block's own "final_rows"
    # reports the R3-blend subtotal only (3), not the true published total
    # (5, which lives at manifest["dataset"]["rows"] instead).
    assert manifest["blend"]["hinted_count"] == 3
    assert manifest["blend"]["anchor_count"] == 0  # legacy anchor, V3_ANCHOR_FRACTION == 0.0, untouched
    assert manifest["blend"]["final_rows"] == 3
    assert manifest["dataset"]["rows"] == 5


def test_build_dataset_anchor_never_in_excluded_offtier(anchor_build_env):
    manifest = v3.build_dataset_cmd(**_build_dataset_anchor_kwargs(anchor_build_env))
    assert manifest["censuses"]["excluded_offtier"]["count"] == 0
    assert not (set(ANCHOR_UIDS) & set(manifest["censuses"]["excluded_offtier"]["uids"]))


def test_build_dataset_anchor_uid_collision_with_hinted_raises_blend_error(anchor_build_env):
    # Re-use "t0" (a hinted uid) as an anchor_solutions uid -- cap1 must
    # refuse (asserted, never silently deduped -- module docstring "cap1").
    rows = _read_jsonl(anchor_build_env["anchor_solutions_path"])
    rows.append(_anchor_solutions_row(
        "t0", question="Colliding anchor Q.", solution_text="Colliding sol \\boxed{9}.",
        answer="9", arxiv_id="2000.00099",
    ))
    _write_jsonl(anchor_build_env["anchor_solutions_path"], rows)
    anchor_build_env["anchor_solutions_sha256"] = _sha256(anchor_build_env["anchor_solutions_path"])
    _resync_anchor_manifest(anchor_build_env)
    anchor_rollouts = _read_jsonl(anchor_build_env["anchor_rollouts_path"])
    anchor_rollouts.append(_rollout_row("t0", 0, "CORRECT output for colliding t0."))
    _write_jsonl(anchor_build_env["anchor_rollouts_path"], anchor_rollouts)

    with pytest.raises(v3.BlendError, match="cap1"):
        v3.build_dataset_cmd(**_build_dataset_anchor_kwargs(anchor_build_env))


def test_build_dataset_anchor_missing_from_rollouts_and_hint_insufficient_censused(anchor_build_env):
    # a0 never shows up in the anchor rollouts at all; a1 gets only WRONG
    # tries -- both distinct attrition classes, named, never silently
    # dropped (mirrors the hinted lane's census shape).
    _write_jsonl(anchor_build_env["anchor_rollouts_path"], [
        _rollout_row("a1", 0, "WRONG output for a1 try0."),
    ])
    manifest = v3.build_dataset_cmd(**_build_dataset_anchor_kwargs(anchor_build_env))
    census = manifest["anchor_solved"]
    assert census["count"] == 0
    assert census["attrition"]["missing_from_rollouts"] == {"count": 1, "uids": ["a0"]}
    assert census["attrition"]["hint_insufficient"] == {"count": 1, "uids": ["a1"]}
    rows = _read_jsonl(manifest["dataset"]["path"])
    assert len(rows) == 3  # hinted only -- neither anchor uid made it in


def test_build_dataset_default_no_anchor_flags_manifest_has_no_anchor_solved_key(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    assert "anchor_solved" not in manifest
    assert manifest["guards"] == list(v3.BUILD_DATASET_GUARD_STEPS)


def test_build_dataset_default_dataset_bytes_unaffected_by_new_anchor_kwargs_existing(env):
    # ADDITIVE-ONLY contract: build once via the plain kwargs (no anchor
    # kwargs even mentioned), once via kwargs that explicitly pass the new
    # anchor params as their own defaults (None) -- both must publish
    # byte-identical dataset content.
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    m1 = v3.build_dataset_cmd(**_build_dataset_kwargs(env, output_dir=env["tmp_path"] / "d1"))
    m2 = v3.build_dataset_cmd(**_build_dataset_kwargs(
        env, output_dir=env["tmp_path"] / "d2",
        anchor_solutions_path=None, anchor_manifest_path=None,
        anchor_rollouts_path=None, hinted_reference_dataset_path=None,
    ))
    assert m1["dataset"]["sha256"] == m2["dataset"]["sha256"]
    assert Path(m1["dataset"]["path"]).read_bytes() == Path(m2["dataset"]["path"]).read_bytes()


# ---- 13e. assert_hinted_subset_byte_unchanged ------------------------------


def test_hinted_subset_byte_unchanged_matching_reference_does_not_raise(env):
    reference_path = _build_hinted_reference_dataset(env)
    hinted_rows = _read_jsonl(reference_path)
    v3.assert_hinted_subset_byte_unchanged(hinted_rows, reference_path)  # must not raise


def test_hinted_subset_byte_unchanged_content_mismatch_refuses(env):
    reference_path = _build_hinted_reference_dataset(env)
    hinted_rows = _read_jsonl(reference_path)
    hinted_rows[0]["completion"][0]["content"] = "a DIFFERENT completion, drifted from stage-1"
    with pytest.raises(v3.HintedReferenceMismatchError, match="differ in content"):
        v3.assert_hinted_subset_byte_unchanged(hinted_rows, reference_path)


def test_hinted_subset_byte_unchanged_uid_set_differs_refuses(env):
    reference_path = _build_hinted_reference_dataset(env)
    hinted_rows = _read_jsonl(reference_path)
    hinted_rows.pop()  # drop one uid entirely -- the sets no longer match
    with pytest.raises(v3.HintedReferenceMismatchError, match="uid set differs"):
        v3.assert_hinted_subset_byte_unchanged(hinted_rows, reference_path)


def test_hinted_subset_byte_unchanged_missing_reference_file_refuses(env, tmp_path):
    with pytest.raises(v3.HintedReferenceMismatchError, match="not found"):
        v3.assert_hinted_subset_byte_unchanged([], tmp_path / "does_not_exist.jsonl")


def test_build_dataset_anchor_stage1_drift_refuses_before_anchor_harvest(anchor_build_env):
    # Mutate the PUBLISHED reference file (simulating stage-1 drift) --
    # build-dataset must refuse via assert_hinted_subset_byte_unchanged
    # before spending any effort on the anchor harvest.
    ref_rows = _read_jsonl(anchor_build_env["hinted_reference_dataset_path"])
    ref_rows[0]["completion"][0]["content"] = "drifted stage-1 content"
    _write_jsonl(anchor_build_env["hinted_reference_dataset_path"], ref_rows)
    with pytest.raises(v3.HintedReferenceMismatchError, match="differ in content"):
        v3.build_dataset_cmd(**_build_dataset_anchor_kwargs(anchor_build_env))


# ---- 13f. eval-set freshness warning (config.EVAL_SET_PATH staleness) -----


def test_eval_set_freshness_warning_fires_on_row_count_mismatch(anchor_env, capsys):
    # anchor_env's eval_set.jsonl fixture has 1 row -- config.
    # V3_ANCHOR_EVAL_SET_EXPECTED_ROWS is 286, so this must warn (never
    # hard-fail: the happy-path bundle build must still succeed).
    assert config.V3_ANCHOR_EVAL_SET_EXPECTED_ROWS != 1
    _make_anchor_bundle(anchor_env)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "286" in captured.err


def test_eval_set_freshness_warning_silent_when_row_count_matches(anchor_env, capsys, monkeypatch):
    monkeypatch.setattr(config, "V3_ANCHOR_EVAL_SET_EXPECTED_ROWS", 1)  # matches the fixture's 1-row eval set
    _make_anchor_bundle(anchor_env)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.err


# ---- 13g. sha-chain generalization is byte-identical for the default label -


def test_generalized_sha_chain_default_label_message_unchanged(env):
    # assert_solutions_sha_chain's generalization (manifest_sha_paths /
    # sidecar_names / label kwargs, all defaulting to the pre-v3b globals)
    # must reproduce the EXACT pre-v3b not-found message for the default
    # (solutions_v3) label -- this pins that byte-identity.
    missing_path = env["tmp_path"] / "does_not_exist_solutions_v3.jsonl"
    with pytest.raises(v3.SolutionsIntegrityError, match=r"solutions_v3\.jsonl yet"):
        v3.assert_solutions_sha_chain(missing_path, env["manifest_path"], {})


def test_generalized_sha_chain_anchor_label_distinct_message(env):
    missing_path = env["tmp_path"] / "does_not_exist_anchor_solutions.jsonl"
    with pytest.raises(v3.SolutionsIntegrityError, match="anchor solutions"):
        v3.assert_solutions_sha_chain(
            missing_path, env["manifest_path"], {},
            manifest_sha_paths=v3.ANCHOR_MANIFEST_SHA_PATHS,
            sidecar_names=v3.ANCHOR_SHA_SIDECAR_NAMES,
            label="anchor solutions",
        )


# ---- 13h. CLI wiring --------------------------------------------------------


def test_cli_parser_builds_make_anchor_bundle_subcommand():
    parser = v3.build_arg_parser()
    args = parser.parse_args([
        "make-anchor-bundle", "--anchor-solutions", "/tmp/a.jsonl", "--bundle-dir", "/tmp/ab",
    ])
    assert args.subcommand == "make-anchor-bundle"
    assert args.anchor_manifest is None
    assert args.split == config.V3_SPLIT_PATH
    assert args.eval_set == config.EVAL_SET_PATH


def test_cli_build_dataset_anchor_flags_optional_default_none():
    parser = v3.build_arg_parser()
    args = parser.parse_args([
        "build-dataset", "--bundle-dir", "/tmp/b", "--solutions", "/tmp/s.jsonl",
        "--rollouts", "/tmp/r.jsonl", "--output-dir", "/tmp/o",
    ])
    assert args.anchor_solutions is None
    assert args.anchor_rollouts is None
    assert args.hinted_reference_dataset == config.V3_STAGE1_HINTED_DATASET_PATH


def test_cli_build_dataset_accepts_anchor_flags():
    parser = v3.build_arg_parser()
    args = parser.parse_args([
        "build-dataset", "--bundle-dir", "/tmp/b", "--solutions", "/tmp/s.jsonl",
        "--rollouts", "/tmp/r.jsonl", "--output-dir", "/tmp/o",
        "--anchor-solutions", "/tmp/anchor_solutions.jsonl",
        "--anchor-rollouts", "/tmp/anchor_rollouts.jsonl",
    ])
    assert str(args.anchor_solutions) == "/tmp/anchor_solutions.jsonl"
    assert str(args.anchor_rollouts) == "/tmp/anchor_rollouts.jsonl"


# ---- 13i. legacy "anchor" mechanism left untouched -------------------------


def test_legacy_anchor_mechanism_names_remain_distinct_from_anchor_solved():
    # Cheap regression pin against the two mechanisms ever being conflated:
    # different config constants, different provenance source_tier values.
    assert config.V3_ANCHOR_FRACTION == 0.0  # Nicky's NO-ANCHOR ruling, untouched by this work
    assert "anchor_solved" != "anchor"
    assert hasattr(config, "V3_ANCHOR_SEED_STRING")  # legacy mechanism's own config, untouched
    assert hasattr(config, "V3_ANCHOR_EVAL_SET_EXPECTED_ROWS")  # this work's own, unrelated config

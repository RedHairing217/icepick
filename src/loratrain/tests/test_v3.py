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
HOLDOUT_UIDS = ["h0", "h1"]

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

    4 train uids (t0-t3, all with solutions rows), 2 holdout uids (h0,h1);
    t0 is band-tier via band_corpus membership (and ALSO appears in the
    wellposed pool with a CONFLICTING "collapse" label -- band_corpus must
    win, see ``test_tier_resolution_*``); t1/t2 resolve via the pool
    (collapse / misdirection, both -> collapse bucket); a 6-row v2/cap1
    anchor pool with uids disjoint from t0-t3. Individual tests mutate one
    piece (rewrite a file, override one kwarg) to exercise a refusal path.
    """
    split_path = tmp_path / "evalharness" / "data" / "corpus_split_200_100.json"
    _write_json(split_path, {
        "train_uids": list(TRAIN_UIDS),
        "holdout_uids": list(HOLDOUT_UIDS),
        "eval_papers": ["9999.00001"],
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
    split_data["train_uids"] = split_data["train_uids"] + [uid]
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


def test_holdout_uid_in_solutions_hard_fail(env):
    _append_solutions_row(env, _solutions_row(
        "h0", question="Holdout Q.", solution_text="Holdout sol \\boxed{9}.", answer="9", arxiv_id="1000.0h0",
    ))
    with pytest.raises(build_dataset.LeakageError, match="HOLDOUT"):
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


def test_bundle_rows_well_formed_no_answer_key_no_holdout(env):
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
    assert not (bundle_uids & set(HOLDOUT_UIDS))


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


def test_blend_ratio_arithmetic(env):
    _write_rollouts(env, _default_rollouts())
    _make_bundle(env)
    manifest = v3.build_dataset_cmd(**_build_dataset_kwargs(env))
    blend = manifest["blend"]
    assert blend["hinted_count"] == 3  # t0, t1, t2 verified; t3 hint_insufficient
    assert blend["hinted_collapse_count"] == 2  # t1, t2
    assert blend["hinted_band_count"] == 1  # t0
    assert blend["anchor_count"] == round(3 * config.V3_ANCHOR_FRACTION / config.V3_HINTED_FRACTION)
    assert blend["anchor_count"] == 1
    assert blend["final_rows"] == 4
    rows = _read_jsonl(manifest["dataset"]["path"])
    assert len(rows) == 4
    anchor_rows = [r for r in rows if r["provenance"]["source_tier"] == "anchor"]
    assert len(anchor_rows) == 1


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

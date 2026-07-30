"""Tests for the dataset-v2 invariants: weight policy + prompt/completion schema.

Pure-function level (no disk, no fixture env -- the build()-level
integration of both fixes is tested in test_build_dataset.py): the
defect-2 policy machinery (``apply_weight_policy`` /
``assert_weight_policy_honored``), the defect-1 schema guard
(``assert_prompt_completion_wellformed``), and the config knobs behind
them. See the work order of 2026-07-29 and README "Dataset v2".
"""

from __future__ import annotations

import hashlib

import pytest

from loratrain import build_dataset, config
from loratrain.build_dataset import (
    TraceIntegrityError,
    WeightPolicyError,
    apply_weight_policy,
    assert_prompt_completion_wellformed,
    assert_weight_policy_honored,
)


def _example(uid, rollout_uid, statement="Prove it.", output="\n\nProof."):
    """A minimal, well-formed v2 example (real wire-format pins)."""
    return {
        "prompt": [
            {"role": "system", "content": config.PASS_AT_K_SYSTEM_PROMPT},
            {"role": "user", "content": statement + config.PASS_AT_K_NO_THINK_SUFFIX},
        ],
        "completion": [{"role": "assistant", "content": output}],
        "provenance": {"uid": uid, "rollout_uid": rollout_uid},
    }


def _flock(counts: dict) -> list:
    """counts = {uid: n_traces} -> a flat example list in harvest order."""
    examples = []
    for uid, n in counts.items():
        examples.extend(_example(uid, f"{uid}-r{i:02d}") for i in range(n))
    return examples


# --- apply_weight_policy: cap1 ---------------------------------------------------


def test_cap1_keeps_exactly_one_row_per_uid():
    examples = _flock({"a": 1, "b": 3, "c": 7})
    kept, block = apply_weight_policy(examples, policy="cap1", cap_k=3, seed=42)

    per_uid = {}
    for ex in kept:
        per_uid[ex["provenance"]["uid"]] = per_uid.get(ex["provenance"]["uid"], 0) + 1
    assert per_uid == {"a": 1, "b": 1, "c": 1}
    assert all("weight" not in ex for ex in kept)
    assert block["rows_before"] == 11
    assert block["rows_after"] == 3
    assert block["policy"] == "cap1"
    assert block["label"] == "cap1"
    assert block["cap_k"] is None
    assert block["weighted"] is False
    assert block["rows_per_uid_before"] == {"1": 1, "3": 1, "7": 1}
    assert block["rows_per_uid_after"] == {"1": 3}


def test_cap1_selection_is_deterministic_and_seeded_not_first_n():
    examples = _flock({"b": 5})
    kept_42a, _ = apply_weight_policy(examples, policy="cap1", cap_k=1, seed=42)
    kept_42b, _ = apply_weight_policy(examples, policy="cap1", cap_k=1, seed=42)
    assert kept_42a == kept_42b  # same seed -> byte-identical selection

    # The selection follows the DOCUMENTED rule (sha256("{seed}:{uid}:{rollout_uid}")
    # ascending), recomputed here independently -- not file order, not "first N".
    def expected_pick(seed):
        return min(
            (f"b-r{i:02d}" for i in range(5)),
            key=lambda r: hashlib.sha256(f"{seed}:b:{r}".encode()).hexdigest(),
        )

    assert kept_42a[0]["provenance"]["rollout_uid"] == expected_pick(42)
    # And a seed for which the rule picks a DIFFERENT trace than seed 42
    # (found by scanning, so this stays true by construction, not luck).
    other_seed = next(s for s in range(1000) if expected_pick(s) != expected_pick(42))
    kept_other, _ = apply_weight_policy(examples, policy="cap1", cap_k=1, seed=other_seed)
    assert kept_other[0]["provenance"]["rollout_uid"] == expected_pick(other_seed)
    assert kept_other[0]["provenance"]["rollout_uid"] != kept_42a[0]["provenance"]["rollout_uid"]


def test_cap1_kept_rows_preserve_harvest_order():
    examples = _flock({"a": 2, "b": 2, "c": 2})
    kept, _ = apply_weight_policy(examples, policy="cap1", cap_k=1, seed=7)
    assert [ex["provenance"]["uid"] for ex in kept] == ["a", "b", "c"]


# --- apply_weight_policy: capk ---------------------------------------------------


def test_capk_caps_at_k_and_keeps_smaller_uids_whole():
    examples = _flock({"a": 1, "b": 3, "c": 7})
    kept, block = apply_weight_policy(examples, policy="capk", cap_k=3, seed=42)

    per_uid = {}
    for ex in kept:
        per_uid[ex["provenance"]["uid"]] = per_uid.get(ex["provenance"]["uid"], 0) + 1
    assert per_uid == {"a": 1, "b": 3, "c": 3}
    assert block["label"] == "cap3"
    assert block["cap_k"] == 3
    assert block["rows_after"] == 7
    assert all("weight" not in ex for ex in kept)


def test_capk_requires_positive_int_cap():
    examples = _flock({"a": 2})
    with pytest.raises(WeightPolicyError, match="positive int"):
        apply_weight_policy(examples, policy="capk", cap_k=0, seed=1)
    with pytest.raises(WeightPolicyError, match="positive int"):
        apply_weight_policy(examples, policy="capk", cap_k=True, seed=1)


# --- apply_weight_policy: inverse ------------------------------------------------


def test_inverse_keeps_all_rows_with_exact_reciprocal_weights():
    examples = _flock({"a": 1, "b": 4, "c": 7})
    kept, block = apply_weight_policy(examples, policy="inverse", cap_k=3, seed=42)

    assert len(kept) == 12  # nothing dropped
    by_uid = {}
    for ex in kept:
        by_uid.setdefault(ex["provenance"]["uid"], []).append(ex)
    for uid, n in (("a", 1), ("b", 4), ("c", 7)):
        assert all(ex["weight"] == 1.0 / n for ex in by_uid[uid])
    assert block["weighted"] is True
    assert block["rows_after"] == block["rows_before"] == 12
    # input examples are never mutated in place
    assert all("weight" not in ex for ex in examples)


def test_unknown_policy_refused():
    with pytest.raises(WeightPolicyError, match="unknown weight policy"):
        apply_weight_policy(_flock({"a": 1}), policy="cap0", cap_k=1, seed=1)


# --- assert_weight_policy_honored ------------------------------------------------


def test_honored_passes_on_policy_outputs():
    examples = _flock({"a": 2, "b": 5})
    for policy in ("cap1", "capk", "inverse"):
        kept, _ = apply_weight_policy(examples, policy=policy, cap_k=2, seed=9)
        assert_weight_policy_honored(kept, policy=policy, cap_k=2)


def test_honored_rejects_cap_violation():
    examples = _flock({"a": 3})
    with pytest.raises(WeightPolicyError, match="exceed the cap1 cap"):
        assert_weight_policy_honored(examples, policy="cap1", cap_k=1)
    with pytest.raises(WeightPolicyError, match="exceed the capk cap"):
        assert_weight_policy_honored(examples, policy="capk", cap_k=2)


def test_honored_rejects_stray_weight_under_cap_policy():
    examples = _flock({"a": 1})
    examples[0]["weight"] = 0.5
    with pytest.raises(WeightPolicyError, match="'weight' field"):
        assert_weight_policy_honored(examples, policy="cap1", cap_k=1)


def test_honored_rejects_wrong_or_missing_inverse_weight():
    examples = _flock({"a": 2})
    examples[0]["weight"] = 0.5
    examples[1]["weight"] = 0.25  # wrong: should be 1/2
    with pytest.raises(WeightPolicyError, match="weight != 1/2"):
        assert_weight_policy_honored(examples, policy="inverse", cap_k=1)

    examples = _flock({"a": 2})  # missing weights entirely
    with pytest.raises(WeightPolicyError):
        assert_weight_policy_honored(examples, policy="inverse", cap_k=1)


# --- assert_prompt_completion_wellformed (defect-1 schema guard) -------------------


def test_wellformed_passes_on_builder_shape():
    assert_prompt_completion_wellformed([_example("a", "a-r00")])


def test_wellformed_rejects_missing_or_misshapen_prompt():
    bad = _example("a", "a-r00")
    del bad["prompt"]
    with pytest.raises(TraceIntegrityError, match="prompt must be exactly"):
        assert_prompt_completion_wellformed([bad])

    bad = _example("a", "a-r00")
    bad["prompt"] = bad["prompt"] + [{"role": "user", "content": "extra turn"}]
    with pytest.raises(TraceIntegrityError, match="prompt must be exactly"):
        assert_prompt_completion_wellformed([bad])


def test_wellformed_rejects_wire_format_drift():
    bad = _example("a", "a-r00")
    bad["prompt"][0]["content"] = "You are a helpful assistant."
    with pytest.raises(TraceIntegrityError, match="PASS_AT_K_SYSTEM_PROMPT"):
        assert_prompt_completion_wellformed([bad])

    bad = _example("a", "a-r00")
    bad["prompt"][1]["content"] = "Prove it."  # no /no_think suffix
    with pytest.raises(TraceIntegrityError, match="no-think suffix"):
        assert_prompt_completion_wellformed([bad])


def test_wellformed_rejects_empty_or_missing_completion():
    bad = _example("a", "a-r00", output="")
    with pytest.raises(TraceIntegrityError, match="non-empty str"):
        assert_prompt_completion_wellformed([bad])

    bad = _example("a", "a-r00")
    bad["completion"] = []
    with pytest.raises(TraceIntegrityError, match="completion must be exactly"):
        assert_prompt_completion_wellformed([bad])

    bad = _example("a", "a-r00")
    bad["completion"] = bad["completion"] * 2
    with pytest.raises(TraceIntegrityError, match="completion must be exactly"):
        assert_prompt_completion_wellformed([bad])


# --- verify_written_dataset policy re-check ---------------------------------------


def test_verify_written_dataset_recheck_catches_policy_violation(tmp_path):
    import json

    examples = _flock({"a": 2})
    rollout_index = {
        (ex["provenance"]["uid"], ex["provenance"]["rollout_uid"]): {"output": "\n\nProof."}
        for ex in examples
    }
    for ex in examples:
        ex["provenance"].update({"verdict": "correct", "verbatim_output": True})
    dataset_path = tmp_path / "sft_train.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")

    # Same file: passes as inverse-shaped? No -- weights are missing, so
    # inverse refuses; and it violates cap1 (2 rows for uid a). Both ways.
    with pytest.raises(WeightPolicyError):
        build_dataset.verify_written_dataset(
            dataset_path, rollout_index, policy="cap1", cap_k=1
        )
    with pytest.raises(WeightPolicyError):
        build_dataset.verify_written_dataset(
            dataset_path, rollout_index, policy="inverse", cap_k=1
        )
    # And with no policy given, the per-row checks still run and pass.
    assert build_dataset.verify_written_dataset(dataset_path, rollout_index) == 2


# --- config knobs ------------------------------------------------------------------


def test_weight_policy_label_derivation():
    assert config.weight_policy_label("cap1", 3) == "cap1"
    assert config.weight_policy_label("capk", 3) == "cap3"
    assert config.weight_policy_label("capk", 5) == "cap5"
    assert config.weight_policy_label("inverse", 3) == "inverse"
    # defaults read the module knobs
    assert config.weight_policy_label() == config.weight_policy_label(
        config.WEIGHT_POLICY, config.WEIGHT_POLICY_CAP_K
    )


def test_validate_config_rejects_bad_weight_policy(monkeypatch):
    monkeypatch.setattr(config, "WEIGHT_POLICY", "capzero")
    with pytest.raises(config.ConfigError, match="WEIGHT_POLICY"):
        config.validate_config()


def test_validate_config_rejects_bad_new_pins(monkeypatch):
    monkeypatch.setattr(config, "GRAD_ACCUM_STEPS", 0)
    monkeypatch.setattr(config, "WARMUP_RATIO", 1.5)
    monkeypatch.setattr(config, "WEIGHT_DECAY", -0.1)
    monkeypatch.setattr(config, "LR_SCHEDULER_TYPE", "")
    with pytest.raises(config.ConfigError) as excinfo:
        config.validate_config()
    message = str(excinfo.value)
    for name in ("GRAD_ACCUM_STEPS", "WARMUP_RATIO", "WEIGHT_DECAY", "LR_SCHEDULER_TYPE"):
        assert name in message


def test_sft_dataset_path_points_at_v2_policy_dir():
    # The upload/train chain must target the v2 build, not the retired v1
    # file -- a stale pointer here would silently ship the defective
    # recipe's dataset when W3 reopens.
    assert config.SFT_DATASET_PATH == (
        config.DATA_V2_DIR / config.weight_policy_label() / "sft_train.jsonl"
    )
    assert config.DATASET_MANIFEST_PATH == (
        config.DATA_V2_DIR / config.weight_policy_label() / "dataset_manifest.json"
    )


def test_wellformed_rejects_think_tags_in_completion():
    # Latent hazard found by the 2026-07-29 masking proof: Qwen3's chat
    # template splits assistant content on '</think>' and re-normalizes the
    # pieces, so a think tag inside a stored-verbatim target would silently
    # change the trained bytes. 0/700 current traces carry one -- the guard
    # makes any future one loud.
    for payload in ("<think>\n\n</think>\n\nProof.", "text with a stray </think> inside"):
        bad = _example("a", "a-r00", output=payload)
        with pytest.raises(TraceIntegrityError, match="think"):
            assert_prompt_completion_wellformed([bad])

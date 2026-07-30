"""Tests for loratrain.build_dataset's W2 build orchestrator (guards + assembly).

Synthetic data only -- no real corpus, no network, no dependency on
out/** or evalharness/data/** existing. Mirrors test_upload_guard.py's
`env` fixture pattern: build a fully valid, hermetic environment once,
then let individual tests mutate one piece of it (a file on disk, or one
`build()` kwarg) to exercise a specific refusal path. See README "Split &
corpus", "Non-negotiable ordering & invariants", and D4 for the
invariants under test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from loratrain import build_dataset, config

# --- fixture data ------------------------------------------------------------

# (uid, source_file, arxiv_id, statement, rollout verdicts, rollout outputs)
# Two source_file groups: fake_a backs u0-u2, fake_b backs u3-u4. Sum of
# n_correct across all five records is 6 -- the happy-path row count.
TRAIN_SPECS = [
    (
        "u0", "out/fake_a/pass_at_k.jsonl", "1000.00000", "Prove statement zero.",
        ["correct", "wrong", "degenerate"],
        ["u0 correct output.", "u0 wrong output.", "u0 degenerate output."],
    ),
    (
        "u1", "out/fake_a/pass_at_k.jsonl", "1000.00001", "Prove statement one.",
        ["correct", "correct"],
        ["\n\nProof: the square root of two is √2.   ", "u1 second correct output."],
    ),
    (
        "u2", "out/fake_a/pass_at_k.jsonl", "1000.00002", "Prove statement two.",
        ["correct", "wrong", "wrong"],
        ["u2 correct output.", "u2 wrong output 1.", "u2 wrong output 2."],
    ),
    (
        "u3", "out/fake_b/pass_at_k.jsonl", "1000.00003", "Prove statement three.",
        ["correct", "degenerate"],
        ["u3 correct output.", "u3 degenerate output."],
    ),
    (
        "u4", "out/fake_b/pass_at_k.jsonl", "1000.00004", "Prove statement four.",
        ["correct"],
        ["u4 correct output."],
    ),
]

EVAL_PAPERS = ["9999.00001", "9999.00002"]

EVAL_SET_ROWS = [
    {
        "uid": "eval-u0", "statement": "Eval problem Alpha.", "answer": "eval-answer-0",
        "arxiv_id": "9999.00001", "eval_slice": "eval_band",
    },
    {
        "uid": "eval-u1", "statement": "Eval problem Beta.", "answer": "eval-answer-1",
        "arxiv_id": "9999.00002", "eval_slice": "anchor_solved",
    },
]

HAPPY_PATH_N_EXAMPLES = sum(verdicts.count("correct") for *_r, verdicts, _o in TRAIN_SPECS)

# A stale earlier-pass line for u1-r00 (append-log reality, mirrors real
# tier1_band): same rollout_uid, DIFFERENT verdict and output. The fixture
# prepends it to fake_a's rollouts, so fake_a has exactly 1 duplicate entry
# and only last-occurrence indexing reconciles u1.
STALE_DUP_LINE = {
    "uid": "u1",
    "rollout_uid": "u1-r00",
    "sample_idx": 0,
    "candidate": "stale-cand",
    "verdict": "wrong",
    "output": "STALE u1-r00 output from an earlier pass -- must never be harvested.",
    "from_cache": False,
}


# --- fixture-building helpers --------------------------------------------------


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _corpus_row(uid, source_file, arxiv_id, statement, answer, verdicts):
    return {
        "uid": uid,
        "statement": statement,
        "answer": answer,
        "arxiv_id": arxiv_id,
        "label": "band",
        "n_correct": verdicts.count("correct"),
        "n_wrong": verdicts.count("wrong"),
        "n_degenerate": verdicts.count("degenerate"),
        "pass_at_k": verdicts.count("correct") / len(verdicts),
        "rollout_uids": [f"{uid}-r{i:02d}" for i in range(len(verdicts))],
        "corpus_provenance": {"source_file": source_file},
    }


def _rollout_rows(uid, verdicts, outputs):
    return [
        {
            "uid": uid,
            "rollout_uid": f"{uid}-r{i:02d}",
            "sample_idx": i,
            "candidate": f"cand{i}",
            "verdict": verdict,
            "output": output,
            "from_cache": False,
        }
        for i, (verdict, output) in enumerate(zip(verdicts, outputs))
    ]


def _corpus_pin(corpus_path: Path):
    """Return (sha256, row_count) for a corpus file already on disk.

    Mirrors assert_corpus_pinned's own counting convention (every row
    here is always newline-terminated by _write_jsonl).
    """
    data = corpus_path.read_bytes()
    return hashlib.sha256(data).hexdigest(), data.count(b"\n")


def _split_pin(split_path: Path) -> str:
    """Full sha256 hex digest of split_path's exact bytes (no truncation).

    2026-07-26: the split's PRIMARY pin is the full sha256
    (assert_split_pinned), matching assert_corpus_pinned's style;
    load_eval_papers' own sha16 check derives its expected value from
    this via ``[:16]``, same as build()/config.py do for the real split.
    """
    return hashlib.sha256(split_path.read_bytes()).hexdigest()


def _write_split_file(env) -> None:
    """(Re)write env["split_path"] from env["eval_papers"] + env["split_backfill_uids"]

    and refresh env["expected_split_sha256"] to match. Called once by
    the `env` fixture itself and again by `_add_backfill_record` after it
    extends env["split_backfill_uids"] -- mirrors how `_add_alt_record`
    refreshes env["expected_corpus_sha256"]/["expected_corpus_rows"]
    after rewriting the corpus. Only the two keys build_dataset.py
    actually reads (`eval_papers`, `train_backfill_7of8_uids`) are
    written -- deliberately minimal vs. the real corpus_split_200_100.json
    schema (train_uids/holdout_uids/etc. are evalharness-lane fields this
    module never touches).
    """
    split_path = env["split_path"]
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "eval_papers": env["eval_papers"],
                "train_backfill_7of8_uids": env["split_backfill_uids"],
            }
        ),
        encoding="utf-8",
    )
    env["expected_split_sha256"] = _split_pin(split_path)


@pytest.fixture
def env(tmp_path):
    """Build a fully valid, hermetic W2 environment under tmp_path.

    5 train records across 2 source_file groups (fake_a: u0-u2, fake_b:
    u3-u4), each with a matching _progress/rollouts.jsonl; a 2-paper
    split file (2026-07-26 schema: eval_papers +
    train_backfill_7of8_uids, the latter EMPTY by default -- see
    `_add_backfill_record` for tests that need a non-empty backfill
    roster); a disjoint 2-row eval_set.jsonl; train_uids.txt listing all
    5 train uids. repo_root == tmp_path, so corpus_provenance.source_file
    resolves under it. Individual tests mutate one piece (rewrite a
    file, override one build() kwarg) to exercise a specific refusal
    path -- mirrors test_upload_guard.py's `env` fixture.
    """
    corpus_rows = [
        _corpus_row(uid, source_file, arxiv_id, statement, f"answer-DIFFERENT-{uid}", verdicts)
        for uid, source_file, arxiv_id, statement, verdicts, _outputs in TRAIN_SPECS
    ]
    corpus_path = tmp_path / "out" / "corpus_pde625" / "band_corpus.jsonl"
    _write_jsonl(corpus_path, corpus_rows)

    rollouts_by_file: dict = {}
    for uid, source_file, _arxiv_id, _statement, verdicts, outputs in TRAIN_SPECS:
        rollouts_by_file.setdefault(source_file, []).extend(_rollout_rows(uid, verdicts, outputs))

    rollout_paths = {}
    for source_file, rows in rollouts_by_file.items():
        rollout_path = tmp_path / Path(source_file).parent / "_progress" / "rollouts.jsonl"
        if source_file == "out/fake_a/pass_at_k.jsonl":
            # Make fake_a append-log-shaped like the real tier1_band file:
            # a STALE line for u1-r00 from an "earlier pass" (different
            # verdict AND output) precedes the clean rows. Last-occurrence
            # indexing must supersede it -- under first-occurrence u1's
            # tally would be 1 correct / 1 wrong and u1 would not
            # reconcile at all. NOTE: tests that rewrite fake_a from
            # env["rollouts_by_file"] drop this line; none of them assert
            # on duplicate behavior.
            rows = [dict(STALE_DUP_LINE)] + rows
        _write_jsonl(rollout_path, rows)
        rollout_paths[source_file] = rollout_path

    eval_set_path = tmp_path / "evalharness" / "data" / "eval_set.jsonl"
    _write_jsonl(eval_set_path, EVAL_SET_ROWS)

    train_uids_path = tmp_path / "evalharness" / "data" / "train_uids.txt"
    train_uids_path.parent.mkdir(parents=True, exist_ok=True)
    train_uids_path.write_text("".join(f"{uid}\n" for uid, *_rest in TRAIN_SPECS), encoding="utf-8")

    corpus_sha256, corpus_rows_n = _corpus_pin(corpus_path)

    env = {
        "tmp_path": tmp_path,
        "repo_root": tmp_path,
        "corpus_path": corpus_path,
        "split_path": tmp_path / "evalharness" / "data" / "corpus_split_200_100.json",
        "train_uids_path": train_uids_path,
        "eval_set_path": eval_set_path,
        "output_dir": tmp_path / "built",
        "expected_corpus_sha256": corpus_sha256,
        "expected_corpus_rows": corpus_rows_n,
        "eval_papers": list(EVAL_PAPERS),
        "split_backfill_uids": [],
        "backfill_trace_sources": {},
        "seed": 424242,
        "corpus_rows": corpus_rows,
        "rollout_paths": rollout_paths,
        "rollouts_by_file": rollouts_by_file,
        "train_uids": [uid for uid, *_rest in TRAIN_SPECS],
    }
    _write_split_file(env)  # sets env["expected_split_sha256"] too
    return env


def _build_kwargs(env, **overrides):
    kwargs = dict(
        corpus_path=env["corpus_path"],
        split_path=env["split_path"],
        train_uids_path=env["train_uids_path"],
        eval_set_path=env["eval_set_path"],
        output_dir=env["output_dir"],
        expected_corpus_sha256=env["expected_corpus_sha256"],
        expected_corpus_rows=env["expected_corpus_rows"],
        expected_split_sha256=env["expected_split_sha256"],
        backfill_trace_sources=env["backfill_trace_sources"],
        seed=env["seed"],
        # Default policy for the pre-v2 test scenarios: inverse keeps EVERY
        # harvested trace (weights aside), so all the row-level expectations
        # written against the v1 one-row-per-trace harvest hold unchanged.
        # cap1/capk behavior gets its own dedicated tests below.
        weight_policy=env.get("weight_policy", "inverse"),
        weight_policy_cap_k=env.get("weight_policy_cap_k", 3),
        repo_root=env["repo_root"],
    )
    kwargs.update(overrides)
    return kwargs


def _add_backfill_record(
    env,
    uid="b0",
    source_dir="out/backfill_src",
    arxiv_id="7000.00001",
    statement="Prove the backfill statement.",
    verdicts=None,
    outputs=None,
    label="solved",
    add_to_train_uids=True,
):
    """Extend env with ONE GGUF-7/8-backfill uid (Nicky's ruling 2026-07-26):

    a synthetic pass_at_k.jsonl row (default n_correct=7/n_wrong=1/
    n_degenerate=0, label="solved" -- the "GGUF 7/8" guarantee) at a
    pinned source path OUTSIDE band_corpus.jsonl, with a matching
    sibling _progress/rollouts.jsonl, registered into both
    env["split_path"]'s train_backfill_7of8_uids AND
    env["backfill_trace_sources"] -- mirrors config.BACKFILL_TRACE_SOURCES
    + the split's declared roster. ``add_to_train_uids`` (default True)
    also appends the uid to train_uids.txt, since build() only harvests
    a pinned backfill uid if it is ALSO part of this build's train_uids
    (symmetric with a corpus-resident uid outside train_uids.txt simply
    not being selected) -- pass False to exercise a uid that is pinned
    but not part of this build. Callers that want to break one piece
    (wrong n_correct/label, missing file, tally mismatch, mapping
    desync) mutate the returned dict's files, or env["backfill_trace_sources"]
    / the split file, afterward. Returns {"uid", "source_file",
    "pass_at_k_path", "rollout_path", "statement"}.
    """
    tmp_path = env["tmp_path"]
    if verdicts is None:
        verdicts = ["correct"] * 7 + ["wrong"]
    if outputs is None:
        outputs = [f"{uid} correct output {i}." for i in range(7)] + [f"{uid} wrong output."]

    source_file = f"{source_dir}/pass_at_k.jsonl"
    row = _corpus_row(uid, source_file, arxiv_id, statement, f"answer-DIFFERENT-{uid}", verdicts)
    del row["corpus_provenance"]  # real backfill source rows carry no corpus_provenance
    row["label"] = label
    row["provenance"] = "extracted"  # flat status string, mirrors the real tier1_band/tier2_7of8 rows

    pass_at_k_path = tmp_path / source_file
    _write_jsonl(pass_at_k_path, [row])

    rollout_path = tmp_path / Path(source_file).parent / "_progress" / "rollouts.jsonl"
    _write_jsonl(rollout_path, _rollout_rows(uid, verdicts, outputs))

    env["split_backfill_uids"] = env["split_backfill_uids"] + [uid]
    env["backfill_trace_sources"] = dict(env["backfill_trace_sources"], **{uid: source_file})
    _write_split_file(env)

    if add_to_train_uids:
        env["train_uids"] = env["train_uids"] + [uid]
        env["train_uids_path"].write_text(
            "".join(f"{u}\n" for u in env["train_uids"]), encoding="utf-8"
        )

    return {
        "uid": uid,
        "source_file": source_file,
        "pass_at_k_path": pass_at_k_path,
        "rollout_path": rollout_path,
        "statement": statement,
    }


# --- 1. happy path -------------------------------------------------------------


def test_happy_path_rows_messages_and_manifest(env):
    manifest = build_dataset.build(**_build_kwargs(env))

    dataset_path = Path(manifest["dataset"]["path"])
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]

    assert HAPPY_PATH_N_EXAMPLES == 6
    assert len(rows) == HAPPY_PATH_N_EXAMPLES

    statement_by_uid = {uid: statement for uid, _sf, _aid, statement, _v, _o in TRAIN_SPECS}

    for row in rows:
        prov = row["provenance"]
        uid = prov["uid"]
        assert row["prompt"][0] == {
            "role": "system",
            "content": "Solve the problem. State only the final answer inside \\boxed{}.",
        }
        assert row["prompt"][1] == {"role": "user", "content": statement_by_uid[uid] + " /no_think"}
        assert len(row["prompt"]) == 2
        assert len(row["completion"]) == 1
        assert row["completion"][0]["role"] == "assistant"
        assert "messages" not in row  # schema v2: prompt/completion replace messages
        assert prov["verdict"] == "correct"
        assert prov["verbatim_output"] is True
        assert prov["corpus_sha256"] == env["expected_corpus_sha256"]
        assert isinstance(prov["sample_idx"], int)

    weird_row = next(
        r for r in rows if r["provenance"]["uid"] == "u1" and r["provenance"]["rollout_uid"] == "u1-r00"
    )
    # Last-occurrence indexing: the harvested u1-r00 is the CLEAN final
    # line, byte-identical, and the stale earlier-pass content appears
    # nowhere in the dataset.
    assert weird_row["completion"][0]["content"] == "\n\nProof: the square root of two is √2.   "
    assert weird_row["provenance"]["sample_idx"] == 0
    assert all(r["completion"][0]["content"] != STALE_DUP_LINE["output"] for r in rows)

    for row in rows:
        prov = row["provenance"]
        assert prov["reconciled_via"] == "routed"
        assert prov["trace_file"] == str(
            Path(prov["source_file"]).parent / "_progress" / "rollouts.jsonl"
        )

    per_uid = manifest["dataset"]["per_uid_trace_counts"]
    assert set(per_uid) == set(env["train_uids"])
    assert per_uid == {"u0": 1, "u1": 2, "u2": 1, "u3": 1, "u4": 1}

    assert manifest["seed"] == env["seed"]
    assert manifest["corpus"]["sha256"] == env["expected_corpus_sha256"]
    assert manifest["eval_paper_split"]["sha256"] == env["expected_split_sha256"]
    assert manifest["dataset"]["sha256"] == build_dataset.sha256_file(dataset_path)
    assert manifest["dataset"]["rows"] == HAPPY_PATH_N_EXAMPLES
    assert manifest["guards"] == list(build_dataset.BUILD_GUARD_STEPS)

    backfill = manifest["backfill_7of8"]
    assert backfill == {"uids": [], "sources": [], "per_uid_trace_counts": {}}

    recon = manifest["reconciliation"]
    assert recon["routed"] == len(env["train_uids"])
    assert recon["unique_alternative"] == 0
    assert set(recon["per_uid"]) == set(env["train_uids"])
    registry = {entry["path"]: entry for entry in recon["registry"]}
    assert registry["out/fake_a/_progress/rollouts.jsonl"]["duplicate_entries"] == 1
    assert registry["out/fake_b/_progress/rollouts.jsonl"]["duplicate_entries"] == 0


# --- 2. determinism --------------------------------------------------------------


def test_determinism_two_builds_byte_identical_dataset(env):
    m1 = build_dataset.build(**_build_kwargs(env, output_dir=env["tmp_path"] / "built1"))
    m2 = build_dataset.build(**_build_kwargs(env, output_dir=env["tmp_path"] / "built2"))

    assert Path(m1["dataset"]["path"]).read_bytes() == Path(m2["dataset"]["path"]).read_bytes()
    assert m1["dataset"]["sha256"] == m2["dataset"]["sha256"]


# --- 3. corpus answer never in targets --------------------------------------------


def test_answer_never_leaks_into_targets(env):
    manifest = build_dataset.build(**_build_kwargs(env))
    dataset_path = Path(manifest["dataset"]["path"])
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]

    answers = {row["answer"] for row in env["corpus_rows"]}
    for row in rows:
        assert row["completion"][0]["content"] not in answers


def test_build_succeeds_with_no_answer_key_at_all(env):
    stripped_rows = []
    for row in env["corpus_rows"]:
        row = dict(row)
        del row["answer"]
        stripped_rows.append(row)
    _write_jsonl(env["corpus_path"], stripped_rows)
    sha256, rows_n = _corpus_pin(env["corpus_path"])

    manifest = build_dataset.build(
        **_build_kwargs(env, expected_corpus_sha256=sha256, expected_corpus_rows=rows_n)
    )
    assert manifest["dataset"]["rows"] == HAPPY_PATH_N_EXAMPLES


# --- 4-6. pin mismatches ----------------------------------------------------------


def test_corpus_sha_mismatch_raises_pin_mismatch_and_writes_nothing(env):
    with pytest.raises(build_dataset.PinMismatchError):
        build_dataset.build(**_build_kwargs(env, expected_corpus_sha256="0" * 64))
    assert not env["output_dir"].exists()


def test_corpus_row_count_mismatch_raises_pin_mismatch(env):
    with pytest.raises(build_dataset.PinMismatchError):
        build_dataset.build(**_build_kwargs(env, expected_corpus_rows=999))
    assert not env["output_dir"].exists()


def test_split_sha_mismatch_raises_pin_mismatch(env):
    # assert_split_pinned -- the PRIMARY, full-sha256 pin (2026-07-26).
    with pytest.raises(build_dataset.PinMismatchError):
        build_dataset.build(**_build_kwargs(env, expected_split_sha256="0" * 64))
    assert not env["output_dir"].exists()


# --- 7-9. split-not-built / retired path ------------------------------------------


def test_missing_train_uids_raises_split_not_built(env):
    env["train_uids_path"].unlink()
    with pytest.raises(build_dataset.SplitNotBuiltError, match="evalharness-build-set"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_missing_eval_set_raises_split_not_built(env):
    env["eval_set_path"].unlink()
    with pytest.raises(build_dataset.SplitNotBuiltError, match="evalharness-build-set"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_retired_path_component_raises_split_not_built_even_if_pins_match(env):
    retired_train_uids = env["tmp_path"] / "evalharness" / "data" / "retired_20260716" / "train_uids.txt"
    retired_train_uids.parent.mkdir(parents=True, exist_ok=True)
    retired_train_uids.write_text(env["train_uids_path"].read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(build_dataset.SplitNotBuiltError, match="(?i)retired"):
        build_dataset.build(**_build_kwargs(env, train_uids_path=retired_train_uids))
    assert not env["output_dir"].exists()


# --- 10-11. train_uids.txt corruption ----------------------------------------------


def test_train_uid_missing_from_corpus_raises_value_error(env):
    text = env["train_uids_path"].read_text(encoding="utf-8")
    env["train_uids_path"].write_text(text + "u-does-not-exist\n", encoding="utf-8")

    with pytest.raises(ValueError):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_duplicate_train_uid_raises_value_error(env):
    text = env["train_uids_path"].read_text(encoding="utf-8")
    env["train_uids_path"].write_text(text + "u0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_duplicate_corpus_uid_raises_value_error(env):
    # A dup-uid corpus would select several rows per train uid, and
    # dedupe_examples downstream would silently mask the collision --
    # select_train_records must refuse instead. The duplicate row gets a
    # DIFFERENT statement so the cross-uid statement-dup guard cannot be
    # what fires here.
    rows = [dict(r) for r in env["corpus_rows"]]
    dup = dict(rows[4])  # second row claiming uid u4
    dup["statement"] = "A different statement for the impostor u4 row."
    rows.append(dup)
    _write_jsonl(env["corpus_path"], rows)
    sha256, rows_n = _corpus_pin(env["corpus_path"])

    with pytest.raises(ValueError, match="duplicate uid"):
        build_dataset.build(**_build_kwargs(env, expected_corpus_sha256=sha256, expected_corpus_rows=rows_n))
    assert not env["output_dir"].exists()


# --- 12-15. leakage / duplicate statements -----------------------------------------


def test_train_uid_also_in_eval_set_raises_leakage_error(env):
    text = env["train_uids_path"].read_text(encoding="utf-8")
    env["train_uids_path"].write_text(text + "eval-u0\n", encoding="utf-8")

    with pytest.raises(build_dataset.LeakageError):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_train_record_arxiv_id_is_eval_paper_raises_leakage_error(env):
    rows = [dict(r) for r in env["corpus_rows"]]
    rows[0]["arxiv_id"] = "9999.00001"  # an eval paper
    _write_jsonl(env["corpus_path"], rows)
    sha256, rows_n = _corpus_pin(env["corpus_path"])

    with pytest.raises(build_dataset.LeakageError):
        build_dataset.build(**_build_kwargs(env, expected_corpus_sha256=sha256, expected_corpus_rows=rows_n))
    assert not env["output_dir"].exists()


def test_cross_uid_statement_dup_raises_duplicate_record_error(env):
    rows = [dict(r) for r in env["corpus_rows"]]
    rows[1]["statement"] = rows[0]["statement"]  # u1 now shares u0's statement
    _write_jsonl(env["corpus_path"], rows)
    sha256, rows_n = _corpus_pin(env["corpus_path"])

    with pytest.raises(build_dataset.DuplicateRecordError):
        build_dataset.build(**_build_kwargs(env, expected_corpus_sha256=sha256, expected_corpus_rows=rows_n))
    assert not env["output_dir"].exists()


def test_train_statement_matches_eval_statement_raises_leakage_error(env):
    rows = [dict(r) for r in env["corpus_rows"]]
    rows[2]["statement"] = EVAL_SET_ROWS[0]["statement"]
    _write_jsonl(env["corpus_path"], rows)
    sha256, rows_n = _corpus_pin(env["corpus_path"])

    with pytest.raises(build_dataset.LeakageError):
        build_dataset.build(**_build_kwargs(env, expected_corpus_sha256=sha256, expected_corpus_rows=rows_n))
    assert not env["output_dir"].exists()


def test_eval_row_without_statement_raises_value_error(env):
    # A statement-less eval row must hard-fail, not be skipped: silently
    # dropping it would shrink the statement set the cross-split leakage
    # check relies on (no warn mode -- build_eval_set already guarantees
    # non-blank statements on every eval row, so this only fires on a
    # malformed eval set).
    rows = [dict(r) for r in EVAL_SET_ROWS]
    del rows[1]["statement"]
    _write_jsonl(env["eval_set_path"], rows)

    with pytest.raises(ValueError, match="statement"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


# --- 16-21. rollout / trace integrity ----------------------------------------------


def test_missing_rollouts_file_raises_trace_integrity_error(env):
    env["rollout_paths"]["out/fake_b/pass_at_k.jsonl"].unlink()

    with pytest.raises(build_dataset.TraceIntegrityError):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_rollout_uid_absent_from_rollouts_file_raises_trace_integrity_error(env):
    rows = env["rollouts_by_file"]["out/fake_a/pass_at_k.jsonl"]
    trimmed = [r for r in rows if r["rollout_uid"] != "u0-r01"]  # drop u0's wrong rollout
    _write_jsonl(env["rollout_paths"]["out/fake_a/pass_at_k.jsonl"], trimmed)

    with pytest.raises(build_dataset.TraceIntegrityError):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_verdict_tally_mismatch_raises_trace_integrity_error(env):
    rows = [dict(r) for r in env["rollouts_by_file"]["out/fake_a/pass_at_k.jsonl"]]
    for r in rows:
        if r["rollout_uid"] == "u0-r00":
            r["verdict"] = "wrong"  # corpus row still says u0 has n_correct=1
    _write_jsonl(env["rollout_paths"]["out/fake_a/pass_at_k.jsonl"], rows)

    with pytest.raises(build_dataset.TraceIntegrityError):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_harvest_forged_rollout_uid_raises_trace_integrity_error(tmp_path):
    # load_rollout_file keys each row as (row["uid"], row["rollout_uid"]),
    # so a lookup via the full pipeline can never surface a row whose own
    # "uid" field disagrees with the key used to find it -- that guard in
    # harvest_correct_traces is defense in depth (same idiom as build()'s
    # nested-shape assert_no_leakage re-check) and is only reachable by
    # exercising harvest_correct_traces directly with a hand-built
    # authoritative map that violates the invariant load_rollout_file
    # would normally hold.
    record = {
        "uid": "u1",
        "statement": "s",
        "arxiv_id": "a1",
        "n_correct": 1,
        "n_wrong": 0,
        "n_degenerate": 0,
        "rollout_uids": ["u1-r00"],
        "corpus_provenance": {"source_file": "out/x/pass_at_k.jsonl"},
    }
    forged_index = {
        ("u1", "u1-r00"): {
            "uid": "u2",  # forged: disagrees with the lookup key's uid half
            "rollout_uid": "u1-r00",
            "sample_idx": 0,
            "verdict": "correct",
            "output": "some output",
        }
    }
    authoritative = {
        "u1": (forged_index, tmp_path / "out/x/_progress/rollouts.jsonl", "routed")
    }
    with pytest.raises(build_dataset.TraceIntegrityError):
        build_dataset.harvest_correct_traces([record], authoritative, "corpus-sha", tmp_path)


def test_correct_rollout_empty_output_raises_trace_integrity_error(env):
    rows = [dict(r) for r in env["rollouts_by_file"]["out/fake_b/pass_at_k.jsonl"]]
    for r in rows:
        if r["rollout_uid"] == "u4-r00":
            r["output"] = ""
    _write_jsonl(env["rollout_paths"]["out/fake_b/pass_at_k.jsonl"], rows)

    with pytest.raises(build_dataset.TraceIntegrityError):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_zero_correct_record_raises_trace_integrity_error(env):
    corpus_rows = [dict(r) for r in env["corpus_rows"]]
    for row in corpus_rows:
        if row["uid"] == "u4":
            row["n_correct"] = 0
            row["n_wrong"] = 1
            row["n_degenerate"] = 0
            row["pass_at_k"] = 0.0
    _write_jsonl(env["corpus_path"], corpus_rows)
    sha256, rows_n = _corpus_pin(env["corpus_path"])

    rollout_rows = [dict(r) for r in env["rollouts_by_file"]["out/fake_b/pass_at_k.jsonl"]]
    for r in rollout_rows:
        if r["rollout_uid"] == "u4-r00":
            r["verdict"] = "wrong"  # tallies stay consistent: n_correct=0, n_wrong=1
    _write_jsonl(env["rollout_paths"]["out/fake_b/pass_at_k.jsonl"], rollout_rows)

    with pytest.raises(build_dataset.TraceIntegrityError):
        build_dataset.build(**_build_kwargs(env, expected_corpus_sha256=sha256, expected_corpus_rows=rows_n))
    assert not env["output_dir"].exists()


# --- rollout reconciliation (append-log reality; module docstring
# "Rollout reconciliation": last-occurrence indexing, routed-or-unique-
# alternative resolution, ambiguity refusal) ------------------------------------------


def test_load_rollout_file_last_occurrence_wins_and_counts_dups(env):
    path = env["rollout_paths"]["out/fake_a/pass_at_k.jsonl"]
    index, duplicate_entries = build_dataset.load_rollout_file(path)

    assert duplicate_entries == 1  # the STALE_DUP_LINE prepended by the fixture
    # Last occurrence wins: the clean final u1-r00 line, not the stale one.
    assert index[("u1", "u1-r00")]["verdict"] == "correct"
    assert index[("u1", "u1-r00")]["output"] == "\n\nProof: the square root of two is √2.   "


def _add_alt_record(env, alt_dir="out/remote_rescore/fake_alt"):
    """Extend env with u5: routed file exists but does NOT reconcile; the
    reconciling rollouts live in a glob-discoverable registry file
    (mirrors the 15 real rows whose traces live in rerun dirs their
    source_file never names). Returns the alt rollouts path."""
    tmp_path = env["tmp_path"]

    u5_row = _corpus_row(
        "u5", "out/fake_c/pass_at_k.jsonl", "1000.00005",
        "Prove statement five.", "answer-DIFFERENT-u5", ["correct", "wrong"],
    )
    corpus_rows = env["corpus_rows"] + [u5_row]
    _write_jsonl(env["corpus_path"], corpus_rows)
    env["corpus_rows"] = corpus_rows
    env["expected_corpus_sha256"], env["expected_corpus_rows"] = _corpus_pin(env["corpus_path"])

    env["train_uids"] = env["train_uids"] + ["u5"]
    env["train_uids_path"].write_text(
        "".join(f"{uid}\n" for uid in env["train_uids"]), encoding="utf-8"
    )

    # Routed file: rollout_uids all present, uids match, but verdicts are
    # (wrong, wrong) vs the corpus row's (1 correct, 1 wrong) -- a stale
    # pass that does NOT reconcile.
    routed_rows = _rollout_rows("u5", ["wrong", "wrong"], ["u5 stale wrong 0.", "u5 stale wrong 1."])
    _write_jsonl(tmp_path / "out/fake_c/_progress/rollouts.jsonl", routed_rows)

    # The reconciling pass, in a file only REGISTRY_GLOBS discovery finds.
    alt_rows = _rollout_rows("u5", ["correct", "wrong"], ["u5 CORRECT output from the rerun pass.", "u5 wrong rerun."])
    alt_path = tmp_path / alt_dir / "_progress" / "rollouts.jsonl"
    _write_jsonl(alt_path, alt_rows)
    return alt_path


def test_unique_alternative_reconciliation_harvests_from_alt_file(env):
    _add_alt_record(env)
    manifest = build_dataset.build(**_build_kwargs(env))

    assert manifest["dataset"]["rows"] == HAPPY_PATH_N_EXAMPLES + 1
    assert manifest["reconciliation"]["routed"] == 5
    assert manifest["reconciliation"]["unique_alternative"] == 1
    assert manifest["reconciliation"]["per_uid"]["u5"] == {
        "trace_file": "out/remote_rescore/fake_alt/_progress/rollouts.jsonl",
        "reconciled_via": "unique_alternative",
    }
    registry_paths = {entry["path"] for entry in manifest["reconciliation"]["registry"]}
    assert "out/remote_rescore/fake_alt/_progress/rollouts.jsonl" in registry_paths
    assert "out/fake_c/_progress/rollouts.jsonl" in registry_paths

    rows = [
        json.loads(line)
        for line in Path(manifest["dataset"]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    u5_rows = [r for r in rows if r["provenance"]["uid"] == "u5"]
    assert len(u5_rows) == 1
    assert u5_rows[0]["completion"][0]["content"] == "u5 CORRECT output from the rerun pass."
    assert u5_rows[0]["provenance"]["reconciled_via"] == "unique_alternative"
    assert u5_rows[0]["provenance"]["trace_file"] == "out/remote_rescore/fake_alt/_progress/rollouts.jsonl"
    # source_file stays the corpus row's original (stale) claim -- the two
    # fields are separately auditable on purpose.
    assert u5_rows[0]["provenance"]["source_file"] == "out/fake_c/pass_at_k.jsonl"


def test_missing_routed_file_falls_through_to_unique_alternative(env):
    # Cross-review gate: a routed file absent from disk is treated as
    # non-reconciling (never a bare file error mid-resolution) -- the
    # record must still resolve via the unique registry alternative.
    _add_alt_record(env)
    (env["tmp_path"] / "out/fake_c/_progress/rollouts.jsonl").unlink()

    manifest = build_dataset.build(**_build_kwargs(env))

    assert manifest["reconciliation"]["unique_alternative"] == 1
    assert manifest["reconciliation"]["per_uid"]["u5"]["reconciled_via"] == "unique_alternative"
    registry_paths = {entry["path"] for entry in manifest["reconciliation"]["registry"]}
    assert "out/fake_c/_progress/rollouts.jsonl" not in registry_paths  # missing file never loaded


def test_missing_routed_file_with_no_alternative_names_it_in_refusal(env):
    # Zero-candidate refusal must say the routed file is MISSING and give
    # expected counts -- not surface a FileNotFoundError.
    (env["tmp_path"] / "out/fake_b/_progress/rollouts.jsonl").unlink()

    with pytest.raises(
        build_dataset.TraceIntegrityError, match="MISSING from disk"
    ) as excinfo:
        build_dataset.build(**_build_kwargs(env))
    assert "Expected counts" in str(excinfo.value)
    assert not env["output_dir"].exists()


def test_ambiguous_reconciliation_raises_trace_integrity_error(env):
    _add_alt_record(env)
    # A SECOND glob-discoverable file that also reconciles u5 -> 2
    # candidates -> refuse; guessing would un-anchor the label<->trace tie.
    alt2_rows = _rollout_rows("u5", ["correct", "wrong"], ["u5 rival correct.", "u5 rival wrong."])
    _write_jsonl(env["tmp_path"] / "out/remote_rescore/fake_alt2/_progress/rollouts.jsonl", alt2_rows)

    with pytest.raises(build_dataset.TraceIntegrityError, match="ambiguous"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_zero_reconciliation_raises_trace_integrity_error(env):
    alt_path = _add_alt_record(env)
    # Break the one reconciling file too: now u5 reconciles nowhere.
    broken = _rollout_rows("u5", ["degenerate", "wrong"], ["u5 broken.", "u5 broken 2."])
    _write_jsonl(alt_path, broken)

    with pytest.raises(build_dataset.TraceIntegrityError, match="NO registry file"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


# --- 22. late-stage failure writes nothing ------------------------------------------


def test_late_stage_failure_writes_nothing(env):
    """Same class of failure as the tally-mismatch test, from the 'nothing
    written' angle: harvest_correct_traces (step 11) is one of the latest
    guards that can still fire before anything hits disk (step 14)."""
    rows = [dict(r) for r in env["rollouts_by_file"]["out/fake_a/pass_at_k.jsonl"]]
    for r in rows:
        if r["rollout_uid"] == "u2-r00":
            r["verdict"] = "wrong"  # corpus row still says u2 has n_correct=1
    _write_jsonl(env["rollout_paths"]["out/fake_a/pass_at_k.jsonl"], rows)

    assert not env["output_dir"].exists()
    with pytest.raises(build_dataset.TraceIntegrityError):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_routed_precedence_pins_over_reconciling_alternative(env):
    # Cross-review recommendation (b): a reconciling ROUTED file wins
    # WITHOUT consulting alternatives -- the provenance claim is honored
    # when consistent. Mirror u0's exact rollouts into a glob-discoverable
    # registry file: both files now reconcile u0, and that must resolve
    # via=routed with no ambiguity error. A future refactor that inverts
    # or removes the short-circuit fails here.
    u0_rows = [r for r in env["rollouts_by_file"]["out/fake_a/pass_at_k.jsonl"] if r["uid"] == "u0"]
    _write_jsonl(env["tmp_path"] / "out/remote_rescore/fake_mirror/_progress/rollouts.jsonl", u0_rows)

    manifest = build_dataset.build(**_build_kwargs(env))

    assert manifest["reconciliation"]["per_uid"]["u0"] == {
        "trace_file": "out/fake_a/_progress/rollouts.jsonl",
        "reconciled_via": "routed",
    }
    assert manifest["reconciliation"]["unique_alternative"] == 0


def test_verify_failure_publishes_no_final_artifacts(env, monkeypatch):
    # Atomic-publish property (cross-review suggestion): if the post-write
    # disk verification fails, neither sft_train.jsonl nor the manifest
    # may exist under their final names; the .tmp stays for forensics.
    monkeypatch.setattr(build_dataset, "verify_written_dataset", lambda *_a, **_k: -1)

    with pytest.raises(build_dataset.TraceIntegrityError, match="NOT published"):
        build_dataset.build(**_build_kwargs(env))

    assert not (env["output_dir"] / "sft_train.jsonl").exists()
    assert not (env["output_dir"] / "dataset_manifest.json").exists()
    assert (env["output_dir"] / "sft_train.jsonl.tmp").exists()


# --- 23. wire-format tripwire ---------------------------------------------------


def test_wire_format_tripwire_matches_pass_at_k_literals():
    # Frozen copies (config.py) that must mirror icepick's pass_at_k wire
    # format byte-for-byte -- src/icepick/processing/pass_at_k/config.py's
    # SYSTEM_PROMPT and backends/qwen_http.py's " /no_think" suffix. If
    # pass@k's wire format ever changes, BOTH sides must move together in
    # one deliberate edit (see config.py's comment above these pins);
    # this test is the tripwire that would catch a one-sided drift.
    assert config.PASS_AT_K_SYSTEM_PROMPT == "Solve the problem. State only the final answer inside \\boxed{}."
    assert config.PASS_AT_K_NO_THINK_SUFFIX == " /no_think"


def test_wire_format_matches_icepick_source_when_present():
    # Cross-review recommendation (a): the frozen-literal tripwire above
    # cannot catch a ONE-SIDED change on the icepick side. When the
    # icepick source tree is present (it is, in this repo layout), extract
    # the actual literals and compare; skip in a hermetic env without it.
    import ast
    import re

    passk_config = config.REPO_ROOT / "src/icepick/processing/pass_at_k/config.py"
    qwen_http = config.REPO_ROOT / "src/icepick/processing/pass_at_k/backends/qwen_http.py"
    if not passk_config.exists() or not qwen_http.exists():
        pytest.skip("icepick pass@k source not present (hermetic env)")

    m = re.search(r'^SYSTEM_PROMPT\s*=\s*(".*")\s*$', passk_config.read_text(encoding="utf-8"), re.M)
    assert m, "SYSTEM_PROMPT literal not found in icepick pass@k config"
    assert ast.literal_eval(m.group(1)) == config.PASS_AT_K_SYSTEM_PROMPT

    m = re.search(r'\+\s*(" /no_think")', qwen_http.read_text(encoding="utf-8"))
    assert m, '" /no_think" suffix literal not found in qwen_http backend'
    assert ast.literal_eval(m.group(1)) == config.PASS_AT_K_NO_THINK_SUFFIX


def test_cli_exposes_no_guard_bypass_flags():
    # Cross-review recommendation (c): the no-pin-override/no-guard-skip
    # CLI rule, pinned as a test. Any future --force/--allow-*/--skip-*/
    # --*-sha/--*-pin style option fails here by name.
    import re

    banned = re.compile(r"(force|allow|skip|override|unsafe|no-verify|no-guard|sha|pin)", re.I)
    parser = build_dataset.build_arg_parser()
    option_strings = [s for action in parser._actions for s in action.option_strings]
    offenders = [s for s in option_strings if banned.search(s)]
    assert not offenders, f"guard-bypass-shaped CLI options found: {offenders}"


# --- 24-25. main() ------------------------------------------------------------------


def test_main_happy_path_writes_dataset_and_manifest(env, monkeypatch, capsys):
    monkeypatch.setattr(config, "EXPECTED_CORPUS_SHA256", env["expected_corpus_sha256"])
    monkeypatch.setattr(config, "EXPECTED_CORPUS_ROWS", env["expected_corpus_rows"])
    monkeypatch.setattr(config, "EXPECTED_SPLIT_SHA256", env["expected_split_sha256"])
    monkeypatch.setattr(config, "BACKFILL_TRACE_SOURCES", env["backfill_trace_sources"])
    monkeypatch.setattr(config, "SEED", env["seed"])

    rc = build_dataset.main(
        [
            "--corpus", str(env["corpus_path"]),
            "--split", str(env["split_path"]),
            "--train-uids", str(env["train_uids_path"]),
            "--eval-set", str(env["eval_set_path"]),
            "--output-dir", str(env["output_dir"]),
            "--repo-root", str(env["repo_root"]),
        ]
    )

    assert rc == 0
    assert (env["output_dir"] / "sft_train.jsonl").exists()
    assert (env["output_dir"] / "dataset_manifest.json").exists()
    summary = json.loads(capsys.readouterr().out)
    # main() builds under config.WEIGHT_POLICY -- default cap1: one trace
    # per uid, so 5 rows (u1's second trace capped away), not the 6-row
    # uncapped harvest.
    assert config.WEIGHT_POLICY == "cap1"
    assert summary["rows"] == len(TRAIN_SPECS)
    assert summary["weight_policy"] == {
        "policy": "cap1",
        "label": "cap1",
        "rows_before": HAPPY_PATH_N_EXAMPLES,
        "rows_after": len(TRAIN_SPECS),
    }


def test_main_missing_split_propagates_split_not_built_error(env, monkeypatch):
    nonexistent_train_uids = env["tmp_path"] / "no_such_dir" / "train_uids.txt"
    nonexistent_eval_set = env["tmp_path"] / "no_such_dir" / "eval_set.jsonl"

    monkeypatch.setattr(config, "CORPUS_PATH", env["corpus_path"])
    monkeypatch.setattr(config, "EVAL_PAPER_SPLIT_PATH", env["split_path"])
    monkeypatch.setattr(config, "TRAIN_UIDS_PATH", nonexistent_train_uids)
    monkeypatch.setattr(config, "EVAL_SET_PATH", nonexistent_eval_set)
    monkeypatch.setattr(config, "DATA_DIR", env["tmp_path"] / "data_default")
    monkeypatch.setattr(config, "EXPECTED_CORPUS_SHA256", env["expected_corpus_sha256"])
    monkeypatch.setattr(config, "EXPECTED_CORPUS_ROWS", env["expected_corpus_rows"])
    monkeypatch.setattr(config, "EXPECTED_SPLIT_SHA256", env["expected_split_sha256"])
    monkeypatch.setattr(config, "BACKFILL_TRACE_SOURCES", env["backfill_trace_sources"])
    monkeypatch.setattr(config, "SEED", env["seed"])

    # main() must not catch this -- refusals propagate to the caller.
    with pytest.raises(build_dataset.SplitNotBuiltError):
        build_dataset.main([])


# --- 26+. GGUF 7/8 backfill (Nicky's ruling 2026-07-26) -----------------------------


def test_backfill_happy_path_harvested_with_provenance_flag(env):
    added = _add_backfill_record(env)

    manifest = build_dataset.build(**_build_kwargs(env))

    assert manifest["dataset"]["rows"] == HAPPY_PATH_N_EXAMPLES + 7  # b0: 7/8 correct
    rows = [
        json.loads(line)
        for line in Path(manifest["dataset"]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    b0_rows = [r for r in rows if r["provenance"]["uid"] == "b0"]
    assert len(b0_rows) == 7
    for row in b0_rows:
        prov = row["provenance"]
        assert prov["backfill_7of8"] is True
        assert prov["verdict"] == "correct"
        assert prov["verbatim_output"] is True
        assert prov["source_file"] == added["source_file"]
        assert prov["reconciled_via"] == "routed"
        assert row["prompt"][1]["content"] == added["statement"] + " /no_think"
    # No wrong-verdict output (b0-r07) ever gets harvested.
    assert all("wrong output" not in r["completion"][0]["content"] for r in b0_rows)

    # Non-backfill rows are explicitly flagged False, not merely absent.
    non_backfill_rows = [r for r in rows if r["provenance"]["uid"] != "b0"]
    assert len(non_backfill_rows) == HAPPY_PATH_N_EXAMPLES
    assert all(r["provenance"]["backfill_7of8"] is False for r in non_backfill_rows)

    backfill_block = manifest["backfill_7of8"]
    assert backfill_block["uids"] == ["b0"]
    assert backfill_block["per_uid_trace_counts"] == {"b0": 7}
    assert backfill_block["sources"] == [
        {
            "path": added["source_file"],
            "sha256": build_dataset.sha256_file(added["pass_at_k_path"]),
        }
    ]
    assert manifest["dataset"]["per_uid_trace_counts"]["b0"] == 7
    assert manifest["train_uids"]["count"] == len(env["train_uids"])  # includes b0
    assert manifest["reconciliation"]["per_uid"]["b0"] == {
        "trace_file": str(Path(added["source_file"]).parent / "_progress" / "rollouts.jsonl"),
        "reconciled_via": "routed",
    }


def test_backfill_uid_not_in_trace_sources_raises_backfill_mapping_error(env):
    # The split declares a backfill uid that config.BACKFILL_TRACE_SOURCES
    # does not pin -- desync, must refuse (both directions matter; this is
    # the "missing from config" direction).
    env["split_backfill_uids"] = env["split_backfill_uids"] + ["b-unpinned"]
    _write_split_file(env)

    with pytest.raises(build_dataset.BackfillMappingError, match="b-unpinned"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_backfill_trace_source_uid_not_in_split_raises_backfill_mapping_error(env):
    # config.BACKFILL_TRACE_SOURCES pins a uid the split does NOT declare
    # in train_backfill_7of8_uids -- the other direction of the same desync.
    env["backfill_trace_sources"] = dict(
        env["backfill_trace_sources"], **{"b-unclaimed": "out/somewhere/pass_at_k.jsonl"}
    )
    # Deliberately do NOT call _write_split_file -- the split must stay
    # silent about "b-unclaimed" for this to be the split-side gap.

    with pytest.raises(build_dataset.BackfillMappingError, match="b-unclaimed"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_backfill_wrong_n_correct_raises_trace_integrity_error(env):
    _add_backfill_record(env, verdicts=["correct"] * 6 + ["wrong"] * 2)

    with pytest.raises(build_dataset.TraceIntegrityError, match="n_correct"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_backfill_wrong_label_raises_trace_integrity_error(env):
    _add_backfill_record(env, label="band")

    with pytest.raises(build_dataset.TraceIntegrityError, match="label"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_backfill_source_file_missing_raises_trace_integrity_error(env):
    added = _add_backfill_record(env)
    added["pass_at_k_path"].unlink()

    with pytest.raises(build_dataset.TraceIntegrityError, match="does not exist"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_backfill_uid_not_found_in_source_file_raises_trace_integrity_error(env):
    added = _add_backfill_record(env)
    _write_jsonl(added["pass_at_k_path"], [])  # source file exists but is empty

    with pytest.raises(build_dataset.TraceIntegrityError, match="no row found"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_backfill_tally_mismatch_raises_trace_integrity_error(env):
    # The pass_at_k.jsonl row says n_correct=7, but its sibling
    # rollouts.jsonl disagrees (one "correct" flipped to "wrong") -- the
    # SAME reconciliation tally-check regular records go through, now
    # exercised via a backfill record.
    added = _add_backfill_record(env)
    rollout_rows = [
        json.loads(line) for line in added["rollout_path"].read_text(encoding="utf-8").splitlines()
    ]
    for r in rollout_rows:
        if r["rollout_uid"] == "b0-r00":
            r["verdict"] = "wrong"
    _write_jsonl(added["rollout_path"], rollout_rows)

    with pytest.raises(build_dataset.TraceIntegrityError):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_backfill_uid_also_in_corpus_raises_value_error(env):
    # Defensive guard: a uid pinned as "not in the corpus, harvest from
    # the rescore pool instead" turning up IN band_corpus.jsonl is a
    # backfill/corpus desync, not a silently-ignorable coincidence. "u0"
    # is already in train_uids.txt (base fixture) -- add_to_train_uids
    # would duplicate it there (a DIFFERENT guard); only the backfill
    # roster/pin registration is needed here.
    _add_backfill_record(env, uid="u0", add_to_train_uids=False)

    with pytest.raises(ValueError, match="backfill/corpus desync"):
        build_dataset.build(**_build_kwargs(env))
    assert not env["output_dir"].exists()


def test_backfill_uid_not_in_train_uids_is_excluded_not_harvested(env):
    # Pinned (split + config agree) but NOT part of THIS build's
    # train_uids.txt -- symmetric with a corpus-resident uid outside
    # train_uids.txt simply not being selected: no error, just excluded.
    _add_backfill_record(env, add_to_train_uids=False)

    manifest = build_dataset.build(**_build_kwargs(env))

    assert manifest["dataset"]["rows"] == HAPPY_PATH_N_EXAMPLES
    assert manifest["backfill_7of8"] == {"uids": [], "sources": [], "per_uid_trace_counts": {}}


def test_old_retired_split_path_raises_split_not_built_error(env):
    # Mirrors the 2026-07-16 retired-path test, updated to the 2026-07-26
    # retirement: evalharness/data/retired_20260726/eval_paper_split.json
    # (the frozen derived-view split, superseded by corpus_split_200_100.json)
    # must still trip assert_not_retired_path -- the canonical split path
    # itself (which contains no "retired" component) must NOT.
    retired_split = (
        env["tmp_path"] / "evalharness" / "data" / "retired_20260726" / "eval_paper_split.json"
    )
    retired_split.parent.mkdir(parents=True, exist_ok=True)
    retired_split.write_text(env["split_path"].read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(build_dataset.SplitNotBuiltError, match="(?i)retired"):
        build_dataset.build(
            **_build_kwargs(
                env,
                split_path=retired_split,
                expected_split_sha256=_split_pin(retired_split),
            )
        )
    assert not env["output_dir"].exists()


def test_leakage_guard_reads_eval_papers_from_new_split_schema(env):
    # The split file build() actually reads is the 2026-07-26
    # corpus_split_200_100.json-shaped file (eval_papers +
    # train_backfill_7of8_uids, full-sha256 pinned) -- confirm the paper-
    # level leakage guard fires off ITS eval_papers, not some vestigial
    # reading of the old eval_paper_split.json shape.
    assert env["split_path"].name == "corpus_split_200_100.json"
    data = json.loads(env["split_path"].read_text(encoding="utf-8"))
    assert set(data.keys()) == {"eval_papers", "train_backfill_7of8_uids"}

    rows = [dict(r) for r in env["corpus_rows"]]
    rows[0]["arxiv_id"] = EVAL_PAPERS[0]  # an eval paper per the NEW split's eval_papers list
    _write_jsonl(env["corpus_path"], rows)
    sha256, rows_n = _corpus_pin(env["corpus_path"])

    with pytest.raises(build_dataset.LeakageError):
        build_dataset.build(**_build_kwargs(env, expected_corpus_sha256=sha256, expected_corpus_rows=rows_n))
    assert not env["output_dir"].exists()


# --- 10. dataset v2: weight policies through the full build() ---------------------
# Pure-function policy tests live in test_weight_policy.py; these exercise
# the ORCHESTRATED path: policy applied after dedupe, manifest audit block,
# per-uid counts, disk re-verification, and determinism per policy.
# TRAIN_SPECS n_correct per uid: u0=1, u1=2, u2=1, u3=1, u4=1 (6 rows uncapped).


def test_build_cap1_one_row_per_uid_and_manifest_block(env):
    manifest = build_dataset.build(**_build_kwargs(env, weight_policy="cap1"))

    assert manifest["dataset"]["rows"] == len(env["train_uids"])  # 5: one per uid
    per_uid = manifest["dataset"]["per_uid_trace_counts"]
    assert per_uid == {"u0": 1, "u1": 1, "u2": 1, "u3": 1, "u4": 1}

    block = manifest["weight_policy"]
    assert block["policy"] == "cap1"
    assert block["label"] == "cap1"
    assert block["rows_before"] == HAPPY_PATH_N_EXAMPLES
    assert block["rows_after"] == 5
    assert block["seed"] == env["seed"]
    assert "sha256" in block["selection_rule"]  # the documented seeded rule
    assert block["rows_per_uid_before"] == {"1": 4, "2": 1}
    assert block["rows_per_uid_after"] == {"1": 5}

    rows = [
        json.loads(line)
        for line in Path(manifest["dataset"]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert all("weight" not in row for row in rows)
    # u1's kept trace follows the documented selection rule, recomputed here.
    u1_kept = next(r for r in rows if r["provenance"]["uid"] == "u1")
    expected = min(
        ("u1-r00", "u1-r01"),
        key=lambda r: hashlib.sha256(f"{env['seed']}:u1:{r}".encode()).hexdigest(),
    )
    assert u1_kept["provenance"]["rollout_uid"] == expected


def test_build_capk_manifest_label_and_counts(env):
    manifest = build_dataset.build(
        **_build_kwargs(env, weight_policy="capk", weight_policy_cap_k=2)
    )
    # k=2 covers every uid's full trace set here, so nothing is dropped --
    # the policy still runs, stamps its block, and re-verifies.
    assert manifest["dataset"]["rows"] == HAPPY_PATH_N_EXAMPLES
    block = manifest["weight_policy"]
    assert block["policy"] == "capk"
    assert block["cap_k"] == 2
    assert block["label"] == "cap2"
    assert block["rows_after"] == HAPPY_PATH_N_EXAMPLES


def test_build_inverse_writes_exact_reciprocal_weights(env):
    manifest = build_dataset.build(**_build_kwargs(env, weight_policy="inverse"))

    assert manifest["dataset"]["rows"] == HAPPY_PATH_N_EXAMPLES  # nothing dropped
    assert manifest["weight_policy"]["weighted"] is True

    rows = [
        json.loads(line)
        for line in Path(manifest["dataset"]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    n_correct = {"u0": 1, "u1": 2, "u2": 1, "u3": 1, "u4": 1}
    for row in rows:
        uid = row["provenance"]["uid"]
        assert row["weight"] == 1.0 / n_correct[uid]  # exact, JSON-round-tripped


def test_build_unknown_policy_refuses_and_writes_nothing(env):
    with pytest.raises(build_dataset.WeightPolicyError):
        build_dataset.build(**_build_kwargs(env, weight_policy="capzero"))
    assert not env["output_dir"].exists()


def test_build_determinism_per_policy(env):
    for policy in ("cap1", "inverse"):
        m1 = build_dataset.build(
            **_build_kwargs(
                env, weight_policy=policy, output_dir=env["tmp_path"] / f"b1_{policy}"
            )
        )
        m2 = build_dataset.build(
            **_build_kwargs(
                env, weight_policy=policy, output_dir=env["tmp_path"] / f"b2_{policy}"
            )
        )
        assert Path(m1["dataset"]["path"]).read_bytes() == Path(m2["dataset"]["path"]).read_bytes()


def test_build_manifest_echoes_trainer_hyperparams(env):
    # The formerly-silent four (grad-accum literal + inherited SFTConfig
    # defaults) must be visible in the dataset manifest (work order
    # 2026-07-29, "ALSO FIX").
    manifest = build_dataset.build(**_build_kwargs(env))
    echoed = manifest["trainer_hyperparams"]
    assert echoed["grad_accum_steps"] == config.GRAD_ACCUM_STEPS == 4
    assert echoed["lr_scheduler_type"] == config.LR_SCHEDULER_TYPE == "linear"
    assert echoed["warmup_ratio"] == config.WARMUP_RATIO == 0.0
    assert echoed["weight_decay"] == config.WEIGHT_DECAY == 0.0
    assert echoed["completion_only_loss"] is True
    assert manifest["sft_schema"]["format"] == "prompt_completion"
    assert manifest["sft_schema"]["version"] == 2
    assert manifest["guards"] == list(build_dataset.BUILD_GUARD_STEPS)
    assert "apply_weight_policy" in manifest["guards"]


def test_main_weight_policy_flag_overrides_config_default(env, monkeypatch, capsys):
    monkeypatch.setattr(config, "EXPECTED_CORPUS_SHA256", env["expected_corpus_sha256"])
    monkeypatch.setattr(config, "EXPECTED_CORPUS_ROWS", env["expected_corpus_rows"])
    monkeypatch.setattr(config, "EXPECTED_SPLIT_SHA256", env["expected_split_sha256"])
    monkeypatch.setattr(config, "BACKFILL_TRACE_SOURCES", env["backfill_trace_sources"])
    monkeypatch.setattr(config, "SEED", env["seed"])

    rc = build_dataset.main(
        [
            "--corpus", str(env["corpus_path"]),
            "--split", str(env["split_path"]),
            "--train-uids", str(env["train_uids_path"]),
            "--eval-set", str(env["eval_set_path"]),
            "--output-dir", str(env["output_dir"]),
            "--repo-root", str(env["repo_root"]),
            "--weight-policy", "inverse",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["weight_policy"]["policy"] == "inverse"
    assert summary["rows"] == HAPPY_PATH_N_EXAMPLES
    rows = [
        json.loads(line)
        for line in (env["output_dir"] / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row.get("weight") == 0.5 for row in rows)  # u1's two traces

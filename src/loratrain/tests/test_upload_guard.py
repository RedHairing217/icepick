"""Tests for loratrain.upload_guard (the RUNBOOK section 5 guarded uploader).

All hermetic: every config path this module touches is monkeypatched onto
tmp_path fixtures, and --execute's subprocess.run is monkeypatched to record
its argv rather than actually invoking scp. No network, no real corpus.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from loratrain import build_dataset, config, upload_guard, verify_base_identity


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _row(uid, rollout_uid, arxiv_id, verdict="correct", verbatim=True):
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prove x"},
            {"role": "assistant", "content": "proof of x"},
        ],
        "provenance": {
            "uid": uid,
            "rollout_uid": rollout_uid,
            "arxiv_id": arxiv_id,
            "source_file": "out/x/pass_at_k.jsonl",
            "verdict": verdict,
            "verbatim_output": verbatim,
            "corpus_sha256": config.EXPECTED_CORPUS_SHA256,
        },
    }


DEFAULT_ROWS = [
    _row("u1", "r1", "1000.00001"),
    _row("u2", "r2", "1000.00002"),
    _row("u3", "r3", "1000.00003"),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Build a fully valid, hermetic upload-guard environment under tmp_path.

    Individual tests then mutate one piece of it to exercise a specific
    refusal path.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset_path = data_dir / "sft_train.jsonl"
    manifest_path = data_dir / "dataset_manifest.json"
    split_path = tmp_path / "eval_paper_split.json"
    eval_set_path = tmp_path / "eval_set.jsonl"
    identity_receipt_path = data_dir / "identity_receipt.json"

    _write_jsonl(dataset_path, DEFAULT_ROWS)
    manifest_path.write_text(json.dumps({"corpus_sha256": config.EXPECTED_CORPUS_SHA256}), encoding="utf-8")

    split_path.write_text(json.dumps({"eval_papers": ["9999.00001", "9999.00002"]}), encoding="utf-8")
    split_sha16 = hashlib.sha256(split_path.read_bytes()).hexdigest()[:16]

    _write_jsonl(eval_set_path, [{"uid": "eval-u1"}, {"uid": "eval-u2"}])

    identity_receipt_path.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "SFT_DATASET_PATH", dataset_path)
    monkeypatch.setattr(config, "DATASET_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(config, "EVAL_PAPER_SPLIT_PATH", split_path)
    monkeypatch.setattr(config, "EXPECTED_SPLIT_SHA256_16", split_sha16)
    monkeypatch.setattr(config, "EVAL_SET_PATH", eval_set_path)
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "pod.example")  # non-loopback hostname
    # TRAIN_SERVER_URL is deliberately NOT patched alongside the IP: since the
    # SSH-tunnel-only decision (RUNBOOK D-R1, 2026-07-25) the URL is tunnel-
    # local -- derived from TRAIN_SERVER_PORT only -- and validate_config
    # rejects any URL carrying the pod's address.
    monkeypatch.setenv("TRAIN_SSH_PORT", "2222")

    return {
        "data_dir": data_dir,
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "split_path": split_path,
        "eval_set_path": eval_set_path,
        "identity_receipt_path": identity_receipt_path,
    }


# --- resolve_ssh_port -------------------------------------------------------


def test_resolve_ssh_port_from_env(monkeypatch):
    monkeypatch.delattr(config, "TRAIN_SERVER_SSH_PORT", raising=False)
    monkeypatch.setenv("TRAIN_SSH_PORT", "2222")
    assert upload_guard.resolve_ssh_port() == 2222


def test_resolve_ssh_port_prefers_config_over_env(monkeypatch):
    # Appendix A applied 2026-07-25: the config attribute exists and wins.
    monkeypatch.setattr(config, "TRAIN_SERVER_SSH_PORT", 40022, raising=False)
    monkeypatch.setenv("TRAIN_SSH_PORT", "2222")
    assert upload_guard.resolve_ssh_port() == 40022


def test_resolve_ssh_port_shipped_config_default(monkeypatch):
    # Config-attr precedence with no env export at all; hermetically pinned
    # (W3 provisioning sets a real per-pod value in the live config).
    monkeypatch.delenv("TRAIN_SSH_PORT", raising=False)
    monkeypatch.setattr(config, "TRAIN_SERVER_SSH_PORT", 22)
    assert upload_guard.resolve_ssh_port() == 22


def test_resolve_ssh_port_missing_refuses(monkeypatch):
    monkeypatch.delattr(config, "TRAIN_SERVER_SSH_PORT", raising=False)
    monkeypatch.delenv("TRAIN_SSH_PORT", raising=False)
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.resolve_ssh_port()


def test_resolve_ssh_port_out_of_range_refuses(monkeypatch):
    monkeypatch.delattr(config, "TRAIN_SERVER_SSH_PORT", raising=False)
    monkeypatch.setenv("TRAIN_SSH_PORT", "70000")
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.resolve_ssh_port()


# --- check_target ------------------------------------------------------------


def test_check_target_loopback_refuses(monkeypatch):
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "127.0.0.1")
    with pytest.raises(upload_guard.UploadRefused) as excinfo:
        upload_guard.check_target()
    assert "1.3" in str(excinfo.value)


def test_check_target_hostname_ok(monkeypatch):
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "pod.example")
    assert upload_guard.check_target() is None


def test_check_target_real_ip_ok(monkeypatch):
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "203.0.113.5")
    assert upload_guard.check_target() is None


# --- check_blocklist / build_scp_command --------------------------------------


def test_check_blocklist_raises_on_offender(tmp_path):
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.check_blocklist([tmp_path / "holdout_v2.jsonl"])


def test_check_blocklist_passes_clean(tmp_path):
    assert upload_guard.check_blocklist([tmp_path / "sft_train.jsonl"]) is None


def test_build_scp_command_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "pod.example")
    files = [tmp_path / "sft_train.jsonl", tmp_path / "run_config.json"]
    cmd = upload_guard.build_scp_command(files, 2222)

    assert cmd[0] == "scp"
    assert cmd[1] == "-P"
    assert cmd[2] == "2222"
    assert cmd[-1] == "root@pod.example:/workspace/run/"
    assert str(files[0]) in cmd
    assert str(files[1]) in cmd


# --- validate_dataset ----------------------------------------------------------


def test_validate_dataset_happy_path(env):
    info = upload_guard.validate_dataset()
    assert info["rows"] == 3
    assert info["sha256"] == build_dataset.sha256_file(env["dataset_path"])


def test_validate_dataset_missing_dataset_refuses(env):
    env["dataset_path"].unlink()
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.validate_dataset()


def test_validate_dataset_bad_json_line_refuses(env):
    env["dataset_path"].write_text("not json\n", encoding="utf-8")
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.validate_dataset()


def test_validate_dataset_leakage_paper_level_surfaces_as_leakage_error(env):
    rows = DEFAULT_ROWS[:2] + [_row("u4", "r4", "9999.00001")]  # arxiv id is an eval paper
    _write_jsonl(env["dataset_path"], rows)
    with pytest.raises(build_dataset.LeakageError):
        upload_guard.validate_dataset()


def test_validate_dataset_leakage_uid_level_surfaces_as_leakage_error(env):
    rows = DEFAULT_ROWS[:2] + [_row("eval-u1", "r4", "1000.00099")]  # uid is in eval_set
    _write_jsonl(env["dataset_path"], rows)
    with pytest.raises(build_dataset.LeakageError):
        upload_guard.validate_dataset()


def test_validate_dataset_missing_eval_set_refuses(env):
    env["eval_set_path"].unlink()
    with pytest.raises(upload_guard.UploadRefused, match="eval_set"):
        upload_guard.validate_dataset()


def test_validate_dataset_missing_manifest_refuses(env):
    env["manifest_path"].unlink()
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.validate_dataset()


def test_validate_dataset_wrong_corpus_sha_refuses(env):
    env["manifest_path"].write_text(json.dumps({"corpus_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.validate_dataset()


def test_validate_dataset_duplicate_uid_rollout_uid_refuses(env):
    rows = list(DEFAULT_ROWS) + [DEFAULT_ROWS[0]]
    _write_jsonl(env["dataset_path"], rows)
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.validate_dataset()


def test_validate_dataset_verdict_wrong_refuses(env):
    rows = DEFAULT_ROWS[:2] + [_row("u4", "r4", "1000.00099", verdict="wrong")]
    _write_jsonl(env["dataset_path"], rows)
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.validate_dataset()


# --- main(): happy dry-run, refusals, --execute -------------------------------


def test_main_happy_dry_run(env, capsys):
    rc = upload_guard.main([])
    captured = capsys.readouterr()

    assert rc == 0
    assert "scp" in captured.out
    assert config.TRAIN_SERVER_IP in captured.out
    assert "DRY RUN" in captured.out


def test_main_loopback_train_server_ip_refuses(env, monkeypatch, capsys):
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "127.0.0.1")
    rc = upload_guard.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "UPLOAD REFUSED" in captured.err


def test_main_missing_identity_receipt_refuses(env, capsys):
    env["identity_receipt_path"].unlink()
    rc = upload_guard.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "UPLOAD REFUSED" in captured.err


def test_main_identity_receipt_not_pass_refuses(env, capsys):
    env["identity_receipt_path"].write_text(json.dumps({"verdict": "FAIL"}), encoding="utf-8")
    rc = upload_guard.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "UPLOAD REFUSED" in captured.err


def test_main_blocklisted_dataset_name_refuses(env, monkeypatch, capsys):
    bad_path = env["data_dir"] / "band_corpus_x.jsonl"
    bad_path.write_bytes(env["dataset_path"].read_bytes())
    monkeypatch.setattr(config, "SFT_DATASET_PATH", bad_path)

    rc = upload_guard.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "UPLOAD REFUSED" in captured.err


def test_main_execute_runs_scp_after_all_checks(env, monkeypatch):
    recorded = {}

    def fake_run(cmd, check=False):
        recorded["cmd"] = cmd
        recorded["check"] = check

    monkeypatch.setattr(upload_guard.subprocess, "run", fake_run)

    rc = upload_guard.main(["--execute"])

    assert rc == 0
    assert recorded["cmd"][0] == "scp"
    assert recorded["check"] is True
    assert recorded["cmd"][-1] == f"root@{config.TRAIN_SERVER_IP}:/workspace/run/"


def test_main_execute_not_called_when_refused(env, monkeypatch):
    calls = []
    monkeypatch.setattr(upload_guard.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(config, "TRAIN_SERVER_IP", "127.0.0.1")  # refuse before scp

    rc = upload_guard.main(["--execute"])

    assert rc == 2
    assert calls == []


def test_manifest_nested_corpus_sha_accepted(monkeypatch, tmp_path):
    # W2's real manifest nests the corpus record; the guard must read it
    # (schema-drift fix 2026-07-26) while still failing on a wrong value.
    import json as _json
    m = tmp_path / "dataset_manifest.json"
    m.write_text(_json.dumps({"corpus": {"sha256": config.EXPECTED_CORPUS_SHA256}}))
    monkeypatch.setattr(config, "DATASET_MANIFEST_PATH", m)
    assert upload_guard._check_manifest_corpus_sha(m) is None  # no raise

def test_manifest_nested_corpus_sha_mismatch_refuses(monkeypatch, tmp_path):
    import json as _json
    m = tmp_path / "dataset_manifest.json"
    m.write_text(_json.dumps({"corpus": {"sha256": "0" * 64}}))
    monkeypatch.setattr(config, "DATASET_MANIFEST_PATH", m)
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard._check_manifest_corpus_sha(m)


# --- write_run_config: base-scheme provenance (T4, 2026-07-30) ---------------


def test_write_run_config_additive_only_under_default_scheme(tmp_path):
    # ADDITIVE-ONLY (T4.2): under the shipped default scheme (fp16), the
    # produced run_config.json must equal exactly the pre-T4 payload shape
    # plus base_scheme + base_source_sha256 -- nothing removed, renamed, or
    # (for this scheme) further added.
    assert config.BASE_SCHEME == config.BASE_SCHEME_FP16  # this test only proves additivity for the shipped default

    path = tmp_path / "run_config.json"
    upload_guard.write_run_config(path)
    produced = json.loads(path.read_text(encoding="utf-8"))

    pre_change_payload = {
        "seeds": list(config.SEEDS),
        "hyperparams": {
            "rank": config.LORA_RANK,
            "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT,
            "lr": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "micro_batch_size": config.MICRO_BATCH_SIZE,
            "max_seq_len": config.MAX_SEQ_LEN,
            "grad_accum_steps": config.GRAD_ACCUM_STEPS,
            "lr_scheduler_type": config.LR_SCHEDULER_TYPE,
            "warmup_ratio": config.WARMUP_RATIO,
            "weight_decay": config.WEIGHT_DECAY,
        },
        "weight_policy": config.WEIGHT_POLICY,
        "weight_policy_label": config.weight_policy_label(),
        "dataset_schema": "prompt_completion.v2",
        "completion_only_loss": True,
        "base_model": config.BASE_MODEL_HF_ID,
        "base_model_revision": verify_base_identity.FP16_REVISION,
        "adapter_format": config.ADAPTER_FORMAT,
        "llamacpp_tag": verify_base_identity.LLAMACPP_TAG,
        "serve_quant": config.SERVE_QUANT,
    }

    expected = dict(pre_change_payload)
    expected["base_scheme"] = config.BASE_SCHEME_FP16
    expected["base_source_sha256"] = verify_base_identity.FP16_REVISION

    assert produced == expected
    assert set(produced) - set(pre_change_payload) == {"base_scheme", "base_source_sha256"}


def test_write_run_config_dequant_scheme_adds_manifest_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    path = tmp_path / "run_config.json"
    receipt = {"verdict": "PASS", "scheme": "dequant_q4km", "chain": {"dequant_manifest_sha256": "ab" * 32}}

    upload_guard.write_run_config(path, receipt)
    produced = json.loads(path.read_text(encoding="utf-8"))

    assert produced["base_scheme"] == "dequant_q4km"
    assert produced["base_source_sha256"] == verify_base_identity.EXPECTED_BASE_GGUF_SHA256
    assert produced["base_manifest_sha256"] == "ab" * 32


def test_write_run_config_dequant_scheme_without_receipt_arg_refuses(tmp_path, monkeypatch):
    # Review fix #8 (fail-open null): this used to emit base_manifest_sha256:
    # null and proceed. A dequant-scheme run_config.json with no manifest-sha
    # chain link is not a shippable degraded artifact -- it must refuse.
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    path = tmp_path / "run_config.json"

    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.write_run_config(path)  # no identity_receipt passed at all
    assert not path.exists()


def test_write_run_config_dequant_scheme_with_receipt_missing_chain_refuses(tmp_path, monkeypatch):
    # Same refusal, but with an identity_receipt argument that just lacks
    # the chain (e.g. a hand-edited/corrupted receipt) rather than being
    # omitted entirely.
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    path = tmp_path / "run_config.json"
    receipt = {"verdict": "PASS", "scheme": "dequant_q4km"}  # no "chain" key at all

    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.write_run_config(path, receipt)
    assert not path.exists()


def test_write_run_config_dequant_scheme_with_empty_chain_sha_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    path = tmp_path / "run_config.json"
    receipt = {"verdict": "PASS", "scheme": "dequant_q4km", "chain": {"dequant_manifest_sha256": None}}

    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.write_run_config(path, receipt)
    assert not path.exists()


def test_write_run_config_fp16_scheme_never_adds_manifest_sha_key(tmp_path):
    path = tmp_path / "run_config.json"
    receipt = {"verdict": "PASS", "chain": {"dequant_manifest_sha256": "ab" * 32}}

    upload_guard.write_run_config(path, receipt)
    produced = json.loads(path.read_text(encoding="utf-8"))

    assert "base_manifest_sha256" not in produced  # fp16 scheme never carries this key at all


# --- check_base_scheme: upload-time chain enforcement (T4 #3) ----------------


def test_check_base_scheme_receipt_without_scheme_key_is_fp16_ok():
    assert config.BASE_SCHEME == config.BASE_SCHEME_FP16
    assert upload_guard.check_base_scheme({"verdict": "PASS"}) is None  # no raise


def test_check_base_scheme_matching_explicit_fp16_ok():
    assert upload_guard.check_base_scheme({"verdict": "PASS", "scheme": "fp16_hf_revision"}) is None


def test_check_base_scheme_mismatch_refuses():
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.check_base_scheme({"verdict": "PASS", "scheme": "dequant_q4km"})


def test_check_base_scheme_dequant_matching_ok(monkeypatch):
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    assert upload_guard.check_base_scheme({"verdict": "PASS", "scheme": "dequant_q4km"}) is None


def test_check_base_scheme_dequant_config_but_fp16_receipt_refuses(monkeypatch):
    # Receipt without "scheme" is treated as fp16 -- must still refuse when
    # config.BASE_SCHEME has been flipped to dequant.
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.check_base_scheme({"verdict": "PASS"})


def test_main_identity_receipt_scheme_mismatch_refuses(env, monkeypatch, capsys):
    env["identity_receipt_path"].write_text(
        json.dumps({"verdict": "PASS", "scheme": "dequant_q4km"}), encoding="utf-8"
    )
    rc = upload_guard.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "UPLOAD REFUSED" in captured.err


def test_main_identity_receipt_no_scheme_key_still_succeeds_under_fp16_default(env):
    # env's identity_receipt fixture is {"verdict": "PASS"} (no scheme key)
    # -- backward compat: this must NOT be refused under the shipped fp16
    # default.
    rc = upload_guard.main([])
    assert rc == 0


def test_main_dequant_scheme_receipt_without_chain_refuses(env, monkeypatch, capsys):
    # Review fix #8, exercised end to end through main(): a dequant-scheme
    # config with a PASS receipt that lacks chain.dequant_manifest_sha256
    # must refuse before ever calling write_run_config's disk write.
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    env["identity_receipt_path"].write_text(
        json.dumps({"verdict": "PASS", "scheme": "dequant_q4km"}), encoding="utf-8"
    )
    rc = upload_guard.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "UPLOAD REFUSED" in captured.err
    assert not (env["data_dir"] / "run_config.json").exists()


# --- check_source_verified: fail-closed whitelist (review fix #1, round 4) --
# Supersedes the round-3 blocklist version -- see the function's own
# docstring for the three fail-open bypass shapes the attack-replay found.

_REAL_PIN = verify_base_identity.EXPECTED_BASE_GGUF_SHA256


def test_check_source_verified_genuine_verified_receipt_proceeds():
    chain = {"source_verified": True, "gguf_sha256": _REAL_PIN, "dequant_manifest_sha256": "def" * 21}
    assert upload_guard.check_source_verified({"verdict": "PASS", "scheme": "dequant_q4km", "chain": chain}) is None


def test_check_source_verified_bypass_null_sha_refuses():
    # Bypass shape #1: source_verified True but gguf_sha256 null -- no
    # value was ever actually compared to the pin.
    chain = {"source_verified": True, "gguf_sha256": None}
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.check_source_verified({"verdict": "PASS", "scheme": "dequant_q4km", "chain": chain})


def test_check_source_verified_bypass_string_false_refuses():
    # Bypass shape #2: source_verified is the STRING "false", not the
    # boolean False -- `"false" is False` is False in Python, so the old
    # blocklist's identity check against False missed this entirely, and
    # the old code never positively required `is True` either.
    chain = {"source_verified": "false", "gguf_sha256": _REAL_PIN}
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.check_source_verified({"verdict": "PASS", "scheme": "dequant_q4km", "chain": chain})


def test_check_source_verified_bypass_wrong_hash_refuses():
    # Bypass shape #3: source_verified True, gguf_sha256 well-formed but
    # WRONG -- the old code never compared it to the pin at all.
    chain = {"source_verified": True, "gguf_sha256": "00" * 32}
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.check_source_verified({"verdict": "PASS", "scheme": "dequant_q4km", "chain": chain})


def test_check_source_verified_explicit_false_refuses():
    chain = {"source_verified": False, "gguf_sha256": None, "manifest_claimed_gguf_sha256": "abc"}
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.check_source_verified({"verdict": "PASS", "scheme": "dequant_q4km", "chain": chain})


def test_check_source_verified_chain_absent_entirely_refuses():
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.check_source_verified({"verdict": "PASS", "scheme": "dequant_q4km"})


def test_check_source_verified_legacy_shape_without_source_verified_key_refuses():
    # Documents the "no legacy carve-out" contract (round 4): a chain that
    # predates this field entirely -- e.g. only gguf_sha256, no
    # source_verified key at all, even the CORRECT pin value -- still
    # refuses. There is no shipped dequant-upload history to grandfather
    # in, so presence of the right hash alone is not enough; the explicit
    # source_verified: True marker is mandatory.
    chain = {"gguf_sha256": _REAL_PIN, "dequant_manifest_sha256": "def" * 21}
    with pytest.raises(upload_guard.UploadRefused):
        upload_guard.check_source_verified({"verdict": "PASS", "scheme": "dequant_q4km", "chain": chain})


def test_main_dequant_scheme_skip_file_sha_receipt_refuses_upload(env, monkeypatch, capsys):
    # End to end: a receipt produced by --skip-file-sha (source_verified
    # False, gguf_sha256 null) must refuse at the upload gate, per the
    # module's own "the upload gate is where the full chain must hold" rule.
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    env["identity_receipt_path"].write_text(
        json.dumps({
            "verdict": "PASS",
            "scheme": "dequant_q4km",
            "chain": {
                "source_verified": False,
                "gguf_sha256": None,
                "manifest_claimed_gguf_sha256": "a" * 64,
                "dequant_manifest_sha256": "b" * 64,
            },
        }),
        encoding="utf-8",
    )
    rc = upload_guard.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "UPLOAD REFUSED" in captured.err
    assert not (env["data_dir"] / "run_config.json").exists()


def test_main_dequant_scheme_verified_receipt_succeeds(env, monkeypatch):
    monkeypatch.setattr(config, "BASE_SCHEME", config.BASE_SCHEME_DEQUANT)
    env["identity_receipt_path"].write_text(
        json.dumps({
            "verdict": "PASS",
            "scheme": "dequant_q4km",
            "chain": {
                "source_verified": True,
                "gguf_sha256": _REAL_PIN,
                "dequant_manifest_sha256": "b" * 64,
            },
        }),
        encoding="utf-8",
    )
    rc = upload_guard.main([])
    assert rc == 0
    produced = json.loads((env["data_dir"] / "run_config.json").read_text(encoding="utf-8"))
    assert produced["base_manifest_sha256"] == "b" * 64

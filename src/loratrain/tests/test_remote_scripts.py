"""Tests for the box-side remote/ scripts: mostly syntax-only, never run.

remote/train_qwen3_lora.py and remote/run_remote_train.sh reference CUDA-box
paths and packages (transformers/peft/trl/torch) absent from this repo, so
most of this suite checks syntax validity only (ast.parse / bash -n) plus an
address scan: no IPv4 literal at all in the trainer, and in the .sh the ONE
permitted literal is the loopback bind of the status server (SSH-tunnel-only,
RUNBOOK D-R1 revised 2026-07-25) -- pod addresses stay in config.py, carried
by the operator's own scp/ssh invocations per RUNBOOK.md, never these files.
This suite also trips if the .sh's STATUS_PORT default drifts from
config.TRAIN_STATUS_BOX_PORT (the two halves of the section 6 tunnel).

One narrow exception (review fix #5, 2026-07-30, ``_load_train_module``):
a handful of pure helper functions in the trainer never touch the heavy
deps (those imports are all lazy, inside ``train()``'s body), so importing
the module itself is safe and those specific functions get real behavioral
tests against ``tmp_path`` fixture dirs instead of text scans.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

from loratrain import config

REMOTE_DIR = Path(__file__).resolve().parents[1] / "remote"
TRAIN_SCRIPT = REMOTE_DIR / "train_qwen3_lora.py"
RUN_SCRIPT = REMOTE_DIR / "run_remote_train.sh"

_IPV4_RE = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
_LOOPBACK = "127.0.0.1"


def _load_train_module():
    """Load train_qwen3_lora.py as a real module via importlib (review fix
    #5, 2026-07-30).

    This is a deliberate, narrow exception to this file's "syntax-only,
    never imported" convention (see the module docstring): the module's
    OWN docstring documents that a handful of pure helper functions
    (``_verify_base_dir_matches_scheme``, ``_fatal``, ``_append_manifest``)
    never touch torch/transformers/peft/trl (those imports are all inside
    ``train()``'s body, never at module scope), so importing the module
    itself is safe -- only calling ``train()``/``main()`` would need the
    heavy deps this repo doesn't install. The review explicitly asked for
    "tests with fixture dirs" (behavioral verification), not another text
    scan, for this gate.
    """
    spec = importlib.util.spec_from_file_location("train_qwen3_lora_under_test", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_module():
    return _load_train_module()


def test_train_script_exists():
    assert TRAIN_SCRIPT.is_file()


def test_run_script_exists():
    assert RUN_SCRIPT.is_file()


def test_train_script_parses_as_valid_python():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(TRAIN_SCRIPT))  # raises SyntaxError on failure


def test_run_remote_train_sh_passes_bash_syntax_check():
    result = subprocess.run(
        ["bash", "-n", str(RUN_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_ipv4_literal_in_train_script():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert not _IPV4_RE.search(source), "found an IPv4 literal in train_qwen3_lora.py"


def test_only_loopback_ipv4_literal_in_run_remote_train_sh():
    # The status server's loopback bind is the ONE address this file may
    # spell (it cannot drift into a pod address by definition); anything
    # else -- a pod IP, an external mapping -- still fails the suite.
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    non_loopback = [
        m.group(0) for m in _IPV4_RE.finditer(source) if m.group(0) != _LOOPBACK
    ]
    assert not non_loopback, (
        f"found non-loopback IPv4 literal(s) in run_remote_train.sh: {non_loopback}"
    )


def test_status_server_binds_loopback():
    # Load-bearing for the SSH-tunnel-only decision: the http.server line
    # must carry --bind 127.0.0.1 so the endpoint is unreachable from the
    # internet even though the port itself is content-restricted.
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    assert re.search(rf"-m http\.server .*--bind {re.escape(_LOOPBACK)}", source), (
        "run_remote_train.sh must start the status server with "
        f"--bind {_LOOPBACK} (RUNBOOK D-R1, SSH-tunnel-only)"
    )


def test_status_port_default_matches_config_box_port():
    # The .sh runs on the box without config.py, so its STATUS_PORT default
    # duplicates config.TRAIN_STATUS_BOX_PORT by necessity -- this is the
    # drift tripwire between the two halves of the section 6 tunnel.
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r'STATUS_PORT="\$\{STATUS_PORT:-(\d+)\}"', source)
    assert match, "run_remote_train.sh no longer declares a STATUS_PORT default"
    assert int(match.group(1)) == config.TRAIN_STATUS_BOX_PORT, (
        f"run_remote_train.sh STATUS_PORT default {match.group(1)} != "
        f"config.TRAIN_STATUS_BOX_PORT {config.TRAIN_STATUS_BOX_PORT}"
    )


def test_train_script_has_no_hardcoded_grad_accum_literal():
    # v2 revision (2026-07-29): gradient_accumulation_steps was a hardcoded
    # `=4` literal (the silent-hyperparameter defect flagged by both 07-28
    # reviews); it must come from the run_config hyperparams now.
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"gradient_accumulation_steps\s*=\s*\d", source), (
        "train_qwen3_lora.py hardcodes gradient_accumulation_steps again -- "
        "it must be a named hyperparameter (hyperparams['grad_accum_steps'])"
    )
    assert "grad_accum_steps" in source


def test_train_script_pins_formerly_silent_sftconfig_defaults():
    # The three inherited-SFTConfig-default knobs must be spelled explicitly
    # in the SFTConfig call, and completion-only loss must be wired for the
    # v2 prompt/completion dataset path.
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    for kwarg in ("lr_scheduler_type", "warmup_ratio", "weight_decay", "completion_only_loss"):
        assert re.search(rf"{kwarg}\s*=", source), (
            f"train_qwen3_lora.py no longer pins {kwarg} explicitly in SFTConfig"
        )


def test_train_script_v1_fallback_defaults_match_what_v1_ran():
    # A v1 run_config.json (no new keys) must reproduce v1 exactly: the
    # .get() fallbacks are pinned to the values v1 actually trained with.
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert re.search(r"""\.get\(\s*['"]grad_accum_steps['"]\s*,\s*4\s*\)""", source)
    assert re.search(r"""\.get\(\s*['"]lr_scheduler_type['"]\s*,\s*['"]linear['"]\s*\)""", source)
    assert re.search(r"""\.get\(\s*['"]warmup_ratio['"]\s*,\s*0\.0\s*\)""", source)
    assert re.search(r"""\.get\(\s*['"]weight_decay['"]\s*,\s*0\.0\s*\)""", source)


def test_train_script_echoes_base_scheme_into_manifest():
    # T4 #4: base_scheme + base_source_sha256 must ride from run_config.json
    # (upload_guard.write_run_config resolves them) through to the per-seed
    # run_manifest.json entry, so a manifest can be traced back to which
    # base the adapter was actually trained against. A run_config.json
    # predating this field falls back to the fp16 scheme label -- same
    # idiom as the four SFTConfig knobs. (Review fix #6: --smoke does NOT
    # get this fallback anymore -- see test_train_script_smoke_marks_entries
    # below -- so this test only covers the non-smoke else-branch.)
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert re.search(
        r"""run_config_data\.get\(\s*['"]base_scheme['"]\s*,\s*['"]fp16_hf_revision['"]\s*\)""", source
    ), "train_qwen3_lora.py must resolve base_scheme from run_config_data with the fp16 fallback"
    assert re.search(
        r"""run_config_data\.get\(\s*['"]base_source_sha256['"]\s*\)""", source
    ), "train_qwen3_lora.py must resolve base_source_sha256 from run_config_data"
    assert 'manifest_entry["base_scheme"] = run_config_data.get' in source
    assert 'manifest_entry["base_source_sha256"] = run_config_data.get' in source


def test_train_script_smoke_mode_never_loads_run_config():
    # --smoke mode's run_config_data must default to {} rather than reading
    # --run-config off disk (the CLI help text calls that flag "ignored in
    # --smoke mode"). Anchored to the `if args.smoke:` branch specifically
    # (test-quality fix #14) -- an unanchored grep for `run_config_data = {}`
    # anywhere in the file would pass even if that assignment moved to the
    # wrong branch or a stray duplicate appeared elsewhere.
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"if args\.smoke:\n(.*?)\n    else:\n", source, re.DOTALL)
    assert match, "train_qwen3_lora.py's train() no longer has an `if args.smoke: ... else:` branch"
    smoke_block = match.group(1)
    assert re.search(r"run_config_data\s*=\s*\{\}", smoke_block), (
        "the --smoke branch (specifically) must set run_config_data = {} "
        "so the base_scheme/base_source_sha256 echo degrades to the fp16/"
        "None defaults instead of reading a run_config.json"
    )


def test_train_script_smoke_marks_entries_instead_of_fabricating_scheme():
    # Review fix #6: a smoke run has no real run_config.json-derived
    # base_scheme -- fabricating the fp16 fallback for it poisoned
    # check_same_base_scheme's manifest scan (the RUNBOOK smoke config's
    # --out lands in the SAME out/run_manifest.json a real campaign
    # appends to). The manifest entry must be marked "smoke": True and
    # must NOT carry a fabricated base_scheme key.
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert 'manifest_entry["smoke"] = True' in source
    match = re.search(r"if args\.smoke:\n(.*?)\n    else:\n", source, re.DOTALL)
    assert match, "train_qwen3_lora.py's train() no longer has an `if args.smoke: ... else:` branch"
    smoke_block = match.group(1)
    assert "base_scheme" not in smoke_block, (
        "the --smoke branch must not set base_scheme on the manifest entry "
        "-- it has no real run_config.json-derived scheme to report"
    )


# --- Review fix #5 (2026-07-30): _verify_base_dir_matches_scheme, with real fixture dirs --


def test_verify_base_dir_matches_scheme_dequant_ok(tmp_path, train_module):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "dequant_manifest.json").write_text(
        json.dumps({"base_scheme": "dequant_q4km", "source_gguf": {"sha256": "abc123"}}), encoding="utf-8"
    )
    train_module._verify_base_dir_matches_scheme(base_dir, "dequant_q4km", "abc123")  # must not raise


def test_verify_base_dir_matches_scheme_dequant_no_source_sha_to_check_ok(tmp_path, train_module):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "dequant_manifest.json").write_text(
        json.dumps({"base_scheme": "dequant_q4km", "source_gguf": {"sha256": "abc123"}}), encoding="utf-8"
    )
    train_module._verify_base_dir_matches_scheme(base_dir, "dequant_q4km", None)  # nothing to compare -- ok


def test_verify_base_dir_matches_scheme_dequant_missing_manifest_exits(tmp_path, train_module):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    with pytest.raises(SystemExit):
        train_module._verify_base_dir_matches_scheme(base_dir, "dequant_q4km", None)


def test_verify_base_dir_matches_scheme_dequant_wrong_manifest_scheme_exits(tmp_path, train_module):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "dequant_manifest.json").write_text(json.dumps({"base_scheme": "something_else"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        train_module._verify_base_dir_matches_scheme(base_dir, "dequant_q4km", None)


def test_verify_base_dir_matches_scheme_dequant_wrong_source_sha_exits(tmp_path, train_module):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "dequant_manifest.json").write_text(
        json.dumps({"base_scheme": "dequant_q4km", "source_gguf": {"sha256": "aaa"}}), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        train_module._verify_base_dir_matches_scheme(base_dir, "dequant_q4km", "bbb")


def test_verify_base_dir_matches_scheme_fp16_without_manifest_ok(tmp_path, train_module):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    train_module._verify_base_dir_matches_scheme(base_dir, "fp16_hf_revision", None)  # must not raise


def test_verify_base_dir_matches_scheme_fp16_with_dequant_manifest_exits(tmp_path, train_module):
    # The other direction: an fp16-scheme run_config pointed at a dir that
    # actually looks like a dequant output must also refuse.
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "dequant_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_module._verify_base_dir_matches_scheme(base_dir, "fp16_hf_revision", None)


# --- Review fix #11 (2026-07-30): _append_manifest atomic publish + hard-fail --


def test_append_manifest_atomic_publish_leaves_no_tmp_file(tmp_path, train_module):
    manifest_path = tmp_path / "run_manifest.json"
    train_module._append_manifest(manifest_path, {"seed": 1})
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["seeds"] == [{"seed": 1}]
    assert not (tmp_path / "run_manifest.json.tmp").exists()


def test_append_manifest_appends_across_calls(tmp_path, train_module):
    manifest_path = tmp_path / "run_manifest.json"
    train_module._append_manifest(manifest_path, {"seed": 1})
    train_module._append_manifest(manifest_path, {"seed": 2})
    seeds = json.loads(manifest_path.read_text(encoding="utf-8"))["seeds"]
    assert [s["seed"] for s in seeds] == [1, 2]


def test_append_manifest_bad_existing_json_hard_fails_instead_of_resetting(tmp_path, train_module):
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("not json {", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_module._append_manifest(manifest_path, {"seed": 1})
    # The corrupt file must survive untouched -- no silent reset to {"seeds": []}.
    assert manifest_path.read_text(encoding="utf-8") == "not json {"


def test_append_manifest_wrong_shape_top_level_list_hard_fails(tmp_path, train_module):
    # Review fix #8, round 3: valid JSON but the wrong shape (a bare list,
    # not an object) used to die with a raw AttributeError
    # (list.setdefault doesn't exist) -- must be a clean hard-fail instead.
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(SystemExit):
        train_module._append_manifest(manifest_path, {"seed": 1})
    assert manifest_path.read_text(encoding="utf-8") == json.dumps([1, 2, 3])  # untouched


def test_append_manifest_wrong_shape_seeds_not_a_list_hard_fails(tmp_path, train_module):
    # An object whose "seeds" key isn't a list (e.g. a string) used to die
    # with AttributeError on `.append` -- must also be a clean hard-fail.
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"seeds": "oops"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        train_module._append_manifest(manifest_path, {"seed": 1})
    assert manifest_path.read_text(encoding="utf-8") == json.dumps({"seeds": "oops"})


def test_append_manifest_seeds_key_absent_from_existing_object_is_fine(tmp_path, train_module):
    # An existing object with no "seeds" key at all is a legitimate
    # starting-fresh shape, not a wrong-shape refusal.
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"note": "hand-created"}), encoding="utf-8")
    train_module._append_manifest(manifest_path, {"seed": 1})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["note"] == "hand-created"
    assert [s["seed"] for s in manifest["seeds"]] == [1]


def test_append_manifest_replaces_same_seed_entry_instead_of_duplicating(tmp_path, train_module):
    # Review fix #8, round 3: run_manifest.json is a CURRENT-STATE record
    # -- the legitimate lost-gguf retrain path (re-run the same seed after
    # its conversion step was lost to a crash) must REPLACE, not duplicate.
    manifest_path = tmp_path / "run_manifest.json"
    train_module._append_manifest(manifest_path, {"seed": 1, "train_loss_final": 0.5})
    train_module._append_manifest(manifest_path, {"seed": 1, "train_loss_final": 0.1})
    seeds = json.loads(manifest_path.read_text(encoding="utf-8"))["seeds"]
    assert len(seeds) == 1
    assert seeds[0]["train_loss_final"] == 0.1


def test_append_manifest_replace_preserves_other_seeds_and_order(tmp_path, train_module):
    manifest_path = tmp_path / "run_manifest.json"
    train_module._append_manifest(manifest_path, {"seed": 1})
    train_module._append_manifest(manifest_path, {"seed": 2})
    train_module._append_manifest(manifest_path, {"seed": 1, "retrained": True})
    seeds = json.loads(manifest_path.read_text(encoding="utf-8"))["seeds"]
    assert [s["seed"] for s in seeds] == [2, 1]
    assert seeds[1]["retrained"] is True


# --- Released defect fix (docs/SESSION_HANDOFF.md 2026-07-30, defect 3) -----


def test_run_remote_train_sh_reads_manifest_from_trainers_actual_out_dir():
    # The trainer (train_qwen3_lora.py) writes run_manifest.json beside its
    # --out adapter dir -- Path(args.out).parent / "run_manifest.json" --
    # and this script invokes --out as "$RUN_DIR/out/adapter_seed$seed", so
    # the trainer's actual write path is "$RUN_DIR/out/run_manifest.json".
    # This script used to read/write "$RUN_DIR/run_manifest.json" (missing
    # the out/ segment): status.json's completed_seeds always stayed empty
    # and the crash-resume skip check always missed, so a relaunch after a
    # mid-run crash would have retrained every seed. Both call sites must
    # point at the trainer's real path, and the buggy path must not
    # reappear.
    source = RUN_SCRIPT.read_text(encoding="utf-8")

    fixed_path = '"$RUN_DIR/out/run_manifest.json"'
    buggy_path = '"$RUN_DIR/run_manifest.json"'

    occurrences = source.count(fixed_path)
    assert occurrences >= 2, (
        "run_remote_train.sh must read/write the manifest at "
        f"{fixed_path} in both the write_status call and the skip-check "
        f"call (>=2 occurrences), found {occurrences}"
    )
    assert buggy_path not in source, (
        "run_remote_train.sh regressed to the pre-fix manifest path "
        f"{buggy_path} (missing the out/ segment) -- the trainer never "
        "writes there, so crash-resume would break again"
    )


def test_run_remote_train_sh_manifest_path_matches_out_dirname():
    # Test-quality fix #15: don't just pin the literal path string -- prove
    # the INVARIANT that makes it correct. The manifest path this script
    # reads/writes must be exactly dirname(the --out argument it passes
    # the trainer). If --out's directory is ever renamed, this forces the
    # manifest-path references to move with it instead of silently
    # drifting apart again (the original defect, restated as an invariant).
    source = RUN_SCRIPT.read_text(encoding="utf-8")

    out_match = re.search(r'--out\s+"([^"]+)"', source)
    assert out_match, "run_remote_train.sh no longer passes --out to the trainer"
    out_arg = out_match.group(1)  # e.g. "$RUN_DIR/out/adapter_seed$seed"
    expected_manifest_dir = out_arg.rsplit("/", 1)[0]  # e.g. "$RUN_DIR/out"

    manifest_refs = re.findall(r'"(\$RUN_DIR/[^"]*run_manifest\.json)"', source)
    assert manifest_refs, "no run_manifest.json path references found"
    for ref in manifest_refs:
        assert ref == f"{expected_manifest_dir}/run_manifest.json", (
            f"run_manifest.json reference {ref!r} does not match dirname(--out) "
            f"{expected_manifest_dir!r} -- the trainer writes run_manifest.json "
            "beside its --out adapter dir, so these must always match"
        )


def test_run_remote_train_sh_skip_check_requires_converted_gguf_artifact():
    # Review fix #7 (round 2), extended by round-3 fix #3: GGUF existence
    # ALONE isn't completion either -- a crash mid-convert can leave a
    # PARTIAL .gguf that still passes is_file(). The skip-check's inline
    # Python must gate on the manifest entry, the final .gguf file, AND
    # its sha line actually being recorded in artifact_shas.txt (which
    # the atomic tmp->sha->mv publish below only ever writes for a
    # complete file).
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"if python - (.*?) <<'PY'\n(.*?)\nPY", source, re.DOTALL)
    assert match, "run_remote_train.sh's skip-check `if python - ... <<'PY'` block not found"
    argv_line, skip_check_body = match.group(1), match.group(2)

    assert "adapter_seed$seed.gguf" in argv_line, (
        "the skip-check must pass the converted adapter_seed$seed.gguf path as an argv "
        "argument to the inline Python, alongside the manifest path and seed"
    )
    assert "artifact_shas.txt" in argv_line, (
        "the skip-check must also pass artifact_shas.txt as an argv argument -- gguf "
        "existence alone is not completion (review fix #3, round 3)"
    )
    assert re.search(r"gguf_path\s*=.*sys\.argv\[3\]", skip_check_body) or "gguf_path" in skip_check_body, (
        "the skip-check's inline Python must read the gguf artifact path argument"
    )
    assert re.search(
        r"not\s+manifest_path\.exists\(\)\s+or\s+not\s+gguf_path\.is_file\(\)\s+or\s+not\s+shas_path\.exists\(\)",
        skip_check_body,
    ), (
        "the skip-check must require manifest_path.exists() AND gguf_path.is_file() AND "
        "shas_path.exists() before it may treat a seed as done"
    )
    assert "str(gguf_path) in recorded_files" in skip_check_body or re.search(
        r"recorded_files", skip_check_body
    ), (
        "the skip-check must confirm the gguf's sha LINE is actually present in "
        "artifact_shas.txt, not just that the file exists (review fix #3, round 3)"
    )


def test_run_remote_train_sh_convert_publish_ordering_is_tmp_then_sha_then_mv():
    # Review fix #3, round 3: the conversion must be atomic -- convert to a
    # .gguf.tmp name, sha256 THAT, append the sha line under the final
    # name, THEN mv .tmp -> final -- so a crash mid-convert never leaves a
    # partial file at the final name for the (now-exact) skip-check above
    # to trip over. Pins the ORDER, not just the presence, of these steps.
    source = RUN_SCRIPT.read_text(encoding="utf-8")

    outfile_match = re.search(r'--outfile\s+"([^"]+\.gguf\.tmp)"', source)
    assert outfile_match, "convert_lora_to_gguf.py must be invoked with a .gguf.tmp --outfile"
    outfile_pos = outfile_match.start()

    sha_append_pos = source.find('>> "$STATUS_DIR/artifact_shas.txt"')
    assert sha_append_pos != -1, "no append to artifact_shas.txt found"

    mv_match = re.search(r'\bmv\s+"[^"]*\.gguf\.tmp"\s+"[^"]*\.gguf"', source)
    assert mv_match, "no mv from the .tmp gguf to its final name found"
    mv_pos = mv_match.start()

    assert outfile_pos < sha_append_pos < mv_pos, (
        "convert (.tmp outfile) -> sha256/record -> mv must happen in that exact order "
        f"(positions: outfile={outfile_pos}, sha_append={sha_append_pos}, mv={mv_pos})"
    )

    # The recorded sha line must name the FINAL filename, not the .tmp one
    # -- otherwise artifact_shas.txt would forever point at a name that no
    # longer exists once the mv completes.
    assert re.search(r'echo\s+"\$\w+\s+\s*\$RUN_DIR/out/adapter_seed\$seed\.gguf"', source), (
        "the recorded artifact_shas.txt line must use the FINAL (non-.tmp) gguf filename"
    )


# --- RUN_DIR trailing-slash normalization (review round 4, fix #5a) ---------


def test_run_remote_train_sh_normalizes_run_dir_trailing_slash():
    # A trailing-slash RUN_DIR (e.g. "/workspace/run/") used to produce
    # double-slash paths ("$RUN_DIR/out//...") that never matched the
    # Path-normalized skip-check comparison -- every completed seed
    # retrained on every resume. Must be stripped exactly once, right
    # after RUN_DIR's initial assignment.
    source = RUN_SCRIPT.read_text(encoding="utf-8")

    assert re.search(r'RUN_DIR="\$\{RUN_DIR%/\}"', source), (
        'run_remote_train.sh must normalize RUN_DIR with RUN_DIR="${RUN_DIR%/}" '
        "(strip a trailing slash) right after it is first set"
    )

    initial_assign_pos = source.find('RUN_DIR="${RUN_DIR:-')
    normalize_pos = source.find('RUN_DIR="${RUN_DIR%/}"')
    status_dir_pos = source.find('STATUS_DIR="${STATUS_DIR:-')
    assert initial_assign_pos != -1 and normalize_pos != -1 and status_dir_pos != -1

    assert initial_assign_pos < normalize_pos < status_dir_pos, (
        "RUN_DIR must be normalized between its initial assignment and any derived "
        "default (e.g. STATUS_DIR) that depends on it, so every downstream path is "
        "already normalized"
    )


def test_run_remote_train_sh_skip_check_normalizes_paths_for_membership_check():
    # Defense in depth alongside the RUN_DIR fix above: the skip-check's
    # membership comparison should normalize both sides so a stray "//"
    # anywhere can never desync the lookup again.
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"if python - (.*?) <<'PY'\n(.*?)\nPY", source, re.DOTALL)
    assert match
    skip_check_body = match.group(2)

    assert "normpath" in skip_check_body, (
        "the skip-check's membership comparison should normalize both sides (e.g. "
        "os.path.normpath) as defense in depth against a trailing-slash RUN_DIR"
    )
    assert re.search(r"os\.path\.normpath\(str\(gguf_path\)\)\s+in\s+recorded_files", skip_check_body), (
        "the actual membership test must compare normalized forms on both sides"
    )


# --- Stale artifact_shas.txt line removal (review round 4, fix #5b) --------


def test_run_remote_train_sh_removes_stale_sha_line_before_appending():
    # A crash between the sha-append and the mv used to leave a stale
    # (possibly conflicting) line for this seed's filename in
    # artifact_shas.txt forever -- a retry only ever appended, never
    # replaced. Must grep -v any existing line for the exact final
    # filename before appending the fresh one.
    source = RUN_SCRIPT.read_text(encoding="utf-8")

    grep_match = re.search(r'grep\s+-v\s+-F\s+"[^"]*adapter_seed\$seed\.gguf"', source)
    assert grep_match, (
        "run_remote_train.sh must strip any existing artifact_shas.txt line for this "
        "seed's final gguf filename (grep -v -F) before appending the fresh sha line"
    )

    grep_pos = grep_match.start()
    append_pos = source.find('>> "$STATUS_DIR/artifact_shas.txt"')
    assert append_pos != -1
    assert grep_pos < append_pos, (
        "the stale-line removal (grep -v) must happen BEFORE the fresh append (>>)"
    )

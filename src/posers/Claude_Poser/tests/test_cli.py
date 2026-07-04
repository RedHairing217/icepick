import csv
import json
import os
from pathlib import Path

from claude_poser.cli import main


FIXTURE = Path(__file__).parent / "fixtures" / "sample_postk.jsonl"


def test_cli_writes_json(tmp_path, capsys):
    out = tmp_path / "scores.json"
    rc = main(["score", "--input", str(FIXTURE), "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["run"]["check"] == "c01_wellposed"
    assert payload["run"]["processor_mode"] == "production"
    assert payload["run"]["input_count"] == 7
    statuses = [r["wellposed_status"] for r in payload["records"]]
    assert statuses.count("pass") >= 4
    assert "flag" in statuses


def test_cli_writes_csv(tmp_path):
    out = tmp_path / "scores.csv"
    rc = main(["score", "--input", str(FIXTURE), "--output", str(out)])
    assert rc == 0
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 7
    assert {"uid", "wellposed_score", "wellposed_status"} <= set(rows[0].keys())
    sidecar = out.with_suffix(".csv.summary.json")
    assert sidecar.exists()
    summary = json.loads(sidecar.read_text())
    assert summary["counts"]


def test_cli_self_test():
    assert main(["self-test"]) == 0


def test_cli_rejects_bad_extension(tmp_path, capsys):
    out = tmp_path / "scores.txt"
    rc = main(["score", "--input", str(FIXTURE), "--output", str(out)])
    assert rc == 2


def test_cli_env_file_loads_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    env_path = tmp_path / "key.env"
    env_path.write_text(
        "ANTHROPIC_API_KEY=sk-ant-fake-for-test\n"
        "ANTHROPIC_MODEL=claude-haiku-4-5-20251001\n",
        encoding="utf-8",
    )
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--env-file", str(env_path),
    ])
    assert rc == 0
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-fake-for-test"
    assert os.environ["ANTHROPIC_MODEL"] == "claude-haiku-4-5-20251001"
    err = capsys.readouterr().err
    assert "loaded 2 key(s)" in err


def test_cli_env_file_missing(tmp_path, capsys):
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--env-file", str(tmp_path / "nope.env"),
    ])
    assert rc == 2


def test_cli_warns_when_judge_without_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--judge",
    ])
    # Run still succeeds — graceful degradation to 'defer' — but warns loudly.
    assert rc == 0
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY is not set" in err


def test_cli_provider_openai_warns_about_openai_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--judge", "--provider", "openai",
    ])
    assert rc == 0
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY is not set" in err
    assert "anthropic" not in err.lower() or "openai" in err.lower()


def test_cli_provider_openai_in_summary(tmp_path, monkeypatch):
    """The run summary records which provider/model would be used."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--provider", "openai",
        "--judge-model", "gpt-4o-mini",
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    params = payload["run"]["parameters"]
    assert params["judge_provider"] == "openai"
    assert params["resolved_model"] == "gpt-4o-mini"
    assert params["openai_api_key_present"] is True
    assert "openai_api_key" not in params  # secret must be stripped


def test_cli_invalid_provider_rejected(tmp_path):
    out = tmp_path / "scores.json"
    try:
        main([
            "score",
            "--input", str(FIXTURE),
            "--output", str(out),
            "--provider", "bogus",
        ])
    except SystemExit as e:
        # argparse choices= raises SystemExit(2) on invalid choice
        assert e.code == 2
        return
    raise AssertionError("invalid provider must be rejected")


# -------------------------------------------------------------------------- #
# Provider-segregated key loading (--anthropic-key-file / --openai-key-file)
# -------------------------------------------------------------------------- #


def _write_key_file(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_segregation_anthropic_provider_loads_only_anthropic_file(
    tmp_path, monkeypatch, capsys
):
    """Under --provider anthropic, the OpenAI key file is not opened."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    anth = _write_key_file(tmp_path / "anthro_key.env",
                           "ANTHROPIC_API_KEY=sk-ant-fake\n")
    openai = _write_key_file(tmp_path / "openai_key.env",
                             "OPENAI_API_KEY=sk-oai-fake\n")
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--provider", "anthropic",
        "--anthropic-key-file", str(anth),
        "--openai-key-file", str(openai),
    ])
    assert rc == 0
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-fake"
    # Segregation: OpenAI key file was NOT loaded into env.
    assert "OPENAI_API_KEY" not in os.environ
    err = capsys.readouterr().err
    assert "[anthropic] loaded" in err
    assert "ignored under --provider anthropic" in err


def test_segregation_openai_provider_loads_only_openai_file(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    anth = _write_key_file(tmp_path / "anthro_key.env",
                           "ANTHROPIC_API_KEY=sk-ant-fake\n")
    openai = _write_key_file(tmp_path / "openai_key.env",
                             "OPENAI_API_KEY=sk-oai-fake\n")
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--provider", "openai",
        "--anthropic-key-file", str(anth),
        "--openai-key-file", str(openai),
    ])
    assert rc == 0
    assert os.environ.get("OPENAI_API_KEY") == "sk-oai-fake"
    assert "ANTHROPIC_API_KEY" not in os.environ
    err = capsys.readouterr().err
    assert "[openai] loaded" in err
    assert "ignored under --provider openai" in err


def test_segregation_warns_when_provider_key_file_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--provider", "anthropic",
        "--anthropic-key-file", str(tmp_path / "missing.env"),
    ])
    assert rc == 2


def test_env_file_is_repeatable(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AP_TEST_A", raising=False)
    monkeypatch.delenv("AP_TEST_B", raising=False)
    f1 = _write_key_file(tmp_path / "a.env", "AP_TEST_A=alpha\n")
    f2 = _write_key_file(tmp_path / "b.env", "AP_TEST_B=beta\n")
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--env-file", str(f1),
        "--env-file", str(f2),
    ])
    assert rc == 0
    assert os.environ.get("AP_TEST_A") == "alpha"
    assert os.environ.get("AP_TEST_B") == "beta"


def test_cli_extracted_judge_policy_default_is_always(tmp_path):
    """Regression guard: the CLI must default to the safer 'always' policy."""
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["run"]["parameters"]["extracted_judge_policy"] == "always"


def test_cli_extracted_judge_policy_can_be_overridden(tmp_path):
    out = tmp_path / "scores.json"
    rc = main([
        "score",
        "--input", str(FIXTURE),
        "--output", str(out),
        "--extracted-judge-policy", "on_scanner_hit",
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["run"]["parameters"]["extracted_judge_policy"] == "on_scanner_hit"


def test_cli_rejects_bad_extracted_judge_policy(tmp_path):
    out = tmp_path / "scores.json"
    try:
        main([
            "score",
            "--input", str(FIXTURE),
            "--output", str(out),
            "--extracted-judge-policy", "nonsense",
        ])
    except SystemExit as e:
        assert e.code == 2
        return
    raise AssertionError("invalid policy must be rejected")

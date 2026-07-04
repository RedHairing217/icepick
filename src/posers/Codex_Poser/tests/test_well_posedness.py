from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codex_poser.well_posedness.judge_providers import (
    JudgeCache,
    default_key_env_path,
    load_judge_config,
    load_key_env,
)
from codex_poser.well_posedness.contracts import PassKRecord
from codex_poser.well_posedness.scoring import score_record


def _judge_reply(determined: bool):
    def judge(_: str) -> str:
        return json.dumps(
            {
                "determined": determined,
                "insufficient_context": not determined,
                "reason": "fixture",
                "confidence": 0.91,
            }
        )

    return judge


class WellPosednessScoringTests(unittest.TestCase):
    def test_computed_record_passes_when_structurally_clean(self) -> None:
        record = PassKRecord.from_raw(
            {
                "source": "generated",
                "provenance": "computed",
                "statement": "Find the value of 2 + 2.",
                "truth": "4",
                "n_correct": 4,
                "n_wrong": 4,
            },
            rid=0,
        )

        result = score_record(record)

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.score, 1.0)

    def test_dangling_reference_flags_extracted_record(self) -> None:
        record = PassKRecord.from_raw(
            {
                "source": "realmath",
                "provenance": "extracted",
                "statement": "Use \\eqref{main} to compute the value.",
                "truth": "0",
                "n_correct": 1,
                "n_wrong": 1,
            },
            rid=0,
        )

        result = score_record(record)

        self.assertEqual(result.status, "flag")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.signals["structural"], {"reference": 1})

    def test_extracted_clean_record_defers_without_judge(self) -> None:
        record = PassKRecord.from_raw(
            {
                "source": "realmath",
                "provenance": "extracted",
                "statement": "Determine x such that x^2 = 4.",
                "truth": "2",
                "n_correct": 1,
                "n_wrong": 1,
            },
            rid=0,
        )

        result = score_record(record)

        self.assertEqual(result.status, "defer")
        self.assertEqual(result.score, 0.5)

    def test_extracted_clean_record_passes_with_judge_majority(self) -> None:
        record = PassKRecord.from_raw(
            {
                "source": "realmath",
                "provenance": "extracted",
                "statement": "Determine x such that x^2 = 4.",
                "truth": "2",
                "n_correct": 1,
                "n_wrong": 1,
            },
            rid=0,
        )

        result = score_record(record, judge=_judge_reply(determined=True), judge_samples=3, judge_uphold=2)

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.signals["judge"]["samples_parsed"], 3)

    def test_extracted_clean_record_flags_with_judge_majority(self) -> None:
        record = PassKRecord.from_raw(
            {
                "source": "realmath",
                "provenance": "extracted",
                "statement": "Determine K(R).",
                "truth": "1",
                "n_correct": 1,
                "n_wrong": 1,
            },
            rid=0,
        )

        result = score_record(record, judge=_judge_reply(determined=False), judge_samples=3, judge_uphold=2)

        self.assertEqual(result.status, "flag")
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.signals["judge"]["insufficient_context"])

    def test_uid_is_stable_across_input_order(self) -> None:
        raw = {
            "source": "generated",
            "statement": "Find the value of 2 + 2.",
            "family": "arithmetic",
        }

        first = PassKRecord.from_raw(raw, rid=0)
        second = PassKRecord.from_raw(raw, rid=99)

        self.assertEqual(first.uid, second.uid)
        self.assertNotEqual(first.rid, second.rid)


class WellPosednessCliTests(unittest.TestCase):
    def test_cli_writes_json_summary(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        fixture = repo / "tests" / "fixtures" / "pass_at_k.jsonl"
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "well_posedness.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codex_poser.well_posedness.cli",
                    "score",
                    "--mode",
                    "flow_testing",
                    "--input",
                    str(fixture),
                    "--output",
                    str(output),
                ],
                cwd=repo,
                env={"PYTHONPATH": str(repo / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["run"]["check_id"], "c01_wellposed")
            self.assertEqual(payload["counts"]["total"], 3)
            self.assertEqual(payload["counts"]["pass"], 1)
            self.assertEqual(payload["counts"]["flag"], 1)
            self.assertEqual(payload["counts"]["defer"], 1)

    def test_cli_writes_csv_rows(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        fixture = repo / "tests" / "fixtures" / "pass_at_k.jsonl"
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "well_posedness.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codex_poser.well_posedness.cli",
                    "score",
                    "--mode",
                    "flow_testing",
                    "--input",
                    str(fixture),
                    "--output",
                    str(output),
                ],
                cwd=repo,
                env={"PYTHONPATH": str(repo / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            csv_text = output.read_text(encoding="utf-8")
            self.assertIn("well_posedness_score", csv_text)
            self.assertIn("defer", csv_text)

    def test_cli_refuses_judge_in_flow_testing_mode(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        fixture = repo / "tests" / "fixtures" / "pass_at_k.jsonl"
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "well_posedness.json"
            key_env = Path(tmpdir) / "key.env"
            key_env.write_text("ANTHROPIC_API_KEY=fake\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codex_poser.well_posedness.cli",
                    "score",
                    "--mode",
                    "flow_testing",
                    "--judge",
                    "--key-env",
                    str(key_env),
                    "--input",
                    str(fixture),
                    "--output",
                    str(output),
                ],
                cwd=repo,
                env={"PYTHONPATH": str(repo / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("--judge is only allowed", completed.stderr)

    def test_cli_records_openai_provider_when_not_calling_judge(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        fixture = repo / "tests" / "fixtures" / "pass_at_k.jsonl"
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "well_posedness.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codex_poser.well_posedness.cli",
                    "score",
                    "--mode",
                    "flow_testing",
                    "--judge-provider",
                    "openai",
                    "--input",
                    str(fixture),
                    "--output",
                    str(output),
                ],
                cwd=repo,
                env={"PYTHONPATH": str(repo / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["parameters"]["judge"]["enabled"])
            self.assertIsNone(payload["parameters"]["judge"]["provider"])

    def test_cli_uses_provider_specific_default_key_env(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        fixture = repo / "tests" / "fixtures" / "pass_at_k.jsonl"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "well_posedness.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codex_poser.well_posedness.cli",
                    "score",
                    "--mode",
                    "production",
                    "--judge",
                    "--judge-provider",
                    "openai",
                    "--input",
                    str(fixture),
                    "--output",
                    str(output),
                ],
                cwd=tmp_path,
                env={"PYTHONPATH": str(repo / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("openai_key.env", completed.stderr)


class KeyEnvTests(unittest.TestCase):
    def test_load_key_env_parses_export_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key_env = Path(tmpdir) / "key.env"
            key_env.write_text(
                "\n".join(
                    [
                        "export ANTHROPIC_API_KEY='fake-secret'",
                        'ANTHROPIC_MODEL="fake-model"',
                        "OPENAI_API_KEY=fake-openai",
                        "OPENAI_MODEL=gpt-test",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            values = load_key_env(key_env)

            self.assertEqual(values["ANTHROPIC_API_KEY"], "fake-secret")
            self.assertEqual(values["ANTHROPIC_MODEL"], "fake-model")
            self.assertEqual(values["OPENAI_API_KEY"], "fake-openai")
            self.assertEqual(values["OPENAI_MODEL"], "gpt-test")

    def test_openai_config_reads_openai_key_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key_env = Path(tmpdir) / "key.env"
            key_env.write_text(
                "OPENAI_API_KEY=fake-openai\nOPENAI_MODEL=gpt-test\n",
                encoding="utf-8",
            )

            config = load_judge_config(provider="openai", key_env_path=key_env)

            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.api_key, "fake-openai")
            self.assertEqual(config.model, "gpt-test")

    def test_default_key_env_paths_are_provider_specific(self) -> None:
        self.assertEqual(default_key_env_path("anthropic"), Path("../anthro_key.env"))
        self.assertEqual(default_key_env_path("openai"), Path("../openai_key.env"))

    def test_judge_cache_separates_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "judge_cache.jsonl"
            anthropic = JudgeCache(cache_path, provider="anthropic", model="same-model")
            anthropic.put("prompt", "anthropic-reply")
            anthropic.save()

            openai = JudgeCache(cache_path, provider="openai", model="same-model")

            self.assertIsNone(openai.get("prompt"))


if __name__ == "__main__":
    unittest.main()

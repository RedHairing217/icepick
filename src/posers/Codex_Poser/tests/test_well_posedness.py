from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from codex_poser.well_posedness.judge_providers import (
    JudgeCache,
    JudgeConfig,
    OpenAIJudge,
    _openai_usage,
    default_key_env_path,
    load_judge_config,
    load_key_env,
)
from codex_poser.well_posedness.contracts import PassKRecord
from codex_poser.well_posedness.scoring import (
    _PROMPT,
    _PROMPT_V2,
    context_lint_hits,
    score_record,
    score_records,
)


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


def _capturing_judge(captured: list[str], determined: bool = True):
    """Fake judge that records every prompt it receives, for asserting on
    rubric-version prompt selection."""

    def judge(prompt: str) -> str:
        captured.append(prompt)
        return json.dumps(
            {
                "determined": determined,
                "insufficient_context": not determined,
                "reason": "fixture",
                "confidence": 0.9,
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

    def test_explicit_computed_provenance_keeps_bypass(self) -> None:
        """Explicit provenance="computed" must still take the by-construction
        bypass — the fail-closed fix must not change behavior for records
        that genuinely declare computed provenance."""
        record = PassKRecord.from_raw(
            {
                "source": "generated",
                "provenance": "computed",
                "family": "arithmetic",
                "statement": "Find the value of 2 + 2.",
                "truth": "4",
                "n_correct": 4,
                "n_wrong": 4,
            },
            rid=0,
        )

        self.assertTrue(record.is_computed)

        result = score_record(record)

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertIn("by construction", result.detail)

    def test_missing_provenance_does_not_bypass_judge(self) -> None:
        """Regression test: a record with NO provenance field (but a
        ``family`` set, e.g. a generated-family record whose provenance
        tag was dropped) must NOT be inferred as "computed" and must NOT
        take the well-posed-by-construction bypass. It should flow through
        the normal judge path like any extracted/unknown-provenance record."""
        record = PassKRecord.from_raw(
            {
                "source": "generated",
                "family": "arithmetic",
                "statement": "Determine x such that x^2 = 4.",
                "truth": "2",
                "n_correct": 1,
                "n_wrong": 1,
            },
            rid=0,
        )

        self.assertEqual(record.provenance, "unknown")
        self.assertFalse(record.is_computed)

        # Without a judge: must defer, not silently pass.
        deferred = score_record(record)
        self.assertEqual(deferred.status, "defer")
        self.assertEqual(deferred.score, 0.5)

        # With a judge: must actually be judged (not short-circuited).
        judged = score_record(
            record, judge=_judge_reply(determined=True), judge_samples=3, judge_uphold=2
        )
        self.assertEqual(judged.status, "pass")
        self.assertEqual(judged.signals["judge"]["samples_parsed"], 3)
        self.assertNotIn("by construction", judged.detail)

    def test_none_and_empty_provenance_normalise_to_unknown(self) -> None:
        """None and empty-string provenance are as unset as a missing key —
        all three must normalise the same way (never "computed")."""
        for provenance_value in (None, ""):
            with self.subTest(provenance_value=provenance_value):
                record = PassKRecord.from_raw(
                    {
                        "source": "generated",
                        "provenance": provenance_value,
                        "family": "arithmetic",
                        "statement": "Determine x such that x^2 = 4.",
                        "truth": "2",
                        "n_correct": 1,
                        "n_wrong": 1,
                    },
                    rid=0,
                )
                self.assertEqual(record.provenance, "unknown")
                self.assertFalse(record.is_computed)

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


class WellPosednessPromptPinTests(unittest.TestCase):
    def test_v1_prompt_hash_is_pinned(self) -> None:
        # _PROMPT text is a billed cache-key interface (judge cache keys off
        # the rendered prompt). Any deliberate v1 edit must update this hash
        # AND the SESSION_HANDOFF note that references it.
        digest = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()
        self.assertEqual(
            digest,
            "d60fea9dacd779e3e679ac63d86e50839ad4a0a0a4452beb3b4c9b9a9fe2b4e4",
        )


class WellPosednessRubricVersionTests(unittest.TestCase):
    def _extracted_record(self) -> PassKRecord:
        return PassKRecord.from_raw(
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

    def test_rubric_v2_selects_v2_prompt_text(self) -> None:
        captured: list[str] = []

        result = score_record(
            self._extracted_record(),
            judge=_capturing_judge(captured),
            judge_samples=1,
            judge_uphold=1,
            rubric_version="v2",
        )

        self.assertIn("Attempt the derivation", captured[0])
        self.assertNotIn("Do not solve it", captured[0])
        self.assertEqual(result.signals["judge"]["rubric_version"], "v2")

    def test_rubric_default_selects_v1_prompt_text(self) -> None:
        captured: list[str] = []

        result = score_record(
            self._extracted_record(),
            judge=_capturing_judge(captured),
            judge_samples=1,
            judge_uphold=1,
        )

        self.assertIn("Do not solve it", captured[0])
        self.assertEqual(result.signals["judge"]["rubric_version"], "v1")

    def test_score_records_threads_rubric_version_and_lint_mode(self) -> None:
        captured: list[str] = []

        scored = score_records(
            [self._extracted_record()],
            judge=_capturing_judge(captured),
            judge_samples=1,
            judge_uphold=1,
            rubric_version="v2",
            context_lint_mode="advisory",
        )

        _, result = scored[0]
        self.assertIn("Attempt the derivation", captured[0])
        self.assertEqual(result.signals["judge"]["rubric_version"], "v2")
        self.assertEqual(result.signals["context_lint"]["mode"], "advisory")


class WellPosednessContextLintTests(unittest.TestCase):
    def test_missing_context_placeholder_class(self) -> None:
        hits = context_lint_hits(
            "Solve the problem given a system with the stated assumptions.",
            "5",
        )

        self.assertIn("missing_context_placeholder", hits["classes"])
        self.assertIn("a system", hits["classes"]["missing_context_placeholder"])
        self.assertGreaterEqual(hits["hit_count"], 1)

    def test_source_local_language_class(self) -> None:
        hits = context_lint_hits(
            "Using the notation defined above, compute the integral.",
            "7",
        )

        self.assertIn("source_local_language", hits["classes"])
        self.assertIn("defined above", hits["classes"]["source_local_language"])

    def test_defines_then_asks_class(self) -> None:
        hits = context_lint_hits(
            "Let T denote the linear operator described. Determine T.",
            "anything",
        )

        self.assertEqual(hits["classes"]["defines_then_asks"], ["T"])

    def test_verbatim_formula_recall_class(self) -> None:
        hits = context_lint_hits(
            "Show that the expression simplifies to x^2 + 1 for all real x.",
            "x^2 + 1",
        )

        self.assertEqual(hits["classes"]["verbatim_formula_recall"], ["x^2 + 1"])

    def test_no_hits_returns_empty_classes_and_zero_count(self) -> None:
        hits = context_lint_hits("Find the value of 2 + 2.", "4")

        self.assertEqual(hits["classes"], {})
        self.assertEqual(hits["hit_count"], 0)


class WellPosednessContextLintAdvisoryTests(unittest.TestCase):
    def test_advisory_attaches_on_computed_pass_path(self) -> None:
        record = PassKRecord.from_raw(
            {
                "source": "generated",
                "provenance": "computed",
                "statement": "Find the value of 2 + 2, as specified.",
                "truth": "4",
                "n_correct": 4,
                "n_wrong": 4,
            },
            rid=0,
        )

        result = score_record(record, context_lint_mode="advisory")

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.signals["context_lint"]["mode"], "advisory")
        self.assertIn(
            "missing_context_placeholder", result.signals["context_lint"]["classes"]
        )

    def test_advisory_attaches_on_defer_path(self) -> None:
        record = PassKRecord.from_raw(
            {
                "source": "realmath",
                "provenance": "extracted",
                "statement": "Using the notation above, determine x such that x^2 = 4.",
                "truth": "2",
                "n_correct": 1,
                "n_wrong": 1,
            },
            rid=0,
        )

        result = score_record(record, context_lint_mode="advisory")

        self.assertEqual(result.status, "defer")
        self.assertEqual(result.signals["context_lint"]["mode"], "advisory")

    def test_advisory_attaches_on_judge_flag_path(self) -> None:
        record = PassKRecord.from_raw(
            {
                "source": "realmath",
                "provenance": "extracted",
                "statement": "Let T denote the operator. Determine T.",
                "truth": "1",
                "n_correct": 1,
                "n_wrong": 1,
            },
            rid=0,
        )

        result = score_record(
            record,
            judge=_judge_reply(determined=False),
            judge_samples=3,
            judge_uphold=2,
            context_lint_mode="advisory",
        )

        self.assertEqual(result.status, "flag")
        self.assertEqual(result.signals["context_lint"]["mode"], "advisory")
        self.assertIn("defines_then_asks", result.signals["context_lint"]["classes"])

    def test_off_mode_attaches_nothing(self) -> None:
        record = PassKRecord.from_raw(
            {
                "source": "realmath",
                "provenance": "extracted",
                "statement": "Using the notation above, determine x such that x^2 = 4.",
                "truth": "2",
                "n_correct": 1,
                "n_wrong": 1,
            },
            rid=0,
        )

        result = score_record(record)

        self.assertNotIn("context_lint", result.signals)

    def test_status_and_score_identical_between_off_and_advisory(self) -> None:
        raw = {
            "source": "realmath",
            "provenance": "extracted",
            "statement": "Using the notation above, determine x such that x^2 = 4.",
            "truth": "2",
            "n_correct": 1,
            "n_wrong": 1,
        }

        off_result = score_record(PassKRecord.from_raw(raw, rid=0))
        advisory_result = score_record(
            PassKRecord.from_raw(raw, rid=0), context_lint_mode="advisory"
        )

        self.assertEqual(off_result.status, advisory_result.status)
        self.assertEqual(off_result.score, advisory_result.score)


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

    def test_cli_defaults_rubric_version_v1_and_lint_mode_off(self) -> None:
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
            self.assertEqual(payload["parameters"]["context_lint_mode"], "off")
            self.assertEqual(payload["parameters"]["judge"]["rubric_version"], "v1")
            self.assertEqual(payload["counts"]["insufficient_context_majority"], 0)

    def test_cli_accepts_rubric_version_and_lint_mode_flags(self) -> None:
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
                    "--context-lint-mode",
                    "advisory",
                    "--judge-rubric-version",
                    "v2",
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
            self.assertEqual(payload["parameters"]["context_lint_mode"], "advisory")
            self.assertEqual(payload["parameters"]["judge"]["rubric_version"], "v2")


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

    def test_openai_config_reads_reasoning_effort_from_key_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key_env = Path(tmpdir) / "key.env"
            key_env.write_text(
                "OPENAI_API_KEY=fake-openai\nOPENAI_MODEL=gpt-5.5\n"
                "OPENAI_REASONING_EFFORT=medium\n",
                encoding="utf-8",
            )

            config = load_judge_config(provider="openai", key_env_path=key_env)

            self.assertEqual(config.model, "gpt-5.5")
            self.assertEqual(config.reasoning_effort, "medium")

    def test_openai_config_reasoning_effort_defaults_to_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key_env = Path(tmpdir) / "key.env"
            key_env.write_text("OPENAI_API_KEY=fake-openai\n", encoding="utf-8")

            scrubbed = {k: v for k, v in os.environ.items()
                        if k != "OPENAI_REASONING_EFFORT"}
            with unittest.mock.patch.dict(os.environ, scrubbed, clear=True):
                config = load_judge_config(provider="openai", key_env_path=key_env)

            self.assertEqual(config.reasoning_effort, "high")

    def test_judge_cache_separates_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "judge_cache.jsonl"
            anthropic = JudgeCache(cache_path, provider="anthropic", model="same-model")
            anthropic.put("prompt", "anthropic-reply")
            anthropic.save()

            openai = JudgeCache(cache_path, provider="openai", model="same-model")

            self.assertIsNone(openai.get("prompt"))


class OpenAIJudgePayloadTests(unittest.TestCase):
    """Wire-format assertions for the Responses-API judge, per model family."""

    def _capture(self, config: JudgeConfig) -> dict:
        captured = {}

        def fake_post_json(url, payload, headers, timeout_seconds, provider):
            captured["url"] = url
            captured["payload"] = payload
            captured["timeout_seconds"] = timeout_seconds
            return {"output_text": "{}", "usage": {}}

        with unittest.mock.patch(
            "codex_poser.well_posedness.judge_providers._post_json",
            side_effect=fake_post_json,
        ):
            OpenAIJudge(config)("prompt-x")
        return captured

    def _capture_payload(self, config: JudgeConfig) -> dict:
        return self._capture(config)["payload"]

    def test_non_reasoning_payload_unchanged(self) -> None:
        """gpt-4.1-mini keeps the exact historical payload (values and key
        order), so requests are byte-identical once serialised."""
        payload = self._capture_payload(
            JudgeConfig(provider="openai", api_key="k", model="gpt-4.1-mini")
        )
        self.assertEqual(
            payload,
            {
                "model": "gpt-4.1-mini",
                "input": "prompt-x",
                "temperature": 0.2,
                "max_output_tokens": 512,
            },
        )
        self.assertEqual(
            list(payload.keys()),
            ["model", "input", "temperature", "max_output_tokens"],
        )

    def test_reasoning_payload_surface(self) -> None:
        """gpt-5.x gets reasoning.effort and a raised token floor, and must
        not carry temperature (the API 400s on it)."""
        payload = self._capture_payload(
            JudgeConfig(provider="openai", api_key="k", model="gpt-5.5")
        )
        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["max_output_tokens"], 4000)
        self.assertNotIn("temperature", payload)

    def test_reasoning_payload_honours_explicit_larger_cap(self) -> None:
        payload = self._capture_payload(
            JudgeConfig(
                provider="openai", api_key="k", model="gpt-5.5",
                max_tokens=8000, reasoning_effort="low",
            )
        )
        self.assertEqual(payload["max_output_tokens"], 8000)
        self.assertEqual(payload["reasoning"], {"effort": "low"})

    def test_reasoning_timeout_floored_legacy_timeout_unchanged(self) -> None:
        """High-effort reasoning overruns short read timeouts; timed-out
        samples become error votes. Reasoning models get a 120s floor
        (explicit larger values win); other models keep theirs exactly."""
        reasoning = self._capture(
            JudgeConfig(provider="openai", api_key="k", model="gpt-5.5")
        )
        self.assertEqual(reasoning["timeout_seconds"], 120.0)
        generous = self._capture(
            JudgeConfig(provider="openai", api_key="k", model="gpt-5.5",
                        timeout_seconds=300.0)
        )
        self.assertEqual(generous["timeout_seconds"], 300.0)
        legacy = self._capture(
            JudgeConfig(provider="openai", api_key="k", model="gpt-4.1-mini")
        )
        self.assertEqual(legacy["timeout_seconds"], 60.0)

    def test_openai_usage_extracts_reasoning_tokens(self) -> None:
        usage = _openai_usage(
            {
                "usage": {
                    "input_tokens": 700,
                    "output_tokens": 950,
                    "output_tokens_details": {"reasoning_tokens": 890},
                }
            }
        )
        self.assertEqual(
            usage,
            {"input_tokens": 700, "output_tokens": 950, "reasoning_tokens": 890},
        )

    def test_openai_usage_no_details_for_legacy_models(self) -> None:
        usage = _openai_usage({"usage": {"input_tokens": 10, "output_tokens": 5}})
        self.assertEqual(usage, {"input_tokens": 10, "output_tokens": 5})


class WellPosednessRubricV3Tests(unittest.TestCase):
    def _capture_prompt(self, rubric_version: str) -> str:
        captured = {}

        def judge(prompt: str) -> str:
            captured["prompt"] = prompt
            return json.dumps(
                {"determined": True, "insufficient_context": False,
                 "reason": "fixture", "confidence": 0.9}
            )

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
        result = score_record(record, judge=judge, rubric_version=rubric_version)
        self.assertEqual(result.signals["judge"]["rubric_version"], rubric_version)
        return captured["prompt"]

    def test_rubric_v3_selects_v3_prompt_text(self) -> None:
        prompt = self._capture_prompt("v3")
        self.assertIn("Sketch — do not write out — the derivation", prompt)
        self.assertIn("If you cannot name the missing ingredient or defect, do not flag.", prompt)
        self.assertNotIn("Do not solve it", prompt)
        self.assertNotIn("Attempt the derivation.", prompt)

    def test_rubric_v2_unaffected_by_v3_addition(self) -> None:
        prompt = self._capture_prompt("v2")
        self.assertIn("Attempt the derivation.", prompt)
        self.assertNotIn("Sketch — do not write out", prompt)


class JudgeMaxTokensFlagTests(unittest.TestCase):
    def test_judge_max_tokens_default_and_explicit(self) -> None:
        from codex_poser.well_posedness.cli import build_parser

        parser = build_parser()
        base = ["score", "--mode", "flow_testing", "--input", "in.jsonl", "--output", "out.json"]
        self.assertEqual(parser.parse_args(base).judge_max_tokens, 512)
        self.assertEqual(
            parser.parse_args(base + ["--judge-max-tokens", "4000"]).judge_max_tokens,
            4000,
        )


if __name__ == "__main__":
    unittest.main()

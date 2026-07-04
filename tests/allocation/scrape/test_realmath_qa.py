"""LLM QA-generation extractor — the deepest in-house extractor.

No network, no API key: the QA generator and the e-print source fetcher
are both injected. Provenance stays ``extracted`` (the answer's truth is
the paper's stated result), so groundtruth keeps these records.
"""

from __future__ import annotations

import io
import socket
import sys
import tarfile
import types

import pytest

from icepick.allocation.scrape import realmath as source

_TEX = r"""
\begin{theorem}
The number of primes below ten is four.
\end{theorem}
\begin{lemma}
Every finite integral domain is a field.
\end{lemma}
"""


def _paper(arxiv_id="2604.00001"):
    return source.Paper(
        arxiv_id=arxiv_id, link=f"http://arxiv.org/abs/{arxiv_id}v1",
        title="Test Paper", abstract="An abstract.",
        primary_category="math.AP", categories=["math.AP"], published="2026-04-01T00:00:00Z",
    )


def _targz(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a qa extraction test")

    monkeypatch.setattr(socket, "socket", _blocked)


# --- classify_answer ----------------------------------------------------------


@pytest.mark.parametrize(
    "answer,tier",
    [
        ("4", "number"),
        ("-17", "number"),
        ("3.5", "number"),
        ("(1, 2)", "tuple"),
        ("x + 1", "expr"),
        ("the golden ratio", None),
        ("", None),
        (None, None),
        # Regression: sympify returns native bool/list/set/tuple for these —
        # they must classify as None, never crash on a missing .is_Symbol.
        ("True", None),
        ("False", None),
        ("[1, 2]", None),
        ("{1, 2}", None),
        ("x == y", None),
        ("not x", None),
        ("()", None),
    ],
)
def test_classify_answer(answer, tier):
    assert source.classify_answer(answer) == tier


# --- qa_extractor -------------------------------------------------------------


def _source_fetcher(paper_arxiv_id):
    return _targz({"main.tex": _TEX})


def test_qa_extractor_builds_verifiable_candidates():
    def generator(statement):
        if "primes" in statement:
            return {"question": "How many primes are below ten?", "answer": "4"}
        return {"question": "Is every finite integral domain a field?", "answer": "yes"}

    candidates = source.qa_extractor(
        _paper(), family="pde", source_fetcher=_source_fetcher, generator=generator,
    )
    # Both theorems get a QA pair, but "yes" is not a verifiable form → dropped.
    assert len(candidates) == 1
    c = candidates[0]
    assert c["statement"] == "How many primes are below ten?"
    assert c["answer"] == "4"
    assert c["tier"] == "number"
    assert c["provenance"] == "extracted"     # truth is the paper's, survives groundtruth
    assert c["truth_policy"] == "extracted"
    assert c["metadata"]["extraction"] == "llm_qa"
    assert c["metadata"]["source_statement"].startswith("The number of primes")
    assert c["family"] == "pde"


def test_qa_extractor_skips_theorems_with_no_answer():
    candidates = source.qa_extractor(
        _paper(), source_fetcher=_source_fetcher, generator=lambda s: None,
    )
    assert candidates == []


def test_qa_extractor_skips_when_generator_raises():
    def broken(statement):
        raise RuntimeError("api blew up")

    candidates = source.qa_extractor(
        _paper(), source_fetcher=_source_fetcher, generator=broken,
    )
    assert candidates == []


def test_qa_extractor_fail_opens_when_gate_flakes():
    def flaky_gate(statement):
        raise RuntimeError("temporary gate outage")

    candidates = source.qa_extractor(
        _paper(), source_fetcher=_source_fetcher, quality_gate=flaky_gate,
        generator=lambda s: {"question": "How many primes are below ten?", "answer": "4"},
    )
    assert candidates


def test_qa_extractor_surfaces_config_errors_instead_of_skipping():
    """A missing key / SDK is systemic — it must not masquerade as 0 results."""
    def misconfigured(statement):
        raise source.QAConfigError("no ANTHROPIC_API_KEY")

    with pytest.raises(source.QAConfigError):
        source.qa_extractor(_paper(), source_fetcher=_source_fetcher, generator=misconfigured)


def test_qa_extractor_survives_a_non_verifiable_boolean_answer():
    """Regression: an LLM 'True'/'False' answer must be dropped, not crash the run."""
    candidates = source.qa_extractor(
        _paper(), source_fetcher=_source_fetcher,
        generator=lambda s: {"question": "Is it true?", "answer": "True"},
    )
    assert candidates == []


def test_qa_extractor_honours_a_computed_provenance_from_the_generator():
    """A generator that computes (not extracts) must be able to say so — and the
    truth policy must follow provenance (computed → trusted), never computed +
    extracted."""
    def computing(statement):
        return {"question": "Q?", "answer": "42", "provenance": "computed"}

    candidates = source.qa_extractor(
        _paper(), source_fetcher=_source_fetcher, generator=computing,
    )
    assert candidates
    assert all(c["provenance"] == "computed" for c in candidates)
    assert all(c["truth_policy"] == "trusted" for c in candidates)


def test_qa_extractor_extracted_answer_defers_to_the_judge():
    candidates = source.qa_extractor(
        _paper(), source_fetcher=_source_fetcher,
        generator=lambda s: {"question": "How many?", "answer": "4"},
    )
    assert candidates
    assert all(c["provenance"] == "extracted" and c["truth_policy"] == "extracted" for c in candidates)


def test_extractor_for_selects_qa():
    assert source.extractor_for("qa") is source.qa_extractor


def test_scrape_with_qa_extraction_mode(monkeypatch):
    feed = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2604.00001v1</id>
    <title>Test Paper</title>
    <summary>An abstract.</summary>
    <arxiv:primary_category term="math.AP"/>
    <category term="math.AP"/>
  </entry>
</feed>"""
    empty = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    monkeypatch.setattr(source, "default_latex_source_fetcher", lambda aid, timeout=30: _targz({"main.tex": _TEX}))
    monkeypatch.setattr(source, "default_qa_quality_gate", lambda statement, **kw: True)
    monkeypatch.setattr(
        source, "default_qa_generator",
        lambda statement, **kw: {"question": "How many primes below ten?", "answer": "4"}
        if "primes" in statement else None,
    )
    result = source.scrape(
        scrape_window={"category": "math.AP", "extraction": "qa"},
        source_name="pde", target_count=10,
        fetcher=lambda q, *, start, max_results: feed if start == 0 else empty,
    )
    assert len(result.candidates) == 1
    assert result.candidates[0]["metadata"]["extraction"] == "llm_qa"
    assert result.candidates[0]["answer"] == "4"


def test_default_qa_generator_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_KEY_FILE", raising=False)
    with pytest.raises(source.QAConfigError, match="ANTHROPIC_API_KEY"):
        source.default_qa_generator("some theorem")


def test_default_qa_calls_use_prompt_cache_and_report_usage(monkeypatch):
    calls = []
    responses = [
        '{"accept": true}',
        '{"question": "Q?", "answer": "42", "is_good_theorem": true}',
    ]

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            text = responses.pop(0)
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text=text)],
                usage=types.SimpleNamespace(
                    input_tokens=100,
                    output_tokens=10,
                    cache_read_input_tokens=80,
                    cache_creation_input_tokens=20,
                ),
            )

    class _Client:
        def __init__(self, api_key):
            self.messages = _Messages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Client))
    usage_rows = []

    assert source.default_qa_quality_gate("Theorem.", usage_callback=usage_rows.append) is True
    assert source.default_qa_generator("Theorem.", usage_callback=usage_rows.append)["answer"] == "42"

    assert calls[0]["system"] == [
        {
            "type": "text",
            "text": source._QA_QUALITY_GATE_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert calls[1]["system"] == [
        {
            "type": "text",
            "text": source._QA_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert usage_rows == [
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        },
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        },
    ]

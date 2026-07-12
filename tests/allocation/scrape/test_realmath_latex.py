"""LaTeX theorem extraction — the deeper in-house extractor.

No network: the e-print source fetcher is injected with an in-memory
gzipped tar built by the tests.
"""

from __future__ import annotations

import gzip
import io
import socket
import tarfile

import pytest

from icepick.allocation.scrape import realmath as source

_TEX = r"""
\documentclass{article}
\begin{document}
\begin{theorem}\label{thm:main}
The sum of the first $n$ odd numbers is $\boxed{n^2}$.
\end{theorem}
Some prose in between that should be ignored.
\begin{lemma}
Every finite integral domain is a field.
\end{lemma}
\begin{proof}
This proof body sits in a non-theorem environment and must not be extracted.
\end{proof}
\end{document}
"""


def _paper(arxiv_id="2604.00001"):
    return source.Paper(
        arxiv_id=arxiv_id, link=f"http://arxiv.org/abs/{arxiv_id}v1",
        title="Test Paper", abstract="An abstract.",
        primary_category="math.AP", categories=["math.AP"], published="2026-04-01T00:00:00Z",
    )


def _targz(files):
    """Build an in-memory .tar.gz of {name: text}."""
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
        raise AssertionError("network access attempted in a latex extraction test")

    monkeypatch.setattr(socket, "socket", _blocked)


# --- extract_tex --------------------------------------------------------------


def test_extract_tex_reads_tex_from_a_gzipped_tar():
    data = _targz({"main.tex": _TEX, "notes.txt": "ignore me"})
    assert "\\begin{theorem}" in source.extract_tex(data)
    assert "ignore me" not in source.extract_tex(data)


def test_extract_tex_handles_a_single_gzipped_tex():
    data = gzip.compress(_TEX.encode("utf-8"))
    assert "\\begin{lemma}" in source.extract_tex(data)


def test_extract_tex_handles_bare_text():
    assert "theorem" in source.extract_tex(_TEX.encode("utf-8"))


# --- extract_theorem_candidates -----------------------------------------------


def test_extract_theorem_candidates_pulls_theorem_like_environments():
    candidates = source.extract_theorem_candidates(_TEX, _paper(), family="pde")
    assert len(candidates) == 2  # theorem + lemma; proof is excluded
    envs = {c["metadata"]["environment"] for c in candidates}
    assert envs == {"theorem", "lemma"}
    thm = next(c for c in candidates if c["metadata"]["environment"] == "theorem")
    assert thm["statement"].startswith("The sum of the first")
    assert "\\label" not in thm["statement"]  # label stripped
    assert thm["answer"] == "n^2"  # boxed answer captured
    assert thm["arxiv_id"] == "2604.00001"
    assert thm["provenance"] == "extracted"
    assert thm["family"] == "pde"


def test_extract_theorem_candidates_ignores_papers_without_theorems():
    assert source.extract_theorem_candidates("just prose, no envs", _paper()) == []


def test_extract_theorem_candidates_cleans_latex_artifacts():
    tex = (
        r"\begin{theorem}[Existence]\label{thm:x}"
        r"Let $u$ solve Eq.~\eqref{eq:main}~\cite{Foo2024}. Then $u = \boxed{0}$."
        r"\end{theorem}"
    )
    [c] = source.extract_theorem_candidates(tex, _paper())
    s = c["statement"]
    assert "\\cite" not in s and "\\eqref" not in s and "\\label" not in s
    assert not s.startswith("[")           # leading [Existence] attribution stripped
    assert s.startswith("Let $u$ solve Eq.")
    assert c["answer"] == "0"              # \boxed answer still captured
    assert c["metadata"]["has_external_refs"] is True  # referenced an external equation


def test_extract_theorem_candidates_no_ref_flag_when_self_contained():
    tex = r"\begin{lemma}Every finite integral domain is a field.\end{lemma}"
    [c] = source.extract_theorem_candidates(tex, _paper())
    assert "has_external_refs" not in c["metadata"]


def test_extract_theorem_candidates_stores_the_raw_pre_clean_body():
    """E1: the raw body survives as an audit trail, ref intact, even though
    the cleaned ``statement`` still has it stripped."""
    tex = r"\begin{theorem}Let $u$ solve Eq.~\eqref{eq:x}.\end{theorem}"
    [c] = source.extract_theorem_candidates(tex, _paper())
    assert "\\eqref{eq:x}" in c["metadata"]["source_statement_raw"]
    assert c["metadata"]["has_external_refs"] is True
    assert "\\eqref" not in c["statement"]  # the cleaned statement is unaffected


def test_extract_theorem_candidates_resolves_refs_to_labeled_content():
    """E5: a \\ref naming a labeled display environment resolves to its body."""
    tex = (
        r"\begin{equation}\label{eq:x} E=mc^2 \end{equation}"
        r"\begin{theorem}The energy satisfies \eqref{eq:x}.\end{theorem}"
    )
    [c] = source.extract_theorem_candidates(tex, _paper())
    assert c["metadata"]["resolved_refs"] == {"eq:x": "E=mc^2"}
    assert "unresolved_refs" not in c["metadata"]


def test_extract_theorem_candidates_resolves_refs_from_hypothesis_environments():
    """The label index also covers assumption/definition/hypothesis envs —
    standing hypotheses stated once and cited by many results afterward."""
    tex = (
        r"\begin{assumption}\label{ass:1} $f$ is Lipschitz. \end{assumption}"
        r"\begin{theorem}Under~\ref{ass:1}, uniqueness holds.\end{theorem}"
    )
    [c] = source.extract_theorem_candidates(tex, _paper())
    assert c["metadata"]["resolved_refs"] == {"ass:1": "$f$ is Lipschitz."}


def test_extract_theorem_candidates_unresolved_ref_to_a_missing_label():
    """A \\ref naming a label nowhere in the paper lands in unresolved_refs,
    not silently dropped."""
    tex = r"\begin{theorem}See~\eqref{eq:ghost}.\end{theorem}"
    [c] = source.extract_theorem_candidates(tex, _paper())
    assert c["metadata"]["unresolved_refs"] == ["eq:ghost"]
    assert "resolved_refs" not in c["metadata"]
    assert c["metadata"]["has_external_refs"] is True


def test_extract_theorem_candidates_partial_resolution_keeps_both_lists():
    tex = (
        r"\begin{equation}\label{eq:x} E=mc^2 \end{equation}"
        r"\begin{theorem}Combine \eqref{eq:x} and \eqref{eq:ghost}.\end{theorem}"
    )
    [c] = source.extract_theorem_candidates(tex, _paper())
    assert c["metadata"]["resolved_refs"] == {"eq:x": "E=mc^2"}
    assert c["metadata"]["unresolved_refs"] == ["eq:ghost"]


def test_extract_theorem_candidates_drops_commented_out_theorems():
    """A commented-out copy of a theorem must not leak or survive as a duplicate."""
    tex = (
        "% \\begin{lemma}\n% An old commented-out version.\n% \\end{lemma}\n"
        "\\begin{lemma}The live statement holds.\\end{lemma}"
    )
    candidates = source.extract_theorem_candidates(tex, _paper())
    assert [c["statement"] for c in candidates] == ["The live statement holds."]


def test_extract_theorem_candidates_strips_inline_comments_but_keeps_escaped_percent():
    tex = (
        "\\begin{theorem}The measure is $50\\%$ of the total.  % TODO: tighten\n"
        "It converges.\\end{theorem}"
    )
    [c] = source.extract_theorem_candidates(tex, _paper())
    assert "TODO" not in c["statement"]          # inline comment stripped
    assert "50\\%" in c["statement"]              # escaped percent preserved
    assert c["statement"].endswith("It converges.")


# --- extractor_for ------------------------------------------------------------


def test_extractor_for_selects_by_name():
    assert source.extractor_for("abstract") is source.default_extractor
    assert source.extractor_for(None) is source.default_extractor
    assert source.extractor_for("latex") is source.latex_extractor


def test_extractor_for_refuses_unknown_modes():
    with pytest.raises(ValueError, match="unknown extraction mode"):
        source.extractor_for("ocr")


# --- latex_extractor + scrape integration -------------------------------------


def test_latex_extractor_fetches_and_mines(monkeypatch):
    data = _targz({"main.tex": _TEX})
    candidates = source.latex_extractor(_paper(), family="pde", source_fetcher=lambda aid: data)
    assert [c["metadata"]["environment"] for c in candidates] == ["theorem", "lemma"]


def test_latex_extractor_skips_papers_whose_source_fails():
    def broken_fetcher(arxiv_id):
        raise OSError("network down")

    assert source.latex_extractor(_paper(), source_fetcher=broken_fetcher) == []


def test_scrape_with_latex_extraction_mode(monkeypatch):
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
    data = _targz({"main.tex": _TEX})
    monkeypatch.setattr(source, "default_latex_source_fetcher", lambda aid, timeout=30: data)

    result = source.scrape(
        scrape_window={"category": "math.AP", "extraction": "latex"},
        source_name="pde", target_count=10,
        fetcher=lambda q, *, start, max_results: feed if start == 0 else empty,
    )
    # One paper, two theorem environments → two candidates.
    assert result.papers_seen == 1
    assert len(result.candidates) == 2
    assert all(c["metadata"]["extraction"] == "latex_theorem" for c in result.candidates)

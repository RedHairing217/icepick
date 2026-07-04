"""Anthropic web_search adapter for publication-status lookup.

Per arXiv paper, makes one Anthropic call per judge sample (default 3),
each one with two tools available:

- ``web_search_20260209`` — Anthropic-hosted web search; Claude issues
  queries and reads results. No client-side execution.
- ``report_verdict`` — a custom (client-side) tool with a strict input
  schema; Claude's terminal call here is what icepick reads as the
  verdict. ``tool_choice = {"type": "any"}`` forces a tool call.

Three samples per paper; the runner upholds the majority verdict. A
``defer`` or ``error`` from any single sample is recorded verbatim in
``judge_votes`` so the runner can compute the majority honestly.

Secrets: groundtruth API access is temporarily disabled pending a cost
review — see ``_build_anthropic_client`` for the exact restore point.
The client is instantiated with a placeholder ``[API key]`` so any
accidental invocation returns 401 from Anthropic without spending
money. The ``anthropic`` SDK is imported lazily so flow_testing mode
runs without the dependency installed.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from icepick.processing.groundtruth.base import (
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_PUBLISHED,
    STATUS_UNPUBLISHED,
    GroundtruthVerdict,
)
from icepick.processing.groundtruth.config import GroundtruthConfig

_DEFAULT_SYSTEM_PROMPT = """\
You are checking whether an academic paper has been peer-reviewed and
indexed in a reputable bibliographic database. Use web_search to find
evidence, then call the report_verdict tool with your conclusion.

PASS BAR — the paper must satisfy BOTH:
  1. Peer-reviewed venue (journal, conference proceedings, refereed workshop)
  2. Indexed in a reputable database: Scopus, Web of Science, DBLP (CS),
     MathSciNet (math), PubMed (life sciences), IEEE Xplore, ACM Digital
     Library, or equivalent. arXiv itself does NOT count as an index.

DO NOT PASS:
  - Predatory journals (check Beall's List or DOAJ status if unsure)
  - Preprint-only postings (no published version traceable)
  - Workshop or symposium papers that are not refereed and not indexed
  - "In submission" or "to appear" without a confirmed venue

Return verdict_status=published only when you have concrete evidence
of BOTH peer review AND indexing. Return unpublished when you find
explicit evidence the paper is preprint-only. Return defer when the
evidence is mixed or you cannot determine the status confidently —
NEVER guess at unpublished.
"""

_REPORT_VERDICT_TOOL = {
    "name": "report_verdict",
    "description": (
        "Report the publication-status verdict for the paper after web_search "
        "has gathered evidence. Call this exactly once, as your final action."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict_status": {
                "type": "string",
                "enum": ["published", "unpublished", "defer"],
                "description": "published if peer-reviewed AND indexed; defer if uncertain",
            },
            "venue": {
                "type": "string",
                "description": "Journal or conference name if published; empty string otherwise",
            },
            "publication_year": {
                "type": "integer",
                "description": "Publication year if known; 0 if unknown",
            },
            "indexed_in": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Indices that list this paper (e.g. ['Scopus', 'DBLP'])",
            },
            "evidence_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URLs supporting the verdict (publisher page, index entry, etc.)",
            },
            "reasoning": {
                "type": "string",
                "description": "One-paragraph explanation of the verdict",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "high only if multiple independent sources confirm",
            },
        },
        "required": ["verdict_status", "reasoning", "confidence"],
    },
}


class AnthropicGroundtruthAdapter:
    """Drives the Anthropic API. One instance per groundtruth run."""

    def __init__(self, cfg: GroundtruthConfig, *, client=None):
        """``client`` injection point lets tests substitute a mock."""
        self.cfg = cfg
        self._client = client

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        self._client = _build_anthropic_client(self.cfg)
        return self._client

    def lookup_paper(
        self,
        *,
        arxiv_id: str,
        paper_title: Optional[str],
        uid_for_error_attribution: str,
    ) -> GroundtruthVerdict:
        """One paper, ``judge_samples`` independent web_search calls, majority verdict.

        Returns a single ``GroundtruthVerdict`` representing the paper.
        The runner copies this verdict (with the per-record uid/source)
        onto every record that shares this arxiv_id.
        """
        votes: list = []
        raw_payloads: list = []
        client = self._ensure_client()
        last_error: Optional[str] = None

        for sample_idx in range(self.cfg.judge_samples):
            try:
                sample = self._one_sample(
                    client=client,
                    arxiv_id=arxiv_id,
                    paper_title=paper_title,
                )
                votes.append(sample)
                raw_payloads.append(sample.get("_raw_payload", {}))
            except Exception as exc:  # noqa: BLE001 — surface any API failure as a vote
                last_error = f"{type(exc).__name__}: {exc}"
                votes.append({"verdict_status": STATUS_ERROR, "error_reason": last_error})
                raw_payloads.append({"error": last_error})

        majority_status = _majority_vote(
            votes,
            uphold=self.cfg.judge_uphold,
        )

        # Pick the most informative passing/failing sample for the merged fields.
        chosen = _choose_representative_sample(votes, majority_status)

        return GroundtruthVerdict(
            uid=uid_for_error_attribution,
            source="",
            verdict_status=majority_status,
            arxiv_id=arxiv_id,
            venue=chosen.get("venue") or None,
            publication_year=(chosen.get("publication_year") or None),
            indexed_in=chosen.get("indexed_in") or [],
            evidence_urls=chosen.get("evidence_urls") or [],
            judge_model=self.cfg.judge_model,
            judge_votes=[v.get("verdict_status", STATUS_ERROR) for v in votes],
            judge_majority=majority_status,
            reasoning=chosen.get("reasoning", ""),
            confidence=chosen.get("confidence"),
            error_reason=(last_error if majority_status == STATUS_ERROR else None),
            raw_payload={"samples": raw_payloads},
        )

    def _one_sample(
        self,
        *,
        client,
        arxiv_id: str,
        paper_title: Optional[str],
    ) -> dict:
        """One web_search-enabled call. Returns the parsed report_verdict input."""
        system = _DEFAULT_SYSTEM_PROMPT
        if self.cfg.custom_bar_instructions:
            system = system + "\n\nADDITIONAL CONSTRAINTS:\n" + self.cfg.custom_bar_instructions

        user_message = _build_user_prompt(arxiv_id=arxiv_id, paper_title=paper_title)

        start = time.monotonic()
        response = client.messages.create(
            model=self.cfg.judge_model,
            max_tokens=4096,
            system=system,
            tools=[
                {"type": "web_search_20260209", "name": "web_search"},
                _REPORT_VERDICT_TOOL,
            ],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_message}],
            timeout=self.cfg.request_timeout_s,
        )
        latency_s = time.monotonic() - start

        verdict_input = _extract_report_verdict_input(response)
        if verdict_input is None:
            # Claude exited without calling report_verdict — record as defer
            # rather than error so a single weird sample doesn't poison the
            # majority. The error_reason field carries the diagnostic.
            return {
                "verdict_status": STATUS_DEFER,
                "reasoning": "model exited without calling report_verdict",
                "confidence": "low",
                "_raw_payload": {
                    "latency_s": latency_s,
                    "stop_reason": getattr(response, "stop_reason", None),
                    "content_summary": _summarise_content(response),
                },
            }

        # NB: don't put verdict_input back into _raw_payload — verdict_input
        # already IS the report_verdict_input contents, so a self-reference
        # would create a circular JSON structure.
        verdict_input["_raw_payload"] = {
            "latency_s": latency_s,
            "stop_reason": getattr(response, "stop_reason", None),
            "usage": _safe_usage(response),
        }
        return verdict_input


def _build_user_prompt(*, arxiv_id: str, paper_title: Optional[str]) -> str:
    lines = [
        "Determine whether this arXiv paper has been peer-reviewed and indexed in a reputable bibliographic database.",
        "",
        f"arXiv ID: {arxiv_id}",
    ]
    if paper_title:
        lines.append(f"Title (extracted at scrape time, may be approximate): {paper_title}")
    lines.extend(
        [
            "",
            "Search the web for evidence. Look for:",
            "  - The paper's entry in Scopus, Web of Science, DBLP, MathSciNet, PubMed, IEEE Xplore, or ACM DL",
            "  - A DOI resolving to a publisher page (Springer, Elsevier, Wiley, ACM, IEEE, Nature, Science, etc.)",
            "  - The paper's bibliographic record on Google Scholar with a non-arxiv venue",
            "",
            "Then call report_verdict with your conclusion. Do NOT guess unpublished — use defer when uncertain.",
        ]
    )
    return "\n".join(lines)


def _extract_report_verdict_input(response) -> Optional[dict]:
    """Find the report_verdict tool_use block in the response, parse its input."""
    content = getattr(response, "content", None) or []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type != "tool_use":
            continue
        if getattr(block, "name", None) != "report_verdict":
            continue
        block_input = getattr(block, "input", None)
        if isinstance(block_input, dict):
            return dict(block_input)
        if isinstance(block_input, str):
            try:
                return json.loads(block_input)
            except json.JSONDecodeError:
                return None
    return None


def _summarise_content(response) -> list:
    """Lightweight content summary for the diagnostic raw_payload."""
    out = []
    for block in getattr(response, "content", None) or []:
        out.append(
            {
                "type": getattr(block, "type", None),
                "name": getattr(block, "name", None),
            }
        )
    return out


def _safe_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out = {}
    for attr in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = getattr(usage, attr, None)
        if value is not None:
            out[attr] = value
    return out


def _majority_vote(votes: list, *, uphold: int) -> str:
    """Uphold a status only when ``uphold`` of N samples agree; otherwise defer.

    Errors are counted as their own bucket. If errors are the strict
    majority, the verdict is ``error``; otherwise errors abstain from
    the count so two real verdicts can still uphold.
    """
    if not votes:
        return STATUS_DEFER
    counts: dict = {}
    for vote in votes:
        status = vote.get("verdict_status", STATUS_ERROR)
        counts[status] = counts.get(status, 0) + 1
    # Error-majority case: nothing salvageable.
    if counts.get(STATUS_ERROR, 0) >= uphold:
        return STATUS_ERROR
    for status in (STATUS_PUBLISHED, STATUS_UNPUBLISHED, STATUS_DEFER):
        if counts.get(status, 0) >= uphold:
            return status
    return STATUS_DEFER


def _choose_representative_sample(votes: list, majority_status: str) -> dict:
    """Pick the sample whose fields populate the merged verdict.

    Prefer a sample that matches the majority; among those, prefer the
    highest confidence. Falls back to the first non-error sample, then
    the first sample of any kind.
    """
    matching = [v for v in votes if v.get("verdict_status") == majority_status]
    if matching:
        return max(matching, key=lambda v: _confidence_rank(v.get("confidence")))
    non_error = [v for v in votes if v.get("verdict_status") != STATUS_ERROR]
    if non_error:
        return non_error[0]
    return votes[0]


def _confidence_rank(value) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(value, 0)


def _build_anthropic_client(cfg: GroundtruthConfig):
    """Load the Anthropic SDK lazily and configure it from cfg.

    KILL SWITCH: groundtruth API access is deliberately disabled pending a
    cost review. The real ``ANTHROPIC_API_KEY`` env / key-file lookup has
    been replaced with a placeholder literal so any accidental invocation
    returns 401 from Anthropic without spending money. ``cfg`` and
    ``_load_env_file`` are still available (dormant) — to re-enable,
    restore the previous body:

        if not os.environ.get("ANTHROPIC_API_KEY") and cfg.anthropic_key_file:
            _load_env_file(cfg.anthropic_key_file)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set and could not be loaded from anthropic_key_file"
            )
        return anthropic.Anthropic()
    """
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "the 'anthropic' package is required in production mode; "
            "install with `pip install icepick[judge]` or add anthropic to your env"
        ) from exc

    return anthropic.Anthropic(api_key="[API key]")


def _load_env_file(path) -> None:
    """Minimal KEY=VALUE loader. Lines starting with # are ignored."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"anthropic_key_file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Don't overwrite anything already set in the environment.
        os.environ.setdefault(key, value)

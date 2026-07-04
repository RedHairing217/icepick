"""Tests for the inverted extracted-provenance default.

Under the new default policy 'always', an extracted-provenance record with
--judge enabled must reach the judge even when the code-tier scanner
returns no hits. This is the fix for the 0/70 realmath false-pass bug.
"""

from __future__ import annotations

import pytest

from claude_poser import judge as judge_mod
from claude_poser.config import WellposedConfig
from claude_poser.schema import normalise_record
from claude_poser.wellposed import check_record


_CLEAN_STATEMENT = (
    r"Let $\mathbb{F}$ be a field with $\operatorname{char}(\mathbb{F}) = 2$ "
    r"and let $k \geq 2$. Compute $l(\mathbb{F})$."
)


def _extracted_rec(statement: str = _CLEAN_STATEMENT):
    return normalise_record(
        {"source": "realmath", "provenance": "extracted", "statement": statement},
        rid=0,
    )


def _stub_caller(replies):
    q = list(replies)
    return lambda cfg, prompt: q.pop(0)


def test_always_policy_defers_even_on_clean_scanner(monkeypatch):
    """The core fix: clean scanner + extracted + judge → still call judge."""
    cfg = WellposedConfig(
        enable_judge=True,
        judge_samples=3,
        judge_uphold=2,
        extracted_judge_policy="always",
    )
    called = {"n": 0}

    def counting_caller(cfg, prompt):
        called["n"] += 1
        return {"verdict": "flag", "insufficient_context": True, "reason": "notation"}

    monkeypatch.setattr(judge_mod, "_call_anthropic_once", counting_caller)
    result = check_record(_extracted_rec(), cfg)
    assert called["n"] == 3, "always policy must call judge 3 times"
    assert result["tier"] == "judge"
    assert result["wellposed_status"] == "insufficient_context"
    assert result["code_hits"] == []  # scanner was genuinely clean; judge caught it


def test_on_scanner_hit_policy_short_circuits_when_clean(monkeypatch):
    """Legacy behavior preserved: clean scanner + on_scanner_hit → no judge."""
    cfg = WellposedConfig(
        enable_judge=True,
        judge_samples=3,
        judge_uphold=2,
        extracted_judge_policy="on_scanner_hit",
    )
    called = {"n": 0}

    def counting_caller(cfg, prompt):
        called["n"] += 1
        return {"verdict": "flag", "insufficient_context": False, "reason": "x"}

    monkeypatch.setattr(judge_mod, "_call_anthropic_once", counting_caller)
    result = check_record(_extracted_rec(), cfg)
    assert called["n"] == 0, "on_scanner_hit must not call judge when scanner clean"
    assert result["tier"] == "code"
    assert result["wellposed_status"] == "pass"


def test_on_scanner_hit_still_defers_when_scanner_fires(monkeypatch):
    cfg = WellposedConfig(
        enable_judge=True,
        judge_samples=3,
        judge_uphold=2,
        extracted_judge_policy="on_scanner_hit",
    )
    rec = _extracted_rec("Using Theorem 3.2 from the previous section, deduce A.")
    called = {"n": 0}

    def counting_caller(cfg, prompt):
        called["n"] += 1
        return {"verdict": "flag", "insufficient_context": False, "reason": "ref"}

    monkeypatch.setattr(judge_mod, "_call_anthropic_once", counting_caller)
    result = check_record(rec, cfg)
    assert called["n"] == 3
    assert result["tier"] == "judge"
    assert result["code_hits"]  # scanner hits recorded as evidence


def test_scanner_hits_ride_along_as_evidence_under_always(monkeypatch):
    """When scanner does fire under 'always', hits are still recorded."""
    cfg = WellposedConfig(
        enable_judge=True,
        judge_samples=3,
        judge_uphold=2,
        extracted_judge_policy="always",
    )
    rec = _extracted_rec("Using Theorem 3.2 from the previous section, deduce A.")
    monkeypatch.setattr(
        judge_mod, "_call_anthropic_once",
        lambda cfg, prompt: {"verdict": "flag", "insufficient_context": True, "reason": "yes"},
    )
    result = check_record(rec, cfg)
    assert result["tier"] == "judge"
    assert result["code_hits"], "scanner hits must be preserved as evidence"


def test_judge_disabled_falls_back_to_code_tier():
    """Judge off + extracted + clean scanner → code-tier pass (unchanged)."""
    cfg = WellposedConfig(enable_judge=False)
    result = check_record(_extracted_rec(), cfg)
    assert result["tier"] == "code"
    assert result["wellposed_status"] == "pass"


def test_judge_disabled_flags_on_scanner_hit():
    cfg = WellposedConfig(enable_judge=False)
    rec = _extracted_rec("See \\ref{thm:main} for details.")
    result = check_record(rec, cfg)
    assert result["tier"] == "code"
    assert result["wellposed_status"] == "flag"


def test_config_rejects_unknown_policy():
    cfg = WellposedConfig(extracted_judge_policy="bogus")
    with pytest.raises(ValueError, match="extracted_judge_policy"):
        cfg.validate()


def test_config_defaults_to_always():
    """Regression guard: the default must be 'always' (correctness > cost)."""
    assert WellposedConfig().extracted_judge_policy == "always"


def test_trusted_provenance_still_short_circuits_regardless_of_policy(monkeypatch):
    """Provenance trust is a stronger signal than any policy setting."""
    cfg = WellposedConfig(
        enable_judge=True,
        extracted_judge_policy="always",
    )
    rec = normalise_record({
        "source": "calc",
        "provenance": "computed",
        "statement": "See Theorem 3.2 above (dangling text ignored for computed).",
    }, rid=0)
    called = {"n": 0}
    monkeypatch.setattr(judge_mod, "_call_anthropic_once",
                        lambda *a, **kw: called.__setitem__("n", called["n"] + 1) or {})
    result = check_record(rec, cfg)
    assert called["n"] == 0
    assert result["tier"] == "code"
    assert result["wellposed_status"] == "pass"

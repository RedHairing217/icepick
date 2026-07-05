from claude_poser.config import WellposedConfig
from claude_poser.judge import judge_wellposed
from claude_poser.judge_cache import JudgeCache
from claude_poser.schema import normalise_record
from claude_poser.wellposed import check_record


def _stub(replies):
    """Return a caller that pops one reply per call."""
    queue = list(replies)

    def caller(cfg, prompt):
        return queue.pop(0)

    return caller


def test_majority_pass():
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2)
    caller = _stub([
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
        {"verdict": "flag", "insufficient_context": False, "reason": "??"},
    ])
    out = judge_wellposed("statement", cfg, caller=caller)
    assert out.majority_verdict == "pass"
    assert out.wellposed_votes == 2


def test_majority_flag():
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2)
    caller = _stub([
        {"verdict": "flag", "insufficient_context": False, "reason": "missing thm"},
        {"verdict": "flag", "insufficient_context": False, "reason": "missing thm"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
    ])
    out = judge_wellposed("s", cfg, caller=caller)
    assert out.majority_verdict == "flag"


def test_insufficient_context_majority():
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2)
    caller = _stub([
        {"verdict": "flag", "insufficient_context": True, "reason": "missing eq"},
        {"verdict": "flag", "insufficient_context": True, "reason": "missing eq"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
    ])
    out = judge_wellposed("s", cfg, caller=caller)
    assert out.majority_verdict == "insufficient_context"


def test_errors_dominate_defers():
    # judge_error_retries=0 isolates the corroboration rule from the retry
    # loop (which would otherwise consume extra stub replies).
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2, judge_error_retries=0)
    caller = _stub([
        {"verdict": "error", "insufficient_context": False, "reason": "boom"},
        {"verdict": "error", "insufficient_context": False, "reason": "boom"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
    ])
    out = judge_wellposed("s", cfg, caller=caller)
    assert out.majority_verdict == "defer"


def test_cache_short_circuits(tmp_path):
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2)
    cache = JudgeCache(tmp_path / "cache.jsonl")
    # First run: populate cache
    real_replies = [
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
    ]
    out1 = judge_wellposed("statement A", cfg, cache=cache, caller=_stub(real_replies))
    assert out1.majority_verdict == "pass"
    # Second run with an empty caller — should be served entirely from cache.
    out2 = judge_wellposed("statement A", cfg, cache=cache, caller=_stub([]))
    assert out2.majority_verdict == "pass"


def test_check_record_uses_judge(monkeypatch):
    cfg = WellposedConfig(enable_judge=True, judge_samples=3, judge_uphold=2)
    rec = {
        "rid": 0, "uid": "u3", "source": "rm", "provenance": "extracted",
        "truth_policy": None,
        "statement": "Using Theorem 3.2 from the previous section, deduce A.",
    }
    # Monkeypatch the real API caller used inside judge.judge_wellposed.
    from claude_poser import judge
    queue = [
        {"verdict": "flag", "insufficient_context": True, "reason": "missing thm"},
        {"verdict": "flag", "insufficient_context": True, "reason": "missing thm"},
        {"verdict": "pass", "insufficient_context": False, "reason": "?"},
    ]
    monkeypatch.setattr(judge, "_call_anthropic_once", lambda cfg, prompt: queue.pop(0))
    result = check_record(rec, cfg)
    assert result["tier"] == "judge"
    assert result["wellposed_status"] == "insufficient_context"
    assert result["wellposed_score"] == 0.0


# --------------------------------------------------------------------------- #
# 2026-07-04 stage-3 kill analysis fixes: parse repair, error retries,
# defer_reason, derived_answer.
# --------------------------------------------------------------------------- #


def test_parse_reply_repairs_latex_escapes():
    """Raw TeX macros inside JSON strings must not become error votes."""
    from claude_poser.judge import _parse_reply

    raw = ('{"verdict": "flag", "insufficient_context": true, '
           '"reason": "the operator \\mathcal{L}_\\alpha is undefined", '
           '"derived_answer": null}')
    reply = _parse_reply(raw)
    assert reply["verdict"] == "flag"
    assert reply["insufficient_context"] is True
    assert "\\mathcal{L}_\\alpha" in reply["reason"]


def test_parse_reply_valid_json_untouched():
    from claude_poser.judge import _parse_reply

    raw = ('{"verdict": "pass", "insufficient_context": false, '
           '"reason": "self-contained", "derived_answer": "s = 1 + \\u03b3"}')
    reply = _parse_reply(raw)
    assert reply["verdict"] == "pass"
    assert reply["derived_answer"] == "s = 1 + γ"


def test_error_sample_retried_before_vote_stands():
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2, judge_error_retries=1)
    # Sample 0 errors once then succeeds on retry; samples 1-2 clean.
    caller = _stub([
        {"verdict": "error", "insufficient_context": False, "reason": "bad JSON: boom"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok (retry)"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
    ])
    out = judge_wellposed("s", cfg, caller=caller)
    assert out.majority_verdict == "pass"
    assert out.error_votes == 0
    assert out.defer_reason is None


def test_persistent_errors_defer_with_judge_errors_reason():
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2, judge_error_retries=0)
    caller = _stub([
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
        {"verdict": "error", "insufficient_context": False, "reason": "boom"},
        {"verdict": "error", "insufficient_context": False, "reason": "boom"},
    ])
    out = judge_wellposed("s", cfg, caller=caller)
    assert out.majority_verdict == "defer"
    assert out.defer_reason == "judge_errors"
    assert out.to_dict()["defer_reason"] == "judge_errors"


def test_split_defer_reason():
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2, judge_error_retries=0)
    caller = _stub([
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
        {"verdict": "flag", "insufficient_context": False, "reason": "nope"},
        {"verdict": "error", "insufficient_context": False, "reason": "boom"},
    ])
    out = judge_wellposed("s", cfg, caller=caller)
    assert out.majority_verdict == "defer"
    assert out.defer_reason == "split"


def test_derived_answer_rides_on_samples():
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2)
    caller = _stub([
        {"verdict": "pass", "insufficient_context": False, "reason": "ok",
         "derived_answer": "1"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok",
         "derived_answer": "1"},
        {"verdict": "flag", "insufficient_context": False, "reason": "??"},
    ])
    out = judge_wellposed("s", cfg, caller=caller)
    dumped = out.to_dict()
    assert dumped["samples"][0]["derived_answer"] == "1"
    assert dumped["samples"][2]["derived_answer"] is None


def test_answer_mismatch_flags_for_review(monkeypatch):
    """Judge passes but derives a different answer than the stored one →
    status unchanged (pass), record flagged answer_mismatch for review."""
    cfg = WellposedConfig(enable_judge=True, judge_samples=3, judge_uphold=2)
    rec = normalise_record({
        "source": "rm", "provenance": "extracted",
        "statement": "What is the exponent s in the product estimate?",
        "answer": "$s = \\frac{n}{p} - 2$",
    }, rid=0)
    from claude_poser import judge
    queue = [
        {"verdict": "pass", "insufficient_context": False, "reason": "standard",
         "derived_answer": "s = n/p - 1"},
        {"verdict": "pass", "insufficient_context": False, "reason": "standard",
         "derived_answer": "n/p - 1"},
        {"verdict": "pass", "insufficient_context": False, "reason": "standard"},
    ]
    monkeypatch.setattr(judge, "_call_anthropic_once", lambda cfg, prompt: queue.pop(0))
    result = check_record(rec, cfg)
    assert result["wellposed_status"] == "pass"
    assert result["judge"]["answer_consistency"] == "mismatch"
    assert "answer_mismatch" in result["review_flags"]


def test_answer_match_no_flag(monkeypatch):
    cfg = WellposedConfig(enable_judge=True, judge_samples=3, judge_uphold=2)
    rec = normalise_record({
        "source": "rm", "provenance": "extracted",
        "statement": "What is the exponent s in the product estimate?",
        "answer": "$s = \\frac{n}{p} - 2$",
    }, rid=0)
    from claude_poser import judge
    queue = [
        {"verdict": "pass", "insufficient_context": False, "reason": "standard",
         "derived_answer": "s = n/p - 2"},
        {"verdict": "pass", "insufficient_context": False, "reason": "standard"},
        {"verdict": "pass", "insufficient_context": False, "reason": "standard"},
    ]
    monkeypatch.setattr(judge, "_call_anthropic_once", lambda cfg, prompt: queue.pop(0))
    result = check_record(rec, cfg)
    assert result["judge"]["answer_consistency"] == "match"
    assert result["review_flags"] == []


def test_no_derived_answers_is_unknown_not_flagged(monkeypatch):
    cfg = WellposedConfig(enable_judge=True, judge_samples=3, judge_uphold=2)
    rec = normalise_record({
        "source": "rm", "provenance": "extracted",
        "statement": "What is the exponent s?", "answer": "$s = 2$",
    }, rid=0)
    from claude_poser import judge
    queue = [
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
    ]
    monkeypatch.setattr(judge, "_call_anthropic_once", lambda cfg, prompt: queue.pop(0))
    result = check_record(rec, cfg)
    assert result["judge"]["answer_consistency"] == "unknown"
    assert result["review_flags"] == []


def test_degenerate_record_flagged_not_killed(monkeypatch):
    """Self-answering record: judge passes it (trivially answerable), but the
    degeneracy scan routes it to review."""
    cfg = WellposedConfig(enable_judge=True, judge_samples=3, judge_uphold=2)
    rec = normalise_record({
        "source": "rm", "provenance": "extracted",
        "statement": ("Let $T$ be a matrix whose eigenvalues are "
                      "$(3\\pm\\sqrt{5})/2$. What are these eigenvalues?"),
        "answer": "$\\dfrac{3\\pm\\sqrt{5}}{2}$",
    }, rid=0)
    from claude_poser import judge
    queue = [
        {"verdict": "pass", "insufficient_context": False, "reason": "stated",
         "derived_answer": "(3\\pm\\sqrt{5})/2"},
    ] * 3
    monkeypatch.setattr(judge, "_call_anthropic_once", lambda cfg, prompt: queue.pop(0))
    result = check_record(rec, cfg)
    assert result["wellposed_status"] == "pass"
    assert "degenerate_candidate" in result["review_flags"]
    assert result["degeneracy_hits"]

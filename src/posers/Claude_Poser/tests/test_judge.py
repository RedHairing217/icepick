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
    cfg = WellposedConfig(judge_samples=3, judge_uphold=2)
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

"""Cache key must include provider AND model so replies don't cross-contaminate."""

from claude_poser.judge_cache import JudgeCache, _key


def test_cache_key_changes_with_provider():
    a = _key("anthropic", "model-x", "prompt", 0)
    b = _key("openai", "model-x", "prompt", 0)
    assert a != b


def test_cache_key_changes_with_model():
    a = _key("anthropic", "claude-haiku-4-5-20251001", "prompt", 0)
    b = _key("anthropic", "claude-opus-4-7", "prompt", 0)
    assert a != b


def test_cache_key_changes_with_sample_id():
    a = _key("anthropic", "m", "prompt", 0)
    b = _key("anthropic", "m", "prompt", 1)
    assert a != b


def test_cache_get_put_isolated_by_provider(tmp_path):
    cache = JudgeCache(tmp_path / "cache.jsonl")
    reply_a = {"verdict": "pass", "insufficient_context": False, "reason": "anth"}
    cache.put("anthropic", "m", "prompt", 0, reply_a)

    assert cache.get("anthropic", "m", "prompt", 0)["reply"] == reply_a
    assert cache.get("openai", "m", "prompt", 0) is None
    assert cache.get("anthropic", "different-model", "prompt", 0) is None


def test_cache_persists_provider_field(tmp_path):
    path = tmp_path / "cache.jsonl"
    JudgeCache(path).put(
        "openai", "gpt-4o-mini", "prompt", 0,
        {"verdict": "pass", "insufficient_context": False, "reason": "ok"},
    )
    reloaded = JudgeCache(path)
    entry = reloaded.get("openai", "gpt-4o-mini", "prompt", 0)
    assert entry["provider"] == "openai"
    assert entry["model"] == "gpt-4o-mini"

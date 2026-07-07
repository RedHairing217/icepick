# src/posers — poser-local agent brief

Applies to everything under `src/posers/` (Claude_Poser + Codex_Poser — the
wellposed judge fleet). The root [AGENTS.md](../../AGENTS.md) binds in full;
read it first. This file adds only poser-local rules.

## Judge caches are keyed on prompt text — edits re-bill

- claude-poser: cache key = `sha256(provider, model, prompt, sample_id)`
  (`Claude_Poser/src/claude_poser/judge_cache.py`).
- codex-poser: cache key = `sha256(identity + prompt)`
  (`Codex_Poser/src/codex_poser/well_posedness/judge_providers.py`).

Any change to judge prompt text — system prompt, rubric wording, even
whitespace — rolls every cached verdict, and the next production run re-bills
every judge call from zero. Prompt text is a billed interface: change it
deliberately, record the change in `docs/SESSION_HANDOFF.md`, and budget for
a cold cache. Don't "tidy" prompt strings in passing.

## Judge model selection — env files, never cascade flags

The billed model per provider comes from the key files' `ANTHROPIC_MODEL` /
`OPENAI_MODEL` lines (`anthro_key.env` / `openai_key.env`). Do NOT steer a
model with the cascade's `--*-judge-model` flags: they are per-BUILD across
providers, so setting one to fix a single combo poisons cross-provider combos
like `claude:openai`. (The posers' standalone `--judge-model` flags are fine
when running a poser CLI directly.)

## Other poser-local rules

- **Key segregation**: each poser refuses to read the other provider's key
  file by design. Never weaken that, and never load keys except via the
  documented flags / path-proxy env vars.
- **Prompt caching is a measured NO here (2026-07-05)**: the largest billed
  judge request is 912 tokens — under every provider's caching floor. Don't
  add `cache_control` plumbing chasing savings; the re-open triggers are
  listed in the root brief.
- **Local Qwen as judge**: pointing the OpenAI-compatible backend at LM
  Studio (`qwen/qwen3-8b`) shares the machine-wide Qwen slot — the max-ONE-
  concurrent-call rule (root brief, invariant 9) applies.
- **Subject-side Qwen sampling (pass@k) is not the posers' business** — its
  byte-identical wire params are invariant 2 in the root brief; posers never
  touch subject sampling.
- **Tests**: both poser suites are covered by the root three-suite pytest
  command; each poser also runs standalone with `pytest` from its own
  directory.

# Icepick — Claude Code entry point

@AGENTS.md

**The canonical working brief for this repo is [AGENTS.md](AGENTS.md)** — the
`@AGENTS.md` line above imports it into context. If the import did not inline
(older CLI, or you are reading this file some other way), stop and read
`AGENTS.md` in full before doing anything else: every safety invariant,
budget gate, hold, and test baseline lives there and binds this session.

**Edit AGENTS.md, not this file.** The brief has exactly one source of truth;
this wrapper holds only Claude-specific notes.

## Claude-only addendum

- **claude-api skill**: consult it before editing code that touches the
  Anthropic API or before making pricing/caching/model-id claims — floors and
  ids change, and this repo's cost findings depend on them.
- **Auto-memory**: this project has `icepick-*` memory entries. Recalled
  memories can be stale — verify them against disk and git before acting,
  and update them at session end when state changed.
- **Session end**: update `docs/SESSION_HANDOFF.md` (the shared cross-session
  ledger — Codex agents and humans write to the same file).

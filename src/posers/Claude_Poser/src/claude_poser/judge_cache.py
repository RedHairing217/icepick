"""On-disk JSONL cache for judge calls.

Cache key includes provider, model, prompt, and sample identity. Changing
any one of those invalidates the entry — we never reuse replies across
providers or models, or collapse samples that were meant to be independent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional


def _key(provider: str, model: str, prompt: str, sample_id: int) -> str:
    h = hashlib.sha256()
    for part in (provider, model, prompt, str(sample_id)):
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class JudgeCache:
    def __init__(self, path: Optional[str | Path]):
        self.path = Path(path) if path else None
        self._mem: dict[str, dict] = {}
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    k = entry.get("key")
                    if k:
                        self._mem[k] = entry

    def get(self, provider: str, model: str, prompt: str, sample_id: int) -> Optional[dict]:
        return self._mem.get(_key(provider, model, prompt, sample_id))

    def put(self, provider: str, model: str, prompt: str, sample_id: int, reply: dict) -> None:
        k = _key(provider, model, prompt, sample_id)
        entry = {
            "key": k,
            "provider": provider,
            "model": model,
            "sample_id": sample_id,
            "reply": reply,
        }
        self._mem[k] = entry
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

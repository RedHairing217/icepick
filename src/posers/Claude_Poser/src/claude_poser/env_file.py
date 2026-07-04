"""Minimal .env loader.

Used to point the CLI at an env file kept outside the repo (e.g.
~/Desktop/helloworld/key.env) so secrets never enter version control.

Rules:
- Lines that are blank or start with '#' are ignored.
- 'export KEY=VALUE' and 'KEY=VALUE' are both accepted.
- Surrounding single or double quotes on the value are stripped.
- Existing environment variables are NOT overridden — explicit shell env wins.
- Malformed lines are skipped and reported in the returned `skipped` list.

The loader intentionally does not interpolate ${VAR}, source other files,
or execute shell — it is a value-pair reader, not a shell.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LINE_RE = re.compile(r"^\s*(?:export\s+)?([^=\s]+)\s*=\s*(.*?)\s*$")


@dataclass
class LoadResult:
    path: Path
    loaded: dict[str, str] = field(default_factory=dict)   # actually set in env
    already_set: list[str] = field(default_factory=list)   # present, left alone
    skipped: list[tuple[int, str]] = field(default_factory=list)  # (line_no, why)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path: str | Path, *, override: bool = False) -> LoadResult:
    """Parse `path` and apply KEY=VALUE pairs to os.environ.

    Args:
        path: file to read. Must exist; FileNotFoundError otherwise.
        override: if True, set values even when a key is already in os.environ.
                  Default False — shell env wins, so CI / explicit exports
                  cannot be silently shadowed by a stale .env file.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"env file not found: {p}")

    result = LoadResult(path=p)
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.rstrip("\r\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = _LINE_RE.match(line)
            if not m:
                result.skipped.append((line_no, "malformed line"))
                continue
            key, value = m.group(1), m.group(2)
            if not _KEY_RE.match(key):
                result.skipped.append((line_no, f"invalid key {key!r}"))
                continue
            value = _strip_quotes(value)
            if key in os.environ and not override:
                result.already_set.append(key)
                continue
            os.environ[key] = value
            result.loaded[key] = value
    return result

"""JSONL ingest for post-pass@k records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import normalise_record


def load_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_normalised(paths: Iterable[str | Path]) -> list[dict]:
    out: list[dict] = []
    rid = 0
    for p in paths:
        for row in load_jsonl(p):
            out.append(normalise_record(row, rid))
            rid += 1
    return out

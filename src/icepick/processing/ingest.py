"""Load JSONL inputs into a stream of ``ProblemRecord``.

Each input is ``(path, source_name)`` so a single run can mix sources
without losing per-source provenance. Records are yielded lazily so the
gate can stream large corpora without holding them all in memory.

Allocation-side adapters (manual mount, CSV/TSV column maps, external
JSONL validation) hand off to this loader by writing canonical JSONL into
``out/intake/.../handoff/`` and then invoking ``load_inputs`` on those
files. This keeps processing's input path uniform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from icepick.contracts.records import ProblemRecord
from icepick.processing import schema


def load_inputs(
    inputs: Iterable[tuple],
    *,
    provenance_overrides: Optional[dict] = None,
    truth_policy_overrides: Optional[dict] = None,
    column_maps: Optional[dict] = None,
) -> Iterator[ProblemRecord]:
    """Yield normalised records across one or more (path, source) inputs.

    ``provenance_overrides`` / ``truth_policy_overrides`` are keyed by
    source name and let allocation declare a batch's policy at ingest
    time. ``column_maps`` is also keyed by source name and lets CSV-ish
    drops rename columns without forking the normaliser.

    ``rid`` increments across the full run for human reference and
    continuity with the experiment log convention. ``uid`` remains stable
    per record regardless of load order.
    """
    rid = 0
    for path, source in inputs:
        prov = (provenance_overrides or {}).get(source)
        policy = (truth_policy_overrides or {}).get(source)
        cmap = (column_maps or {}).get(source)
        for raw in _iter_jsonl(path):
            yield schema.from_raw(
                raw,
                source=source,
                rid=rid,
                provenance_override=prov,
                truth_policy_override=policy,
                column_map=cmap,
            )
            rid += 1


def _iter_jsonl(path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"input not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON ({exc.msg})"
                ) from exc

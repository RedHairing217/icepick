"""Poser contracts — protocol, dataclasses, canonical enum, uid helpers.

Every adapter implements the same three-method shape:

    plan(records, cfg, work_dir) -> PoserRequest
    run(request)                  -> PoserRunResult
    normalise(raw_path, input_uids) -> list[PoserVerdict]

The runner composes them. Subprocess details, env vars, calibration
flags — all of that lives inside the adapters. The runner only knows
about the contract surface here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol


# --- canonical verdict enum ------------------------------------------------

STATUS_WELL_POSED = "well_posed"
STATUS_ILL_POSED = "ill_posed"
STATUS_DEFER = "defer"
STATUS_ERROR = "error"

CANONICAL_STATUSES = (
    STATUS_WELL_POSED,
    STATUS_ILL_POSED,
    STATUS_DEFER,
    STATUS_ERROR,
)

# Scores are recorded for downstream analysis. The gate keys off
# verdict_status, not verdict_score.
STATUS_SCORE = {
    STATUS_WELL_POSED: 1.0,
    STATUS_DEFER: 0.5,
    STATUS_ILL_POSED: 0.0,
    STATUS_ERROR: 0.0,
}


# --- dataclasses -----------------------------------------------------------


@dataclass
class PoserVerdict:
    """One poser's canonical verdict on one record.

    ``raw_payload`` preserves the poser-specific verdict verbatim so the
    adapter renaming is non-destructive. ``verdict_detail`` may include
    ``original_status`` (the pre-canonical token) and ``error_reason``
    when ``verdict_status == 'error'``.
    """

    uid: str
    source: str
    verdict_status: str
    verdict_score: float
    poser_name: str
    poser_model: str
    verdict_detail: dict = field(default_factory=dict)
    verdict_signals: dict = field(default_factory=dict)
    raw_payload: dict = field(default_factory=dict)

    def to_jsonl_row(self) -> dict:
        return {
            "uid": self.uid,
            "source": self.source,
            "verdict_status": self.verdict_status,
            "verdict_score": self.verdict_score,
            "poser_name": self.poser_name,
            "poser_model": self.poser_model,
            "verdict_detail": self.verdict_detail,
            "verdict_signals": self.verdict_signals,
            "raw_payload": self.raw_payload,
        }


@dataclass
class PoserRequest:
    """Subprocess invocation plan."""

    argv: list
    env: dict
    input_path: Path
    output_path: Path
    cache_path: Optional[Path] = None
    poser_name: str = ""


@dataclass
class PoserRunResult:
    """Subprocess outcome."""

    exit_code: int
    stdout: str
    stderr: str
    output_path: Path
    wall_clock_seconds: float


# --- adapter protocol ------------------------------------------------------


class PoserAdapter(Protocol):
    """Common adapter surface. ``name`` is ``'claude'`` or ``'codex'``."""

    name: str

    def plan(
        self, records: list, cfg, work_dir: Path
    ) -> PoserRequest: ...

    def run(self, request: PoserRequest) -> PoserRunResult: ...

    def normalise(
        self, raw_output_path: Path, input_uids: list
    ) -> list: ...


# --- uid helpers -----------------------------------------------------------


def compute_uid(source: str, statement: str) -> str:
    """Stable 32-hex uid. Matches Anthro_Poser's convention so it round-trips.

    SHA256 truncated to 32 hex chars (128 bits). Collision probability is
    negligible at any realistic corpus size. Using a separator byte
    (0x1F, ASCII unit separator) prevents source/statement ambiguity for
    pathological inputs.
    """
    digest = hashlib.sha256(
        f"{source}\x1f{statement}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def inject_uid(records: Iterable[dict]) -> list:
    """Return a new list with ``uid`` ensured on every record.

    Defends against the two posers using different default hash
    functions. icepick computes uid up front and trusts it to round-trip
    through whichever poser is invoked. Records that already carry a
    uid are left alone.
    """
    out = []
    for raw in records:
        record = dict(raw)
        if not record.get("uid"):
            source = record.get("source", "")
            statement = record.get("statement") or record.get("question") or record.get("prompt") or ""
            record["uid"] = compute_uid(source, statement)
        out.append(record)
    return out

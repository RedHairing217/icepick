"""Record normalisation for post-pass@k inputs.

Only fields needed by the c01 check are required. The rest pass through.
"""

from __future__ import annotations

import hashlib
from typing import Any


PROVENANCE_VALUES = ("computed", "extracted", "manual", "external", "unknown")


def compute_uid(source: str, statement: str) -> str:
    """Stable content uid so records join across runs and input order changes."""
    h = hashlib.sha256()
    h.update((source or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update((statement or "").encode("utf-8"))
    return h.hexdigest()[:32]


def normalise_record(row: dict, rid: int) -> dict:
    """Coerce an input row into the minimum shape the c01 check needs.

    Unknown provenance is treated conservatively as 'extracted' downstream
    (judge defers rather than trusting silently); we preserve the raw value
    in the record so the run summary can report on it.
    """
    source = str(row.get("source") or row.get("dataset") or "unknown")
    statement = str(row.get("statement") or row.get("problem") or row.get("question") or "")
    provenance = str(row.get("provenance") or "unknown").lower()
    if provenance not in PROVENANCE_VALUES:
        provenance = "unknown"

    truth_policy = str(row.get("truth_policy") or "").lower() or None

    uid = row.get("uid") or compute_uid(source, statement)

    return {
        "rid": rid,
        "uid": uid,
        "source": source,
        "statement": statement,
        "provenance": provenance,
        "truth_policy": truth_policy,
        # Stored ground-truth answer, when the input carries one. Feeds the
        # degeneracy scan (answer-in-statement) and the judge answer-
        # consistency audit; both are review signals, not gates.
        "answer": row.get("answer"),
        "family": row.get("family"),
        "tier": row.get("tier"),
        "pass_at_k": row.get("pass_at_k"),
        "n_correct": row.get("n_correct"),
        "n_wrong": row.get("n_wrong"),
        "n_degenerate": row.get("n_degenerate"),
        "raw": row,
    }


def is_self_contained_provenance(record: dict) -> bool:
    """c01 trusts computed-provenance records as self-contained.

    Manually mounted records with truth_policy='trusted' are treated the same.
    Everything else (extracted, manual+unknown, external, unknown) runs the
    dangling-reference scan and may defer to the judge.
    """
    if record["provenance"] == "computed":
        return True
    if record["provenance"] == "manual" and record.get("truth_policy") == "trusted":
        return True
    return False

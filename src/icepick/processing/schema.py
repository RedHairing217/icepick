"""Normalisation from raw rows onto ``ProblemRecord``.

Every input shape — generated families, RealMath-style extracted records,
external JSONL drops, manually mounted batches — funnels through
``ProblemRecord.from_raw``. Treat family and source as data, not code: a
new source should be onboarded by declaring its provenance and a column
map, never by adding a new branch here.

The stable ``uid`` is a content hash of source and statement. It does not
depend on load order and does not depend on which files are present in a
given run. Verdicts and buckets join back to records across runs by uid.

Label derivation: trust an explicit stored ``band`` / ``misdirection`` /
``solved`` / ``collapse`` first (the RealMath harvest writes its own
labels). Otherwise derive from pass rate against the band. The confident-
wrong attractor only sub-sorts the below-band region; it never overrides
a band or solved verdict.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from icepick.contracts.records import (
    BAND_HI,
    BAND_LO,
    PROVENANCE_COMPUTED,
    PROVENANCE_EXTRACTED,
    PROVENANCE_VALUES,
    TRUTH_POLICY_EXTRACTED,
    TRUTH_POLICY_TRUSTED,
    TRUTH_POLICY_UNKNOWN,
    TRUTH_POLICY_VALUES,
    ProblemRecord,
)


def from_raw(
    raw: dict,
    source: str,
    rid: int,
    *,
    provenance_override: Optional[str] = None,
    truth_policy_override: Optional[str] = None,
    column_map: Optional[dict] = None,
) -> ProblemRecord:
    """Normalise one raw row onto ``ProblemRecord``.

    ``provenance_override`` and ``truth_policy_override`` are how
    allocation declares a mounted or external batch's policy at ingest
    time. ``column_map`` lets CSV/TSV inputs rename their columns onto the
    canonical field names without forking this function.
    """
    row = _apply_column_map(raw, column_map)

    statement = _first(row, ("statement", "question", "prompt", "problem"), "")
    family = row.get("family") or None
    tier = row.get("tier") or None
    params = row.get("params") if row.get("params") is not None else row.get("detail")
    if family and params is None:
        params = None  # generated rows may legitimately have no detail

    provenance = _resolve_provenance(row, family, provenance_override)
    truth_policy = _resolve_truth_policy(row, provenance, truth_policy_override)

    truth_strings = _collect_truth_strings(row)
    answer_value = row.get("answer_value") if "answer_value" in row else row.get("answer")

    n_correct = _coerce_int(row.get("n_correct", row.get("correct")))
    n_wrong = _coerce_int(
        row.get("n_wrong", row.get("wrong", row.get("wrong_complete")))
    )
    n_degenerate = _coerce_int(row.get("n_degenerate", row.get("degenerate")))

    pass_at_k = _coerce_float(row.get("pass_at_k"))
    modal_wrong = _first(row, ("modal_wrong", "top_wrong_value"), None)
    top_wrong_share = _coerce_float(row.get("top_wrong_share")) or 0.0

    label = _normalise_label(row, pass_at_k, top_wrong_share)

    return ProblemRecord(
        rid=rid,
        uid=content_id(source, statement),
        source=source,
        provenance=provenance,
        statement=statement,
        truth_strings=truth_strings,
        answer_value=answer_value,
        tier=tier,
        family=family,
        params=params,
        truth_policy=truth_policy,
        label=label,
        pass_at_k=pass_at_k,
        n_correct=n_correct,
        n_wrong=n_wrong,
        n_degenerate=n_degenerate,
        modal_wrong=str(modal_wrong) if modal_wrong not in (None, "") else None,
        top_wrong_share=top_wrong_share,
        raw=raw,
    )


def content_id(source: str, statement: str) -> str:
    """Stable 12-hex content id. Independent of load order and file set."""
    digest = hashlib.sha1(f"{source}\x00{statement}".encode("utf-8")).hexdigest()
    return digest[:12]


def _apply_column_map(raw: dict, column_map: Optional[dict]) -> dict:
    """Project source columns onto canonical names without mutating ``raw``."""
    if not column_map:
        return raw
    projected = dict(raw)
    for canonical, source_key in column_map.items():
        if source_key in raw and canonical not in projected:
            projected[canonical] = raw[source_key]
    return projected


def _resolve_provenance(
    row: dict, family: Optional[str], override: Optional[str]
) -> str:
    if override:
        if override not in PROVENANCE_VALUES:
            raise ValueError(
                f"unknown provenance {override!r}; allowed: {PROVENANCE_VALUES}"
            )
        return override
    stored = row.get("provenance")
    if stored in PROVENANCE_VALUES:
        return stored
    return PROVENANCE_COMPUTED if family else PROVENANCE_EXTRACTED


def _resolve_truth_policy(
    row: dict, provenance: str, override: Optional[str]
) -> str:
    """Default trust matches provenance; explicit values always win.

    Computed → trusted (c02 trusts the harvest-time check).
    Extracted → extracted (c02 defers to judge residue).
    Manual / external → unknown (route conservatively).
    """
    if override:
        if override not in TRUTH_POLICY_VALUES:
            raise ValueError(
                f"unknown truth_policy {override!r}; allowed: {TRUTH_POLICY_VALUES}"
            )
        return override
    stored = row.get("truth_policy")
    if stored in TRUTH_POLICY_VALUES:
        return stored
    if provenance == PROVENANCE_COMPUTED:
        return TRUTH_POLICY_TRUSTED
    if provenance == PROVENANCE_EXTRACTED:
        return TRUTH_POLICY_EXTRACTED
    return TRUTH_POLICY_UNKNOWN


def _collect_truth_strings(row: dict) -> list:
    """Surface forms of the correct answer used by leakage."""
    explicit = row.get("truth_strings")
    if isinstance(explicit, list) and explicit:
        return [str(x) for x in explicit if x not in (None, "")]
    out = []
    for key in ("truth", "answer"):
        value = row.get(key)
        if value not in (None, ""):
            out.append(str(value))
    return out


def _normalise_label(row: dict, pass_at_k: Optional[float], top_wrong_share: float) -> str:
    stored = row.get("label")
    if stored in ("band", "misdirection", "solved", "collapse"):
        return stored
    if pass_at_k is None:
        return "other"
    if pass_at_k > BAND_HI:
        return "solved"
    if pass_at_k >= BAND_LO:
        return "band"
    return "misdirection" if top_wrong_share >= 0.5 else "collapse"


def _first(row: dict, keys, default):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _coerce_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

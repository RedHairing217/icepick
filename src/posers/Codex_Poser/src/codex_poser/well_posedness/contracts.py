"""Contracts for post-pass@k records and well-posedness output."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

BAND_LO = 0.125
BAND_HI = 0.75

PASS = "pass"
FLAG = "flag"
DEFER = "defer"
ERROR = "error"


@dataclass(frozen=True)
class PassKRecord:
    """Normalised post-pass@k record for well-posedness scoring."""

    rid: int
    uid: str
    source: str
    statement: str
    truth_strings: list[str]
    answer_value: Any
    tier: str | None
    family: str | None
    params: dict[str, Any] | None
    provenance: str
    label: str
    pass_at_k: float | None
    n_correct: int
    n_wrong: int
    n_degenerate: int
    modal_wrong: str | None
    top_wrong_share: float | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return self.n_correct + self.n_wrong + self.n_degenerate

    @property
    def n_decided(self) -> int:
        return self.n_correct + self.n_wrong

    @property
    def is_computed(self) -> bool:
        return self.provenance == "computed"

    def summary(self) -> dict[str, Any]:
        return {
            "rid": self.rid,
            "uid": self.uid,
            "source": self.source,
            "provenance": self.provenance,
            "family": self.family,
            "label": self.label,
            "pass_at_k": self.pass_at_k,
            "n_correct": self.n_correct,
            "n_wrong": self.n_wrong,
            "n_degenerate": self.n_degenerate,
        }

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        rid: int,
        default_source: str = "unknown",
    ) -> "PassKRecord":
        statement = str(_first(raw, ("statement", "question", "prompt", "problem"), "") or "")
        source = str(_first(raw, ("source",), default_source) or default_source)
        family = _optional_str(_first(raw, ("family",), None))
        tier = _optional_str(_first(raw, ("tier",), None))
        params = _params(raw, family)
        provenance = _normalise_provenance(raw, family)

        n_correct = _int(_first(raw, ("n_correct", "correct"), 0))
        n_wrong = _int(_first(raw, ("n_wrong", "wrong", "wrong_complete"), 0))
        n_degenerate = _int(_first(raw, ("n_degenerate", "degenerate"), 0))
        pass_at_k = _float_or_none(_first(raw, ("pass_at_k", "pass_rate"), None))
        if pass_at_k is None:
            total = n_correct + n_wrong + n_degenerate
            pass_at_k = round(n_correct / total, 6) if total else None

        top_wrong_share = _float_or_none(_first(raw, ("top_wrong_share",), None))
        label = _normalise_label(raw, pass_at_k, top_wrong_share)
        truth_strings = _truth_strings(raw)
        answer_value = _first(raw, ("answer_value", "answer", "gold_answer", "truth"), None)
        uid = str(_first(raw, ("uid",), "") or _content_uid(source, statement))
        modal_wrong = _optional_str(_first(raw, ("modal_wrong", "top_wrong_value"), None))

        return cls(
            rid=rid,
            uid=uid,
            source=source,
            statement=statement,
            truth_strings=truth_strings,
            answer_value=answer_value,
            tier=tier,
            family=family,
            params=params,
            provenance=provenance,
            label=label,
            pass_at_k=pass_at_k,
            n_correct=n_correct,
            n_wrong=n_wrong,
            n_degenerate=n_degenerate,
            modal_wrong=modal_wrong,
            top_wrong_share=top_wrong_share,
            raw=raw,
        )


@dataclass(frozen=True)
class WellPosednessResult:
    """One record's c01 well-posedness score."""

    check_id: str
    status: str
    score: float
    detail: str
    signals: dict[str, Any] = field(default_factory=dict)

    def to_record(self, record: PassKRecord) -> dict[str, Any]:
        output = record.summary()
        output.update(
            {
                "well_posedness_check": self.check_id,
                "well_posedness_status": self.status,
                "well_posedness_score": self.score,
                "well_posedness_detail": self.detail,
                "signals": self.signals,
            }
        )
        return output


def _content_uid(source: str, statement: str) -> str:
    digest = hashlib.sha1(f"{source}\x00{statement}".encode("utf-8")).hexdigest()
    return digest[:12]


def _first(raw: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return default


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truth_strings(raw: dict[str, Any]) -> list[str]:
    value = raw.get("truth_strings")
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, list):
                return [str(item) for item in loaded if item not in (None, "")]
        if stripped:
            return [stripped]

    values = []
    for key in ("truth", "answer", "gold_answer", "answer_value"):
        item = raw.get(key)
        if item not in (None, ""):
            values.append(str(item))
    return list(dict.fromkeys(values))


def _params(raw: dict[str, Any], family: str | None) -> dict[str, Any] | None:
    value = raw.get("params")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            return loaded
    detail = raw.get("detail")
    if family and isinstance(detail, dict):
        return detail
    return None


def _normalise_provenance(raw: dict[str, Any], family: str | None) -> str:
    """Normalise the raw ``provenance`` field.

    Fail-closed: ``computed`` provenance grants the well-posed-by-
    construction bypass in ``scoring.py`` (the record is never sent to a
    judge), so that category must never be *inferred* — only an explicit
    ``provenance: "computed"`` on the input row may return "computed" here.
    Missing, empty, or unrecognised provenance normalises to "unknown"
    regardless of ``family`` or ``truth_policy``, which routes the record
    through the ordinary defer/judge path exactly like extracted, manual,
    or external records.
    """
    value = str(raw.get("provenance") or "").strip().lower()
    if value in {"computed", "extracted", "manual", "external", "unknown"}:
        return value
    return "unknown"


def _normalise_label(
    raw: dict[str, Any],
    pass_at_k: float | None,
    top_wrong_share: float | None,
) -> str:
    stored = str(raw.get("label") or "").strip().lower()
    if stored in {"solved", "band", "misdirection", "collapse", "too_easy", "too_hard"}:
        return stored
    if pass_at_k is None:
        return "unknown"
    if pass_at_k > BAND_HI:
        return "solved"
    if pass_at_k >= BAND_LO:
        return "band"
    return "misdirection" if (top_wrong_share or 0.0) >= 0.5 else "collapse"

"""Record and check-result shapes shared across subsystems.

``ProblemRecord`` is the silo-agnostic view of one corpus problem. Every
ingest source — generated families, RealMath-style extracted records,
externally supplied JSONL, manually mounted batches — is normalised onto
this shape at the boundary.

Concrete normalisation logic lives in ``icepick.processing.schema``; this
module owns the dataclass and the small enums everyone agrees on. Keeping
the dataclass here means the allocation subsystem can validate handoff
files against the same shape processing will consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

BAND_LO = 0.125
BAND_HI = 0.75

PASS = "pass"
FLAG = "flag"
ERROR = "error"
DEFER = "defer"

PROVENANCE_COMPUTED = "computed"
PROVENANCE_EXTRACTED = "extracted"
PROVENANCE_MANUAL = "manual"
PROVENANCE_EXTERNAL = "external"
PROVENANCE_VALUES = (
    PROVENANCE_COMPUTED,
    PROVENANCE_EXTRACTED,
    PROVENANCE_MANUAL,
    PROVENANCE_EXTERNAL,
)

TRUTH_POLICY_TRUSTED = "trusted"
TRUTH_POLICY_EXTRACTED = "extracted"
TRUTH_POLICY_UNKNOWN = "unknown"
TRUTH_POLICY_VALUES = (
    TRUTH_POLICY_TRUSTED,
    TRUTH_POLICY_EXTRACTED,
    TRUTH_POLICY_UNKNOWN,
)


@dataclass
class CheckResult:
    """One checker's verdict on one record."""

    check_id: str
    status: str
    detail: Optional[str] = None
    score: Optional[float] = None
    payload: Optional[dict] = None


@dataclass
class ProblemRecord:
    """A single corpus problem in silo-agnostic form.

    ``rid`` is the load-order index, kept for human reference. ``uid`` is the
    stable content id — a hash of source and statement, unchanged by input
    order or by which files are present. Verdicts and buckets join back to
    the record across runs by ``uid``.

    ``provenance`` is one of: ``computed`` (truth produced at harvest,
    well-posed by construction, trusted by c02); ``extracted`` (scraped from
    a paper, defers to judge); ``manual`` (mounted by a human operator);
    ``external`` (handed in as JSONL from another system). ``truth_policy``
    encodes whether c02 should trust, defer to, or be conservative with the
    supplied truth. Unknown-policy records route conservatively.
    """

    rid: int
    uid: str
    source: str
    provenance: str
    statement: str
    truth_strings: list
    answer_value: Any
    tier: Optional[str]
    family: Optional[str]
    params: Optional[dict]
    truth_policy: str
    label: str
    pass_at_k: Optional[float]
    n_correct: int
    n_wrong: int
    n_degenerate: int
    modal_wrong: Optional[str]
    top_wrong_share: float
    raw: dict = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return self.n_correct + self.n_wrong

    @property
    def is_computed(self) -> bool:
        return self.provenance == PROVENANCE_COMPUTED

    @property
    def is_band(self) -> bool:
        if self.label == "band":
            return True
        if self.pass_at_k is None:
            return False
        return BAND_LO <= self.pass_at_k <= BAND_HI

    def summary(self) -> dict:
        return {
            "rid": self.rid,
            "uid": self.uid,
            "source": self.source,
            "provenance": self.provenance,
            "truth_policy": self.truth_policy,
            "family": self.family or self.tier,
            "label": self.label,
            "pass_at_k": self.pass_at_k,
            "n_correct": self.n_correct,
            "n_wrong": self.n_wrong,
            "n_degenerate": self.n_degenerate,
        }

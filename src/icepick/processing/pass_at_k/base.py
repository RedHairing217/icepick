"""Pass@k stage primitives: rollout/record dataclasses and the backend protocol.

The stage's contract with the rest of the pipeline:

Input JSONL rows carry ``statement`` (or ``question``/``prompt``/``problem``)
and ``truth`` (or ``answer``). ``uid`` is injected up front if absent, using
the same convention as the wellposed fleet (:func:`icepick.processing.poser.base.compute_uid`).

Output rows gain ``pass_at_k``, ``n_correct``, ``n_wrong``, ``n_degenerate``,
``label``, ``modal_wrong``, ``top_wrong_share`` and ``rollout_uids``; every
original field is preserved. Records that arrive with ``pass_at_k`` already
set pass through untouched — their label is taken as-is (non-goal: no
band-relabeling of ModelBreaker's records).

Labels match ``processing/schema.py:_normalise_label`` exactly (parity is
pinned by a test): ``solved`` / ``band`` / ``misdirection`` / ``collapse``.
``drop`` is this stage's own pre-filter label for records it refuses to
score (garbage truth, unverifiable truth tier) or whose rollouts were
degenerate-dominated; ``drop_reason`` says which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

# --- rollout verdicts --------------------------------------------------------

ROLLOUT_CORRECT = "correct"
ROLLOUT_WRONG = "wrong"
ROLLOUT_DEGENERATE = "degenerate"  # no extractable candidate, or backend error text
ROLLOUT_VERDICTS = (ROLLOUT_CORRECT, ROLLOUT_WRONG, ROLLOUT_DEGENERATE)

# --- labels ------------------------------------------------------------------

LABEL_SOLVED = "solved"
LABEL_BAND = "band"
LABEL_MISDIRECTION = "misdirection"
LABEL_COLLAPSE = "collapse"
LABEL_DROP = "drop"
LABEL_VALUES = (LABEL_SOLVED, LABEL_BAND, LABEL_MISDIRECTION, LABEL_COLLAPSE, LABEL_DROP)

DROP_GARBAGE_TRUTH = "garbage_truth"  # truth is LaTeX macros only / empty
DROP_UNVERIFIABLE = "unverifiable_truth"  # classify() tier not number/tuple/expr
DROP_DEGENERATE = "degenerate_dominated"  # >= half the rollouts had no scoreable answer

# Share of wrong answers the modal wrong value must reach for a sub-band
# record to be ``misdirection`` rather than ``collapse``. Ported from
# ModelBreaker's --misdir-thresh default; identical to the threshold baked
# into processing/schema.py:_normalise_label.
MISDIRECTION_THRESHOLD = 0.5

# Fraction of rollouts that must be degenerate before the record is dropped
# instead of scored (ModelBreaker's "degenerate" label).
DEGENERATE_DROP_FRACTION = 0.5


def rollout_uid(uid: str, sample_idx: int) -> str:
    """Deterministic per-rollout id; joins output rows to rollouts.jsonl."""
    return f"{uid}-r{sample_idx:02d}"


# --- dataclasses -------------------------------------------------------------


@dataclass
class RolloutResult:
    """One of the k samples for one record."""

    rollout_uid: str
    sample_idx: int
    raw_output: str  # exactly what the backend returned, <think> tags included
    candidate: Optional[str]  # extracted answer, None when degenerate
    verdict: str  # one of ROLLOUT_VERDICTS
    from_cache: bool = False

    def to_jsonl_row(self, uid: str) -> dict:
        return {
            "uid": uid,
            "rollout_uid": self.rollout_uid,
            "sample_idx": self.sample_idx,
            "output": self.raw_output,
            "candidate": self.candidate,
            "verdict": self.verdict,
            "from_cache": self.from_cache,
        }


@dataclass
class PassAtKRecord:
    """The stamped scoring result for one record."""

    uid: str
    source: str
    pass_at_k: Optional[float]  # None only for label="drop" pre-filters
    n_correct: int
    n_wrong: int
    n_degenerate: int
    label: str  # one of LABEL_VALUES
    modal_wrong: Optional[str]
    top_wrong_share: float
    rollout_uids: list = field(default_factory=list)
    drop_reason: Optional[str] = None  # set iff label == "drop"

    def stamp(self, record: dict) -> dict:
        """Merge onto the original row; original fields win nothing, we win.

        Returns a new dict — the input row is never mutated.
        """
        out = dict(record)
        out["pass_at_k"] = self.pass_at_k
        out["n_correct"] = self.n_correct
        out["n_wrong"] = self.n_wrong
        out["n_degenerate"] = self.n_degenerate
        out["label"] = self.label
        out["modal_wrong"] = self.modal_wrong
        out["top_wrong_share"] = self.top_wrong_share
        out["rollout_uids"] = list(self.rollout_uids)
        if self.drop_reason is not None:
            out["drop_reason"] = self.drop_reason
        return out


# --- backend protocol --------------------------------------------------------


class ModelBackend(Protocol):
    """Subject-model surface. Injectable so runner tests use fakes.

    ``call`` returns exactly ``k`` raw model outputs for one question.
    Implementations do NOT retry — retry/backoff lives at the runner layer
    (mirrors the cascade). Implementations raise on transport errors and
    never return partial lists.
    """

    name: str

    def call(
        self,
        question: str,
        *,
        k: int,
        temperature: float,
        max_tokens: int,
        think: bool,
        timeout: float,
    ) -> list: ...

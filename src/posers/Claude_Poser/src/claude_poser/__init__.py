"""Claude_Poser: isolated c01 well-posedness check.

Input: post-pass@k JSONL records (one problem per line).
Output: JSON or CSV with a well-posedness score per record.

The rest of the ModelBreaker processing pipeline is presumed to live elsewhere;
this package only owns the c01 check, its judge tier, and score reporting.
"""

from .wellposed import check_record, check_records
from .scoring import score_from_check
from .schema import normalise_record, compute_uid

__all__ = [
    "check_record",
    "check_records",
    "score_from_check",
    "normalise_record",
    "compute_uid",
]

"""Processing stage: pass@k difficulty scoring.

Runs each record's problem ``k`` times against a subject model, verifies
every rollout against the truth answer (sympy-backed numeric/symbolic
equivalence, ported from ModelBreaker's realmath verifier), and stamps
``pass_at_k`` + ``label`` onto the record for the difficulty-band filter.

Pipeline placement (recommended):

    allocation -> processing groundtruth -> processing wellposed-cascade
               -> processing pass_at_k -> final_corpus.jsonl

or standalone on any handoff JSONL via
``icepick processing pass_at_k --input <handoff> --output-dir <out>``.

Restartability matches the scraper: pause/restart acceptable, full kill
unacceptable. Every finished rollout and record is committed to
``<output_dir>/_progress/`` as it happens; a resumed run re-bills nothing
it already paid for.
"""

from icepick.processing.pass_at_k.base import (  # noqa: F401
    LABEL_BAND,
    LABEL_COLLAPSE,
    LABEL_DROP,
    LABEL_MISDIRECTION,
    LABEL_SOLVED,
    LABEL_VALUES,
    ModelBackend,
    PassAtKRecord,
    RolloutResult,
)
from icepick.processing.pass_at_k.config import (  # noqa: F401
    BACKEND_VALUES,
    PassAtKConfig,
)

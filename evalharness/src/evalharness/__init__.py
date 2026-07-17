"""LoRA eval harness for qwen3-8b on the pde625 band corpus.

Standalone sub-repo (mirrors the ``src/posers/*`` pattern): no import
dependency on ``icepick`` itself. ``run_eval.py`` talks to the ``icepick``
package only by shelling out to its installed console script
(``icepick processing pass_at_k``); it never imports icepick modules.

See ``docs/eval_harness_design.md`` in the parent icepick repo for the
full design (frozen split rationale, measurement protocol, sub-repo
layout). This package implements exactly that design:

  build_eval_set.py -- split + remote-rescore cascade -> eval_set.jsonl
                        (eval-band + anchor-solved + anchor-fail) and
                        train_uids.txt (final band minus eval papers).
  run_eval.py        -- subprocess-drives ``icepick processing pass_at_k``
                        for greedy pass@1 (primary) and optional k=8
                        temp=0.7 x3 (secondary, distributional).
  report.py          -- paired greedy diff, exact McNemar, anchor drift,
                        markdown report to stdout + file.
"""

from __future__ import annotations

__version__ = "0.1.0"

"""loratrain -- LoRA training arm for qwen3-8b on the pde625 band corpus.

Standalone sub-repo (mirrors the ``evalharness/`` and ``src/posers/*``
pattern): stdlib-only, zero import dependency on ``icepick`` or on
``evalharness``. This package TRAINS; it never measures -- measurement
belongs exclusively to ``evalharness/``. The only contract between the
two is file-shaped: loratrain consumes evalharness's derived
``train_uids.txt`` (and, downstream, its eval set / baseline run), and
produces a LoRA adapter for evalharness to score.

See ``README.md`` in this directory for the full design (decisions
D1-D4, the exact train->serve recipe, and the phased W0-W5 plan). This
package implements exactly that design, as stubs:

  config.py        -- single source of truth: server address, pins,
                       hyperparameters, paths, validate_config().
  build_dataset.py -- W2 STUB + REAL guards: leakage / dedupe / trace
                       integrity, harvesting verified-correct rollouts
                       into an SFT jsonl.
  train_lora.py     -- W3 STUB + REAL ordering guard + remote-client
                       payload shape (thin client to the operator's
                       training server).
  export_serve.py   -- W4 STUB: PEFT adapter -> GGUF adapter ->
                       llama-server command builder.

Every stub validates its real guards before raising
``NotImplementedError`` with its gate name, so "train before baseline"
etc. are structurally hard from day one, per README.
"""

from __future__ import annotations

__version__ = "0.1.0"

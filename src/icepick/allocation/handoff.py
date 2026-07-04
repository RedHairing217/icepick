"""Write canonical JSONL handoff into ``out/intake/.../handoff/``.

Mounted source files are never modified in place. Manual mounts and CSV
column maps produce derived handoff files here; processing consumes
those files via the same ``load_inputs`` path as any other JSONL.
"""

from __future__ import annotations


def write_handoff(records, *, source, output_dir):
    raise NotImplementedError("allocation.handoff.write_handoff is not yet implemented")

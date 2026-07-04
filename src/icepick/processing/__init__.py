"""Processing subsystem.

Three working stages, run in this order:

  1. ``ingest``       - load JSONL into normalised ``ProblemRecord``s
                        (``schema.from_raw`` does the work).
  2. ``groundtruth``  - publication-status filter via Anthropic web_search.
                        Discards records whose source arXiv paper isn't
                        peer-reviewed and indexed.
  3. ``poser``        - well-posedness filter via the Claude_Poser /
                        Codex_Poser fleet (one or more provider
                        combinations in parallel). The poser stage IS
                        the gate — records that pass it are the final
                        filtered corpus.

Does NOT decide what new data to acquire and does NOT call allocation or
agent modules.
"""

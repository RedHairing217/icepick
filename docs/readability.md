# Human-readability principles

The repo should be easy for a human operator to scan before it is clever.
Every implementation review must explicitly reference these seven
procedures (copied verbatim from the spec):

1. **Structure and scanning** — few, purpose-labelled top-level folders.
   Stable contracts in `contracts/`, processing in `processing/`,
   allocation in `allocation/`, hosts in `llm_hosts/`, chat in `agent/`.
   Generated outputs under `out/` with the same shape every run.
2. **Naming and labeling** — domain names (`gate`, `routing`, `triage`,
   `confirm`, `manifest`, `bucket`, `calibration_replay`). No `utils`,
   `helpers`, `misc`. Host roles labelled `subject` and `manager`
   everywhere. Processor modes labelled `production` and `flow_testing`
   everywhere. Records carry both `rid` and `uid`. Every output stamps
   `source`, `provenance`, `processor_mode`, and `created_at`.
3. **Visual hierarchy** — markdown reports lead with outcome, then
   counts, then action items, then per-record detail. Summaries have
   stable top-level sections (`run`, `inputs`, `capabilities`, `counts`,
   `buckets`, `review_lanes`, `parameters`, `warnings`). CLI shows run
   id, mode, input count, bucket counts, warnings, output paths — and
   nothing else by default.
4. **Progressive disclosure** — default commands show summaries only.
   `--verbose` for per-check detail. `--explain UID` for one-record
   drilldown. Detailed judge replies and raw model responses live in
   logs unless explicitly requested.
5. **Forms** — manifests are form-like with one clear field per
   decision. Enums over free text. Approval fields are easy to locate.
   Dangerous or costly choices grouped together. Ambiguous forms are
   refused, not guessed.
6. **Familiar structures** — standard Python package layout. JSONL for
   record streams, JSON for summaries/manifests, Markdown for human
   handoffs. Conventional CLI verbs: `plan`, `run`, `gate`, `confirm`,
   `summary`, `list`, `show`, `validate`.
7. **Standardization** — every stage writes a summary file and a
   machine-readable output. Every stage records mode and input paths.
   Every failure carries a code, a short label, and a human-readable
   detail. Every bucket and lane has a stable filename. Every command
   supports `--output-dir`. Every destructive or costly command supports
   `--dry-run` or requires approval.

A feature is not complete until its files, names, CLI output, and reports
are readable by a human operator.

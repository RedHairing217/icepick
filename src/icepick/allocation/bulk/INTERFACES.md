# arxiv_bulk — Frozen Interface Contract (Gate F0)

Status: FROZEN 2026-07-06 (orchestrator: Fable). Builders code against THIS file.
Any change to a signature or invariant below goes through the orchestrator, not a builder.

Provenance: the prior-session design skeleton is not on disk; §5 invariants were
reconstructed from the build command set + repo conventions and are binding as written here.

## 0. Provider decision memo (Gate F1, decided early — recon forced it)

- **Provider: S3 requester-pays** (`s3://arxiv/src/`). It is the ONLY provider of LaTeX
  source chunks. `gs://arxiv-dataset` is ruled out empirically (2026-07-06): PDF-only
  (no `src/` prefix), stale since 2020-12-02.
- Bucket is requester-pays for EVERYTHING including `src/arXiv_src_manifest.xml`
  (anonymous fetch → 403, verified). AWS credentials required for all paid paths.
- **This machine has no AWS credentials or CLI.** All paid paths (manifest fetch ≈ $0.005,
  single-chunk probe ≈ $0.05, batch downloads) are unreachable until Nicky provides
  credentials at the W4 gate. The $60 hard stop is therefore satisfied by construction
  during W2/W3.
- Pre-manifest spend bracket (documented basis, NOT for approval): src chunks ≈ 500 MB
  each; recent arXiv src growth very roughly 40–50 GB/month → whole 18-month window
  (2025-01 → 2026-07) ≈ $63–81 at $0.09/GB (OVER the stop); a 1–2 month window ≈ $4–9.
  Real numbers come from the src manifest at W4; window scoping is Nicky's call there.
- Code consequence: implement the S3 client behind a provider seam (`ProviderClient`,
  §3). Do NOT build a GCS client.
- Manifest acquisition is an OPERATOR step, not a run step: the src manifest XML is
  fetched once (with creds, at W4+) to a local path, and production plans carry that
  path in `scrape_window["manifest_path"]`. `estimate()` and `run()` parse it locally
  and never fetch it.

## 1. bulk/manifest.py  (owner: sonnet-A)

```python
EGRESS_USD_PER_GB = 0.09          # AWS us-east-1 → internet, basis for all cost math

class ManifestError(ValueError): ...

@dataclass(frozen=True)
class ManifestEntry:              # one <file> element of arXiv_src_manifest.xml
    filename: str                 # S3 key, e.g. "src/arXiv_src_2501_001.tar"
    yymm: str                     # "2501"; KEEP AS STRING ("0001" = Jan 2000 exists)
    seq_num: int
    first_item: str               # arXiv id of first paper in chunk
    last_item: str
    num_items: int
    size_bytes: int               # <size>
    md5sum: str                   # tar-level checksum, hex
    content_md5sum: str
    timestamp: str                # "YYYY-MM-DD HH:MM:SS" verbatim, never parsed

def parse_manifest(xml_text: str) -> list[ManifestEntry]
    # pure; ManifestError on malformed XML or any missing field

def select_chunks(entries: list[ManifestEntry], *, year: int,
                  month: Optional[int] = None) -> list[ManifestEntry]
    # yymm window; month=None → whole year; empty result is valid, not an error

def chunks_for_ids(entries: list[ManifestEntry],
                   wanted_ids: set[str]) -> list[ManifestEntry]
    # minimal chunk set whose [first_item, last_item] ranges cover the wanted ids.
    # New-style ids only ("2501.00123"): compare (yymm, numeric suffix).
    # Old-style ids (e.g. "math/0501123") are OUT OF SCOPE v1: skip + warning string
    # channel (return shape stays list; warnings via module-level helper or raise —
    # builder documents choice in docstring; window ≥ 2025-01 makes this unreachable).

@dataclass(frozen=True)
class ChunkRollup:
    chunk_count: int
    total_bytes: int
    egress_usd: float             # total_bytes / 1 GiB? NO — decimal GB (1e9), × 0.09

def rollup(entries: list[ManifestEntry]) -> ChunkRollup
```

Fixture: hand-written `tests/fixtures/arxiv_bulk/src_manifest_sample.xml`, ~6 entries
across ≥3 months (e.g. 2412, 2501 ×2 chunks, 2502), field values shaped exactly per
the documented schema (filename, yymm, seq_num, first_item, last_item, num_items,
size, md5sum, content_md5sum, timestamp).

## 2. bulk/category_index.py  (owner: sonnet-B)

```python
@dataclass(frozen=True)
class OAIResponse:                # the injectable-fetcher return, minimal by design
    status: int                   # 200 | 503 | ...
    retry_after: Optional[float]  # parsed Retry-After seconds, None if absent
    text: str                     # XML body ("" on non-200)

# fetcher seam: Callable[[str], OAIResponse]  — takes the full request URL

@dataclass(frozen=True)
class PaperMeta:
    arxiv_id: str
    primary_category: str         # first entry of <categories>
    categories: tuple[str, ...]
    title: str                    # single-line, whitespace-collapsed
    # NO abstract — deliberate (index size); see §4 Paper construction

class CategoryIndex:
    def __init__(self, cache_dir: Path): ...
    def build(self, *, oai_set: str, fetcher: Callable[[str], OAIResponse],
              base_url: str = "https://oaipmh.arxiv.org/oai",
              metadata_prefix: str = "arXiv",
              sleeper: Callable[[float], None] = time.sleep,
              from_date: Optional[str] = None) -> None
        # AMENDED at F2 (builder catch): sleeper was mandated by prose below
        # but missing from the frozen signature; it is part of the contract.
        # AMENDED at F3 (W3 H2): from_date ("YYYY-MM-DD") adds an OAI `from`
        # datestamp bound to the INITIAL request only (tokens carry state).
        # SUPERSET semantics, not a filter: datestamp >= submission date always,
        # so every window submission is included; yymm id filtering still
        # happens client-side in ids_for. Bounds the walk to the window era
        # instead of the whole set history.
        # ListRecords paging via resumptionToken until exhausted. SERIAL requests
        # only. Honor 503 + Retry-After (sleep exactly retry_after when given;
        # default backoff when absent, capped, journaled). Each fetched page is
        # cached to cache_dir BEFORE the next request → build is resumable after
        # a kill without refetching completed pages (resumption tokens expire
        # daily; page cache is what survives).
    def lookup(self, arxiv_id: str) -> Optional[PaperMeta]
    def ids_for(self, *, category: str, yymm: str,
                primary_only: bool) -> list[str]
        # yymm filter = new-style id prefix match
    oai_requests: int             # EVERY issued HTTP request, retries included
    # AMENDED at F2 — landed as public attrs, realmath-shaped, lifetime totals
    # per instance (adapter snapshots per run):
    rate_limit_events: int
    rate_limit_backoff_seconds: float
    rate_limit_statuses: dict[str, int]   # str(status) keys, realmath's shape
```

OAI facts the fixtures must model: metadataPrefix `arXiv` carries categories + title;
resumptionToken paging; datestamps are MODIFICATION dates → no submission-date
filtering server-side (that is exactly why yymm filtering happens client-side on ids).
Fixtures: 2 canned ListRecords pages (page 2 via resumptionToken) + one 503-with-
Retry-After sequence. Sleeping must go through an injectable `sleeper` (default
`time.sleep`) so tests assert the honored delay without waiting.

## 3. bulk/chunk_store.py  (owner: sonnet-C)

```python
class ChecksumError(RuntimeError): ...

# provider seam — the ONLY thing an S3/other client must implement:
# class ProviderClient(Protocol):
#     def download(self, key: str, dest: Path) -> None

def s3_client(*, region: str = "us-east-1") -> ProviderClient
    # boto3 imported LAZILY here (dep lives in the [bulk] extra, already added);
    # RequestPayer="requester" on every call; creds from the standard AWS chain,
    # never from icepick config. Unit-tested only via the seam, never live.

class ChunkStore:
    def __init__(self, client: ProviderClient, *, work_dir: Path,
                 max_resident: int = 2): ...
    def fetch(self, entry: ManifestEntry) -> Path
        # download to work_dir, verify md5 against entry.md5sum → ChecksumError
        # (and delete the corrupt file); enforce ≤ max_resident chunks on disk
        # (fetch of a third blocks/raises per builder choice — document it;
        # counters: chunk_downloads += 1, chunk_bytes += entry.size_bytes.
        # A re-fetch of a still-resident, checksum-verified chunk is a no-op
        # (no counter increment) — resume support.
    def extract_matching(self, chunk_path: Path,
                         wanted_ids: set[str]) -> Iterator[tuple[str, bytes]]
        # stream tar members; yield (arxiv_id, raw member bytes) for wanted ids;
        # member bytes are OPAQUE (gz vs tar.gz sniffing stays downstream, exactly
        # where realmath's default fetcher leaves it today)
    def purge(self, chunk_path: Path) -> None
    def release(self, entry_or_path) -> None   # purge accepting entry or path
        # AMENDED at F2: original prose said "extract-then-delete convenience"
        # but the signature carries no wanted_ids — extract stays the §4
        # adapter's job; release() is purge-by-entry-or-path.
    chunk_downloads: int          # verified fetches only
    chunk_bytes: int              # bytes of verified fetches only
    corrupt_downloads: int        # AMENDED at F2: checksum-failed transfers —
    corrupt_bytes: int            # they billed real egress; invariant 2 requires
                                  # them counted; adapter surfaces when nonzero
```

Residency semantics (decided at F2): fetch beyond max_resident RAISES
(RuntimeError naming limit + remedy) — blocking would deadlock the §4
single-threaded fetch→extract→release loop. Refetch-of-resident bypasses the
cap; adoption of a pre-existing verified file is cap-checked. Residency is
tracked per store instance; one store per work_dir is the only supported
topology.

Assumed inner layout (UNVERIFIED until the W4 probe — keep it isolated in ONE
member-name→arxiv_id helper): members like `2501/2501.00123.gz`. Tests build the
synthetic 2-paper tarball at test time into tmp_path (no binary fixture committed).

## 4. adapters/arxiv_bulk.py  (owner: opus-D)

`SOURCE_ARXIV_BULK = "arxiv_bulk"` — ALREADY LANDED in `contracts/manifests.py`
(constant, `SOURCE_TYPES`, `requires_calls()`); import it, never redefine it.

Mirror `realmath_scrape` stage-for-stage. Import public pieces from
`icepick.allocation.adapters.realmath_scrape`: `ScrapeRunResult`, `PAPERS_PER_RECORD`,
`CANDIDATES_PER_PAPER`, `ESTIMATE_SAFETY_MULTIPLIER`, and `normalise` machinery (below).

- `plan(request: dict) -> ProposedPlan` — same required/optional fields as realmath;
  `scrape_window` fields for bulk = realmath's set + `manifest_path` (str, LOCAL path;
  required for production, absent for flow_testing). `extraction` limited to
  {"latex", "qa"} — validate and refuse others.
- `estimate(plan) -> dict` — same key shape as realmath's estimate dict; `call_kinds`
  covers `oai_requests` / `chunk_downloads` / `qa_calls`; ADD two keys:
  `expected_chunk_bytes`, `expected_egress_usd` (from manifest rollup of the window —
  parse `manifest_path` locally). Reuse the planning ratios + safety multiplier;
  estimates round AGAINST the operator, never under.
- `run(manifest, *, now=None) -> ScrapeRunResult` — production flow:
  parse manifest_path → `select_chunks` by window → `CategoryIndex` (cache under the
  run's progress dir or manifest-specified cache_dir) → `ids_for(category, yymm,
  primary_only)` → cap by `max_papers` → `chunks_for_ids` → per chunk:
  `fetch` → `extract_matching` → build LOCAL `source_fetcher` (dict-backed;
  `(arxiv_id, **kw) -> bytes`; missing id → skip paper + warning, never network) →
  `qa_extractor` / `latex_extractor` from `icepick.allocation.scrape.realmath` with
  `ScrapeCheckpoint(progress_dir)` + `caching_generator` for QA → delete chunk →
  next chunk (≤2 resident). `Paper` construction: `Paper(arxiv_id, link=
  f"https://arxiv.org/abs/{arxiv_id}", title=meta.title, abstract="",
  primary_category=meta.primary_category, categories=list(meta.categories),
  published="")` — abstract/published deliberately empty; VERIFY no latex/qa
  extractor path reads them (if one does, STOP and note for orchestrator).
- Budget: `call_budget` caps `oai_requests + chunk_downloads + qa_calls`
  (`total_calls` = that sum; `chunk_bytes` is telemetry, not a call). Exhaustion =
  checkpointed PAUSE, not error: `interrupted=True`, resumable — mirror realmath's
  `_BudgetExhausted` semantics (import-with-note or minimal local equivalent marked
  `# F2-MERGE:`).
- Resume: ScrapeCheckpoint gives paper-level resume + QA cache. Chunk-level resume:
  journal completed chunks to `chunks_done.jsonl` ALONGSIDE the checkpoint's ledger
  files, i.e. directly inside `checkpoint.progress_dir` (which already IS the
  `<run_dir>/_progress` directory — AMENDED at F2: the original wording doubled the
  `_progress` segment and was implemented literally; fixed) via adapter-local helper —
  do NOT modify checkpoint.py. A journaled chunk is never re-downloaded on resume.
- `acquisition` dict (production): `{"oai_requests", "chunk_downloads", "chunk_bytes",
  "corrupt_downloads", "corrupt_bytes", "qa_calls", "qa_model", "rate_limit_events",
  "rate_limit_backoff_seconds", "rate_limit_statuses", "token_usage", "total_calls",
  "call_budget", "resumed_papers", "spend_rows"}` — `spend_rows` = `[["oai_request", n],
  ["chunk_download", n], ["chunk_gb", round(bytes/1e9, 3)], ["qa_generation (<model>)",
  n]]` plus `["chunk_download_corrupt", n]` ONLY when n>0; `_write_report` already
  renders `spend_rows` when present (landed at F0). (Key list AMENDED at F3 — it was
  stale vs the §3 corrupt-counter amendment; W3 review L4.)
- Budget nuance (deliberate, W3 L1): a checksum-failed transfer REFUNDS its
  call_budget charge — `total_calls` counts successful acquisition calls, matching
  realmath semantics — while its real egress stays visible via
  corrupt_downloads/corrupt_bytes + the corrupt spend row. Charge only what the
  store actually downloaded: an ADOPTED resident chunk (resume) must not increment
  chunk_downloads/chunk_bytes nor consume budget (W3 M3).
- Pause retention (deliberate, W3 M2): a run interrupted mid-chunk (budget pause /
  Ctrl-C) RETAINS the in-flight chunk file in the run's own `_chunks/` dir — resume
  adopts it free instead of re-billing egress (invariant 3 outranks tidiness). The
  pause path emits a warning naming the retained path. Completed runs end with
  `_chunks/` empty.
- estimate() OAI term (AMENDED at F3, W3 H2): for production (manifest_path
  present), price the index walk from the manifest itself — records ≈ sum of
  `num_items` over entries with `yymm >=` window start (an ALL-categories superset
  of the set walk; deliberate over-provision of an unpriced unit — never under);
  pages = ceil(records / 1000), safety multiplier applied. The old
  target_count-scaled term is wrong (walk size is corpus-shaped, not sample-shaped).
- Run layout: IDENTICAL tree to realmath (manifest.json, handoff/records.jsonl,
  handoff/surplus_records.jsonl, raw/*.jsonl, reports/source_report.md). Import
  `_write_run` from realmath_scrape with an `# F2-PROMOTE: _write_run` comment at the
  import site (same for any other private helper you need: `_paper_pool`,
  `_canonical_record`, `_read_fixture_candidates`, ... — import + mark, NEVER copy).
- `normalise(raw_outputs) -> NormaliseResult` — thin wrapper over realmath's
  canonicalisation path so records satisfy the same `_CANONICAL_KEYS`; provenance
  `extracted`, truth policy per realmath conventions; source stamped from
  `raw_outputs["source_name"]` as realmath does.
- flow_testing: replay `manifest.calibration_sheet` fixture EXACTLY like realmath
  (same JSONL row shape: link/question/answer/tier/truth + optional title/arxiv_id);
  no network, no manifest_path needed, auto-approvable by creator.
- Surplus: cap overflow rows are NEVER dropped — they flow to surplus_records.jsonl
  exactly as realmath does (house rule: never reject accepted theorems).

Tests (mirror realmath's file-per-concern): test_arxiv_bulk_plan.py, _estimate.py,
_flow_testing.py, _production.py, _resume.py under tests/allocation/adapters/.
Production tests inject EVERYTHING (fake index pages via fetcher seam, fake provider
client, fixture QA generator); copy the socket-blocking `_no_network` autouse-fixture
pattern from tests/allocation/scrape/test_realmath_qa.py into each new test file.
Fixture: tests/fixtures/arxiv_bulk/qa_candidates.jsonl (flow_testing sheet, ≥6 rows,
shape-parity with tests/fixtures/realmath/qa_candidates.jsonl).

## 5. Invariants (reconstructed §4 — BINDING, reviewed adversarially at W3)

1. **Approval-gated spend.** `run()` only from an `ApprovedManifest`; production
   manifests require human approval; flow_testing may be creator-approved.
   `requires_calls()` includes `SOURCE_ARXIV_BULK` (landed).
2. **Complete budget accounting.** Every acquisition unit (oai_requests,
   chunk_downloads, chunk_bytes, qa_calls) appears in `acquisition` AND renders in
   reports/source_report.md via `spend_rows`. No paid/countable action outside a
   counter. `call_budget` enforced as pause-not-error.
3. **Resume never double-bills.** Re-running the same command resumes: checkpointed
   papers not re-processed, cached QA not re-called, journaled chunks not
   re-downloaded, still-resident verified chunks not re-fetched.
4. **Offline-only tests.** No socket use anywhere in tests (socket-guard fixture in
   every new test file); all inputs are on-disk fixtures; flow_testing is
   deterministic. No live OAI call, no S3 call, no scrape during W2/W3 — ever.
5. **Disk discipline.** ≤2 chunks resident; chunks deleted after extraction (a
   PAUSED run retains its in-flight chunk for free resume adoption — see §4);
   writes confined to the run's own output tree + declared cache/work dirs.
   SCOPE CLARIFIED at F3 (W3 M4): the `out/` prohibition binds BUILD/VERIFICATION
   agents in this effort (live batch runs + batch 9 are HELD and off-limits to
   them). At runtime the shipped adapter legitimately writes NEW run trees under
   whatever output_dir the operator approves — including out/ — and must never
   write outside its own `<output_dir>/runs/<run_id>` tree (which run_dir_for's
   shape already enforces).
6. **Credential hygiene.** AWS creds only via the standard AWS chain; QA key only via
   the established key-file mechanism; nothing key-shaped in code, fixtures, logs,
   reports, or commits.
7. **Schema parity.** Handoff records satisfy realmath's `_CANONICAL_KEYS`; identical
   run layout; `ScrapeRunResult`/`NormaliseResult` imported, not redefined; surplus
   preserved, never discarded.
8. **Ownership boundary.** Builders write ONLY their §-assigned files. Orchestrator-only:
   `cli.py`, `contracts/manifests.py`, `allocation/scrape/checkpoint.py`,
   `allocation/scrape/realmath.py`, `adapters/realmath_scrape.py`, `pyproject.toml`,
   `CLAUDE.md`, `docs/**`, this file, all `__init__.py`. Private helpers you need:
   import + `# F2-PROMOTE:` mark, or minimal local stub + `# F2-MERGE:` mark. Never
   copy logic, never edit another owner's file.

## 6. File ownership

| path | owner |
| --- | --- |
| src/icepick/allocation/bulk/manifest.py, tests/allocation/bulk/test_manifest.py, tests/fixtures/arxiv_bulk/src_manifest_sample.xml | sonnet-A |
| src/icepick/allocation/bulk/category_index.py, tests/allocation/bulk/test_category_index.py, tests/fixtures/arxiv_bulk/oai_*.xml | sonnet-B |
| src/icepick/allocation/bulk/chunk_store.py, tests/allocation/bulk/test_chunk_store.py | sonnet-C |
| src/icepick/allocation/adapters/arxiv_bulk.py, tests/allocation/adapters/test_arxiv_bulk_*.py, tests/fixtures/arxiv_bulk/qa_candidates.jsonl | opus-D |
| everything else | orchestrator |

Baseline before this work: **449 passed, 3 skipped** (measured 2026-07-06; CLAUDE.md's
428 is stale — orchestrator fixes at W5). Each builder leaves the FULL suite green.

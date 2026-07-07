"""Chunk store: download, verify, stream-extract, and purge arXiv src tars.

Contract: INTERFACES.md §3 (frozen 2026-07-06).  The consuming adapter is
``icepick.allocation.adapters.arxiv_bulk``; the provider decision (§0) is S3
requester-pays, ``s3://arxiv/src/`` — the only live provider, reached solely
through the :class:`ProviderClient` seam so unit tests never touch boto3 or
the network.

Duck-typed manifest entries
    ``fetch``/``release`` accept any object exposing the frozen §1
    ``ManifestEntry`` attributes this module reads: ``.filename`` (the S3
    key, e.g. ``"src/arXiv_src_2501_001.tar"``), ``.size_bytes`` and
    ``.md5sum``.  The real ``icepick.allocation.bulk.manifest.ManifestEntry``
    satisfies this, but the import is type-checking-only so this module (and
    its tests) stand independently of ``manifest.py`` landing.

Residency policy (builder choice: RAISE, not block)
    At most ``max_resident`` chunks may be on disk at once (invariant §5.5).
    A ``fetch`` that would register a chunk beyond the cap raises
    ``RuntimeError`` immediately.  Rationale: the adapter drives this store
    from a single-threaded fetch → extract → release loop, so no other actor
    can ever free a slot while a blocked ``fetch`` waits — blocking would
    deadlock and hide the caller's missing ``purge``/``release``.  Raising
    surfaces the bug at the call site; ``purge``/``release`` frees a slot.

Counter semantics (verified and corrupt tracked SEPARATELY)
    ``chunk_downloads += 1`` and ``chunk_bytes += entry.size_bytes`` happen
    only when a download completes AND its md5 matches ``entry.md5sum``.
    A checksum mismatch deletes the corrupt file and raises
    :class:`ChecksumError`; the verified counters stay untouched, but the
    transfer DID bill real egress, so it is recorded in
    ``corrupt_downloads``/``corrupt_bytes`` (§3 amendment at F2; invariant
    §5.2 — no paid action outside a counter).  A re-fetch of a
    still-resident verified chunk is a no-op with no increment; likewise a
    pre-existing on-disk file that verifies is adopted without a download or
    increment (resume support, invariant §5.3).  A pre-existing file that
    fails verification (e.g. a partial download from a killed run) is
    deleted and re-downloaded fresh.

Member-name → arXiv-id mapping
    Isolated in the single helper :func:`_member_arxiv_id`.  The assumed
    inner layout — members like ``2501/2501.00123.gz`` — is UNVERIFIED until
    the W4 probe (INTERFACES.md §3); swap that one helper if the probe shows
    a different layout.  Member bytes stay opaque (gz vs tar.gz sniffing is
    downstream's job, exactly where realmath's default fetcher leaves it).
"""

from __future__ import annotations

import hashlib
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterator, Optional, Protocol, Union

if TYPE_CHECKING:  # runtime-independent of manifest.py (see module docstring)
    from icepick.allocation.bulk.manifest import ManifestEntry


class ChecksumError(RuntimeError):
    """A downloaded chunk's md5 did not match the manifest ``md5sum``."""


class ProviderClient(Protocol):
    """The ONLY surface a chunk provider must implement (§3 seam)."""

    def download(self, key: str, dest: Path) -> None:
        """Download object ``key`` to local path ``dest``."""
        ...  # pragma: no cover — protocol stub


_S3_BUCKET = "arxiv"  # s3://arxiv/src/ — manifest filenames already carry "src/"


class _S3Client:
    """Requester-pays S3 provider for bucket ``arxiv``.

    Built only via :func:`s3_client`; never constructed in unit tests
    (tests exercise the seam with fakes — invariant §5.4).
    """

    def __init__(self, s3: object) -> None:
        self._s3 = s3

    def download(self, key: str, dest: Path) -> None:
        # RequestPayer="requester" on EVERY transfer: the bucket 403s
        # anonymous/unmarked requests and the account owns the egress bill.
        self._s3.download_file(  # type: ignore[attr-defined]
            Bucket=_S3_BUCKET,
            Key=key,
            Filename=str(dest),
            ExtraArgs={"RequestPayer": "requester"},
        )


def s3_client(*, region: str = "us-east-1") -> ProviderClient:
    """Build the live S3 provider client (requester-pays, bucket ``arxiv``).

    boto3 is imported lazily HERE — the dependency lives in the ``[bulk]``
    extra, so this module must import cleanly without it.  Credentials come
    exclusively from the standard AWS chain (env vars, shared config,
    instance role); icepick config never supplies or sees them (invariant
    §5.6).

    Raises:
        ImportError: with install guidance, when boto3 is not installed.
    """
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for the arxiv_bulk S3 client but is not "
            "installed; install the extra: pip install 'icepick[bulk]'"
        ) from exc
    return _S3Client(boto3.client("s3", region_name=region))


# --- member-name mapping (ONE helper; assumed layout, UNVERIFIED until W4) ----

_NEWSTYLE_ID_RE = re.compile(r"^\d{4}\.\d{4,6}$")


def _member_arxiv_id(member_name: str) -> Optional[str]:
    """Map a tar member name to a new-style arXiv id, or ``None`` to skip.

    Assumed layout (UNVERIFIED until the W4 probe): ``2501/2501.00123.gz``
    → ``"2501.00123"``.  Anything that is not ``<basename ending .gz>``
    whose stem parses as ``YYMM.NNNN[NN]`` returns ``None`` — unknown or
    unparseable member names are skipped by callers, never fatal.  If the
    probe reveals a different layout, change ONLY this helper.
    """
    basename = PurePosixPath(member_name).name
    if not basename.endswith(".gz"):
        return None
    stem = basename[: -len(".gz")]
    if not _NEWSTYLE_ID_RE.match(stem):
        return None
    return stem


class ChunkStore:
    """Fetch-verify-extract-purge lifecycle for arXiv src chunks.

    See the module docstring for the residency (raise-not-block), counter,
    and resume semantics; all are load-bearing parts of the §3 contract.

    Attributes:
        chunk_downloads: completed, checksum-VERIFIED downloads only.
        chunk_bytes: sum of ``entry.size_bytes`` over those same downloads.
        corrupt_downloads: checksum-FAILED transfers (file deleted, fetch
            raised ChecksumError).  Kept separate from the verified
            counters — the two never overlap — because a failed transfer
            still billed real egress (invariant §5.2).
        corrupt_bytes: actual bytes transferred by those failed downloads,
            measured from the bad file on disk before deletion (falls back
            to ``entry.size_bytes`` if its size is unreadable).
    """

    def __init__(
        self,
        client: ProviderClient,
        *,
        work_dir: Path,
        max_resident: int = 2,
    ) -> None:
        self._client = client
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._max_resident = max_resident
        self._resident: set[Path] = set()
        self.chunk_downloads: int = 0
        self.chunk_bytes: int = 0
        self.corrupt_downloads: int = 0
        self.corrupt_bytes: int = 0

    # --- fetch ----------------------------------------------------------------

    def fetch(self, entry: "ManifestEntry") -> Path:
        """Download ``entry`` into ``work_dir`` and return the verified path.

        No-op (no download, no counters) if the chunk is already resident
        and verified.  Raises :class:`ChecksumError` after deleting the file
        when the downloaded bytes do not match ``entry.md5sum``.  Raises
        ``RuntimeError`` when a new chunk would exceed ``max_resident``
        (raise-not-block: see module docstring); free a slot with
        :meth:`purge` or :meth:`release` first.
        """
        dest = self._local_path_for(entry.filename)

        if dest in self._resident and dest.exists():
            return dest  # still-resident + verified at fetch time → no-op
        self._resident.discard(dest)  # registered but vanished → refetch

        if dest.exists():
            # Pre-existing file from a previous run: adopt it if it
            # verifies (resume, no download/counters); else it is a stale
            # partial — delete and download fresh.
            if _md5_of(dest) == entry.md5sum:
                self._check_capacity()
                self._resident.add(dest)
                return dest
            dest.unlink()

        self._check_capacity()

        try:
            self._client.download(entry.filename, dest)
        except BaseException:
            dest.unlink(missing_ok=True)  # never leave partials behind
            raise

        if _md5_of(dest) != entry.md5sum:
            # The failed transfer still billed real egress: record it under
            # the corrupt counters (never the verified ones).  Measure the
            # actual bytes on disk before deleting the bad file.
            try:
                transferred = dest.stat().st_size
            except OSError:
                transferred = entry.size_bytes
            self.corrupt_downloads += 1
            self.corrupt_bytes += transferred
            dest.unlink(missing_ok=True)
            raise ChecksumError(
                f"md5 mismatch for {entry.filename}: downloaded file does "
                f"not match manifest md5sum {entry.md5sum}; corrupt file "
                "deleted, egress recorded under corrupt_downloads/"
                "corrupt_bytes only"
            )

        self._resident.add(dest)
        self.chunk_downloads += 1
        self.chunk_bytes += entry.size_bytes
        return dest

    # --- extract ----------------------------------------------------------------

    def extract_matching(
        self, chunk_path: Path, wanted_ids: set[str]
    ) -> Iterator[tuple[str, bytes]]:
        """Stream tar members, yielding ``(arxiv_id, raw bytes)`` for wanted ids.

        Members are read one at a time into memory — nothing is ever
        extracted to disk.  Member bytes are OPAQUE (gz vs tar.gz sniffing
        stays downstream).  Member names that do not map to an arXiv id
        (see :func:`_member_arxiv_id`) are skipped silently.  Stops early
        once every wanted id has been yielded.
        """
        if not wanted_ids:
            return
        found: set[str] = set()
        with tarfile.open(chunk_path, mode="r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                arxiv_id = _member_arxiv_id(member.name)
                if arxiv_id is None or arxiv_id not in wanted_ids:
                    continue
                handle = tar.extractfile(member)
                if handle is None:  # unreadable special member — skip
                    continue
                with handle:
                    payload = handle.read()
                yield arxiv_id, payload
                found.add(arxiv_id)
                if found >= wanted_ids:
                    return

    # --- purge / release --------------------------------------------------------

    def purge(self, chunk_path: Path) -> None:
        """Delete the chunk file and free its residency slot (idempotent)."""
        path = Path(chunk_path)
        path.unlink(missing_ok=True)
        self._resident.discard(path)

    def release(self, entry_or_path: Union["ManifestEntry", Path, str]) -> None:
        """Purge a chunk by manifest entry OR by path (extraction is §4's job).

        Accepts a manifest entry (anything with a ``.filename`` attribute)
        or a path-like; afterwards the chunk file is gone from disk.
        """
        filename = getattr(entry_or_path, "filename", None)
        if filename is not None:
            self.purge(self._local_path_for(filename))
        else:
            self.purge(Path(entry_or_path))

    # --- internals ----------------------------------------------------------------

    def _local_path_for(self, key: str) -> Path:
        """Local resting place for an S3 key: work_dir / basename(key)."""
        return self._work_dir / PurePosixPath(key).name

    def _check_capacity(self) -> None:
        if len(self._resident) >= self._max_resident:
            raise RuntimeError(
                f"chunk residency limit reached ({self._max_resident} "
                f"resident in {self._work_dir}); purge()/release() a chunk "
                "before fetching another (invariant: ≤ max_resident chunks "
                "on disk)"
            )


def _md5_of(path: Path) -> str:
    """Hex md5 of a file, streamed in 1 MiB blocks (chunks are ~500 MB)."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

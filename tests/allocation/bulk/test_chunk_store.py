"""ChunkStore unit tests — offline only, provider seam faked, tars synthetic.

No committed binary fixtures: every tarball is built at test time into
``tmp_path`` via the ``tarfile`` module.  Manifest entries are local
stand-ins (duck-typed per the chunk_store docstring) so these tests do not
depend on ``bulk/manifest.py``.  boto3 is never imported for real: the
factory tests inject fakes into ``sys.modules``.
"""

from __future__ import annotations

import hashlib
import io
import socket
import sys
import tarfile
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from icepick.allocation.bulk import chunk_store
from icepick.allocation.bulk.chunk_store import ChecksumError, ChunkStore, s3_client


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a chunk store test")

    monkeypatch.setattr(socket, "socket", _blocked)


# --- stand-ins (duck-typed ManifestEntry + scripted provider) -----------------


@dataclass(frozen=True)
class _Entry:
    """Local stand-in exposing the attributes chunk_store duck-types on."""

    filename: str
    size_bytes: int
    md5sum: str
    first_item: str = "2501.00001"
    last_item: str = "2501.99999"


def _entry_for(payload: bytes, filename: str = "src/arXiv_src_2501_001.tar") -> _Entry:
    return _Entry(
        filename=filename,
        size_bytes=len(payload),
        md5sum=hashlib.md5(payload).hexdigest(),
    )


class _FakeClient:
    """Provider seam fake: writes prepared bytes per key; records calls."""

    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.calls: list[str] = []

    def download(self, key: str, dest: Path) -> None:
        self.calls.append(key)
        Path(dest).write_bytes(self.payloads[key])


def _store(tmp_path: Path, payloads: dict[str, bytes], **kwargs) -> tuple[ChunkStore, _FakeClient]:
    client = _FakeClient(payloads)
    return ChunkStore(client, work_dir=tmp_path / "work", **kwargs), client


def _make_tar(path: Path, members: dict[str, bytes], with_dir: bool = False) -> None:
    """Build a synthetic chunk tarball with the given member-name→bytes map."""
    with tarfile.open(path, "w") as tar:
        if with_dir:
            info = tarfile.TarInfo(name="2501/")
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


# --- fetch: verify, counters, cleanup ------------------------------------------


def test_fetch_happy_path_downloads_verifies_and_counts(tmp_path):
    payload = b"tar bytes for chunk 001"
    entry = _entry_for(payload)
    store, client = _store(tmp_path, {entry.filename: payload})

    path = store.fetch(entry)

    assert path == tmp_path / "work" / "arXiv_src_2501_001.tar"
    assert path.read_bytes() == payload
    assert client.calls == [entry.filename]
    assert store.chunk_downloads == 1
    assert store.chunk_bytes == entry.size_bytes
    assert store.corrupt_downloads == 0  # clean transfer: nothing corrupt
    assert store.corrupt_bytes == 0


def test_fetch_checksum_mismatch_deletes_file_and_counts_only_corrupt_egress(tmp_path):
    good = b"the bytes the manifest promises"
    entry = _entry_for(good)
    # Scripted corruption: provider delivers different bytes than promised.
    corrupt_payload = b"corrupted transfer!!"
    store, _ = _store(tmp_path, {entry.filename: corrupt_payload})

    with pytest.raises(ChecksumError, match="md5 mismatch"):
        store.fetch(entry)

    assert not (tmp_path / "work" / "arXiv_src_2501_001.tar").exists()
    assert store.chunk_downloads == 0  # verified counters untouched
    assert store.chunk_bytes == 0
    assert store.corrupt_downloads == 1  # but the billed egress is recorded
    assert store.corrupt_bytes == len(corrupt_payload)  # actual bytes on disk


def test_counters_accumulate_across_distinct_fetches(tmp_path):
    p1, p2 = b"chunk one bytes", b"chunk two bytes, longer"
    e1 = _entry_for(p1, "src/arXiv_src_2501_001.tar")
    e2 = _entry_for(p2, "src/arXiv_src_2501_002.tar")
    store, _ = _store(tmp_path, {e1.filename: p1, e2.filename: p2})

    store.fetch(e1)
    store.fetch(e2)

    assert store.chunk_downloads == 2
    assert store.chunk_bytes == e1.size_bytes + e2.size_bytes


# --- fetch: resume / no-op semantics --------------------------------------------


def test_refetch_of_resident_chunk_is_noop_without_counter_increment(tmp_path):
    payload = b"chunk payload"
    entry = _entry_for(payload)
    store, client = _store(tmp_path, {entry.filename: payload})

    first = store.fetch(entry)
    second = store.fetch(entry)

    assert first == second
    assert client.calls == [entry.filename]  # exactly one download
    assert store.chunk_downloads == 1
    assert store.chunk_bytes == entry.size_bytes


def test_fetch_adopts_preexisting_verified_file_without_download(tmp_path):
    payload = b"left behind by a previous, killed run"
    entry = _entry_for(payload)
    work = tmp_path / "work"
    work.mkdir()
    (work / "arXiv_src_2501_001.tar").write_bytes(payload)

    store, client = _store(tmp_path, {entry.filename: payload})
    path = store.fetch(entry)

    assert path.read_bytes() == payload
    assert client.calls == []  # resume: never re-downloaded
    assert store.chunk_downloads == 0
    assert store.chunk_bytes == 0


def test_fetch_replaces_preexisting_corrupt_file(tmp_path):
    payload = b"full, correct chunk bytes"
    entry = _entry_for(payload)
    work = tmp_path / "work"
    work.mkdir()
    (work / "arXiv_src_2501_001.tar").write_bytes(b"partial junk from a kill")

    store, client = _store(tmp_path, {entry.filename: payload})
    path = store.fetch(entry)

    assert path.read_bytes() == payload  # stale partial replaced
    assert client.calls == [entry.filename]
    assert store.chunk_downloads == 1


# --- residency cap ----------------------------------------------------------------


def _three_entries_store(tmp_path):
    payloads = {
        f"src/arXiv_src_2501_00{i}.tar": f"chunk {i} bytes".encode() for i in (1, 2, 3)
    }
    entries = [_entry_for(data, key) for key, data in payloads.items()]
    store, client = _store(tmp_path, payloads, max_resident=2)
    return store, client, entries


def test_third_fetch_beyond_max_resident_raises(tmp_path):
    store, _, (e1, e2, e3) = _three_entries_store(tmp_path)
    store.fetch(e1)
    store.fetch(e2)  # 2 resident: fine

    with pytest.raises(RuntimeError, match="residency limit"):
        store.fetch(e3)

    assert store.chunk_downloads == 2  # the refused fetch counted nothing
    assert not (tmp_path / "work" / "arXiv_src_2501_003.tar").exists()


def test_purge_frees_a_residency_slot(tmp_path):
    store, _, (e1, e2, e3) = _three_entries_store(tmp_path)
    p1 = store.fetch(e1)
    store.fetch(e2)

    store.purge(p1)
    p3 = store.fetch(e3)  # slot freed → allowed again

    assert p3.exists()
    assert store.chunk_downloads == 3


def test_refetch_of_resident_chunk_at_cap_is_still_noop(tmp_path):
    store, client, (e1, e2, _) = _three_entries_store(tmp_path)
    p1 = store.fetch(e1)
    store.fetch(e2)

    assert store.fetch(e1) == p1  # not a capacity violation
    assert client.calls == [e1.filename, e2.filename]


# --- extract_matching ---------------------------------------------------------------


_PAPERS = {
    "2501/2501.00123.gz": b"gz-opaque bytes of paper 00123",
    "2501/2501.00456.gz": b"gz-opaque bytes of paper 00456",
    "2501/2501.00789.gz": b"gz-opaque bytes of paper 00789",
}


def test_extract_matching_yields_only_wanted_ids_with_raw_bytes(tmp_path):
    chunk = tmp_path / "chunk.tar"
    _make_tar(chunk, _PAPERS)
    store, _ = _store(tmp_path, {})

    got = dict(store.extract_matching(chunk, {"2501.00123", "2501.00789"}))

    assert got == {
        "2501.00123": _PAPERS["2501/2501.00123.gz"],
        "2501.00789": _PAPERS["2501/2501.00789.gz"],
    }


def test_extract_matching_skips_weird_member_names_not_fatal(tmp_path):
    members = dict(_PAPERS)
    members.update(
        {
            "README": b"top-level stray file",
            "2501/notes.txt": b"not a paper",
            "2501/2501.badid.gz": b"unparseable stem",
            "9999/withdrawn": b"no extension at all",
        }
    )
    chunk = tmp_path / "chunk.tar"
    _make_tar(chunk, members, with_dir=True)  # plus a directory member
    store, _ = _store(tmp_path, {})

    got = dict(store.extract_matching(chunk, {"2501.00456", "2501.00123"}))

    assert got == {
        "2501.00123": _PAPERS["2501/2501.00123.gz"],
        "2501.00456": _PAPERS["2501/2501.00456.gz"],
    }


def test_extract_matching_streams_lazily_and_writes_nothing_to_disk(tmp_path):
    chunk = tmp_path / "chunk.tar"
    _make_tar(chunk, _PAPERS)
    store, _ = _store(tmp_path, {})
    before = {p for p in tmp_path.rglob("*")}

    it = store.extract_matching(chunk, {"2501.00123", "2501.00456"})
    assert iter(it) is it  # a lazy iterator, not a materialised collection

    first = next(it)
    assert first[0] in {"2501.00123", "2501.00456"}
    list(it)  # drain

    assert {p for p in tmp_path.rglob("*")} == before  # nothing extracted to disk


def test_extract_matching_empty_wanted_set_yields_nothing(tmp_path):
    chunk = tmp_path / "chunk.tar"
    _make_tar(chunk, _PAPERS)
    store, _ = _store(tmp_path, {})

    assert list(store.extract_matching(chunk, set())) == []


# --- purge / release -----------------------------------------------------------------


def test_purge_removes_file_and_is_idempotent(tmp_path):
    payload = b"chunk to purge"
    entry = _entry_for(payload)
    store, _ = _store(tmp_path, {entry.filename: payload})
    path = store.fetch(entry)

    store.purge(path)
    assert not path.exists()
    store.purge(path)  # second purge: no error


def test_release_accepts_a_manifest_entry(tmp_path):
    payload = b"chunk to release by entry"
    entry = _entry_for(payload)
    store, _ = _store(tmp_path, {entry.filename: payload})
    path = store.fetch(entry)

    store.release(entry)

    assert not path.exists()


def test_release_accepts_a_path(tmp_path):
    payload = b"chunk to release by path"
    entry = _entry_for(payload)
    store, _ = _store(tmp_path, {entry.filename: payload})
    path = store.fetch(entry)

    store.release(path)

    assert not path.exists()


# --- s3_client factory (never live; fakes injected via sys.modules) -------------------


def test_s3_client_raises_clear_error_when_boto3_absent(monkeypatch):
    # None in sys.modules makes `import boto3` fail even if it IS installed —
    # this also proves the import is lazy (module import already succeeded).
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(ImportError, match=r"boto3.*icepick\[bulk\]"):
        s3_client()


def test_s3_client_wires_requester_pays_and_standard_chain(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}
    downloads: list[dict[str, object]] = []

    class _FakeS3:
        def download_file(self, *, Bucket, Key, Filename, ExtraArgs=None):
            downloads.append(
                {"Bucket": Bucket, "Key": Key, "Filename": Filename, "ExtraArgs": ExtraArgs}
            )

    def _client(service, region_name=None):
        recorded["service"] = service
        recorded["region"] = region_name
        return _FakeS3()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = _client  # creds args absent → standard AWS chain only
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    client = s3_client(region="eu-west-1")
    dest = tmp_path / "arXiv_src_2501_001.tar"
    client.download("src/arXiv_src_2501_001.tar", dest)

    assert recorded == {"service": "s3", "region": "eu-west-1"}
    assert downloads == [
        {
            "Bucket": "arxiv",
            "Key": "src/arXiv_src_2501_001.tar",
            "Filename": str(dest),
            "ExtraArgs": {"RequestPayer": "requester"},
        }
    ]


def test_member_name_mapping_isolated_helper():
    assert chunk_store._member_arxiv_id("2501/2501.00123.gz") == "2501.00123"
    assert chunk_store._member_arxiv_id("2501/2501.123456.gz") == "2501.123456"
    for weird in ("README", "2501/notes.txt", "2501/2501.00123", "2501/", "x.gz"):
        assert chunk_store._member_arxiv_id(weird) is None

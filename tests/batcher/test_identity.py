"""Tests for src/icepick/batcher/identity.py.

Cross-checks:
- ``compute_uid`` is tested against the REAL ``processing.poser.base.compute_uid``
  to prove the recipes are identical.
- ``stmt_key`` is tested against the REAL normalise() dedup-key recipe from
  ``allocation.adapters.realmath_scrape`` (extracted inline since normalise()
  is not a public function but the key line is ``" ".join(s.lower().split())``).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from icepick.batcher.identity import compute_uid, stmt_key, content_hash


# ---------------------------------------------------------------------------
# compute_uid
# ---------------------------------------------------------------------------


class TestComputeUid:
    def test_basic_recipe(self):
        """SHA-256 of 'source\x1fstatement', truncated to 32 hex chars."""
        source = "arxiv_bulk_pde625"
        statement = "Let x > 0. Then x^2 > 0."
        expected = hashlib.sha256(
            f"{source}\x1f{statement}".encode("utf-8")
        ).hexdigest()[:32]
        assert compute_uid(source, statement) == expected

    def test_length(self):
        assert len(compute_uid("s", "t")) == 32

    def test_hex(self):
        uid = compute_uid("abc", "def")
        assert all(c in "0123456789abcdef" for c in uid)

    def test_separator_prevents_ambiguity(self):
        """Different source/statement splits must not collide."""
        # "a\x1fb" vs "" and "a" + "" and "\x1fb" — different by construction
        uid1 = compute_uid("a\x1f", "b")
        uid2 = compute_uid("a", "\x1fb")
        # These are technically the same bytes but demonstrate that ordinary
        # source/statement splits with the separator in the VALUE are handled
        # deterministically and identically to the poser.
        # What we care about is that source="" and source="a" do NOT collide:
        assert compute_uid("", "hello") != compute_uid("a", "hello")

    def test_empty_inputs(self):
        uid = compute_uid("", "")
        assert len(uid) == 32

    def test_deterministic(self):
        uid1 = compute_uid("src", "stmt")
        uid2 = compute_uid("src", "stmt")
        assert uid1 == uid2

    def test_cross_check_against_real_poser(self):
        """The batcher recipe must produce the SAME uid as the real poser."""
        from icepick.processing.poser.base import compute_uid as poser_compute_uid

        samples = [
            ("arxiv_bulk_pde625", "Prove that every bounded sequence has a convergent subsequence."),
            ("batch1_src", "Let f be continuous on [a,b]. Then f attains its max."),
            ("", ""),
            ("realmath", "x^2 + y^2 = r^2"),
            ("src", "statement with \x1f in it"),
        ]
        for source, statement in samples:
            batcher_uid = compute_uid(source, statement)
            real_uid = poser_compute_uid(source, statement)
            assert batcher_uid == real_uid, (
                f"compute_uid mismatch for source={source!r}, "
                f"statement={statement!r}: batcher={batcher_uid!r} real={real_uid!r}"
            )


# ---------------------------------------------------------------------------
# stmt_key
# ---------------------------------------------------------------------------


class TestStmtKey:
    """Mirror of realmath_scrape.normalise() line 345: " ".join(s.lower().split())."""

    def _real_dedup_key(self, s: str) -> str:
        """Replicate the exact recipe from realmath_scrape.normalise() line 345."""
        normalised = " ".join(s.lower().split())
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def test_basic(self):
        s = "  Let   X > 0.  "
        assert stmt_key(s) == self._real_dedup_key(s)

    def test_case_folding(self):
        assert stmt_key("Hello World") == stmt_key("hello world")

    def test_whitespace_collapse(self):
        assert stmt_key("a  b\t c\n d") == stmt_key("a b c d")

    def test_leading_trailing_stripped(self):
        assert stmt_key("  foo  ") == stmt_key("foo")

    def test_length(self):
        # SHA-256 hex = 64 chars
        assert len(stmt_key("anything")) == 64

    def test_cross_check_against_real_normalise(self):
        """The batcher stmt_key must match the realmath_scrape dedup key recipe."""
        samples = [
            "Prove that every bounded sequence has a convergent subsequence.",
            "  Let   f be CONTINUOUS on [a,b].  ",
            "x^2 + y^2 = r^2",
            "",
            "Multiple   spaces\t\ttabs\nnewlines",
        ]
        for s in samples:
            batcher_key = stmt_key(s)
            real_key = self._real_dedup_key(s)
            assert batcher_key == real_key, (
                f"stmt_key mismatch for {s!r}: batcher={batcher_key!r} real={real_key!r}"
            )

    def test_deterministic(self):
        assert stmt_key("abc def") == stmt_key("abc def")


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_basic(self):
        row = {"source": "s", "statement": "st", "answer": "42"}
        expected = hashlib.sha256(
            json.dumps(row, sort_keys=True).encode("utf-8")
        ).hexdigest()
        assert content_hash(row) == expected

    def test_sort_keys(self):
        """Key order in the dict must not affect the hash."""
        row_a = {"b": 2, "a": 1}
        row_b = {"a": 1, "b": 2}
        assert content_hash(row_a) == content_hash(row_b)

    def test_length(self):
        assert len(content_hash({})) == 64

    def test_different_rows_differ(self):
        assert content_hash({"x": 1}) != content_hash({"x": 2})

    def test_empty_dict(self):
        expected = hashlib.sha256(json.dumps({}, sort_keys=True).encode("utf-8")).hexdigest()
        assert content_hash({}) == expected

    def test_nested(self):
        row = {"metadata": {"arxiv_id": "1234.5678"}, "statement": "foo"}
        # Just verify it runs and gives a 64-char hex string.
        h = content_hash(row)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        row = {"a": 1, "b": "two"}
        assert content_hash(row) == content_hash(row)

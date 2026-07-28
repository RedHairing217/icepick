"""Tests for loratrain.build_dataset.assert_verified_correct."""

from __future__ import annotations

import pytest

from loratrain.build_dataset import TraceIntegrityError, assert_verified_correct


def _example(**provenance_overrides):
    provenance = {
        "uid": "u1",
        "rollout_uid": "r1",
        "arxiv_id": "1000.00001",
        "verdict": "correct",
        "verbatim_output": True,
    }
    provenance.update(provenance_overrides)
    return {"provenance": provenance}


def test_good_example_passes():
    assert assert_verified_correct(_example()) is None


def test_verdict_wrong_raises():
    with pytest.raises(TraceIntegrityError):
        assert_verified_correct(_example(verdict="wrong"))


def test_verdict_degenerate_raises():
    with pytest.raises(TraceIntegrityError):
        assert_verified_correct(_example(verdict="degenerate"))


def test_verbatim_output_false_raises():
    with pytest.raises(TraceIntegrityError):
        assert_verified_correct(_example(verbatim_output=False))


def test_verbatim_output_missing_raises():
    example = _example()
    del example["provenance"]["verbatim_output"]
    with pytest.raises(TraceIntegrityError):
        assert_verified_correct(example)


def test_missing_rollout_uid_raises():
    example = _example()
    del example["provenance"]["rollout_uid"]
    with pytest.raises(TraceIntegrityError):
        assert_verified_correct(example)

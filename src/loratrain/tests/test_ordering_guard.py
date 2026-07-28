"""Tests for loratrain.train_lora's ordering guard and client-shape stubs."""

from __future__ import annotations

import hashlib

import pytest

from loratrain import config
from loratrain.train_lora import (
    ENDPOINT_TRAIN,
    OrderingError,
    assert_baseline_captured,
    build_job_payload,
    submit_job,
)


def test_missing_baseline_raises_ordering_error(tmp_path):
    missing = tmp_path / "baseline_greedy.jsonl"
    with pytest.raises(OrderingError, match="baseline"):
        assert_baseline_captured(missing)


def test_empty_baseline_raises_ordering_error(tmp_path):
    empty = tmp_path / "baseline_greedy.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(OrderingError):
        assert_baseline_captured(empty)


def test_whitespace_only_baseline_raises_ordering_error(tmp_path):
    whitespace_only = tmp_path / "baseline_greedy.jsonl"
    whitespace_only.write_text("   \n\n   \n", encoding="utf-8")
    with pytest.raises(OrderingError):
        assert_baseline_captured(whitespace_only)


def test_nonempty_baseline_returns_its_sha256(tmp_path):
    baseline = tmp_path / "baseline_greedy.jsonl"
    baseline.write_text('{"uid": "a", "output": "42"}\n', encoding="utf-8")
    expected = hashlib.sha256(baseline.read_bytes()).hexdigest()
    assert assert_baseline_captured(baseline) == expected


def test_build_job_payload_embeds_sha_and_hyperparams(tmp_path):
    dataset_path = tmp_path / "sft_train.jsonl"
    baseline_sha256 = "deadbeef" * 8
    payload = build_job_payload(dataset_path, baseline_sha256)

    assert payload["baseline_greedy_sha256"] == baseline_sha256
    assert payload["dataset_file"] == str(dataset_path)
    assert payload["base_model"] == config.BASE_MODEL_HF_ID
    assert payload["adapter_format"] == config.ADAPTER_FORMAT
    assert payload["seed"] == config.SEED
    assert payload["hyperparams"] == {
        "rank": config.LORA_RANK,
        "alpha": config.LORA_ALPHA,
        "dropout": config.LORA_DROPOUT,
        "lr": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "micro_batch_size": config.MICRO_BATCH_SIZE,
        "max_seq_len": config.MAX_SEQ_LEN,
    }


def test_submit_job_raises_notimplemented_with_derived_url():
    with pytest.raises(NotImplementedError) as excinfo:
        submit_job({"dummy": "payload"})
    assert config.TRAIN_SERVER_URL in str(excinfo.value)
    assert ENDPOINT_TRAIN in str(excinfo.value)

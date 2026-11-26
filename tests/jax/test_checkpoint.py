"""Round-trip the checkpoint format through the TPU run publisher."""

from pathlib import Path

import numpy as np
import pytest

from hamic_jax.checkpoint import (
    PARAMS_NAME,
    CheckpointError,
    CheckpointPublisher,
    CheckpointResolver,
    LocalObjectStore,
    load_checkpoint,
    save_checkpoint,
)
from hamic_jax.data import DataCursor
from hamic_jax.runs import write_run_manifest


def test_publishing_preserves_the_local_checkpoint(tmp_path: Path) -> None:
    local = tmp_path / "local"
    restored = tmp_path / "restored"
    metadata = {"step": 3, "backend": "jax"}
    cursor = DataCursor(7, 24, 1, 2)
    params = {"weight": np.arange(4, dtype=np.float32)}
    save_checkpoint(local, params, (), 3, cursor, metadata)

    store = LocalObjectStore(tmp_path / "store")
    prefix = "runs/round-trip"
    CheckpointPublisher(store, prefix).publish(local, metadata)
    resolver = CheckpointResolver(store, prefix)
    assert resolver.committed()[0].step == 3
    assert resolver.restore_latest(restored) is not None
    for directory in (local, restored):
        loaded, _, step, saved_cursor, saved_metadata = load_checkpoint(directory)
        np.testing.assert_array_equal(loaded["weight"], params["weight"])
        assert (step, saved_cursor, saved_metadata) == (3, cursor, metadata)
    with pytest.raises(CheckpointError, match="already committed"):
        CheckpointPublisher(store, prefix).publish(local, metadata)


def test_resume_does_not_start_over_when_only_checkpoint_is_corrupt(tmp_path: Path) -> None:
    local = tmp_path / "local"
    metadata = {"step": 3}
    save_checkpoint(local, {"weight": np.zeros(1)}, (), 3, DataCursor(), metadata)
    store = LocalObjectStore(tmp_path / "store")
    prefix = "runs/round-trip"
    checkpoint = CheckpointPublisher(store, prefix).publish(local, metadata)
    store.put_bytes(f"{checkpoint.prefix}/{PARAMS_NAME}", b"truncated", "simulate damage")
    with pytest.raises(CheckpointError, match="No valid committed checkpoint"):
        CheckpointResolver(store, prefix).restore_latest(tmp_path / "restored")


def test_run_manifest_refuses_a_different_tokenized_corpus(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    manifest = {"run_id": "round-trip", "backend": "jax", "identity": {}, "corpus_sha256": "a"}
    write_run_manifest(store, manifest, "create run")
    with pytest.raises(RuntimeError, match="different tokenized corpus"):
        write_run_manifest(store, {**manifest, "corpus_sha256": "b"}, "resume run")

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
from hamic_jax.data import (
    MANIFEST_NAME,
    DataCursor,
    TokenizedCorpus,
    WindowedPackedBatches,
    fetch_corpus_metadata,
)
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


def test_windowed_batches_resume_and_evict_across_windows(tmp_path: Path) -> None:
    import hashlib
    import json

    remote_root = tmp_path / "remote"
    prefix = "tokenized/test-win/tok"
    src_dir = remote_root / prefix
    (src_dir / "train").mkdir(parents=True)
    shards_meta = []
    for idx in range(3):
        tokens = np.arange(10 + idx * 10, 18 + idx * 10, dtype=np.uint16)
        offsets = np.array([0, 4, 8], dtype=np.int64)
        t_path = src_dir / "train" / f"train-{idx:05d}.tokens.npy"
        o_path = src_dir / "train" / f"train-{idx:05d}.offsets.npy"
        np.save(t_path, tokens)
        np.save(o_path, offsets)
        shards_meta.append({
            "tokens": int(tokens.size),
            "documents": 2,
            "files": {
                "tokens": {
                    "path": t_path.name,
                    "sha256": hashlib.sha256(t_path.read_bytes()).hexdigest(),
                },
                "offsets": {
                    "path": o_path.name,
                    "sha256": hashlib.sha256(o_path.read_bytes()).hexdigest(),
                },
            },
        })
    (src_dir / MANIFEST_NAME).write_text(
        json.dumps({
            "format": "hamic-tokenized-npy-v1",
            "splits": {"train": {"shards": shards_meta}},
        }),
        encoding="utf-8",
    )

    store = LocalObjectStore(remote_root)
    local_dir = tmp_path / "local"
    corpus, fetcher = fetch_corpus_metadata(store, prefix, local_dir)
    assert not (local_dir / "train" / "train-00000.tokens.npy").exists()

    batches = WindowedPackedBatches(
        corpus, "train", seq_len=4, batch_size=2, bos_id=0, eos_id=2,
        shard_window=1, fetcher=fetcher, seed=7,
    )
    b0 = batches.next_batch()
    b1 = batches.next_batch()
    cursor = batches.cursor()
    b2 = batches.next_batch()
    b3 = batches.next_batch()
    b4 = batches.next_batch()
    assert batches.has_completed_window
    evicted = batches.evict_completed()
    assert evicted
    for idx in evicted:
        assert not (local_dir / "train" / f"train-{idx:05d}.tokens.npy").exists()

    resumed_dir = tmp_path / "resumed"
    r_corpus, r_fetcher = fetch_corpus_metadata(store, prefix, resumed_dir)
    resumed = WindowedPackedBatches(
        r_corpus, "train", seq_len=4, batch_size=2, bos_id=0, eos_id=2,
        shard_window=1, fetcher=r_fetcher, seed=7, cursor=cursor,
    )
    np.testing.assert_array_equal(resumed.next_batch(), b2)
    np.testing.assert_array_equal(resumed.next_batch(), b3)
    np.testing.assert_array_equal(resumed.next_batch(), b4)
    assert b0.shape == (2, 4) and b1.shape == (2, 4)


def test_checkpoint_publisher_prunes_oldest_remote_checkpoints(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "store")
    prefix = "runs/prune-test"
    publisher = CheckpointPublisher(store, prefix)
    for step in (1, 2, 3):
        local = tmp_path / f"local-{step}"
        metadata = {"step": step, "backend": "jax"}
        save_checkpoint(local, {"weight": np.full(2, step, dtype=np.float32)}, (), step, DataCursor(step), metadata)
        publisher.publish(local, metadata)
    pruned = publisher.prune(1)
    assert pruned == [1, 2]
    resolver = CheckpointResolver(store, prefix)
    assert [item.step for item in resolver.committed()] == [3]
    restored = tmp_path / "restored"
    assert resolver.restore_latest(restored) is not None
    loaded, _, step, _, _ = load_checkpoint(restored)
    assert step == 3
    np.testing.assert_array_equal(loaded["weight"], np.full(2, 3, dtype=np.float32))


def test_evict_shards_purges_store_cache(tmp_path: Path) -> None:
    from unittest.mock import MagicMock
    from hamic_jax.data import ShardFetcher

    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "train").mkdir(parents=True)
    tok = corpus_dir / "train" / "train-00000.tokens.npy"
    tok.write_bytes(b"abc")
    (corpus_dir / MANIFEST_NAME).write_text(
        '{"format": "hamic-tokenized-npy-v1", "splits": {"train": {"shards": [{"tokens": 1, "documents": 1, "files": {"tokens": {"path": "train-00000.tokens.npy", "sha256": "unused"}}}]}}}',
        encoding="utf-8",
    )
    corpus = TokenizedCorpus(corpus_dir)
    store = MagicMock()
    fetcher = ShardFetcher(store, "tokenized/v/c", corpus)
    fetcher.evict_shards("train", [0])
    assert not tok.exists()
    store.purge_cache.assert_called_once_with()

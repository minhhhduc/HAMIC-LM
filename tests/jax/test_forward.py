"""Tiny CPU forward checks for both families, independent of torch."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax")

from hamic_jax.data import MANIFEST_NAME, TokenizedCorpus  # noqa: E402
from hamic_jax.model import encoder_hidden, forward, loss as model_loss  # noqa: E402
from hamic_jax.params import init_params  # noqa: E402
from hamic_jax.sharding import DeviceLayout  # noqa: E402
from hamic_jax.spec import ModelPreset, TrainingConfig  # noqa: E402
from hamic_jax.train import Trainer  # noqa: E402


def _encoder() -> ModelPreset:
    return ModelPreset(
        name="tiny-encoder",
        architecture="modernbert",
        num_layers=2,
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        local_attention=128,
        sliding_window=64,
    )


def _decoder() -> ModelPreset:
    return ModelPreset(
        name="tiny-decoder",
        architecture="llama",
        num_layers=2,
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        norm_eps=1e-6,
        local_attention=None,
        sliding_window=None,
    )


@pytest.mark.parametrize("preset_factory", [_encoder, _decoder])
def test_forward_shapes_and_loss(preset_factory: Callable[[], ModelPreset]) -> None:
    preset = preset_factory()
    params = init_params(preset, jax.random.PRNGKey(0))
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 128, (2, 8)), jnp.int32)
    labels = ids.at[:, :2].set(-100)
    output = forward(params, preset, ids, labels)
    assert output.logits.shape == (2, 8, 128)
    assert output.hidden.shape == (2, 8, 32)
    assert output.loss is not None
    assert np.isfinite(np.asarray(output.loss))


def test_encoder_hidden_matches_forward_prefix() -> None:
    preset = _encoder()
    params = init_params(preset, jax.random.PRNGKey(1))
    ids = jnp.arange(8, dtype=jnp.int32)[None, :]
    hidden = encoder_hidden(params, preset, ids)
    assert hidden.shape == (1, 8, 32)
    assert forward(params, preset, ids).hidden.shape == (1, 8, 32)


@pytest.mark.parametrize("preset_factory", [_encoder, _decoder])
def test_recompute_matches_plain_forward_and_grad(
    preset_factory: Callable[[], ModelPreset],
) -> None:
    """Rematerialisation must not change the forward value or the parameter gradient."""
    preset = preset_factory()
    params = init_params(preset, jax.random.PRNGKey(2))
    ids = jnp.asarray(np.random.default_rng(1).integers(0, 128, (2, 8)), jnp.int32)
    labels = ids.at[:, :2].set(-100)

    plain = forward(params, preset, ids, labels, recompute=False)
    remat = forward(params, preset, ids, labels, recompute=True)
    np.testing.assert_allclose(np.asarray(plain.logits), np.asarray(remat.logits))
    np.testing.assert_allclose(np.asarray(plain.loss), np.asarray(remat.loss))

    def gradient(recompute: bool) -> dict[str, jnp.ndarray]:
        return jax.grad(
            lambda tree: forward(tree, preset, ids, labels, recompute=recompute).loss
        )(params)

    plain_grad = gradient(False)
    remat_grad = gradient(True)
    for name in plain_grad:
        np.testing.assert_allclose(
            np.asarray(plain_grad[name]),
            np.asarray(remat_grad[name]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_sharded_mlm_step_matches_a_full_batch(tmp_path: Path, monkeypatch) -> None:
    devices = jax.local_devices()
    if len(devices) < 2:
        pytest.skip("requires two local JAX devices")

    tokens = np.arange(40, 56, dtype=np.uint16)
    (tmp_path / "train").mkdir()
    np.save(tmp_path / "train" / "tokens.npy", tokens)
    np.save(tmp_path / "train" / "offsets.npy", [0, len(tokens)])
    files = {name: {"path": f"{name}.npy"} for name in ("tokens", "offsets")}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({
        "format": "hamic-tokenized-npy-v1",
        "splits": {"train": {"shards": [{"files": files}]}},
    }))
    corpus = TokenizedCorpus(tmp_path)
    config = TrainingConfig(
        "mesh-test", "tiny-encoder", str(tmp_path), "mlm", 4, 2, 2, 32, 1e-3,
        precision="fp32",
    )
    ids = np.array([[[40, 41, 42, 43], [44, 45, 46, 47]]] * 2, dtype=np.int32)
    labels = np.array([
        [[40, -100, -100, -100], [44, 45, 46, -100]],
        [[-100, -100, -100, -100], [44, 45, -100, -100]],
    ], dtype=np.int32)

    choose_layout = DeviceLayout.for_batch

    def update(device_count):
        monkeypatch.setattr(
            DeviceLayout, "for_batch", lambda batch: choose_layout(batch, devices[:device_count])
        )
        trainer = Trainer(config, _encoder(), corpus)
        monkeypatch.setattr(trainer, "_micro_batches", lambda: (ids, labels))
        metrics = trainer.train_step_metrics()
        return (*metrics, *jax.tree.leaves(trainer.params))

    single = update(1)
    sharded = update(2)
    for expected, actual in zip(single, sharded, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sparse_capacity", [6, 2])
def test_sparse_mlm_loss_matches_dense_value_and_grad(sparse_capacity: int) -> None:
    preset = _encoder()
    params = init_params(preset, jax.random.PRNGKey(3))
    ids = jnp.asarray(np.random.default_rng(2).integers(0, 128, (2, 8)), jnp.int32)
    labels = jnp.full((2, 8), -100, dtype=jnp.int32).at[0, :3].set(ids[0, :3]).at[1, 1].set(ids[1, 1])

    dense_val, dense_grad = jax.value_and_grad(
        lambda tree: model_loss(tree, preset, ids, labels, sparse_capacity=None)
    )(params)
    sparse_val, sparse_grad = jax.value_and_grad(
        lambda tree: model_loss(tree, preset, ids, labels, sparse_capacity=sparse_capacity)
    )(params)

    np.testing.assert_allclose(np.asarray(sparse_val), np.asarray(dense_val), rtol=1e-5, atol=1e-6)
    for name in dense_grad:
        np.testing.assert_allclose(
            np.asarray(sparse_grad[name]),
            np.asarray(dense_grad[name]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_modernbert_init_uses_truncated_normal_and_output_scaling() -> None:
    import math

    preset = _encoder()
    params = init_params(preset, jax.random.PRNGKey(4))
    std_in = 0.02
    std_out = 0.02 / math.sqrt(2.0 * preset.num_layers)
    cutoff = 2.0
    assert float(jnp.max(jnp.abs(params["model.layers.0.attn.Wqkv.weight"]))) <= cutoff * std_in + 1e-6
    assert float(jnp.max(jnp.abs(params["model.layers.0.attn.Wo.weight"]))) <= cutoff * std_out + 1e-6
    assert float(jnp.std(params["model.layers.0.attn.Wo.weight"])) < float(
        jnp.std(params["model.layers.0.attn.Wqkv.weight"])
    )

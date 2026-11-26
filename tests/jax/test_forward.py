"""Tiny CPU forward checks for both families, independent of torch."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax")

from hamic_jax.model import encoder_hidden, forward  # noqa: E402
from hamic_jax.params import init_params  # noqa: E402
from hamic_jax.spec import ModelPreset  # noqa: E402


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

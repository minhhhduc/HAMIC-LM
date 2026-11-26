"""Attention kernel switch: the audit path must match the fused primitive."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from hamic_jax.attention import attention


def test_reference_matches_fused() -> None:
    rng = np.random.default_rng(0)
    query = jnp.asarray(rng.normal(size=(2, 4, 4, 8)), jnp.float32)
    key = jnp.asarray(rng.normal(size=(2, 4, 2, 8)), jnp.float32)
    value = jnp.asarray(rng.normal(size=(2, 4, 2, 8)), jnp.float32)
    mask = jnp.ones((2, 1, 4, 4), dtype=bool)

    fused = attention(query, key, value, mask, implementation="xla")
    reference = attention(query, key, value, mask, implementation="reference")
    np.testing.assert_allclose(
        np.asarray(reference), np.asarray(fused), rtol=1e-4, atol=1e-4
    )


def test_unknown_implementation_is_rejected() -> None:
    query = jnp.ones((1, 2, 1, 4), jnp.float32)
    with pytest.raises(ValueError):
        attention(query, query, query, None, implementation="flash")

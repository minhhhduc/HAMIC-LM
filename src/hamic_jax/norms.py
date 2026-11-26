"""Normalization layers used by the three HAMIC architectures."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["layer_norm", "rms_norm"]

Array = jax.Array


def layer_norm(x: Array, weight: Array, eps: float = 1e-5) -> Array:
    """Weight-only LayerNorm with float32 statistics."""
    dtype = x.dtype
    value = x.astype(jnp.float32)
    mean = jnp.mean(value, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(value - mean), axis=-1, keepdims=True)
    normalized = (value - mean) * jax.lax.rsqrt(variance + eps)
    return normalized.astype(dtype) * weight


def rms_norm(x: Array, weight: Array, eps: float = 1e-6) -> Array:
    """Llama/Qwen RMSNorm, including HF's cast-before-weight step."""
    dtype = x.dtype
    value = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(value), axis=-1, keepdims=True)
    normalized = (value * jax.lax.rsqrt(variance + eps)).astype(dtype)
    return normalized * weight
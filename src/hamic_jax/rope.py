"""Rotary position embeddings, matching the pinned Hugging Face implementation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hamic_jax.spec import ModelPreset

__all__ = ["RopeTables", "apply_rope", "rotate_half", "rope_tables"]

Array = jax.Array


class RopeTables(NamedTuple):
    """Cosine and sine tables shaped ``[seq_len, head_dim]``."""

    cos: Array
    sin: Array


def rope_tables(
    preset: ModelPreset,
    seq_len: int,
    dtype: jnp.dtype = jnp.float32,
    offset: int = 0,
) -> RopeTables:
    """Build RoPE tables for absolute positions ``offset .. offset + seq_len``.

    Hugging Face computes the inverse frequencies in float32, multiplies by the position,
    concatenates the two halves, and only then casts to the activation dtype. Position ids
    are absolute column indices; they do not reset at document boundaries.
    """
    dim = preset.head_dim
    if dim % 2:
        raise ValueError("RoPE requires an even head_dim")
    exponent = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    inv_freq = 1.0 / (jnp.float32(preset.rope_theta) ** exponent)
    positions = jnp.arange(offset, offset + seq_len, dtype=jnp.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return RopeTables(cos=emb.cos().astype(dtype), sin=emb.sin().astype(dtype))


def rotate_half(x: Array) -> Array:
    """Rotate the last dimension by swapping and negating its halves."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return jnp.concatenate((-second, first), axis=-1)


def apply_rope(q: Array, k: Array, tables: RopeTables) -> tuple[Array, Array]:
    """Apply RoPE to ``[batch, seq, heads, head_dim]`` queries and keys.

    HF rotates in float32 and casts back; the same cast is kept here for parity.
    """
    dtype = q.dtype
    cos = tables.cos[None, :, None, :]
    sin = tables.sin[None, :, None, :]
    qf = q.astype(jnp.float32)
    kf = k.astype(jnp.float32)
    q_out = qf * cos + rotate_half(qf) * sin
    k_out = kf * cos + rotate_half(kf) * sin
    return q_out.astype(dtype), k_out.astype(dtype)
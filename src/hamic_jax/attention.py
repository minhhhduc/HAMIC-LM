"""Attention masks and fused scaled dot-product attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "attention",
    "causal_mask",
    "decoder_mask",
    "encoder_mask",
]

Array = jax.Array


def encoder_mask(
    input_ids: Array,
    attention_mask: Array | None,
    pad_id: int,
    bos_id: int,
    document_mask: bool,
) -> Array:
    """Bidirectional mask shaped ``[batch, 1, seq, seq]``.

    ``attention_mask`` is the usual 1-for-real-token mask. When ``document_mask`` is true,
    tokens may only attend inside their own document; a document starts at each ``bos``.
    """
    valid = (input_ids != pad_id) if attention_mask is None else attention_mask.astype(bool)
    mask = valid[:, None, :] & valid[:, :, None]
    if document_mask:
        document = jnp.cumsum(input_ids == bos_id, axis=1) - 1
        document = jnp.maximum(document, 0)
        mask = mask & (document[:, :, None] == document[:, None, :])
    return mask[:, None, :, :]


def causal_mask(seq_len: int) -> Array:
    """Lower-triangular boolean mask shaped ``[1, 1, seq, seq]``."""
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, None, :, :]


def decoder_mask(
    input_ids: Array,
    attention_mask: Array | None,
    pad_id: int,
    bos_id: int,
    document_mask: bool,
) -> Array:
    """Causal mask, optionally restricted to the same document."""
    valid = (input_ids != pad_id) if attention_mask is None else attention_mask.astype(bool)
    seq_len = input_ids.shape[1]
    mask = causal_mask(seq_len) & valid[:, None, None, :]
    if document_mask:
        document = jnp.cumsum(input_ids == bos_id, axis=1) - 1
        document = jnp.maximum(document, 0)
        mask = mask & (document[:, :, None] == document[:, None, :])[:, None, :, :]
    return mask


def attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None,
    scale: float | None = None,
) -> Array:
    """Fused scaled dot-product attention over ``[batch, seq, heads, dim]`` arrays.

    ``jax.nn.dot_product_attention`` is XLA's attention primitive: it keeps the logits in
    float32, applies the mask with a finite dtype minimum, and lets XLA fuse the matmuls
    and softmax on TPU. It also discovers grouped-query attention from the head counts,
    so the decoder never materialises ``repeat_kv``.

    HF eager attention multiplies the logits by ``head_dim ** -0.5``; the fused primitive
    does the same, after the QK product, which keeps the fp32 rounding order.
    """
    return jax.nn.dot_product_attention(query, key, value, mask=mask, scale=scale)
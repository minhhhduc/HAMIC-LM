"""Attention masks and scaled dot-product attention.

The default implementation is XLA's fused attention primitive, which is the only
attention kernel available on v5e. ``reference`` is an explicit fp32 softmax used to
audit the fused path; it is slower and no production config selects it.
"""

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

_FUSED_ALIASES = ("xla", "sdpa", "flex_attention")


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


def _reference_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None,
    scale: float | None,
) -> Array:
    """Materialise logits and softmax in fp32; a slow audit of the fused kernel."""
    if key.shape[2] != query.shape[2]:
        repeat = query.shape[2] // key.shape[2]
        key = jnp.repeat(key, repeat, axis=2)
        value = jnp.repeat(value, repeat, axis=2)
    factor = query.shape[-1] ** -0.5 if scale is None else scale
    logits = jnp.einsum(
        "bqhd,bkhd->bhqk",
        query.astype(jnp.float32),
        key.astype(jnp.float32),
    )
    logits = logits * factor
    if mask is not None:
        # Padding rows can be fully masked. Attending to every key there keeps the
        # softmax finite; their outputs are ignored by the loss.
        empty = ~jnp.any(mask, axis=-1, keepdims=True)
        logits = jnp.where(mask | empty, logits, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(logits, axis=-1)
    context = jnp.einsum("bhqk,bkhd->bqhd", weights, value.astype(jnp.float32))
    return context.astype(value.dtype)


def attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None,
    scale: float | None = None,
    *,
    implementation: str = "xla",
) -> Array:
    """Scaled dot-product attention over ``[batch, seq, heads, dim]`` arrays.

    ``jax.nn.dot_product_attention`` is XLA's attention primitive: it keeps the logits in
    float32, applies the mask with a finite dtype minimum, and lets XLA fuse the matmuls
    and softmax on TPU. It also discovers grouped-query attention from the head counts,
    so the decoder never materialises ``repeat_kv``. TPU v5e has no FlexAttention kernel;
    the config's ``flex_attention`` and ``sdpa`` both resolve to this primitive, while
    ``reference`` runs the explicit audit path above.

    HF eager attention multiplies the logits by ``head_dim ** -0.5``; the fused primitive
    does the same, after the QK product, which keeps the fp32 rounding order.
    """
    if implementation in _FUSED_ALIASES:
        return jax.nn.dot_product_attention(query, key, value, mask=mask, scale=scale)
    if implementation == "reference":
        return _reference_attention(query, key, value, mask, scale)
    raise ValueError(
        f"unknown attention implementation {implementation!r}; "
        f"use one of {_FUSED_ALIASES + ('reference',)}"
    )
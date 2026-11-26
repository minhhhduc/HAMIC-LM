"""JAX forwards for the three HAMIC architecture families.

The implementation mirrors the pinned Hugging Face modules:

* ModernBERT: pre-LayerNorm, bias-free projections, GeGLU, tied decoder.
* Llama: pre-RMSNorm, SwiGLU, grouped-query attention, tied LM head.
* Qwen3: Llama plus per-head RMSNorm on queries and keys before RoPE.

The tree is the flat ``dict[str, Array]`` produced by :mod:`hamic_jax.params`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from hamic_jax.attention import attention, decoder_mask, encoder_mask
from hamic_jax.mlp import geglu, swiglu
from hamic_jax.norms import layer_norm, rms_norm
from hamic_jax.rope import apply_rope, rope_tables
from hamic_jax.spec import ModelPreset, SpecialTokens

__all__ = ["ModelOutput", "cross_entropy", "encoder_hidden", "forward", "loss"]

Array = jax.Array
_IGNORE_INDEX = -100


class ModelOutput(NamedTuple):
    """Logits and optional loss from one forward pass."""

    logits: Array
    loss: Array | None
    hidden: Array


def _dense(x: Array, weight: Array) -> Array:
    return x @ weight.T


def _linear_bias(x: Array, weight: Array, bias: Array) -> Array:
    return x @ weight.T + bias


def encoder_hidden(
    params: dict[str, Array],
    preset: ModelPreset,
    input_ids: Array,
    attention_mask: Array | None = None,
    document_mask: bool = True,
    special: SpecialTokens | None = None,
) -> Array:
    """Return the final encoder hidden states, before the MLM head."""
    special = special or SpecialTokens()
    hidden = params["model.embeddings.tok_embeddings.weight"][input_ids]
    hidden = layer_norm(hidden, params["model.embeddings.norm.weight"], eps=preset.norm_eps)
    mask = encoder_mask(
        input_ids,
        attention_mask,
        pad_id=special.pad_id,
        bos_id=special.bos_id,
        document_mask=document_mask,
    )
    tables = rope_tables(preset, input_ids.shape[1], dtype=hidden.dtype)
    heads = preset.num_heads
    head_dim = preset.head_dim
    for layer in range(preset.num_layers):
        prefix = f"model.layers.{layer}"
        if layer > 0:
            attn_input = layer_norm(hidden, params[f"{prefix}.attn_norm.weight"], eps=preset.norm_eps)
        else:
            attn_input = hidden
        qkv = _dense(attn_input, params[f"{prefix}.attn.Wqkv.weight"])
        qkv = qkv.reshape(input_ids.shape[0], input_ids.shape[1], 3, heads, head_dim)
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        query, key = apply_rope(query, key, tables)
        context = attention(query, key, value, mask)
        context = context.reshape(input_ids.shape[0], input_ids.shape[1], heads * head_dim)
        hidden = hidden + _dense(context, params[f"{prefix}.attn.Wo.weight"])
        mlp_input = layer_norm(hidden, params[f"{prefix}.mlp_norm.weight"], eps=preset.norm_eps)
        hidden = hidden + geglu(
            mlp_input,
            params[f"{prefix}.mlp.Wi.weight"],
            params[f"{prefix}.mlp.Wo.weight"],
        )
    return layer_norm(hidden, params["model.final_norm.weight"], eps=preset.norm_eps)


def _decoder_forward(
    params: dict[str, Array],
    preset: ModelPreset,
    input_ids: Array,
    attention_mask: Array | None,
    document_mask: bool,
    special: SpecialTokens,
) -> Array:
    hidden = params["model.embed_tokens.weight"][input_ids]
    mask = decoder_mask(
        input_ids,
        attention_mask,
        pad_id=special.pad_id,
        bos_id=special.bos_id,
        document_mask=document_mask,
    )
    tables = rope_tables(preset, input_ids.shape[1], dtype=hidden.dtype)
    heads = preset.num_heads
    kv_heads = preset.num_kv_heads
    head_dim = preset.head_dim
    for layer in range(preset.num_layers):
        prefix = f"model.layers.{layer}"
        attn_input = rms_norm(
            hidden, params[f"{prefix}.input_layernorm.weight"], eps=preset.norm_eps
        )
        query = _dense(attn_input, params[f"{prefix}.self_attn.q_proj.weight"])
        key = _dense(attn_input, params[f"{prefix}.self_attn.k_proj.weight"])
        value = _dense(attn_input, params[f"{prefix}.self_attn.v_proj.weight"])
        query = query.reshape(input_ids.shape[0], input_ids.shape[1], heads, head_dim)
        key = key.reshape(input_ids.shape[0], input_ids.shape[1], kv_heads, head_dim)
        value = value.reshape(input_ids.shape[0], input_ids.shape[1], kv_heads, head_dim)
        if preset.architecture == "qwen3":
            query = rms_norm(
                query, params[f"{prefix}.self_attn.q_norm.weight"], eps=preset.norm_eps
            )
            key = rms_norm(key, params[f"{prefix}.self_attn.k_norm.weight"], eps=preset.norm_eps)
        query, key = apply_rope(query, key, tables)
        context = attention(query, key, value, mask)
        context = context.reshape(
            input_ids.shape[0], input_ids.shape[1], heads * head_dim
        )
        hidden = hidden + _dense(context, params[f"{prefix}.self_attn.o_proj.weight"])
        mlp_input = rms_norm(
            hidden, params[f"{prefix}.post_attention_layernorm.weight"], eps=preset.norm_eps
        )
        hidden = hidden + swiglu(
            mlp_input,
            params[f"{prefix}.mlp.gate_proj.weight"],
            params[f"{prefix}.mlp.up_proj.weight"],
            params[f"{prefix}.mlp.down_proj.weight"],
        )
    hidden = rms_norm(hidden, params["model.norm.weight"], eps=preset.norm_eps)
    logits = _dense(hidden, params["model.embed_tokens.weight"])
    return logits


def cross_entropy(logits: Array, labels: Array, causal: bool) -> Array:
    """Masked mean cross-entropy.

    For causal models the logits and labels are shifted by one, matching
    ``ForCausalLMLoss`` exactly (the last label is discarded). ``-100`` positions are
    ignored in both the numerator and the denominator.
    """
    if causal:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
    valid = labels != _IGNORE_INDEX
    safe_labels = jnp.where(valid, labels, 0)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), safe_labels
    )
    numerator = jnp.sum(jnp.where(valid, token_loss, 0.0))
    denominator = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    return numerator / denominator


def forward(
    params: dict[str, Array],
    preset: ModelPreset,
    input_ids: Array,
    labels: Array | None = None,
    attention_mask: Array | None = None,
    *,
    document_mask: bool = True,
    objective: str | None = None,
    special: SpecialTokens | None = None,
) -> ModelOutput:
    """Run one forward pass and optionally compute the training loss.

    Args:
        params: Flat parameter tree.
        preset: Model shape.
        input_ids: ``[batch, seq]`` integer token ids.
        labels: Masked-LM labels or causal-LM targets; ``-100`` is ignored.
        attention_mask: Optional ``[batch, seq]`` real-token mask. When absent, pad ids
            are masked out.
        document_mask: Restrict attention to the document that starts at each bos token.
        objective: ``"mlm"`` or ``"clm"``. Defaults by architecture.
        special: Special-token ids; defaults to HAMIC's fixed layout.
    """
    special = special or SpecialTokens()
    if preset.is_encoder:
        hidden = encoder_hidden(
            params, preset, input_ids, attention_mask, document_mask, special
        )
        head = jax.nn.gelu(_dense(hidden, params["head.dense.weight"]), approximate=False)
        head = layer_norm(head, params["head.norm.weight"], eps=preset.norm_eps)
        embedding = params["model.embeddings.tok_embeddings.weight"]
        logits = _linear_bias(head, embedding, params["decoder.bias"])
    else:
        hidden = _decoder_forward(
            params, preset, input_ids, attention_mask, document_mask, special
        )
        logits = hidden
    loss_value: Array | None = None
    if labels is not None:
        chosen = objective or ("mlm" if preset.is_encoder else "clm")
        loss_value = cross_entropy(logits, labels, causal=chosen == "clm")
    return ModelOutput(logits=logits, loss=loss_value, hidden=hidden)


def loss(
    params: dict[str, Array],
    preset: ModelPreset,
    input_ids: Array,
    labels: Array,
    attention_mask: Array | None = None,
    *,
    document_mask: bool = True,
    objective: str | None = None,
) -> Array:
    """Return only the scalar loss (convenience for ``jax.value_and_grad``)."""
    return forward(
        params,
        preset,
        input_ids,
        labels,
        attention_mask,
        document_mask=document_mask,
        objective=objective,
    ).loss
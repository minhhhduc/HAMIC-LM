"""Flat parameter trees keyed by Hugging Face ``state_dict`` names.

A flat ``dict[str, jax.Array]`` is deliberate: it is easy to shard across TPU hosts, easy
to compare with a torch ``state_dict``, and has no framework-specific wrapper. Tied
weights appear once; :func:`state_dict` materializes the aliases on export.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from hamic_jax.spec import Architecture, ModelPreset

__all__ = [
    "count_parameters",
    "init_params",
    "parameter_shapes",
    "state_dict",
    "tied_aliases",
    "validate_params",
]

Array = jax.Array


def tied_aliases(preset: ModelPreset) -> dict[str, str]:
    """Return output-head aliases pointing at the input embedding leaf."""
    if not preset.tie_word_embeddings:
        return {}
    if preset.is_encoder:
        return {"decoder.weight": "model.embeddings.tok_embeddings.weight"}
    return {"lm_head.weight": "model.embed_tokens.weight"}


def parameter_shapes(preset: ModelPreset) -> dict[str, tuple[int, ...]]:
    """Return every unique leaf's shape, keyed by its canonical HF name."""
    if preset.is_encoder:
        return _encoder_shapes(preset)
    return _decoder_shapes(preset)


def _encoder_shapes(preset: ModelPreset) -> dict[str, tuple[int, ...]]:
    hidden = preset.hidden_size
    inner = preset.intermediate_size
    vocab = preset.vocab_size
    if preset.head_dim * preset.num_heads != hidden:
        raise ValueError("ModernBERT requires head_dim * num_heads == hidden_size")
    shapes: dict[str, tuple[int, ...]] = {
        "model.embeddings.tok_embeddings.weight": (vocab, hidden),
        "model.embeddings.norm.weight": (hidden,),
    }
    for layer in range(preset.num_layers):
        prefix = f"model.layers.{layer}"
        if layer > 0:
            shapes[f"{prefix}.attn_norm.weight"] = (hidden,)
        shapes[f"{prefix}.attn.Wqkv.weight"] = (3 * hidden, hidden)
        shapes[f"{prefix}.attn.Wo.weight"] = (hidden, hidden)
        shapes[f"{prefix}.mlp_norm.weight"] = (hidden,)
        shapes[f"{prefix}.mlp.Wi.weight"] = (2 * inner, hidden)
        shapes[f"{prefix}.mlp.Wo.weight"] = (hidden, inner)
    shapes["model.final_norm.weight"] = (hidden,)
    shapes["head.dense.weight"] = (hidden, hidden)
    shapes["head.norm.weight"] = (hidden,)
    shapes["decoder.bias"] = (vocab,)
    return shapes


def _decoder_shapes(preset: ModelPreset) -> dict[str, tuple[int, ...]]:
    hidden = preset.hidden_size
    inner = preset.intermediate_size
    vocab = preset.vocab_size
    q_width = preset.num_heads * preset.head_dim
    kv_width = preset.num_kv_heads * preset.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
    }
    for layer in range(preset.num_layers):
        prefix = f"model.layers.{layer}"
        shapes[f"{prefix}.self_attn.q_proj.weight"] = (q_width, hidden)
        shapes[f"{prefix}.self_attn.k_proj.weight"] = (kv_width, hidden)
        shapes[f"{prefix}.self_attn.v_proj.weight"] = (kv_width, hidden)
        shapes[f"{prefix}.self_attn.o_proj.weight"] = (hidden, q_width)
        if preset.architecture == Architecture.QWEN3:
            shapes[f"{prefix}.self_attn.q_norm.weight"] = (preset.head_dim,)
            shapes[f"{prefix}.self_attn.k_norm.weight"] = (preset.head_dim,)
        shapes[f"{prefix}.mlp.gate_proj.weight"] = (inner, hidden)
        shapes[f"{prefix}.mlp.up_proj.weight"] = (inner, hidden)
        shapes[f"{prefix}.mlp.down_proj.weight"] = (hidden, inner)
        shapes[f"{prefix}.input_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden,)
    shapes["model.norm.weight"] = (hidden,)
    return shapes


def count_parameters(preset: ModelPreset) -> int:
    """Return the exact leaf count, with tied weights counted once."""
    return sum(math.prod(shape) for shape in parameter_shapes(preset).values())


def init_params(
    preset: ModelPreset,
    key: Array,
    dtype: Any = jnp.float32,
) -> dict[str, Array]:
    """Initialize a flat tree with the HF post-init convention.

    Matrices are normal(0, 0.02); one-dimensional norm weights are ones; decoder biases
    are zeros. This is not a bitwise reproduction of torch RNG, but it matches the
    documented initializer family. Trained parity always starts from a converted
    checkpoint.
    """
    leaves: dict[str, Array] = {}
    for index, (name, shape) in enumerate(parameter_shapes(preset).items()):
        leaf_key = jax.random.fold_in(key, index)
        if name.endswith("bias"):
            value = jnp.zeros(shape, dtype=dtype)
        elif len(shape) == 1:
            value = jnp.ones(shape, dtype=dtype)
        else:
            value = jax.random.normal(leaf_key, shape, dtype=jnp.float32) * 0.02
            value = value.astype(dtype)
        leaves[name] = value
    return leaves


def state_dict(params: dict[str, Array], preset: ModelPreset) -> dict[str, Array]:
    """Materialize every HF key, including tied aliases."""
    exported = dict(params)
    for alias, source in tied_aliases(preset).items():
        exported[alias] = params[source]
    return exported


def validate_params(params: dict[str, Array], preset: ModelPreset) -> None:
    """Raise if a tree is missing a leaf or has the wrong shape."""
    expected = parameter_shapes(preset)
    missing = sorted(set(expected) - set(params))
    extra = sorted(set(params) - set(expected))
    if missing or extra:
        raise ValueError(f"parameter keys differ: missing={missing}, extra={extra}")
    for name, shape in expected.items():
        if tuple(params[name].shape) != shape:
            raise ValueError(f"{name}: expected {shape}, got {tuple(params[name].shape)}")
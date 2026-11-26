"""Feed-forward blocks for encoder and decoder presets."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["geglu", "swiglu"]

Array = jax.Array


def geglu(x: Array, wi: Array, wo: Array) -> Array:
    """ModernBERT MLP: ``Wo(gelu(Wi(x)[:inner]) * Wi(x)[inner:])``.

    ``Wi`` has shape ``[2 * inner, hidden]``. Hugging Face chunks the projection into
    ``(input, gate)`` and applies the activation to the **first** chunk.
    """
    projected = x @ wi.T
    inner = projected.shape[-1] // 2
    value, gate = projected[..., :inner], projected[..., inner:]
    return (jax.nn.gelu(value, approximate=False) * gate) @ wo.T


def swiglu(x: Array, gate: Array, up: Array, down: Array) -> Array:
    """Llama/Qwen MLP: ``down(silu(gate(x)) * up(x))``."""
    return (jax.nn.silu(x @ gate.T) * (x @ up.T)) @ down.T
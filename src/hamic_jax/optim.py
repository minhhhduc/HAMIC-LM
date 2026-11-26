"""Optimizer built from Optax, not hand-written update math.

Optax owns AdamW and the schedule composition. The schedule reproduces HAMIC's WSD shape
(linear warmup, constant, linear decay) and cosine shape with ``optax.join_schedules``.

Global-norm clipping is the one piece written in JAX: the oracle's
``torch.nn.utils.clip_grad_norm_`` divides by ``||g|| + 1e-6`` and clips only when
``||g|| >= max_norm``. ``optax.clip_by_global_norm`` omits the epsilon, so it is not
numerically equivalent and the final-audit rule requires the JAX version.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from hamic_jax.spec import TrainingConfig

__all__ = ["build_optimizer", "clip_by_global_norm", "learning_rate_schedule"]

Array = jax.Array

_CLIP_EPS = 1e-6


def clip_by_global_norm(max_norm: float) -> optax.GradientTransformation:
    """Clip gradients by the oracle's global-norm formula.

    ``torch.nn.utils.clip_grad_norm_`` computes the total norm in float32 and scales by
    ``min(1, max_norm / (norm + 1e-6))``. The norm is reduced across the whole tree, so
    this remains a single global scalar under data parallelism; XLA lowers the reduction
    to an all-reduce when the leaves are sharded.
    """

    def init_fn(params: Any) -> tuple[()]:
        del params
        return ()

    def update_fn(
        updates: Any, state: tuple[()], params: Any | None = None
    ) -> tuple[Any, tuple[()]]:
        del params
        leaves = jax.tree.leaves(updates)
        if not leaves:
            return updates, state
        squared = sum(
            jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves
        )
        norm = jnp.sqrt(squared)
        scale = jnp.minimum(
            1.0, jnp.asarray(max_norm, jnp.float32) / (norm + _CLIP_EPS)
        )
        clipped = jax.tree.map(
            lambda leaf: (leaf.astype(jnp.float32) * scale).astype(leaf.dtype), updates
        )
        return clipped, state

    return optax.GradientTransformation(init_fn, update_fn)


def learning_rate_schedule(config: TrainingConfig) -> optax.Schedule:
    """Return the LR schedule for a run, in absolute learning-rate units."""
    total = max(1, config.total_steps)
    warmup = max(1, config.warmup_steps)
    peak = float(config.learning_rate)
    floor = peak * float(config.min_lr_ratio)
    start = peak / warmup
    # The oracle returns ``(step + 1) / warmup``; ending the linear ramp one step
    # before the boundary reproduces it exactly instead of being one step slow.
    warm = optax.linear_schedule(
        init_value=start,
        end_value=peak,
        transition_steps=max(1, warmup - 1),
    )

    if config.lr_schedule == "wsd":
        decay_start = max(warmup, total - max(1, round(total * config.decay_ratio)))
        hold = optax.constant_schedule(peak)
        # The oracle applies the first decay step at ``decay_start``, so the decay branch
        # starts one step below the peak and reaches the floor on the final step.
        span = max(1, total - decay_start)
        decay = optax.linear_schedule(
            init_value=peak - (peak - floor) / span,
            end_value=floor,
            transition_steps=max(1, span - 1),
        )
        if decay_start <= warmup:
            # Short runs and large decay ratios collapse the constant phase. Joining
            # two schedules at the same boundary is rejected by optax, and the oracle
            # itself starts decaying once warmup ends.
            return optax.join_schedules([warm, decay], [warmup])
        return optax.join_schedules([warm, hold, decay], [warmup, decay_start])

    if total <= warmup:
        return warm
    cosine = optax.cosine_decay_schedule(
        init_value=peak,
        decay_steps=total - warmup,
        alpha=floor / peak,
    )
    return optax.join_schedules([warm, cosine], [warmup])


def build_optimizer(config: TrainingConfig, params: dict[str, Array]) -> optax.GradientTransformation:
    """AdamW with decay on matrices only, plus global-norm clipping."""
    matrix_mask = jax.tree.map(lambda leaf: leaf.ndim >= 2, params)
    return optax.chain(
        clip_by_global_norm(config.grad_clip),
        optax.adamw(
            learning_rate=learning_rate_schedule(config),
            b1=config.betas[0],
            b2=config.betas[1],
            eps=1e-8,
            weight_decay=config.weight_decay,
            mask=matrix_mask,
        ),
    )
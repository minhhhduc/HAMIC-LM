"""JIT training loop for the JAX backend.

Mask generation and packing stay on the host (deterministic NumPy), matching the GPU
backend's split between CPU data work and accelerator compute. The step itself is a pure
JAX function with gradient accumulation via ``lax.scan``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from hamic_jax.data import PackedBatches, TokenizedCorpus
from hamic_jax.model import forward
from hamic_jax.objectives import CLMObjective, MLMObjective, MaskedBatch
from hamic_jax.optim import build_optimizer
from hamic_jax.params import init_params
from hamic_jax.sharding import DeviceLayout
from hamic_jax.spec import Architecture, ModelPreset, TrainingConfig

__all__ = ["TrainState", "Trainer", "evaluate", "evaluate_metrics"]

Array = jax.Array

# Validation masks each batch with a distinct, run-independent seed, as the oracle does.
_EVAL_SEED_OFFSET = 1 << 40


@dataclass
class TrainState:
    """Everything needed to resume a run."""

    params: dict[str, Array]
    optimizer: Any
    step: int
    consumed_tokens: int


class Trainer:
    """Train one flat parameter tree on packed batches."""

    def __init__(
        self,
        config: TrainingConfig,
        preset: ModelPreset,
        corpus: TokenizedCorpus,
        params: dict[str, Array] | None = None,
        seed: int | None = None,
        dtype: Any | None = None,
    ) -> None:
        if (
            not config.cross_document_attention
            and preset.architecture != Architecture.MODERNBERT
        ):
            raise ValueError(
                "cross_document_attention=False is supported for ModernBERT only"
            )
        if (
            config.sparse_prediction
            and preset.architecture != Architecture.MODERNBERT
        ):
            raise ValueError("sparse_prediction is a ModernBERT option")
        self.config = config
        self.preset = preset
        self.dtype = dtype if dtype is not None else (
            jnp.bfloat16 if config.precision == "bf16" else jnp.float32
        )
        self.batches = PackedBatches(
            corpus,
            "train",
            seq_len=config.seq_len,
            batch_size=config.micro_batch_size,
            bos_id=preset.special_tokens.bos_id,
            eos_id=preset.special_tokens.eos_id,
            seed=config.seed if seed is None else seed,
            shuffle=config.shuffle,
            source_weights=dict(config.source_weights) or None,
        )
        self.layout = DeviceLayout.for_batch(config.micro_batch_size)
        key = jax.random.PRNGKey(config.seed)
        master = init_params(preset, key, dtype=jnp.float32) if params is None else params
        master = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype=jnp.float32), master)
        self.params = self.layout.replicate(master)
        self.optimizer = build_optimizer(config, self.params)
        self.optimizer_state = self.layout.replicate(self.optimizer.init(self.params))
        self.step = 0
        self.objective: MLMObjective | CLMObjective
        if config.objective == "mlm":
            from pathlib import Path

            from hamic_jax.objectives import (
                NgramTables,
                TokenClasses,
                mine_pmi_ngrams,
                mine_pmi_pairs,
                word_tokens,
            )

            pairs = None
            ngrams = None
            classes: TokenClasses | None = None
            cache = Path(config.tokenized)
            if config.masking == "pmi":
                pairs_file = cache / "pmi_pairs.npy"
                if pairs_file.is_file():
                    pairs = np.load(pairs_file)
                else:
                    pairs = mine_pmi_pairs(
                        corpus.documents("train"),
                        preset.vocab_size,
                        preset.special_tokens.special_ids,
                        max_tokens=config.pmi_max_tokens,
                        min_count=config.pmi_min_count,
                        min_pmi=config.pmi_min_pmi,
                        max_pairs=config.pmi_max_pairs,
                        eligible=word_tokens(corpus.tokenizer_file()),
                        discount_k=config.pmi_discount_k,
                    )
            if config.masking == "pmi_ngram":
                classes = TokenClasses.from_tokenizer_file(corpus.tokenizer_file())
                stem = (
                    f"pmi-ngrams-{config.pmi_max_ngram}-{config.pmi_min_count}"
                    f"-{config.pmi_min_doc_count}-{config.pmi_min_pmi:g}"
                )
                ngram_file = cache / f"{stem}.npz"
                ngrams = (
                    NgramTables.load(ngram_file)
                    if ngram_file.is_file()
                    else mine_pmi_ngrams(
                        corpus.documents("train"),
                        preset.vocab_size,
                        preset.special_tokens.special_ids,
                        classes=classes,
                        max_n=config.pmi_max_ngram,
                        max_tokens=config.pmi_max_tokens,
                        min_count=config.pmi_min_count,
                        min_doc_count=config.pmi_min_doc_count,
                        min_pmi=config.pmi_min_pmi,
                        max_units=config.pmi_max_pairs,
                        discount_k=config.pmi_discount_k,
                        max_extension_share=config.pmi_max_extension_share,
                    )
                )
            if config.whole_word_masking and classes is None:
                classes = TokenClasses.from_tokenizer_file(corpus.tokenizer_file())
            self.objective = MLMObjective(
                vocab_size=preset.vocab_size,
                special=preset.special_tokens,
                seed=config.seed,
                mask_rate=config.mask_rate,
                mask_rate_final=config.mask_rate_final,
                schedule_start=config.mask_schedule_start,
                total_steps=config.total_steps * config.grad_accum_steps,
                masking=config.masking,
                span_max_length=config.span_max_length,
                pmi_pairs=pairs,
                pmi_keep=config.pmi_unit_share,
                pmi_max_unit_length=config.pmi_max_unit_length,
                whole_word_masking=config.whole_word_masking,
                ngram_tables=ngrams,
                token_classes=classes,
            )
        else:
            self.objective = CLMObjective()
        self._step_fn = self._build_step()

    def _build_step(
        self,
    ) -> Callable[..., tuple[dict[str, Array], Any, Array, Array]]:
        optimizer = self.optimizer
        preset = self.preset
        config = self.config
        layout = self.layout

        def loss_fn(
            params: dict[str, Array], input_ids: Array, labels: Array
        ) -> Array:
            output = forward(
                params,
                preset,
                input_ids,
                labels,
                document_mask=not config.cross_document_attention,
                objective=config.objective,
                attention_impl=config.attention,
                recompute=config.recompute,
            )
            assert output.loss is not None
            return output.loss

        def step(
            params: dict[str, Array],
            optimizer_state: Any,
            input_ids: Array,
            labels: Array,
        ) -> tuple[dict[str, Array], Any, Array, Array]:
            compute = jax.tree.map(lambda leaf: leaf.astype(self.dtype), params)

            def body(
                grad_sum: dict[str, Array],
                batch: tuple[Array, Array],
            ) -> tuple[dict[str, Array], Array]:
                micro_ids, micro_labels = batch
                loss, grads = jax.value_and_grad(loss_fn)(
                    compute, micro_ids, micro_labels
                )
                return jax.tree.map(jnp.add, grad_sum, grads), loss

            zeros = jax.tree.map(jnp.zeros_like, params)
            grad_sum, losses = jax.lax.scan(body, zeros, (input_ids, labels))
            # Every chip must apply the same global-batch gradient. The gradients are
            # already the sum over the local shard, so an average over chips gives the
            # gradient of the mean loss over the full batch.
            grad_sum = jax.tree.map(
                lambda value: jax.lax.pmean(value, "data"),
                grad_sum,
            )
            mean_loss = jax.lax.pmean(jnp.mean(losses), "data")
            mean_grads = jax.tree.map(
                lambda value: value.astype(jnp.float32)
                / max(1, config.grad_accum_steps),
                grad_sum,
            )
            norm = jnp.sqrt(
                sum(
                    jnp.sum(jnp.square(value))
                    for value in jax.tree.leaves(mean_grads)
                )
            )
            updates, new_optimizer_state = optimizer.update(
                mean_grads, optimizer_state, params
            )
            new_params = optax.apply_updates(params, updates)
            return new_params, new_optimizer_state, mean_loss, norm

        mapped = jax.shard_map(
            step,
            mesh=layout.mesh,
            in_specs=(
                layout.replicated_spec(self.params),
                layout.replicated_spec(self.optimizer_state),
                layout.stack_sharding.spec,
                layout.stack_sharding.spec,
            ),
            out_specs=(
                layout.replicated_spec(self.params),
                layout.replicated_spec(self.optimizer_state),
                layout.param_sharding.spec,
                layout.param_sharding.spec,
            ),
        )
        return jax.jit(mapped, donate_argnums=(0, 1, 2, 3))

    def _micro_batches(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        ids_rows = []
        label_rows = []
        for index in range(self.config.grad_accum_steps):
            rows = self.batches.next_batch()
            masked: MaskedBatch = self.objective(
                rows, self.step * self.config.grad_accum_steps + index
            )
            ids_rows.append(masked.input_ids)
            label_rows.append(masked.labels)
        return np.stack(ids_rows), np.stack(label_rows)

    def train_step_metrics(self) -> tuple[Array, Array]:
        """Run one optimizer step and return ``(loss, gradient_norm)`` device scalars.

        Neither scalar is synchronised here. Callers that log every step can read both
        once at the end of a log window, which keeps the TPU queue full.
        """
        ids, labels = self._micro_batches()
        new_params, new_optimizer_state, loss, norm = self._step_fn(
            self.params,
            self.optimizer_state,
            self.layout.place_stack(ids),
            self.layout.place_stack(labels),
        )
        self.params = new_params
        self.optimizer_state = new_optimizer_state
        self.step += 1
        return loss, norm

    def train_step(self) -> Array:
        """Run one optimizer step and return its mean loss as a device scalar."""
        loss, _ = self.train_step_metrics()
        return loss

    def restore(self, params: Any, optimizer_state: Any) -> None:
        """Replace the state after loading a checkpoint on a different device layout."""
        master = jax.tree.map(
            lambda leaf: jnp.asarray(leaf, dtype=jnp.float32),
            params,
        )
        self.params = self.layout.replicate(master)
        self.optimizer_state = self.layout.replicate(optimizer_state)

    def train(self, steps: int, callback: Callable[[int, float], None] | None = None) -> None:
        """Run up to ``steps`` optimizer steps."""
        for _ in range(steps):
            loss = self.train_step()
            if callback is not None:
                callback(self.step, float(loss))
        if steps:
            jax.block_until_ready(loss)

    def cursor(self) -> Any:
        """Return the packer cursor for checkpoint metadata."""
        return self.batches.cursor()

    def state(self) -> TrainState:
        """Bundle the resumable state."""
        return TrainState(
            params=self.params,
            optimizer=self.optimizer_state,
            step=self.step,
            consumed_tokens=self.batches.consumed_tokens,
        )


def evaluate_metrics(
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    params: dict[str, Array],
    sequences: int | None = None,
    batch_size: int | None = None,
    *,
    masking: str | None = None,
    mask_rate: float | None = None,
    pmi_pairs: np.ndarray[Any, Any] | None = None,
) -> tuple[float, float | None]:
    """Mean validation loss and, for causal LM, bits per byte.

    ``masking`` selects the MLM view (``token``, ``span`` or ``pmi``). Eval notebooks
    pass all three against the same parameters; the training objective is unchanged.

    The loss is the token-weighted mean over every batch: each shard contributes its
    local sum and valid-token count through ``psum``, so splitting the batch over chips
    does not change the reported number. Bits per byte additionally needs the byte
    length of each target token. It is only available for causal LM with a byte-level
    tokenizer next to the shards; MLM targets have no single next-token byte count.
    """
    count = sequences or config.eval_sequences
    batch = batch_size or config.micro_batch_size
    layout = DeviceLayout.for_batch(batch)
    batches = PackedBatches(
        corpus,
        "validation",
        seq_len=config.seq_len,
        batch_size=batch,
        bos_id=preset.special_tokens.bos_id,
        eos_id=preset.special_tokens.eos_id,
        seed=config.seed,
        shuffle=False,
    )
    objective: Any
    if config.objective == "mlm":
        view = masking or "token"
        rate = mask_rate if mask_rate is not None else (config.eval_mask_rate or config.mask_rate)
        objective = masking_views(
            config,
            preset,
            corpus,
            (view,),
            mask_rate=rate,
            pmi_pairs=pmi_pairs,
        )[view]
    else:
        if masking not in (None, "token"):
            raise ValueError("masking views only apply to the MLM objective")
        objective = CLMObjective()
    byte_table: Array | None = None
    if config.objective == "clm":
        try:
            byte_table = layout.replicate(
                jnp.asarray(corpus.token_byte_lengths(), dtype=jnp.float32)
            )
        except (FileNotFoundError, KeyError, ValueError):
            byte_table = None
    dtype = jnp.bfloat16 if config.precision == "bf16" else jnp.float32
    compute = layout.replicate(
        jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype=dtype), params)
    )

    def eval_step(
        shard_params: dict[str, Array],
        input_ids: Array,
        labels: Array,
    ) -> tuple[Array, Array, Array]:
        output = forward(
            shard_params,
            preset,
            input_ids,
            labels,
            document_mask=not config.cross_document_attention,
            objective=config.objective,
            attention_impl=config.attention,
            recompute=config.recompute,
        )
        assert output.loss is not None
        targets = labels[:, 1:] if config.objective == "clm" else labels
        valid = targets != -100
        local_count = jnp.sum(valid.astype(jnp.float32))
        local_nats = output.loss * local_count
        global_nats = jax.lax.psum(local_nats, "data")
        global_count = jax.lax.psum(local_count, "data")
        global_loss = global_nats / jnp.maximum(global_count, 1.0)
        if byte_table is None:
            zero = jnp.zeros((), dtype=jnp.float32)
            return global_loss, global_nats, zero
        safe = jnp.where(valid, targets, 0)
        local_bytes = jnp.sum(byte_table[safe] * valid.astype(jnp.float32))
        return (
            global_loss,
            global_nats,
            jax.lax.psum(local_bytes, "data"),
        )

    mapped = jax.shard_map(
        eval_step,
        mesh=layout.mesh,
        in_specs=(
            layout.replicated_spec(compute),
            layout.batch_sharding.spec,
            layout.batch_sharding.spec,
        ),
        out_specs=(
            layout.param_sharding.spec,
            layout.param_sharding.spec,
            layout.param_sharding.spec,
        ),
    )
    jitted = jax.jit(mapped)
    losses: list[float] = []
    nats = 0.0
    target_bytes = 0.0
    for index in range(max(1, math.ceil(count / batch))):
        rows = batches.next_batch()
        masked = objective(rows, _EVAL_SEED_OFFSET + index)
        loss, batch_nats, batch_bytes = jitted(
            compute,
            layout.place_batch(masked.input_ids),
            layout.place_batch(masked.labels),
        )
        losses.append(float(loss))
        nats += float(batch_nats)
        target_bytes += float(batch_bytes)
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    bits_per_byte = (
        nats / (math.log(2) * target_bytes)
        if byte_table is not None and target_bytes > 0
        else None
    )
    return mean_loss, bits_per_byte


def evaluate(
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    params: dict[str, Array],
    sequences: int | None = None,
    batch_size: int | None = None,
    *,
    masking: str | None = None,
    mask_rate: float | None = None,
    pmi_pairs: np.ndarray[Any, Any] | None = None,
) -> float:
    """Mean validation loss over a deterministic prefix of the validation split."""
    loss, _ = evaluate_metrics(
        config,
        preset,
        corpus,
        params,
        sequences,
        batch_size,
        masking=masking,
        mask_rate=mask_rate,
        pmi_pairs=pmi_pairs,
    )
    return loss

def masking_views(
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    views: tuple[str, ...] = ("token", "span", "pmi"),
    *,
    mask_rate: float | None = None,
    pmi_pairs: np.ndarray[Any, Any] | None = None,
) -> dict[str, MLMObjective]:
    """Build one MLM objective per requested masking view.

    PMI pairs are mined once per call. Callers evaluating several checkpoints can pass
    a precomputed ``pmi_pairs`` table to avoid mining for every run.
    """
    if config.objective != "mlm":
        raise ValueError("masking views only apply to the MLM objective")
    rate = mask_rate if mask_rate is not None else (config.eval_mask_rate or config.mask_rate)
    objectives: dict[str, MLMObjective] = {}
    for view in views:
        if view not in ("token", "span", "pmi"):
            raise ValueError(f"unsupported masking view: {view!r}")
        pairs = pmi_pairs
        if view == "pmi" and pairs is None:
            from hamic_jax.objectives import mine_pmi_pairs, word_tokens

            pairs = mine_pmi_pairs(
                corpus.documents("train"),
                preset.vocab_size,
                preset.special_tokens.special_ids,
                max_tokens=config.pmi_max_tokens,
                min_count=config.pmi_min_count,
                min_pmi=config.pmi_min_pmi,
                max_pairs=config.pmi_max_pairs,
                eligible=word_tokens(corpus.tokenizer_file()),
                discount_k=config.pmi_discount_k,
            )
        objectives[view] = MLMObjective(
            vocab_size=preset.vocab_size,
            special=preset.special_tokens,
            seed=config.seed,
            mask_rate=rate,
            masking=view,
            span_max_length=config.span_max_length,
            pmi_pairs=pairs,
            pmi_keep=config.pmi_unit_share,
            pmi_max_unit_length=config.pmi_max_unit_length,
            total_steps=1,
        )
    return objectives


def evaluate_views(
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    params: dict[str, Array],
    views: tuple[str, ...] = ("token", "span", "pmi"),
    sequences: int | None = None,
    batch_size: int | None = None,
    *,
    mask_rate: float | None = None,
    pmi_pairs: np.ndarray[Any, Any] | None = None,
) -> dict[str, float]:
    """Evaluate the same parameters under each requested masking view."""
    mining = pmi_pairs
    if config.objective == "mlm" and "pmi" in views and mining is None:
        mining = masking_views(
            config,
            preset,
            corpus,
            ("pmi",),
            mask_rate=mask_rate,
        )["pmi"].pmi_pairs
    return {
        view: evaluate(
            config,
            preset,
            corpus,
            params,
            sequences=sequences,
            batch_size=batch_size,
            masking=view,
            mask_rate=mask_rate,
            pmi_pairs=mining,
        )
        for view in views
    }
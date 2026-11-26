"""Pure specification objects for the JAX backend.

The GPU implementation is the oracle, but ``hamic_jax`` never imports it. This module
therefore owns the small amount of shared vocabulary (preset fields, special-token ids,
training fields) needed by the TPU path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "Architecture",
    "ModelPreset",
    "SpecialTokens",
    "TrainingConfig",
    "read_yaml",
]

SUPPORTED_ARCHITECTURES = ("modernbert", "llama", "qwen3")


class Architecture(str):
    """Architecture names accepted by the JAX model."""

    MODERNBERT = "modernbert"
    LLAMA = "llama"
    QWEN3 = "qwen3"


@dataclass(frozen=True)
class SpecialTokens:
    """HAMIC special tokens.

    The order is fixed by ``src/hamic/tokenization/spec.py``: bos, pad, eos, unk, mask,
    then 32 sentinels. Regular vocabulary starts after the sentinels.
    """

    bos: str = "<s>"
    pad: str = "<pad>"
    eos: str = "</s>"
    unk: str = "<unk>"
    mask: str = "<mask>"
    sentinel_template: str = "<extra_id_{index}>"
    num_sentinels: int = 32

    @property
    def sentinels(self) -> tuple[str, ...]:
        """The sentinel token strings in id order."""
        return tuple(self.sentinel_template.format(index=i) for i in range(self.num_sentinels))

    @property
    def ordered(self) -> tuple[str, ...]:
        """All special tokens in id order."""
        return (self.bos, self.pad, self.eos, self.unk, self.mask, *self.sentinels)

    @property
    def bos_id(self) -> int:
        """Id of the begin-of-sequence token."""
        return 0

    @property
    def pad_id(self) -> int:
        """Id of the padding token."""
        return 1

    @property
    def eos_id(self) -> int:
        """Id of the end-of-sequence token."""
        return 2

    @property
    def mask_id(self) -> int:
        """Id of the mask token."""
        return 4

    @property
    def special_ids(self) -> tuple[int, ...]:
        """Every reserved id, including sentinels."""
        return tuple(range(len(self.ordered)))

    @property
    def first_regular_id(self) -> int:
        """First id that may be masked or sampled as a random replacement."""
        return len(self.ordered)


SPECIAL_TOKENS = SpecialTokens()


@dataclass(frozen=True)
class ModelPreset:
    """A resolved HAMIC model shape.

    Args:
        name: Human-readable preset name.
        architecture: ``modernbert``, ``llama``, or ``qwen3``.
        num_layers: Transformer blocks.
        hidden_size: Residual width.
        num_heads: Query heads.
        num_kv_heads: Key/value heads (equal to ``num_heads`` for ModernBERT).
        head_dim: Per-head width; defaults to ``hidden_size // num_heads``.
        intermediate_size: MLP width.
        vocab_size: Tokenizer size.
        max_position_embeddings: Context length used by RoPE.
        rope_theta: RoPE base.
        norm_eps: RMSNorm epsilon (decoder families only).
        tie_word_embeddings: Whether the output head reuses the input embedding.
        attention_bias: Whether q/k/v/o projections carry biases (always false in HAMIC).
        mlp_bias: Whether MLP projections carry biases (always false in HAMIC).
        local_attention: ModernBERT local-attention window; ``None`` disables.
        sliding_window: ModernBERT sliding window value recorded for metadata.
        sparse_prediction: Training-only ModernBERT shortcut; false at export.
    """

    name: str
    architecture: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    local_attention: int | None = 128
    sliding_window: int | None = 64
    sparse_prediction: bool = False
    special_tokens: SpecialTokens = field(default=SPECIAL_TOKENS, repr=False)

    def __post_init__(self) -> None:
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {self.architecture!r}")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.attention_bias or self.mlp_bias:
            raise ValueError("HAMIC presets use bias-free projections")

    @property
    def is_encoder(self) -> bool:
        """Whether this preset is a bidirectional masked-LM encoder."""
        return self.architecture == Architecture.MODERNBERT

    @property
    def num_parameters(self) -> int:
        """Exact parameter count of the flat tree, counting tied weights once."""
        from hamic_jax.params import count_parameters

        return count_parameters(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view."""
        return {
            "name": self.name,
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
            "mlp_bias": self.mlp_bias,
            "local_attention": self.local_attention,
            "sliding_window": self.sliding_window,
            "sparse_prediction": self.sparse_prediction,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Fields needed to reproduce a HAMIC optimization trajectory.

    Unknown YAML keys are rejected so a typo cannot silently change a run.
    """

    run_id: str
    model_preset: str
    tokenized: str
    objective: str
    seq_len: int
    micro_batch_size: int
    grad_accum_steps: int
    total_tokens: float
    learning_rate: float
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1234
    mask_rate: float = 0.15
    mask_rate_final: float | None = None
    mask_schedule_start: float = 0.0
    lr_schedule: str = "cosine"
    decay_ratio: float = 0.2
    cross_document_attention: bool = True
    source_weights: Mapping[str, float] = field(default_factory=dict)
    precision: str = "bf16"
    eval_sequences: int = 256
    eval_mask_rate: float | None = None
    checkpoint_every_minutes: float = 45.0
    checkpoint_every_steps: int = 0
    log_every_steps: int = 10
    eval_every_steps: int = 500
    sparse_prediction: bool = False
    attention: str = "sdpa"
    masking: str = "token"
    span_max_length: int = 5
    pmi_unit_share: float = 1.0
    pmi_max_tokens: int = 100_000_000
    pmi_min_count: int = 50
    pmi_min_pmi: float = 3.0
    pmi_max_pairs: int = 200_000
    pmi_max_ngram: int = 3
    pmi_min_doc_count: int = 5
    pmi_max_extension_share: float = 0.6
    pmi_discount_k: float = 0.0
    pmi_chunk_tokens: int | None = None
    pmi_max_unit_length: int | None = None
    whole_word_masking: bool = False
    variant_share: float = 0.0
    shuffle: bool = True
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.objective not in ("mlm", "clm"):
            raise ValueError("objective must be 'mlm' or 'clm'")
        if self.lr_schedule not in ("cosine", "wsd"):
            raise ValueError("lr_schedule must be 'cosine' or 'wsd'")
        if self.masking not in ("token", "span", "pmi", "pmi_ngram"):
            raise ValueError("unsupported masking policy")
        if not 2 <= self.pmi_max_ngram <= 3:
            raise ValueError("pmi_max_ngram must be 2 or 3")
        if not 0 < self.decay_ratio <= 1:
            raise ValueError("decay_ratio must be in (0, 1]")
        if self.seq_len <= 0 or self.micro_batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch geometry must be positive")

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed by one optimizer step on one data-parallel rank."""
        return self.seq_len * self.micro_batch_size * self.grad_accum_steps

    @property
    def total_steps(self) -> int:
        """Number of optimizer steps in the run (ceiling, matching the oracle)."""
        return max(1, math.ceil(self.total_tokens / self.tokens_per_step))

    @property
    def warmup_steps(self) -> int:
        """Linear warmup length, at least one step."""
        return max(1, round(self.total_steps * self.warmup_ratio))

    def identity(self) -> dict[str, Any]:
        """Fields that define the trajectory and therefore gate strict resume."""
        return {
            "run_id": self.run_id,
            "model_preset": self.model_preset,
            "tokenized": self.tokenized,
            "objective": self.objective,
            "seq_len": self.seq_len,
            "micro_batch_size": self.micro_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "total_tokens": self.total_tokens,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "min_lr_ratio": self.min_lr_ratio,
            "betas": list(self.betas),
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "lr_schedule": self.lr_schedule,
            "decay_ratio": self.decay_ratio,
            "source_weights": dict(sorted(self.source_weights.items())),
        }


_MODEL_KEYS = {
    "name",
    "architecture",
    "label_parameters",
    "label_tolerance",
    "num_layers",
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "norm_eps",
    "local_attention",
    "sliding_window",
    "tie_word_embeddings",
    "sparse_prediction",
}

_TRAIN_KEYS = {
    "version",
    "run_id",
    "model_preset",
    "tokenized",
    "objective",
    "mask_rate",
    "mask_rate_final",
    "mask_schedule_start",
    "seq_len",
    "micro_batch_size",
    "grad_accum_steps",
    "total_tokens",
    "learning_rate",
    "warmup_ratio",
    "min_lr_ratio",
    "betas",
    "weight_decay",
    "grad_clip",
    "precision",
    "seed",
    "device",
    "compile",
    "sparse_prediction",
    "attention",
    "lr_schedule",
    "decay_ratio",
    "cross_document_attention",
    "source_weights",
    "masking",
    "pmi_unit_share",
    "span_max_length",
    "pmi_max_tokens",
    "pmi_min_count",
    "pmi_min_pmi",
    "pmi_max_pairs",
    "pmi_discount_k",
    "pmi_chunk_tokens",
    "pmi_max_unit_length",
    "pmi_max_ngram",
    "pmi_min_doc_count",
    "pmi_max_extension_share",
    "whole_word_masking",
    "variant_share",
    "shuffle",
    "deterministic",
    "log_every_steps",
    "eval_every_steps",
    "eval_sequences",
    "eval_mask_rate",
    "checkpoint_every_minutes",
    "checkpoint_every_steps",
}


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping without importing the GPU package."""
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{where} has unknown keys: {unknown}")


def model_preset_from_mapping(data: Mapping[str, Any]) -> ModelPreset:
    """Build a :class:`ModelPreset` from a parsed model YAML mapping."""
    _reject_unknown(data, _MODEL_KEYS, "model preset")
    architecture = str(data["architecture"])
    supported = {Architecture.MODERNBERT, Architecture.LLAMA, Architecture.QWEN3}
    if architecture not in supported:
        raise ValueError(
            f"unsupported architecture {architecture!r}; the JAX backend implements "
            f"{sorted(supported)} (legacy RoBERTa pilots are out of v1 scope)"
        )
    hidden = int(data["hidden_size"])
    heads = int(data["num_heads"])
    head_dim = int(data.get("head_dim", hidden // heads))
    if architecture == Architecture.MODERNBERT:
        kv_heads = heads
    else:
        if "num_kv_heads" not in data:
            raise ValueError("decoder presets must declare num_kv_heads")
        kv_heads = int(data["num_kv_heads"])
    return ModelPreset(
        name=str(data["name"]),
        architecture=architecture,
        num_layers=int(data["num_layers"]),
        hidden_size=hidden,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=int(data["intermediate_size"]),
        vocab_size=int(data["vocab_size"]),
        max_position_embeddings=int(data["max_position_embeddings"]),
        rope_theta=float(data.get("rope_theta", 10_000.0)),
        norm_eps=float(
            data.get(
                "norm_eps",
                1e-5 if architecture == Architecture.MODERNBERT else 1e-6,
            )
        ),
        tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
        local_attention=(
            int(data["local_attention"])
            if data.get("local_attention") is not None
            else (128 if architecture == Architecture.MODERNBERT else None)
        ),
        sliding_window=(
            int(data["sliding_window"])
            if data.get("sliding_window") is not None
            else (64 if architecture == Architecture.MODERNBERT else None)
        ),
        sparse_prediction=bool(data.get("sparse_prediction", False)),
    )


def training_config_from_mapping(data: Mapping[str, Any]) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from a parsed train YAML mapping."""
    _reject_unknown(data, _TRAIN_KEYS, "train config")
    betas = tuple(float(v) for v in data.get("betas", (0.9, 0.95)))
    if len(betas) != 2:
        raise ValueError("betas must have two values")
    weights = {str(k): float(v) for k, v in (data.get("source_weights") or {}).items()}
    return TrainingConfig(
        run_id=str(data["run_id"]),
        model_preset=str(data["model_preset"]),
        tokenized=str(data["tokenized"]),
        objective=str(data["objective"]),
        seq_len=int(data["seq_len"]),
        micro_batch_size=int(data["micro_batch_size"]),
        grad_accum_steps=int(data["grad_accum_steps"]),
        total_tokens=float(data["total_tokens"]),
        learning_rate=float(data["learning_rate"]),
        warmup_ratio=float(data.get("warmup_ratio", 0.02)),
        min_lr_ratio=float(data.get("min_lr_ratio", 0.1)),
        betas=(betas[0], betas[1]),
        weight_decay=float(data.get("weight_decay", 0.1)),
        grad_clip=float(data.get("grad_clip", 1.0)),
        seed=int(data.get("seed", 1234)),
        mask_rate=float(data.get("mask_rate", 0.15)),
        mask_rate_final=(
            float(data["mask_rate_final"]) if data.get("mask_rate_final") is not None else None
        ),
        mask_schedule_start=float(data.get("mask_schedule_start", 0.0)),
        lr_schedule=str(data.get("lr_schedule", "cosine")),
        decay_ratio=float(data.get("decay_ratio", 0.2)),
        cross_document_attention=bool(data.get("cross_document_attention", True)),
        source_weights=weights,
        precision=str(data.get("precision", "bf16")),
        eval_sequences=int(data.get("eval_sequences", 256)),
        eval_mask_rate=(
            float(data["eval_mask_rate"])
            if data.get("eval_mask_rate") is not None
            else None
        ),
        checkpoint_every_minutes=float(data.get("checkpoint_every_minutes", 45.0)),
        checkpoint_every_steps=int(data.get("checkpoint_every_steps", 0)),
        log_every_steps=int(data.get("log_every_steps", 10)),
        eval_every_steps=int(data.get("eval_every_steps", 500)),
        sparse_prediction=bool(data.get("sparse_prediction", False)),
        attention=str(data.get("attention", "sdpa")),
        masking=str(data.get("masking", "token")),
        span_max_length=int(data.get("span_max_length", 5)),
        pmi_unit_share=float(data.get("pmi_unit_share", 1.0)),
        pmi_max_tokens=int(data.get("pmi_max_tokens", 100_000_000)),
        pmi_min_count=int(data.get("pmi_min_count", 50)),
        pmi_min_pmi=float(data.get("pmi_min_pmi", 3.0)),
        pmi_max_pairs=int(data.get("pmi_max_pairs", 200_000)),
        pmi_max_ngram=int(data.get("pmi_max_ngram", 3)),
        pmi_min_doc_count=int(data.get("pmi_min_doc_count", 5)),
        pmi_max_extension_share=float(data.get("pmi_max_extension_share", 0.6)),
        pmi_discount_k=float(data.get("pmi_discount_k", 0.0)),
        pmi_chunk_tokens=(
            int(data["pmi_chunk_tokens"]) if data.get("pmi_chunk_tokens") is not None else None
        ),
        pmi_max_unit_length=(
            int(data["pmi_max_unit_length"])
            if data.get("pmi_max_unit_length") is not None
            else None
        ),
        whole_word_masking=bool(data.get("whole_word_masking", False)),
        variant_share=float(data.get("variant_share", 0.0)),
        shuffle=bool(data.get("shuffle", True)),
        deterministic=bool(data.get("deterministic", False)),
    )
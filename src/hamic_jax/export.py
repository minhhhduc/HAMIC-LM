"""Export a JAX parameter tree as a stock Hugging Face model directory.

No torch call is made. Safetensors is written through ``safetensors.numpy``; the config
is the JSON that the pinned Transformers classes expect. Tied heads are materialized
twice because safetensors has no aliasing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from safetensors.numpy import load_file, save_file

from hamic_jax.params import state_dict, validate_params
from hamic_jax.spec import Architecture, ModelPreset

__all__ = ["export_pretrained", "import_weights", "model_config"]

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def model_config(preset: ModelPreset) -> dict[str, Any]:
    """Return the Transformers configuration for a preset."""
    special = preset.special_tokens
    common: dict[str, Any] = {
        "vocab_size": preset.vocab_size,
        "hidden_size": preset.hidden_size,
        "intermediate_size": preset.intermediate_size,
        "num_hidden_layers": preset.num_layers,
        "num_attention_heads": preset.num_heads,
        "max_position_embeddings": preset.max_position_embeddings,
        "tie_word_embeddings": preset.tie_word_embeddings,
        "bos_token_id": special.bos_id,
        "eos_token_id": special.eos_id,
        "pad_token_id": special.pad_id,
        "cls_token_id": special.bos_id,
        "sep_token_id": special.eos_id,
        "mask_token_id": special.mask_id,
        "dtype": "bfloat16",
        "transformers_version": "5.17.0",
    }
    if preset.architecture == Architecture.MODERNBERT:
        return {
            "architectures": ["ModernBertForMaskedLM"],
            "model_type": "modernbert",
            **common,
            "num_key_value_heads": preset.num_kv_heads,
            "head_dim": preset.head_dim,
            "hidden_activation": "gelu",
            "classifier_activation": "gelu",
            "classifier_bias": False,
            "decoder_bias": True,
            "norm_bias": False,
            "attention_bias": False,
            "mlp_bias": False,
            "norm_eps": preset.norm_eps,
            "local_attention": preset.local_attention,
            "sliding_window": preset.sliding_window,
            "layer_types": ["full_attention"] * preset.num_layers,
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": preset.rope_theta,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": preset.rope_theta,
                },
            },
            "sparse_prediction": False,
            "sparse_pred_ignore_index": -100,
        }
    if preset.architecture == Architecture.QWEN3:
        return {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            **common,
            "num_key_value_heads": preset.num_kv_heads,
            "head_dim": preset.head_dim,
            "hidden_act": "silu",
            "rms_norm_eps": preset.norm_eps,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": preset.rope_theta,
            },
            "attention_bias": False,
            "mlp_bias": False,
            "use_sliding_window": False,
            "sliding_window": None,
            "layer_types": ["full_attention"] * preset.num_layers,
        }
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        **common,
        "num_key_value_heads": preset.num_kv_heads,
        "head_dim": preset.head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": preset.norm_eps,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": preset.rope_theta,
        },
        "attention_bias": False,
        "mlp_bias": False,
    }


def export_pretrained(
    preset: ModelPreset,
    params: Mapping[str, Any],
    output_dir: str | Path,
    tokenizer_dir: str | Path | None = None,
    lineage: Mapping[str, Any] | None = None,
) -> Path:
    """Write safetensors, ``config.json``, tokenizer files, and ``hamic_model.json``."""
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"{out} already exists; exports are immutable")
    tree = {name: np.asarray(value) for name, value in params.items()}
    validate_params(tree, preset)
    exported = {name: np.asarray(value) for name, value in state_dict(tree, preset).items()}
    out.mkdir(parents=True)
    save_file(exported, out / "model.safetensors")
    (out / "config.json").write_text(
        json.dumps(model_config(preset), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if tokenizer_dir is not None:
        source = Path(tokenizer_dir)
        for name in TOKENIZER_FILES:
            candidate = source / name
            if candidate.is_file():
                shutil.copyfile(candidate, out / name)
        metadata = source / "hamic_tokenizer.json"
        if metadata.is_file():
            shutil.copyfile(metadata, out / metadata.name)
    info = {
        "name": preset.name,
        "preset": preset.to_dict(),
        "parameters": int(sum(value.size for value in tree.values())),
        "lineage": dict(lineage or {}),
    }
    (out / "hamic_model.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    special = preset.special_tokens
    generation = {
        "bos_token_id": special.bos_id,
        "eos_token_id": special.eos_id,
        "pad_token_id": special.pad_id,
    }
    (out / "generation_config.json").write_text(
        json.dumps(generation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def import_weights(path: str | Path, preset: ModelPreset) -> dict[str, np.ndarray[Any, Any]]:
    """Load a safetensors ``state_dict`` and keep only the unique leaves.

    Tied aliases are checked for equality and then dropped, because the JAX tree stores
    one copy.
    """
    file = Path(path)
    if file.is_dir():
        file = file / "model.safetensors"
    values = load_file(file)
    tree: dict[str, np.ndarray[Any, Any]] = {}
    from hamic_jax.params import parameter_shapes, tied_aliases

    expected = parameter_shapes(preset)
    for name in expected:
        if name not in values:
            raise KeyError(f"missing weight: {name}")
        tree[name] = values[name]
    for alias, source in tied_aliases(preset).items():
        if alias in values and not np.array_equal(values[alias], values[source]):
            raise ValueError(f"tied alias {alias} differs from {source}")
    validate_params(tree, preset)
    return tree
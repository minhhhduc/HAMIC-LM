"""Strict loaders for the HAMIC YAML files.

The JAX backend reads the same files as the GPU backend but parses them itself, so it
works in an environment where ``hamic`` and ``torch`` are not installed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hamic_jax.spec import (
    ModelPreset,
    TrainingConfig,
    model_preset_from_mapping,
    read_yaml,
    training_config_from_mapping,
)

__all__ = [
    "PRESET_FILES",
    "available_presets",
    "config_root",
    "load_preset",
    "load_training_config",
]

PRESET_FILES = {
    "hamic-es": "hamic-es.yaml",
    "hamic-em": "hamic-em.yaml",
    "hamic-d500m": "hamic-d500m.yaml",
    "hamic-d500m-m": "hamic-d500m-m.yaml",
    "hamic-d1b": "hamic-d1b.yaml",
    "hamic-d1b-m": "hamic-d1b-m.yaml",
}


def config_root() -> Path:
    """Return the repository's ``configs`` directory."""
    root = Path(__file__).resolve().parents[2]
    candidate = root / "configs"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Cannot find configs/ next to {root}")
    return candidate


def _repo_root() -> Path:
    return config_root().parent


def load_preset(name_or_path: str | Path) -> ModelPreset:
    """Load a model preset by name (e.g. ``hamic-es``) or YAML path.

    Raises:
        FileNotFoundError: If neither a file nor a known preset matches.
        ValueError: If the YAML contains unknown keys or invalid geometry.
    """
    text = str(name_or_path)
    path = Path(text)
    if not path.is_file():
        filename = PRESET_FILES.get(text, f"{text}.yaml")
        path = config_root() / "models" / filename
    if not path.is_file():
        known = ", ".join(sorted(PRESET_FILES))
        raise FileNotFoundError(f"No preset {text!r}; known presets: {known}")
    return model_preset_from_mapping(read_yaml(path))


def available_presets() -> dict[str, Path]:
    """Return the known preset names and their YAML paths."""
    return {
        name: config_root() / "models" / filename
        for name, filename in sorted(PRESET_FILES.items())
    }


def load_training_config(path: str | Path) -> tuple[TrainingConfig, ModelPreset]:
    """Load a run configuration and its referenced model preset.

    ``model_preset`` is repository-relative in every HAMIC YAML; it is resolved from the
    repository root, not from the caller's working directory.
    """
    source = Path(path)
    config = training_config_from_mapping(read_yaml(source))
    tokenized = Path(config.tokenized)
    if not tokenized.is_absolute():
        config = replace(config, tokenized=str(_repo_root() / tokenized))
    preset_path = Path(config.model_preset)
    if not preset_path.is_absolute():
        preset_path = _repo_root() / preset_path
    return config, load_preset(preset_path)
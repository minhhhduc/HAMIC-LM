"""Runtime dependency checks for the TPU notebooks.

The notebooks import this module with the standard library only. It installs the
third-party packages that ``hamic_jax`` needs when they are absent, using pip's Python
entry point without spawning a child process.
"""

from __future__ import annotations

from collections.abc import Mapping

_BASE = {
    "optax": "optax",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
    "yaml": "pyyaml",
    "tokenizers": "tokenizers",
    "matplotlib": "matplotlib",
    "ipywidgets": "ipywidgets",
    "pandas": "pandas",
    "tqdm": "tqdm",
}
_DOWNSTREAM = {
    "transformers": "transformers",
    "datasets": "datasets",
}


def _missing(dependencies: Mapping[str, str]) -> list[str]:
    """Return distributions whose import module is not available."""
    import importlib.util

    return [
        distribution
        for module, distribution in dependencies.items()
        if importlib.util.find_spec(module) is None
    ]


def ensure_dependencies(*, downstream: bool = False) -> list[str]:
    """Install any missing runtime distributions and return what was installed.

    Args:
        downstream: Also require the Hugging Face fine-tuning dependencies.

    Returns:
        Distribution names passed to pip. Empty when the environment is already complete.

    Raises:
        RuntimeError: pip is unavailable or refuses the installation.
    """
    dependencies = dict(_BASE)
    if downstream:
        dependencies.update(_DOWNSTREAM)
    missing = _missing(dependencies)
    if not missing:
        return []
    try:
        from pip._internal.cli.main import main as pip_main
    except ImportError as exc:  # pragma: no cover - pip is present on Colab/TPU images
        raise RuntimeError(
            "pip is unavailable; install the missing distributions manually: "
            + ", ".join(missing)
        ) from exc
    code = pip_main(
        [
            "install",
            "--quiet",
            "--disable-pip-version-check",
            *missing,
        ]
    )
    if code:
        raise RuntimeError(f"pip could not install {missing}")
    import importlib

    importlib.invalidate_caches()
    still_missing = _missing(dependencies)
    if still_missing:
        raise RuntimeError(f"installed but still unimportable: {still_missing}")
    return missing


__all__ = ["ensure_dependencies"]
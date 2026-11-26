"""The JAX package must import without torch or the GPU package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("optax")

SOURCE = Path(__file__).resolve().parents[2] / "src"


def test_import_does_not_pull_torch_or_hamic() -> None:
    script = """
import sys
import hamic_jax
import hamic_jax.checkpoint
import hamic_jax.data
import hamic_jax.downstream
import hamic_jax.export
import hamic_jax.model
import hamic_jax.objectives
import hamic_jax.optim
import hamic_jax.params
import hamic_jax.presets
import hamic_jax.session
import hamic_jax.sharding
import hamic_jax.train
assert "torch" not in sys.modules, "torch was imported"
assert "hamic" not in sys.modules, "the GPU package was imported"
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
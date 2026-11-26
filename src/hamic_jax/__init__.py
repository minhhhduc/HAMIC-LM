"""JAX/TPU backend for HAMIC.

This package is deliberately independent of ``hamic`` and ``torch``. It reads HAMIC's
configuration and tokenized files directly, and it writes the same Hugging Face
``state_dict`` names on export.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0.dev0+tpu"
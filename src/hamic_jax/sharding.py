"""Device layout for one TPU host running data-parallel training.

Every optimizer step shards the packed batch over the local chips. Parameters and
optimizer state are replicated: a v5e-8 has 16 GiB per chip, which holds the full
fp32 state of every public pretraining preset up to ``hamic-d500m``. The batch is
split only along the ``data`` mesh axis, and each shard computes the same gradient
after an explicit all-reduce, so the update is identical to a single-device run of
the same global batch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

__all__ = ["DeviceLayout", "device_label", "peak_memory_gib"]


@dataclass(frozen=True)
class DeviceLayout:
    """A one-dimensional data-parallel mesh and its named shardings."""

    mesh: Mesh
    data_parallel: int

    @classmethod
    def for_batch(
        cls,
        batch_size: int,
        devices: Sequence[Any] | None = None,
    ) -> DeviceLayout:
        """Use the largest divisor of ``batch_size`` among the available chips.

        A batch that is smaller than the chip count (a drill, or an evaluation batch)
        keeps every chip busy only when it divides evenly. Uneven sharding would pad
        the batch and change the loss, so the remainder stays idle instead.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        available = tuple(jax.local_devices() if devices is None else devices)
        if not available:
            raise RuntimeError("no JAX devices are visible")
        count = next(
            (size for size in range(len(available), 0, -1) if batch_size % size == 0),
            1,
        )
        array = np.asarray(available[:count], dtype=object)
        return cls(mesh=Mesh(array, ("data",)), data_parallel=count)

    @property
    def param_sharding(self) -> NamedSharding:
        """Replicated sharding for parameters and optimizer state."""
        return NamedSharding(self.mesh, P())

    @property
    def batch_sharding(self) -> NamedSharding:
        """Sharding for ``[batch, seq]`` inputs: split the batch over the mesh."""
        return NamedSharding(self.mesh, P(None, "data"))

    @property
    def stack_sharding(self) -> NamedSharding:
        """Sharding for ``[accum, batch, seq]`` stacks: split only the batch axis."""
        return NamedSharding(self.mesh, P(None, "data", None))

    def replicate(self, tree: Any) -> Any:
        """Place every array leaf on all chips with the replicated sharding."""
        return jax.tree.map(
            lambda leaf: jax.device_put(leaf, self.param_sharding),
            tree,
        )

    def replicated_spec(self, tree: Any) -> Any:
        """Replicated ``PartitionSpec`` for array leaves; ``None`` for static leaves.

        ``shard_map`` treats a ``None`` spec as a static argument. Optax state normally
        holds only arrays, but a transform may add a static leaf; mapping that leaf to
        ``PartitionSpec()`` would make ``shard_map`` try to shard it and fail.
        """
        return jax.tree.map(
            lambda leaf: P() if isinstance(leaf, (jax.Array, np.ndarray)) else None,
            tree,
        )

    def place_batch(self, rows: Any) -> jax.Array:
        """Place one host batch on the mesh, sharded along its first axis."""
        array = rows if isinstance(rows, jax.Array) else jnp.asarray(rows, dtype=jnp.int32)
        return jax.device_put(array, self.batch_sharding)

    def place_stack(self, rows: Any) -> jax.Array:
        """Place ``[accum, batch, seq]`` training input, splitting the batch axis."""
        array = rows if isinstance(rows, jax.Array) else jnp.asarray(rows, dtype=jnp.int32)
        return jax.device_put(array, self.stack_sharding)


def device_label() -> str:
    """Short device name for metric records, such as ``v5e-8`` or ``cpu-1``."""
    devices = jax.local_devices()
    backend = jax.default_backend()
    kind = str(devices[0].device_kind) if devices else backend
    if backend == "tpu":
        lowered = kind.lower()
        # JAX reports the v5e and v6e chips as "TPU v5 lite" and "TPU v6 lite".
        name = "v5e" if "v5 lite" in lowered else "v6e" if "v6 lite" in lowered else None
        if name is None:
            match = re.search(r"v\d+[a-z]?", lowered)
            name = match.group(0) if match is not None else "tpu"
        return f"{name}-{len(devices)}"
    return f"{backend}-{len(devices)}"


def peak_memory_gib() -> float | None:
    """Peak device memory in GiB when the backend exposes memory statistics."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except (RuntimeError, ValueError, NotImplementedError):
        return None
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use", stats.get("bytes_in_use"))
    return round(float(peak) / 2**30, 2) if peak else None
"""CPU checks for the mesh helpers that the TPU trainer depends on."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from hamic_jax.sharding import DeviceLayout, device_label, peak_memory_gib


def test_layout_picks_a_divisor_and_rejects_zero() -> None:
    with pytest.raises(ValueError):
        DeviceLayout.for_batch(0)
    layout = DeviceLayout.for_batch(6)
    assert 6 % layout.data_parallel == 0
    assert layout.mesh.devices.size == layout.data_parallel


def test_stack_sharding_keeps_the_accumulation_axis_replicated() -> None:
    """Sharding the accumulation axis would give each chip a different micro-step."""
    layout = DeviceLayout.for_batch(6)
    assert layout.stack_sharding.spec == P(None, "data", None)
    stacked = layout.place_stack(np.zeros((2, 6, 4), np.int32))
    assert stacked.shape == (2, 6, 4)
    assert stacked.sharding.spec == P(None, "data", None)


def test_replicated_spec_marks_non_array_leaves_static() -> None:
    layout = DeviceLayout.for_batch(6)
    tree = {"weight": jnp.ones((2, 2), jnp.float32), "empty": (), "none": None}
    spec = layout.replicated_spec(tree)
    assert spec["weight"] == P()
    assert spec["empty"] == ()
    assert spec["none"] is None


def test_device_label_and_memory_are_reported() -> None:
    label = device_label()
    assert label and "-" in label
    memory = peak_memory_gib()
    assert memory is None or memory >= 0.0

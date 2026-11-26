"""Parameter counts are execution-verified; the JAX shapes must reproduce them exactly."""

from __future__ import annotations

import pytest

from hamic_jax.params import count_parameters, parameter_shapes
from hamic_jax.presets import load_preset

MEASURED = {
    "hamic-es": 150_906_432,
    "hamic-em": 348_796_992,
    "hamic-d500m": 486_176_000,
    "hamic-d500m-m": 521_828_608,
    "hamic-d1b": 1_021_143_040,
    "hamic-d1b-m": 1_011_064_832,
}


@pytest.mark.parametrize(("name", "expected"), MEASURED.items())
def test_parameter_count(name: str, expected: int) -> None:
    assert count_parameters(load_preset(name)) == expected


def test_shapes_are_unique_and_tied_weights_are_absent() -> None:
    preset = load_preset("hamic-es")
    shapes = parameter_shapes(preset)
    assert len(shapes) == len(set(shapes))
    assert "decoder.weight" not in shapes
    assert "head.dense.bias" not in shapes
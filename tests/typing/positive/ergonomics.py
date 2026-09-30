"""Static types of scaling, sums, Dimensionless numbers, parse and reinterpret."""

from typing import Any, Literal, assert_type

import jax
import numpy as np
import numpy.typing as npt
import torch

from quantype import (
    Dimensionless,
    EnergyDensity,
    Frequency,
    InverseTime,
    Length,
    Pressure,
    Time,
    u,
)
from quantype.core import Unit
from quantype.kinds import LengthKind
from quantype.systems import SI

length = 2 * u.nm
step = Time[float](0.5, u.fs)
floats: npt.NDArray[np.float64] = np.array([1.0, 2.0])
positions = Length[npt.NDArray[np.float32]]([1, 2], u.angstrom)

# Any real scalar keeps the storage; arrays make or keep array storage.
assert_type(length * np.float32(2), Length[float])
assert_type(np.float64(2) * length, Length[float])
assert_type(length / np.int64(2), Length[float])
assert_type(step * np.arange(3), Time[npt.NDArray[np.float64]])
assert_type(step * floats, Time[npt.NDArray[np.float64]])
assert_type(positions * floats, Length[npt.NDArray[np.float32]])
assert_type(1 / step, InverseTime[float])
# An array on the left is typed by NumPy's own stubs, which return Any.

# Builtin sum() starts from zero, which an empty iterable returns.
assert_type(sum([length, 2 * u.nm]), Length[float] | Literal[0])

# Dimensionless values take plain numbers.
ratio = (1 * u.eV) / (4 * u.eV)
assert_type(ratio + 1, Dimensionless[float])
assert_type(1 - ratio, Dimensionless[float])
assert_type(ratio < 1, bool)
as_float: float = float(ratio)

# The presented unit, and naming equal dimensions explicitly.
named_unit: Unit[LengthKind] = length.unit
unnamed_unit: Unit[Any] | None = (length * step).unit
density = (1 * u.eV) / (1 * u.angstrom_cubed)
assert_type(density, EnergyDensity[float])
assert_type(Pressure.reinterpret(density), Pressure[float])
assert_type(Frequency.reinterpret(1 / step), Frequency[float])
assert_type(Length.reinterpret(Length[float, SI](1, u.nm)), Length[float, SI])

# A string holds one number.
assert_type(Length.parse("5 nm"), Length[float])
wire = {"kind": "Length", "magnitude": 1.0, "unit": "nm"}
assert_type(Length.parse(wire), Length[float | npt.NDArray[np.float64]])

# Format specs apply to the magnitude.
assert_type(f"{length:.3f}", str)


def backend_arrays(values: jax.Array, tensor: torch.Tensor) -> None:
    assert_type(length * values, Length[jax.Array])
    assert_type(length * tensor, Length[torch.Tensor])
    held = Length[jax.Array](values, u.angstrom)
    assert_type(held * values, Length[jax.Array])
    assert_type(held * 2.0, Length[jax.Array])

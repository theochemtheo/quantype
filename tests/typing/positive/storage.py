"""A quantity type's NumPy form, read through quantype.typing.SupportsToNumpy."""

from typing import assert_type

import numpy as np
import numpy.typing as npt

from quantype import Energy, Force, Length, Quantity, u
from quantype.kinds import EnergyKind
from quantype.products import LengthEnergy
from quantype.systems import ASE, SI, NoSystem
from quantype.typing import SupportsToNumpy

type Array = npt.NDArray[np.float64]


def numpy_form[B](value: SupportsToNumpy[B] | None) -> B:
    assert value is not None
    return value.to_numpy()


energy = Energy.from_value(1.5, ASE)
assert_type(numpy_form(energy), Energy[Array, ASE])
assert_type(numpy_form(Energy[float](1.5, u.eV)), Energy[Array, NoSystem])
forces = Force.from_value(np.zeros(3), SI)
assert_type(numpy_form(forces), Force[Array, SI])
product = Length.from_value(2.0, SI) * Energy.from_value(3.0, SI)
assert_type(numpy_form(product), LengthEnergy[Array, SI])


def from_base(quantity: Quantity[EnergyKind, float, ASE]) -> None:
    assert_type(numpy_form(quantity), Quantity[EnergyKind, Array, ASE])


def from_optional(quantity: Energy[float, ASE] | None) -> None:
    assert_type(numpy_form(quantity), Energy[Array, ASE])

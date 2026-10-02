"""These assert_type checks are the portable public API specification."""

from collections.abc import Callable
from typing import assert_type

import jax
import numpy as np
import numpy.typing as npt
import torch

import quantype.numpy as qnp
from quantype import (
    Area,
    Dimensionless,
    Energy,
    EnergyDensity,
    EnergyPerAtom,
    Force,
    ForceConstant,
    Length,
    Pressure,
    Temperature,
    TemperatureDifference,
    TemperatureRate,
    Velocity,
    Volume,
    ujax,
    utorch,
)
from quantype import units as u
from quantype.products import EnergyPerTime, LengthTime

r = 2.0 * u.angstrom
t = 4.0 * u.fs
e = 3.0 * u.eV
assert_type(r, Length[float])
assert_type(2 * u.nm, Length[float])
assert_type(e.to(u.hartree), Energy[float])
assert_type(e.magnitude(u.eV), float)
assert_type(-4.2 * u.eV_per_atom, EnergyPerAtom[float])
assert_type(r**2, Area[float])
assert_type(r**3, Volume[float])
assert_type(r * r, Area[float])
assert_type((r * r) * r, Volume[float])
assert_type(r / t, Velocity[float])
assert_type(e / r, Force[float])
assert_type((e / r) * r, Energy[float])
assert_type((e / r) / (r**2), Pressure[float])
assert_type(e / (r**3), EnergyDensity[float])
assert_type(r * t, LengthTime[float])
assert_type(e / t, EnergyPerTime[float])

t1 = 300 * u.K
t2 = 280 * u.K
delta = t1 - t2
assert_type(delta, TemperatureDifference[float])
assert_type(t1 + delta, Temperature[float])
assert_type(delta + t1, Temperature[float])
assert_type(delta / (2 * u.s), TemperatureRate[float])

array: npt.NDArray[np.float64] = np.array([1.0, 2.0, 3.0])
positions = u.angstrom(array)
assert_type(positions, Length[npt.NDArray[np.float64]])
assert_type(u.angstrom * array, Length[npt.NDArray[np.float64]])
assert_type(positions.sum(), Length[npt.NDArray[np.float64]])
assert_type(positions.mean(), Length[npt.NDArray[np.float64]])
assert_type(qnp.sqrt(positions**2), Length[npt.NDArray[np.float64]])
assert_type(qnp.sin(u.degree(array)), Dimensionless[npt.NDArray[np.float64]])
assert_type(qnp.exp(u.one(array)), Dimensionless[npt.NDArray[np.float64]])


def harmonic_jax(x: Length[jax.Array]) -> Energy[jax.Array]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


def harmonic_torch(x: Length[torch.Tensor]) -> Energy[torch.Tensor]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


assert_type(ujax.grad(harmonic_jax), Callable[[Length[jax.Array]], Force[jax.Array]])
assert_type(
    ujax.hessian(harmonic_jax),
    Callable[[Length[jax.Array]], ForceConstant[jax.Array]],
)
tx = u.angstrom(torch.tensor([1.0, 2.0, 3.0], requires_grad=True))
assert_type(tx, Length[torch.Tensor])
assert_type(utorch.grad(harmonic_torch(tx), tx), Force[torch.Tensor])
assert_type(Energy.from_value(array), Energy[npt.NDArray[np.float64]])
assert_type(
    Length.parse({"kind": "Length", "magnitude": 1.0, "unit": "angstrom"}),
    Length[float | npt.NDArray[np.float64]],
)

"""Unit systems in the static type. Also executed at runtime to catch drift."""

from collections.abc import Callable
from typing import assert_type

import jax
import numpy as np
import numpy.typing as npt
import torch

from quantype import (
    Area,
    Energy,
    Force,
    ForceConstant,
    Length,
    Temperature,
    TemperatureDifference,
    Time,
    ujax,
    utorch,
)
from quantype import units as u
from quantype.systems import SI, Atomistic, Metal, Real, UnitSystem

kJ_per_mol = Energy.define_unit(  # noqa: N816 -- conventional unit spelling
    "typing:kJ_per_mol", reference=u.joule, scale=1e3 / 6.02214076e23
)


class Gromacs(UnitSystem, name="typing:gromacs"):
    length = u.nanometer
    energy = kJ_per_mol
    time = u.picosecond


class GromacsFs(Gromacs, name="typing:gromacs-fs"):
    time = u.femtosecond


# --- Construction and algebra within one system -----------------------------------

x_si = Length[float, SI](2.0, u.nm)
e_si = Energy[float, SI](3.0, u.eV)
x_metal = Length[float, Metal](20.0, u.angstrom)
assert_type(x_si, Length[float, SI])
assert_type(x_si + x_si, Length[float, SI])
assert_type(2.0 * x_si, Length[float, SI])
assert_type(x_si * x_si, Area[float, SI])
assert_type(x_si**2, Area[float, SI])
assert_type(e_si / x_si, Force[float, SI])
assert_type((e_si / x_si) * x_si, Energy[float, SI])
assert_type(x_si.system, type[SI])
assert_type(x_si.mean(), Length[float, SI])
assert_type(x_si.max(), Length[float, SI])
assert_type(x_si < 2.0 * x_si, bool)
assert_type(u.sqrt(x_si * x_si), Length[float, SI])
assert_type(Length[float, SI].from_value(2.0), Length[float, SI])
assert_type(Length[float, SI].parse("2 nm"), Length[float, SI])
point = Temperature[float, SI](20.0, u.celsius)
assert_type(point - point, TemperatureDifference[float, SI])

# --- The default system keeps today's annotations -----------------------------------

x_default = 2.0 * u.nm
assert_type(x_default, Length[float])
assert_type(x_default, Length[float, Atomistic])
assert_type(Length.from_value(2.0), Length[float])
assert_type(Length[float](2, u.nm), Length[float, Atomistic])


def cutoff_today(r: Length[float]) -> Length[float]:
    return r + r


cutoff_today(x_default)

# --- Generic code over the system ---------------------------------------------------


def work[S: UnitSystem](f: Force[float, S], d: Length[float, S]) -> Energy[float, S]:
    return f * d


assert_type(work(e_si / x_si, x_si), Energy[float, SI])
f_metal = Force[float, Metal](1.0, u.eV_per_angstrom)
assert_type(work(f_metal, x_metal), Energy[float, Metal])

# --- Explicit bridges and system-specific kernels -------------------------------------

assert_type(x_metal.to_system(SI), Length[float, SI])
assert_type(x_si + x_metal.to_system(SI), Length[float, SI])
positions: npt.NDArray[np.float64] = np.zeros(3)


def lammps_real_kernel(r: Length[npt.NDArray[np.float64], Real]) -> None:
    del r


lammps_real_kernel(Length[npt.NDArray[np.float64], Real](positions, u.angstrom))
lammps_real_kernel(Length[npt.NDArray[np.float64], SI](positions, u.nm).to_system(Real))

# --- Custom systems need no stub changes ----------------------------------------------

cutoff = Length[float, Gromacs](1.2, u.nm)
assert_type(cutoff + cutoff, Length[float, Gromacs])
assert_type(cutoff.to_system(GromacsFs), Length[float, GromacsFs])
raw: npt.NDArray[np.float32] = np.zeros((4, 3), dtype=np.float32)
forces = Force[npt.NDArray[np.float32], Gromacs].from_value(raw)
assert_type(forces, Force[npt.NDArray[np.float32], Gromacs])
assert_type(forces.to_system(Atomistic), Force[npt.NDArray[np.float32]])
assert_type(forces[0], Force[npt.NDArray[np.float32], Gromacs])

# --- Autodiff stays generic over the system -------------------------------------------


def harmonic_jax(x: Length[jax.Array, SI]) -> Energy[jax.Array, SI]:
    k = ForceConstant[float, SI](2.0, u.eV_per_angstrom_squared)
    return 0.5 * k * (x**2).sum()


def harmonic_torch(x: Length[torch.Tensor, Metal]) -> Energy[torch.Tensor, Metal]:
    k = ForceConstant[float, Metal](2.0, u.eV_per_angstrom_squared)
    return 0.5 * k * (x**2).sum()


assert_type(
    ujax.grad(harmonic_jax),
    Callable[[Length[jax.Array, SI]], Force[jax.Array, SI]],
)
assert_type(
    ujax.hessian(harmonic_jax),
    Callable[[Length[jax.Array, SI]], ForceConstant[jax.Array, SI]],
)
tx = Length[torch.Tensor, Metal](torch.tensor([1.0], requires_grad=True), u.angstrom)
assert_type(utorch.grad(harmonic_torch(tx), tx), Force[torch.Tensor, Metal])
step = Time[float, Metal](1.0, u.fs)
assert_type(step.to_system(SI), Time[float, SI])

"""Static types of quantype.numpy and the autodiff adapters."""

from collections.abc import Callable
from typing import Any, assert_type

import jax
import numpy as np
import numpy.typing as npt
import torch

import quantype.numpy as qnp
from quantype import (
    Angle,
    Area,
    Dimensionless,
    Energy,
    Force,
    ForceConstant,
    Length,
    Quantity,
    Temperature,
    TemperatureDifference,
    u,
    ujax,
    utorch,
)
from quantype.kinds import ForceKind, LengthKind, Mul
from quantype.systems import SI

type Array = npt.NDArray[np.float64]

angle = Angle[float](60, u.deg)
ratio = (1 * u.eV) / (2 * u.eV)
positions = Length[Array]([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], u.angstrom)
force = Force[Array]([1.0, 0.0, 0.0], u.eV_per_angstrom)

# Unit rules, typed with the catalogue's classes.
assert_type(qnp.cos(angle), Dimensionless[float])
assert_type(qnp.arccos(ratio), Angle[float])
assert_type(qnp.exp(ratio), Dimensionless[float])
assert_type(qnp.sqrt(Area[float, SI](4, u.angstrom_squared)), Length[float, SI])
assert_type(qnp.arctan2(1 * u.nm, 1 * u.nm), Angle[float])
assert_type(qnp.linalg.norm(positions, axis=-1), Length[Array])
assert_type(qnp.stack([positions, positions]), Length[Array])
assert_type(qnp.where(np.array([True, False]), positions, positions), Length[Array])
# pyright infers Any for dot of NumPy arrays (their shapes are Any); scalars are exact.
assert_type(qnp.dot(Force[float](1, u.eV_per_angstrom), 1 * u.nm), Energy[float])
assert_type(qnp.cross(positions, force), Quantity[Mul[LengthKind, ForceKind], Array])
assert_type(qnp.std(Temperature[Array]([300.0], u.K)), TemperatureDifference[Array])
assert_type(qnp.sum(positions, axis=0), Length[Array])
assert_type(qnp.allclose(positions, positions), bool)

# Plain values go to their own backend.
assert_type(qnp.cos(0.5), float)
assert_type(qnp.sqrt(np.float32(4)), np.float32)

# Array methods keep the kind.
assert_type(positions.T, Length[Array])
assert_type(positions.reshape(6), Length[Array])
assert_type(Temperature[Array]([300.0], u.K).std(), TemperatureDifference[Array])


def energy(x: Length[jax.Array], k: ForceConstant[float]) -> Energy[jax.Array]:
    return 0.5 * k * (x**2).sum()


def differentiate(x: Length[jax.Array], k: ForceConstant[float]) -> None:
    # Further arguments pass through; the first is differentiated.
    assert_type(ujax.grad(energy)(x, k), Force[jax.Array])
    assert_type(ujax.grad(energy)(x, k=k), Force[jax.Array])
    assert_type(
        ujax.value_and_grad(energy)(x, k), tuple[Energy[jax.Array], Force[jax.Array]]
    )
    assert_type(ujax.hessian(energy)(x, k), ForceConstant[jax.Array])
    assert_type(ujax.grad(energy, argnums=(0,)), Callable[..., Any])


def gradients(x: Length[torch.Tensor], y: Length[torch.Tensor]) -> None:
    energy = 0.5 * (2.0 * u.eV_per_angstrom_squared) * (x**2).sum()
    assert_type(utorch.grad(energy, x), Force[torch.Tensor])
    several = utorch.grad(energy, [x, y])
    assert_type(several[0], Quantity[Any, torch.Tensor])

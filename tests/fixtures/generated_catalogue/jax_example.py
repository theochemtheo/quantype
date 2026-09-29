"""Generated JAX adapters return quantities from the application catalogue."""

import jax
import numpy as np
from labquantities import Energy, Force, ForceConstant, Length, u, ujax


def energy(x: Length[jax.Array]) -> Energy[jax.Array]:
    return Energy.from_canonical((x.value**2).sum())


def test_jit_gradient() -> None:
    x = Length[jax.Array]([1, 2], u.angstrom)

    gradient = ujax.jit(ujax.grad(energy))(x)

    assert isinstance(gradient, Force)
    np.testing.assert_array_equal(gradient.value, [2, 4])


def test_hessian() -> None:
    x = Length[jax.Array]([1, 2], u.angstrom)

    hessian = ujax.hessian(energy)(x)

    assert isinstance(hessian, ForceConstant)
    np.testing.assert_array_equal(hessian.value, 2 * np.eye(2))

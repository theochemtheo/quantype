"""Generated JAX adapters return quantities from the application catalogue."""

import jax
import numpy as np
from labquantities import Energy, Force, ForceConstant, Length, Quantity, u, ujax


def energy(x: Length[jax.Array]) -> Energy[jax.Array]:
    return Energy.from_value((x.value**2).sum())


def test_jit_gradient() -> None:
    x = Length[jax.Array]([1, 2], u.angstrom)

    gradient = ujax.jit(ujax.grad(energy))(x)

    assert isinstance(gradient, Force)
    np.testing.assert_array_equal(gradient.value, [2, 4])


def test_structural_transforms() -> None:
    q = Length[jax.Array]([2, 3], u.angstrom) * (3 * u.fs)
    leaves, _ = jax.tree_util.tree_flatten(q)
    assert len(leaves) == 1
    doubled = ujax.jit(lambda value: value + value)(q)
    inverse = ujax.jit(lambda value: 1 / value)(q)
    mapped = ujax.vmap(lambda value: 1 / value)(q)
    assert type(doubled) is type(q)  # a product class keeps its class
    assert isinstance(q, Quantity)
    assert type(inverse) is Quantity
    assert type(mapped) is Quantity
    np.testing.assert_array_equal(doubled.value, [12, 18])
    np.testing.assert_allclose(inverse.value, [1 / 6, 1 / 9])
    np.testing.assert_array_equal(mapped.value, inverse.value)
    assert (inverse + u.one(1) / q).kind == inverse.kind


def test_hessian() -> None:
    x = Length[jax.Array]([1, 2], u.angstrom)

    hessian = ujax.hessian(energy)(x)

    assert isinstance(hessian, ForceConstant)
    np.testing.assert_array_equal(hessian.value, 2 * np.eye(2))

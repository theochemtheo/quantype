"""Backend transformations retain canonical units and differentiable storage."""

from collections.abc import Callable
from typing import Any, Protocol, TypeGuard, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from quantype import (
    Energy,
    Force,
    ForceConstant,
    Length,
    Quantity,
    _generated,
    u,
    ujax,
    utorch,
)


class _JaxTransforms(Protocol):
    def jit[**P, R](self, fun: Callable[P, R]) -> Callable[P, R]: ...

    def vmap[**P, R](self, fun: Callable[P, R]) -> Callable[P, R]: ...


class _JaxNumpy(Protocol):
    def array(self, value: object) -> jax.Array: ...

    def sum(self, value: jax.Array) -> jax.Array: ...


class _JaxTrees(Protocol):
    def tree_flatten(self, tree: Quantity[Any, Any]) -> tuple[list[Any], object]: ...

    def tree_unflatten(self, tree: object, leaves: list[Any]) -> Quantity[Any, Any]: ...


# Keep upstream JAX's incomplete annotations at the backend boundary; quantity
# inputs and transformation results retain their concrete types in the tests.
_jax = cast("_JaxTransforms", jax)
_jnp = cast("_JaxNumpy", jnp)
_trees = cast("_JaxTrees", jax.tree_util)


def _is_quantity_class(candidate: object) -> TypeGuard[type[Quantity[Any, Any]]]:
    return isinstance(candidate, type) and issubclass(candidate, Quantity)


def _jax_harmonic(x: Length[jax.Array]) -> Energy[jax.Array]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


def _torch_harmonic(x: Length[torch.Tensor]) -> Energy[torch.Tensor]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


def _identity[T](value: T) -> T:
    return value


def test_jax_jit_and_vmap() -> None:
    x = u.angstrom(_jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    energies = ujax.jit(ujax.vmap(_jax_harmonic))(x)
    assert type(energies) is Energy
    np.testing.assert_allclose(energies.value, [5.0, 25.0])
    gradients = ujax.jit(ujax.vmap(ujax.grad(_jax_harmonic)))(x)
    assert type(gradients) is Force
    np.testing.assert_allclose(gradients.value, 2.0 * x.value)


def test_jax_every_quantity_class_is_a_single_leaf_pytree() -> None:
    classes: list[Any] = []
    classes.extend(
        candidate
        for candidate in vars(_generated).values()
        if _is_quantity_class(candidate) and candidate is not Quantity
    )
    for cls in classes:
        quantity = cls.from_canonical(_jnp.array(2.0))
        leaves, tree = _trees.tree_flatten(quantity)
        assert len(leaves) == 1
        assert leaves[0] is quantity.value
        restored = _trees.tree_unflatten(tree, leaves)
        assert type(restored) is cls
        assert restored.kind == quantity.kind


def test_jax_pytree_preserves_display_and_structural_kind() -> None:
    quantity = u.angstrom(_jnp.array([1.0, 2.0])).to(u.nm)
    restored = _jax.jit(_identity)(quantity)
    np.testing.assert_allclose(restored.value, quantity.value)
    assert getattr(restored, "_display") == getattr(quantity, "_display")  # noqa: B009
    structural = quantity * u.eV(_jnp.array([3.0, 4.0]))
    assert type(structural) is Quantity
    leaves, tree = _trees.tree_flatten(structural)
    assert len(leaves) == 1
    assert leaves[0] is structural.value
    rebuilt = _trees.tree_unflatten(tree, leaves)
    assert type(rebuilt) is Quantity
    assert rebuilt.kind == structural.kind
    mapped = _jax.vmap(_identity)(structural)
    assert mapped.kind == structural.kind
    np.testing.assert_allclose(mapped.value, structural.value)


def test_jax_derivatives_and_conversion_invariance() -> None:
    gradient = ujax.grad(_jax_harmonic)
    hessian = ujax.hessian(_jax_harmonic)
    for x in (
        u.angstrom(_jnp.array([1.0, 2.0, 3.0])),
        u.nm(_jnp.array([0.1, 0.2, 0.3])),
        u.angstrom(_jnp.array([1.0, 2.0, 3.0])).to(u.nm),
    ):
        first = _jax.jit(gradient)(x)
        second = _jax.jit(hessian)(x)
        assert type(first) is Force
        assert type(second) is ForceConstant
        np.testing.assert_allclose(first.value, [2.0, 4.0, 6.0], rtol=1e-6)
        np.testing.assert_allclose(second.value, 2.0 * np.eye(3), rtol=1e-6)
        np.testing.assert_allclose((-first).value, [-2.0, -4.0, -6.0])


def test_jax_raw_model_boundary_and_single_evaluation() -> None:
    calls: list[None] = []

    def model(x: Length[jax.Array]) -> Energy[jax.Array]:
        calls.append(None)
        return Energy.from_canonical(_jnp.sum(x.value**2))

    x = u.angstrom(_jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(ujax.grad(model)(x).value, [2.0, 4.0])
    assert len(calls) == 1
    calls.clear()
    np.testing.assert_allclose(ujax.hessian(model)(x).value, 2.0 * np.eye(2))
    assert len(calls) == 1


def test_jax_rejects_nonscalar_energy() -> None:
    def vector_energy(x: Length[jax.Array]) -> Energy[jax.Array]:
        return Energy.from_canonical(x.value**2)

    with pytest.raises(TypeError, match="scalar"):
        ujax.grad(vector_energy)(u.angstrom(_jnp.array([1.0, 2.0])))


def test_torch_graph_and_higher_derivative() -> None:
    raw = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x = u.angstrom(raw)
    energy = _torch_harmonic(x)
    first = utorch.grad(energy, x, create_graph=True)
    assert isinstance(first, Force)
    assert type(first.value) is torch.Tensor
    assert first.value.requires_grad
    assert first.value.grad_fn is not None
    torch.testing.assert_close(first.value, 2.0 * raw)
    second = utorch.grad(first.sum(), x)
    assert isinstance(second, ForceConstant)
    torch.testing.assert_close(second.value, torch.full_like(raw, 2.0))
    torch.testing.assert_close((-first).value, -2.0 * raw)


def test_torch_conversion_preserves_graph_and_canonical_derivative() -> None:
    raw = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    x = u.nm(raw).to(u.angstrom)
    energy = _torch_harmonic(x)
    first = utorch.grad(energy, x, retain_graph=True)
    torch.testing.assert_close(first.value, torch.tensor([2.0, 4.0, 6.0]))
    # Differentiating with respect to the *raw nm* leaf includes the scale factor,
    # whereas the unit-aware adapter differentiates w.r.t. canonical angstroms.
    (raw_gradient,) = torch.autograd.grad(energy.value, raw)
    torch.testing.assert_close(raw_gradient, 10.0 * first.value)


def test_torch_raw_model_boundary() -> None:
    x = u.angstrom(torch.tensor([1.0, 2.0], requires_grad=True))
    energy = Energy.from_canonical(torch.sum(x.value**2))
    torch.testing.assert_close(utorch.grad(energy, x).value, 2.0 * x.value)


def test_torch_rejects_nonscalar_energy() -> None:
    x = u.angstrom(torch.tensor([1.0, 2.0], requires_grad=True))
    energy = Energy.from_canonical(x.value**2)
    with pytest.raises(ValueError, match="scalar"):
        utorch.grad(energy, x)

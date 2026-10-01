"""Backend transformations retain canonical units and differentiable storage."""

from collections.abc import Callable
from typing import Any, Protocol, TypeGuard, cast

import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
import numpy as np

from quantype import (
    Energy,
    Force,
    ForceConstant,
    Length,
    Quantity,
    _generated,
    u,
    ujax,
)
from quantype.products import LengthEnergy
from quantype.systems import CGS, SI, Atomic, Atomistic, Metal, Real, UnitSystem


def _dynamic(cls: object) -> Any:  # noqa: ANN401 -- runtime-chosen parameters
    """Parameterize by a runtime value, which a type expression cannot name."""
    return cls


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
        quantity = cls.from_value(_jnp.array(2.0))
        leaves, tree = _trees.tree_flatten(quantity)
        assert len(leaves) == 1
        assert leaves[0] is quantity.value
        restored = _trees.tree_unflatten(tree, leaves)
        assert type(restored) is cls
        assert restored.kind == quantity.kind


def test_jax_pytree_keeps_kind_and_system_but_not_display() -> None:
    quantity = u.angstrom(_jnp.array([1.0, 2.0])).to(u.nm)
    restored = _jax.jit(_identity)(quantity)
    np.testing.assert_allclose(restored.value, quantity.value)
    # Display units are presentation, not tree structure (UX-017).
    assert getattr(restored, "_display") is None  # noqa: B009
    assert restored.system is quantity.system
    _, presented = _trees.tree_flatten(quantity)
    _, plain = _trees.tree_flatten(u.angstrom(_jnp.array([1.0, 2.0])))
    assert presented == plain
    # A product class is created on first use, then registered as a pytree.
    structural = quantity * u.eV(_jnp.array([3.0, 4.0]))
    assert type(structural) is LengthEnergy
    leaves, tree = _trees.tree_flatten(structural)
    assert len(leaves) == 1
    assert leaves[0] is structural.value
    rebuilt = _trees.tree_unflatten(tree, leaves)
    assert type(rebuilt) is LengthEnergy
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
        return Energy.from_value(_jnp.sum(x.value**2))

    x = u.angstrom(_jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(ujax.grad(model)(x).value, [2.0, 4.0])
    assert len(calls) == 1
    calls.clear()
    np.testing.assert_allclose(ujax.hessian(model)(x).value, 2.0 * np.eye(2))
    assert len(calls) == 1


def test_jax_rejects_nonscalar_energy() -> None:
    def vector_energy(x: Length[jax.Array]) -> Energy[jax.Array]:
        return Energy.from_value(x.value**2)

    with pytest.raises(TypeError, match="scalar"):
        ujax.grad(vector_energy)(u.angstrom(_jnp.array([1.0, 2.0])))


class Gromacs(UnitSystem, name="test-autodiff:gromacs"):
    length = u.nanometer
    energy = Energy.define_unit(
        "test-autodiff:kJ_per_mol", reference=u.joule, scale=1e3 / 6.02214076e23
    )
    time = u.picosecond


SYSTEMS = pytest.mark.parametrize(
    "system",
    [Atomistic, Metal, Real, SI, CGS, Atomic, Gromacs],
    ids=lambda system: system.__name__,
)


@SYSTEMS
def test_jax_transformations_in_each_system(system: type[UnitSystem]) -> None:
    length: Any = _dynamic(Length)[jax.Array, system]
    spring: Any = _dynamic(ForceConstant)[float, system](2.0, u.eV_per_angstrom_squared)

    def energy(x: Any) -> Any:  # noqa: ANN401 -- system-generic helper
        return 0.5 * spring * (x**2).sum()

    x = length(_jnp.array([0.1, 0.2]), u.nm)
    force = _jax.jit(ujax.grad(energy))(x)
    assert type(force) is Force
    assert force.system is system
    np.testing.assert_allclose(force.magnitude(u.eV_per_angstrom), [2, 4], rtol=1e-5)
    curvature = _jax.jit(ujax.hessian(energy))(x)
    assert type(curvature) is ForceConstant
    np.testing.assert_allclose(
        curvature.magnitude(u.eV_per_angstrom_squared), 2 * np.eye(2), rtol=1e-5
    )
    step = length(_jnp.array(1.0), u.angstrom)
    lax: Any = cast("Any", jax).lax

    def advance(carry: Any, _: object) -> tuple[Any, None]:  # noqa: ANN401
        return carry + step, None

    carry, _ = lax.scan(advance, length(_jnp.array(0.0), u.nm), None, length=3)
    assert carry.system is system
    np.testing.assert_allclose(carry.magnitude(u.angstrom), 3.0, rtol=1e-5)
    chosen = lax.cond(
        False,  # noqa: FBT003 -- the predicate under test
        lambda: length(_jnp.array(1.0), u.nm),
        lambda: length(_jnp.array(5.0), u.angstrom),
    )
    np.testing.assert_allclose(chosen.magnitude(u.angstrom), 5.0, rtol=1e-5)


def test_jax_grad_requires_one_system() -> None:
    def leaks(x: Length[jax.Array, SI]) -> Energy[jax.Array]:
        del x
        return Energy.from_value(_jnp.array(1.0))

    untyped: Any = ujax.grad
    with pytest.raises(TypeError, match=r"argument's unit system \(SI\)"):
        untyped(leaks)(Length[jax.Array, SI](_jnp.array(1.0), u.nm))

"""The prototype scenarios that need JAX; see ``test_scenarios.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
import numpy as np

import quantype.numpy as qnp
from quantype import Angle, Dimensionless, Energy, Force, ForceConstant, Length, u, ujax
from quantype.systems import SI, Atomistic, Metal, UnitSystem

if TYPE_CHECKING:
    from collections.abc import Callable

SYSTEMS = pytest.mark.parametrize("system", [Atomistic, SI], ids=["Atomistic", "SI"])
# JAX's upstream annotations are incomplete; keep them at this boundary.
_jax: Any = jax
_jit: Any = _jax.jit
_lax: Any = _jax.lax
_jnp: Any = jnp


def _dynamic(cls: object) -> Any:  # noqa: ANN401 -- runtime-chosen parameters
    """Parameterize by a runtime value, which a type expression cannot name."""
    return cls


def _double(x: Any) -> Any:  # noqa: ANN401 -- any quantity
    return x * 2.0


def _suffix(system: type[UnitSystem]) -> str:
    return "" if system is Atomistic else f", {system.__name__}"


@SYSTEMS
def test_jax_traces_once_per_system(system: type[UnitSystem]) -> None:
    traces = 0

    def double_sum(x: Any) -> Any:  # noqa: ANN401 -- system-generic helper
        nonlocal traces
        traces += 1
        return (x * 2.0).sum()

    compiled: Callable[[Any], Any] = _jit(double_sum)
    length: Any = _dynamic(Length)[jax.Array, system]
    for unit in (u.nm, u.angstrom, u.bohr):
        compiled(length(_jnp.ones(3), unit))
    assert traces == 1


@SYSTEMS
def test_jax_structure_ignores_display(system: type[UnitSystem]) -> None:
    length: Any = _dynamic(Length)[jax.Array, system]
    suffix = _suffix(system)
    fallback = "20.0 Å" if system is Atomistic else "2e-09 m"
    doubled = _jit(_double)(length(_jnp.array(1.0), u.nm))
    assert repr(doubled) == f"Length({fallback}{suffix})"
    step = length(_jnp.array(1.0), u.angstrom)

    def advance(carry: Any, _: object) -> tuple[Any, None]:  # noqa: ANN401
        return carry + step, None

    carry, _ = _lax.scan(advance, length(_jnp.array(0.0), u.nm), None, length=3)
    assert carry.magnitude(u.angstrom) == pytest.approx(3.0)
    chosen = _lax.cond(
        True,  # noqa: FBT003 -- the predicate under test
        lambda: length(_jnp.array(1.0), u.nm),
        lambda: length(_jnp.array(10.0), u.angstrom),
    )
    assert chosen.magnitude(u.angstrom) == pytest.approx(10.0)


def test_jax_cond_across_systems_is_rejected() -> None:
    si = Length[jax.Array, SI](_jnp.array(1.0), u.nm)
    metal = Length[jax.Array, Metal](_jnp.array(10.0), u.angstrom)
    with pytest.raises(TypeError, match="pytree structure"):
        _lax.cond(True, lambda: si, lambda: metal)  # noqa: FBT003


@SYSTEMS
def test_force_from_grad(system: type[UnitSystem]) -> None:
    spring: Any = _dynamic(ForceConstant)[float, system](2.0, u.eV_per_angstrom_squared)

    def energy(x: Any) -> Any:  # noqa: ANN401 -- system-generic helper
        return 0.5 * spring * (x**2).sum()

    length: Any = _dynamic(Length)[jax.Array, system]
    derivative: Any = ujax.grad(energy)
    force = derivative(length(_jnp.array([0.1, 0.2]), u.nm))
    assert force.kind == "Force"
    assert force.system is system
    np.testing.assert_allclose(
        force.magnitude(u.eV_per_angstrom), [2.0, 4.0], rtol=1e-6
    )


def test_backend_scalars_and_traced_values_scale_quantities() -> None:
    length = Length[jax.Array]([1.0, 2.0], u.angstrom)

    def scale(factor: jax.Array, value: Length[jax.Array]) -> Length[jax.Array]:
        return factor * value

    scaled: Any = _jit(scale)(_jnp.array(3.0), length)
    np.testing.assert_allclose(np.asarray(scaled.value, dtype=np.float64), [3, 6])
    weighted = length * _jnp.ones(2)
    np.testing.assert_allclose(np.asarray(weighted.value, dtype=np.float64), [1, 2])


def _pair_energy(x: Length[jax.Array], k: ForceConstant[float]) -> Energy[jax.Array]:
    return 0.5 * k * (x**2).sum()


def test_value_and_grad_evaluates_once_and_passes_parameters_through() -> None:
    x = Length[jax.Array](_jnp.array([1.0, 2.0]), u.angstrom)
    k = ForceConstant[float](2.0, u.eV_per_angstrom_squared)
    energy, gradient = ujax.value_and_grad(_pair_energy)(x, k)
    assert isinstance(energy, Energy)
    assert isinstance(gradient, Force)
    assert float(energy.value) == 5.0
    np.testing.assert_allclose(np.asarray(gradient.value, dtype=np.float64), [2, 4])
    compiled: Any = _jit(ujax.value_and_grad(_pair_energy))
    energy, gradient = compiled(x, k)
    np.testing.assert_allclose(np.asarray(gradient.value, dtype=np.float64), [2, 4])
    hessian = ujax.hessian(_pair_energy)(x, k)
    assert isinstance(hessian, ForceConstant)


def test_argnums_differentiates_several_quantities() -> None:
    def energy(x: Length[jax.Array], y: Length[jax.Array]) -> Energy[jax.Array]:
        k = 2.0 * u.eV_per_angstrom_squared
        return 0.5 * k * ((x**2).sum() + (y**2).sum())

    x = Length[jax.Array](_jnp.array([1.0]), u.angstrom)
    y = Length[jax.Array](_jnp.array([3.0]), u.angstrom)
    gradients: Any = ujax.grad(energy, argnums=(0, 1))(x, y)
    assert [type(g) for g in gradients] == [Force, Force]
    np.testing.assert_allclose(np.asarray(gradients[1].value, dtype=np.float64), [6])
    second: Any = ujax.grad(energy, argnums=1)(x, y)
    assert isinstance(second, Force)
    untyped: Any = ujax.grad(energy, argnums=1)
    with pytest.raises(TypeError, match="Differentiated argument 1 must be a quantity"):
        untyped(x, 3.0)


def test_quantype_numpy_works_on_jax_arrays() -> None:
    angles = Angle[jax.Array](_jnp.array([0.0, 90.0]), u.deg)
    cosines = qnp.cos(angles)
    assert isinstance(cosines, Dimensionless)
    np.testing.assert_allclose(
        np.asarray(cosines.value, dtype=np.float64), [1, 0], atol=1e-6
    )
    positions = Length[jax.Array](_jnp.array([[3.0, 4.0]]), u.angstrom)
    distances = qnp.linalg.norm(positions, axis=-1)
    np.testing.assert_allclose(np.asarray(distances.value, dtype=np.float64), [5])

    def distance(x: Length[jax.Array]) -> Length[jax.Array]:
        return qnp.linalg.norm(x, axis=-1).sum()

    traced: Any = _jit(distance)(positions)
    np.testing.assert_allclose(float(traced.value), 5)

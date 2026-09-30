"""quantype.numpy and NumPy's dispatch protocols apply the same unit rules."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import quantype.numpy as qnp
from quantype import (
    Angle,
    Area,
    Dimensionless,
    Energy,
    Force,
    Length,
    Pressure,
    Quantity,
    Temperature,
    TemperatureDifference,
    u,
)
from quantype.systems import SI, Metal

if TYPE_CHECKING:
    from types import ModuleType

    from quantype.core import Unit
    from quantype.kinds import LengthKind

type Array = npt.NDArray[np.float64]
# NumPy's own functions dispatch on quantities at runtime; NumPy's stubs do not
# accept them, so typed code uses quantype.numpy and these tests an untyped alias.
numpy: Any = np


def _lengths(values: object, unit: Unit[LengthKind] = u.angstrom) -> Length[Array]:
    return Length[Array](values, unit)


@pytest.mark.parametrize("namespace", [qnp, numpy], ids=["qnp", "np"])
def test_trigonometry_takes_angles_and_gives_dimensionless(
    namespace: ModuleType,
) -> None:
    angle = Angle[float](60, u.deg)
    cosine = cast("Dimensionless[float]", namespace.cos(angle))
    assert isinstance(cosine, Dimensionless)
    assert float(cosine) == pytest.approx(0.5)
    back = namespace.arccos(cosine)
    assert isinstance(back, Angle)
    assert back.magnitude(u.deg) == pytest.approx(60)
    with pytest.raises(TypeError, match="cos expects Angle, received Length"):
        namespace.cos(1 * u.nm)
    with pytest.raises(TypeError, match="arccos expects Dimensionless"):
        namespace.arccos(angle)


@pytest.mark.parametrize("namespace", [qnp, numpy], ids=["qnp", "np"])
def test_exponentials_take_dimensionless(namespace: ModuleType) -> None:
    ratio = (1 * u.eV) / (1 * u.eV)
    assert float(namespace.exp(ratio)) == pytest.approx(math.e)
    assert float(namespace.log(ratio)) == 0
    with pytest.raises(TypeError, match="exp expects Dimensionless, received Energy"):
        namespace.exp(1 * u.eV)


@pytest.mark.parametrize("namespace", [qnp, numpy], ids=["qnp", "np"])
def test_square_roots_undo_squares(namespace: ModuleType) -> None:
    area = Area[float](9, u.angstrom_squared)
    assert namespace.sqrt(area) == 3 * u.angstrom
    unnamed = (3 * u.fs) * (3 * u.fs)
    assert namespace.sqrt(unnamed) == 3 * u.fs  # a structural square
    with pytest.raises(TypeError, match="sqrt needs a squared kind"):
        namespace.sqrt(1 * u.eV)


def test_plain_values_go_to_their_backend() -> None:
    assert qnp.cos(0.0) == 1.0
    np.testing.assert_allclose(qnp.sqrt(np.array([4.0, 9.0])), [2, 3])
    assert qnp.sum(np.ones(3)) == 3


def test_norms_distances_and_reductions() -> None:
    positions = _lengths([[0.0, 0, 0], [3, 4, 0]])
    for norms in (
        qnp.linalg.norm(positions, axis=-1),
        numpy.linalg.norm(positions, axis=-1),
    ):
        found: Length[Array] = norms
        assert isinstance(found, Length)
        np.testing.assert_allclose(found.value, [0, 5])
    np.testing.assert_allclose(numpy.sum(positions, axis=0).value, [3, 4, 0])
    np.testing.assert_allclose(qnp.mean(positions, axis=0).value, [1.5, 2, 0])
    assert numpy.max(positions) == 4 * u.angstrom
    assert numpy.shape(positions) == (2, 3)


def test_joining_choosing_and_limiting() -> None:
    first, second = _lengths([1.0, 2.0]), _lengths([3.0, 4.0], u.nm)
    for namespace in (qnp, numpy):
        stacked = namespace.stack([first, second])
        assert isinstance(stacked, Length)
        assert stacked.shape == (2, 2)
        chosen = namespace.where(np.array([True, False]), first, second)
        np.testing.assert_allclose(chosen.value, [1, 40])
        np.testing.assert_allclose(namespace.maximum(first, second).value, [30, 40])
        limited = namespace.clip(first, 1.5 * u.angstrom, None)
        np.testing.assert_allclose(limited.value, [1.5, 2])
    mixed: list[Any] = [first, Energy[Array]([1.0, 2.0], u.eV)]
    with pytest.raises(TypeError, match="quantities of one kind"):
        qnp.stack(mixed)
    with pytest.raises(TypeError, match="Cannot combine Atomistic and SI"):
        qnp.maximum(first, first.to_system(SI))


def test_products_follow_the_algebra() -> None:
    force = Force[Array]([1.0, 0.0, 0.0], u.eV_per_angstrom)
    displacement = _lengths([2.0, 5.0, 0.0])
    work = numpy.dot(force, displacement)
    assert isinstance(work, Energy)
    assert work.magnitude(u.eV) == pytest.approx(2)
    torque = qnp.cross(displacement, force)
    assert type(torque) is Quantity  # r cross F is a torque, not a work
    np.testing.assert_allclose(torque.value, [0, 0, -5])


def test_temperature_spreads_are_differences() -> None:
    points = Temperature[Array]([300.0, 310.0], u.K)
    assert isinstance(numpy.std(points), TemperatureDifference)
    assert isinstance(qnp.diff(points), TemperatureDifference)
    assert isinstance(points.std(), TemperatureDifference)
    with pytest.raises(TypeError, match="cumsum is meaningless"):
        points.cumsum()


def test_closeness_needs_tolerances_of_the_same_kind() -> None:
    first = _lengths([1.0, 2.0])
    assert qnp.allclose(first, first.to(u.nm), atol=1e-12 * u.angstrom)
    np.testing.assert_array_equal(
        numpy.isclose(first, _lengths([1.0, 3.0])), [True, False]
    )
    tolerance: Any = 1e-9
    with pytest.raises(TypeError, match="atol must be a quantity"):
        qnp.isclose(first, first, atol=tolerance)
    np.testing.assert_array_equal(qnp.isnan(first), [False, False])


def test_array_methods_keep_meaning() -> None:
    positions = _lengths([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], u.nm)
    assert positions.T.shape == (3, 2)
    assert positions.reshape(3, 2).shape == positions.reshape((3, 2)).shape == (3, 2)
    assert positions.transpose(1, 0).shape == (3, 2)
    assert positions[:1].squeeze().shape == (3,)
    assert positions.reshape(3, 2).unit is u.nanometer
    np.testing.assert_allclose(positions[0].cumsum().magnitude(u.nm), [1, 3, 6])
    assert positions.dtype == np.float64
    assert (1 * u.nm).dtype is None


def test_arrays_on_the_left_use_reflected_operators() -> None:
    length = 2 * u.nm
    np.testing.assert_allclose(
        (numpy.array([1.0, 2.0]) * length).magnitude(u.nm), [2, 4]
    )
    np.testing.assert_allclose((numpy.ones(2) / length).value, [0.05, 0.05])
    ratio = (1 * u.eV) / (2 * u.eV)
    np.testing.assert_array_equal(numpy.array([0.0, 1.0]) < ratio, [True, False])


def test_functions_without_unit_rules_explain_the_boundary() -> None:
    positions: Any = _lengths([1.0, 2.0])
    with pytest.raises(TypeError, match=r"np\.fft does not know the units"):
        numpy.fft.fft(positions)
    with pytest.raises(TypeError, match=r"np\.add\.reduce does not know"):
        numpy.add.reduce(positions)
    with pytest.raises(TypeError, match="Implicit array coercion"):
        numpy.asarray(positions)


def test_overridden_kinds_rescale_through_functions() -> None:
    pressures = Pressure[Array, Metal]([3.0, 4.0], u.bar)
    norm = qnp.linalg.norm(pressures)
    assert norm.system is Metal
    assert norm.magnitude(u.bar) == pytest.approx(5)
    angle = Angle[float, SI](90, u.deg)
    assert float(qnp.sin(angle)) == pytest.approx(1)


def test_every_function_passes_plain_arrays_to_numpy() -> None:
    a, b = np.array([1.0, 2.0, 0.0]), np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(qnp.arctan2(a, b), np.arctan2(a, b))
    np.testing.assert_allclose(qnp.maximum(a, b), [1, 2, 0])
    np.testing.assert_allclose(qnp.clip(a, 0.5, 1.5), [1, 1.5, 0.5])
    np.testing.assert_allclose(qnp.where(a > 1, a, b), [0, 2, 0])
    assert qnp.allclose(a, a)
    np.testing.assert_array_equal(qnp.isclose(a, b), [False, False, True])
    assert qnp.dot(a, b) == 2
    np.testing.assert_allclose(qnp.cross(a, b), [0, 0, 1])
    np.testing.assert_allclose(qnp.linalg.norm(a), 5**0.5)
    assert qnp.stack([a, b]).shape == qnp.concatenate([a[None], b[None]]).shape
    assert qnp.reshape(a, (3, 1)).shape == (3, 1)
    np.testing.assert_allclose(qnp.zeros_like(a), 0)
    np.testing.assert_allclose(qnp.cumsum(a), [1, 3, 3])
    np.testing.assert_allclose(qnp.diff(a), [1, -2])
    assert qnp.std(a) == np.std(a)
    assert qnp.absolute(-2.0) == 2
    assert qnp.sqrt(4.0) == 2


def test_dot_and_cross_need_two_quantities() -> None:
    length: Any = 1 * u.nm
    number: Any = 2.0
    with pytest.raises(TypeError, match="dot of a quantity needs another"):
        qnp.dot(length, number)
    with pytest.raises(TypeError, match="cross of a quantity needs another"):
        qnp.cross(length, np.ones(3))
    with pytest.raises(TypeError, match="norm is meaningless"):
        qnp.linalg.norm(Temperature[Array]([300.0], u.K))

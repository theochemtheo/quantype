"""Everyday Python and NumPy habits work on quantities, or fail clearly."""

from __future__ import annotations

import operator
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from quantype import (
    Dimensionless,
    Energy,
    EnergyDensity,
    Frequency,
    Length,
    Pressure,
    Quantity,
    Temperature,
    TemperatureDifference,
    Time,
    Volume,
    u,
)
from quantype.systems import SI, Metal
from quantype.testing import assert_allclose

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    "factor",
    [2, 2.0, np.float32(2), np.float64(2), np.int64(2), Fraction(2)],
    ids=lambda f: type(f).__name__,
)
def test_real_scalars_scale_quantities(factor: float) -> None:
    length = Length[float](2, u.nm)
    for scaled in (length * factor, factor * length, length / (1 / factor)):
        assert scaled.magnitude(u.nm) == pytest.approx(4)
        assert type(scaled.value) is float


def test_numerical_arrays_scale_quantities() -> None:
    step = Time[float](0.5, u.fs)
    grid = step * np.arange(4)
    assert isinstance(grid, Time)
    np.testing.assert_allclose(grid.magnitude(u.fs), [0, 0.5, 1, 1.5])
    assert_allclose(np.arange(4) * step, grid)
    weights = np.array([1.0, 0.5])
    positions = Length[npt.NDArray[np.float32]]([2, 4], u.angstrom)
    weighted = positions * weights
    assert_allclose(weighted, u.angstrom(np.array([2.0, 2.0])))
    assert weighted.value.dtype == np.float32
    assert (positions * np.float64(2)).value.dtype == np.float32
    reciprocal = np.ones(2) / Length[float](2, u.angstrom)
    assert_allclose(reciprocal, 1 / u.angstrom(np.array([2.0, 2.0])))


@pytest.mark.parametrize("factor", [True, np.True_, 1j, "2", None, [1, 2]])
def test_non_real_factors_are_rejected(factor: object) -> None:
    length: Any = Length[float](2, u.nm)
    with pytest.raises(TypeError, match="scale by real numbers or numerical arrays"):
        _ = length * factor
    with pytest.raises(TypeError, match="real numbers or numerical arrays"):
        _ = length * np.array([1, 2], dtype=bool)


def test_absolute_temperatures_scale() -> None:
    assert (2 * (300 * u.K)).value == 600


def test_zero_is_the_additive_identity() -> None:
    lengths = [1 * u.nm, 2 * u.nm, 5 * u.angstrom]
    total = sum(lengths)
    assert isinstance(total, Length)
    assert total.magnitude(u.nm) == pytest.approx(3.5)
    # Only the reflected forms are typed, which is what sum() needs.
    length: Any = 1 * u.nm
    assert length - 0 == length
    assert length + 0 == length
    assert (0 - 1 * u.nm) == -1 * u.nm
    untyped: Any = 1 * u.nm
    for other in (1, 0.0, np.int64(0)):
        with pytest.raises(TypeError, match="give it a unit"):
            _ = untyped + other
    points: list[Any] = [300 * u.K, 310 * u.K]
    with pytest.raises(TypeError, match=r"such as 10 \* u\.delta_K"):
        sum(points)


def test_dimensionless_values_mix_with_plain_numbers() -> None:
    ratio = (1 * u.eV) / (4 * u.eV)
    assert repr(ratio) == "Dimensionless(0.25)"
    assert str(ratio) == "0.25"
    assert float(ratio) == 0.25
    assert isinstance(ratio + 1, Dimensionless)
    assert (1 - ratio).value == 0.75
    assert ratio < 1
    assert ratio == 0.25
    assert ratio != 0.5
    ratios = Dimensionless[npt.NDArray[np.float64]]([0.5, 2.0], u.one)
    np.testing.assert_array_equal(ratios > 1, [False, True])
    with pytest.raises(TypeError, match="scalar Dimensionless"):
        float(ratios)
    length: Any = 1 * u.nm
    with pytest.raises(TypeError, match="Only a scalar Dimensionless quantity"):
        float(length)


def test_truthiness_is_ambiguous() -> None:
    for quantity in (0 * u.nm, 1 * u.nm, Length[npt.NDArray[np.float64]]([1], u.nm)):
        with pytest.raises(TypeError, match="truth value of a Length quantity"):
            bool(quantity)


def test_quantities_from_different_systems_are_unequal() -> None:
    atomistic = 2 * u.nm
    si: Any = atomistic.to_system(SI)
    assert atomistic != si
    assert atomistic not in [si, 3 * u.nm]
    assert len({atomistic, si}) == 2
    with pytest.raises(TypeError, match="Cannot combine Atomistic and SI"):
        _ = atomistic < si


def test_format_specs_apply_to_the_magnitude() -> None:
    length = Length[float](2, u.nm)
    assert f"{length:.3f}" == "2.000 nm"
    assert f"{length:>8.1e}" == " 2.0e+00 nm"
    assert f"{length}" == "2.0 nm"
    array = Length[npt.NDArray[np.float64]]([1, 2.5], u.angstrom)
    assert f"{array:.2f}" == "[1.00 2.50] Å"
    assert f"{(1 * u.eV) / (4 * u.eV):.1%}" == "25.0%"


def test_unit_names_the_presented_unit() -> None:
    assert (2 * u.nm).unit is u.nanometer
    assert Length.from_value(1.0).unit is u.angstrom
    assert Length[float, SI].from_value(1.0).unit is u.meter
    assert Pressure[float, Metal].from_value(1.0).unit is u.bar
    assert ((1 * u.nm) * (1 * u.fs)).unit is None


def test_reinterpret_names_equal_dimensions_explicitly() -> None:
    density = (1 * u.eV) / (1 * u.angstrom_cubed)
    assert isinstance(density, EnergyDensity)
    pressure = Pressure.reinterpret(density)
    assert isinstance(pressure, Pressure)
    assert pressure.magnitude(u.GPa) == pytest.approx(160.2176634)
    assert Frequency.reinterpret(1 / (1 * u.ps)).magnitude(u.THz) == pytest.approx(1)
    # Three factors whose relations agree are named, however they are grouped.
    assert isinstance((2 * u.eV) * (3 * u.angstrom) / (1 * u.eV), Length)
    structural = (2 * u.angstrom) * (3 * u.fs) * (1 * u.eV) / ((1 * u.eV) * (1 * u.fs))
    assert type(structural) is Quantity
    assert Length.reinterpret(structural).magnitude(u.angstrom) == pytest.approx(6)
    same = Length.reinterpret(2 * u.nm)
    assert same.unit is u.nanometer  # the same kind keeps its display unit


def test_reinterpret_keeps_system_and_rescales_overridden_kinds() -> None:
    density = Energy[float, Metal](1, u.eV) / Volume[float, Metal](1, u.angstrom_cubed)
    pressure = Pressure.reinterpret(density)
    assert pressure.system is Metal
    assert pressure.value == pytest.approx(1.602176634e6)  # bar in Metal
    assert pressure.magnitude(u.GPa) == pytest.approx(160.2176634)


def test_reinterpret_refuses_different_dimensions_and_temperature_points() -> None:
    untyped: Any = 1 * u.eV
    with pytest.raises(TypeError, match="Energy as Length: their dimensions differ"):
        Length.reinterpret(untyped)
    with pytest.raises(TypeError, match="Temperature as TemperatureDifference"):
        TemperatureDifference.reinterpret(300 * u.K)
    with pytest.raises(TypeError, match="TemperatureDifference as Temperature"):
        Temperature.reinterpret(10 * u.delta_K)
    number: Any = 2.0
    with pytest.raises(TypeError, match="Expected a quantity"):
        Length.reinterpret(number)


@pytest.mark.parametrize(
    "call",
    [
        lambda: Length[float](2, cast("Any", "nm")),
        lambda: (2 * u.nm).to(cast("Any", "nm")),
        lambda: (2 * u.nm).magnitude(cast("Any", "nm")),
        lambda: Length.define_unit("lab:x", reference=cast("Any", "nm")),
    ],
)
def test_string_units_get_a_type_error_naming_the_unit_object(
    call: Callable[[], object],
) -> None:
    with pytest.raises(TypeError, match=r"such as u\.nm, not strings"):
        call()


def test_a_device_passed_to_to_explains_display_units() -> None:
    with pytest.raises(TypeError, match="to move a tensor to another device"):
        (2 * u.nm).to(cast("Any", "cpu"))


@pytest.mark.parametrize("number", [0, 2, 2.0, np.float64(2), np.array([2.0])])
def test_comparing_with_a_plain_number_needs_a_unit(number: object) -> None:
    length: Any = 2 * u.nm
    other: Any = number
    for compare in (operator.eq, operator.ne, operator.lt, operator.ge):
        with pytest.raises(TypeError, match=r"Cannot compare Length and .*; give it a"):
            compare(length, other)
    with pytest.raises(TypeError, match="give it a unit"):
        operator.eq(other, length)


def test_non_numbers_are_unequal_and_dimensionless_values_take_numbers() -> None:
    length: Any = 2 * u.nm
    assert length != None  # noqa: E711 -- the comparison under test
    assert length != "2 nm"
    assert length in [None, 2 * u.nm]
    ratio = (1 * u.nm) / (4 * u.nm)
    assert ratio == 0.25
    assert ratio < 1

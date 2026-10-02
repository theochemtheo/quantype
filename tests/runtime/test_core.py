from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

# Runtime boundary tests deliberately inspect the internal algebra.
# pyright: reportPrivateUsage=false
import numpy as np
import numpy.typing as npt
import pytest

import quantype.numpy as qnp
import quantype.products
from quantype import (
    Area,
    Energy,
    EnergyDensity,
    Entropy,
    Force,
    Length,
    Pressure,
    Quantity,
    Temperature,
    TemperatureDifference,
    TemperatureRate,
    Velocity,
    Volume,
)
from quantype import units as u
from quantype._internal._registry import POWERS, QUANTITIES, RELATIONS, UNITS
from quantype._internal._semantics import KINDS
from quantype.core import _wrap, dimensions, get_unit, result_kind
from quantype.systems import SI

if TYPE_CHECKING:
    from collections.abc import Callable


def test_scalar_workflow() -> None:
    r = 2.0 * u.angstrom
    t = 4.0 * u.fs
    e = -3.2 * u.eV
    assert isinstance(r, Length)
    assert isinstance(r / t, Velocity)
    assert isinstance(e / r, Force)
    assert isinstance(r**2, Area)
    assert isinstance(r**3, Volume)
    assert isinstance((e / r) / (r**2), Pressure)
    assert (r / t).value == 0.5
    assert (e / r).value == -1.6
    assert (r**2).value == 4.0
    assert (r**3).value == 8.0
    assert isinstance((e / r) * r, Energy)
    assert isinstance(e / (r**3), EnergyDensity)


@pytest.mark.parametrize(
    ("unit", "canonical"),
    [
        ("meter", 1e10),
        ("centimeter", 1e8),
        ("nanometer", 10.0),
        ("bohr", 0.529177210903),
        ("second", 1e15),
        ("picosecond", 1e3),
        ("hartree", 27.211386245988),
        ("rydberg", 13.605693122994),
        ("joule", 1 / 1.602176634e-19),
        ("newton", 1 / 1.602176634e-9),
        ("pascal", 1 / 1.602176634e11),
    ],
)
def test_conversions(unit: str, canonical: float) -> None:
    # Runtime catalogue iteration intentionally bypasses the nominal static API.
    constructor: Any = get_unit(unit)
    q = constructor(1)
    assert q.value == pytest.approx(canonical)
    assert q.magnitude(constructor) == pytest.approx(1.0)


def test_explicit_conversion() -> None:
    q = 1.0 * u.hartree
    converted = q.to(u.eV)
    assert isinstance(converted, Energy)
    assert converted.value == q.value
    assert converted.magnitude() == pytest.approx(27.211386245988)
    assert q.magnitude(u.hartree) == 1.0
    assert repr(5 * u.angstrom) == "Length(5.0 Å)"
    assert str(5 * u.angstrom) == "5.0 Å"
    wrong: Any = u.second
    with pytest.raises(ValueError, match=r"Expected Energy.*Time"):
        q.to(wrong)


def test_temperature_affine_algebra() -> None:
    cold = 0 * u.celsius
    warm = 300 * u.K
    assert cold.value == pytest.approx(273.15)
    difference = warm - cold
    assert isinstance(difference, TemperatureDifference)
    assert difference.value == pytest.approx(26.85)
    assert isinstance(cold + difference, Temperature)
    assert (cold + difference).value == pytest.approx(300)
    assert (difference + cold).value == pytest.approx(300)
    assert (warm - difference).value == pytest.approx(cold.value)
    assert isinstance(difference / (2 * u.s), TemperatureRate)
    untyped: Any = warm
    with pytest.raises(TypeError, match="Cannot add two absolute Temperatures"):
        _ = untyped + warm
    array: Any = Temperature[npt.NDArray[np.float64]]([280, 300], u.K)
    with pytest.raises(TypeError, match="Cannot sum absolute Temperatures"):
        array.sum()
    assert array.mean().value == pytest.approx(290)


def test_absolute_temperatures_multiply_and_scale() -> None:
    # Kelvin-scaled storage makes products of absolute temperatures meaningful.
    warm = 300 * u.K
    assert (2 * warm).value == 600
    assert (warm / warm).value == 1
    assert (warm**2).value == 90000
    assert (-warm).value == -300
    boltzmann = Entropy[float](8.617333262e-5, u.eV_per_kelvin)
    thermal = boltzmann * warm
    assert isinstance(thermal, Energy)
    assert thermal.value == pytest.approx(0.025852, rel=1e-4)
    assert isinstance(thermal / boltzmann, Temperature)
    assert isinstance(thermal / warm, Entropy)


def test_scaling_drops_an_offset_display_unit() -> None:
    # Twice 20 °C is 586.3 K; shown in °C it would read as a misleading 313.15.
    doubled = 2 * (20 * u.celsius)
    assert doubled.value == pytest.approx(586.3)
    assert repr(doubled) == "Temperature(586.3 K)"
    assert repr(2 * (300 * u.K).to(u.kelvin)) == "Temperature(600.0 K)"
    assert repr(-(1 * u.delta_celsius)) == "TemperatureDifference(-1.0 Δ°C)"


def test_semantic_dimensions() -> None:
    pressure: Any = _wrap("Pressure", 1.0, None)
    density = _wrap("EnergyDensity", 1.0, None)
    per_volume = _wrap("EnergyPerVolume", 1.0, None)
    assert pressure.dimensions == density.dimensions == per_volume.dimensions
    with pytest.raises(TypeError, match=r"Pressure.*EnergyDensity"):
        _ = pressure + density
    assert (
        _wrap("Frequency", 1.0, None).dimensions
        == _wrap("InverseTime", 1.0, None).dimensions
    )
    assert (
        _wrap("EnergyPerAtom", 1.0, None).dimensions
        != _wrap("Energy", 1.0, None).dimensions
    )
    assert (
        _wrap("ParticleDensity", 1.0, None).dimensions
        != _wrap("ElectronDensity", 1.0, None).dimensions
    )


def test_structural_arithmetic_keeps_dimensions() -> None:
    uncommon = (2 * u.angstrom) * (3 * u.fs)
    assert isinstance(uncommon, Quantity)
    assert uncommon.kind == "Mul[Length,Time]"
    assert uncommon.value == 6
    assert uncommon.dimensions == (1, 0, 1, 0, 0, 0, 0, 0)
    # No global dimensional simplification invents a semantic named kind.
    again: Any = uncommon
    quotient = again / (2 * u.angstrom)
    assert quotient.dimensions == (0, 0, 1, 0, 0, 0, 0, 0)
    assert "Å" in repr(uncommon)
    assert "Mul" not in repr(uncommon)


def test_numpy_workflow() -> None:
    raw = np.array([1.0, 2.0, 3.0])
    positions = u.angstrom(raw)
    assert positions.value is raw
    assert isinstance(positions.sum().value, np.ndarray)
    assert positions.sum().value == 6
    assert positions.mean().value == 2
    np.testing.assert_allclose(qnp.sqrt(positions**2).value, raw)
    np.testing.assert_allclose(qnp.sin(u.degree(np.array([0.0, 90.0]))).value, [0, 1])
    np.testing.assert_allclose((u.angstrom * raw).value, raw)
    np.testing.assert_allclose((raw * u.angstrom).value, raw)
    with pytest.raises(TypeError, match="Implicit array coercion"):
        np.asarray(positions)
    untyped: Any = positions
    with pytest.raises(TypeError, match="sin expects Angle, received Length"):
        np.sin(untyped)


@pytest.mark.parametrize("function", [np.sum, np.mean, np.concatenate])
def test_numpy_functions_keep_units(function: Callable[[Any], Any]) -> None:
    quantity: Any = u.angstrom(np.array([1.0, 2.0]))
    argument = [quantity, quantity] if function is np.concatenate else quantity
    assert isinstance(function(argument), Length)


@pytest.mark.parametrize(
    "function", [np.fft.fft, np.cumprod, np.linalg.inv], ids=lambda f: f.__name__
)
def test_other_numpy_functions_explain_explicit_boundary(
    function: Callable[[Any], Any],
) -> None:
    quantity: Any = u.angstrom(np.eye(2))
    with pytest.raises(TypeError, match=r"quantype\.numpy.*\.magnitude\(unit\)"):
        function(quantity)


def test_unit_first_list_error_explains_construction() -> None:
    untyped: Any = u.nm
    with pytest.raises(
        TypeError, match=r"typed quantity constructor or numpy\.asarray"
    ):
        untyped([1, 2])


def test_registry_is_dimensionally_consistent() -> None:
    for (operation, left, right), result in RELATIONS.items():
        sign = 1 if operation == "mul" else -1
        expected = tuple(
            a + sign * b
            for a, b in zip(dimensions(left), dimensions(right), strict=True)
        )
        assert dimensions(result) == expected, (operation, left, right, result)
    for (kind, exponent), result in POWERS.items():
        assert dimensions(result) == tuple(n * exponent for n in dimensions(kind))
    for kind, spec in QUANTITIES.items():
        assert UNITS[spec.canonical_unit].kind == kind
        assert UNITS[spec.canonical_unit].scale == 1
        assert UNITS[spec.canonical_unit].offset == 0


def test_misused_constructors_and_classes_explain_themselves() -> None:
    with pytest.raises(TypeError, match=r"Expected a unit, such as u\.angstrom"):
        Length[float](2, cast("Any", 3))
    with pytest.raises(TypeError, match="dtype requires a parameterized"):
        Length(2.0, u.nm, dtype=cast("Any", np.float32))
    with pytest.raises(TypeError, match="Construct a named quantity"):
        Quantity(2.0, u.nm)
    with pytest.raises(TypeError, match="from_value requires a named quantity"):
        Quantity.from_value(1.0)
    with pytest.raises(TypeError, match="reinterpret requires a named quantity"):
        Quantity.reinterpret(2 * u.nm)
    with pytest.raises(ValueError, match="Expected Length; received Energy"):
        Length.define_unit("lab:wrong", reference=cast("Any", u.eV))
    with pytest.raises(ValueError, match="must not be empty"):
        Length.define_unit(" ", reference=u.nm)


def test_operations_without_physical_meaning_raise() -> None:
    length = 2 * u.nm
    with pytest.raises(TypeError, match="absolute Temperature from 0"):
        _ = 0 - cast("Any", 300 * u.K)
    with pytest.raises(TypeError, match="Cannot subtract Temperature and Length"):
        _ = cast("Any", 300 * u.K) - length
    with pytest.raises(TypeError, match="Only integer quantity powers"):
        _ = length ** cast("Any", 1.5)
    with pytest.raises(TypeError, match="not supported between"):
        _ = length < cast("Any", "short")
    with pytest.raises(ValueError, match="A scalar has no reduction axis"):
        length.sum(axis=0)
    flag: Any = True
    with pytest.raises(TypeError, match="Boolean values are not physical"):
        u.nm(flag)
    with pytest.raises(TypeError, match="scale by real numbers"):
        _ = length * cast("Any", np.dtype("float64"))


def test_scalar_storage_hashes_by_value_and_arrays_do_not_hash() -> None:
    first = Length[np.float64](2, u.nm)
    assert hash(first) == hash(Length[np.float64](20, u.angstrom))
    with pytest.raises(TypeError, match="unhashable array storage: ndarray"):
        hash(u.nm(np.array([1.0])))


def test_structural_results_display_their_units() -> None:
    product = (2 * u.nm) * (3 * u.fs) * u.eV(1)
    assert repr(product**2) == "Quantity(3600.0 (((Å * fs) * eV))^2)"
    assert repr(product - product) == "Quantity(0.0 ((Å * fs) * eV))"
    assert repr(u.nm) == "nm"


def test_internal_entry_points_reject_unsupported_requests() -> None:
    with pytest.raises(ValueError, match="Unknown physical operation 'add'"):
        result_kind("add", KINDS["Length"], KINDS["Length"])
    with pytest.raises(TypeError, match="from_value requires a named quantity"):
        Quantity[Any, float, SI].from_value(1.0)
    with pytest.raises(TypeError, match="NotImplemented"):
        np.add(cast("Any", u.nm), 1)


def test_product_classes_are_listed_for_completion() -> None:
    assert "LengthTime" in dir(quantype.products)

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Runtime boundary tests deliberately inspect the internal algebra.
# pyright: reportPrivateUsage=false
import numpy as np
import pytest

from quantype import (
    Area,
    Energy,
    EnergyDensity,
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
from quantype.core import _wrap, dimensions, get_unit

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
    assert repr(5 * u.angstrom) == "5.0 Å"
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
    operations: tuple[Callable[[], Any], ...] = (
        lambda: untyped + warm,
        lambda: untyped * 2,
        lambda: untyped**2,
    )
    for operation in operations:
        with pytest.raises(TypeError, match="Temperature"):
            operation()


def test_semantic_dimensions() -> None:
    pressure: Any = _wrap("Pressure", 1.0)
    density = _wrap("EnergyDensity", 1.0)
    per_volume = _wrap("EnergyPerVolume", 1.0)
    assert pressure.dimensions == density.dimensions == per_volume.dimensions
    with pytest.raises(TypeError, match=r"Pressure.*EnergyDensity"):
        _ = pressure + density
    assert _wrap("Frequency", 1.0).dimensions == _wrap("InverseTime", 1.0).dimensions
    assert _wrap("EnergyPerAtom", 1.0).dimensions != _wrap("Energy", 1.0).dimensions
    assert (
        _wrap("ParticleDensity", 1.0).dimensions
        != _wrap("ElectronDensity", 1.0).dimensions
    )


def test_structural_arithmetic_keeps_dimensions() -> None:
    uncommon = (2 * u.angstrom) * (3 * u.fs)
    assert isinstance(uncommon, Quantity)
    assert uncommon.kind == "Mul[Length,Time]"
    assert uncommon.value == 6
    assert uncommon.dimensions == (1, 0, 1, 0, 0, 0, 0)
    # No global dimensional simplification invents a semantic named kind.
    again: Any = uncommon
    quotient = again / (2 * u.angstrom)
    assert quotient.dimensions == (0, 0, 1, 0, 0, 0, 0)
    assert "Å" in repr(uncommon)
    assert "Mul" not in repr(uncommon)


def test_numpy_workflow() -> None:
    raw = np.array([1.0, 2.0, 3.0])
    positions = u.angstrom(raw)
    assert positions.value is raw
    assert isinstance(positions.sum().value, np.ndarray)
    assert positions.sum().value == 6
    assert positions.mean().value == 2
    np.testing.assert_allclose(u.sqrt(positions**2).value, raw)
    np.testing.assert_allclose(u.sin(u.degree(np.array([0.0, 90.0]))).value, [0, 1])
    np.testing.assert_allclose((u.angstrom * raw).value, raw)
    np.testing.assert_allclose((raw * u.angstrom).value, raw)
    with pytest.raises(TypeError, match="Implicit array coercion"):
        np.asarray(positions)
    untyped: Any = positions
    with pytest.raises(TypeError, match="quantity arithmetic"):
        np.sin(untyped)


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

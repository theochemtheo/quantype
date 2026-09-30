"""Run with pytest after generating labquantities in this fixture project."""

import pytest
from labquantities import (
    InverseTime,
    Length,
    Pressure,
    SurfaceTension,
    TemperatureDifference,
    u,
)

from quantype import Length as OriginalLength
from quantype import u as original_units


def test_physical_algebra() -> None:
    length = Length[float](2, u.length.angstrom)
    pressure = Pressure[float](3, u.pressure.pascal)

    assert isinstance(length * pressure, SurfaceTension)
    assert isinstance(pressure * length, SurfaceTension)
    assert (length * pressure).value == (pressure * length).value
    assert not isinstance(length, OriginalLength)
    product = length * (3 * u.fs)
    assert (product + product).value == 12
    assert (product * 2).mean().value == 12
    inverse = 1.0 / (2 * u.fs)
    assert isinstance(inverse, InverseTime)
    assert inverse.value == 0.5
    inverse_tension = 1.0 / (length * pressure)
    assert inverse_tension.kind == "Div[Dimensionless,SurfaceTension]"
    assert inverse_tension.value == 1.0 / (length * pressure).value


def test_serialization_roundtrip() -> None:
    length = Length[float](2, u.length.nanometer)

    restored = Length.parse(length.to(u.length.nanometer).to_dict())

    assert isinstance(restored, Length)
    assert restored.value == length.value
    assert Length.parse("2 nm", units=(u.nm,)).value == length.value
    assert u.nm is u.length.nanometer
    assert u.nm is not original_units.nm
    with pytest.raises(ValueError, match="shadows builtin"):
        Length.parse("2 nm", units=(original_units.nm,))


def test_math_preserves_catalogue() -> None:
    length = Length[float](2, u.length.angstrom)

    result = u.sqrt(length**2)

    assert isinstance(result, Length)
    assert result.value == length.value


def test_temperature_difference() -> None:
    cold = u.celsius(0)
    warm = u.kelvin(300)

    difference = warm - cold

    assert isinstance(difference, TemperatureDifference)
    assert difference.value == warm.value - cold.value

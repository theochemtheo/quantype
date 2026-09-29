"""Run with pytest after generating labquantities in this fixture project."""

from labquantities import (
    Length,
    Pressure,
    SurfaceTension,
    TemperatureDifference,
    u,
)

from quantype import Length as OriginalLength


def test_physical_algebra() -> None:
    length = Length[float](2, u.length.angstrom)
    pressure = Pressure[float](3, u.pressure.pascal)

    assert isinstance(length * pressure, SurfaceTension)
    assert isinstance(pressure * length, SurfaceTension)
    assert (length * pressure).value == (pressure * length).value
    assert not isinstance(length, OriginalLength)


def test_serialization_roundtrip() -> None:
    length = Length[float](2, u.length.nanometer)

    restored = Length.parse(length.to(u.length.nanometer).to_dict())

    assert isinstance(restored, Length)
    assert restored.value == length.value


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

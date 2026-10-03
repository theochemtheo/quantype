"""Run with pytest after generating labquantities in this fixture project."""

import labquantities.numpy as qnp
import pytest
from labquantities import (
    InverseTime,
    Length,
    Pressure,
    Quantity,
    SurfaceTension,
    TemperatureDifference,
    u,
)

from quantype import Length as OriginalLength
from quantype import u as original_units
from quantype.systems import SI, Atomistic


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
    assert inverse_tension.kind == "Pow[SurfaceTension,-1]"  # as tension ** -1
    assert inverse_tension.value == 1.0 / (length * pressure).value


def test_structural_reciprocals() -> None:
    quantities = (
        (2 * u.angstrom) * (3 * u.fs),
        (2 * u.angstrom) / (3 * u.bohr_magneton),
        (2 * u.angstrom) ** 4,
    )
    for q in quantities:
        # Product classes belong to this catalogue's structural base.
        assert isinstance(q, Quantity)
        scalar_inverse = 1 / q
        assert type(scalar_inverse) is Quantity
        explicit_inverse = u.one(1) / q
        assert (scalar_inverse + explicit_inverse).value == 2 / q.value
        assert (1 / (1 / q)).kind == (u.one(1) / scalar_inverse).kind
    mixed = (2 * u.angstrom) * (3 * original_units.fs)
    with pytest.raises(TypeError, match="mixed-catalogue"):
        1 / mixed
    # An explicit numerator remains usable without merging nominal identities.
    explicit = u.one(1) / mixed
    assert explicit.value == 1 / mixed.value
    other = original_units.one(1) / quantities[0]
    with pytest.raises(TypeError, match="Cannot add"):
        other + 1 / quantities[0]


def test_unit_systems_cover_new_kinds() -> None:
    tension = Length[float, SI](2, u.nm) * Pressure[float, SI](3, u.pascal)
    assert isinstance(tension, SurfaceTension)
    assert tension.system is SI
    assert tension.value == pytest.approx(6e-9)
    unit = SI.unit_for(SurfaceTension)
    assert unit.name == "si:SurfaceTension"
    assert unit.symbol == "J/m^2"
    assert Atomistic.unit_for(SurfaceTension) is u.surface_tension.surface_tension
    assert tension.to_system(Atomistic).magnitude(unit) == pytest.approx(6e-9)
    wire = tension.to_dict()
    assert wire["unit"] == "si:SurfaceTension"
    assert SurfaceTension[float, SI].parse(wire).value == pytest.approx(6e-9)
    restored = SurfaceTension.parse(wire, units=SI.units)
    assert restored.magnitude(unit) == pytest.approx(6e-9)
    assert qnp.sqrt(Length[float, SI](2, u.nm) ** 2).system is SI


def test_portable_unit_factors() -> None:
    assert u.lab_sqrt2(2).value == 2 * 2**0.5
    assert u.lab_point(1).value == 2.5


def test_unit_lookup() -> None:
    assert Length.unit_named("lab_sqrt2") is u.lab_sqrt2
    assert Length.unit_named("Å") is u.angstrom
    tension = u.surface_tension.surface_tension
    assert SurfaceTension.unit_named("surface_tension") is tension
    with pytest.raises(ValueError, match="Expected Length; received Pressure"):
        Length.unit_named("pascal")
    with pytest.raises(ValueError, match="Unknown unit 'furlong' for Length"):
        Length.unit_named("furlong")


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

    result = qnp.sqrt(length**2)

    assert isinstance(result, Length)
    assert result.value == length.value
    ratio = length / length
    assert type(qnp.exp(ratio)) is type(ratio)  # the catalogue's own Dimensionless
    assert type(qnp.arccos(ratio)).__module__.startswith("labquantities")


def test_temperature_difference() -> None:
    cold = u.celsius(0)
    warm = u.kelvin(300)

    difference = warm - cold

    assert isinstance(difference, TemperatureDifference)
    assert difference.value == warm.value - cold.value

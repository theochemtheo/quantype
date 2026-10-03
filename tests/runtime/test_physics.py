"""Mechanics, electrostatics and LAMMPS's non-coherent storage units."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

from quantype import (
    Acceleration,
    Area,
    Charge,
    ElectricField,
    Energy,
    EnergyDensity,
    Force,
    Length,
    Mass,
    MassDensity,
    Momentum,
    Pressure,
    Quantity,
    Time,
    Velocity,
    Volume,
    u,
)
from quantype._internal._registry import close_relations, unit_specs
from quantype.catalogue import QuantitySpec, UnitSpec, builtin_catalogue
from quantype.products import LengthMass
from quantype.serialization import load_npz, save_npz
from quantype.systems import SI, Atomistic, Metal, Real, UnitSystem

if TYPE_CHECKING:
    from pathlib import Path

# LAMMPS's own conversion factors (src/update.cpp), which it hard-codes to
# about seven digits from older CODATA values.
LAMMPS_METAL = {"mvv2e": 1.0364269e-4, "nktv2p": 1.6021765e6}
LAMMPS_REAL = {"mvv2e": 48.88821291**2, "nktv2p": 68568.415, "qe2f": 23.060549}
LAMMPS_DIGITS = 1e-6
# 1 Da Å²/fs² in eV, from CODATA 2022: the factor Atomistic mass arithmetic applies.
DA_ANGSTROM2_PER_FS2 = 1.66053906892e-27 * 1e10 / 1.602176634e-19


def test_atomistic_stores_mass_in_daltons() -> None:
    assert Atomistic.unit_for(Mass) is u.dalton
    assert Atomistic.unit_for(MassDensity) is u.gram_per_cubic_centimeter
    assert Atomistic.unit_for(Momentum) is u.eV_fs_per_angstrom
    assert Mass[float](12, u.Da).value == 12
    assert (12 * u.Da).value == 12
    assert Mass.from_value(12.0).magnitude(u.Da) == 12
    assert (1 * u.m_e).value == pytest.approx(5.485799090441e-4, rel=1e-9)
    assert repr(Mass.from_value(1.0)) == "Mass(1.0 Da)"


class Ase(UnitSystem, name="test-physics:ase"):
    length = u.angstrom
    energy = u.eV
    time = u.ase_time


def test_ase_units_make_daltons_coherent() -> None:
    fs = 1e5 * math.sqrt(1.66053906892e-27 / 1.602176634e-19)
    assert (1 * u.ase_time).magnitude(u.fs) == pytest.approx(fs, rel=1e-12)
    assert (1 * u.ase_velocity).magnitude(u.angstrom_per_fs) == pytest.approx(
        1 / fs, rel=1e-12
    )
    speed = Velocity[float](1, u.ase_velocity)
    derived = Length[float](1, u.angstrom) / Time[float](1, u.ase_time)
    assert speed.value == pytest.approx(derived.value, rel=1e-12)
    kinetic = Mass[float](1, u.Da) * speed * speed
    assert kinetic.magnitude(u.eV) == pytest.approx(1, rel=1e-12)
    momentum = Mass[float](1, u.Da) * speed
    assert momentum.magnitude(u.ase_momentum) == pytest.approx(1, rel=1e-12)
    assert (momentum * speed).magnitude(u.eV) == pytest.approx(1, rel=1e-12)
    # ase.units.fs, from ASE's default CODATA 2014.
    assert 1 / unit_specs("2014")["ase_time"].scale == pytest.approx(
        0.09822694788464063, rel=1e-14
    )


def test_a_system_on_ase_time_stores_ase_units() -> None:
    assert Ase.unit_for(Mass) is u.dalton
    assert Ase.unit_for(Velocity) is u.ase_velocity
    assert Ase.unit_for(Momentum) is u.ase_momentum
    velocities = Velocity[npt.NDArray[np.float64], Ase].from_value(np.full(3, 0.5))
    momenta = Mass[float, Ase](2, u.Da) * velocities
    np.testing.assert_allclose(momenta.value, np.full(3, 1.0))


def test_atomistic_mass_arithmetic_rescales() -> None:
    mass = Mass[float](1, u.Da)
    speed = Velocity[float](1, u.angstrom_per_fs)
    kinetic = mass * speed * speed
    assert isinstance(kinetic, Energy)
    assert kinetic.value == pytest.approx(DA_ANGSTROM2_PER_FS2, rel=1e-12)
    momentum = mass * speed
    assert isinstance(momentum, Momentum)
    assert momentum.value == pytest.approx(DA_ANGSTROM2_PER_FS2, rel=1e-12)
    acceleration = Force[float](1, u.eV_per_angstrom) / mass
    assert isinstance(acceleration, Acceleration)
    assert acceleration.value == pytest.approx(1 / DA_ANGSTROM2_PER_FS2, rel=1e-12)
    assert (momentum / speed).value == pytest.approx(1, rel=1e-12)
    density = Mass[float](1, u.gram) / Volume[float](1, u.cubic_centimeter)
    assert isinstance(density, MassDensity)
    assert density.value == pytest.approx(1, rel=1e-12)
    per_cubic_angstrom = mass / Volume[float](1, u.angstrom_cubed)
    assert per_cubic_angstrom.value == pytest.approx(1.66053906892, rel=1e-12)
    assert (per_cubic_angstrom * Volume[float](1, u.angstrom_cubed)).value == (
        pytest.approx(1, rel=1e-12)
    )


def test_metal_stores_lammps_units() -> None:
    assert Metal.unit_for(Mass) is u.gram_per_mole
    assert Metal.unit_for(Pressure) is u.bar
    assert Metal.unit_for(MassDensity) is u.gram_per_cubic_centimeter
    assert Metal.unit_for(Velocity) is u.angstrom_per_ps
    assert Metal.unit_for(ElectricField) is u.volt_per_angstrom
    assert Metal.unit_for(Charge) is u.elementary_charge


def test_real_stores_lammps_units() -> None:
    assert Real.unit_for(Mass) is u.gram_per_mole
    assert Real.unit_for(Pressure) is u.atmosphere
    assert Real.unit_for(ElectricField) is u.volt_per_angstrom
    assert Real.unit_for(Force) is u.kcal_per_mol_per_angstrom


def test_metal_arithmetic_matches_lammps_conversion_factors() -> None:
    mass = Mass[float, Metal](1, u.gram_per_mole)
    speed = Velocity[float, Metal](1, u.angstrom_per_ps)
    kinetic = mass * speed * speed
    assert isinstance(kinetic, Energy)
    assert kinetic.value == pytest.approx(LAMMPS_METAL["mvv2e"], rel=LAMMPS_DIGITS)
    acceleration = Force[float, Metal](1, u.eV_per_angstrom) / mass
    assert isinstance(acceleration, Acceleration)
    assert acceleration.value == pytest.approx(
        1 / LAMMPS_METAL["mvv2e"], rel=LAMMPS_DIGITS
    )
    pressure = Energy[float, Metal](1, u.eV) / Volume[float, Metal](1, u.angstrom_cubed)
    assert isinstance(pressure, EnergyDensity)  # nominally distinct from Pressure
    bar = Force[float, Metal](1, u.eV_per_angstrom) / Area[float, Metal](
        1, u.angstrom_squared
    )
    assert isinstance(bar, Pressure)
    assert repr(bar).endswith(" bar, Metal)")
    assert bar.value == pytest.approx(LAMMPS_METAL["nktv2p"], rel=LAMMPS_DIGITS)
    work = Pressure[float, Metal](1, u.bar) * Volume[float, Metal](1, u.angstrom_cubed)
    assert work.value == pytest.approx(1 / LAMMPS_METAL["nktv2p"], rel=LAMMPS_DIGITS)
    density = Mass[float, Metal](1, u.gram) / Volume[float, Metal](
        1, u.cubic_centimeter
    )
    assert isinstance(density, MassDensity)
    assert density.value == pytest.approx(1)


def test_real_arithmetic_matches_lammps_conversion_factors() -> None:
    mass = Mass[float, Real](1, u.gram_per_mole)
    speed = Velocity[float, Real](1, u.angstrom_per_fs)
    kinetic = mass * speed * speed
    assert kinetic.value == pytest.approx(LAMMPS_REAL["mvv2e"], rel=LAMMPS_DIGITS)
    force = Charge[float, Real](1, u.e) * ElectricField[float, Real](
        1, u.volt_per_angstrom
    )
    assert force.value == pytest.approx(LAMMPS_REAL["qe2f"], rel=LAMMPS_DIGITS)
    bar = Force[float, Real](1, u.kcal_per_mol_per_angstrom) / Area[float, Real](
        1, u.angstrom_squared
    )
    assert bar.value == pytest.approx(LAMMPS_REAL["nktv2p"], rel=LAMMPS_DIGITS)


def test_overridden_kinds_agree_with_the_default_system() -> None:
    mass = Mass[float, Metal](39.948, u.Da)
    speed = Velocity[float, Metal](5, u.angstrom_per_ps)
    in_metal = (mass * speed * speed).to_system(Atomistic)
    in_default = mass.to_system(Atomistic) * speed.to_system(Atomistic) ** 2
    assert in_metal.value == pytest.approx(in_default.value, rel=1e-12)
    pressure = Pressure[float, Metal](1.5, u.GPa)
    assert pressure.to_system(Atomistic).magnitude(u.GPa) == pytest.approx(1.5)
    assert pressure.magnitude(u.bar) == pytest.approx(15000)


def test_structural_results_are_stored_coherently() -> None:
    mass = Mass[float, Metal](1, u.gram_per_mole)
    moment = mass * Length[float, Metal](1, u.angstrom)
    assert type(moment) is LengthMass  # factors in catalogue order
    assert repr(moment).endswith("(Å * eV ps^2/Å^2), Metal)")
    assert moment.value == pytest.approx(LAMMPS_METAL["mvv2e"], rel=LAMMPS_DIGITS)


def test_overridden_kinds_serialize_in_their_units(tmp_path: Path) -> None:
    pressure = Pressure[float, Metal].from_value(3.0)
    assert pressure.to_dict() == {"kind": "Pressure", "magnitude": 3.0, "unit": "bar"}
    assert Pressure[float, Metal].parse("3 bar").value == pytest.approx(3.0)
    path = tmp_path / "frame.npz"
    save_npz(path, p=Pressure[npt.NDArray[np.float64], Metal].from_value(np.ones(2)))
    restored = load_npz(path, "p", Pressure[npt.NDArray[np.float64], Real])
    np.testing.assert_allclose(restored.magnitude(u.bar), [1, 1])


class Gromacs(UnitSystem, name="test-physics:gromacs"):
    length = u.nanometer
    energy = u.kJ_per_mol
    time = u.picosecond
    overrides = (u.bar,)


def test_custom_systems_can_override_kinds() -> None:
    # GROMACS is coherent in its base units, except for pressure in bar.
    assert Gromacs.unit_for(Pressure) is u.bar
    assert Gromacs.unit_for(Mass).scale == pytest.approx(u.Da.scale, rel=1e-8)
    force = Force[float, Gromacs](1, u.eV_per_angstrom)
    pressure = force / Area[float, Gromacs](1, u.angstrom_squared)
    assert pressure.magnitude(u.GPa) == pytest.approx(160.2176634)
    assert pressure.value == pytest.approx(1.602176634e6)  # bar


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ((u.nanometer,), TypeError, "overrides Length, a base axis"),
        ((u.celsius,), TypeError, "overrides Temperature, a base axis"),
        ((u.bar, u.atmosphere), TypeError, "overrides Pressure twice"),
        (("bar",), TypeError, "must contain units; received str"),
    ],
)
def test_invalid_overrides_are_rejected_at_definition(
    overrides: tuple[Any, ...], error: type[Exception], message: str
) -> None:
    # A class body assigning ``overrides`` cannot read a parameter of that name.
    cases: tuple[Any, ...] = overrides
    with pytest.raises(error, match=message):

        class Broken(UnitSystem, name="test-physics:broken"):  # pyright: ignore[reportUnusedClass]
            length = u.angstrom
            energy = u.eV
            time = u.fs
            overrides = cases


def test_derivatives_rescale_through_overridden_kinds() -> None:
    jax = pytest.importorskip("jax")
    ujax = pytest.importorskip("quantype.ujax")

    def pressure(volume: Volume[Any, Metal]) -> Quantity[Any, Any, Metal]:
        # P(V) = c / V, stored in bar.
        result: Quantity[Any, Any, Metal] = Pressure[Any, Metal].from_value(2.0) * (
            Volume[float, Metal](1, u.angstrom_cubed) / volume
        )
        return result

    volume = Volume[Any, Metal].from_value(jax.numpy.asarray(2.0))
    slope = ujax.grad(pressure)(volume)
    # dP/dV = -c / V²: -0.5 bar/Å³, stored coherently as eV/Å⁶.
    expected = -0.5 / LAMMPS_METAL["nktv2p"]
    assert float(slope.value) == pytest.approx(expected, rel=LAMMPS_DIGITS)


def test_torch_derivatives_rescale_through_overridden_kinds() -> None:
    torch = pytest.importorskip("torch")
    utorch = pytest.importorskip("quantype.utorch")
    raw = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    volume = Volume[Any, Metal].from_value(raw)
    pressure = Pressure[Any, Metal].from_value(torch.tensor(2.0, dtype=torch.float64))
    output = pressure * (Volume[float, Metal](1, u.angstrom_cubed) / volume)
    slope = utorch.grad(output, volume)
    expected = -0.5 / LAMMPS_METAL["nktv2p"]
    assert float(slope.value) == pytest.approx(expected, rel=LAMMPS_DIGITS)


def test_divisions_undo_declared_multiplications() -> None:
    assert isinstance((2 * u.eV) / (1 * u.eV_per_angstrom), Length)
    assert isinstance((2 * u.eV) / (1 * u.GPa), Volume)
    # A declared division wins over the one a multiplication would imply.
    assert isinstance((2 * u.eV) / (1 * u.angstrom_cubed), EnergyDensity)


def test_ambiguous_divisions_must_be_declared() -> None:
    # Force * Length and Force * Time both give Energy (dimensions aside), so
    # Energy / Force could be either Length or Time.
    declared = {
        ("mul", "Force", "Length"): "Energy",
        ("mul", "Force", "Time"): "Energy",
    }
    with pytest.raises(ValueError, match="Ambiguous division Energy / Force"):
        close_relations(declared)
    closed = close_relations({**declared, ("div", "Energy", "Force"): "Length"})
    assert closed["div", "Energy", "Force"] == "Length"


def test_application_catalogues_gain_undoing_divisions() -> None:
    catalogue = builtin_catalogue().extend(
        quantities={
            "SurfaceTension": QuantitySpec((-2, 1, 0, 0, 0, 0, 0, 0), "surface_tension")
        },
        units={"surface_tension": UnitSpec("SurfaceTension")},
        relations={("mul", "Pressure", "Length"): "SurfaceTension"},
    )
    assert catalogue.algebra["div", "SurfaceTension", "Pressure"] == "Length"
    assert catalogue.algebra["div", "SurfaceTension", "Length"] == "Pressure"
    assert ("div", "SurfaceTension", "Length") not in catalogue.relations


def test_new_units_convert_correctly() -> None:
    assert (1 * u.Da).magnitude(u.kg) == pytest.approx(1.66053906892e-27, rel=1e-10)
    assert (1 * u.debye).magnitude(u.coulomb_meter) == pytest.approx(
        3.33564e-30, rel=1e-5
    )
    assert (1 * u.atm).magnitude(u.bar) == pytest.approx(1.01325)
    assert (1 * u.kJ_per_mol).magnitude(u.kcal_per_mol) == pytest.approx(1 / 4.184)
    assert (1 * u.inverse_centimeter).magnitude(u.THz) == pytest.approx(0.0299792458)
    assert (1 * u.statcoulomb).magnitude(u.coulomb) == pytest.approx(3.33564095e-10)
    assert (1 * u.e).magnitude(u.coulomb) == pytest.approx(1.602176634e-19)
    assert math.isclose((1 * u.m_e).magnitude(u.Da), 5.485799090441e-4, rel_tol=1e-9)
    assert Length.parse("5 Angstrom") == Length.parse("5 Ang") == 5 * u.angstrom
    assert SI.unit_for(Charge) is u.coulomb

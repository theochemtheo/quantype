"""Unit systems: coherence, validation, conversion, and a custom system end to end."""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, TypeAdapter

from quantype import (
    Energy,
    Force,
    Length,
    Pressure,
    Temperature,
    TemperatureDifference,
    Time,
    u,
)
from quantype._internal._registry import RELATIONS
from quantype._internal._systems import range_table
from quantype._internal._unit import known_kinds
from quantype.core import get_unit
from quantype.serialization import load_npz, save_npz
from quantype.systems import (
    CGS,
    SI,
    Atomic,
    Atomistic,
    Metal,
    Real,
    StorageRangeWarning,
    UnitSystem,
)

BUILTIN = (Atomistic, Metal, Real, SI, CGS, Atomic)
DOCS = Path(__file__).parents[2] / "docs" / "units.md"

# GROMACS uses kJ/mol, which is not a catalogue unit: define it first.
kilojoule_per_mole = Energy.define_unit(
    "gromacs:kJ_per_mol", reference=u.joule, scale=1e3 / 6.02214076e23, symbol="kJ/mol"
)


def _dynamic(cls: object) -> Any:  # noqa: ANN401 -- runtime-chosen parameters
    """Parameterize by a runtime value, which a type expression cannot name."""
    return cls


class Gromacs(UnitSystem, name="gromacs"):
    length = u.nanometer
    energy = kilojoule_per_mole
    time = u.picosecond


class GromacsFs(Gromacs, name="gromacs-fs"):
    time = u.femtosecond


SYSTEMS = (*BUILTIN, Gromacs, GromacsFs)


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.__name__)
def test_systems_are_coherent(system: type[UnitSystem]) -> None:
    for (operation, left, right), result in RELATIONS.items():
        lhs = system.unit_for(left).scale
        rhs = system.unit_for(right).scale
        expected = lhs * rhs if operation == "mul" else lhs / rhs
        assert math.isclose(system.unit_for(result).scale, expected, rel_tol=1e-9), (
            operation,
            left,
            right,
        )


@pytest.mark.parametrize("system", BUILTIN, ids=lambda s: s.__name__)
def test_builtin_systems_name_every_builtin_kind(system: type[UnitSystem]) -> None:
    for kind in known_kinds():
        unit = system.unit_for(kind)
        assert get_unit(unit.name, kind=kind) is unit, (system, kind)
    assert system.units == ()


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        (Atomistic, ("angstrom", "electron_volt", "femtosecond", "bohr_magneton")),
        (Metal, ("angstrom", "electron_volt", "picosecond", "bohr_magneton")),
        (Real, ("angstrom", "kcal_per_mol", "femtosecond", "bohr_magneton")),
        (SI, ("meter", "joule", "second", "ampere_meter_squared")),
        (CGS, ("centimeter", "erg", "second", "erg_per_gauss")),
        (Atomic, ("bohr", "hartree", "atomic_time", "atomic_magnetic_moment")),
    ],
)
def test_builtin_base_units(
    system: type[UnitSystem], expected: tuple[str, ...]
) -> None:
    kinds = ("Length", "Energy", "Time", "MagneticMoment")
    assert tuple(system.unit_for(kind).name for kind in kinds) == expected
    assert system.unit_for(Temperature) is u.kelvin
    assert system.unit_for(TemperatureDifference) is u.delta_kelvin
    assert system.unit_for("AtomCount") is u.atom


def test_unit_for_accepts_classes_names_and_kinds() -> None:
    assert SI.unit_for(Pressure) is u.pascal
    assert SI.unit_for("Pressure") is u.pascal
    assert SI.unit_for(u.pascal.semantic) is u.pascal
    with pytest.raises(TypeError, match="named quantity kind"):
        SI.unit_for("NotAKind")
    with pytest.raises(TypeError, match="abstract"):
        UnitSystem.unit_for(Length)
    with pytest.raises(TypeError, match="used as classes"):
        SI()


@pytest.mark.parametrize("source", SYSTEMS, ids=lambda s: s.__name__)
@pytest.mark.parametrize("target", SYSTEMS, ids=lambda s: s.__name__)
def test_to_system_round_trips(
    source: type[UnitSystem], target: type[UnitSystem]
) -> None:
    for kind in known_kinds():
        unit: Any = Atomistic.unit_for(kind)
        sample: object = unit(1.0)
        cls: Any = type(sample)
        original = cls[float, source](1.5, unit)
        converted = original.to_system(target)
        assert converted.system is target
        assert converted.magnitude(unit) == pytest.approx(1.5, rel=1e-12)
        back = converted.to_system(source)
        assert back.value == pytest.approx(original.value, rel=1e-12)


def test_temperatures_never_gain_an_offset() -> None:
    point = Temperature[float](20.0, u.celsius)
    for system in BUILTIN:
        assert point.to_system(system).value == pytest.approx(293.15)
        assert point.to_system(system).magnitude(u.celsius) == 20.0


# --- Validation at class definition ------------------------------------------


def test_system_requires_a_name() -> None:
    body = {"length": u.nm, "energy": u.eV, "time": u.fs}
    with pytest.raises(TypeError, match=re.escape("requires name=...")):
        type("Unnamed", (UnitSystem,), body)


def test_system_names_are_unique_in_a_hierarchy() -> None:
    with pytest.raises(TypeError, match="reuses the system name 'gromacs'"):

        class Again(Gromacs, name="gromacs"):  # pyright: ignore[reportUnusedClass]
            pass


def test_system_requires_length_energy_and_time() -> None:
    with pytest.raises(TypeError, match="Incomplete is missing a time unit"):

        class Incomplete(UnitSystem, name="incomplete"):  # pyright: ignore[reportUnusedClass]
            length = u.nm
            energy = u.eV


def test_base_units_must_match_their_axis() -> None:
    wrong: Any = u.nanometer
    message = "Mixed.energy must be an Energy unit; received Length (nanometer)"
    with pytest.raises(TypeError, match=re.escape(message)):

        class Mixed(UnitSystem, name="mixed"):  # pyright: ignore[reportUnusedClass]
            length = u.nm
            energy = wrong
            time = u.fs


def test_temperature_base_must_not_have_an_offset() -> None:
    with pytest.raises(ValueError, match="must not have an offset; use kelvin"):

        class Celsius(UnitSystem, name="celsius"):  # pyright: ignore[reportUnusedClass]
            length = u.nm
            energy = u.eV
            time = u.fs
            temperature = u.celsius


def test_derived_scales_must_fit_float64() -> None:
    huge = Length.define_unit("lab:huge", reference=u.angstrom, scale=1e200)
    with pytest.raises(ValueError, match="Huge gives Area a scale that overflows"):

        class Huge(UnitSystem, name="huge"):  # pyright: ignore[reportUnusedClass]
            length = huge
            energy = u.eV
            time = u.fs


# --- A custom system end to end ---------------------------------------------


def test_gromacs_storage_and_bridge() -> None:
    cutoff = Length[float, Gromacs](1.2, u.nm)
    assert cutoff.value == 1.2
    assert repr(cutoff) == "Length(1.2 nm, Gromacs)"
    raw = np.ones((4, 3), dtype=np.float32)
    forces = Force[npt.NDArray[np.float32], Gromacs].from_value(raw)
    assert forces.value is raw
    assert forces.system is Gromacs
    assert Gromacs.unit_for(Force).name == "gromacs:Force"
    assert str(forces[0, 0]) == "1.0 kJ/mol/nm"
    atomistic = forces.to_system(Atomistic)
    expected = kilojoule_per_mole.scale / u.nanometer.scale  # eV/Å per kJ/mol/nm
    np.testing.assert_allclose(atomistic.value, expected, rtol=1e-6)
    assert atomistic.value.dtype == np.float32
    mixed: Any = forces
    with pytest.raises(TypeError, match=r"Cannot combine Gromacs and Atomistic"):
        _ = mixed + atomistic
    distance = np.asarray(2, dtype=np.float32)
    energy = forces[0, 0] * Length[npt.NDArray[np.float32], Gromacs].from_value(
        distance
    )
    assert type(energy) is Energy
    assert energy.magnitude(kilojoule_per_mole) == 2


def test_inherited_system_is_separate() -> None:
    step = Time[float, Gromacs](1, u.ps)
    fine = step.to_system(GromacsFs)
    assert fine.value == pytest.approx(1000)
    assert Gromacs.unit_for(Length) is GromacsFs.unit_for(Length)
    mixed: Any = step
    with pytest.raises(TypeError, match="Gromacs and GromacsFs"):
        _ = mixed + fine


def test_gromacs_units_decode_at_the_boundary(tmp_path: Path) -> None:
    assert kilojoule_per_mole in Gromacs.units
    force_unit = Gromacs.unit_for(Force)
    assert force_unit in Gromacs.units
    assert u.nanometer not in Gromacs.units
    force = Force[float, Gromacs].from_value(3.0)
    wire = force.to_dict()
    assert wire == {"kind": "Force", "magnitude": 3.0, "unit": "gromacs:Force"}
    # Untyped entry points need the definitions; any system can decode them.
    with pytest.raises(ValueError, match="Unknown unit"):
        Force.parse(wire)
    atomistic = Force.parse(wire, units=Gromacs.units)
    assert atomistic.system is Atomistic
    assert atomistic.magnitude(force_unit) == pytest.approx(3.0)
    # A target type that names the system finds its units automatically.
    adapter = TypeAdapter(Force[float, Gromacs])
    typed = adapter.validate_json(json.dumps(wire))
    assert typed.system is Gromacs
    assert typed.value == 3.0
    assert adapter.validate_python(wire, context={"units": Gromacs.units}).value == 3
    assert Force[float, Gromacs].parse(wire).value == 3.0
    path = tmp_path / "frame.npz"
    save_npz(
        path, forces=Force[npt.NDArray[np.float64], Gromacs].from_value(np.ones(2))
    )
    restored = load_npz(path, "forces", Force[npt.NDArray[np.float64], Gromacs])
    np.testing.assert_array_equal(restored.value, [1, 1])
    in_si = load_npz(
        path, "forces", Force[npt.NDArray[np.float64], SI], units=Gromacs.units
    )
    np.testing.assert_allclose(in_si.value, force_unit.scale / SI.unit_for(Force).scale)


def test_custom_units_resolve_by_kind() -> None:
    other = Pressure.define_unit("gromacs:Force", reference=u.pascal)
    wire = {"kind": "Force", "magnitude": 1.0, "unit": "gromacs:Force"}
    parsed = Force.parse(wire, units=(other, *Gromacs.units))
    assert parsed.magnitude(Gromacs.unit_for(Force)) == pytest.approx(1.0)


# --- Numerical range -----------------------------------------------------------


def test_documented_range_table_is_current() -> None:
    table = range_table(BUILTIN, ("float32", "float64"))
    assert table in DOCS.read_text()


@pytest.mark.parametrize("system", BUILTIN, ids=lambda s: s.__name__)
def test_float64_is_safe_in_every_builtin_system(system: type[UnitSystem]) -> None:
    assert system.check_range(np.float64) == ()


def test_float32_range_by_system() -> None:
    flagged = {
        system.__name__: {issue.kind for issue in system.check_range(np.float32)}
        for system in BUILTIN
    }
    assert flagged["Atomistic"] == flagged["Metal"] == flagged["Real"] == set()
    assert flagged["Atomic"] == set()
    assert {"Energy", "Volume"} <= flagged["SI"]
    energy = next(i for i in SI.check_range("float32") if i.kind == "Energy")
    assert energy.squared
    assert energy.problem == "underflow"
    assert str(energy).startswith("SI: Energy (1.6e-23 J)² would underflow float32")
    assert Gromacs.check_range(np.float32) == ()


def test_float32_variance_is_wrong_only_where_flagged() -> None:
    def variance(system: type[UnitSystem]) -> float:
        energies = _dynamic(Energy)[npt.NDArray[np.float32], system]([0.0, 0.002], u.eV)
        deviation = energies - energies.mean()
        return float((deviation**2).mean().value) * system.unit_for(Energy).scale ** 2

    assert variance(Atomistic) == pytest.approx(1e-6, rel=1e-6)
    assert variance(SI) != pytest.approx(1e-6, rel=1e-3)


def test_narrow_numpy_construction_warns_on_underflow() -> None:
    with pytest.warns(StorageRangeWarning, match="subnormal"):
        Energy[npt.NDArray[np.float32], SI]([1e-22], u.eV)
    # NumPy's own cast warning follows the caller's errstate.
    with (
        pytest.warns(StorageRangeWarning, match="overflow"),
        np.errstate(over="ignore"),
    ):
        Length[np.float16](1e5, u.angstrom)
    with pytest.warns(StorageRangeWarning, match="subnormal"):
        Energy[npt.NDArray[np.float32]]([1e-22], u.eV).to_system(SI)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Energy[npt.NDArray[np.float32], SI]([1.0], u.eV)
        Energy[npt.NDArray[np.float64], SI]([1e-22], u.eV)


# --- Pydantic ------------------------------------------------------------------


class SIConfig(BaseModel):
    cutoff: Length[float, SI]
    timestep: Time[float, Metal]


def test_pydantic_fields_choose_their_system() -> None:
    config = SIConfig.model_validate({"cutoff": "0.5 nm", "timestep": "2 fs"})
    assert config.cutoff.system is SI
    assert config.cutoff.value == pytest.approx(5e-10)
    assert config.timestep.value == pytest.approx(0.002)
    dumped = json.loads(config.model_dump_json())
    assert dumped["cutoff"] == {"kind": "Length", "magnitude": 0.5, "unit": "nanometer"}
    assert SIConfig.model_validate_json(config.model_dump_json()) == config
    with pytest.raises(
        ValueError, match=r"Expected a quantity in SI; received one in Atomistic"
    ):
        SIConfig(cutoff=u.nm(1.0), timestep=Time[float, Metal](1, u.fs))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

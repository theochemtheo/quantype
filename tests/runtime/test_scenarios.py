"""The unit-handling prototype scenarios, as contracts for the chosen design.

Ported from ``notes/alpha-release/unit-prototypes/scenarios.py``. Each scenario
runs in the default system and in SI, the system most unlike it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from quantype import (
    Energy,
    Length,
    Temperature,
    TemperatureDifference,
    u,
)
from quantype.systems import SI, Atomistic, Metal, Real, UnitSystem

SYSTEMS = pytest.mark.parametrize("system", [Atomistic, SI], ids=["Atomistic", "SI"])


def _dynamic(cls: object) -> Any:  # noqa: ANN401 -- runtime-chosen parameters
    """Parameterize by a runtime value, which a type expression cannot name."""
    return cls


def _suffix(system: type[UnitSystem]) -> str:
    return "" if system is Atomistic else f", {system.__name__}"


@SYSTEMS
def test_presentation(system: type[UnitSystem]) -> None:
    length: Any = _dynamic(Length)[float, system]
    energy: Any = _dynamic(Energy)[float, system]
    two_nm, five_a = length(2, u.nm), length(5, u.angstrom)
    suffix = _suffix(system)
    assert repr(two_nm) == f"Length(2.0 nm{suffix})"
    assert two_nm.magnitude() == 2.0
    assert repr(two_nm + five_a) == f"Length(2.5 nm{suffix})"
    assert repr(five_a + two_nm) == f"Length(25.0 Å{suffix})"
    ratio = energy(3, u.eV) / two_nm
    expected = "0.15 eV/Å" if system is Atomistic else "2.4032649509999997e-10 N"
    assert repr(ratio) == f"Force({expected}{suffix})"
    assert ratio.magnitude(u.eV_per_angstrom) == pytest.approx(0.15)
    array: Any = _dynamic(Length)[npt.NDArray[np.float64], system]
    assert repr(array([1.0, 2.0, 3.0], u.nm).mean()) == f"Length(2.0 nm{suffix})"


@SYSTEMS
def test_temperatures(system: type[UnitSystem]) -> None:
    point: Any = _dynamic(Temperature)[float, system]
    difference: Any = _dynamic(TemperatureDifference)[float, system]
    suffix = _suffix(system)
    assert repr(point(20, u.celsius) + difference(5, u.delta_celsius)) == (
        f"Temperature(25.0 °C{suffix})"
    )
    assert repr(point(30, u.celsius) - point(20, u.celsius)) == (
        f"TemperatureDifference(10.0 Δ°C{suffix})"
    )
    assert repr(difference(5, u.delta_kelvin) + point(20, u.celsius)) == (
        f"Temperature(25.0 °C{suffix})"
    )


@SYSTEMS
def test_raw_values_are_in_the_system(system: type[UnitSystem]) -> None:
    length: Any = _dynamic(Length)[float, system]
    a, b = length(2, u.nm).value, length(20, u.angstrom).value
    assert a == pytest.approx(b)
    assert a == pytest.approx(20.0 if system is Atomistic else 2e-9)


@SYSTEMS
@pytest.mark.parametrize(
    ("text", "magnitude", "unit"),
    [
        ("0.5 nm", 0.5, "nanometer"),
        ("10 bohr", 10.0, "bohr"),
        # Exact echo: storage cannot hold 20.1 °C exactly, the echo can.
        ("20.1 degC", 20.1, "celsius"),
    ],
)
def test_configs_round_trip_in_their_units(
    system: type[UnitSystem], text: str, magnitude: float, unit: str
) -> None:
    cls: Any = Temperature if unit == "celsius" else Length
    wire = cls[float, system].parse(text).to_dict()
    assert wire["magnitude"] == magnitude
    assert wire["unit"] == unit


@SYSTEMS
def test_float32_variance(system: type[UnitSystem]) -> None:
    energy: Any = _dynamic(Energy)[npt.NDArray[np.float32], system]
    energies = energy([0.0, 0.002], u.eV)
    variance = ((energies - energies.mean()) ** 2).mean()
    in_ev2 = float(variance.value) * system.unit_for(Energy).scale ** 2
    if system is Atomistic:
        assert in_ev2 == pytest.approx(1e-6, rel=1e-6)
    else:
        # SI underflows float32 here, as check_range and the docs report.
        assert in_ev2 == pytest.approx(9.826e-7, rel=1e-3)


def test_kernel_boundary_in_lammps_real_units() -> None:
    x = (2.0 * u.nm).to_system(Real)
    e = (1.0 * u.eV).to_system(Real)
    assert x.value == pytest.approx(20.0)
    assert e.value == pytest.approx(23.06054783061903)


def test_systems_never_mix() -> None:
    si: Any = Length[float, SI](2.0, u.nm)
    metal = Length[float, Metal](5.0, u.angstrom)
    with pytest.raises(TypeError, match=r"Cannot combine SI and Metal.*to_system"):
        _ = si + metal
    with pytest.raises(TypeError, match="Cannot combine SI and Metal"):
        _ = si * metal
    assert (si + metal.to_system(SI)).magnitude(u.nm) == pytest.approx(2.5)
    # Typed .value depends on the system, by design and visibly.
    assert si.value == pytest.approx(2e-9)
    assert Length[float, Metal](2.0, u.nm).value == pytest.approx(20.0)

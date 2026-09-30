"""Constants are exact in every unit system, so they adopt their operand's."""

from __future__ import annotations

import math
import subprocess
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from quantype import (
    Action,
    Charge,
    Energy,
    Entropy,
    Force,
    Length,
    Mass,
    Quantity,
    Temperature,
    Velocity,
    codata,
    constants,
    u,
)
from quantype.systems import SI, Atomic, Metal, Real, UnitSystem


def test_values_match_codata() -> None:
    values = codata.values()
    assert constants.k_B.magnitude(u.joule_per_kelvin) == pytest.approx(
        values.boltzmann_constant
    )
    assert constants.hbar.magnitude(u.joule_second) == pytest.approx(
        values.planck_constant / (2 * math.pi)
    )
    assert constants.e.magnitude(u.coulomb) == pytest.approx(values.elementary_charge)
    assert constants.m_e.magnitude(u.kg) == pytest.approx(values.electron_mass)
    assert constants.m_u.magnitude(u.Da) == pytest.approx(1)
    assert constants.c.magnitude(u.meter_per_second) == pytest.approx(299792458)
    # LAMMPS metal's qqr2e, to its seven digits.
    assert constants.k_e.to_system(Metal).value == pytest.approx(14.399645, rel=1e-6)


def test_constants_are_typed_by_kind() -> None:
    assert isinstance(constants.k_B.to_system(SI), Entropy)
    assert isinstance(constants.hbar.to_system(SI), Action)
    assert isinstance(constants.e.to_system(SI), Charge)
    assert isinstance(constants.m_e.to_system(SI), Mass)
    assert isinstance(constants.c.to_system(SI), Velocity)
    assert type(constants.k_e.to_system(SI)) is Quantity


def test_values_are_easy_to_inspect() -> None:
    assert repr(constants.k_B) == "Constant(k_B = 8.617333262145179e-05 eV/K)"
    assert repr(constants.k_B.to_system(SI)) == "Entropy(1.380649e-23 J/K, SI)"
    assert constants.k_B.magnitude(u.joule_per_kelvin) == pytest.approx(1.380649e-23)
    assert repr(constants.hbar.to(u.joule_second)).startswith("Action(1.05457")
    assert constants.k_B.to_system(Atomic).value == pytest.approx(
        constants.k_B.magnitude(u.hartree_per_kelvin)
    )
    wrong: Any = u.eV
    with pytest.raises(ValueError, match="Expected Entropy"):
        constants.k_B.magnitude(wrong)


@pytest.mark.parametrize("system", [SI, Metal, Real, Atomic], ids=lambda s: s.__name__)
def test_constants_adopt_the_operand_system(system: type[UnitSystem]) -> None:
    dynamic: Any = Temperature
    point: Temperature[float, Any] = dynamic[float, system](300, u.K)
    thermal = constants.k_B * point
    assert isinstance(thermal, Energy)
    adopted: object = thermal.system
    assert adopted is system
    assert thermal.magnitude(u.eV) == pytest.approx(0.025852, rel=1e-4)
    assert (point * constants.k_B).magnitude(u.eV) == pytest.approx(
        thermal.magnitude(u.eV)
    )
    assert isinstance(thermal / constants.k_B, Temperature)


def test_coulomb_energy_through_the_constant() -> None:
    energy = constants.k_e * (1 * u.e) * (1 * u.e) / (1 * u.angstrom)
    assert energy.dimensions == Energy[float](1, u.eV).dimensions
    assert energy.value == pytest.approx(14.3996, rel=1e-5)


def test_constant_arithmetic() -> None:
    doubled = 2 * constants.k_B
    assert type(doubled) is type(constants.k_B)
    assert doubled.magnitude(u.eV_per_kelvin) == pytest.approx(
        2 * constants.k_B.magnitude(u.eV_per_kelvin)
    )
    assert (np.float32(2) * constants.k_B).magnitude(u.eV_per_kelvin) == (
        doubled.magnitude(u.eV_per_kelvin)
    )
    assert (-constants.k_B).magnitude(u.eV_per_kelvin) < 0
    hbar_c = constants.hbar * constants.c
    assert hbar_c.to_system(SI).value == pytest.approx(3.16152677e-26)
    untyped: Any = constants.k_B
    with pytest.raises(TypeError, match="integer constant powers"):
        _ = untyped**0.5


def test_constants_work_with_arrays() -> None:
    points = Temperature[npt.NDArray[np.float64]]([300, 600], u.K)
    thermal = constants.k_B * points
    np.testing.assert_allclose(thermal.magnitude(u.eV), [0.025852, 0.051704], rtol=1e-4)
    force = (1 * u.e) * (1 * u.volt_per_angstrom)
    assert isinstance(force, Force)
    assert isinstance(constants.m_u * (1 * u.angstrom_per_fs), Quantity)
    assert isinstance(Length[float](1, u.nm) * constants.e, Quantity)


def test_importing_constants_does_not_fix_the_edition() -> None:
    code = (
        "import quantype.constants\n"
        "from quantype import codata\n"
        "codata.use('2018')\n"
        "from quantype.constants import k_B\n"
        "assert k_B.to_system(__import__('quantype.systems').systems.SI).value "
        "== 1.380649e-23\n"
    )
    result = subprocess.run(  # noqa: S603 -- a fixed interpreter and test source
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr

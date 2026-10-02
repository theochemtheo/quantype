"""Physical constants, exact in every unit system.

A constant has no unit system of its own: arithmetic with a quantity adopts
that quantity's system, so ``k_B * T`` is an energy in ``T``'s system. Values
come from the process's CODATA edition (see :mod:`quantype.codata`) and are
resolved on first use, so importing this module does not fix the edition.

``k_B.to_system(SI)``, ``k_B.magnitude(u.joule_per_kelvin)`` and ``repr(k_B)``
show a constant's value in a system or unit.
"""

# Constants resolve lazily through __getattr__, like units, and share core's
# package-private constructor.
# ruff: noqa: F822
# pyright: reportUnsupportedDunderAll=false, reportPrivateUsage=false
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from quantype import _generated, codata
from quantype._internal._semantics import KINDS, Semantic, power, product
from quantype.core import Constant, _constant

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["c", "e", "epsilon_0", "h", "hbar", "k_B", "k_e", "m_e", "m_u"]

# Keep the generated per-kind classes registered before any constant is built.
del _generated


def _coulomb_kind(*, inverse: bool) -> Semantic:
    """Energy length per charge squared (k_e), or its inverse (ε0)."""
    energy_length = product("mul", KINDS["Energy"], KINDS["Length"])
    charge_squared = power(KINDS["Charge"], 2)
    if inverse:
        return product("div", charge_squared, energy_length)
    return product("div", energy_length, charge_squared)


def _permittivity(values: codata.Codata) -> float:
    # F/m is C²/(J m); the reference units are e²/(eV Å).
    return values.vacuum_electric_permittivity / (values.elementary_charge * 1e10)


# Name -> kind and value in the reference units (Å, eV, fs, K, e).
_DEFINITIONS: dict[
    str, tuple[Callable[[], Semantic], Callable[[codata.Codata], float]]
] = {
    "k_B": (
        lambda: KINDS["Entropy"],
        lambda v: v.boltzmann_constant / v.elementary_charge,
    ),
    "h": (
        lambda: KINDS["Action"],
        lambda v: v.planck_constant / v.elementary_charge / 1e-15,
    ),
    "hbar": (
        lambda: KINDS["Action"],
        lambda v: v.planck_constant / (2 * math.pi) / v.elementary_charge / 1e-15,
    ),
    "e": (lambda: KINDS["Charge"], lambda _: 1.0),
    "m_e": (
        lambda: KINDS["Mass"],
        lambda v: v.electron_mass / (v.elementary_charge * 1e-10),
    ),
    "m_u": (
        lambda: KINDS["Mass"],
        lambda v: v.atomic_mass_constant / (v.elementary_charge * 1e-10),
    ),
    "c": (lambda: KINDS["Velocity"], lambda v: v.speed_of_light_in_vacuum * 1e-5),
    "epsilon_0": (lambda: _coulomb_kind(inverse=True), _permittivity),
    "k_e": (
        lambda: _coulomb_kind(inverse=False),
        lambda v: 1 / (4 * math.pi * _permittivity(v)),
    ),
}


def __getattr__(name: str) -> Constant[Any]:
    definition = _DEFINITIONS.get(name)
    if definition is None:
        raise AttributeError(name)
    kind, value = definition
    constant = _constant(name, kind(), value(codata.values()))
    globals()[name] = constant
    return constant

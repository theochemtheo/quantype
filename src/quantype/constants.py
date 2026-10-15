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
from fractions import Fraction
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


def _exact(value: float) -> Fraction:
    """A CODATA value as the exact decimal the table states."""
    return Fraction(repr(value))


def _permittivity(v: codata.Codata) -> Fraction:
    # F/m is C²/(J m); the reference units are e²/(eV Å).
    return _exact(v.vacuum_electric_permittivity) / (
        _exact(v.elementary_charge) * Fraction("1e10")
    )


# Name -> kind and value in the reference units (Å, eV, fs, K, e), exact except
# where π enters, and rounded once on first use.
_DEFINITIONS: dict[
    str, tuple[Callable[[], Semantic], Callable[[codata.Codata], Fraction | float]]
] = {
    "k_B": (
        lambda: KINDS["Entropy"],
        lambda v: _exact(v.boltzmann_constant) / _exact(v.elementary_charge),
    ),
    "h": (
        lambda: KINDS["Action"],
        lambda v: (
            _exact(v.planck_constant) / _exact(v.elementary_charge) / Fraction("1e-15")
        ),
    ),
    "hbar": (
        lambda: KINDS["Action"],
        lambda v: (
            float(
                _exact(v.planck_constant)
                / _exact(v.elementary_charge)
                / Fraction("1e-15")
            )
            / (2 * math.pi)
        ),
    ),
    "e": (lambda: KINDS["Charge"], lambda _: 1),
    "m_e": (
        lambda: KINDS["Mass"],
        lambda v: (
            _exact(v.electron_mass) / (_exact(v.elementary_charge) * Fraction("1e-10"))
        ),
    ),
    "m_u": (
        lambda: KINDS["Mass"],
        lambda v: (
            _exact(v.atomic_mass_constant)
            / (_exact(v.elementary_charge) * Fraction("1e-10"))
        ),
    ),
    "c": (
        lambda: KINDS["Velocity"],
        lambda v: _exact(v.speed_of_light_in_vacuum) * Fraction("1e-5"),
    ),
    "epsilon_0": (lambda: _coulomb_kind(inverse=True), _permittivity),
    "k_e": (
        lambda: _coulomb_kind(inverse=False),
        lambda v: 1 / (4 * math.pi * float(_permittivity(v))),
    ),
}


def __getattr__(name: str) -> Constant[Any]:
    definition = _DEFINITIONS.get(name)
    if definition is None:
        raise AttributeError(name)
    kind, value = definition
    constant = _constant(name, kind(), float(value(codata.values())))
    globals()[name] = constant
    return constant

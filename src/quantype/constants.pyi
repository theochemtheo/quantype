# Conventional symbols keep their case.
# ruff: noqa: N816
# The per-kind constant classes are private to the generated package.
# pyright: reportPrivateUsage=false
from quantype._generated import (
    _ActionConstant,
    _ChargeConstant,
    _EntropyConstant,
    _MassConstant,
    _VelocityConstant,
)
from quantype.core import Constant
from quantype.kinds import ChargeKind, Div, EnergyKind, LengthKind, Mul, Pow

__all__ = ["c", "e", "epsilon_0", "h", "hbar", "k_B", "k_e", "m_e", "m_u"]

#: Boltzmann constant.
k_B: _EntropyConstant
#: Planck constant.
h: _ActionConstant
#: Reduced Planck constant, h / 2π.
hbar: _ActionConstant
#: Elementary charge.
e: _ChargeConstant
#: Electron mass.
m_e: _MassConstant
#: Atomic mass constant (one dalton).
m_u: _MassConstant
#: Speed of light in vacuum.
c: _VelocityConstant
#: Vacuum electric permittivity.
epsilon_0: Constant[Div[Pow[ChargeKind, int], Mul[EnergyKind, LengthKind]]]
#: Coulomb constant, 1 / (4π ε0).
k_e: Constant[Div[Mul[EnergyKind, LengthKind], Pow[ChargeKind, int]]]

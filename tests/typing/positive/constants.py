"""Constants adopt their operand's unit system, statically too."""

# The per-kind constant classes are private to the generated package.
# pyright: reportPrivateUsage=false

from typing import assert_type

import numpy as np
import numpy.typing as npt

from quantype import Energy, Entropy, Force, Mass, Momentum, Temperature, constants, u
from quantype._generated import (
    _DimensionlessConstant,
    _EnergyConstant,
    _EntropyConstant,
)
from quantype.products import EntropySquared, VelocityAction
from quantype.systems import SI, Metal

point = 300 * u.K
point_si = Temperature[float, SI](300, u.K)
points = Temperature[npt.NDArray[np.float64]]([300, 600], u.K)

assert_type(constants.k_B * point, Energy[float])
assert_type(point * constants.k_B, Energy[float])
assert_type(constants.k_B * point_si, Energy[float, SI])
assert_type(constants.k_B * points, Energy[npt.NDArray[np.float64]])
assert_type((2 * u.eV) / constants.k_B, Temperature[float])
assert_type(constants.k_B.to_system(SI), Entropy[float, SI])
assert_type(constants.k_B.to(u.joule_per_kelvin), Entropy[float])
assert_type(constants.k_B.magnitude(u.joule_per_kelvin), float)
assert_type(2 * constants.k_B, _EntropyConstant)
assert_type(constants.m_u.to_system(Metal), Mass[float, Metal])
assert_type(constants.m_u * (1 * u.angstrom_per_fs), Momentum[float])
assert_type(constants.e * (1 * u.volt_per_angstrom), Force[float])

# Products of constants are named as products of quantities are.
assert_type((constants.hbar * constants.c).to_system(SI), VelocityAction[float, SI])
assert_type((constants.hbar * constants.c) / (1 * u.nm), Energy[float])
assert_type((2 / constants.k_B) * (1 * u.eV), Temperature[float])
assert_type((1 * u.eV) * (2 / constants.k_B), Temperature[float])
assert_type((constants.k_B**2).to_system(SI), EntropySquared[float, SI])
assert_type(constants.k_B / constants.k_B, _DimensionlessConstant)
assert_type(constants.m_e * constants.c * constants.c, _EnergyConstant)

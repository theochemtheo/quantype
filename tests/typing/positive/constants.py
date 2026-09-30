"""Constants adopt their operand's unit system, statically too."""

# The per-kind constant classes are private to the generated package.
# pyright: reportPrivateUsage=false

from typing import assert_type

import numpy as np
import numpy.typing as npt

from quantype import Energy, Entropy, Force, Mass, Momentum, Temperature, constants, u
from quantype._generated import _EntropyConstant
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

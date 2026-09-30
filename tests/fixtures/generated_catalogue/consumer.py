"""Portable static contract for an application-generated catalogue."""

from typing import assert_type

import numpy as np
from labquantities import InverseTime, Length, Pressure, SurfaceTension, Temperature, u
from labquantities.kinds import (
    DimensionlessKind,
    Div,
    LengthKind,
    Mul,
    SurfaceTensionKind,
    TimeKind,
)

from quantype.core import Quantity

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)

assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])

product = length * (3 * u.fs)
assert_type(product + product, Quantity[Mul[LengthKind, TimeKind], float])
assert_type(product * 2, Quantity[Mul[LengthKind, TimeKind], float])
assert_type(product.mean(), Quantity[Mul[LengthKind, TimeKind], float])
assert_type(1.0 / (2 * u.fs), InverseTime[float])
assert_type(
    1.0 / (length * pressure),
    Quantity[Div[DimensionlessKind, SurfaceTensionKind], float],
)
assert_type(u.angstrom(np.float64(2)), Length[np.float64])
assert_type((300 * u.K).mean(), Temperature[float])

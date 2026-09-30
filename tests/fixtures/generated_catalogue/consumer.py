"""Portable static contract for an application-generated catalogue."""

from typing import assert_type

import numpy as np
from labquantities import (
    InverseTime,
    Length,
    Pressure,
    Quantity,
    SurfaceTension,
    Temperature,
    u,
)
from labquantities.kinds import (
    DimensionlessKind,
    Div,
    LengthKind,
    Mul,
    SurfaceTensionKind,
    TimeKind,
)

from quantype.systems import SI, Atomistic

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)

assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])

product = length * (3 * u.fs)
assert_type(product + product, Quantity[Mul[LengthKind, TimeKind], float])
assert_type(product * 2, Quantity[Mul[LengthKind, TimeKind], float])
assert_type(product.mean(), Quantity[Mul[LengthKind, TimeKind], float])
assert_type(
    1 / product,
    Quantity[Div[DimensionlessKind, Mul[LengthKind, TimeKind]], float],
)
assert_type(
    1 / product + u.one(1) / product,
    Quantity[Div[DimensionlessKind, Mul[LengthKind, TimeKind]], float],
)
assert_type(1.0 / (2 * u.fs), InverseTime[float])
assert_type(
    1.0 / (length * pressure),
    Quantity[Div[DimensionlessKind, SurfaceTensionKind], float],
)
assert_type(u.angstrom(np.float64(2)), Length[np.float64])
assert_type((300 * u.K).mean(), Temperature[float])

length_si = Length[float, SI](2, u.nm)
pressure_si = Pressure[float, SI](3, u.pascal)
assert_type(length_si * pressure_si, SurfaceTension[float, SI])
assert_type((length_si * pressure_si).to_system(Atomistic), SurfaceTension[float])
assert_type(u.sqrt(length_si * length_si), Length[float, SI])

"""Portable static contract for an application-generated catalogue."""

from typing import assert_type

import labquantities.numpy as qnp
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
    TimeKind,
)
from labquantities.products import LengthTime, PerSurfaceTension

from quantype.systems import SI, Atomistic

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)

assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])

product = length * (3 * u.fs)
assert_type(product, LengthTime[float])
assert_type((3 * u.fs) * length, LengthTime[float])
assert_type(product + product, LengthTime[float])
assert_type(product * 2, LengthTime[float])
assert_type(product.mean(), LengthTime[float])
assert_type(product / (3 * u.fs), Length[float])
assert_type(
    1 / product,
    Quantity[Div[DimensionlessKind, Mul[LengthKind, TimeKind]], float],
)
assert_type(
    1 / product + u.one(1) / product,
    Quantity[Div[DimensionlessKind, Mul[LengthKind, TimeKind]], float],
)
assert_type(1.0 / (2 * u.fs), InverseTime[float])
assert_type(1.0 / (length * pressure), PerSurfaceTension[float])
assert_type(u.angstrom(np.float64(2)), Length[np.float64])
assert_type((300 * u.K).mean(), Temperature[float])

length_si = Length[float, SI](2, u.nm)
pressure_si = Pressure[float, SI](3, u.pascal)
assert_type(length_si * pressure_si, SurfaceTension[float, SI])
assert_type((length_si * pressure_si).to_system(Atomistic), SurfaceTension[float])
assert_type(qnp.sqrt(length_si * length_si), Length[float, SI])
assert_type(Length.unit_named("angstrom")(2.0), Length[float])
assert_type(SurfaceTension.kind, str)
assert_type(LengthTime.kind, str)

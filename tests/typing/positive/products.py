"""Products named by their factors, whatever their order, grouping or powers."""

from typing import assert_type

import numpy as np
import numpy.typing as npt

import quantype.numpy as qnp
from quantype import (
    Acceleration,
    Charge,
    Dimensionless,
    Energy,
    InverseTime,
    Length,
    Mass,
    Momentum,
    Quantity,
    Time,
    Velocity,
    constants,
    u,
)
from quantype.kinds import ChargeKind, LengthKind, Mul, TimeKind
from quantype.products import (
    LengthEntropy,
    LengthTime,
    PerLength,
    VelocitySquared,
)
from quantype.systems import SI

Array = npt.NDArray[np.float64]
m = Mass[float](1.0, u.Da)
v = Velocity[float](1.0, u.angstrom_per_fs)
p = Momentum[float](1.0, u.eV_fs_per_angstrom)
length = Length[float](1.0, u.angstrom)
t = Time[float](1.0, u.fs)
q = Charge[float](1.0, u.e)
lengths: Length[Array] = Length[Array](np.ones(3), u.angstrom)
times: Time[Array] = Time[Array](np.ones(3), u.fs)

# Three factors are named whatever their order, grouping or powers.
assert_type(m * v**2, Energy[float])
assert_type(v**2 * m, Energy[float])
assert_type(m * (v * v), Energy[float])
assert_type(p**2 / m, Energy[float])
assert_type(m * length / t, Momentum[float])
assert_type(length / t**2, Acceleration[float])
assert_type(length * t / t, Length[float])
assert_type(t * length / t, Length[float])
assert_type(length / t**2 * m * length, Energy[float])

# Two factors give one product class in either order; x * x is x ** 2.
assert_type(length * t, LengthTime[float])
assert_type(t * length, LengthTime[float])
assert_type(length * t + t * length, LengthTime[float])
assert_type(v * v, VelocitySquared[float])
assert_type(v**2, VelocitySquared[float])
assert_type(1 / length, PerLength[float])
assert_type(length**-1, PerLength[float])
assert_type(t**-1, InverseTime[float])
assert_type((length * t) / (t * length), Dimensionless[float])
assert_type(qnp.sqrt(v**2), Velocity[float])
assert_type(qnp.dot(length, t), LengthTime[float])

# Storage broadcasts through a product class.
assert_type(lengths * t / t, Length[Array])
assert_type(length * t / times, Length[Array])
assert_type((length * t + lengths * t) / t, Length[Array])

# Operations that keep a kind keep the product class too.
assert_type(-(length * t) / t, Length[float])
assert_type(abs(length * t) / t, Length[float])
assert_type(2.0 * (length * t) / t, Length[float])
assert_type((lengths * t).sum() / t, Length[Array])
assert_type((lengths * t).std() / times, Length[Array])
assert_type((length * t).to_system(SI) / t.to_system(SI), Length[float, SI])
assert_type(LengthTime.from_value(1.0, SI), LengthTime[float, SI])

# Constants take part like any other factor, from either side.
assert_type(constants.k_B * length, LengthEntropy[float])
assert_type(length * constants.k_B, LengthEntropy[float])
assert_type(lengths * constants.k_B, LengthEntropy[Array])

# Products of more factors keep their structure.
assert_type(length * t * q, Quantity[Mul[Mul[LengthKind, TimeKind], ChargeKind], float])
assert_type(
    q * (length * t), Quantity[Mul[ChargeKind, Mul[LengthKind, TimeKind]], float]
)

# apply keeps the quantity's own class, product classes included, and storage.
assert_type(lengths.apply(np.sort), Length[Array])
assert_type((length * t).apply(lambda raw: raw * 2), LengthTime[float])
bound = lengths.to_system(SI)
assert_type(bound.apply(np.sort), Length[Array, SI])
assert_type((length * t).to_system(SI).apply(abs), LengthTime[float, SI])

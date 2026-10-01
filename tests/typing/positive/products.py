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
m = Mass[float].from_value(1.0)
v = Velocity[float].from_value(1.0)
p = Momentum[float].from_value(1.0)
length = Length[float].from_value(1.0)
t = Time[float].from_value(1.0)
q = Charge[float].from_value(1.0)
lengths: Length[Array] = Length[Array].from_value(np.ones(3))
times: Time[Array] = Time[Array].from_value(np.ones(3))

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
assert_type(LengthTime[float].from_value(1.0), LengthTime[float])

# Constants take part like any other factor, from either side.
assert_type(constants.k_B * length, LengthEntropy[float])
assert_type(length * constants.k_B, LengthEntropy[float])
assert_type(lengths * constants.k_B, LengthEntropy[Array])

# Products of more factors keep their structure.
assert_type(length * t * q, Quantity[Mul[Mul[LengthKind, TimeKind], ChargeKind], float])
assert_type(
    q * (length * t), Quantity[Mul[ChargeKind, Mul[LengthKind, TimeKind]], float]
)

# with_value keeps the quantity's own class, product classes included.
assert_type(lengths.with_value(np.sort(lengths.value)), Length[Array])
assert_type((length * t).with_value(2.0), LengthTime[float])

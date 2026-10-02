"""Structural computations, affine means, reciprocals, and scalar storage."""

from typing import assert_type

import numpy as np
import numpy.typing as npt

from quantype import Dimensionless, InverseTime, Length, Quantity, Temperature, u
from quantype.kinds import (
    DimensionlessKind,
    Div,
    EnergyKind,
    LengthKind,
    Mul,
    Pow,
    TimeKind,
)
from quantype.products import EnergyPerTime, LengthTime, PerLength

product = (2 * u.angstrom) * (3 * u.fs)
assert_type(product + product, LengthTime[float])
assert_type(product - product, LengthTime[float])
assert_type(product * 2, LengthTime[float])
assert_type(2 * product, LengthTime[float])
assert_type(product / 2, LengthTime[float])
assert_type(-product, LengthTime[float])
assert_type(+product, LengthTime[float])
assert_type(abs(product), LengthTime[float])
assert_type(product.sum(), LengthTime[float])
assert_type(product.mean(), LengthTime[float])
assert_type(
    product * (2 * u.eV),
    Quantity[Mul[Mul[LengthKind, TimeKind], EnergyKind], float],
)
assert_type(
    product / (2 * u.eV),
    Quantity[Div[Mul[LengthKind, TimeKind], EnergyKind], float],
)
assert_type(
    1.0 / product,
    Quantity[Div[DimensionlessKind, Mul[LengthKind, TimeKind]], float],
)
ratio = (2 * u.eV) / (3 * u.fs)
assert_type(ratio + ratio, EnergyPerTime[float])
assert_type(ratio * 2, EnergyPerTime[float])
assert_type(ratio.mean(), EnergyPerTime[float])


def structural_power(q: Quantity[Pow[LengthKind, int], float]) -> None:
    assert_type(q + q, Quantity[Pow[LengthKind, int], float])
    assert_type(q * 2, Quantity[Pow[LengthKind, int], float])
    assert_type(q.mean(), Quantity[Pow[LengthKind, int], float])


array: npt.NDArray[np.float64] = np.array([1.0, 2.0])
array_product = u.angstrom(array) * (3 * u.fs)
assert_type(product + array_product, LengthTime[npt.NDArray[np.float64]])
assert_type(array_product + product, LengthTime[npt.NDArray[np.float64]])
assert_type(
    array_product * (2 * u.eV),
    Quantity[Mul[Mul[LengthKind, TimeKind], EnergyKind], npt.NDArray[np.float64]],
)
assert_type(
    product * u.eV(array),
    Quantity[Mul[Mul[LengthKind, TimeKind], EnergyKind], npt.NDArray[np.float64]],
)
assert_type(
    (2 * u.eV) * array_product,
    Quantity[Mul[EnergyKind, Mul[LengthKind, TimeKind]], npt.NDArray[np.float64]],
)
assert_type(1.0 / (2 * u.fs), InverseTime[float])
assert_type(1.0 / u.fs(array), InverseTime[npt.NDArray[np.float64]])
assert_type(1.0 / u.one(2), Dimensionless[float])
assert_type(1.0 / u.angstrom(2), PerLength[float])
assert_type(u.angstrom(np.float64(2)), Length[np.float64])
assert_type(u.nm(np.float32(2)), Length[np.float32])
assert_type(np.float64(2) * u.nm, Length[np.float64])
assert_type(u.nm * np.float64(2), Length[np.float64])
assert_type((300 * u.K).mean(), Temperature[float])
assert_type(+(300 * u.K), Temperature[float])

"""Typed constructors, custom units, and binary restoration."""

from typing import assert_type

import numpy as np
import numpy.typing as npt
import torch

from quantype import Area, Length, Temperature, u
from quantype.serialization import load_npz

assert_type(Length[np.float64](2, u.length.nanometer), Length[np.float64])
assert_type(
    Length[npt.NDArray[np.float64]]([[1, 2, 3]], u.nm), Length[npt.NDArray[np.float64]]
)
assert_type(
    Length[torch.Tensor]([1, 2], u.nm, dtype=torch.float64), Length[torch.Tensor]
)
assert_type(
    load_npz("positions.npz", "positions", Length[npt.NDArray[np.float64]]),
    Length[npt.NDArray[np.float64]],
)
bleb = Temperature.define_unit("lab:bleb", reference=u.celsius, scale=2**0.5)
assert_type(bleb(2), Temperature[float])
assert_type(Temperature[np.float64](2, bleb), Temperature[np.float64])

# Without a storage parameter, a float, NumPy float, or array is the storage.
assert_type(Length(2.0, u.nm), Length[float])
assert_type(Length(np.float32(2), u.nm), Length[np.float32])
values: npt.NDArray[np.float64] = np.zeros(3)
assert_type(Length(values, u.nm), Length[npt.NDArray[np.float64]])
assert_type(Length(torch.zeros(3), u.nm), Length[torch.Tensor])
assert_type(Length(2.0, u.nm) * Length(3.0, u.nm), Area[float])

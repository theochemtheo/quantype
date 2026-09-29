"""Shape-checked NumPy constructors reusable across all physical kinds.

These are deliberately constructors, not ndarray subclasses: subsequent NumPy
operations retain ordinary NumPy shape semantics rather than falsely promising
that arbitrary arithmetic preserves shape or tensor symmetry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def Vector3(value: ArrayLike) -> NDArray[np.float64]:  # noqa: N802
    """Build an independent three-component real vector."""
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (3,):
        raise ValueError(f"Expected vector shape (3,); received {array.shape}")
    return array


def SymmetricTensor3(value: ArrayLike) -> NDArray[np.float64]:  # noqa: N802
    """Build an independent symmetric 3x3 real tensor (NumPy allclose tolerance)."""
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (3, 3):
        raise ValueError(f"Expected tensor shape (3, 3); received {array.shape}")
    if not np.allclose(array, array.T, rtol=1e-7, atol=1e-12):
        raise ValueError("Expected a symmetric 3x3 tensor")
    return array

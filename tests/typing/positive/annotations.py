"""Quantity annotations read at runtime with quantype.typing."""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from quantype import Force, Quantity
from quantype.systems import SI, UnitSystem
from quantype.typing import QuantityType, quantity_type

annotation = quantity_type(Force[npt.NDArray[np.float64], SI])
assert_type(annotation, QuantityType | None)
assert annotation is not None
assert_type(annotation.kind, type[Quantity[Any, Any, Any]])
assert_type(annotation.storage, object)
assert_type(annotation.system, type[UnitSystem])
assert annotation.kind is Force
assert annotation.system is SI
assert quantity_type(float) is None

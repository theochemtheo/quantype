"""An intentionally small, explicit, backend-preserving numerical surface."""

# Package-private wrapping is shared by the narrow numerical adapters.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from typing import Any

from quantype.core import Quantity, _wrap


def _unary(name: str, value: Any) -> Any:
    if isinstance(value, (int, float)):
        return getattr(math, name)(value)
    module = type(value).__module__
    if module.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return getattr(jnp, name)(value)
    if module.startswith("torch"):
        import torch

        return getattr(torch, name)(value)
    import numpy as np

    return np.asarray(getattr(np, name)(value))


def sqrt(quantity: Quantity[Any, Any]) -> Quantity[Any, Any]:
    if quantity.kind != "Area":
        raise TypeError(f"sqrt expects Area, received {quantity.kind}")
    return _wrap("Length", _unary("sqrt", quantity.value))


def sin(quantity: Quantity[Any, Any]) -> Quantity[Any, Any]:
    if quantity.kind != "Angle":
        raise TypeError(f"sin expects Angle, received {quantity.kind}")
    return _wrap("Dimensionless", _unary("sin", quantity.value))


def exp(quantity: Quantity[Any, Any]) -> Quantity[Any, Any]:
    if quantity.kind != "Dimensionless":
        raise TypeError(f"exp expects Dimensionless, received {quantity.kind}")
    return _wrap("Dimensionless", _unary("exp", quantity.value))

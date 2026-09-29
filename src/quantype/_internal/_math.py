"""An intentionally small, explicit, backend-preserving numerical surface."""

# Package-private wrapping is shared by the narrow numerical adapters.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from typing import Any

from quantype._internal._semantics import KINDS
from quantype.core import Quantity, _wrap


def _unary(name: str, value: Any) -> Any:
    module = type(value).__module__
    if module == "builtins" and isinstance(value, (int, float)):
        return getattr(math, name)(value)
    if module.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return getattr(jnp, name)(value)
    if module.startswith("torch"):
        import torch

        return getattr(torch, name)(value)
    import numpy as np

    result = getattr(np, name)(value)
    return result if isinstance(value, np.generic) else np.asarray(result)


def sqrt(quantity: Quantity[Any, Any]) -> Quantity[Any, Any]:
    if quantity._semantic is not KINDS["Area"]:  # noqa: SLF001
        raise TypeError(f"sqrt expects Area, received {quantity.kind}")
    return _wrap("Length", _unary("sqrt", quantity.value))


def sin(quantity: Quantity[Any, Any]) -> Quantity[Any, Any]:
    if quantity._semantic is not KINDS["Angle"]:  # noqa: SLF001
        raise TypeError(f"sin expects Angle, received {quantity.kind}")
    return _wrap("Dimensionless", _unary("sin", quantity.value))


def exp(quantity: Quantity[Any, Any]) -> Quantity[Any, Any]:
    if quantity._semantic is not KINDS["Dimensionless"]:  # noqa: SLF001
        raise TypeError(f"exp expects Dimensionless, received {quantity.kind}")
    return _wrap("Dimensionless", _unary("exp", quantity.value))

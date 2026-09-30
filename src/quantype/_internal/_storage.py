"""Explicit storage conversion. Optional backends are imported only on request."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAliasType, cast, get_args, get_origin

if TYPE_CHECKING:
    from collections.abc import Iterable

# Backend APIs and NumPy's runtime dtype generics form a dynamic boundary.
# ruff: noqa: ANN401, PLC0415


def reject_booleans(value: Any) -> None:
    """Reject observable booleans before numerical coercion erases their types."""
    if (
        isinstance(value, bool)
        or getattr(getattr(value, "dtype", None), "kind", None) == "b"
    ):
        raise ValueError("Quantity magnitudes must be real numbers, not booleans")
    if isinstance(value, (list, tuple)):
        for item in cast("Iterable[Any]", value):
            reject_booleans(item)


def validate_floating_storage(value: Any) -> None:
    """Inspect existing storage without converting, detaching, or moving it."""
    module = type(value).__module__
    if module.startswith("torch"):
        valid = value.is_floating_point()
    elif module.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        valid = jnp.issubdtype(value.dtype, jnp.floating)
    else:
        valid = value.dtype.kind == "f"
    if not valid:
        raise ValueError(
            "Unit-first construction requires a real floating dtype; "
            "use a typed quantity constructor to convert integer magnitudes"
        )


def storage_origin(storage: Any) -> Any:
    origin = get_origin(storage) or storage
    while isinstance(origin, TypeAliasType):
        origin = get_origin(origin.__value__) or origin.__value__
    return origin


def _numpy(value: Any, storage: Any, dtype: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError("Quantity magnitudes must be real numbers")
    origin = storage_origin(storage)
    if origin is not np.ndarray:
        if dtype is not None:
            raise TypeError("Scalar dtype is already specified by the storage type")
        if array.ndim != 0:
            raise ValueError("Expected scalar storage, received an array")
        return origin(array)
    arguments = get_args(storage)
    scalar = arguments[-1] if arguments else np.float64
    if get_origin(scalar) is np.dtype:
        scalar = get_args(scalar)[0]
    if scalar is Any:
        scalar = np.float64
    if np.dtype(scalar).kind != "f":
        raise ValueError("Quantity storage must use a real floating dtype")
    if dtype is not None and np.dtype(dtype) != np.dtype(scalar):
        raise ValueError("dtype conflicts with the NumPy storage annotation")
    return np.asarray(array, dtype=scalar)


def _torch(value: Any, dtype: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        invalid = value.is_complex() or value.dtype == torch.bool
    else:
        invalid = np.asarray(value).dtype.kind not in "iuf"
    if invalid:
        raise ValueError("Quantity magnitudes must be real numbers")
    tensor = torch.as_tensor(value, dtype=dtype)
    if dtype is not None and not tensor.is_floating_point():
        raise ValueError("Quantity storage must use a real floating dtype")
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor


def _jax(value: Any, dtype: Any) -> Any:
    import jax.numpy as jnp
    import numpy as np

    # Inspect dtype without transferring existing backend arrays to the host.
    source_dtype = getattr(value, "dtype", None)
    if source_dtype is None:
        source_dtype = np.asarray(value).dtype
    if not (
        jnp.issubdtype(source_dtype, jnp.floating)
        or jnp.issubdtype(source_dtype, jnp.integer)
    ):
        raise ValueError("Quantity magnitudes must be real numbers")
    if dtype is not None and not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError("Quantity storage must use a real floating dtype")
    array = cast("Any", jnp).asarray(value, dtype=dtype)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.float32)
    if dtype is not None and np.dtype(array.dtype) != np.dtype(dtype):
        raise ValueError("Requested JAX dtype is unavailable; check jax_enable_x64")
    return array


def convert(value: Any, storage: Any, *, dtype: Any = None) -> Any:
    import numpy as np

    reject_booleans(value)
    origin = storage_origin(storage)
    if origin in (float, np.ndarray) or (
        isinstance(origin, type) and issubclass(origin, np.floating)
    ):
        return _numpy(value, storage, dtype)
    module = getattr(cast("Any", origin), "__module__", "")
    if module.startswith("torch"):
        return _torch(value, dtype)
    if module.startswith(("jax", "jaxlib")):
        return _jax(value, dtype)
    raise TypeError(f"Unsupported quantity storage {storage!r}")


def host_array(value: Any) -> Any:
    """Serialization only: intentionally detach and transfer to host."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"Unsupported wire dtype {array.dtype}")
    return array


def backend_name(value: object) -> str:
    module = type(value).__module__
    if module.startswith("torch"):
        return "torch"
    if module.startswith(("jax", "jaxlib")):
        return "jax"
    if module.startswith("numpy"):
        return "numpy"
    return "python"

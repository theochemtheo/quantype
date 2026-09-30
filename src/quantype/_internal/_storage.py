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


def _numpy(value: Any, storage: Any, dtype: Any, scale: float, offset: float) -> Any:
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
        return origin(_numpy_units(array, origin, scale, offset))
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
    return np.asarray(_numpy_units(array, scalar, scale, offset), dtype=scalar)


def _numpy_units(value: Any, dtype: Any, scale: float, offset: float) -> Any:
    import numpy as np

    if scale == 1 and offset == 0:
        return value
    raw = np.asarray(value, dtype=np.result_type(value.dtype, dtype, np.float64))
    return unit_conversion(raw, scale, offset)


def _torch(value: Any, dtype: Any, scale: float, offset: float) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        invalid = value.is_complex() or value.dtype == torch.bool
    else:
        invalid = np.asarray(value).dtype.kind not in "iuf"
    if invalid:
        raise ValueError("Quantity magnitudes must be real numbers")
    if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("Quantity storage must use a real floating dtype")
    if scale == 1 and offset == 0:
        tensor = torch.as_tensor(value, dtype=dtype)
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.get_default_dtype())
        return tensor
    # Host inputs must not round to the target dtype before unit conversion.
    tensor = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=torch.float64)
    )
    inferred = torch.as_tensor(value)
    target_dtype = dtype or (
        inferred.dtype if inferred.is_floating_point() else torch.get_default_dtype()
    )
    work_dtype = torch.float32 if tensor.device.type == "mps" else torch.float64
    tensor = tensor.to(work_dtype)
    return unit_conversion(tensor, scale, offset).to(target_dtype)


def _jax(value: Any, dtype: Any, scale: float, offset: float) -> Any:
    import jax
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
    target_dtype = dtype
    if target_dtype is None:
        target_dtype = (
            jax.dtypes.canonicalize_dtype(source_dtype)
            if jnp.issubdtype(source_dtype, jnp.floating)
            else jnp.float32
        )
    if dtype is not None and np.dtype(jax.dtypes.canonicalize_dtype(dtype)) != np.dtype(
        dtype
    ):
        raise ValueError("Requested JAX dtype is unavailable; check jax_enable_x64")
    # Integer inputs go straight to floating storage, never through int32.
    if scale == 1 and offset == 0:
        return cast("Any", jnp).asarray(value, dtype=target_dtype)
    work_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    array = cast("Any", jnp).asarray(value, dtype=work_dtype)
    return unit_conversion(array, scale, offset).astype(target_dtype)


def convert(
    value: Any,
    storage: Any,
    *,
    dtype: Any = None,
    scale: float = 1.0,
    offset: float = 0.0,
) -> Any:
    """Validate, convert units in working precision, then cast to target storage."""
    import numpy as np

    reject_booleans(value)
    origin = storage_origin(storage)
    if origin in (float, np.ndarray) or (
        isinstance(origin, type) and issubclass(origin, np.floating)
    ):
        return _numpy(value, storage, dtype, scale, offset)
    module = getattr(cast("Any", origin), "__module__", "")
    if module.startswith("torch"):
        return _torch(value, dtype, scale, offset)
    if module.startswith(("jax", "jaxlib")):
        return _jax(value, dtype, scale, offset)
    raise TypeError(f"Unsupported quantity storage {storage!r}")


def unit_conversion(
    value: Any, scale: float, offset: float, *, inverse: bool = False
) -> Any:
    """Apply an affine conversion without narrowing factors or changing storage.

    Existing low-precision input rounding is irreversible. Working precision
    avoids additional overflow/underflow from casting the conversion factors.
    Backend casts stay on device and retain tracers and differentiation graphs.
    """
    if scale == 1 and offset == 0:
        return value
    raw, limit = _working_value(value)
    if limit is not None:
        # Compare Python scalars: NumPy weak-scalar comparisons can themselves
        # overflow by narrowing a large Python factor to a float32 finfo value.
        maximum = float(limit.max)
        minimum = float(limit.tiny * limit.eps)
        if (
            float(scale) > maximum
            or float(scale) < minimum
            or abs(float(offset)) > maximum
            or (offset != 0 and abs(float(offset)) < minimum)
        ):
            raise ValueError(
                "Unit conversion factors exceed available working precision"
            )
    if inverse:
        if offset != 0:
            raw = raw - offset
        if scale != 1:
            raw = raw / scale
    else:
        if scale != 1:
            raw = raw * scale
        if offset != 0:
            raw = raw + offset
    return _restore_storage(value, raw)


def _working_value(value: Any) -> tuple[Any, Any]:
    module = type(value).__module__
    if module.startswith("torch"):
        import torch

        torch_dtype = torch.float32 if value.device.type == "mps" else torch.float64
        return value.to(torch_dtype), torch.finfo(torch_dtype)
    if module.startswith(("jax", "jaxlib")):
        import jax
        import jax.numpy as jnp

        jax_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
        return value.astype(jax_dtype), cast("Any", jnp).finfo(jax_dtype)
    if module.startswith("numpy"):
        import numpy as np

        numpy_dtype = np.result_type(value.dtype, np.float64)
        return np.asarray(value, dtype=numpy_dtype), np.finfo(numpy_dtype)
    return value, None


def _restore_storage(value: Any, raw: Any) -> Any:
    module = type(value).__module__
    if module.startswith("torch"):
        return raw.to(value.dtype)
    if module.startswith(("jax", "jaxlib")):
        return raw.astype(value.dtype)
    if module.startswith("numpy"):
        import numpy as np

        if isinstance(value, np.ndarray):
            return np.asarray(raw, dtype=cast("Any", value).dtype)
        return value.dtype.type(raw)
    return raw


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

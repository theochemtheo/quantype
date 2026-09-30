"""Regression coverage for the library audit's numerical and unit boundaries."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import TypeAdapter, ValidationError

from quantype import Length, u
from quantype.core import get_unit
from quantype.serialization import load_npz, save_npz

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from jax import Array


class _JaxNumpy(Protocol):
    def asarray(self, value: object, *, dtype: object = None) -> Array: ...


class _JaxTransforms(Protocol):
    def jit(self, function: Callable[[Array], Array]) -> Callable[[Array], Array]: ...

    def grad(self, function: Callable[[Array], Array]) -> Callable[[Array], Array]: ...


@pytest.mark.parametrize("unit_name", ["angstrom", "nm"])
@pytest.mark.parametrize(
    "raw",
    [
        np.array([True]),
        np.array([1 + 2j]),
        np.array(["bad"]),
        np.array([object()], dtype=object),
        np.array([100], dtype=np.int8),
        np.int64(100),
        np.bool_(True),  # noqa: FBT003 -- explicitly test boolean storage
        np.complex128(1 + 2j),
    ],
)
def test_unit_first_rejects_nonfloating_numpy_storage(
    raw: object, unit_name: str
) -> None:
    unit: Any = getattr(u, unit_name)
    with pytest.raises(ValueError, match="real floating dtype"):
        unit(raw)


@pytest.mark.parametrize(
    "scalar_type", [np.float16, np.float32, np.float64, np.longdouble]
)
def test_unit_first_preserves_numpy_floating_scalars(
    scalar_type: type[np.floating[Any]],
) -> None:
    raw = scalar_type(2)
    assert u.angstrom(raw).value is raw
    converted = u.nm(raw)
    assert type(converted.value) is scalar_type
    assert converted.value == 20
    assert type(u.angstrom(2).value) is float
    assert type(u.angstrom(2.0).value) is float


def test_python_scalar_subclasses_still_promote_to_float() -> None:
    class PythonFloat(float):
        pass

    class PythonInt(int):
        pass

    assert type(u.angstrom(PythonFloat(2)).value) is float
    assert type(u.angstrom(PythonInt(2)).value) is float


def test_integer_conversion_remains_available_explicitly() -> None:
    quantity = Length[npt.NDArray[np.float64]](
        np.array([100], dtype=np.int8), u.angstrom
    )
    np.testing.assert_array_equal((quantity**2).value, [10000])
    assert quantity.value.dtype == np.float64


@pytest.mark.parametrize(
    "raw",
    [
        [True, 2],
        [1.5, False],
        [[1, 2], [True, 3]],
        (1, (True, 2)),
        [np.bool_(True), 2],  # noqa: FBT003 -- explicitly test mixed booleans
        [np.array([True]), np.array([2])],
    ],
)
def test_mixed_booleans_are_rejected_before_coercion(raw: object) -> None:
    payload = {"kind": "Length", "magnitude": raw, "unit": "angstrom"}
    with pytest.raises(ValueError, match="booleans"):
        Length[npt.NDArray[np.float64]](raw, u.angstrom)
    with pytest.raises(ValueError, match="booleans"):
        Length.parse(payload)
    with pytest.raises(ValidationError, match="booleans"):
        TypeAdapter(Length[npt.NDArray[np.float64]]).validate_python(payload)


@pytest.mark.parametrize("raw", [[True, 2], [[1, 2], [True, 3]]])
def test_json_mixed_booleans_are_rejected(raw: object) -> None:
    wire = json.dumps({"kind": "Length", "magnitude": raw, "unit": "angstrom"})
    with pytest.raises(ValidationError, match="booleans"):
        TypeAdapter(Length[npt.NDArray[np.float64]]).validate_json(wire)


def test_builtin_definitions_are_canonical_across_scopes_and_aliases() -> None:
    assert get_unit("angstrom", kind=u.angstrom.semantic) is u.angstrom
    assert get_unit("nm", kind=u.nm.semantic) is u.nm
    assert get_unit("nanometer") is u.nm
    assert get_unit("nm", kind=u.nm.semantic) is u.length.nanometer
    assert Length.parse("1 angstrom", units=(u.angstrom,)).value == 1
    assert Length.parse("1 nm", units=(u.nm,)).value == 10
    adapter = TypeAdapter(Length[float])
    assert adapter.validate_python("1 nm", context={"units": (u.nm,)}).value == 10
    with pytest.raises(ValueError, match="Duplicate"):
        Length.parse("1 nm", units=(u.nm, u.length.nanometer))


def test_archive_accepts_explicit_builtin_units(tmp_path: Path) -> None:
    path = tmp_path / "length.npz"
    quantity = u.nm(np.array([1.0, 2.0])).to(u.nm)
    save_npz(path, length=quantity)
    restored = load_npz(path, "length", Length[npt.NDArray[np.float64]], units=(u.nm,))
    np.testing.assert_array_equal(restored.value, quantity.value)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_custom_builtin_shadowing_is_still_rejected(scale: float) -> None:
    shadow = Length.define_unit("angstrom", reference=u.angstrom, scale=scale)
    with pytest.raises(ValueError, match="shadows builtin"):
        Length.parse("1 angstrom", units=(shadow,))


def test_torch_unit_first_validation_and_graph_preservation() -> None:
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    for raw in (torch.tensor([True]), torch.tensor([100]), torch.tensor([1 + 2j])):
        with pytest.raises(ValueError, match="real floating dtype"):
            u.angstrom(raw)
        with pytest.raises(ValueError, match="real floating dtype"):
            u.nm(raw)
    with pytest.raises(ValueError, match="booleans"):
        Length[torch.Tensor]([True, 2], u.angstrom)
    raw_float = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    assert u.angstrom(raw_float).value is raw_float
    converted = u.nm(raw_float).value
    assert converted.dtype == raw_float.dtype
    assert converted.device == raw_float.device
    (gradient,) = torch.autograd.grad(converted.sum(), raw_float)
    np.testing.assert_array_equal(gradient.numpy(), [10, 10])
    assert Length[torch.Tensor]([1, 2], u.angstrom).mean().value == 1.5


def test_jax_unit_first_validation_and_tracer_preservation() -> None:
    pytest.importorskip("jax")
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    arrays = cast("_JaxNumpy", jnp)
    transforms = cast("_JaxTransforms", jax)

    def convert_units(value: jax.Array) -> jax.Array:
        return u.nm(value).value

    def summed(value: jax.Array) -> jax.Array:
        return convert_units(value).sum()

    for raw in (
        arrays.asarray([True]),
        arrays.asarray([100]),
        arrays.asarray([1 + 2j]),
    ):
        with pytest.raises(ValueError, match="real floating dtype"):
            u.angstrom(raw)
        with pytest.raises(ValueError, match="real floating dtype"):
            u.nm(raw)
    with pytest.raises(ValueError, match="booleans"):
        Length[jax.Array]([[1, 2], [True, 3]], u.angstrom)
    for dtype in (jnp.float32, jnp.bfloat16):
        raw_float = arrays.asarray([1.0, 2.0], dtype=dtype)
        assert u.angstrom(raw_float).value is raw_float
        converted = transforms.jit(convert_units)(raw_float)
        assert converted.dtype == raw_float.dtype
        np.testing.assert_array_equal(converted, [10, 20])
    gradient = transforms.grad(summed)(arrays.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(gradient, [10, 10])


def test_structural_arithmetic_and_reciprocals() -> None:
    product = (2 * u.angstrom) * (3 * u.fs)
    assert (product + product).value == 12
    assert (product - product).value == 0
    assert (product * 2).value == 12
    assert (2 * product).value == 12
    assert (product / 2).value == 3
    assert (-product).value == -6
    assert abs(-product).value == 6
    assert product.mean().value == 6
    assert product.sum().value == 6
    assert (1.0 / (2 * u.fs)).kind == "InverseTime"
    assert (1.0 / (2 * u.fs)).value == 0.5
    inverse_product = 1.0 / product
    assert inverse_product.kind == "Div[Dimensionless,Mul[Length,Time]]"
    assert inverse_product.value == pytest.approx(1 / 6)

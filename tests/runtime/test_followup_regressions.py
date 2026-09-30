"""Regression tests for the follow-up audit's numerical and JSON boundaries."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import TypeAdapter

from quantype import Length, Temperature, Time, u
from quantype.serialization import load_npz, save_npz

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("array_storage", [False, True])
def test_float16_construction_converts_before_rounding(*, array_storage: bool) -> None:
    storage: Any = npt.NDArray[np.float16] if array_storage else np.float16
    with np.errstate(over="raise", invalid="raise"):
        length = Length[storage](3e-6, u.meter)
        time = Time[storage](1e-11, u.second)
        point = Temperature[storage](0, u.celsius)
    assert length.value.dtype == np.float16
    assert time.value.dtype == np.float16
    assert isinstance(length.value, np.ndarray) == array_storage
    assert length.value == np.float16(3e-6 * u.meter.scale)
    assert time.value == np.float16(1e-11 * u.second.scale)
    assert point.value == np.float16(u.celsius.offset)


@pytest.mark.parametrize("array_storage", [False, True])
def test_float16_presentation_and_unit_first_storage(*, array_storage: bool) -> None:
    storage: Any = npt.NDArray[np.float16] if array_storage else np.float16
    with np.errstate(over="raise", invalid="raise"):
        magnitude = Length[storage](30000, u.angstrom).magnitude(u.meter)
        raw = np.asarray(3e-6, dtype=np.float16) if array_storage else np.float16(3e-6)
        canonical = u.meter(raw).value
    assert magnitude.dtype == np.float16
    assert magnitude == np.float16(30000 / u.meter.scale)
    assert canonical.dtype == np.float16
    assert canonical == np.float16(float(raw) * u.meter.scale)
    assert isinstance(canonical, np.ndarray) == array_storage


def test_float16_npz_conversion_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "length.npz"
    source = Length[npt.NDArray[np.float64]]([3e-6], u.meter).to(u.meter)
    save_npz(path, length=source)
    restored = load_npz(path, "length", Length[npt.NDArray[np.float16]])
    np.testing.assert_array_equal(restored.value, [np.float16(30000)])
    assert restored.value.dtype == np.float16
    # Loading remembers metres, which float16 can hold only as a subnormal.
    # Presenting in ångströms keeps the round trip exact.
    save_npz(path, length=restored.to(u.angstrom))
    again = load_npz(path, "length", Length[npt.NDArray[np.float16]])
    np.testing.assert_array_equal(again.value, restored.value)
    # Noncanonical float16 wire magnitudes round normally in their final dtype.
    save_npz(path, length=restored.to(u.meter))
    presented = load_npz(path, "length", Length[npt.NDArray[np.float16]])
    expected = np.float16(float(np.float16(30000 / u.meter.scale)) * u.meter.scale)
    np.testing.assert_array_equal(presented.value, [expected])


def test_tiny_input_survives_large_numpy_unit_factor() -> None:
    unit = Length.define_unit("lab:huge", reference=u.angstrom, scale=1e100)
    assert Length[np.float16](1e-100, unit).value == 1


@pytest.mark.parametrize("x64", [False, True])
def test_jax_integer_and_low_precision_boundaries(tmp_path: Path, *, x64: bool) -> None:
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")
    program = f"""
import json
import numpy as np
import jax
import jax.numpy as jnp
from quantype import Length, Time, u
from quantype.serialization import load_npz
jax.config.update('jax_enable_x64', {x64!r})
for raw in (
    np.array([2**40], dtype=np.int64),
    np.array([2**32 + 1], dtype=np.uint64),
    np.int64(2**40), np.uint64(2**32 + 1), [2**40], 2**40,
):
    q = Length[jax.Array](raw, u.angstrom)
    assert q.value.dtype == jnp.float32
    np.testing.assert_array_equal(q.value, np.asarray(raw, dtype=np.float32))
    explicit = Length[jax.Array](raw, u.angstrom, dtype='float32')
    np.testing.assert_array_equal(explicit.value, q.value)
path = {str(tmp_path / "integer.npz")!r}
np.savez(path, array_0=np.array([2**40], dtype=np.int64),
    metadata=np.asarray(json.dumps({{'version': 1, 'quantities': {{
        'q': {{'kind': 'Length', 'unit': 'angstrom', 'array': 'array_0'}}}}}})))
np.testing.assert_array_equal(load_npz(path, 'q', Length[jax.Array]).value, [2**40])
if not {x64!r}:
    try:
        Length[jax.Array]([1], u.meter, dtype='float64')
    except ValueError as exc:
        assert 'unavailable' in str(exc)
    else:
        raise AssertionError('unavailable float64 was accepted')
unit = Length.define_unit('lab:huge', reference=u.angstrom, scale=1e100)
if {x64!r}:
    np.testing.assert_array_equal(Length[jax.Array]([1e-100], unit).value, [1])
else:
    try:
        with np.errstate(over='raise', invalid='raise'):
            Length[jax.Array]([1e-100], unit)
    except ValueError as exc:
        assert 'working precision' in str(exc)
    else:
        raise AssertionError('unrepresentable conversion factor was accepted')
for dtype in (jnp.float16, jnp.bfloat16):
    q = Length[jax.Array]([3e-6], u.meter, dtype=dtype)
    np.testing.assert_allclose(q.value.astype(jnp.float32), [30000], rtol=.01)
    t = Time[jax.Array]([1e-11], u.second, dtype=dtype)
    np.testing.assert_allclose(t.value.astype(jnp.float32), [10000], rtol=.01)
    assert q.value.dtype == dtype
    @jax.jit
    def convert(raw):
        return Length[jax.Array](raw, u.meter, dtype=dtype).value
    np.testing.assert_array_equal(convert(jnp.asarray([3e-6])), q.value)
    result = Length[jax.Array]([30000], u.angstrom, dtype=dtype).magnitude(u.meter)
    np.testing.assert_allclose(result.astype(jnp.float32), [3e-6], rtol=.02)
gradient = jax.grad(lambda raw: Length[jax.Array](raw, u.nm).value)(jnp.asarray(2.))
assert gradient == 10
"""
    subprocess.run([sys.executable, "-c", program], check=True)  # noqa: S603


def test_torch_float16_construction_and_graph() -> None:
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    q = Length[torch.Tensor]([3e-6], u.meter, dtype=torch.float16)
    t = Time[torch.Tensor]([1e-11], u.second, dtype=torch.float16)
    assert q.value.dtype == torch.float16
    assert q.value.item() == 30000
    assert t.value.item() == 10000
    result = Length[torch.Tensor]([30000], u.angstrom, dtype=torch.float16)
    assert result.magnitude(u.meter).item() == float(np.float16(3e-6))
    raw = torch.tensor([3e-6], dtype=torch.float64, requires_grad=True)
    converted = Length[torch.Tensor](raw, u.meter, dtype=torch.float16)
    assert converted.value.device == raw.device
    (gradient,) = torch.autograd.grad(converted.value.sum(), raw)
    np.testing.assert_array_equal(gradient.numpy(), [u.meter.scale])
    unit = Length.define_unit("lab:huge", reference=u.angstrom, scale=1e100)
    assert Length[torch.Tensor]([1e-100], unit, dtype=torch.float16).value.item() == 1
    for numpy_dtype, torch_dtype in (
        (np.float16, torch.float16),
        (np.float32, torch.float32),
        (np.float64, torch.float64),
    ):
        source = np.array([2.0], dtype=numpy_dtype)
        converted = Length[torch.Tensor](source, u.nm)
        assert converted.value.dtype == torch_dtype
        assert converted.value.item() == 20


@pytest.mark.parametrize("shape", [None, (), (2,), (2, 2), (0, 2)])
def test_longdouble_json_primitives(shape: tuple[int, ...] | None) -> None:
    if shape is None:
        scalar = Length[np.longdouble](np.longdouble("1.125"), u.angstrom)
        wire = scalar.to_dict()
        serialized = TypeAdapter(Length[np.longdouble]).dump_json(scalar)
    else:
        array = Length[npt.NDArray[np.longdouble]](np.full(shape, 1.125), u.angstrom)
        wire = array.to_dict()
        serialized = TypeAdapter(Length[npt.NDArray[np.longdouble]]).dump_json(array)
    assert json.loads(json.dumps(wire)) == wire
    assert json.loads(serialized) == wire


@pytest.mark.parametrize("extreme", ["large", "small"])
def test_longdouble_json_rejects_extended_range(extreme: str) -> None:
    if np.finfo(np.longdouble).max == np.finfo(np.float64).max:
        pytest.skip("longdouble has no extended range on this platform")
    raw = (
        np.finfo(np.longdouble).max
        if extreme == "large"
        else np.finfo(np.longdouble).tiny
    )
    q = Length[np.longdouble](raw, u.angstrom)
    with pytest.raises(ValueError, match="range of JSON binary64"):
        q.to_dict()


def test_longdouble_npz_preserves_dtype(tmp_path: Path) -> None:
    path = tmp_path / "extended.npz"
    q = Length[npt.NDArray[np.longdouble]](
        np.array(["1.000000000000000001", "2"], dtype=np.longdouble), u.angstrom
    )
    save_npz(path, q=q)
    restored = load_npz(path, "q", Length[npt.NDArray[np.longdouble]])
    assert restored.value.dtype == np.longdouble
    np.testing.assert_array_equal(restored.value, q.value)

"""Contracts introduced by the refactor, including optional-backend boundaries."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from importlib import metadata
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

import quantype
from quantype import Energy, Length, Temperature, u
from quantype._internal._registry import QUANTITIES
from quantype.serialization import load_npz, save_npz
from quantype.systems import Atomistic
from quantype.testing import assert_allclose

if TYPE_CHECKING:
    from pathlib import Path


def test_typed_constructor_converts_storage_and_units() -> None:
    scalar = Length[np.float64, Atomistic](2, u.length.nanometer)
    assert type(scalar.value) is np.float64
    assert scalar.value == 20
    array = Length[npt.NDArray[np.float32], Atomistic]([[1, 2, 3]], u.nm)
    assert array.value.dtype == np.float32
    np.testing.assert_array_equal(array.value, [[10, 20, 30]])
    raw = np.ones(3)
    assert Length[npt.NDArray[np.float64]](raw, u.length.nanometer).magnitude() is raw
    assert Length.from_value(raw, Atomistic).value is raw
    unbound = Length[npt.NDArray[np.float32]]([[1, 2, 3]], u.nm)
    assert unbound.dtype == np.float32
    np.testing.assert_array_equal(unbound.magnitude(u.angstrom), [[10, 20, 30]])
    wrong: Any = u.eV
    with pytest.raises(ValueError, match="Expected Length"):
        Length[np.float64](2, wrong)
    with pytest.raises(ValueError, match="scalar storage"):
        Length[np.float64]([1, 2], u.nm)
    zero = Temperature[float](0, u.temperature.celsius)
    assert zero.magnitude(u.kelvin) == 273.15


def test_import_defers_numerical_dependencies() -> None:
    code = (
        "import sys; import quantype; "
        "assert not {'numpy', 'scipy', 'torch', 'jax', 'pydantic'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_npz_roundtrip_and_backend_metadata(tmp_path: Path) -> None:
    path = tmp_path / "quantities.npz"
    positions = Length[npt.NDArray[np.float32]]([[1, 2, 3]], u.nm).to(u.nm)
    energy = Energy[np.float64](2, u.eV)
    save_npz(path, positions=positions, energy=energy, record_backend=True)
    restored = load_npz(path, "positions", Length[npt.NDArray[np.float64]])
    assert restored.dtype == np.float64
    np.testing.assert_array_equal(restored.magnitude(), positions.magnitude())
    assert load_npz(path, "energy", Energy[np.float64]).magnitude(u.eV) == 2
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        assert metadata["quantities"]["positions"]["source_backend"] == "numpy"
    with pytest.raises(ValueError, match="Expected Energy"):
        load_npz(path, "positions", Energy[npt.NDArray[np.float64]])


def test_npz_custom_units_and_zero_dimensional_storage(tmp_path: Path) -> None:
    bleb = Temperature.define_unit("lab:bleb", reference=u.celsius, scale=np.sqrt(2))
    point = Temperature[npt.NDArray[np.float64]](1, bleb).to(bleb)
    path = tmp_path / "point.npz"
    save_npz(path, point=point)
    restored = load_npz(
        path, "point", Temperature[npt.NDArray[np.float64]], units=(bleb,)
    )
    assert restored.shape == ()
    assert_allclose(restored, point)


def test_torch_constructor_and_archive(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    raw = torch.tensor([1.0, 2.0], requires_grad=True)
    length = Length[torch.Tensor](raw, u.nm, dtype=torch.float64)
    assert length.dtype == torch.float64
    (gradient,) = torch.autograd.grad(length.magnitude(u.angstrom).sum(), raw)
    np.testing.assert_array_equal(gradient.numpy(), [10, 10])
    path = tmp_path / "torch.npz"
    save_npz(path, length=length, record_backend=True)
    restored = load_npz(path, "length", Length[torch.Tensor], dtype=torch.float64)
    assert not restored.magnitude().requires_grad
    np.testing.assert_array_equal(
        restored.magnitude(u.angstrom).detach().numpy(), [10, 20]
    )


def test_jax_constructor_and_archive(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    import jax  # noqa: PLC0415

    length = Length[jax.Array]([1, 2], u.nm)
    path = tmp_path / "jax.npz"
    save_npz(path, length=length, record_backend=True)
    restored = load_npz(path, "length", Length[jax.Array])
    np.testing.assert_array_equal(restored.magnitude(u.angstrom), [10, 20])


def test_version_matches_the_installed_distribution() -> None:
    assert quantype.__version__ == metadata.version("quantype")
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        getattr(quantype, "missing")  # noqa: B009


def test_each_kind_module_holds_the_runtime_class() -> None:
    for name in QUANTITIES:
        module = importlib.import_module(f"quantype._kinds.{name}")
        assert getattr(module, name) is getattr(quantype, name)

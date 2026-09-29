"""Binary decoding establishes physical and numerical invariants without pickle."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

from quantype import Length, u
from quantype.serialization import load_npz, save_npz

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"version": 2, "quantities": {}},
        {"version": True, "quantities": {}},
        {"version": 1, "quantities": {"q": {"kind": "Length"}}},
    ],
)
def test_malformed_metadata(tmp_path: Path, metadata: object) -> None:
    path = tmp_path / "invalid.npz"
    np.savez(path, metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match=r"archive|Archive"):
        load_npz(path, "q", Length[npt.NDArray[np.float64]])


def test_pickle_arrays_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "object.npz"
    metadata = {
        "version": 1,
        "quantities": {"q": {"kind": "Length", "unit": "angstrom", "array": "value"}},
    }
    np.savez(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        value=np.array([object()], dtype=object),
    )
    with pytest.raises(ValueError, match="allow_pickle=False"):
        load_npz(path, "q", Length[npt.NDArray[np.float64]])
    with pytest.raises(ValueError, match="dtype"):
        save_npz(path, q=Length.from_canonical(np.array([object()], dtype=object)))


def test_npy_uses_an_explicit_external_unit(tmp_path: Path) -> None:
    quantity = Length[npt.NDArray[np.float64]]([1, 2], u.nm)
    path = tmp_path / "positions.npy"
    np.save(path, quantity.magnitude(u.nm), allow_pickle=False)
    restored = Length[npt.NDArray[np.float64]](np.load(path, allow_pickle=False), u.nm)
    np.testing.assert_array_equal(restored.value, quantity.value)


def test_storage_dtype_and_unit_invariants() -> None:
    for value in (True, [True], 1j, [1j], ["1"]):
        with pytest.raises(ValueError, match="real numbers"):
            Length[npt.NDArray[np.float64]](value, u.nm)
    scalar = Length[np.float32](2, u.nm)
    assert type(scalar.sum().value) is np.float32
    assert type(u.sqrt(scalar**2).value) is np.float32
    unit: Any = u.nm
    with pytest.raises(AttributeError, match="assign"):
        unit.scale = 100
    assert u.dimensionless.one(2).value == 2
    assert u.energy_density.energy_density(2).kind == "EnergyDensity"

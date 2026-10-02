"""Binary decoding establishes physical and numerical invariants without pickle."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import quantype.numpy as qnp
from quantype import Length, u
from quantype.serialization import load_npz, save_npz

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(None, id="not-an-object"),
        pytest.param({}, id="missing-fields"),
        pytest.param({"version": 2, "quantities": {}}, id="unknown-version"),
        pytest.param({"version": True, "quantities": {}}, id="boolean-version"),
        pytest.param(
            {"version": 1, "quantities": {"q": {"kind": "Length"}}},
            id="incomplete-quantity",
        ),
        pytest.param({"version": 1, "quantities": []}, id="quantities-not-named"),
        pytest.param({"version": 1, "quantities": {}}, id="quantity-missing"),
    ],
)
def test_malformed_metadata(tmp_path: Path, metadata: object) -> None:
    path = tmp_path / "invalid.npz"
    np.savez(path, metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match=r"archive|Archive"):
        load_npz(path, "q", Length[npt.NDArray[np.float64]])


def test_loading_object_arrays_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "object.npz"
    metadata = {
        "version": 1,
        "quantities": {"q": {"kind": "Length", "unit": "angstrom", "array": "value"}},
    }
    # Bypass quantype's writer to construct an archive it must refuse to load.
    np.savez(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        value=np.array([object()], dtype=object),
    )
    with pytest.raises(ValueError, match="allow_pickle=False"):
        load_npz(path, "q", Length[npt.NDArray[np.float64]])


def test_saving_object_arrays_is_rejected(tmp_path: Path) -> None:
    quantity = Length.from_value(np.array([object()], dtype=object))

    with pytest.raises(ValueError, match="dtype"):
        save_npz(tmp_path / "object.npz", q=quantity)


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
    assert type(qnp.sqrt(scalar**2).value) is np.float32
    unit: Any = u.nm
    with pytest.raises(AttributeError, match="assign"):
        unit.scale = 100
    assert u.dimensionless.one(2).value == 2
    assert u.energy_density.energy_density(2).kind == "EnergyDensity"


def _archive(path: Path, entry: dict[str, str], **arrays: object) -> None:
    metadata = {"version": 1, "quantities": {"q": entry}}
    np.savez(path, metadata=np.asarray(json.dumps(metadata)), **cast("Any", arrays))


@pytest.mark.parametrize(
    ("entry", "arrays", "message"),
    [
        pytest.param(
            {"kind": "Length", "unit": "electron_volt", "array": "value"},
            {"value": np.ones(1)},
            "Expected Length; received Energy",
            id="unit-of-another-kind",
        ),
        pytest.param(
            {"kind": "Length", "unit": "angstrom", "array": "missing"},
            {"value": np.ones(1)},
            "missing the quantity array",
            id="array-missing",
        ),
        pytest.param(
            {"kind": "Length", "unit": "angstrom", "array": "value"},
            {"value": np.array([True])},
            "real numerical dtype",
            id="boolean-array",
        ),
    ],
)
def test_archive_entries_must_match_their_arrays(
    tmp_path: Path, entry: dict[str, str], arrays: dict[str, object], message: str
) -> None:
    path = tmp_path / "invalid.npz"
    _archive(path, entry, **arrays)
    with pytest.raises(ValueError, match=message):
        load_npz(path, "q", Length[npt.NDArray[np.float64]])


def test_archives_need_metadata_and_a_parameterized_target(tmp_path: Path) -> None:
    path = tmp_path / "plain.npz"
    np.savez(path, value=np.ones(1))
    with pytest.raises(ValueError, match="missing physical metadata"):
        load_npz(path, "q", Length[npt.NDArray[np.float64]])
    with pytest.raises(TypeError, match="parameterized quantity type"):
        load_npz(path, "q", cast("Any", Length))


def test_python_scalars_are_recorded_as_python_storage(tmp_path: Path) -> None:
    path = tmp_path / "scalar.npz"
    save_npz(path, record_backend=True, q=2 * u.nm)
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata"]))
    assert metadata["quantities"]["q"]["source_backend"] == "python"


def test_numpy_storage_annotations_must_be_real_floating() -> None:
    with pytest.raises(ValueError, match="real floating dtype"):
        Length[npt.NDArray[np.int64]]([1], u.nm)
    with pytest.raises(ValueError, match="conflicts with the NumPy storage"):
        Length[npt.NDArray[np.float64]]([1.0], u.nm, dtype=np.float32)
    with pytest.raises(TypeError, match="already specified by the storage type"):
        Length[np.float64](1.0, u.nm, dtype=np.float32)
    with pytest.raises(TypeError, match="Unsupported quantity storage"):
        cast("Any", Length)[object](1.0, u.nm)
    spelled = Length[np.ndarray[Any, np.dtype[np.float32]]]([1.0], u.nm)
    assert spelled.value.dtype == np.float32
    unspecified = Length[npt.NDArray[Any]]([1.0], u.nm)
    assert unspecified.value.dtype == np.float64


def test_numpy_scalar_storage_keeps_its_dtype_when_scaled() -> None:
    scaled = Length[np.float32](2, u.nm) * np.float64(2)
    assert type(scaled.value) is np.float32

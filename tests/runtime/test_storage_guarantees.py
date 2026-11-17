"""Storage guarantees that readers handing out read-only arrays rely on."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import quantype
import quantype.numpy as qnp
from quantype import Energy, Force, Length, Quantity, u
from quantype.catalogue import builtin_catalogue
from quantype.products import LengthEnergy, LengthTime
from quantype.systems import ASE, SI, Atomistic, NoSystem

IN_PLACE = (
    "__iadd__",
    "__isub__",
    "__imul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__imatmul__",
    "__setitem__",
    "__delitem__",
)


def _read_only(shape: tuple[int, ...] = (4, 3)) -> npt.NDArray[np.float64]:
    raw = np.arange(float(np.prod(shape))).reshape(shape)
    raw.flags.writeable = False
    return raw


def test_from_value_keeps_the_array_it_is_given() -> None:
    raw = _read_only()
    for forces in (
        Force.from_value(raw, Atomistic),
        Force.from_value(raw, SI),
    ):
        assert forces.value is raw
        assert not forces.value.flags.writeable


def test_construction_without_a_system_keeps_the_array_it_is_given() -> None:
    raw = _read_only()
    for unit in (u.eV_per_angstrom, u.hartree_per_bohr):
        forces = Force[npt.NDArray[np.float64]](raw, unit)
        assert forces.magnitude(unit) is raw
        assert forces.magnitude() is raw


def test_the_storage_unit_test_is_exact_equality() -> None:
    assert Atomistic.unit_for(Force) == u.eV_per_angstrom
    assert SI.unit_for(Force) == u.newton
    assert Atomistic.unit_for(Force) != u.hartree_per_bohr
    renamed = Force.define_unit("lab:eV_per_A", reference=u.eV_per_angstrom)
    assert renamed.scale == u.eV_per_angstrom.scale
    assert Atomistic.unit_for(Force) != renamed


def test_basic_indexing_shares_memory() -> None:
    raw = _read_only()
    forces = Force[npt.NDArray[np.float64]](raw, u.eV_per_angstrom)
    for key in (slice(1, 3), 0, (slice(None), 1), np.s_[::2]):
        selected = forces[key]
        assert np.shares_memory(selected.magnitude(), raw)
        assert not selected.magnitude().flags.writeable


def test_integer_and_boolean_array_indexing_select_rows() -> None:
    forces = Force[npt.NDArray[np.float64]](_read_only(), u.eV_per_angstrom)
    rows = forces[np.array([0, 2])]
    np.testing.assert_array_equal(rows.magnitude(), forces.magnitude()[[0, 2]])
    mask = forces[np.array([True, False, True, False])]
    np.testing.assert_array_equal(mask.magnitude(), rows.magnitude())
    assert isinstance(rows, Force)


def test_an_element_is_zero_dimensional_until_item() -> None:
    forces = Force.from_value(_read_only(), SI)
    element = forces[1, 2]
    assert isinstance(element.value, np.ndarray)
    assert element.value.shape == ()
    scalar = element.item()
    assert type(scalar.value) is float
    assert scalar.value == 5.0
    assert scalar.system is SI
    assert isinstance(scalar, Force)


def test_to_numpy_returns_float64_storage_as_the_same_array() -> None:
    raw = _read_only((3,))
    forces = Force.from_value(raw, SI)
    converted = forces.to_numpy()
    assert converted.value is raw
    assert converted.system is SI
    assert isinstance(converted, Force)


@pytest.mark.parametrize(
    "raw", [1.5, np.float64(1.5), np.array([1.5, 2.0], dtype=np.float32)]
)
def test_to_numpy_copies_other_storage_into_float64(raw: object) -> None:
    energy = Energy.from_value(raw, ASE)
    converted = energy.to_numpy()
    stored = converted.value
    assert type(stored) is np.ndarray
    assert stored.dtype == np.float64
    assert not np.shares_memory(stored, np.asarray(raw))
    np.testing.assert_array_equal(stored, np.asarray(raw, dtype=np.float64))


def test_to_numpy_keeps_kind_unit_and_system() -> None:
    energy = Energy[float](1.5, u.hartree)
    converted = energy.to_numpy()
    assert isinstance(converted, Energy)
    assert converted.system is NoSystem
    assert converted.unit is u.hartree
    assert converted.item() == energy
    product = (Length.from_value(2.0, SI) * Energy.from_value(3.0, SI)).to_numpy()
    assert isinstance(product, LengthEnergy)
    assert product.value.shape == ()


def test_stacking_float_storage_gives_float64_array_storage() -> None:
    energies = qnp.stack([Energy.from_value(1.0, ASE), Energy.from_value(2.0, ASE)])
    assert isinstance(energies, Energy)
    assert energies.value.dtype == np.float64
    assert energies.value.shape == (2,)


CLASSES: list[object] = [
    Quantity,
    LengthTime,
    *(getattr(quantype, name) for name in builtin_catalogue().quantities),
]


@pytest.mark.parametrize("cls", CLASSES)
def test_quantities_have_no_in_place_operators(cls: object) -> None:
    for name in IN_PLACE:
        assert not hasattr(cls, name), name


def test_augmented_assignment_rebinds_and_leaves_storage_alone() -> None:
    raw = _read_only((3,))
    before = raw.copy()
    lengths = Length[npt.NDArray[np.float64]](raw, u.angstrom)
    original = lengths
    lengths += Length[npt.NDArray[np.float64]](raw, u.nm)
    lengths *= 2.0
    lengths -= original
    assert lengths is not original
    assert original.magnitude() is raw
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(lengths.magnitude(u.angstrom), 21 * before)


def test_numpy_cannot_write_into_a_quantity() -> None:
    lengths: Any = Length[npt.NDArray[np.float64]](np.zeros(3), u.nm)
    with pytest.raises(TypeError, match="does not know the units"):
        np.add(lengths, lengths, out=lengths)
    plain = np.zeros(3)
    with pytest.raises(TypeError):
        plain[:] = lengths


def test_the_installed_version_is_public() -> None:
    assert isinstance(quantype.__version__, str)
    assert quantype.__version__.count(".") >= 2

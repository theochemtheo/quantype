"""Storage guarantees that readers handing out read-only arrays rely on."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import quantype
from quantype import Force, Length, Quantity, u
from quantype.catalogue import builtin_catalogue
from quantype.products import LengthTime
from quantype.systems import SI, Atomistic

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
        Force.from_value(raw),
        Force[npt.NDArray[np.float64]].from_value(raw),
        Force[npt.NDArray[np.float64], SI].from_value(raw),
    ):
        assert forces.value is raw
        assert not forces.value.flags.writeable


def test_the_storage_unit_test_is_exact_equality() -> None:
    assert Atomistic.unit_for(Force) == u.eV_per_angstrom
    assert SI.unit_for(Force) == u.newton
    assert Atomistic.unit_for(Force) != u.hartree_per_bohr
    renamed = Force.define_unit("lab:eV_per_A", reference=u.eV_per_angstrom)
    assert renamed.scale == u.eV_per_angstrom.scale
    assert Atomistic.unit_for(Force) != renamed


def test_basic_indexing_shares_memory() -> None:
    raw = _read_only()
    forces = Force.from_value(raw)
    for key in (slice(1, 3), 0, (slice(None), 1), np.s_[::2]):
        selected = forces[key]
        assert np.shares_memory(selected.value, raw)
        assert not selected.value.flags.writeable


def test_integer_and_boolean_array_indexing_select_rows() -> None:
    forces = Force.from_value(_read_only())
    rows = forces[np.array([0, 2])]
    np.testing.assert_array_equal(rows.value, forces.value[[0, 2]])
    mask = forces[np.array([True, False, True, False])]
    np.testing.assert_array_equal(mask.value, rows.value)
    assert isinstance(rows, Force)


def test_an_element_is_zero_dimensional_until_item() -> None:
    forces = Force[npt.NDArray[np.float64], SI].from_value(_read_only())
    element = forces[1, 2]
    assert isinstance(element.value, np.ndarray)
    assert element.value.shape == ()
    scalar = element.item()
    assert type(scalar.value) is float
    assert scalar.value == 5.0
    assert scalar.system is SI
    assert isinstance(scalar, Force)


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
    lengths = Length.from_value(raw)
    original = lengths
    lengths += Length.from_value(raw)
    lengths *= 2.0
    lengths -= original
    assert lengths is not original
    assert original.value is raw
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(lengths.value, 3 * before)


def test_numpy_cannot_write_into_a_quantity() -> None:
    lengths: Any = Length.from_value(np.zeros(3))
    with pytest.raises(TypeError, match="does not know the units"):
        np.add(lengths, lengths, out=lengths)
    plain = np.zeros(3)
    with pytest.raises(TypeError):
        plain[:] = lengths


def test_the_installed_version_is_public() -> None:
    assert isinstance(quantype.__version__, str)
    assert quantype.__version__.count(".") >= 2

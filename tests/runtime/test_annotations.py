"""quantity_type reads a quantity annotation's kind, storage, and unit system."""

from __future__ import annotations

import types
from typing import Annotated, Any, cast, get_args, get_type_hints

import numpy as np
import numpy.typing as npt
import pytest

from quantype import Energy, Force, Length, Quantity, u
from quantype.kinds import ForceKind
from quantype.products import LengthTime
from quantype.systems import SI, Atomistic, UnitSystem
from quantype.typing import QuantityType, quantity_type

ARRAY = npt.NDArray[np.float64]


class _Gromacs(UnitSystem, name="test-annotations:gromacs"):
    length = u.nanometer
    energy = u.kJ_per_mol
    time = u.picosecond


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (Force[float], QuantityType(Force, float, Atomistic)),
        (Force[npt.NDArray[np.float64]], QuantityType(Force, ARRAY, Atomistic)),
        (Force[float, Atomistic], QuantityType(Force, float, Atomistic)),
        (Force[float, SI], QuantityType(Force, float, SI)),
        (Force[np.float32, _Gromacs], QuantityType(Force, np.float32, _Gromacs)),
        (LengthTime[float], QuantityType(LengthTime, float, Atomistic)),
        (Annotated[Force[float], "meta"], QuantityType(Force, float, Atomistic)),
        (Force, QuantityType(Force, None, Atomistic)),
    ],
)
def test_annotations_give_kind_storage_and_system(
    annotation: object, expected: QuantityType
) -> None:
    assert quantity_type(annotation) == expected


@pytest.mark.parametrize(
    "annotation",
    [float, np.float64, ARRAY, list[str], None, 3, "Force"],
)
def test_other_annotations_are_not_quantity_types(annotation: object) -> None:
    assert quantity_type(annotation) is None


def test_an_optional_quantity_is_a_union_holding_one() -> None:
    optional = Force[float] | None
    assert isinstance(optional, types.UnionType)
    assert quantity_type(optional) is None
    inner = [quantity_type(member) for member in get_args(optional)]
    assert inner == [QuantityType(Force, float, Atomistic), None]


class _Record:
    forces: Annotated[Force[npt.NDArray[np.float64]], "REF_forces"]
    energy: Energy[float, SI] | None


def test_class_annotations_read_through_get_type_hints() -> None:
    hints = get_type_hints(_Record, include_extras=True)
    assert quantity_type(hints["forces"]) == QuantityType(Force, ARRAY, Atomistic)
    energy, none = get_args(hints["energy"])
    assert quantity_type(energy) == QuantityType(Energy, float, SI)
    assert none is type(None)


def test_subscripting_the_kind_changes_the_storage() -> None:
    scalar = quantity_type(Energy[float, SI])
    assert scalar is not None
    kind: Any = scalar.kind
    batched = kind[ARRAY, scalar.system]
    assert quantity_type(batched) == QuantityType(Energy, ARRAY, SI)
    assert quantity_type(kind[ARRAY]) == QuantityType(Energy, ARRAY, Atomistic)
    raw = np.array([1.0, 2.0])
    assert batched.from_value(raw).value is raw


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        ("Force[SI]", "storage type first"),
        ("Force[float, int]", "Expected a unit system"),
        ("Force[float, SI, int]", "more than a storage type and a system"),
        ("Quantity[ForceKind, float]", "not a named quantity type"),
        ("Quantity", "not a named quantity type"),
    ],
)
def test_malformed_quantity_annotations_raise(annotation: str, message: str) -> None:
    namespace = {"Force": Force, "Quantity": Quantity, "ForceKind": ForceKind}
    resolved = eval(annotation, {"SI": SI, **namespace})  # noqa: S307 -- fixed text
    with pytest.raises(TypeError, match=message):
        quantity_type(resolved)


def test_constructing_with_too_many_type_arguments_raises() -> None:
    alias = cast("Any", Length)[float, SI, int]
    with pytest.raises(TypeError, match="more than a storage type and a system"):
        alias(1, u.nm)

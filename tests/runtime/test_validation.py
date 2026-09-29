"""Boundary validation and reusable shape-constructor demonstrations."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import BaseModel, TypeAdapter, ValidationError

from quantype import Force, Length, Pressure, Temperature, Time
from quantype import units as u
from quantype._validation import parse_quantity, serialize_quantity
from quantype.shapes import SymmetricTensor3, Vector3


class Simulation(BaseModel):
    timestep: Time[float]
    cutoff: Length[float]
    positions: Length[NDArray[np.float64]]
    temperature: Temperature[float]


def test_scalar_roundtrip_and_display_unit() -> None:
    length = parse_quantity(Length, {"value": 0.5, "units": "nm"})
    assert length.kind == "Length"
    assert length.value == pytest.approx(5.0)
    assert serialize_quantity(length) == {"value": 5.0, "units": "angstrom"}
    assert serialize_quantity(length, u.nm) == {"value": 0.5, "units": "nanometer"}
    assert serialize_quantity(length.to(u.nm)) == {"value": 0.5, "units": "nanometer"}
    assert parse_quantity(Length, length) is length
    assert parse_quantity(Length, "5 angstrom").value == pytest.approx(5.0)
    restored = parse_quantity(
        Length, json.loads(json.dumps(serialize_quantity(length)))
    )
    assert restored.value == length.value


def test_array_roundtrip() -> None:
    quantity = parse_quantity(Length, {"value": [[1, 2, 3]], "units": "nm"})
    values: Any = quantity.value
    assert isinstance(values, np.ndarray)
    restored = parse_quantity(
        Length, json.loads(json.dumps(serialize_quantity(quantity, u.nm)))
    )
    np.testing.assert_allclose(restored.value, quantity.value)


def test_pydantic_model_roundtrip() -> None:
    model = Simulation.model_validate(
        {
            "timestep": "2 fs",
            "cutoff": {"value": 0.5, "units": "nm"},
            "positions": {"value": [[0, 0, 0], [1, 2, 3]], "units": "angstrom"},
            "temperature": "20 degC",
        }
    )
    assert model.cutoff.value == pytest.approx(5.0)
    assert model.temperature.value == pytest.approx(293.15)
    restored = Simulation.model_validate_json(model.model_dump_json())
    np.testing.assert_array_equal(restored.positions.value, model.positions.value)
    assert restored.cutoff.value == model.cutoff.value


def test_bare_annotation_and_useful_json_schema() -> None:
    # Bare named classes intentionally support runtime validation.
    adapter: TypeAdapter[Any] = TypeAdapter(Length)
    assert adapter.validate_python("5 angstrom").value == 5.0
    schema = Simulation.model_json_schema()
    cutoff = schema["properties"]["cutoff"]["anyOf"][0]
    assert cutoff["properties"]["value"]["type"] == "number"
    assert "angstrom" in cutoff["properties"]["units"]["enum"]
    assert set(cutoff["required"]) == {"value", "units"}
    output_schema = Simulation.model_json_schema(mode="serialization")
    position_value = output_schema["properties"]["positions"]["properties"]["value"]
    assert position_value["type"] == "array"


def test_storage_annotations_do_not_lie() -> None:
    scalar = TypeAdapter(Length[float])
    array = TypeAdapter(Length[NDArray[np.float64]])
    with pytest.raises(ValidationError, match="storage"):
        scalar.validate_python({"value": [1, 2], "units": "angstrom"})
    with pytest.raises(ValidationError, match="storage"):
        array.validate_python("1 angstrom")
    with pytest.raises(ValidationError, match="storage"):
        scalar.validate_python(u.angstrom(np.array([1.0])))


@pytest.mark.parametrize(
    "data",
    [
        {"value": 1, "units": "eV"},
        "1 eV",
        u.eV(1.0),
    ],
)
def test_wrong_kind_errors(data: object) -> None:
    with pytest.raises(ValueError, match="Expected Length; received Energy"):
        parse_quantity(Length, data)


def test_dimensionally_equal_semantics_are_not_interchangeable() -> None:
    density = u.eV(1.0) / (u.angstrom(1.0) ** 3)
    with pytest.raises(ValueError, match="Expected Pressure; received EnergyDensity"):
        parse_quantity(Pressure, density)


@pytest.mark.parametrize(
    "data",
    [
        {"value": 1},
        {"value": 1, "units": "angstrom", "extra": 1},
        {"value": True, "units": "angstrom"},
        {"value": ["not numeric"], "units": "angstrom"},
        {"value": [[1], [1, 2]], "units": "angstrom"},
        {"value": 1, "units": "__import__('os')"},
        {"value": 1, "units": 42},
        "not-a-number angstrom",
        "angstrom",
        42,
    ],
)
def test_invalid_payload(data: object) -> None:
    with pytest.raises(ValueError, match=r".+"):
        parse_quantity(Length, data)


def test_shapes_compose_with_physical_kinds() -> None:
    position = u.angstrom(Vector3([1, 2, 3]))
    force = u.eV_per_angstrom(Vector3([4, 5, 6]))
    stress = u.eV_per_angstrom_cubed(SymmetricTensor3(np.eye(3)))
    assert isinstance(position, Length)
    assert isinstance(force, Force)
    assert isinstance(stress, Pressure)
    assert position.value.shape == force.value.shape == (3,)
    assert stress.value.shape == (3, 3)
    with pytest.raises(ValueError, match="shape"):
        Vector3([1, 2])
    with pytest.raises(ValueError, match="shape"):
        SymmetricTensor3(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="symmetric"):
        SymmetricTensor3([[1, 2, 0], [0, 1, 0], [0, 0, 1]])


def test_compound_unit_and_public_parse_boundary() -> None:
    force = Force.parse({"value": 0.01, "units": "eV / angstrom"})
    assert force.kind == "Force"
    assert force.value == 0.01
    length = Length.parse({"value": [1, 2], "units": "nm"})
    np.testing.assert_array_equal(length.value, [10, 20])
    converted = Length.parse(u.angstrom(np.array([1, 2])))
    values: Any = converted.value
    assert values.dtype == np.float64
    with pytest.raises(ValueError, match="Expected Length"):
        Length.parse({"value": 1.0, "units": "second"})


def test_unnamed_serialization_has_clear_error() -> None:
    quantity = (2 * u.angstrom) * (3 * u.fs)
    with pytest.raises(ValueError, match="named quantity"):
        quantity.to_dict()

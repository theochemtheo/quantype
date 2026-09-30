"""Physical meaning, typed restoration, and custom units at external boundaries."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import BaseModel, TypeAdapter, ValidationError

from quantype import Length, Pressure, Temperature, Time, u
from quantype.serialization import parse_quantity, to_dict


class Simulation(BaseModel):
    timestep: Time[float]
    cutoff: Length[float]
    positions: Length[NDArray[np.float64]]
    temperature: Temperature[float]


def test_scalar_roundtrip_and_display_unit() -> None:
    length = Length.parse({"kind": "Length", "magnitude": 0.5, "unit": "nm"})
    assert length.value == pytest.approx(5)
    assert to_dict(length) == {"kind": "Length", "magnitude": 5.0, "unit": "angstrom"}
    wire = {"kind": "Length", "magnitude": 0.5, "unit": "nanometer"}
    assert to_dict(length, u.nm) == wire
    assert length.to(u.nm).to_dict() == wire
    assert parse_quantity(Length, length) is length
    assert Length.parse(wire).value == length.value


def test_pydantic_roundtrip() -> None:
    model = Simulation.model_validate(
        {
            "timestep": "2 fs",
            "cutoff": "0.5 nm",
            "temperature": "20 degC",
            "positions": {"kind": "Length", "magnitude": [[1, 2, 3]], "unit": "nm"},
        }
    )
    restored = Simulation.model_validate_json(model.model_dump_json())
    np.testing.assert_array_equal(restored.positions.value, [[10, 20, 30]])
    assert restored.cutoff.value == model.cutoff.value
    assert restored.temperature.value == pytest.approx(293.15)


def test_schema_and_bare_annotation() -> None:
    adapter: TypeAdapter[Any] = TypeAdapter(Length)
    assert adapter.validate_python("5 angstrom").value == 5
    schema = Simulation.model_json_schema()["properties"]["cutoff"]["anyOf"][0]
    assert set(schema["required"]) == {"kind", "magnitude", "unit"}
    assert schema["properties"]["magnitude"]["type"] == "number"


@pytest.mark.parametrize("storage", [float, np.float16, np.float32, np.float64])
def test_scalar_json_schema(storage: object) -> None:
    quantity_class: Any = Length
    adapter: TypeAdapter[Any] = TypeAdapter(quantity_class[storage])
    schemas = (
        adapter.json_schema()["anyOf"][0],
        adapter.json_schema(mode="serialization"),
    )
    for schema in schemas:
        assert schema["properties"]["magnitude"]["type"] == "number"


def test_zero_dimensional_array_json_schema_and_roundtrip() -> None:
    adapter = TypeAdapter(Length[NDArray[np.float64]])
    original = Length[NDArray[np.float64]](2, u.nm)
    schemas = (
        adapter.json_schema()["anyOf"][0],
        adapter.json_schema(mode="serialization"),
    )
    for schema in schemas:
        magnitude = schema["properties"]["magnitude"]
        assert {branch["type"] for branch in magnitude["anyOf"]} == {"number", "array"}
    # Catch disagreement with the serializer's return schema on all Pydantic v2s.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wire = adapter.dump_json(original)
    restored = adapter.validate_json(wire)
    assert restored.value.shape == ()
    assert restored.value.dtype == np.float64
    assert restored.value == original.value


def test_storage_conversion() -> None:
    scalar = TypeAdapter(Length[np.float64])
    assert type(scalar.validate_python("2 nm").value) is np.float64
    array = TypeAdapter(Length[NDArray[np.float32]])
    result = array.validate_python(
        {"kind": "Length", "magnitude": [1, 2], "unit": "nm"}
    )
    assert result.value.dtype == np.float32
    with pytest.raises(ValidationError, match="storage"):
        scalar.validate_python(u.angstrom(np.array([1.0])))


@pytest.mark.parametrize(
    "data", ["1 eV", u.eV(1), {"kind": "Energy", "magnitude": 1, "unit": "eV"}]
)
def test_wrong_kind(data: object) -> None:
    with pytest.raises(ValueError, match="Expected Length; received Energy"):
        Length.parse(data)


def test_nominal_semantics() -> None:
    density = u.eV(1) / u.angstrom(1) ** 3
    with pytest.raises(ValueError, match="Expected Pressure; received EnergyDensity"):
        Pressure.parse(density)
    with pytest.raises(ValueError, match="named quantity"):
        (u.angstrom(2) * u.fs(3)).to_dict()


@pytest.mark.parametrize("value", [True, [True], ["bad"], [[1], [1, 2]]])
def test_invalid_magnitude(value: object) -> None:
    with pytest.raises(ValueError, match=r".+"):
        Length.parse({"kind": "Length", "magnitude": value, "unit": "angstrom"})


@pytest.mark.parametrize(
    "data",
    [
        42,
        "angstrom",
        "bad angstrom",
        {"magnitude": 1},
        {"kind": "Length", "magnitude": 1, "unit": "__import__('os')"},
    ],
)
def test_invalid_payload(data: object) -> None:
    with pytest.raises(ValueError, match=r".+"):
        Length.parse(data)


def test_custom_affine_unit() -> None:
    bleb = Temperature.define_unit(
        "lab:bleb", reference=u.temperature.celsius, scale=math.sqrt(2)
    )
    zero = Temperature[np.float64](0, bleb)
    one = Temperature[np.float64](1, bleb)
    assert zero.value == pytest.approx(273.15)
    assert one.magnitude(u.celsius) == pytest.approx(math.sqrt(2))
    wire = one.to(bleb).to_dict()
    assert wire["unit"] == "lab:bleb"
    with pytest.raises(ValueError, match="Unknown unit"):
        Temperature.parse(wire)
    restored = Temperature.parse(wire, units=(bleb,))
    assert restored.value == one.value
    adapter = TypeAdapter(Temperature[np.float64])
    assert adapter.validate_python(wire, context={"units": (bleb,)}).value == one.value
    with pytest.raises(ValueError, match="Duplicate"):
        Temperature.parse(wire, units=(bleb, bleb))


def test_invalid_unit_definitions() -> None:
    for scale in (0, -1, math.nan, math.inf):
        with pytest.raises(ValueError, match="scale"):
            Length.define_unit("invalid", reference=u.nm, scale=scale)
    with pytest.raises(ValueError, match="offset"):
        Length.define_unit("shifted", reference=u.nm, offset=1)

"""Physical meaning, typed restoration, and custom units at external boundaries."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import BaseModel, TypeAdapter, ValidationError

from quantype import (
    Dimensionless,
    Energy,
    Force,
    Length,
    Pressure,
    Temperature,
    Time,
    u,
)
from quantype._internal._registry import unit_specs
from quantype.core import Quantity, Unit, get_unit
from quantype.serialization import parse_quantity, resolve_unit, to_dict
from quantype.systems import UnitSystem


class Simulation(BaseModel):
    timestep: Time[float]
    cutoff: Length[float]
    positions: Length[NDArray[np.float64]]
    temperature: Temperature[float]


def test_scalar_roundtrip_and_display_unit() -> None:
    length = Length.parse({"kind": "Length", "magnitude": 0.5, "unit": "nm"})
    assert length.value == pytest.approx(5)
    # The input unit is remembered, so a config's "0.5 nm" is written back as-is.
    wire = {"kind": "Length", "magnitude": 0.5, "unit": "nanometer"}
    assert to_dict(length) == wire
    assert to_dict(length, u.angstrom) == {
        "kind": "Length",
        "magnitude": 5.0,
        "unit": "angstrom",
    }
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
    assert set(schema["required"]) == {"magnitude", "unit"}  # the field names the kind
    assert schema["properties"]["magnitude"]["type"] == "number"
    written = Simulation.model_json_schema(mode="serialization")["properties"]
    assert set(written["cutoff"]["required"]) == {"kind", "magnitude", "unit"}


def test_objects_may_leave_out_the_kind_their_field_names() -> None:
    assert Length.parse({"magnitude": 2, "unit": "nm"}) == 2 * u.nm
    adapter = TypeAdapter(Length[float])
    assert adapter.validate_python({"magnitude": 2, "unit": "nm"}) == 2 * u.nm
    with pytest.raises(ValidationError, match="Expected Length; received Energy"):
        adapter.validate_python({"kind": "Energy", "magnitude": 2, "unit": "eV"})
    with pytest.raises(ValidationError, match="and optionally 'kind'"):
        adapter.validate_python({"magnitude": 2})


def test_python_mode_dumps_keep_quantities() -> None:
    adapter = TypeAdapter(Length[float])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert adapter.dump_python(2 * u.nm) == 2 * u.nm
        assert adapter.dump_python(2 * u.nm, mode="json") == {
            "kind": "Length",
            "magnitude": 2.0,
            "unit": "nanometer",
        }


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5 nm", 50.0),
        ("5nm", 50.0),
        ("  -1.5e-1  nm ", -1.5),
        ("1e-9m", 10.0),
        (".5 Angstrom", 0.5),
        ("2 Ang", 2.0),
    ],
)
def test_quantity_strings_allow_an_optional_space(text: str, expected: float) -> None:
    assert Length.parse(text).value == pytest.approx(expected)
    assert TypeAdapter(Length[float]).validate_python(text).value == pytest.approx(
        expected
    )


@pytest.mark.parametrize("text", ["nm", "five nm", ""])
def test_quantity_strings_need_a_number_and_a_unit(text: str) -> None:
    with pytest.raises(ValueError, match="Expected '<number> <unit>'"):
        Length.parse(text)


def test_quantity_string_schema_is_portable() -> None:
    # JSON Schema patterns are ECMA-262: no Python-only named groups or flags.
    pattern = TypeAdapter(Length[float]).json_schema()["anyOf"][1]["pattern"]
    assert "(?P<" not in pattern
    assert "(?i" not in pattern


def _every_unit() -> list[Unit[Any]]:
    return list(dict.fromkeys(get_unit(name) for name in unit_specs()))


@pytest.mark.parametrize("unit", _every_unit(), ids=lambda unit: unit.name)
def test_printed_quantities_parse_back(unit: Unit[Any]) -> None:
    # Calls are typed on each kind's own unit class; this test spans all kinds.
    quantity = cast("Quantity[Any, float, Any]", cast("Any", unit)(1.5))
    restored = type(quantity).parse(str(quantity))
    assert restored.unit is unit
    assert restored == quantity


@pytest.mark.parametrize("text", ["5", "5 ", "15", "1.5", "1e5", " -2 "])
def test_a_number_without_a_unit_says_so(text: str) -> None:
    with pytest.raises(ValueError, match="has no unit"):
        Length.parse(text)


def test_dimensionless_values_parse_from_a_bare_number() -> None:
    assert Dimensionless.parse("0.5") == u.one(0.5)
    assert Dimensionless.parse(str(u.one(0.25))) == u.one(0.25)
    adapter = TypeAdapter(Dimensionless[float])
    assert adapter.validate_python("0.5") == u.one(0.5)
    with pytest.raises(ValidationError, match="has no unit"):
        TypeAdapter(Length[float]).validate_python("0.5")


class _Gromacs(UnitSystem, name="test-validation:gromacs"):
    length = u.nanometer
    energy = u.kJ_per_mol
    time = u.picosecond


def test_system_units_parse_from_their_symbols() -> None:
    force = Force[float, _Gromacs].from_value(2.0)
    assert force.unit is not None
    assert force.unit.symbol == "kJ/mol/nm"
    assert Force[float, _Gromacs].parse(str(force)) == force


def test_an_ambiguous_symbol_names_the_candidates() -> None:
    first = Length.define_unit("lab:first", reference=u.nm, symbol="lu")
    second = Length.define_unit("lab:second", reference=u.nm, scale=2.0, symbol="lu")
    with pytest.raises(ValueError, match="lab:first, lab:second"):
        Length.parse("1 lu", units=(first, second))


def test_units_must_be_named_by_strings_of_the_expected_kind() -> None:
    with pytest.raises(ValueError, match="Unit identifier must be a string"):
        Length.parse({"magnitude": 1, "unit": 5})
    custom = Energy.define_unit("lab:spark", reference=u.eV, scale=2.0)
    with pytest.raises(ValueError, match="Expected Length; received Energy"):
        Length.parse({"magnitude": 1, "unit": "lab:spark"}, units=(custom,))
    with pytest.raises(ValueError, match="Unknown unit"):
        resolve_unit("lab:nothing")


def test_parse_restores_float64_array_storage() -> None:
    narrow = Length[NDArray[np.float32]]([1.0], u.nm)
    restored = Length.parse(narrow).value
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == np.float64


def test_parse_keeps_float64_arrays_as_they_are() -> None:
    restored = Length.parse({"magnitude": [1.0, 2.0], "unit": "nm"}).value
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == np.float64

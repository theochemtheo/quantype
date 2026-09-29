"""Explicit wire-format boundaries; numerical kernels never call this module."""

# Pydantic catches ValueError, not TypeError. This module implements core's
# private boundary hooks and necessarily inspects its presentation metadata.
# ruff: noqa: TRY004, SLF001
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAliasType,
    TypeGuard,
    cast,
    get_args,
    get_origin,
)

import numpy as np
import numpy.typing as npt
from pydantic_core import core_schema

from quantype._registry import QUANTITIES, UNITS
from quantype.core import Quantity, Unit, get_unit

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler


def _is_quantity(value: object) -> TypeGuard[Quantity[Any, Any]]:
    return isinstance(value, Quantity)


def _wire_value(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("Quantity values must be numbers, not booleans")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            array: npt.NDArray[Any] = np.asarray(cast("Any", value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Quantity values must be rectangular numeric arrays"
            ) from exc
        if array.dtype.kind not in "iuf":
            raise ValueError("Quantity arrays must contain real numbers")
        return array.astype(np.float64)
    raise ValueError("Quantity value must be a number or a numeric array")


def parse_quantity(cls: type[Quantity[Any, Any]], data: object) -> Quantity[Any, Any]:
    """Parse a quantity, enforcing nominal kind rather than dimensional equality."""
    expected = cls._kind
    if _is_quantity(data):
        if data.kind != expected:
            raise ValueError(f"Expected {expected}; received {data.kind}")
        return data
    if isinstance(data, str):
        pieces = data.strip().split(maxsplit=1)
        if len(pieces) != 2:  # noqa: PLR2004 - magnitude and unit
            raise ValueError(f"Expected {expected} as '<number> <unit>'")
        try:
            value: object = float(pieces[0])
        except ValueError as exc:
            raise ValueError(f"Expected {expected} with a numeric magnitude") from exc
        name: object = pieces[1]
    elif isinstance(data, Mapping):
        payload = cast("Mapping[object, object]", data)
        if set(payload) != {"value", "units"}:
            raise ValueError("Quantity objects require exactly 'value' and 'units'")
        value, name = payload["value"], payload["units"]
    else:
        raise ValueError(
            f"Expected {expected} as a quantity, string, or value/units object"
        )
    if not isinstance(name, str):
        raise ValueError(f"Expected {expected} with a string unit name")
    unit = get_unit(name)
    if unit.kind != expected:
        raise ValueError(f"Expected {expected}; received {unit.kind} (unit {name!r})")
    # Catalogue lookup is a dynamic boundary; named unit constructors carry the
    # more precise public callable signatures in generated stubs.
    return cast("Quantity[Any, Any]", cast("Any", unit)(_wire_value(value)))


def serialize_quantity(
    q: Quantity[Any, Any], unit: Unit[Any] | None = None
) -> dict[str, object]:
    """Export a scalar/list payload; accelerator transfers happen only here."""
    if q.kind not in QUANTITIES:
        raise ValueError(
            "Serialization requires a named quantity with a registered unit; "
            "rewrap canonical data in an explicitly chosen physical kind"
        )
    name = (
        unit.name
        if unit is not None
        else q._display or QUANTITIES[q.kind].canonical_unit
    )
    selected = get_unit(name)
    value: Any = q.magnitude(selected)
    # PyTorch requires an explicit detach and host transfer. JAX arrays support
    # NumPy conversion here, outside tracing and differentiable computation.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, (int, float)):
        raw: object = float(value)
    else:
        raw = np.asarray(value).tolist()
    return {"value": raw, "units": selected.name}


def pydantic_schema(
    cls: type[Quantity[Any, Any]],
    source_type: Any,  # noqa: ANN401 - Pydantic's schema hook contract
    handler: GetCoreSchemaHandler,
) -> core_schema.CoreSchema:
    """Build a Pydantic v2 boundary with an honest numerical-storage check."""
    del handler
    arguments = get_args(source_type)
    storage = arguments[-1] if arguments else Any
    storage_origin = get_origin(storage) or storage
    # NumPy's NDArray is a PEP 695 alias, not directly an ndarray GenericAlias.
    # Resolve aliases for the storage-class check without claiming shape typing.
    while isinstance(storage_origin, TypeAliasType):
        value = storage_origin.__value__
        storage_origin = get_origin(value) or value

    def validate(data: object) -> Quantity[Any, Any]:
        quantity = parse_quantity(cls, data)
        if storage is Any:
            return quantity
        if storage_origin is float:
            valid = isinstance(quantity.value, float)
        elif storage_origin is np.ndarray:
            valid = isinstance(quantity.value, np.ndarray)
        elif isinstance(storage_origin, type):
            valid = isinstance(quantity.value, cast("type[object]", storage_origin))
        else:
            raise ValueError(f"Unsupported quantity storage annotation {storage!r}")
        if not valid:
            raise ValueError(
                f"Expected {cls._kind} storage {storage!r}; "
                f"received {type(quantity.value).__name__}"
            )
        return quantity

    names = [
        name
        for registered, spec in UNITS.items()
        if spec.kind == cls._kind
        for name in (registered, *spec.aliases)
    ]
    number = core_schema.float_schema()
    array = core_schema.list_schema(core_schema.any_schema())
    value_schema = (
        number
        if storage_origin is float
        else array
        if storage_origin is np.ndarray
        else core_schema.union_schema([number, array])
    )
    payload = core_schema.typed_dict_schema(
        {
            "value": core_schema.typed_dict_field(value_schema),
            "units": core_schema.typed_dict_field(core_schema.literal_schema(names)),
        },
        extra_behavior="forbid",
    )
    return core_schema.no_info_plain_validator_function(
        validate,
        json_schema_input_schema=core_schema.union_schema(
            [payload, core_schema.str_schema(pattern=r"^\s*\S+\s+\S.*$")]
        ),
        serialization=core_schema.plain_serializer_function_ser_schema(
            serialize_quantity, return_schema=payload
        ),
        metadata={"quantity_kind": cls._kind},
    )

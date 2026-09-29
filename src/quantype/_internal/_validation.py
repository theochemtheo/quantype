"""Pydantic v2 integration; wire codecs and storage conversion have separate owners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np
from pydantic_core import core_schema

from quantype._internal._storage import convert, storage_origin
from quantype.serialization import parse_quantity, to_dict

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

    from quantype.core import Quantity, Unit

# Runtime schema/storage inspection is the deliberately dynamic Pydantic boundary.
# ruff: noqa: ANN401, SLF001
# pyright: reportPrivateUsage=false

serialize_quantity = to_dict


def pydantic_schema(
    cls: type[Quantity[Any, Any]], source_type: Any, handler: GetCoreSchemaHandler
) -> core_schema.CoreSchema:
    del handler
    arguments = get_args(source_type)
    storage = arguments[-1] if arguments else Any
    origin = storage_origin(storage)

    def validate(data: object, info: core_schema.ValidationInfo) -> Quantity[Any, Any]:
        units: tuple[Unit[Any], ...] = ()
        context: Any = info.context
        if isinstance(context, dict):
            units = tuple(cast("dict[str, Any]", context).get("units", ()))
        quantity = parse_quantity(cls, data, units=units)
        if storage is Any:
            return quantity
        try:
            raw = convert(quantity.value, storage)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Expected {cls._kind} storage {storage!r}: {exc}"
            ) from exc
        result = cls.from_canonical(raw)
        result._display = quantity._display
        return result

    number = core_schema.float_schema()
    array = core_schema.list_schema(core_schema.any_schema())
    value_schema = (
        number
        if origin is float or origin is np.float64
        else array
        if origin is np.ndarray
        else core_schema.union_schema([number, array])
    )
    payload = core_schema.typed_dict_schema(
        {
            "kind": core_schema.typed_dict_field(
                core_schema.literal_schema([cls._kind])
            ),
            "magnitude": core_schema.typed_dict_field(value_schema),
            # A context may supply custom units, so a fixed builtin enum would
            # falsely reject valid schema inputs.
            "unit": core_schema.typed_dict_field(core_schema.str_schema()),
        },
        extra_behavior="forbid",
    )

    def validate_input(
        data: object,
        handler: core_schema.ValidatorFunctionWrapHandler,
        info: core_schema.ValidationInfo,
    ) -> Quantity[Any, Any]:
        # The inner schema describes the wire shape. Parsing the original value
        # ourselves avoids Pydantic coercing boolean magnitudes into numbers.
        del handler
        return validate(data, info)

    wrapper = getattr(core_schema, "with_info_wrap_validator_function", None)
    if wrapper is None:
        # Pydantic 2.0 used this name for the same context-aware boundary.
        wrapper = getattr(core_schema, "general_wrap_validator_function")  # noqa: B009
    return cast(
        "core_schema.CoreSchema",
        wrapper(
            validate_input,
            schema=core_schema.union_schema(
                [payload, core_schema.str_schema(pattern=r"^\s*\S+\s+\S.*$")]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                to_dict, return_schema=payload
            ),
            metadata={"quantity_kind": cls._kind},
        ),
    )

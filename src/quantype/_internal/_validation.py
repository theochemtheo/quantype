"""Pydantic v2 integration; wire codecs and storage conversion have separate owners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np
from pydantic_core import core_schema

from quantype._internal._storage import convert, storage_origin
from quantype._internal._systems import Atomistic, require_system
from quantype.core import _wrap
from quantype.serialization import (
    DIMENSIONLESS_STRING,
    QUANTITY_STRING,
    parse_quantity,
    to_dict,
)

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

    from quantype.core import Quantity, Unit

# Runtime schema/storage inspection is the deliberately dynamic Pydantic boundary.
# ruff: noqa: ANN401, SLF001
# pyright: reportPrivateUsage=false

serialize_quantity = to_dict


def _unchanged(value: object) -> object:
    return value


def pydantic_schema(
    cls: type[Quantity[Any, Any, Any]],
    source_type: Any,
    handler: GetCoreSchemaHandler,
) -> core_schema.CoreSchema:
    del handler
    arguments = get_args(source_type)
    storage = arguments[0] if arguments else Any
    system = require_system(arguments[1]) if len(arguments) > 1 else Atomistic
    origin = storage_origin(storage)

    def validate(
        data: object, info: core_schema.ValidationInfo
    ) -> Quantity[Any, Any, Any]:
        units: tuple[Unit[Any], ...] = ()
        context: Any = info.context
        if isinstance(context, dict):
            units = tuple(cast("dict[str, Any]", context).get("units", ()))
        quantity = parse_quantity(cls, data, units=units, system=system)
        if storage is Any:
            return quantity
        try:
            raw = convert(quantity.value, storage)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Expected {cls._kind} storage {storage!r}: {exc}"
            ) from exc
        # The echo is a Python float, so only float storage can keep it.
        echo = quantity._echo if origin is float else None
        return _wrap(cls._semantic, raw, system, display=quantity._display, echo=echo)

    number = core_schema.float_schema()
    array = core_schema.list_schema(core_schema.any_schema())
    scalar_storage = origin is float or (
        isinstance(origin, type) and issubclass(origin, np.floating)
    )
    # Arrays include zero-dimensional storage, which serializes as a number.
    value_schema = (
        number if scalar_storage else core_schema.union_schema([number, array])
    )

    def payload_schema(*, kind_required: bool) -> core_schema.TypedDictSchema:
        return core_schema.typed_dict_schema(
            {
                "kind": core_schema.typed_dict_field(
                    core_schema.literal_schema([cls._kind]), required=kind_required
                ),
                "magnitude": core_schema.typed_dict_field(value_schema),
                # A context may supply custom units, so a fixed builtin enum would
                # falsely reject valid schema inputs.
                "unit": core_schema.typed_dict_field(core_schema.str_schema()),
            },
            extra_behavior="forbid",
        )

    # Input may leave out "kind", which the field's type names; output has it.
    payload = payload_schema(kind_required=False)
    written = payload_schema(kind_required=True)

    def validate_input(
        data: object,
        handler: core_schema.ValidatorFunctionWrapHandler,
        info: core_schema.ValidationInfo,
    ) -> Quantity[Any, Any, Any]:
        # The inner schema describes the wire shape. Parsing the original value
        # ourselves avoids Pydantic coercing boolean magnitudes into numbers.
        del handler
        return validate(data, info)

    # Dimensionless values print as a bare number, so the schema accepts one.
    pattern = DIMENSIONLESS_STRING if cls._kind == "Dimensionless" else QUANTITY_STRING
    wrapper = getattr(core_schema, "with_info_wrap_validator_function", None)
    if wrapper is None:
        # Pydantic 2.0 used this name for the same context-aware boundary.
        wrapper = getattr(core_schema, "general_wrap_validator_function")  # noqa: B009
    return cast(
        "core_schema.CoreSchema",
        wrapper(
            validate_input,
            # Python-mode dumps pass quantities through this inner schema,
            # unchanged. Pydantic 2.0 has no "any" serialization schema.
            schema=core_schema.union_schema(
                [payload, core_schema.str_schema(pattern=pattern)],
                serialization=core_schema.plain_serializer_function_ser_schema(
                    _unchanged
                ),
            ),
            # Python-mode dumps keep quantities, as they keep datetimes; JSON
            # mode writes the kind/magnitude/unit object.
            serialization=core_schema.plain_serializer_function_ser_schema(
                to_dict, return_schema=written, when_used="json"
            ),
            metadata={"quantity_kind": cls._kind},
        ),
    )

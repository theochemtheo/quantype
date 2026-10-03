"""Value/unit JSON and pickle-free NumPy archives at an explicit host boundary.

NPY contains only an array; use NPZ when physical metadata must travel with it.
Recorded backends are provenance. The requested target type controls restoration,
including its unit system: the wire format records units, never a system.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeGuard, cast, get_args, get_origin

import numpy as np

from quantype import codata

from quantype._internal._lookup import resolve_unit as resolve_unit  # noqa: PLC0414
from quantype._internal._lookup import unit_of_kind
from quantype._internal._semantics import Kind
from quantype._internal._storage import (
    backend_name,
    convert,
    host_array,
    reject_booleans,
    unit_conversion,
)
from quantype._internal._systems import (
    Atomistic,
    UnitSystem,
    into_system,
    require_system,
)
from quantype.core import Quantity, Unit, _wrap

if TYPE_CHECKING:
    from pathlib import Path

# Codecs are dynamic boundaries between array libraries and wire data.
# Pydantic consumes these validation failures as ValueErrors.
# ruff: noqa: SLF001, TRY004
# pyright: reportPrivateUsage=false


def selected_unit(
    quantity: Quantity[Any, Any, Any], unit: Unit[Any] | None = None
) -> Unit[Any]:
    if not isinstance(quantity._semantic, Kind):
        raise ValueError("Serialization requires a named quantity")
    selected = (
        unit or quantity._display or quantity._system.unit_for(quantity._semantic)
    )
    quantity._check_unit(selected)
    return selected


def to_dict(
    quantity: Quantity[Any, Any, Any], unit: Unit[Any] | None = None
) -> dict[str, object]:
    selected = selected_unit(quantity, unit)
    return {
        "kind": quantity.kind,
        "magnitude": _json_numbers(host_array(quantity.magnitude(selected)).tolist()),
        "unit": selected.name,
    }


def _json_numbers(value: object) -> object:
    """JSON uses binary64 numbers; extended-range values must not become infinity."""
    if isinstance(value, list):
        return [_json_numbers(item) for item in cast("list[object]", value)]
    if isinstance(value, np.floating):
        scalar = cast("Any", value)
        number = float(scalar)
        if np.isfinite(scalar) and (
            not np.isfinite(number) or (scalar != 0 and number == 0)
        ):
            raise ValueError("Magnitude exceeds the range of JSON binary64 numbers")
        return number
    return value


_NUMBER = (
    r"[-+]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    r"|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?|[nN][aA][nN])"
)
# "<number> <unit>", the space optional: "0.5 nm", "0.5nm", "1e-3 eV", "1eV".
# A unit starts with neither a digit nor a point, so "15" is a number without a
# unit rather than 1 of a unit "5". Also the JSON-schema pattern, so it keeps to
# syntax ECMA-262 regexes share.
QUANTITY_STRING = rf"^\s*({_NUMBER})\s*([^\s\d.+-].*?)\s*$"
# Dimensionless values print as a bare number, so they parse from one too.
DIMENSIONLESS_STRING = rf"^\s*({_NUMBER})\s*([^\s\d.+-].*?)?\s*$"
_QUANTITY_STRING = re.compile(QUANTITY_STRING)
_NUMBER_STRING = re.compile(rf"^\s*({_NUMBER})\s*$")


def _is_quantity(value: object) -> TypeGuard[Quantity[Any, Any, Any]]:
    return isinstance(value, Quantity)


def _wire_fields(data: object, expected: str) -> tuple[object, str]:
    if isinstance(data, str):
        number = _NUMBER_STRING.match(data)
        if number is not None:
            if expected != "Dimensionless":
                raise ValueError(
                    f"{data!r} has no unit; write it as '<number> <unit>', "
                    f"such as '{number[1]} nm'"
                )
            return float(number[1]), "one"
        match = _QUANTITY_STRING.match(data)
        if match is None:
            raise ValueError(f"Expected '<number> <unit>'; received {data!r}")
        value: object = float(match[1])
        name: object = match[2]
    elif isinstance(data, Mapping):
        # The target names the kind, so "kind" may be left out, as it is from a
        # string; written data always includes it, and it must then match.
        payload = cast("Mapping[object, object]", data)
        if set(payload) - {"kind"} != {"magnitude", "unit"}:
            raise ValueError(
                "Quantity objects have 'magnitude' and 'unit', and optionally 'kind'"
            )
        if payload.get("kind", expected) != expected:
            raise ValueError(f"Expected {expected}; received {payload['kind']}")
        value, name = payload["magnitude"], payload["unit"]
    else:
        raise ValueError("Expected a quantity, string, or kind/magnitude/unit object")
    if not isinstance(name, str):
        raise ValueError("Unit identifier must be a string")
    return value, name


def parse_quantity(
    cls: type[Quantity[Any, Any, Any]],
    data: object,
    *,
    units: Iterable[Unit[Any]] = (),
    system: type[UnitSystem] = Atomistic,
) -> Quantity[Any, Any, Any]:
    """Decode into ``system``'s storage, remembering the unit the data used."""
    if _is_quantity(data):
        if data._semantic is not cls._semantic:
            raise ValueError(f"Expected {cls._kind}; received {data.kind}")
        if data._system is not system:
            raise ValueError(
                f"Expected a quantity in {system.__name__}; received one in "
                f"{data._system.__name__}. Convert it explicitly with .to_system(...)"
            )
        return data
    value, name = _wire_fields(data, cls._kind)
    unit = unit_of_kind(name, cast("Kind", cls._semantic), units, system)
    reject_booleans(value)
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(
            "Quantity magnitudes must be real numbers, not booleans or strings"
        )
    raw = float(array) if array.ndim == 0 else array.astype(np.float64)
    scale, offset = into_system(unit, system)
    stored = unit_conversion(raw, scale, offset)
    echo = raw if isinstance(raw, float) else None
    return _wrap(cls._semantic, stored, system, display=unit, echo=echo)


def save_npz(
    path: str | Path,
    *,
    record_backend: bool = False,
    **quantities: Quantity[Any, Any, Any],
) -> None:
    """Save named quantities plus versioned metadata; never write pickled objects."""
    arrays: dict[str, Any] = {}
    entries: dict[str, object] = {}
    for index, (name, quantity) in enumerate(quantities.items()):
        unit = selected_unit(quantity)
        key = f"array_{index}"
        arrays[key] = host_array(quantity.magnitude(unit))
        entry = {"kind": quantity.kind, "unit": unit.name, "array": key}
        if record_backend:
            entry["source_backend"] = backend_name(quantity.value)
        entries[name] = entry
    # The CODATA edition is provenance: it fixes what units such as bohr meant.
    metadata = {"version": 1, "codata": codata.edition(), "quantities": entries}
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(path, **arrays)


def _archive_entry(document: object, name: str) -> Mapping[str, str]:
    if not isinstance(document, dict):
        raise ValueError("Expected a quantity archive metadata object")
    metadata = cast("dict[str, object]", document)
    if (
        set(metadata) - {"codata"} != {"version", "quantities"}
        or type(metadata["version"]) is not int
        or metadata["version"] != 1
        or not isinstance(metadata.get("codata", ""), str)
    ):
        raise ValueError("Unsupported quantity archive version or metadata fields")
    entries = metadata["quantities"]
    if not isinstance(entries, dict):
        raise ValueError("Expected named quantities in archive metadata")
    entry = cast("dict[str, object]", entries).get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"Missing or invalid archive quantity {name!r}")
    fields = cast("dict[str, object]", entry)
    required = {"kind", "unit", "array"}
    if (
        not required <= fields.keys()
        or fields.keys() - required - {"source_backend"}
        or not all(isinstance(value, str) for value in fields.values())
    ):
        raise ValueError("Invalid quantity archive entry")
    return cast("Mapping[str, str]", fields)


def load_npz[Q](
    path: str | Path,
    name: str,
    target: type[Q],
    *,
    units: Iterable[Unit[Any]] = (),
    dtype: object = None,
) -> Q:
    """Restore to the explicit quantity/storage/system type, whatever the source.

    ``Length[NDArray[np.float64], SI]`` restores meters. Archives are system
    independent: one written from any system decodes into any other.
    """
    cls = get_origin(target) or target
    arguments = get_args(target)
    if not arguments or not isinstance(cls, type) or not issubclass(cls, Quantity):
        raise TypeError("Restoration requires a parameterized quantity type")
    storage = arguments[0]
    system = require_system(arguments[1]) if len(arguments) > 1 else Atomistic
    kind = cast("Kind", cls._semantic)
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive:
            raise ValueError("Archive is missing physical metadata")
        entry = _archive_entry(json.loads(str(archive["metadata"])), name)
        if entry["kind"] != cls._kind:
            raise ValueError(f"Expected {cls._kind}; received {entry['kind']}")
        unit = unit_of_kind(entry["unit"], kind, units, system)
        if entry["array"] not in archive:
            raise ValueError("Archive is missing the quantity array")
        array = archive[entry["array"]]
        if array.dtype.kind not in "iuf":
            raise ValueError("Archive magnitude must have a real numerical dtype")
        scale, offset = into_system(unit, system)
        raw = convert(array, storage, dtype=dtype, scale=scale, offset=offset)
        return cast("Q", _wrap(cls._semantic, raw, system, display=unit))

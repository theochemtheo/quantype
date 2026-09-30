"""Value/unit JSON and pickle-free NumPy archives at an explicit host boundary.

NPY contains only an array; use NPZ when physical metadata must travel with it.
Recorded backends are provenance. The requested target type controls restoration.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeGuard, cast, get_args, get_origin

import numpy as np

from quantype._internal._semantics import Kind
from quantype._internal._storage import (
    backend_name,
    convert,
    host_array,
    reject_booleans,
)
from quantype.core import Quantity, Unit, _wrap, canonical_unit, get_unit

if TYPE_CHECKING:
    from pathlib import Path

# Codecs are dynamic boundaries between array libraries and wire data.
# Pydantic consumes these validation failures as ValueErrors.
# ruff: noqa: SLF001, TRY004
# pyright: reportPrivateUsage=false


def resolve_unit(
    name: str, units: Iterable[Unit[Any]] = (), *, kind: Kind | None = None
) -> Unit[Any]:
    """Resolve explicit definitions without mutating the builtin catalogue."""
    definitions: dict[str, Unit[Any]] = {}
    for unit in units:
        if unit.name in definitions:
            raise ValueError(f"Duplicate unit definition {unit.name!r}")
        try:
            builtin = get_unit(unit.name, kind=kind)
        except ValueError:
            builtin = None
        if builtin is not None and builtin is not unit:
            raise ValueError(f"Custom unit shadows builtin {unit.name!r}")
        definitions[unit.name] = unit
    if name in definitions:
        return definitions[name]
    return get_unit(name, kind=kind)


def selected_unit(
    quantity: Quantity[Any, Any], unit: Unit[Any] | None = None
) -> Unit[Any]:
    if not isinstance(quantity._semantic, Kind):
        raise ValueError("Serialization requires a named quantity")
    selected = unit or quantity._display or canonical_unit(quantity._semantic)
    quantity._check_unit(selected)
    return selected


def to_dict(
    quantity: Quantity[Any, Any], unit: Unit[Any] | None = None
) -> dict[str, object]:
    selected = selected_unit(quantity, unit)
    return {
        "kind": quantity.kind,
        "magnitude": host_array(quantity.magnitude(selected)).tolist(),
        "unit": selected.name,
    }


def _is_quantity(value: object) -> TypeGuard[Quantity[Any, Any]]:
    return isinstance(value, Quantity)


def _wire_fields(data: object, expected: str) -> tuple[object, str]:
    if isinstance(data, str):
        pieces = data.strip().split(maxsplit=1)
        if len(pieces) != 2:  # noqa: PLR2004 -- magnitude and unit
            raise ValueError("Expected '<number> <unit>'")
        value: object = float(pieces[0])
        name: object = pieces[1]
    elif isinstance(data, Mapping):
        payload = cast("Mapping[object, object]", data)
        if set(payload) != {"kind", "magnitude", "unit"}:
            raise ValueError(
                "Quantity objects require exactly 'kind', 'magnitude', and 'unit'"
            )
        if payload["kind"] != expected:
            raise ValueError(f"Expected {expected}; received {payload['kind']}")
        value, name = payload["magnitude"], payload["unit"]
    else:
        raise ValueError("Expected a quantity, string, or kind/magnitude/unit object")
    if not isinstance(name, str):
        raise ValueError("Unit identifier must be a string")
    return value, name


def parse_quantity(
    cls: type[Quantity[Any, Any]], data: object, *, units: Iterable[Unit[Any]] = ()
) -> Quantity[Any, Any]:
    if _is_quantity(data):
        if data._semantic is not cls._semantic:
            raise ValueError(f"Expected {cls._kind}; received {data.kind}")
        return data
    value, name = _wire_fields(data, cls._kind)
    unit = resolve_unit(name, units, kind=cast("Kind", cls._semantic))
    if unit.semantic is not cls._semantic:
        raise ValueError(f"Expected {cls._kind}; received {unit.kind} (unit {name!r})")
    reject_booleans(value)
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(
            "Quantity magnitudes must be real numbers, not booleans or strings"
        )
    raw = float(array) if array.ndim == 0 else array.astype(np.float64)
    return _wrap(cls._semantic, unit.canonical(raw))


def save_npz(
    path: str | Path, *, record_backend: bool = False, **quantities: Quantity[Any, Any]
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
    arrays["metadata"] = np.asarray(json.dumps({"version": 1, "quantities": entries}))
    np.savez(path, **arrays)


def _archive_entry(document: object, name: str) -> Mapping[str, str]:
    if not isinstance(document, dict):
        raise ValueError("Expected a quantity archive metadata object")
    metadata = cast("dict[str, object]", document)
    if (
        set(metadata) != {"version", "quantities"}
        or type(metadata["version"]) is not int
        or metadata["version"] != 1
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
    """Restore to the explicit quantity/storage type, regardless of source backend."""
    cls = get_origin(target) or target
    storage = get_args(target)
    if not storage or not isinstance(cls, type) or not issubclass(cls, Quantity):
        raise TypeError("Restoration requires a parameterized quantity type")
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive:
            raise ValueError("Archive is missing physical metadata")
        entry = _archive_entry(json.loads(str(archive["metadata"])), name)
        if entry["kind"] != cls._kind:
            raise ValueError(f"Expected {cls._kind}; received {entry['kind']}")
        unit = resolve_unit(entry["unit"], units, kind=cast("Kind", cls._semantic))
        if unit.semantic is not cls._semantic:
            raise ValueError(f"Expected {cls._kind}; received {unit.kind}")
        if entry["array"] not in archive:
            raise ValueError("Archive is missing the quantity array")
        array = archive[entry["array"]]
        if array.dtype.kind not in "iuf":
            raise ValueError("Archive magnitude must have a real numerical dtype")
        raw = convert(array, storage[-1], dtype=dtype)
        canonical = convert(unit.canonical(raw), storage[-1], dtype=dtype)
        return cast("Q", _wrap(cls._semantic, canonical))

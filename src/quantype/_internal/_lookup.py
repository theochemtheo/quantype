"""Unit lookup by name at decoding boundaries, without mutating the catalogue."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantype._internal._systems import Atomistic, UnitSystem
from quantype._internal._unit import Unit, get_unit, known_kinds, units_of

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quantype._internal._semantics import Kind


def resolve_unit(
    name: str,
    units: Iterable[Unit[Any]] = (),
    *,
    kind: Kind | None = None,
    system: type[UnitSystem] = Atomistic,
) -> Unit[Any]:
    """Resolve explicit definitions without mutating the builtin catalogue.

    The target system's own units (custom bases and derived identifiers such as
    ``gromacs:Force``) are found automatically after explicit definitions.
    """
    definitions = _definitions(units, kind)
    named = [unit for (key, _), unit in definitions.items() if key == name]
    # Different kinds may share an identifier: prefer the requested kind.
    for unit in named:
        if unit.semantic is kind:
            return unit
    if kind is not None and system is not Atomistic:
        for unit in (system.unit_for(kind), *system.units):
            if unit.name == name and unit.semantic is kind:
                return unit
    if named:
        return named[0]
    try:
        return get_unit(name, kind=kind)
    except ValueError:
        if kind is None:
            raise
        return _by_symbol(name, kind, [*definitions.values(), *_system_units(system)])


def unit_of_kind(
    name: str,
    kind: Kind,
    units: Iterable[Unit[Any]] = (),
    system: type[UnitSystem] = Atomistic,
) -> Unit[Any]:
    """The unit of ``kind`` that ``name`` names: an identifier, else a symbol.

    A name of another kind loses to a symbol of this one, and is otherwise an
    error that names both kinds.
    """
    definitions = tuple(units)
    unit = resolve_unit(name, definitions, kind=kind, system=system)
    if unit.semantic is kind:
        return unit
    try:
        return _by_symbol(name, kind, [*definitions, *_system_units(system)])
    except ValueError:
        raise ValueError(
            f"Expected {kind}; received {unit.kind} (unit {name!r})"
        ) from None


def _system_units(system: type[UnitSystem]) -> list[Unit[Any]]:
    return [*system.units, *(system.unit_for(kind) for kind in known_kinds())]


def _by_symbol(name: str, kind: Kind, extra: Iterable[Unit[Any]]) -> Unit[Any]:
    """A unit of ``kind`` whose display symbol is ``name``, so printed values parse."""
    candidates = list(
        dict.fromkeys(
            unit for unit in (*extra, *units_of(kind)) if unit.semantic is kind
        )
    )
    found = [unit for unit in candidates if unit.symbol == name]
    if len(found) > 1:
        names = ", ".join(unit.name for unit in found)
        raise ValueError(f"Unit symbol {name!r} is ambiguous; use one of {names}")
    if not found:
        known = ", ".join(unit.name for unit in candidates)
        raise ValueError(
            f"Unknown unit {name!r} for {kind}"
            + (f"; use one of {known}" if known else "")
        )
    return found[0]


def _definitions(
    units: Iterable[Unit[Any]], kind: Kind | None
) -> dict[tuple[str, Kind], Unit[Any]]:
    definitions: dict[tuple[str, Kind], Unit[Any]] = {}
    for unit in units:
        if (unit.name, unit.semantic) in definitions:
            raise ValueError(f"Duplicate unit definition {unit.name!r}")
        try:
            builtin = get_unit(unit.name, kind=kind)
        except ValueError:
            builtin = None
        if builtin is not None and builtin is not unit:
            raise ValueError(f"Custom unit shadows builtin {unit.name!r}")
        definitions[unit.name, unit.semantic] = unit
    return definitions

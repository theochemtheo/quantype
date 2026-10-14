"""Storage units: the unit a quantity without a unit system keeps its numbers in.

Construction keeps the input unit. Products, quotients and powers compose their
operands' units without converting, and take the name of a catalogue unit of
the same scale when one exists, so kcal/mol/Å times Å is stored in kcal/mol.
They show in a named unit chosen from the operands' units (``shown``).
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any, Literal, cast, override

from quantype._internal._semantics import Kind, Semantic
from quantype._internal._systems import (
    DISPLAY_SYSTEMS,
    AnySystem,
    Atomistic,
    NoSystem,
    UnitSystem,
    coherent_symbol,
    factor,
    into_system,
)
from quantype._internal._unit import Unit, units_of

# A composed scale takes a catalogue unit's name when they agree this closely,
# which absorbs the rounding of each operand's scale.
_TOLERANCE = 1e-12
# Symbols holding these are parenthesized as an operand: (Å/fs)^2.
_COMPOUND = frozenset("/·^ ")


class ComposedUnit(Unit[Any]):
    """A product, quotient or power of units, which the catalogue doesn't name.

    ``systems`` are the display systems that store this unit's kind in it, so
    products of a quantity shown in it can show in one of them. ``power_of``
    is the base and exponent of a power, so a root undoes it.
    """

    systems: tuple[type[UnitSystem], ...]
    power_of: tuple[Unit[Any], float] | None

    def __init__(
        self,
        kind: Semantic,
        scale: float,
        symbol: str,
        *,
        systems: tuple[type[UnitSystem], ...] = (),
        power_of: tuple[Unit[Any], float] | None = None,
    ) -> None:
        super().__init__(symbol, kind, scale, symbol=symbol)
        object.__setattr__(self, "systems", systems)
        object.__setattr__(self, "power_of", power_of)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ComposedUnit)
            and other.symbol == self.symbol
            and other.scale == self.scale
            and other.semantic == self.semantic
        )

    @override
    def __hash__(self) -> int:
        return hash((self.symbol, self.scale, self.semantic))


def _grouped(symbol: str) -> str:
    return f"({symbol})" if _COMPOUND & set(symbol) else symbol


@cache
def _catalogue_systems(unit: Unit[Any]) -> tuple[type[UnitSystem], ...]:
    kind = unit.semantic
    if not isinstance(kind, Kind):
        return ()
    return tuple(
        system
        for system in DISPLAY_SYSTEMS
        if math.isclose(system.unit_for(kind).scale, unit.scale, rel_tol=_TOLERANCE)
    )


def _systems(unit: Unit[Any]) -> tuple[type[UnitSystem], ...]:
    if isinstance(unit, ComposedUnit):
        return unit.systems
    return _catalogue_systems(unit)


def _named(kind: Semantic, scale: float) -> Unit[Any] | None:
    """The catalogue unit of ``kind`` with this scale, preferring the canonical one."""
    if not isinstance(kind, Kind):
        return None
    matches = [
        unit
        for unit in units_of(kind)
        if not unit.offset and math.isclose(unit.scale, scale, rel_tol=_TOLERANCE)
    ]
    for unit in matches:
        if unit.name == kind.canonical_unit:
            return unit
    return matches[0] if matches else None


def _unit(
    kind: Semantic,
    scale: float,
    symbol: str,
    power_of: tuple[Unit[Any], float] | None = None,
) -> Unit[Any]:
    named = _named(kind, scale)
    if named is not None:
        return named
    return ComposedUnit(kind, scale, symbol, power_of=power_of)


def _is_one(unit: Unit[Any]) -> bool:
    return unit.symbol == "1" and unit.scale == 1


@cache
def compose(
    operation: Literal["mul", "div"], kind: Semantic, left: Unit[Any], right: Unit[Any]
) -> Unit[Any]:
    """The unit of a product or quotient of quantities stored in these units."""
    if _is_one(right):
        return recast(kind, left)
    if operation == "mul" and _is_one(left):
        return recast(kind, right)
    if operation == "mul" and left == right:
        return raised(kind, left, 2)
    a, b = _grouped(left.symbol), _grouped(right.symbol)
    if operation == "mul":
        return _unit(kind, left.scale * right.scale, f"{a}·{b}")
    return _unit(kind, left.scale / right.scale, f"{a}/{b}")


@cache
def raised(kind: Semantic, unit: Unit[Any], exponent: float) -> Unit[Any]:
    """The unit of a power of a quantity stored in ``unit``."""
    if isinstance(unit, ComposedUnit) and unit.power_of is not None:
        base, inner = unit.power_of
        total = inner * exponent
        return recast(kind, base) if total == 1 else raised(kind, base, total)
    shown = int(exponent) if float(exponent).is_integer() else exponent
    symbol = f"{_grouped(unit.symbol)}^{shown}"
    return _unit(kind, unit.scale**exponent, symbol, (unit, exponent))


@cache
def recast(kind: Semantic, unit: Unit[Any]) -> Unit[Any]:
    """``unit``'s scale as a unit of another kind: ΔK for K, Pa for J/m³."""
    if unit.semantic == kind:
        return unit
    return _named(kind, unit.scale) or ComposedUnit(kind, unit.scale, unit.symbol)


@cache
def reference_unit(kind: Semantic) -> Unit[Any]:
    """The unit of scale 1: Å, eV, fs, and their products."""
    if isinstance(kind, Kind):
        return _named(kind, 1.0) or Atomistic.unit_for(kind)
    if isinstance(kind.right, int):
        return raised(kind, reference_unit(kind.left), kind.right)
    operation = cast("Literal['mul', 'div']", kind.operation)
    return compose(
        operation, kind, reference_unit(kind.left), reference_unit(kind.right)
    )


@cache
def system_unit(system: type[UnitSystem], kind: Semantic) -> Unit[Any]:
    """The unit ``system`` stores ``kind`` in, as a storage unit."""
    if isinstance(kind, Kind):
        return system.unit_for(kind)
    return ComposedUnit(
        kind,
        factor(system, kind),
        _coherent_symbol(system, kind),
        systems=(system,) if system in DISPLAY_SYSTEMS else (),
    )


@cache
def neutral(kind: Semantic) -> Unit[Any]:
    """A constant's unit, which every display system can show it in."""
    unit = reference_unit(kind)
    return ComposedUnit(kind, unit.scale, unit.symbol, systems=DISPLAY_SYSTEMS)


def _coherent_symbol(system: type[UnitSystem], kind: Semantic) -> str:
    if isinstance(kind, Kind):
        return coherent_symbol(system, kind)
    left = _grouped(_coherent_symbol(system, kind.left))
    if isinstance(kind.right, int):
        return f"{left}^{kind.right}"
    right = _grouped(_coherent_symbol(system, kind.right))
    return f"{left}·{right}" if kind.operation == "mul" else f"{left}/{right}"


def storage_map(
    unit: Unit[Any], system: type[AnySystem]
) -> tuple[float, float, Unit[Any] | None]:
    """How magnitudes in ``unit`` are stored in ``system``: scale, offset, unit.

    Without a system the numbers stay in ``unit``, except that offset units
    (°C, °F) store kelvin. In a unit system there is no storage unit.
    """
    if system is not NoSystem:
        return (*into_system(unit, cast("type[UnitSystem]", system)), None)
    if unit.offset:
        return unit.scale, unit.offset, reference_unit(unit.semantic)
    return 1.0, 0.0, unit


@cache
def shown(kind: Semantic, *operands: Unit[Any]) -> Unit[Any]:
    """The unit a product, quotient or power of quantities shown in ``operands``
    shows in.

    It is the kind's unit in the first display system that stores each
    operand's kind in that operand's unit, else in ``Atomistic``, whose units
    are the reference units, so eV/nm shows as eV/Å and J/m as N. Operand order
    doesn't matter.
    """
    first, *rest = (_systems(unit) for unit in operands)
    systems = [system for system in first if all(system in other for other in rest)]
    chosen = systems[0] if systems else Atomistic
    # mypy reads a class's __hash__ as unbound, so a class isn't Hashable to it.
    return system_unit(chosen, kind)  # type: ignore[arg-type]


def presented(unit: Unit[Any], kind: Semantic) -> Unit[Any]:
    """The unit to show a quantity stored in ``unit`` in, when nothing chose one.

    A composed unit of a named kind shows in the kind's ``Atomistic`` unit.
    """
    if isinstance(unit, ComposedUnit) and isinstance(kind, Kind):
        return Atomistic.unit_for(kind)
    return unit

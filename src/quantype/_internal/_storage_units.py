"""Units without a unit system: the unit a quantity keeps its numbers in.

Construction keeps the input unit. Products, quotients and powers compose their
operands' units as a product of catalogue units with exponents, in which an
identical unit cancels exactly, so (eV/nm)·nm is eV. A composed unit takes the
name of the kind's unit in a display system when its scale is that unit's, so
J/m is N and kcal/mol/Å·Å is kcal/mol. A product is shown the same way, from
its operands' displayed units, and its symbol parses back (``parsed``).
"""

from __future__ import annotations

import math
import re
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, cast, override

from quantype._internal._registry import BASIS
from quantype._internal._semantics import Expression, Kind, Semantic
from quantype._internal._systems import (
    DISPLAY_SYSTEMS,
    AnySystem,
    Atomistic,
    NoSystem,
    UnitSystem,
    _resolve,  # pyright: ignore[reportPrivateUsage]
    coherence,
    coherent_symbol,
    into_system,
)
from quantype._internal._unit import Unit, units_of

if TYPE_CHECKING:
    from collections.abc import Callable

# A composed scale takes a catalogue unit's name when they agree this closely,
# which absorbs the rounding of each operand's scale.
_TOLERANCE = 1e-12
# Leaf symbols holding these are parenthesized in a composed symbol.
_COMPOUND = frozenset("/·* ")

type Factors = tuple[tuple[Unit[Any], float], ...]


class ComposedUnit(Unit[Any]):
    """A product of catalogue units with exponents: eV/nm, Da·nm^2/ps^2.

    ``factors`` are the units, which are never composed themselves, with
    nonzero exponents, sorted by sign and symbol.
    """

    factors: Factors

    def __init__(self, kind: Semantic, factors: Factors) -> None:
        scale = math.prod(unit.scale**exponent for unit, exponent in factors)
        symbol = _symbol(factors)
        super().__init__(symbol, kind, scale, symbol=symbol)
        object.__setattr__(self, "factors", factors)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ComposedUnit)
            and other.semantic == self.semantic
            and other.factors == self.factors
        )

    @override
    def __hash__(self) -> int:
        return hash((self.semantic, self.factors))


def _exponent(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _term(unit: Unit[Any], exponent: float) -> str:
    symbol = unit.symbol
    if exponent == 1:
        return f"({symbol})" if _COMPOUND & set(symbol) else symbol
    if (_COMPOUND | {"^"}) & set(symbol):
        symbol = f"({symbol})"
    return f"{symbol}^{_exponent(exponent)}"


def _symbol(factors: Factors) -> str:
    if len(factors) == 1 and factors[0][1] == 1:
        return factors[0][0].symbol
    above = [_term(unit, exponent) for unit, exponent in factors if exponent > 0]
    below = [_term(unit, -exponent) for unit, exponent in factors if exponent < 0]
    text = "·".join(above) or "1"
    if len(below) == 1:
        return f"{text}/{below[0]}"
    return f"{text}/({'·'.join(below)})" if below else text


def factors_of(unit: Unit[Any]) -> Factors:
    """``unit`` as catalogue units with exponents; an offset unit as kelvin."""
    if isinstance(unit, ComposedUnit):
        return unit.factors
    if unit.offset:
        return ((reference_unit(unit.semantic), 1.0),)
    if unit.symbol == "1" and unit.scale == 1:
        return ()
    return ((unit, 1.0),)


def _combined(*parts: tuple[Factors, float]) -> Factors:
    exponents: dict[Unit[Any], float] = {}
    for factors, power in parts:
        for unit, exponent in factors:
            exponents[unit] = exponents.get(unit, 0.0) + exponent * power
    # Sorted, so a product's unit doesn't depend on its operands' order.
    return tuple(
        sorted(
            ((unit, exponent) for unit, exponent in exponents.items() if exponent),
            key=lambda pair: (pair[1] < 0, pair[0].symbol, pair[0].name),
        )
    )


def _built(kind: Semantic, factors: Factors) -> Unit[Any]:
    """A unit of ``kind`` from ``factors``: a catalogue unit, when it is one."""
    if not factors:
        return reference_unit(kind)
    if len(factors) == 1 and factors[0][1] == 1:
        return recast(kind, factors[0][0])
    return ComposedUnit(kind, factors)


def settled(kind: Semantic, unit: Unit[Any]) -> Unit[Any]:
    """``unit``, or the kind's unit in a display system when the scales agree.

    Only the name changes: J/m becomes N, and nm·fs/ps stays as it is, though
    it has a picometer's scale.
    """
    if not isinstance(unit, ComposedUnit) or not isinstance(kind, Kind):
        return unit
    for system in DISPLAY_SYSTEMS:
        candidate = system.unit_for(kind)
        if not candidate.offset and math.isclose(
            candidate.scale, unit.scale, rel_tol=_TOLERANCE
        ):
            return candidate
    return unit


@cache
def compose(
    operation: Literal["mul", "div"], kind: Semantic, left: Unit[Any], right: Unit[Any]
) -> Unit[Any]:
    """The unit of a product or quotient of quantities in these units."""
    power = 1.0 if operation == "mul" else -1.0
    factors = _combined((factors_of(left), 1.0), (factors_of(right), power))
    return settled(kind, _built(kind, factors))


@cache
def raised(kind: Semantic, unit: Unit[Any], exponent: float) -> Unit[Any]:
    """The unit of a power of a quantity in ``unit``."""
    return settled(kind, _built(kind, _combined((factors_of(unit), exponent))))


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


@cache
def recast(kind: Semantic, unit: Unit[Any]) -> Unit[Any]:
    """``unit`` as a unit of another kind of its dimensions: ΔK for K."""
    if unit.semantic == kind:
        return unit
    named = _named(kind, unit.scale)
    if named is not None:
        return named
    factors = factors_of(unit)
    return ComposedUnit(kind, factors) if factors else reference_unit(kind)


@cache
def reference_unit(kind: Semantic) -> Unit[Any]:
    """The unit of scale 1: Å, eV, fs, and their products."""
    if isinstance(kind, Kind):
        return _named(kind, 1.0) or Atomistic.unit_for(kind)
    return _structural(kind, reference_unit)


def _structural(kind: Expression, leaf: Callable[[Kind], Unit[Any]]) -> Unit[Any]:
    """A unit of a structural kind, from a unit for each named kind in it."""

    def factors(semantic: Semantic) -> Factors:
        if isinstance(semantic, Kind):
            return factors_of(leaf(semantic))
        if isinstance(semantic.right, int):
            return _combined((factors(semantic.left), semantic.right))
        power = 1.0 if semantic.operation == "mul" else -1.0
        right = factors(semantic.right)
        return _combined((factors(semantic.left), 1.0), (right, power))

    return _built(kind, factors(kind))


@cache
def system_unit(system: type[UnitSystem], kind: Semantic) -> Unit[Any]:
    """The unit ``system`` stores ``kind`` in; structural kinds are coherent."""
    if isinstance(kind, Kind):
        return system.unit_for(kind)
    return _structural(kind, lambda leaf: _coherent(system, leaf))


def _coherent(system: type[UnitSystem], kind: Kind) -> Unit[Any]:
    unit = system.unit_for(kind)
    factor = coherence(system, kind)
    if factor == 1:
        return unit
    return Unit(
        f"{system.name}:coherent:{kind}",
        kind,
        unit.scale / factor,
        symbol=coherent_symbol(system, kind),
    )


def _axis(unit: Unit[Any]) -> str | None:
    """The base axis a unit of a base kind measures, such as length for nm."""
    kind = unit.semantic
    powers = [power for power in kind.dimensions if power]
    if not isinstance(kind, Kind) or powers != [1]:
        return None
    return BASIS[kind.dimensions.index(1)]


@cache
def adapted(kind: Semantic, *context: Unit[Any]) -> Unit[Any]:
    """A constant's unit, in the base units ``context`` uses where it has them.

    So k_B times a temperature in K is shown in eV, and a kcal/mol energy over
    k_B in K: the constant brings no units of its own beyond the reference ones.
    """
    chosen: dict[str, Unit[Any]] = {}
    for unit in context:
        for leaf, _ in factors_of(unit):
            axis = _axis(leaf)
            if axis is not None:
                chosen.setdefault(axis, leaf)
    bases = _resolve(Atomistic)
    factors = tuple(
        (chosen.get(axis, bases[axis]), float(power))
        for axis, power in zip(BASIS, kind.dimensions, strict=True)
        if power
    )
    return _built(kind, factors)


def _systems(unit: Unit[Any]) -> tuple[type[UnitSystem], ...]:
    """The display systems that store each unit in ``unit``'s kind in it."""
    leaves = [leaf for leaf, _ in factors_of(unit)]
    return tuple(
        system
        for system in DISPLAY_SYSTEMS
        if all(
            isinstance(leaf.semantic, Kind)
            and math.isclose(
                system.unit_for(leaf.semantic).scale, leaf.scale, rel_tol=_TOLERANCE
            )
            for leaf in leaves
        )
    )


@cache
def _expanded(leaf: Unit[Any]) -> Factors:
    """``leaf`` as the base units of a display system that derives it from them."""
    kind = leaf.semantic
    if (
        not isinstance(kind, Kind)
        or _axis(leaf) is not None
        or not any(kind.dimensions)
    ):
        return ((leaf, 1.0),)
    for system in DISPLAY_SYSTEMS:
        if coherence(system, kind) == 1 and math.isclose(
            system.unit_for(kind).scale, leaf.scale, rel_tol=_TOLERANCE
        ):
            bases = _resolve(system)
            return tuple(
                (bases[axis], float(power))
                for axis, power in zip(BASIS, kind.dimensions, strict=True)
                if power
            )
    return ((leaf, 1.0),)


@cache
def simplest(kind: Semantic, unit: Unit[Any]) -> Unit[Any]:
    """``unit`` in its simplest form, for ``simplified()``.

    A unit a display system derives from its base units, such as kcal/mol/Å in
    ``Real``, expands into them. Units of one dimension then merge into the
    first of them by symbol, so (kcal/mol/Å)·nm is kcal/mol. A named kind then
    takes its unit in the first display system that holds every remaining
    unit, so Da·Å^2/fs^2 is eV, else a catalogue unit of the same scale in a
    display system. What remains is the merged product.
    """
    expanded = _combined(
        *((_expanded(leaf), exponent) for leaf, exponent in factors_of(unit))
    )
    merged: dict[tuple[int, ...], tuple[Unit[Any], float]] = {}
    for leaf, exponent in expanded:
        key = tuple(leaf.semantic.dimensions)
        target, total = merged.get(key, (leaf, 0.0))
        merged[key] = (target, total + exponent)
    factors = tuple((leaf, total) for leaf, total in merged.values() if total)
    simplified = _built(kind, factors)
    if isinstance(kind, Kind) and factors:
        systems = _systems(simplified)
        if systems:
            return systems[0].unit_for(kind)
    return settled(kind, simplified)


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


_TOKENS = re.compile(r"\s*([()·*/^]|[^\s()·*/^]+)")


def parsed(text: str, kind: Kind, leaf: Callable[[str], Unit[Any]]) -> Unit[Any] | None:
    """The unit a composed symbol such as ``eV/nm`` or ``Da·nm^2/ps^2`` names.

    ``leaf`` resolves a catalogue name or symbol, raising ``ValueError`` for
    neither. Parenthesized text is tried as one unit first, so ``(kcal/mol)·nm``
    works. Operators apply left to right, as in ``a/b·c = (a/b)·c``. None if
    ``text`` isn't composed or isn't of ``kind``'s dimensions.
    """
    tokens = _TOKENS.findall(text)
    if not set(tokens) & set("()·*/^"):
        return None
    try:
        factors = _Parser(tokens, leaf).factors()
    except (ValueError, IndexError):
        return None
    dimensions = [0.0] * len(kind.dimensions)
    for unit, exponent in factors:
        for index, power in enumerate(unit.semantic.dimensions):
            dimensions[index] += power * exponent
    if dimensions != [float(power) for power in kind.dimensions]:
        return None
    return _built(kind, factors)


class _Parser:
    def __init__(self, tokens: list[str], leaf: Callable[[str], Unit[Any]]) -> None:
        self.tokens = tokens
        self.position = 0
        self.leaf = leaf

    def factors(self) -> Factors:
        factors = self._term()
        while self.position < len(self.tokens):
            operator = self._take()
            if operator not in {"·", "*", "/"}:
                raise ValueError(operator)
            power = -1.0 if operator == "/" else 1.0
            factors = _combined((factors, 1.0), (self._term(), power))
        return factors

    def _take(self) -> str:
        self.position += 1
        return self.tokens[self.position - 1]

    def _term(self) -> Factors:
        factors = self._atom()
        if self.position < len(self.tokens) and self.tokens[self.position] == "^":
            self._take()
            factors = _combined((factors, float(self._take())))
        return factors

    def _atom(self) -> Factors:
        text = self._take()
        if text != "(":
            return factors_of(self.leaf(text))
        start, depth = self.position, 1
        while depth:
            depth += {"(": 1, ")": -1}.get(self._take(), 0)
        inner = self.tokens[start : self.position - 1]
        try:
            return factors_of(self.leaf("".join(inner)))
        except ValueError:
            return _Parser(inner, self.leaf).factors()

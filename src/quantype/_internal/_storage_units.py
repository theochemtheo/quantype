"""Units without a unit system: the unit a quantity keeps its numbers in.

Construction keeps the input unit. Products, quotients and powers compose their
operands' units as a product of catalogue units with exponents, in which an
identical unit cancels exactly, so (eV/nm)·nm is eV, and a right operand's unit
that would cancel a left one of its dimension converts into it, so eV·nm/Å is
eV. A composed unit takes the name of the kind's unit in a display system when
its scale is that unit's, so J/m is N and kcal/mol/Å·Å is kcal/mol. A product
is shown the same way, from its operands' displayed units, and its symbol
parses back (``parsed``).
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

# A composed scale takes a catalogue unit's name when they agree this closely:
# a few ulps, the rounding of each operand's scale, so a name never stands for
# a unit whose scale differs.
_TOLERANCE = 4e-15
# Leaf symbols holding these are parenthesized in a composed symbol.
_COMPOUND = frozenset("/·* ")

type Factors = tuple[tuple[Unit[Any], float], ...]


class ComposedUnit(Unit[Any]):
    """A product of catalogue units with exponents: eV/nm, Da·nm^2/ps^2.

    ``factors`` are the units, which are never composed themselves, with
    nonzero exponents, sorted by sign and symbol. The name composes the units'
    names, since a custom unit may reuse a catalogue unit's symbol.
    """

    factors: Factors

    def __init__(self, kind: Semantic, factors: Factors) -> None:
        scale = math.prod(unit.scale**exponent for unit, exponent in factors)
        name = _composed(factors, lambda unit: unit.name)
        symbol = _composed(factors, lambda unit: unit.symbol)
        super().__init__(name, kind, scale, symbol=symbol)
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


def _term(text: str, exponent: float) -> str:
    if exponent == 1:
        return f"({text})" if _COMPOUND & set(text) else text
    if (_COMPOUND | {"^"}) & set(text):
        text = f"({text})"
    return f"{text}^{_exponent(exponent)}"


def _composed(factors: Factors, text: Callable[[Unit[Any]], str]) -> str:
    if len(factors) == 1 and factors[0][1] == 1:
        return text(factors[0][0])
    if not factors:
        return "1"
    above = [_term(text(unit), power) for unit, power in factors if power > 0]
    if not above:
        # ns^-1 rather than 1/ns, which would read as a number before a unit.
        return "·".join(_term(text(unit), power) for unit, power in factors)
    below = [_term(text(unit), -power) for unit, power in factors if power < 0]
    numerator = "·".join(above)
    if len(below) == 1:
        return f"{numerator}/{below[0]}"
    return f"{numerator}/({'·'.join(below)})" if below else numerator


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
    if not factors and isinstance(kind, Kind):
        return reference_unit(kind)
    if not factors:
        return ComposedUnit(kind, ())
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
) -> tuple[Unit[Any], float, float]:
    """The unit of a product or quotient of quantities in these units.

    A unit of the right operand that would cancel one of the left's dimension
    converts into it, so eV·nm/Å is eV and nm·fs/ps is nm; the product of the
    numbers then takes the factor ``source / target``. The result shows no
    unit its operands don't.
    """
    power = 1.0 if operation == "mul" else -1.0
    mine = factors_of(left)
    source = target = 1.0
    theirs: list[tuple[Unit[Any], float]] = []
    for unit, exponent in _combined((factors_of(right), power)):
        into = _cancelled(mine, unit, exponent)
        if into is not unit:
            above, below = (unit, into) if exponent > 0 else (into, unit)
            source *= above.scale ** abs(exponent)
            target *= below.scale ** abs(exponent)
        theirs.append((into, exponent))
    factors = _combined((mine, 1.0), (tuple(theirs), 1.0))
    return settled(kind, _built(kind, factors)), source, target


def _cancelled(mine: Factors, unit: Unit[Any], exponent: float) -> Unit[Any]:
    """The unit in ``mine`` that ``unit`` to ``exponent`` would cancel, else it."""
    dimensions = unit.semantic.dimensions
    if not any(dimensions):
        return unit
    for leaf, own in mine:
        if own * exponent < 0 and leaf.semantic.dimensions == dimensions:
            return leaf
    return unit


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


def _merged(factors: Factors) -> Factors:
    """Units of one dimension merged into the one with the largest power."""
    groups: dict[tuple[int, ...], list[tuple[Unit[Any], float]]] = {}
    for leaf, exponent in factors:
        groups.setdefault(tuple(leaf.semantic.dimensions), []).append((leaf, exponent))
    return tuple(
        (max(group, key=lambda pair: abs(pair[1]))[0], total)
        for group in groups.values()
        if (total := sum(exponent for _, exponent in group))
    )


@cache
def rooted(kind: Semantic, unit: Unit[Any], exponent: float) -> Unit[Any]:
    """The displayed unit of a root of a quantity shown in ``unit``.

    Units of one dimension merge first, so the root of nm·Å shows in nm
    rather than nm^0.5·Å^0.5.
    """
    return settled(kind, _built(kind, _combined((_merged(factors_of(unit)), exponent))))


@cache
def simplest(kind: Semantic, unit: Unit[Any]) -> Unit[Any]:
    """``unit`` in its simplest form, for ``simplified()``.

    A catalogue unit is already simplest. A unit a display system derives from
    its base units, such as kcal/mol/Å in ``Real``, expands into them. Units of
    one dimension then merge into the one with the largest power, so
    (kcal/mol/Å)·nm is kcal/mol and Pa·Å is J/m^2. A named kind then takes its
    unit in the first display system that holds every remaining unit, so
    Da·Å^2/fs^2 is eV, else a catalogue unit of the same scale in a display
    system. What remains is the merged product.
    """
    if not isinstance(unit, ComposedUnit):
        return unit
    factors = _merged(
        _combined(*((_expanded(leaf), exponent) for leaf, exponent in factors_of(unit)))
    )
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
    (°C, °F) store kelvin and a dimensionless composed unit, such as nm/Å,
    plain numbers. In a unit system there is no storage unit.
    """
    if system is not NoSystem:
        return (*into_system(unit, cast("type[UnitSystem]", system)), None)
    plain = isinstance(unit, ComposedUnit) and not any(unit.semantic.dimensions)
    if unit.offset or plain:
        return unit.scale, unit.offset, reference_unit(unit.semantic)
    return 1.0, 0.0, unit


_TOKENS = re.compile(r"[()·*/^]|[^\s()·*/^]+")


def parsed(text: str, kind: Kind, leaf: Callable[[str], Unit[Any]]) -> Unit[Any] | None:
    """The unit a composed symbol such as ``eV/nm`` or ``Da·nm^2/ps^2`` names.

    ``leaf`` resolves a catalogue name or symbol, raising ``ValueError`` for
    neither. Parenthesized text is tried as one unit first, so ``(kcal/mol)·nm``
    works. Operators apply left to right, as in ``a/b·c = (a/b)·c``. None if
    ``text`` isn't composed; ``ValueError`` if it can't be read, names a unit
    with an offset, or isn't of ``kind``'s dimensions.
    """
    tokens = list(_TOKENS.finditer(text))
    if not {token.group() for token in tokens} & set("()·*/^"):
        return None
    try:
        factors = _Parser(text, tokens, leaf).factors()
    except IndexError:
        raise ValueError(f"Can't read the unit {text!r}") from None
    dimensions = [0.0] * len(kind.dimensions)
    for unit, exponent in factors:
        for index, power in enumerate(unit.semantic.dimensions):
            dimensions[index] += power * exponent
    if dimensions != [float(power) for power in kind.dimensions]:
        raise ValueError(f"{text!r} is not a unit of {kind}")
    return _built(kind, factors)


class _Parser:
    def __init__(
        self, text: str, tokens: list[re.Match[str]], leaf: Callable[[str], Unit[Any]]
    ) -> None:
        self.text = text
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

    def _next(self) -> str:
        return self.tokens[self.position].group()

    def _take(self) -> str:
        self.position += 1
        return self.tokens[self.position - 1].group()

    def _term(self) -> Factors:
        factors = self._atom()
        if self.position < len(self.tokens) and self._next() == "^":
            self._take()
            exponent = self._take()
            try:
                power = float(exponent)
            except ValueError:
                power = math.nan
            if not math.isfinite(power):
                raise ValueError(f"Can't read the exponent {exponent!r}")
            factors = _combined((factors, power))
        return factors

    def _atom(self) -> Factors:
        text = self._take()
        if text != "(":
            return self._leaf(text)
        start, depth = self.position, 1
        while depth:
            depth += {"(": 1, ")": -1}.get(self._take(), 0)
        inner = self.tokens[start : self.position - 1]
        # A leaf symbol such as ``eV ps/Å`` keeps its spaces.
        whole = self.text[inner[0].start() : inner[-1].end()]
        try:
            unit = self.leaf(whole)
        except ValueError:
            return _Parser(self.text, inner, self.leaf).factors()
        return self._leaf(whole, unit)

    def _leaf(self, text: str, unit: Unit[Any] | None = None) -> Factors:
        unit = self.leaf(text) if unit is None else unit
        if unit.offset:
            raise ValueError(f"{unit.symbol} has an offset; a product can't hold it")
        return factors_of(unit)

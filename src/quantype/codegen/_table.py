"""The naming table: which unnamed products a catalogue's relations name.

Named kinds are atoms. An unnamed product of two named kinds is a canonical
*pair*: ``Length * Time`` and ``Time * Length`` are one pair, ``x * x`` is
``x ** 2``, and ``1 / x`` is ``x ** -1``. A named kind times or over a pair is
named when the groupings of its factors that declared relations name agree on
one kind, so ``Mass * Velocity ** 2`` is an ``Energy`` however it is written.
Relations are never treated as equations, so kinds with equal dimensions stay
distinct, and a product that two groupings name differently stays unnamed.

The table holds every pair, its public class name, and the entries. Runtime
arithmetic and the generated stubs both read it, so they agree on every result.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quantype.catalogue import Catalogue

type Operation = Literal["mul", "div", "pow"]
type Product = Literal["mul", "div"]
# A product of named kinds as sorted (kind, exponent) factors, without zeros
# or Dimensionless.
type Monomial = tuple[tuple[str, int], ...]

# The integer powers a pair can be: squares, cubes and their reciprocals.
POWERS = (2, 3, -1, -2)


@dataclass(frozen=True)
class Pair:
    """A canonical unnamed product of two named kinds, or a power of one."""

    operation: Operation
    left: str
    right: str | int
    name: str

    @property
    def marker(self) -> str:
        """The structural kind marker the class is generic over in stubs."""
        if self.operation == "pow":
            return f"Pow[{self.left}Kind, Literal[{self.right}]]"
        expression = "Mul" if self.operation == "mul" else "Div"
        return f"{expression}[{self.left}Kind, {self.right}Kind]"

    @property
    def monomial(self) -> Monomial:
        if self.operation == "pow":
            return _monomial({self.left: int(self.right)})
        sign = 1 if self.operation == "mul" else -1
        return _monomial({self.left: 1, str(self.right): sign})


@dataclass(frozen=True)
class Table:
    pairs: Mapping[str, Pair]
    # Every spelling of a pair from two named kinds: ("mul", "Time", "Length"),
    # ("mul", "Length", "Length"), ("div", "Dimensionless", "Length"), and
    # ("pow", "Length", 2) all find their pair's name.
    spellings: Mapping[tuple[Operation, str, str | int], str]
    # (operation, left, right) -> named result, where one operand is a pair's
    # name and the other a named kind.
    entries: Mapping[tuple[Product, str, str], str]

    def pair(self, operation: Operation, left: str, right: str | int) -> Pair | None:
        name = self.spellings.get((operation, left, right))
        return None if name is None else self.pairs[name]


def _monomial(factors: Mapping[str, int]) -> Monomial:
    return tuple(
        sorted(
            (kind, exponent)
            for kind, exponent in factors.items()
            if exponent and kind != "Dimensionless"
        )
    )


def _combine(left: Monomial, right: Monomial, sign: int) -> Monomial:
    factors = dict(left)
    for kind, exponent in right:
        factors[kind] = factors.get(kind, 0) + sign * exponent
    return _monomial(factors)


def _splits(monomial: Monomial) -> list[tuple[Monomial, Monomial]]:
    """Every way to write ``monomial`` as a product of two non-empty parts."""
    kinds = [kind for kind, _ in monomial]
    ranges = [range(min(0, e), max(0, e) + 1) for _, e in monomial]
    splits: list[tuple[Monomial, Monomial]] = []
    for exponents in product(*ranges):
        first = _monomial(dict(zip(kinds, exponents, strict=True)))
        second = _monomial(
            {kind: e - x for (kind, e), x in zip(monomial, exponents, strict=True)}
        )
        if first and second:
            splits.append((first, second))
    return splits


class _Namer:
    """Which named kinds the groupings of a monomial's factors produce."""

    def __init__(self, catalogue: Catalogue) -> None:
        self.algebra = catalogue.algebra
        self.powers = catalogue.powers
        self.cache: dict[Monomial, frozenset[str]] = {}

    def power(self, kind: str, exponent: int) -> str | None:
        return named_power(self.algebra, self.powers, kind, exponent)

    def names(self, monomial: Monomial) -> frozenset[str]:
        cached = self.cache.get(monomial)
        if cached is None:
            cached = self.cache[monomial] = self._names(monomial)
        return cached

    def _names(self, monomial: Monomial) -> frozenset[str]:
        if not monomial:
            return frozenset({"Dimensionless"})
        if len(monomial) == 1:
            kind, exponent = monomial[0]
            if exponent == 1:
                return frozenset({kind})
            named = self.power(kind, exponent)
            if named is not None:
                return frozenset({named})
        found: set[str] = set()
        for first, second in _splits(monomial):
            inverse = _monomial({kind: -e for kind, e in second})
            found |= self._related("mul", first, second)
            found |= self._related("div", first, inverse)
        return frozenset(found)

    def _related(self, operation: Product, left: Monomial, right: Monomial) -> set[str]:
        """What a declared relation makes of the names of two parts."""
        return {
            result
            for first in self.names(left)
            for second in self.names(right)
            if (result := self.algebra.get((operation, first, second))) is not None
        }


def named_power(
    algebra: Mapping[tuple[str, str, str], str],
    powers: Mapping[tuple[str, int], str],
    kind: str,
    exponent: int,
) -> str | None:
    """A named power, including ``x * x`` and ``1 / x`` relations."""
    named = powers.get((kind, exponent))
    if named is None and exponent == 2:  # noqa: PLR2004 -- x ** 2 is x * x
        named = algebra.get(("mul", kind, kind))
    if named is None and exponent == -1:
        named = algebra.get(("div", "Dimensionless", kind))
    return named


def pair_name(pair: Pair) -> str:
    """The public class name: kind names in catalogue order (LengthTime)."""
    if pair.operation == "mul":
        return f"{pair.left}{pair.right}"
    if pair.operation == "div":
        return f"{pair.left}Per{pair.right}"
    return {
        2: f"{pair.left}Squared",
        3: f"{pair.left}Cubed",
        -1: f"Per{pair.left}",
        -2: f"Per{pair.left}Squared",
    }[int(pair.right)]


def _candidates(catalogue: Catalogue, namer: _Namer) -> list[Pair]:
    """Every unnamed product of two named kinds, in canonical form."""
    kinds = [kind for kind in catalogue.quantities if kind != "Dimensionless"]
    algebra = catalogue.algebra
    pairs: list[Pair] = []
    for index, left in enumerate(kinds):
        pairs += [
            Pair("pow", left, exponent, "")
            for exponent in POWERS
            if namer.power(left, exponent) is None
        ]
        pairs += [
            Pair("mul", left, right, "")
            for right in kinds[index + 1 :]
            if ("mul", left, right) not in algebra
        ]
        pairs += [
            Pair("div", left, right, "")
            for right in kinds
            if right != left and ("div", left, right) not in algebra
        ]
    return [
        Pair(pair.operation, pair.left, pair.right, pair_name(pair)) for pair in pairs
    ]


def naming_table(catalogue: Catalogue, reserved: frozenset[str] = frozenset()) -> Table:
    """Every product class of two kinds, and what each names with a third.

    ``reserved`` holds names a pair class must not take, besides the kinds.
    """
    namer = _Namer(catalogue)
    kinds = [kind for kind in catalogue.quantities if kind != "Dimensionless"]
    entries: dict[tuple[Product, str, str], str] = {}
    used: dict[str, Pair] = {}
    for pair in _candidates(catalogue, namer):
        found = _entries(pair, kinds, namer)
        if pair.name in used:
            raise ValueError(
                f"Products {used[pair.name].marker} and {pair.marker} would both "
                f"be named {pair.name}; rename a quantity"
            )
        used[pair.name] = pair
        entries |= found
    clashes = sorted(
        name for name in used if name in catalogue.quantities or name in reserved
    )
    if clashes:
        raise ValueError(
            f"Generated product classes {clashes} clash with quantity or API names; "
            "rename the quantity or declare the relation that names the product"
        )
    return Table(used, _spellings(used), entries)


def _entries(
    pair: Pair, kinds: list[str], namer: _Namer
) -> dict[tuple[Product, str, str], str]:
    """A named kind times or over ``pair``, from either side, when named."""
    found: dict[tuple[Product, str, str], str] = {}
    operations: tuple[Product, ...] = ("mul", "div")
    for kind, operation in product(kinds, operations):
        sign = 1 if operation == "mul" else -1
        single = _monomial({kind: 1})
        keys: tuple[tuple[tuple[Product, str, str], Monomial], ...] = (
            ((operation, kind, pair.name), _combine(single, pair.monomial, sign)),
            ((operation, pair.name, kind), _combine(pair.monomial, single, sign)),
        )
        for key, monomial in keys:
            names = namer.names(monomial)
            if len(names) == 1:
                found[key] = next(iter(names))
    return found


def _spellings(
    pairs: Mapping[str, Pair],
) -> dict[tuple[Operation, str, str | int], str]:
    spellings: dict[tuple[Operation, str, str | int], str] = {}
    for name, pair in pairs.items():
        spellings[pair.operation, pair.left, pair.right] = name
        if pair.operation == "mul":
            spellings["mul", str(pair.right), pair.left] = name
        elif pair.operation == "pow" and pair.right == 2:  # noqa: PLR2004
            spellings["mul", pair.left, pair.left] = name
        elif pair.operation == "pow" and pair.right == -1:
            spellings["div", "Dimensionless", pair.left] = name
    return spellings

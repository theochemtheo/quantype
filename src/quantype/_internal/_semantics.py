"""Physical identities and expression trees, independent of numerical storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from quantype._internal._registry import POWERS, QUANTITIES, RELATIONS, close_relations

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, eq=False)
class Kind:
    """Nominal identity: equal dimensions never make two kinds interchangeable."""

    name: str
    dimensions: tuple[int, ...]
    canonical_unit: str
    affine: bool = False

    @override
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Expression:
    operation: Literal["mul", "div", "pow"]
    left: Semantic
    right: Semantic | int

    @property
    def dimensions(self) -> tuple[int, ...]:
        if isinstance(self.right, int):
            return tuple(n * self.right for n in self.left.dimensions)
        sign = 1 if self.operation == "mul" else -1
        return tuple(
            a + sign * b
            for a, b in zip(self.left.dimensions, self.right.dimensions, strict=True)
        )

    @override
    def __str__(self) -> str:
        op = {"mul": "Mul", "div": "Div", "pow": "Pow"}[self.operation]
        return f"{op}[{self.left},{self.right}]"


type Semantic = Kind | Expression

KINDS = {
    name: Kind(name, spec.dimensions, spec.canonical_unit, name == "Temperature")
    for name, spec in QUANTITIES.items()
}
TEMPERATURE_DIFFERENCES = {KINDS["Temperature"]: KINDS["TemperatureDifference"]}
DIMENSIONLESS_KINDS = dict.fromkeys(KINDS.values(), KINDS["Dimensionless"])
# Each catalogue's kinds by name, keyed by its dimensionless kind, so a function
# can name the Angle or Dimensionless of its operand's own catalogue.
CATALOGUE_KINDS: dict[Kind, dict[str, Kind]] = {KINDS["Dimensionless"]: KINDS}


def named_kind(semantic: Semantic, name: str) -> Kind:
    """The kind called ``name`` in the catalogue ``semantic`` belongs to."""
    kinds = CATALOGUE_KINDS[dimensionless_kind(semantic)]
    try:
        return kinds[name]
    except KeyError as exc:
        raise TypeError(f"This quantity's catalogue has no {name} kind") from exc


def dimensionless_kind(semantic: Semantic) -> Kind:
    """Resolve a reciprocal's numerator without importing another catalogue.

    An expression spanning different catalogues has no unambiguous implicit
    scalar identity. Its caller can supply an explicit dimensionless numerator.
    """
    if isinstance(semantic, Kind):
        try:
            return DIMENSIONLESS_KINDS[semantic]
        except KeyError as exc:
            raise TypeError(
                "Scalar reciprocal requires a catalogue dimensionless kind"
            ) from exc
    left = dimensionless_kind(semantic.left)
    if not isinstance(semantic.right, int) and left is not dimensionless_kind(
        semantic.right
    ):
        raise TypeError(
            "Scalar reciprocal of a mixed-catalogue expression requires "
            "an explicit dimensionless quantity numerator"
        )
    return left


def addition(left: Semantic, right: Semantic, *, subtract: bool) -> Semantic:
    if isinstance(left, Kind) and left.affine:
        if left is right:
            if not subtract:
                raise TypeError(
                    "Cannot add two absolute Temperatures; to shift one, add a "
                    "TemperatureDifference, such as 10 * u.delta_K"
                )
            return TEMPERATURE_DIFFERENCES[left]
        if TEMPERATURE_DIFFERENCES[left] is right:
            return left
    elif isinstance(right, Kind) and right.affine:
        if not subtract and TEMPERATURE_DIFFERENCES[right] is left:
            return right
    elif left == right:
        return left
    operation = "subtract" if subtract else "add"
    raise TypeError(f"Cannot {operation} {left} and {right}")


PRODUCTS = {
    (op, KINDS[left], KINDS[right]): KINDS[result]
    for (op, left, right), result in close_relations(RELATIONS).items()
}
EXPONENTS = {
    (KINDS[name], power): KINDS[result] for (name, power), result in POWERS.items()
}


# The naming table (quantype.codegen._table), loaded from each catalogue's
# generated products module on first need. An unnamed product of two named kinds
# is a canonical pair: every spelling of it finds the same expression.
PAIRS: dict[tuple[str, Kind, Kind | int], Expression] = {}
# Each pair's runtime class, as (module, class name).
PAIR_CLASSES: dict[Expression, tuple[str, str]] = {}
# A named kind times or over a pair, from either side, when that is named.
ENTRIES: dict[tuple[str, Semantic, Semantic], Kind] = {}
_LOADERS: list[Callable[[], object]] = []


def defer_table(loader: Callable[[], object]) -> None:
    """Load a catalogue's naming table when a product first needs it."""
    _LOADERS.append(loader)


def _load_tables() -> None:
    while _LOADERS:
        _LOADERS.pop()()


def product(
    operation: Literal["mul", "div"], left: Semantic, right: Semantic
) -> Semantic:
    # Absolute temperatures are stored kelvin-scaled in every system, so their
    # products and ratios are well defined; only their sums are not.
    if isinstance(left, Kind) and isinstance(right, Kind):
        known = PRODUCTS.get((operation, left, right))
        if known is not None:
            return known
        _load_tables()
        pair = PAIRS.get((operation, left, right))
        if pair is not None:
            return pair
        return Expression(operation, left, right)
    _load_tables()
    named = ENTRIES.get((operation, left, right))
    if named is not None:
        return named
    if operation == "div" and left == right and left in PAIR_CLASSES:
        return dimensionless_kind(left)
    return Expression(operation, left, right)


def power(kind: Semantic, exponent: int) -> Semantic:
    if isinstance(kind, Kind):
        known = EXPONENTS.get((kind, exponent))
        # x ** 2 is x * x, and x ** -1 is 1 / x, wherever those are named.
        if known is None and exponent == 2:  # noqa: PLR2004
            known = PRODUCTS.get(("mul", kind, kind))
        dimensionless = DIMENSIONLESS_KINDS.get(kind)
        if known is None and exponent == -1 and dimensionless is not None:
            known = PRODUCTS.get(("div", dimensionless, kind))
        if known is not None:
            return known
        _load_tables()
        pair = PAIRS.get(("pow", kind, exponent))
        if pair is not None:
            return pair
    return Expression("pow", kind, exponent)

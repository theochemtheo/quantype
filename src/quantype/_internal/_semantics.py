"""Physical identities and expression trees, independent of numerical storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, override

from quantype._internal._registry import POWERS, QUANTITIES, RELATIONS, close_relations


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
                raise TypeError("Cannot add two absolute Temperatures")
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


def product(
    operation: Literal["mul", "div"], left: Semantic, right: Semantic
) -> Semantic:
    # Absolute temperatures are stored kelvin-scaled in every system, so their
    # products and ratios are well defined; only their sums are not.
    if isinstance(left, Kind) and isinstance(right, Kind):
        known = PRODUCTS.get((operation, left, right))
        if known is not None:
            return known
    return Expression(operation, left, right)


def power(kind: Semantic, exponent: int) -> Semantic:
    if isinstance(kind, Kind):
        known = EXPONENTS.get((kind, exponent))
        if known is not None:
            return known
    return Expression("pow", kind, exponent)

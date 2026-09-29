"""Declarative input shared by builtin and external catalogue generation."""

from __future__ import annotations

import keyword
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from quantype._registry import BASIS, QuantitySpec, UnitSpec

__all__ = ["Catalogue", "QuantitySpec", "UnitSpec", "builtin_catalogue"]

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class Catalogue:
    quantities: Mapping[str, QuantitySpec]
    units: Mapping[str, UnitSpec]
    relations: Mapping[tuple[str, str, str], str] = field(
        default_factory=dict[tuple[str, str, str], str]
    )
    powers: Mapping[tuple[str, int], str] = field(
        default_factory=dict[tuple[str, int], str]
    )

    def __post_init__(self) -> None:
        for attribute in ("quantities", "units", "relations", "powers"):
            object.__setattr__(
                self, attribute, MappingProxyType(dict(getattr(self, attribute)))
            )
        self.validate()

    def validate(self) -> None:
        self._validate_quantities()
        self._validate_units()
        self._validate_algebra()

    def _validate_quantities(self) -> None:
        reserved = {"Quantity", "Unit", "Mul", "Div", "Pow"}
        for name, spec in self.quantities.items():
            if (
                not name.isidentifier()
                or keyword.iskeyword(name)
                or name in reserved
                or len(spec.dimensions) != len(BASIS)
            ):
                raise ValueError(f"Invalid quantity definition {name!r}")
            if any(type(exponent) is not int for exponent in spec.dimensions):
                raise ValueError("Dimensions must have integer exponents")
            canonical = self.units.get(spec.canonical_unit)
            if (
                canonical is None
                or canonical.kind != name
                or canonical.scale != 1
                or canonical.offset != 0
            ):
                raise ValueError(f"Invalid canonical unit for {name}")

    def _validate_units(self) -> None:
        identifiers: set[str] = set()
        for name, unit in self.units.items():
            if (
                not name.isidentifier()
                or keyword.iskeyword(name)
                or unit.kind not in self.quantities
            ):
                raise ValueError(f"Invalid unit definition {name!r}")
            if (
                not math.isfinite(unit.scale)
                or unit.scale <= 0
                or not math.isfinite(unit.offset)
            ):
                raise ValueError(f"Invalid conversion for {name}")
            if unit.offset and unit.kind != "Temperature":
                raise ValueError("Only absolute-temperature units may have offsets")
            for identifier in (name, *unit.aliases):
                if not identifier.strip() or identifier in identifiers:
                    raise ValueError(f"Empty or duplicate unit name {identifier!r}")
                identifiers.add(identifier)

    def _validate_algebra(self) -> None:
        for (operation, left, right), result in self.relations.items():
            if not {left, right, result} <= self.quantities.keys():
                raise ValueError("Unknown quantity in relation")
            if operation not in {"mul", "div"} or "Temperature" in (left, right):
                raise ValueError("Invalid or affine multiplicative relationship")
            sign = 1 if operation == "mul" else -1
            expected = tuple(
                a + sign * b
                for a, b in zip(
                    self.quantities[left].dimensions,
                    self.quantities[right].dimensions,
                    strict=True,
                )
            )
            if expected != self.quantities[result].dimensions:
                raise ValueError(
                    f"Dimensionally inconsistent relation {left} {operation} {right}"
                )
            if (
                operation == "mul"
                and self.relations.get((operation, right, left), result) != result
            ):
                raise ValueError("Conflicting commutative relationship")
        for (name, exponent), result in self.powers.items():
            if (
                not {name, result} <= self.quantities.keys()
                or type(exponent) is not int
            ):
                raise ValueError("Unknown quantity or invalid exponent in power")
            if (
                name == "Temperature"
                or tuple(n * exponent for n in self.quantities[name].dimensions)
                != self.quantities[result].dimensions
            ):
                raise ValueError(f"Invalid power {name} ** {exponent}")

    def extend(
        self,
        *,
        quantities: Mapping[str, QuantitySpec],
        units: Mapping[str, UnitSpec],
        relations: Mapping[tuple[str, str, str], str] | None = None,
        powers: Mapping[tuple[str, int], str] | None = None,
    ) -> Catalogue:
        for old, new in (
            (self.quantities, quantities),
            (self.units, units),
            (self.relations, relations or {}),
            (self.powers, powers or {}),
        ):
            if old.keys() & new.keys():
                raise ValueError("An extension cannot replace existing definitions")
        return Catalogue(
            {**self.quantities, **quantities},
            {**self.units, **units},
            {**self.relations, **(relations or {})},
            {**self.powers, **(powers or {})},
        )


def builtin_catalogue() -> Catalogue:
    from quantype._registry import (  # noqa: PLC0415
        POWERS,
        QUANTITIES,
        RELATIONS,
        unit_specs,
    )

    return Catalogue(QUANTITIES, unit_specs(), RELATIONS, POWERS)

"""Bind an explicitly imported generated catalogue to the shared physical engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantype._internal._semantics import (
    CATALOGUE_KINDS,
    DIMENSIONLESS_KINDS,
    EXPONENTS,
    PRODUCTS,
    TEMPERATURE_DIFFERENCES,
    Kind,
)
from quantype._internal._unit import _CATALOGUE_UNITS, _UNIT_LOOKUPS, Unit

if TYPE_CHECKING:
    from quantype.catalogue import Catalogue

# pyright: reportPrivateUsage=false


class RuntimeCatalogue:
    def __init__(self, catalogue: Catalogue) -> None:
        self.kinds = {
            name: Kind(
                name, spec.dimensions, spec.canonical_unit, name == "Temperature"
            )
            for name, spec in catalogue.quantities.items()
        }
        if "Temperature" in self.kinds:
            TEMPERATURE_DIFFERENCES[self.kinds["Temperature"]] = self.kinds[
                "TemperatureDifference"
            ]
        if "Dimensionless" in self.kinds:
            DIMENSIONLESS_KINDS.update(
                dict.fromkeys(self.kinds.values(), self.kinds["Dimensionless"])
            )
            CATALOGUE_KINDS[self.kinds["Dimensionless"]] = self.kinds
        self.units: dict[str, Unit[Any]] = {}
        by_kind: dict[Kind, list[Unit[Any]]] = {
            kind: [] for kind in self.kinds.values()
        }
        for name, spec in catalogue.units.items():
            kind = self.kinds[spec.kind]
            unit: Unit[Any] = Unit(name, kind, spec.scale, spec.offset, spec.symbol)
            by_kind[kind].append(unit)
            for identifier in (name, *spec.aliases):
                self.units[identifier] = unit
        for kind, units in by_kind.items():
            _UNIT_LOOKUPS[kind] = self.get_unit
            _CATALOGUE_UNITS[kind] = tuple(units)
        for (op, left, right), result in catalogue.algebra.items():
            PRODUCTS[op, self.kinds[left], self.kinds[right]] = self.kinds[result]
        for (name, exponent), result in catalogue.powers.items():
            EXPONENTS[self.kinds[name], exponent] = self.kinds[result]

    def get_unit(self, name: str) -> Unit[Any]:
        try:
            return self.units[name]
        except KeyError as exc:
            raise ValueError(f"Unknown unit {name!r}") from exc

"""Bind an explicitly imported generated catalogue to the shared physical engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantype._semantics import EXPONENTS, PRODUCTS, TEMPERATURE_DIFFERENCES, Kind
from quantype._unit import _CANONICAL_UNITS, _UNIT_LOOKUPS, Unit

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
        self.units: dict[str, Unit[Any]] = {}
        for name, spec in catalogue.units.items():
            kind = self.kinds[spec.kind]
            unit: Unit[Any] = Unit(name, kind, spec.scale, spec.offset, spec.symbol)
            for identifier in (name, *spec.aliases):
                self.units[identifier] = unit
            if name == kind.canonical_unit:
                _CANONICAL_UNITS[kind] = unit
        for kind in self.kinds.values():
            _UNIT_LOOKUPS[kind] = self.get_unit
        for (op, left, right), result in catalogue.relations.items():
            PRODUCTS[op, self.kinds[left], self.kinds[right]] = self.kinds[result]
            if op == "mul":
                PRODUCTS[op, self.kinds[right], self.kinds[left]] = self.kinds[result]
        for (name, exponent), result in catalogue.powers.items():
            EXPONENTS[self.kinds[name], exponent] = self.kinds[result]

    def get_unit(self, name: str) -> Unit[Any]:
        try:
            return self.units[name]
        except KeyError as exc:
            raise ValueError(f"Unknown unit {name!r}") from exc

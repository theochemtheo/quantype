"""Declarative input shared by builtin and external catalogue generation."""

from __future__ import annotations

import keyword
import math
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

from quantype._internal._registry import (
    BASIS,
    QuantitySpec,
    UnitSpec,
    close_relations,
)

__all__ = ["Catalogue", "QuantitySpec", "UnitSpec", "builtin_catalogue"]

if TYPE_CHECKING:
    from collections.abc import Mapping


# These bindings exist in generated units modules before lazy units are resolved.
# A declaration must not promise a unit where runtime exposes an implementation
# object or a public math helper instead.
_UNIT_MODULE_BINDINGS = frozenset(
    {
        # Stub imports.
        "Any",
        "Unit",
        "overload",
        "np",
        # Private runtime helpers.
        "_typing",
        "_get_unit",
        "_runtime",
        "_NAMES",
        "globals",
        "__getattr__",
        "__all__",
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__builtins__",
    }
)


# Names imported or assigned by the runtime/stub renderers and package scaffold.
# Keep this shared validation contract aligned when adding generated bindings.
_QUANTITY_BINDINGS = _UNIT_MODULE_BINDINGS | frozenset(
    {
        "Quantity",
        "Mul",
        "Div",
        "Pow",
        "Literal",
        "override",
        "np",
        "npt",
        "_units",
        "_BaseQuantity",
        "_StructuralQuantity",
        "u",
        "Callable",
        "Array",
        "Tensor",
        "_generated",
        "_catalogue",
        "numpy",
        "units",
        "kinds",
        "ujax",
        "utorch",
        # Unit-system bindings imported by generated stubs.
        "TypeVar",
        "UnitSystem",
        "Atomistic",
        "systems",
        # These generic parameters occur alongside named-class references in
        # generated methods, so they cannot also name a quantity class.
        "V",
        "W",
        "K",
        "S",
        "T",
    }
)


def _namespace_name(kind: str) -> str:
    return "".join(
        ("_" + char.lower()) if char.isupper() else char for char in kind
    ).lstrip("_")


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
        close_relations(self.relations)

    @cached_property
    def algebra(self) -> Mapping[tuple[str, str, str], str]:
        """Declared relations, plus the symmetric products and undoing divisions."""
        return MappingProxyType(close_relations(self.relations))

    def _validate_quantities(self) -> None:
        reserved = _QUANTITY_BINDINGS | {
            binding
            for kind in self.quantities
            for binding in (f"{kind}Kind", f"_{kind}Namespace", f"_{kind}Unit")
        }
        namespaces: set[str] = set()
        for name, spec in self.quantities.items():
            if (
                not name.isidentifier()
                or keyword.iskeyword(name)
                or name in reserved
                or (name.startswith("__") and name.endswith("__"))
                or len(spec.dimensions) != len(BASIS)
            ):
                raise ValueError(f"Invalid quantity definition {name!r}")
            namespace = _namespace_name(name)
            if (
                not namespace
                or keyword.iskeyword(namespace)
                or namespace in _UNIT_MODULE_BINDINGS
                or namespace in namespaces
            ):
                raise ValueError(f"Conflicting quantity namespace {namespace!r}")
            namespaces.add(namespace)
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
        reserved = _UNIT_MODULE_BINDINGS | {
            binding
            for kind in self.quantities
            for binding in (kind, f"{kind}Kind", f"_{kind}Namespace", f"_{kind}Unit")
        }
        identifiers: set[str] = set()
        for name, unit in self.units.items():
            if (
                not name.isidentifier()
                or keyword.iskeyword(name)
                or unit.kind not in self.quantities
            ):
                raise ValueError(f"Invalid unit definition {name!r}")
            if (
                isinstance(unit.scale, bool)
                or isinstance(unit.offset, bool)
                or not math.isfinite(float(unit.scale))
                or float(unit.scale) <= 0
                or not math.isfinite(float(unit.offset))
                or (unit.offset != 0 and float(unit.offset) == 0)
            ):
                raise ValueError(f"Invalid conversion for {name}")
            if unit.offset and unit.kind != "Temperature":
                raise ValueError("Only absolute-temperature units may have offsets")
            for identifier in (name, *unit.aliases):
                if not identifier.strip() or identifier in identifiers:
                    raise ValueError(f"Empty or duplicate unit name {identifier!r}")
                if identifier in reserved or (
                    identifier.startswith("__") and identifier.endswith("__")
                ):
                    raise ValueError(
                        f"Unit name {identifier!r} conflicts with "
                        "a generated API binding"
                    )
                identifiers.add(identifier)

    def _validate_algebra(self) -> None:
        for (operation, left, right), result in self.relations.items():
            if not {left, right, result} <= self.quantities.keys():
                raise ValueError("Unknown quantity in relation")
            if operation not in {"mul", "div"}:
                raise ValueError(f"Invalid relation operation {operation!r}")
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
                tuple(n * exponent for n in self.quantities[name].dimensions)
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
    from quantype._internal._registry import (  # noqa: PLC0415
        POWERS,
        QUANTITIES,
        RELATIONS,
        unit_specs,
    )

    return Catalogue(QUANTITIES, unit_specs(), RELATIONS, POWERS)

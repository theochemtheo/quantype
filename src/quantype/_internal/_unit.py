"""Immutable conversion definitions and explicit catalogue lookup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, cast, override

from quantype._internal._semantics import KINDS, Kind, Semantic

if TYPE_CHECKING:
    from collections.abc import Callable

    from quantype.core import Quantity

# Numerical dispatch is intentionally dynamic; importing core is deferred to
# construction to keep unit definitions independent of quantity implementation.
# ruff: noqa: ANN401, PLC0415
# pyright: reportPrivateUsage=false


@dataclass(frozen=True, init=False, repr=False, match_args=False)
class Unit[K]:
    name: str
    kind: str
    semantic: Semantic
    scale: float
    offset: float
    symbol: str
    __array_priority__: ClassVar[int] = 10000

    def __init__(
        self,
        name: str,
        kind: str | Semantic,
        scale: float = 1.0,
        offset: float = 0.0,
        symbol: str | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("Unit identifier must not be empty")
        if (
            isinstance(scale, bool)
            or not math.isfinite(scale)
            or scale <= 0
            or not math.isfinite(offset)
        ):
            raise ValueError(
                "Unit scale must be finite and positive; offset must be finite"
            )
        semantic = KINDS[kind] if isinstance(kind, str) else kind
        if offset and not (isinstance(semantic, Kind) and semantic.affine):
            raise ValueError("Only absolute-temperature units may have an offset")
        for attribute, value in (
            ("name", name),
            ("kind", str(semantic)),
            ("semantic", semantic),
            ("scale", scale),
            ("offset", offset),
            ("symbol", symbol if symbol is not None else name),
        ):
            object.__setattr__(self, attribute, value)

    @override
    def __hash__(self) -> int:
        # Units key the caches every product consults; hash each once.
        cached = self.__dict__.get("_hash")
        if isinstance(cached, int):
            return cached
        fields = (self.name, self.semantic, self.scale, self.offset, self.symbol)
        computed = hash(fields)
        object.__setattr__(self, "_hash", computed)
        return computed

    def __call__[V](self, value: V) -> Quantity[K, V, Any]:
        """Unit-first construction without a system: stored and shown in this unit.

        Units with an offset (°C) store kelvin.
        """
        from quantype._internal._storage import unit_conversion
        from quantype._internal._storage_units import reference_unit
        from quantype._internal._systems import NoSystem
        from quantype.core import _wrap

        raw: Any = value
        echo: float | None = None
        if isinstance(raw, bool):
            raise TypeError("Boolean values are not physical magnitudes")
        # NumPy float64 is also a Python float: inspect numerical storage first.
        if hasattr(raw, "shape") and hasattr(raw, "dtype"):
            from quantype._internal._storage import validate_floating_storage

            validate_floating_storage(raw)
        elif isinstance(raw, (int, float)):
            raw = echo = float(raw)
        elif isinstance(raw, Unit):
            raise TypeError(_UNIT_ARITHMETIC)
        else:
            raise TypeError(
                "Use a real scalar or numerical array as a magnitude; "
                "for lists, use a typed quantity constructor or numpy.asarray "
                "with a floating dtype first"
            )
        storage: Any = self
        if self.offset:
            raw = unit_conversion(raw, self.scale, self.offset)
            storage = reference_unit(self.semantic)
        return cast(
            "Quantity[K, V, Any]",
            _wrap(
                self.semantic,
                raw,
                NoSystem,
                display=cast("Any", self),
                echo=echo,
                storage=storage,
            ),
        )

    def __mul__(self, value: Any) -> Quantity[K, Any, Any]:
        return self(value)

    def __rmul__(self, value: Any) -> Quantity[K, Any, Any]:
        return self(value)

    def __truediv__(self, other: object) -> NoReturn:
        raise TypeError(_UNIT_ARITHMETIC)

    def __rtruediv__(self, other: object) -> NoReturn:
        raise TypeError(_UNIT_ARITHMETIC)

    def __pow__(self, exponent: object) -> NoReturn:
        raise TypeError(_UNIT_ARITHMETIC)

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        if ufunc.__name__ == "multiply" and method == "__call__" and not kwargs:
            other = inputs[1] if inputs[0] is self else inputs[0]
            return self(other)
        return NotImplemented

    @override
    def __repr__(self) -> str:
        return self.symbol


_UNIT_ARITHMETIC = (
    "Units only scale magnitudes; combine quantities instead, as in "
    "(1.0 * u.eV) / (1.0 * u.angstrom), or name a composed unit, as in "
    "Force.unit_named('eV/Å')"
)
_UNIT_LOOKUPS: dict[Kind, Callable[[str], Unit[Any]]] = {}
# Units of kinds from explicitly imported generated catalogues, in catalogue order.
_CATALOGUE_UNITS: dict[Kind, tuple[Unit[Any], ...]] = {}


@cache
def _builtin_units() -> dict[str, Unit[Any]]:
    from quantype._internal._registry import unit_specs

    units: dict[str, Unit[Any]] = {}
    for name, spec in unit_specs().items():
        unit: Unit[Any] = Unit(name, spec.kind, spec.scale, spec.offset, spec.symbol)
        for identifier in (name, *spec.aliases):
            units[identifier] = unit
    return units


def get_unit(name: str, *, kind: Semantic | None = None) -> Unit[Any]:
    if isinstance(kind, Kind) and kind in _UNIT_LOOKUPS:
        return _UNIT_LOOKUPS[kind](name)
    try:
        return _builtin_units()[name]
    except KeyError as exc:
        raise ValueError(f"Unknown unit {name!r}") from exc


@cache
def _builtin_kind_units() -> dict[Kind, tuple[Unit[Any], ...]]:
    grouped: dict[Kind, list[Unit[Any]]] = {}
    for unit in dict.fromkeys(_builtin_units().values()):
        grouped.setdefault(cast("Kind", unit.semantic), []).append(unit)
    return {kind: tuple(units) for kind, units in grouped.items()}


def units_of(kind: Kind) -> tuple[Unit[Any], ...]:
    """Every named catalogue unit of a kind, in catalogue order."""
    if kind in _CATALOGUE_UNITS:
        return _CATALOGUE_UNITS[kind]
    return _builtin_kind_units().get(kind, ())


def known_kinds() -> tuple[Kind, ...]:
    """Built-in kinds, then kinds of generated catalogues imported so far."""
    builtin = tuple(KINDS.values())
    return builtin + tuple(kind for kind in _CATALOGUE_UNITS if kind not in builtin)

"""Immutable conversion definitions and explicit catalogue lookup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

from quantype._internal._semantics import KINDS, Kind

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
    semantic: Kind
    scale: float
    offset: float
    symbol: str
    __array_priority__: ClassVar[int] = 10000

    def __init__(
        self,
        name: str,
        kind: str | Kind,
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
        if offset and not semantic.affine:
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

    @cached_property
    def _default_storage(self) -> tuple[float, float]:
        """The affine map into the default system, for unit-first construction."""
        from quantype._internal._systems import Atomistic, into_system

        return into_system(cast("Any", self), Atomistic)

    def __call__[V](self, value: V) -> Quantity[K, V, Any]:
        """Unit-first construction: default-system storage, displayed in this unit."""
        from quantype._internal._storage import unit_conversion
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
        else:
            raise TypeError(
                "Use a real scalar or numerical array as a magnitude; "
                "for lists, use a typed quantity constructor or numpy.asarray "
                "with a floating dtype first"
            )
        stored = unit_conversion(raw, *self._default_storage)
        return cast(
            "Quantity[K, V, Any]",
            _wrap(self.semantic, stored, None, display=cast("Any", self), echo=echo),
        )

    def __mul__(self, value: Any) -> Quantity[K, Any, Any]:
        return self(value)

    def __rmul__(self, value: Any) -> Quantity[K, Any, Any]:
        return self(value)

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


def get_unit(name: str, *, kind: Kind | None = None) -> Unit[Any]:
    if kind is not None and kind in _UNIT_LOOKUPS:
        return _UNIT_LOOKUPS[kind](name)
    try:
        return _builtin_units()[name]
    except KeyError as exc:
        raise ValueError(f"Unknown unit {name!r}") from exc


@cache
def _builtin_kind_units() -> dict[Kind, tuple[Unit[Any], ...]]:
    grouped: dict[Kind, list[Unit[Any]]] = {}
    for unit in dict.fromkeys(_builtin_units().values()):
        grouped.setdefault(unit.semantic, []).append(unit)
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

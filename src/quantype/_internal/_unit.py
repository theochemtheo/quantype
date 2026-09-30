"""Immutable conversion definitions and explicit catalogue lookup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
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

    def __call__[V](self, value: V) -> Quantity[K, V]:
        from quantype.core import _wrap

        raw: Any = value
        if isinstance(raw, bool):
            raise TypeError("Boolean values are not physical magnitudes")
        if isinstance(raw, (int, float)):
            raw = float(raw)
        elif not (hasattr(raw, "shape") and hasattr(raw, "dtype")):
            raise TypeError(
                "Use a real scalar or numerical array as a magnitude; "
                "for lists, use a typed quantity constructor or numpy.asarray first"
            )
        return cast("Quantity[K, V]", _wrap(self.semantic, self.canonical(raw)))

    def canonical(self, raw: Any) -> Any:
        if self.scale != 1:
            raw = raw * self.scale
        if self.offset != 0:
            raw = raw + self.offset
        return raw

    def __mul__(self, value: Any) -> Quantity[K, Any]:
        return self(value)

    def __rmul__(self, value: Any) -> Quantity[K, Any]:
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


_CANONICAL_UNITS: dict[Kind, Unit[Any]] = {}
_UNIT_LOOKUPS: dict[Kind, Callable[[str], Unit[Any]]] = {}


def canonical_unit(kind: Kind) -> Unit[Any]:
    if kind in _CANONICAL_UNITS:
        return _CANONICAL_UNITS[kind]
    return get_unit(kind.canonical_unit)


@cache
def get_unit(name: str, *, kind: Kind | None = None) -> Unit[Any]:
    if kind is not None and kind in _UNIT_LOOKUPS:
        return _UNIT_LOOKUPS[kind](name)
    from quantype._internal._registry import unit_specs

    for identifier, spec in unit_specs().items():
        if name == identifier or name in spec.aliases:
            return Unit(identifier, spec.kind, spec.scale, spec.offset, spec.symbol)
    raise ValueError(f"Unknown unit {name!r}")

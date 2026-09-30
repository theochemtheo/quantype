# mypy: disable-error-code=overload-overlap
# pyright: reportOverlappingOverload=false
from typing import Any, Self, overload

import numpy as np
import numpy.typing as npt

from quantype._internal._kind_types import Div, Mul, NonAffineKind, Pow
from quantype._internal._semantics import Semantic
from quantype._internal._unit import Unit as Unit
from quantype._internal._unit import canonical_unit as canonical_unit
from quantype._internal._unit import get_unit as get_unit
from quantype.kinds import DimensionlessKind

__all__ = [
    "Quantity",
    "Unit",
    "canonical_unit",
    "dimensions",
    "get_unit",
    "result_kind",
]

type _StructuralQuantity[A, B, V] = (
    Quantity[Mul[A, B], V] | Quantity[Div[A, B], V] | Quantity[Pow[A, B], V]
)

class Quantity[K, V]:
    _kind: str
    _semantic: Semantic
    _display: Unit[K] | None
    def __init__(
        self, value: object, unit: Unit[K], *, dtype: object = ...
    ) -> None: ...
    @classmethod
    def define_unit(
        cls,
        name: str,
        *,
        reference: Unit[K],
        scale: float = ...,
        offset: float = ...,
        symbol: str | None = ...,
    ) -> Unit[K]: ...
    @property
    def value(self) -> V: ...
    @property
    def kind(self) -> str: ...
    @property
    def dimensions(self) -> tuple[int, ...]: ...
    @classmethod
    def from_canonical[W](cls, value: W) -> Quantity[K, W]: ...
    @classmethod
    def parse(
        cls, data: object, *, units: tuple[Unit[Any], ...] = ...
    ) -> Quantity[K, float | npt.NDArray[np.float64]]: ...
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any: ...
    def _check_unit(self, unit: Unit[K]) -> None: ...
    def magnitude(self, unit: Unit[K] | None = ...) -> V: ...
    def to(self, unit: Unit[K]) -> Self: ...
    def to_dict(self, unit: Unit[K] | None = ...) -> dict[str, object]: ...
    def __pos__(self) -> Self: ...
    def mean(
        self, axis: int | tuple[int, ...] | None = ..., *, keepdims: bool = ...
    ) -> Self: ...
    # Named quantities have catalogue-generated operators. Restrict base
    # arithmetic to structural trees so affine points cannot inherit it.
    # K remains anchored to the class, preserving nominal tree equality.
    @overload
    def __add__[A, B, W](
        self: _StructuralQuantity[A, B, float], other: Quantity[K, W], /
    ) -> Quantity[K, W]: ...
    @overload
    def __add__[A, B](
        self: _StructuralQuantity[A, B, V], other: Quantity[K, float], /
    ) -> Quantity[K, V]: ...
    @overload
    def __add__[A, B](
        self: _StructuralQuantity[A, B, V], other: Quantity[K, V], /
    ) -> Quantity[K, V]: ...
    @overload
    def __sub__[A, B, W](
        self: _StructuralQuantity[A, B, float], other: Quantity[K, W], /
    ) -> Quantity[K, W]: ...
    @overload
    def __sub__[A, B](
        self: _StructuralQuantity[A, B, V], other: Quantity[K, float], /
    ) -> Quantity[K, V]: ...
    @overload
    def __sub__[A, B](
        self: _StructuralQuantity[A, B, V], other: Quantity[K, V], /
    ) -> Quantity[K, V]: ...
    @overload
    def __mul__[A, B](
        self: _StructuralQuantity[A, B, V], other: float, /
    ) -> Quantity[K, V]: ...
    @overload
    def __mul__[A, B, L: NonAffineKind, W](
        self: _StructuralQuantity[A, B, float], other: Quantity[L, W], /
    ) -> Quantity[Mul[K, L], W]: ...
    @overload
    def __mul__[A, B, L: NonAffineKind](
        self: _StructuralQuantity[A, B, V], other: Quantity[L, float], /
    ) -> Quantity[Mul[K, L], V]: ...
    @overload
    def __mul__[A, B, L: NonAffineKind](
        self: _StructuralQuantity[A, B, V], other: Quantity[L, V], /
    ) -> Quantity[Mul[K, L], V]: ...
    @overload
    def __truediv__[A, B](
        self: _StructuralQuantity[A, B, V], other: float, /
    ) -> Quantity[K, V]: ...
    @overload
    def __truediv__[A, B, L: NonAffineKind, W](
        self: _StructuralQuantity[A, B, float], other: Quantity[L, W], /
    ) -> Quantity[Div[K, L], W]: ...
    @overload
    def __truediv__[A, B, L: NonAffineKind](
        self: _StructuralQuantity[A, B, V], other: Quantity[L, float], /
    ) -> Quantity[Div[K, L], V]: ...
    @overload
    def __truediv__[A, B, L: NonAffineKind](
        self: _StructuralQuantity[A, B, V], other: Quantity[L, V], /
    ) -> Quantity[Div[K, L], V]: ...
    def __rmul__[A, B](
        self: _StructuralQuantity[A, B, V], other: float, /
    ) -> Quantity[K, V]: ...
    def __rtruediv__[A, B](
        self: _StructuralQuantity[A, B, V], other: float, /
    ) -> Quantity[Div[DimensionlessKind, K], V]: ...
    def __pow__[A, B, N: int](
        self: _StructuralQuantity[A, B, V], exponent: N, /
    ) -> Quantity[Pow[K, N], V]: ...
    def __neg__[A, B](self: _StructuralQuantity[A, B, V]) -> Quantity[K, V]: ...
    def __abs__[A, B](self: _StructuralQuantity[A, B, V]) -> Quantity[K, V]: ...
    def sum[A, B](
        self: _StructuralQuantity[A, B, V],
        axis: int | tuple[int, ...] | None = ...,
        *,
        keepdims: bool = ...,
    ) -> Quantity[K, V]: ...

def result_kind(op: str, left: Semantic, right: Semantic) -> Semantic: ...
def dimensions(kind: str | Semantic) -> tuple[int, ...]: ...
def _wrap(kind: str | Semantic, value: Any) -> Quantity[Any, Any]: ...

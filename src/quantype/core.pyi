# mypy: disable-error-code=overload-overlap
# pyright: reportOverlappingOverload=false
from collections.abc import Iterator
from typing import Any, ClassVar, Generic, Self, overload, override

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

from quantype._internal._kind_types import Div, Mul, NonAffineKind, Pow
from quantype._internal._semantics import Semantic
from quantype._internal._systems import Atomistic, UnitSystem
from quantype._internal._unit import Unit as Unit
from quantype._internal._unit import get_unit as get_unit
from quantype.kinds import DimensionlessKind

__all__ = [
    "Quantity",
    "Unit",
    "dimensions",
    "get_unit",
    "result_kind",
]

# PEP 695 cannot express the system default before Python 3.13. Public
# parameter names read better in diagnostics than private ones.
# ruff: noqa: PYI001
K = TypeVar("K")
V = TypeVar("V")
S = TypeVar("S", bound=UnitSystem, default=Atomistic)

type _StructuralQuantity[A, B, X, Y: UnitSystem] = (
    Quantity[Mul[A, B], X, Y] | Quantity[Div[A, B], X, Y] | Quantity[Pow[A, B], X, Y]
)

class Quantity(Generic[K, V, S]):
    _kind: str
    _semantic: Semantic
    _system: type[UnitSystem]
    _display: Unit[K] | None
    _echo: float | None
    __hash__: ClassVar[None]  # type: ignore[assignment]
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
    @property
    def system(self) -> type[S]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @classmethod
    def from_value[W](cls, value: W) -> Quantity[K, W, S]: ...
    @classmethod
    def parse(
        cls, data: object, *, units: tuple[Unit[Any], ...] = ...
    ) -> Quantity[K, float | npt.NDArray[np.float64], S]: ...
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any: ...
    def _check_unit(self, unit: Unit[K]) -> None: ...
    def magnitude(self, unit: Unit[K] | None = ...) -> V: ...
    def to(self, unit: Unit[K]) -> Self: ...
    def to_system[T: UnitSystem](self, system: type[T]) -> Quantity[K, V, T]: ...
    def to_dict(self, unit: Unit[K] | None = ...) -> dict[str, object]: ...
    def __pos__(self) -> Self: ...
    def mean(
        self, axis: int | tuple[int, ...] | None = ..., *, keepdims: bool = ...
    ) -> Self: ...
    def max(
        self, axis: int | tuple[int, ...] | None = ..., *, keepdims: bool = ...
    ) -> Self: ...
    def min(
        self, axis: int | tuple[int, ...] | None = ..., *, keepdims: bool = ...
    ) -> Self: ...
    def __getitem__(self, key: Any, /) -> Self: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Self]: ...
    # Value equality: bool for scalar storage, element-wise for arrays.
    @override
    def __eq__(self, other: object, /) -> Any: ...
    @override
    def __ne__(self, other: object, /) -> Any: ...
    # Ordering requires the same kind and system; K and S anchor to the class.
    @overload
    def __lt__(
        self: Quantity[K, float, S], other: Quantity[K, float, S], /
    ) -> bool: ...
    @overload
    def __lt__(self, other: Quantity[K, Any, S], /) -> Any: ...
    @overload
    def __le__(
        self: Quantity[K, float, S], other: Quantity[K, float, S], /
    ) -> bool: ...
    @overload
    def __le__(self, other: Quantity[K, Any, S], /) -> Any: ...
    @overload
    def __gt__(
        self: Quantity[K, float, S], other: Quantity[K, float, S], /
    ) -> bool: ...
    @overload
    def __gt__(self, other: Quantity[K, Any, S], /) -> Any: ...
    @overload
    def __ge__(
        self: Quantity[K, float, S], other: Quantity[K, float, S], /
    ) -> bool: ...
    @overload
    def __ge__(self, other: Quantity[K, Any, S], /) -> Any: ...
    # Named quantities have catalogue-generated operators. Restrict base
    # arithmetic to structural trees so affine points cannot inherit it.
    # K remains anchored to the class, preserving nominal tree equality.
    @overload
    def __add__[A, B, W](
        self: _StructuralQuantity[A, B, float, S], other: Quantity[K, W, S], /
    ) -> Quantity[K, W, S]: ...
    @overload
    def __add__[A, B](
        self: _StructuralQuantity[A, B, V, S], other: Quantity[K, float, S], /
    ) -> Quantity[K, V, S]: ...
    @overload
    def __add__[A, B](
        self: _StructuralQuantity[A, B, V, S], other: Quantity[K, V, S], /
    ) -> Quantity[K, V, S]: ...
    @overload
    def __sub__[A, B, W](
        self: _StructuralQuantity[A, B, float, S], other: Quantity[K, W, S], /
    ) -> Quantity[K, W, S]: ...
    @overload
    def __sub__[A, B](
        self: _StructuralQuantity[A, B, V, S], other: Quantity[K, float, S], /
    ) -> Quantity[K, V, S]: ...
    @overload
    def __sub__[A, B](
        self: _StructuralQuantity[A, B, V, S], other: Quantity[K, V, S], /
    ) -> Quantity[K, V, S]: ...
    @overload
    def __mul__[A, B](
        self: _StructuralQuantity[A, B, V, S], other: float, /
    ) -> Quantity[K, V, S]: ...
    @overload
    def __mul__[A, B, L: NonAffineKind, W](
        self: _StructuralQuantity[A, B, float, S], other: Quantity[L, W, S], /
    ) -> Quantity[Mul[K, L], W, S]: ...
    @overload
    def __mul__[A, B, L: NonAffineKind](
        self: _StructuralQuantity[A, B, V, S], other: Quantity[L, float, S], /
    ) -> Quantity[Mul[K, L], V, S]: ...
    @overload
    def __mul__[A, B, L: NonAffineKind](
        self: _StructuralQuantity[A, B, V, S], other: Quantity[L, V, S], /
    ) -> Quantity[Mul[K, L], V, S]: ...
    @overload
    def __truediv__[A, B](
        self: _StructuralQuantity[A, B, V, S], other: float, /
    ) -> Quantity[K, V, S]: ...
    @overload
    def __truediv__[A, B, L: NonAffineKind, W](
        self: _StructuralQuantity[A, B, float, S], other: Quantity[L, W, S], /
    ) -> Quantity[Div[K, L], W, S]: ...
    @overload
    def __truediv__[A, B, L: NonAffineKind](
        self: _StructuralQuantity[A, B, V, S], other: Quantity[L, float, S], /
    ) -> Quantity[Div[K, L], V, S]: ...
    @overload
    def __truediv__[A, B, L: NonAffineKind](
        self: _StructuralQuantity[A, B, V, S], other: Quantity[L, V, S], /
    ) -> Quantity[Div[K, L], V, S]: ...
    def __rmul__[A, B](
        self: _StructuralQuantity[A, B, V, S], other: float, /
    ) -> Quantity[K, V, S]: ...
    def __rtruediv__[A, B](
        self: _StructuralQuantity[A, B, V, S], other: float, /
    ) -> Quantity[Div[DimensionlessKind, K], V, S]: ...
    def __pow__[A, B, N: int](
        self: _StructuralQuantity[A, B, V, S], exponent: N, /
    ) -> Quantity[Pow[K, N], V, S]: ...
    def __neg__[A, B](self: _StructuralQuantity[A, B, V, S]) -> Quantity[K, V, S]: ...
    def __abs__[A, B](self: _StructuralQuantity[A, B, V, S]) -> Quantity[K, V, S]: ...
    def sum[A, B](
        self: _StructuralQuantity[A, B, V, S],
        axis: int | tuple[int, ...] | None = ...,
        *,
        keepdims: bool = ...,
    ) -> Quantity[K, V, S]: ...

def result_kind(op: str, left: Semantic, right: Semantic) -> Semantic: ...
def dimensions(kind: str | Semantic) -> tuple[int, ...]: ...
def _wrap(
    kind: str | Semantic,
    value: Any,
    system: type[UnitSystem] | None,
    *,
    display: Unit[Any] | None = ...,
    echo: float | None = ...,
) -> Quantity[Any, Any, Any]: ...
def _parse(
    cls: type[Quantity[Any, Any, Any]],
    data: object,
    units: tuple[Unit[Any], ...],
    system: type[UnitSystem],
) -> Quantity[Any, Any, Any]: ...

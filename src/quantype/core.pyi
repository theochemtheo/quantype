from typing import Any, Self

import numpy as np
import numpy.typing as npt

from quantype._internal._semantics import Semantic
from quantype._internal._unit import Unit as Unit
from quantype._internal._unit import canonical_unit as canonical_unit
from quantype._internal._unit import get_unit as get_unit

__all__ = [
    "Quantity",
    "Unit",
    "canonical_unit",
    "dimensions",
    "get_unit",
    "result_kind",
]

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

def result_kind(op: str, left: Semantic, right: Semantic) -> Semantic: ...
def dimensions(kind: str | Semantic) -> tuple[int, ...]: ...
def _wrap(kind: str | Semantic, value: Any) -> Quantity[Any, Any]: ...

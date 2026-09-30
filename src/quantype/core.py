"""Backend-neutral runtime. The adjacent stub is the static public contract.

Numerical operations intentionally use the stored object's own operators. Dynamic
typing is confined to this dispatch boundary, not exposed as a user's result type.
"""

# Runtime metadata is deliberately shared by the package's boundary modules.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

from quantype._internal._semantics import (
    KINDS,
    Kind,
    Semantic,
    addition,
    power,
    product,
)
from quantype._internal._unit import Unit, canonical_unit, get_unit

__all__ = [
    "Quantity",
    "Unit",
    "canonical_unit",
    "dimensions",
    "get_unit",
    "result_kind",
]

if TYPE_CHECKING:
    from types import GenericAlias


def result_kind(op: str, left: Semantic, right: Semantic) -> Semantic:
    """Shared physical algebra for arithmetic and autodiff."""
    if op not in {"mul", "div"}:
        raise ValueError(f"Unknown physical operation {op!r}")
    return product(cast("Any", op), left, right)


def dimensions(kind: str | Semantic) -> tuple[int, ...]:
    return (KINDS[kind] if isinstance(kind, str) else kind).dimensions


def _symbol(kind: Semantic) -> str:
    if isinstance(kind, Kind):
        return canonical_unit(kind).symbol
    if isinstance(kind.right, int):
        return f"({_symbol(kind.left)})^{kind.right}"
    operator = "*" if kind.operation == "mul" else "/"
    return f"({_symbol(kind.left)} {operator} {_symbol(kind.right)})"


_CLASSES: dict[Kind, type[Quantity[Any, Any]]] = {}


def _wrap(kind: str | Semantic, value: Any) -> Quantity[Any, Any]:
    # No coercion: JAX reconstruction also supplies sentinel leaves.
    semantic = KINDS[kind] if isinstance(kind, str) else kind
    cls = _CLASSES[semantic] if isinstance(semantic, Kind) else Quantity
    result = cast("Any", object.__new__(cls))
    result._value = value
    result._semantic = semantic
    result._display = None
    return cast("Quantity[Any, Any]", result)


class Quantity[K, V]:
    """A semantic quantity with canonical storage and an optional display unit."""

    _kind: str = ""
    _semantic: Semantic
    __array_priority__: ClassVar[int] = 10000

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.__dict__.get("_kind"):
            if "_semantic" not in cls.__dict__:
                cls._semantic = KINDS[cls._kind]
            if isinstance(cls._semantic, Kind):
                _CLASSES[cls._semantic] = cls

    @classmethod
    def __class_getitem__(cls, parameters: Any) -> GenericAlias:
        from quantype._internal._construction import StorageAlias

        return StorageAlias(cls, parameters)

    def __init__(self, value: object, unit: Unit[K], *, dtype: object = None) -> None:
        if dtype is not None:
            raise TypeError("dtype requires a parameterized quantity storage type")
        if not self._kind:
            raise TypeError("Construct a named quantity or use a unit")
        self._check_unit(unit)
        self._value = cast("V", cast("Any", unit)(value).value)
        self._display: Unit[K] | None = None

    @classmethod
    def define_unit(
        cls,
        name: str,
        *,
        reference: Unit[K],
        scale: float = 1.0,
        offset: float = 0.0,
        symbol: str | None = None,
    ) -> Unit[K]:
        if reference.semantic is not cls._semantic:
            raise ValueError(f"Expected {cls._kind}; received {reference.kind}")
        return Unit(
            name,
            reference.semantic,
            reference.scale * scale,
            reference.scale * offset + reference.offset,
            symbol,
        )

    @property
    def value(self) -> V:
        """The canonical numerical value; explicitly bypasses physical typing."""
        return self._value

    @property
    def kind(self) -> str:
        return str(self._semantic)

    @property
    def dimensions(self) -> tuple[int, ...]:
        return self._semantic.dimensions

    @classmethod
    def from_canonical[W](cls, value: W) -> Quantity[K, W]:
        if not cls._kind:
            raise TypeError("from_canonical requires a named quantity class")
        return cast("Quantity[K, W]", _wrap(cls._semantic, value))

    @classmethod
    def parse(cls, data: object, *, units: tuple[Unit[Any], ...] = ()) -> Self:
        import numpy as np

        from quantype.serialization import parse_quantity

        result = parse_quantity(cast("Any", cls), data, units=cast("Any", units))
        raw: Any = result.value
        if not isinstance(raw, (float, np.ndarray)):
            raise TypeError(
                "parse accepts scalar/NumPy storage; use a typed Pydantic field "
                "to validate backend quantities"
            )
        if isinstance(raw, np.ndarray):
            array = cast("Any", raw)
            if array.dtype != np.float64:
                result = _wrap(cls._semantic, np.asarray(array, dtype=np.float64))
        return cast("Self", result)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from quantype._internal._validation import pydantic_schema

        return pydantic_schema(cast("Any", cls), source_type, handler)

    def _check_unit(self, unit: Unit[K]) -> None:
        if unit.semantic is not self._semantic:
            raise ValueError(
                f"Expected {self.kind}; unit {unit.name!r} represents {unit.kind}"
            )

    def magnitude(self, unit: Unit[K] | None = None) -> V:
        if unit is None:
            if self._display is None:
                return self.value
            unit = self._display
        self._check_unit(unit)
        raw: Any = self.value
        if unit.offset != 0:
            raw = raw - unit.offset
        if unit.scale != 1:
            raw = raw / unit.scale
        return cast("V", raw)

    def to(self, unit: Unit[K]) -> Self:
        self._check_unit(unit)
        result = _wrap(self._semantic, self.value)
        result._display = unit
        return cast("Self", result)

    def to_dict(self, unit: Unit[K] | None = None) -> dict[str, object]:
        from quantype.serialization import to_dict

        return to_dict(cast("Any", self), cast("Any", unit))

    def _add_sub(self, other: object, *, subtract: bool) -> Quantity[Any, Any]:
        if not isinstance(other, Quantity):
            raise TypeError(f"Expected a quantity, received {type(other).__name__}")
        rhs = cast("Quantity[Any, Any]", other)
        kind = addition(self._semantic, rhs._semantic, subtract=subtract)
        lhs: Any = self.value
        return _wrap(kind, lhs - rhs.value if subtract else lhs + rhs.value)

    def __add__(self, other: object) -> Quantity[Any, Any]:
        return self._add_sub(other, subtract=False)

    def __sub__(self, other: object) -> Quantity[Any, Any]:
        return self._add_sub(other, subtract=True)

    def __mul__(self, other: Any) -> Quantity[Any, Any]:
        lhs: Any = self.value
        if isinstance(other, Quantity):
            rhs = cast("Quantity[Any, Any]", other)
            return _wrap(product("mul", self._semantic, rhs._semantic), lhs * rhs.value)
        self._check_scalar(other)
        return _wrap(self._semantic, lhs * other)

    def __rmul__(self, other: Any) -> Quantity[Any, Any]:
        return self * other

    def __truediv__(self, other: Any) -> Quantity[Any, Any]:
        lhs: Any = self.value
        if isinstance(other, Quantity):
            rhs = cast("Quantity[Any, Any]", other)
            return _wrap(product("div", self._semantic, rhs._semantic), lhs / rhs.value)
        self._check_scalar(other)
        return _wrap(self._semantic, lhs / other)

    def __rtruediv__(self, other: Any) -> Quantity[Any, Any]:
        self._check_scalar(other)
        return _wrap(
            product("div", KINDS["Dimensionless"], self._semantic), other / self.value
        )

    def _check_scalar(self, value: object) -> None:
        if isinstance(self._semantic, Kind) and self._semantic.affine:
            raise TypeError(
                "Scale a TemperatureDifference, not an absolute Temperature"
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("Quantity scaling requires a real scalar")

    def __pow__(self, exponent: object) -> Quantity[Any, Any]:
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise TypeError("Only integer quantity powers are supported")
        kind = power(self._semantic, exponent)
        raw: Any = self.value
        return _wrap(kind, raw**exponent)

    def __neg__(self) -> Self:
        self._check_scalar(-1)
        raw: Any = self.value
        return cast("Self", _wrap(self._semantic, -raw))

    def __pos__(self) -> Self:
        return self

    def __abs__(self) -> Self:
        self._check_scalar(1)
        raw: Any = self.value
        return cast("Self", _wrap(self._semantic, abs(raw)))

    def _reduce(self, method: str, axis: int | None, *, keepdims: bool) -> Self:
        raw: Any = self.value
        if isinstance(raw, (int, float)):
            if axis is not None:
                raise ValueError("A scalar has no reduction axis")
            return self
        if type(raw).__module__.startswith("torch"):
            # PyTorch calls the axis `dim`; dim=None reduces all dimensions.
            reduced = getattr(raw, method)(dim=axis, keepdim=keepdims)
        else:
            reduced = getattr(raw, method)(axis=axis, keepdims=keepdims)
        if type(raw).__module__.startswith("numpy"):
            import numpy as np

            # Preserve the storage type, including when reduction produces 0-D.
            if isinstance(raw, np.ndarray):
                reduced = np.asarray(reduced)
        return cast("Self", _wrap(self._semantic, reduced))

    def sum(self, axis: int | None = None, *, keepdims: bool = False) -> Self:
        if isinstance(self._semantic, Kind) and self._semantic.affine:
            raise TypeError("Cannot sum absolute Temperatures")
        return self._reduce("sum", axis, keepdims=keepdims)

    def mean(self, axis: int | None = None, *, keepdims: bool = False) -> Self:
        return self._reduce("mean", axis, keepdims=keepdims)

    def __array__(self, dtype: object = None, copy: object = None) -> Any:
        raise TypeError(
            "Implicit array coercion drops units; use .value or .magnitude(unit)"
        )

    def __array_ufunc__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("Use quantity arithmetic or quantype.units math helpers")

    def __array_function__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "Use quantity methods such as .sum() and .mean(), or pass "
            ".value or .magnitude(unit) to NumPy explicitly"
        )

    @override
    def __repr__(self) -> str:
        symbol = self._display.symbol if self._display else _symbol(self._semantic)
        return f"{self.magnitude()!s} {symbol}"

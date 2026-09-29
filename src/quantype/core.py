"""Backend-neutral runtime. The adjacent stub is the static public contract.

Numerical operations intentionally use the stored object's own operators. Dynamic
typing is confined to this dispatch boundary, not exposed as a user's result type.
"""

# Runtime metadata is deliberately shared by the package's boundary modules.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from functools import cache
from typing import Any, ClassVar, Self, cast, override

from quantype._registry import POWERS, QUANTITIES, RELATIONS, UNITS


def result_kind(op: str, left: str, right: str) -> str:
    """Resolve semantic algebra, never infer semantics from dimensions alone."""
    if "Temperature" in (left, right):
        raise TypeError("Absolute Temperature cannot participate in products or ratios")
    known = RELATIONS.get((op, left, right))
    if known is not None:
        return known
    if op not in {"mul", "div"}:
        raise ValueError(f"Unknown physical operation {op!r}")
    return f"{'Mul' if op == 'mul' else 'Div'}[{left},{right}]"


def _expression_parts(kind: str) -> tuple[str, str, str]:
    operator, body = kind.split("[", 1)
    body = body[:-1]
    depth = 0
    for index, char in enumerate(body):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            return operator, body[:index], body[index + 1 :]
    raise ValueError(f"Unknown quantity kind {kind!r}")


def dimensions(kind: str) -> tuple[int, ...]:
    if kind in QUANTITIES:
        return QUANTITIES[kind].dimensions
    operation, left, right = _expression_parts(kind)
    lhs = dimensions(left)
    if operation == "Pow":
        return tuple(exponent * int(right) for exponent in lhs)
    rhs = dimensions(right)
    sign = 1 if operation == "Mul" else -1
    return tuple(a + sign * b for a, b in zip(lhs, rhs, strict=True))


def _symbol(kind: str) -> str:
    if kind in QUANTITIES:
        return get_unit(QUANTITIES[kind].canonical_unit).symbol
    operation, left, right = _expression_parts(kind)
    if operation == "Pow":
        return f"({_symbol(left)})^{right}"
    return f"({_symbol(left)} {'*' if operation == 'Mul' else '/'} {_symbol(right)})"


def _wrap(kind: str, value: Any) -> Quantity[Any, Any]:
    # No coercion here: also used to reconstruct JAX pytrees with sentinel leaves.
    from quantype import _generated

    cls = getattr(_generated, kind, Quantity)
    result = cast("Any", object.__new__(cls))
    result._value = value
    result._kind = kind
    result._display = None
    return cast("Quantity[Any, Any]", result)


@cache
def get_unit(name: str) -> Unit[Any]:
    """Look up a catalogue name or alias; never evaluate unit expressions."""
    for identifier, spec in UNITS.items():
        if name == identifier or name in spec.aliases:
            return Unit(identifier, spec.kind, spec.scale, spec.offset, spec.symbol)
    raise ValueError(f"Unknown unit {name!r}")


class Unit[K]:
    __array_priority__: ClassVar[int] = 10000

    def __init__(
        self,
        name: str,
        kind: str,
        scale: float = 1.0,
        offset: float = 0.0,
        symbol: str | None = None,
    ) -> None:
        if scale <= 0:
            raise ValueError("Unit scale must be positive")
        self.name = name
        self.kind = kind
        self.scale = scale
        self.offset = offset
        self.symbol = symbol if symbol is not None else name

    def __call__[V](self, value: V) -> Quantity[K, V]:
        raw: Any = value
        if isinstance(raw, bool):
            raise TypeError("Boolean values are not physical magnitudes")
        if isinstance(raw, (int, float)):
            raw = float(raw)
        elif not (hasattr(raw, "shape") and hasattr(raw, "dtype")):
            raise TypeError("Use a real scalar or numerical array as a magnitude")
        # Identity conversions must preserve tensor leaves and backend storage.
        if self.scale != 1:
            raw = raw * self.scale
        if self.offset != 0:
            raw = raw + self.offset
        result = _wrap(self.kind, raw)
        return cast("Quantity[K, V]", result)

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


class Quantity[K, V]:
    """A semantic quantity with canonical storage and an optional display unit."""

    _kind: str = ""
    __array_priority__: ClassVar[int] = 10000

    def __init__(self, value: V) -> None:
        if not self._kind:
            raise TypeError("Construct a named quantity or use a unit")
        self._value = value
        self._display: str | None = None

    @property
    def value(self) -> V:
        """The canonical numerical value; explicitly bypasses physical typing."""
        return self._value

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def dimensions(self) -> tuple[int, ...]:
        return dimensions(self.kind)

    @classmethod
    def from_canonical[W](cls, value: W) -> Quantity[K, W]:
        if not cls._kind:
            raise TypeError("from_canonical requires a named quantity class")
        return cast("Quantity[K, W]", _wrap(cls._kind, value))

    @classmethod
    def parse(cls, data: object) -> Self:
        import numpy as np

        from quantype._validation import parse_quantity

        result = parse_quantity(cast("Any", cls), data)
        raw: Any = result.value
        if not isinstance(raw, (float, np.ndarray)):
            raise TypeError(
                "parse accepts scalar/NumPy storage; use a typed Pydantic field "
                "to validate backend quantities"
            )
        if isinstance(raw, np.ndarray):
            array = cast("Any", raw)
            if array.dtype != np.float64:
                result = parse_quantity(
                    cast("Any", cls),
                    {"value": array, "units": QUANTITIES[result.kind].canonical_unit},
                )
        return cast("Self", result)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from quantype._validation import pydantic_schema

        return pydantic_schema(cast("Any", cls), source_type, handler)

    def _check_unit(self, unit: Unit[K]) -> None:
        if unit.kind != self.kind:
            raise ValueError(
                f"Expected {self.kind}; unit {unit.name!r} represents {unit.kind}"
            )

    def magnitude(self, unit: Unit[K] | None = None) -> V:
        if unit is None:
            if self._display is None:
                return self.value
            unit = cast("Unit[K]", get_unit(self._display))
        self._check_unit(unit)
        raw: Any = self.value
        if unit.offset != 0:
            raw = raw - unit.offset
        if unit.scale != 1:
            raw = raw / unit.scale
        return cast("V", raw)

    def to(self, unit: Unit[K]) -> Self:
        self._check_unit(unit)
        result = _wrap(self.kind, self.value)
        result._display = unit.name
        return cast("Self", result)

    def to_dict(self, unit: Unit[K] | None = None) -> dict[str, object]:
        from quantype._validation import serialize_quantity

        return serialize_quantity(cast("Any", self), cast("Any", unit))

    def _add_sub(self, other: object, *, subtract: bool) -> Quantity[Any, Any]:
        if not isinstance(other, Quantity):
            raise TypeError(f"Expected a quantity, received {type(other).__name__}")
        rhs = cast("Quantity[Any, Any]", other)
        left, right = self.kind, rhs.kind
        kind = left
        if left == "Temperature" and right == "Temperature":
            if not subtract:
                raise TypeError("Cannot add two absolute Temperatures")
            kind = "TemperatureDifference"
        elif (left == "Temperature" and right == "TemperatureDifference") or (
            left == "TemperatureDifference" and right == "Temperature" and not subtract
        ):
            kind = "Temperature"
        elif left != right:
            operation = "subtract" if subtract else "add"
            raise TypeError(f"Cannot {operation} {left} and {right}")
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
            return _wrap(result_kind("mul", self.kind, rhs.kind), lhs * rhs.value)
        self._check_scalar(other)
        return _wrap(self.kind, lhs * other)

    def __rmul__(self, other: Any) -> Quantity[Any, Any]:
        return self * other

    def __truediv__(self, other: Any) -> Quantity[Any, Any]:
        lhs: Any = self.value
        if isinstance(other, Quantity):
            rhs = cast("Quantity[Any, Any]", other)
            return _wrap(result_kind("div", self.kind, rhs.kind), lhs / rhs.value)
        self._check_scalar(other)
        return _wrap(self.kind, lhs / other)

    def __rtruediv__(self, other: Any) -> Quantity[Any, Any]:
        self._check_scalar(other)
        return _wrap(result_kind("div", "Dimensionless", self.kind), other / self.value)

    def _check_scalar(self, value: object) -> None:
        if self.kind == "Temperature":
            raise TypeError(
                "Scale a TemperatureDifference, not an absolute Temperature"
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("Quantity scaling requires a real scalar")

    def __pow__(self, exponent: object) -> Quantity[Any, Any]:
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise TypeError("Only integer quantity powers are supported")
        if self.kind == "Temperature":
            raise TypeError("Absolute Temperature cannot be exponentiated")
        kind = POWERS.get((self.kind, exponent), f"Pow[{self.kind},{exponent}]")
        raw: Any = self.value
        return _wrap(kind, raw**exponent)

    def __neg__(self) -> Self:
        self._check_scalar(-1)
        raw: Any = self.value
        return cast("Self", _wrap(self.kind, -raw))

    def __pos__(self) -> Self:
        return self

    def __abs__(self) -> Self:
        self._check_scalar(1)
        raw: Any = self.value
        return cast("Self", _wrap(self.kind, abs(raw)))

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
            reduced = np.asarray(reduced)
        return cast("Self", _wrap(self.kind, reduced))

    def sum(self, axis: int | None = None, *, keepdims: bool = False) -> Self:
        if self.kind == "Temperature":
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

    @override
    def __repr__(self) -> str:
        symbol = get_unit(self._display).symbol if self._display else _symbol(self.kind)
        return f"{self.magnitude()!s} {symbol}"

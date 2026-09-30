"""Backend-neutral runtime. The adjacent stub is the static public contract.

Numerical operations intentionally use the stored object's own operators. Dynamic
typing is confined to this dispatch boundary, not exposed as a user's result type.

Each quantity holds raw numbers in the coherent units of its unit system, its
physical kind, the system, and optionally the unit it was given in (display).
"""

# Runtime metadata is deliberately shared by the package's boundary modules.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

from quantype._internal._semantics import (
    DIMENSIONLESS_KINDS,
    KINDS,
    TEMPERATURE_DIFFERENCES,
    Kind,
    Semantic,
    addition,
    dimensionless_kind,
    power,
    product,
)
from quantype._internal._systems import (
    Atomistic,
    UnitSystem,
    factor,
    into_system,
    require_system,
)
from quantype._internal._unit import Unit, get_unit

__all__ = [
    "Quantity",
    "Unit",
    "dimensions",
    "get_unit",
    "result_kind",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import GenericAlias


def result_kind(op: str, left: Semantic, right: Semantic) -> Semantic:
    """Shared physical algebra for arithmetic and autodiff."""
    if op not in {"mul", "div"}:
        raise ValueError(f"Unknown physical operation {op!r}")
    return product(cast("Any", op), left, right)


def dimensions(kind: str | Semantic) -> tuple[int, ...]:
    return (KINDS[kind] if isinstance(kind, str) else kind).dimensions


def _symbol(kind: Semantic, system: type[UnitSystem]) -> str:
    if isinstance(kind, Kind):
        return system.unit_for(kind).symbol
    if isinstance(kind.right, int):
        return f"({_symbol(kind.left, system)})^{kind.right}"
    operator_symbol = "*" if kind.operation == "mul" else "/"
    left, right = _symbol(kind.left, system), _symbol(kind.right, system)
    return f"({left} {operator_symbol} {right})"


_CLASSES: dict[Kind, type[Quantity[Any, Any, Any]]] = {}
_STRUCTURAL_CLASSES: dict[Kind, type[Quantity[Any, Any, Any]]] = {}


def _wrap(
    kind: str | Semantic,
    value: Any,
    system: type[UnitSystem] | None,
    *,
    display: Unit[Any] | None = None,
    echo: float | None = None,
) -> Quantity[Any, Any, Any]:
    """Wrap trusted raw numbers; ``system=None`` means the default system."""
    # No coercion: JAX reconstruction also supplies sentinel leaves.
    semantic = KINDS[kind] if isinstance(kind, str) else kind
    if isinstance(semantic, Kind):
        cls = _CLASSES[semantic]
    else:
        # The leftmost kind owns the result API; reciprocal inference separately
        # checks all leaves so mixed-catalogue scalar numerators stay ambiguous.
        leaf = semantic.left
        while not isinstance(leaf, Kind):
            leaf = leaf.left
        owner = DIMENSIONLESS_KINDS.get(leaf)
        cls = Quantity if owner is None else _STRUCTURAL_CLASSES.get(owner, Quantity)
    result = cast("Any", object.__new__(cls))
    result._value = value
    result._semantic = semantic
    result._system = Atomistic if system is None else system
    result._display = display
    result._echo = echo
    return cast("Quantity[Any, Any, Any]", result)


def _delta_unit(point: Unit[Any], kind: Semantic) -> Unit[Any] | None:
    """The temperature-difference unit matching a point unit (°C → Δ°C)."""
    if not isinstance(kind, Kind):
        return None
    try:
        unit = get_unit(f"delta_{point.name}", kind=kind)
    except ValueError:
        return None
    return unit if unit.semantic is kind and unit.scale == point.scale else None


def _sum_display(
    kind: Semantic, left: Quantity[Any, Any, Any], right: Quantity[Any, Any, Any]
) -> Unit[Any] | None:
    # The left operand's unit wins when it can present the result.
    for display in (left._display, right._display):
        if display is not None and display.semantic is kind:
            return display
    if (
        left._display is not None
        and isinstance(left._semantic, Kind)
        and TEMPERATURE_DIFFERENCES.get(left._semantic) is kind
    ):
        return _delta_unit(left._display, kind)
    return None


class Quantity[K, V, S: UnitSystem]:
    """A semantic quantity: raw numbers in a unit system, and a display unit."""

    _kind: str = ""
    _semantic: Semantic
    _system: type[UnitSystem]
    _display: Unit[K] | None
    _echo: float | None
    __array_priority__: ClassVar[int] = 10000
    # Equality compares values, and array storage is mutable.
    __hash__: ClassVar[None] = None  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "_catalogue_dimensionless" in cls.__dict__:
            _STRUCTURAL_CLASSES[cls.__dict__["_catalogue_dimensionless"]] = cls
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
        built = cast("Any", unit)(value)
        self._value = cast("V", built._value)
        self._system = Atomistic
        self._display = unit
        self._echo = built._echo

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
        """Raw numbers in the unit system's units; explicitly bypasses units."""
        return self._value

    @property
    def kind(self) -> str:
        return str(self._semantic)

    @property
    def dimensions(self) -> tuple[int, ...]:
        return self._semantic.dimensions

    @property
    def system(self) -> type[UnitSystem]:
        return self._system

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple[int, ...](getattr(self._value, "shape", ()))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def from_value[W](cls, value: W) -> Quantity[K, W, Any]:
        """Trusted wrap of raw numbers already in the default system's units."""
        if not cls._kind:
            raise TypeError("from_value requires a named quantity class")
        return cast("Quantity[K, W, Any]", _wrap(cls._semantic, value, None))

    @classmethod
    def parse(cls, data: object, *, units: tuple[Unit[Any], ...] = ()) -> Self:
        return cast("Self", _parse(cls, data, units, Atomistic))

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
        """Magnitude in ``unit``, else the display unit, else the system's unit."""
        if self._echo is not None and (unit is None or unit == self._display):
            return cast("V", self._echo)
        selected: Unit[Any] | None = unit or self._display
        if selected is None:
            if not isinstance(self._semantic, Kind):
                return self._value
            selected = self._system.unit_for(self._semantic)
        self._check_unit(selected)
        from quantype._internal._storage import unit_conversion

        scale, offset = into_system(selected, self._system)
        return cast("V", unit_conversion(self._value, scale, offset, inverse=True))

    def to(self, unit: Unit[K]) -> Self:
        """Present in ``unit``; the stored numbers are unchanged."""
        self._check_unit(unit)
        return cast(
            "Self", _wrap(self._semantic, self._value, self._system, display=unit)
        )

    def to_system(self, system: type[UnitSystem]) -> Quantity[K, V, Any]:
        """Rescale the raw numbers into another unit system; the display is kept."""
        target = require_system(system)
        raw: Any = self._value
        if target is not self._system:
            from quantype._internal._storage import (
                unit_conversion,
                warn_if_out_of_range,
            )

            # Temperatures share kelvin-scaled bases, so there is never an offset.
            ratio = factor(self._system, self._semantic) / factor(
                target, self._semantic
            )
            converted = unit_conversion(raw, ratio, 0.0)
            if type(raw).__module__.startswith("numpy"):
                import numpy as np

                exact = np.asarray(raw, dtype=np.float64) * ratio
                warn_if_out_of_range(exact, converted)
            raw = converted
        return cast(
            "Quantity[K, V, Any]",
            _wrap(self._semantic, raw, target, display=self._display, echo=self._echo),
        )

    def to_dict(self, unit: Unit[K] | None = None) -> dict[str, object]:
        from quantype.serialization import to_dict

        return to_dict(cast("Any", self), cast("Any", unit))

    def _same_system(self, other: Quantity[Any, Any, Any]) -> None:
        if other._system is not self._system:
            raise TypeError(
                f"Cannot combine {self._system.__name__} and "
                f"{other._system.__name__} quantities; convert one explicitly "
                "with .to_system(...)"
            )

    def _with(self, raw: Any, *, display: bool = True) -> Self:
        """Same kind and system; same-kind operations keep the display unit."""
        return cast(
            "Self",
            _wrap(
                self._semantic,
                raw,
                self._system,
                display=self._display if display else None,
            ),
        )

    def _add_sub(self, other: object, *, subtract: bool) -> Quantity[Any, Any, Any]:
        if not isinstance(other, Quantity):
            raise TypeError(f"Expected a quantity, received {type(other).__name__}")
        rhs = cast("Quantity[Any, Any, Any]", other)
        self._same_system(rhs)
        kind = addition(self._semantic, rhs._semantic, subtract=subtract)
        lhs: Any = self._value
        raw = lhs - rhs._value if subtract else lhs + rhs._value
        return _wrap(kind, raw, self._system, display=_sum_display(kind, self, rhs))

    def __add__(self, other: object) -> Quantity[Any, Any, Any]:
        return self._add_sub(other, subtract=False)

    def __sub__(self, other: object) -> Quantity[Any, Any, Any]:
        return self._add_sub(other, subtract=True)

    def __mul__(self, other: Any) -> Quantity[Any, Any, Any]:
        lhs: Any = self._value
        if isinstance(other, Quantity):
            rhs = cast("Quantity[Any, Any, Any]", other)
            self._same_system(rhs)
            kind = product("mul", self._semantic, rhs._semantic)
            return _wrap(kind, lhs * rhs._value, self._system)
        self._check_scalar(other)
        return self._with(lhs * other)

    def __rmul__(self, other: Any) -> Quantity[Any, Any, Any]:
        return self * other

    def __truediv__(self, other: Any) -> Quantity[Any, Any, Any]:
        lhs: Any = self._value
        if isinstance(other, Quantity):
            rhs = cast("Quantity[Any, Any, Any]", other)
            self._same_system(rhs)
            kind = product("div", self._semantic, rhs._semantic)
            return _wrap(kind, lhs / rhs._value, self._system)
        self._check_scalar(other)
        return self._with(lhs / other)

    def __rtruediv__(self, other: Any) -> Quantity[Any, Any, Any]:
        self._check_scalar(other)
        numerator = dimensionless_kind(self._semantic)
        kind = product("div", numerator, self._semantic)
        return _wrap(kind, other / self._value, self._system)

    def _check_scalar(self, value: object) -> None:
        if isinstance(self._semantic, Kind) and self._semantic.affine:
            raise TypeError(
                "Scale a TemperatureDifference, not an absolute Temperature"
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("Quantity scaling requires a real scalar")

    def __pow__(self, exponent: object) -> Quantity[Any, Any, Any]:
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise TypeError("Only integer quantity powers are supported")
        kind = power(self._semantic, exponent)
        raw: Any = self._value
        return _wrap(kind, raw**exponent, self._system)

    def __neg__(self) -> Self:
        self._check_scalar(-1)
        raw: Any = self._value
        return self._with(-raw)

    def __pos__(self) -> Self:
        return self

    def __abs__(self) -> Self:
        self._check_scalar(1)
        raw: Any = self._value
        return self._with(abs(raw))

    def _compare(self, other: object, compare: Callable[[Any, Any], Any]) -> Any:
        if not isinstance(other, Quantity):
            return NotImplemented
        rhs = cast("Quantity[Any, Any, Any]", other)
        self._same_system(rhs)
        if rhs._semantic != self._semantic:
            raise TypeError(f"Cannot compare {self.kind} and {rhs.kind}")
        return compare(self._value, rhs._value)

    @override
    def __eq__(self, other: object) -> Any:
        if not isinstance(other, Quantity):
            return NotImplemented
        rhs = cast("Quantity[Any, Any, Any]", other)
        if rhs._semantic != self._semantic:
            return False
        return self._compare(rhs, operator.eq)

    @override
    def __ne__(self, other: object) -> Any:
        if not isinstance(other, Quantity):
            return NotImplemented
        rhs = cast("Quantity[Any, Any, Any]", other)
        if rhs._semantic != self._semantic:
            return True
        return self._compare(rhs, operator.ne)

    def __lt__(self, other: object) -> Any:
        return self._compare(other, operator.lt)

    def __le__(self, other: object) -> Any:
        return self._compare(other, operator.le)

    def __gt__(self, other: object) -> Any:
        return self._compare(other, operator.gt)

    def __ge__(self, other: object) -> Any:
        return self._compare(other, operator.ge)

    def __getitem__(self, key: Any) -> Self:
        raw: Any = self._value
        if isinstance(raw, (int, float)):
            raise TypeError("Scalar quantities cannot be indexed")
        selected = raw[key]
        if type(raw).__module__.startswith("numpy"):
            import numpy as np

            # Preserve ndarray storage when indexing selects a single element.
            if isinstance(raw, np.ndarray):
                selected = np.asarray(selected)
        return self._with(selected)

    def __len__(self) -> int:
        raw: Any = self._value
        if isinstance(raw, (int, float)):
            raise TypeError("Scalar quantities have no length")
        return len(raw)

    def __iter__(self) -> Iterator[Self]:
        for index in range(len(self)):
            yield self[index]

    def _reduce(self, method: str, axis: int | None, *, keepdims: bool) -> Self:
        raw: Any = self._value
        if isinstance(raw, (int, float)):
            if axis is not None:
                raise ValueError("A scalar has no reduction axis")
            return self._with(raw)
        if type(raw).__module__.startswith("torch"):
            # PyTorch calls the axis `dim`, and its max/min also return indices.
            if method in {"max", "min"}:
                dims = () if axis is None else axis
                reduced = getattr(raw, f"a{method}")(dim=dims, keepdim=keepdims)
            else:
                reduced = getattr(raw, method)(dim=axis, keepdim=keepdims)
        else:
            reduced = getattr(raw, method)(axis=axis, keepdims=keepdims)
        if type(raw).__module__.startswith("numpy"):
            import numpy as np

            # Preserve the storage type, including when reduction produces 0-D.
            if isinstance(raw, np.ndarray):
                reduced = np.asarray(reduced)
        return self._with(reduced)

    def sum(self, axis: int | None = None, *, keepdims: bool = False) -> Self:
        if isinstance(self._semantic, Kind) and self._semantic.affine:
            raise TypeError("Cannot sum absolute Temperatures")
        return self._reduce("sum", axis, keepdims=keepdims)

    def mean(self, axis: int | None = None, *, keepdims: bool = False) -> Self:
        return self._reduce("mean", axis, keepdims=keepdims)

    def max(self, axis: int | None = None, *, keepdims: bool = False) -> Self:
        return self._reduce("max", axis, keepdims=keepdims)

    def min(self, axis: int | None = None, *, keepdims: bool = False) -> Self:
        return self._reduce("min", axis, keepdims=keepdims)

    def __array__(self, dtype: object = None, copy: object = None) -> Any:
        raise TypeError(
            "Implicit array coercion drops units; use .value or .magnitude(unit)"
        )

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        # A NumPy scalar on the left (np.float64(2) * q) dispatches here.
        if (
            method == "__call__"
            and not kwargs
            and len(inputs) == 2  # noqa: PLR2004 -- binary ufunc
            and ufunc.__name__ in {"multiply", "divide"}
        ):
            left, right = inputs
            if left is self and not isinstance(right, Quantity):
                return self * right if ufunc.__name__ == "multiply" else self / right
            if right is self and not isinstance(left, Quantity):
                if ufunc.__name__ == "multiply":
                    return self.__rmul__(left)
                return self.__rtruediv__(left)
        raise TypeError("Use quantity arithmetic or quantype.units math helpers")

    def __array_function__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "Use quantity methods such as .sum() and .mean(), or pass "
            ".value or .magnitude(unit) to NumPy explicitly"
        )

    def _presentation(self) -> str:
        if self._display is not None:
            symbol = self._display.symbol
        else:
            symbol = _symbol(self._semantic, self._system)
        return f"{self.magnitude()!s} {symbol}"

    @override
    def __str__(self) -> str:
        return self._presentation()

    @override
    def __repr__(self) -> str:
        name = type(self).__name__ if isinstance(self._semantic, Kind) else "Quantity"
        system = "" if self._system is Atomistic else f", {self._system.__name__}"
        return f"{name}({self._presentation()}{system})"


def _parse(
    cls: type[Quantity[Any, Any, Any]],
    data: object,
    units: tuple[Unit[Any], ...],
    system: type[UnitSystem],
) -> Quantity[Any, Any, Any]:
    import numpy as np

    from quantype.serialization import parse_quantity

    result: Any = parse_quantity(cast("Any", cls), data, units=units, system=system)
    raw: Any = result.value
    if not isinstance(raw, (float, np.ndarray)):
        raise TypeError(
            "parse accepts scalar/NumPy storage; use a typed Pydantic field "
            "to validate backend quantities"
        )
    if isinstance(raw, np.ndarray):
        array = cast("Any", raw)
        if array.dtype != np.float64:
            result = _wrap(
                cls._semantic,
                np.asarray(array, dtype=np.float64),
                system,
                display=result._display,
            )
    return result

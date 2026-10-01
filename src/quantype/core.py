"""Backend-neutral runtime. The adjacent stub is the static public contract.

Numerical operations intentionally use the stored object's own operators. Dynamic
typing is confined to this dispatch boundary, not exposed as a user's result type.

Each quantity holds raw numbers in the coherent units of its unit system, its
physical kind, the system, and optionally the unit it was given in (display).
"""

# Runtime metadata is deliberately shared by the package's boundary modules.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import numbers
import operator
import sys
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

from quantype._internal._products import class_for
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
from quantype._internal._storage import keep_storage, real_operand
from quantype._internal._systems import (
    Atomistic,
    UnitSystem,
    coherence,
    coherent_symbol,
    factor,
    into_system,
    is_coherent,
    require_system,
)
from quantype._internal._unit import Unit, get_unit

__all__ = [
    "Constant",
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
    """A structural quantity's symbol: its leaves in their coherent units."""
    if isinstance(kind, Kind):
        return coherent_symbol(system, kind)
    if isinstance(kind.right, int):
        return f"({_symbol(kind.left, system)})^{kind.right}"
    operator_symbol = "*" if kind.operation == "mul" else "/"
    left, right = _symbol(kind.left, system), _symbol(kind.right, system)
    return f"({left} {operator_symbol} {right})"


def _numpy_bool() -> type:
    """NumPy's boolean scalar type when NumPy is loaded; it is not a Python bool."""
    numpy = sys.modules.get("numpy")
    return bool if numpy is None else cast("type", numpy.bool_)


def _rescaled(raw: Any, scale: float) -> Any:
    """Apply a coherence factor; the usual factor of 1 leaves storage untouched."""
    return raw if scale == 1.0 else raw * scale


def _product_scale(
    system: type[UnitSystem], left: Semantic, right: Semantic, result: Semantic, op: str
) -> float:
    """The factor taking raw operands' product or ratio to the result's storage."""
    if is_coherent(system):
        return 1.0
    lhs, rhs = coherence(system, left), coherence(system, right)
    return (lhs * rhs if op == "mul" else lhs / rhs) / coherence(system, result)


def require_unit(unit: object, kind: str, hint: str = "") -> None:
    """Reject anything but a unit with a message that names the right object.

    ``hint`` explains a likely mix-up when the string isn't a unit name.
    """
    if isinstance(unit, Unit):
        return
    semantic = KINDS.get(kind)
    example = f"u.{semantic.canonical_unit}" if semantic is not None else "u.nm"
    if isinstance(unit, str):
        try:
            known: Unit[Any] | None = get_unit(unit)
        except ValueError:
            known = None
        if known is not None and unit.isidentifier():
            example, hint = f"u.{unit}", ""
        message = (
            f"Units are objects, such as {example}, not strings. To read a "
            f"quantity from text, use {kind or 'Length'}.parse('<number> <unit>')."
        )
    else:
        message = f"Expected a unit, such as {example}; received {type(unit).__name__}."
    raise TypeError(f"{message} {hint}".rstrip())


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
    elif (pair := class_for(semantic)) is not None:
        cls = pair
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
        require_unit(reference, cls._kind)
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
    def unit(self) -> Unit[K] | None:
        """The unit ``magnitude()`` and ``repr`` use; None for unnamed products."""
        if self._display is not None:
            return self._display
        if isinstance(self._semantic, Kind):
            return cast("Unit[K]", self._system.unit_for(self._semantic))
        return None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple[int, ...](getattr(self._value, "shape", ()))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def from_value[W](cls, value: W) -> Quantity[K, W, Any]:
        """Trusted wrap of raw numbers already in the default system's units."""
        if getattr(cls, "_semantic", None) is None:
            raise TypeError("from_value requires a named quantity class")
        return cast("Quantity[K, W, Any]", _wrap(cls._semantic, value, None))

    @classmethod
    def reinterpret[W, T: UnitSystem](
        cls, quantity: Quantity[Any, W, T]
    ) -> Quantity[K, W, T]:
        """The same physical value named as this kind, which must share its dimensions.

        Kinds are nominal, so equal dimensions never convert implicitly: an
        ``EnergyDensity`` is not a ``Pressure`` until reinterpreted. The unit
        system and storage are kept. Absolute and difference temperatures are
        points and vectors, so reinterpreting between them is refused.
        """
        if not cls._kind:
            raise TypeError("reinterpret requires a named quantity class")
        if not isinstance(quantity, Quantity):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected a quantity, received {type(quantity).__name__}")
        source, target = quantity._semantic, cls._semantic
        if source.dimensions != target.dimensions:
            raise TypeError(
                f"Cannot reinterpret {source} as {target}: their dimensions differ"
            )
        if source is not target and any(
            isinstance(kind, Kind) and kind.affine for kind in (source, target)
        ):
            raise TypeError(
                f"Cannot reinterpret {source} as {target}; subtract a reference "
                "temperature, or add a difference to one"
            )
        system = quantity._system
        scale = coherence(system, source) / coherence(system, target)
        raw = _rescaled(quantity._value, scale)
        display = quantity._display if source is target else None
        return cast("Quantity[K, W, T]", _wrap(target, raw, system, display=display))

    @classmethod
    def parse(cls, data: object, *, units: tuple[Unit[Any], ...] = ()) -> Self:
        return cast("Self", _parse(cls, data, units, Atomistic))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from quantype._internal._validation import pydantic_schema

        return pydantic_schema(cast("Any", cls), source_type, handler)

    def _check_unit(self, unit: Unit[K], hint: str = "") -> None:
        require_unit(unit, self._kind, hint)
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
        self._check_unit(
            unit,
            "`.to` sets the display unit; to move a tensor to another device, "
            "move its `.value` and wrap it again with `from_value`.",
        )
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

    def _scaled(self, raw: Any) -> Self:
        """Scaling keeps the display unit, unless it has an offset (°C).

        Twice 20 °C is 586.3 K, which a Celsius display would show as 313.15 °C.
        """
        display = self._display is None or not self._display.offset
        return self._with(raw, display=display)

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

    def _is_dimensionless(self) -> bool:
        semantic = self._semantic
        return (
            isinstance(semantic, Kind) and DIMENSIONLESS_KINDS.get(semantic) is semantic
        )

    def _plain(self, other: object) -> Quantity[Any, Any, Any] | None:
        """A plain number or array as this Dimensionless kind, if it is one."""
        operand = real_operand(other) if self._is_dimensionless() else None
        if operand is None:
            return None
        return _wrap(self._semantic, operand, self._system)

    def _add_sub(
        self, other: object, *, subtract: bool, reflected: bool = False
    ) -> Quantity[Any, Any, Any]:
        if not isinstance(other, Quantity):
            # Zero means the same in every unit, so builtin sum() works.
            if type(other) is int and other == 0 and not self._is_dimensionless():
                if not (reflected and subtract):
                    return self
                if isinstance(self._semantic, Kind) and self._semantic.affine:
                    raise TypeError("Cannot subtract an absolute Temperature from 0")
                return -self
            plain = self._plain(other)
            if plain is None:
                raise TypeError(
                    f"Cannot {'subtract' if subtract else 'add'} "
                    f"{type(other).__name__} and {self.kind}; give it a unit"
                )
            other = plain
        rhs = cast("Quantity[Any, Any, Any]", other)
        left, right = (rhs, self) if reflected else (self, rhs)
        left._same_system(right)
        kind = addition(left._semantic, right._semantic, subtract=subtract)
        lhs: Any = left._value
        raw = lhs - right._value if subtract else lhs + right._value
        return _wrap(kind, raw, self._system, display=_sum_display(kind, left, right))

    def __add__(self, other: object) -> Quantity[Any, Any, Any]:
        return self._add_sub(other, subtract=False)

    def __radd__(self, other: object) -> Quantity[Any, Any, Any]:
        return self._add_sub(other, subtract=False, reflected=True)

    def __sub__(self, other: object) -> Quantity[Any, Any, Any]:
        return self._add_sub(other, subtract=True)

    def __rsub__(self, other: object) -> Quantity[Any, Any, Any]:
        return self._add_sub(other, subtract=True, reflected=True)

    def __mul__(self, other: Any) -> Quantity[Any, Any, Any]:
        lhs: Any = self._value
        if isinstance(other, Quantity):
            rhs = cast("Quantity[Any, Any, Any]", other)
            self._same_system(rhs)
            kind = product("mul", self._semantic, rhs._semantic)
            scale = _product_scale(
                self._system, self._semantic, rhs._semantic, kind, "mul"
            )
            return _wrap(kind, _rescaled(lhs * rhs._value, scale), self._system)
        if isinstance(other, Constant):
            return NotImplemented  # the constant adopts this quantity's system
        return self._scaled(keep_storage(lhs, lhs * self._operand(other)))

    def __rmul__(self, other: Any) -> Quantity[Any, Any, Any]:
        return self * other

    def __truediv__(self, other: Any) -> Quantity[Any, Any, Any]:
        lhs: Any = self._value
        if isinstance(other, Quantity):
            rhs = cast("Quantity[Any, Any, Any]", other)
            self._same_system(rhs)
            kind = product("div", self._semantic, rhs._semantic)
            scale = _product_scale(
                self._system, self._semantic, rhs._semantic, kind, "div"
            )
            return _wrap(kind, _rescaled(lhs / rhs._value, scale), self._system)
        if isinstance(other, Constant):
            return NotImplemented  # the constant adopts this quantity's system
        return self._scaled(keep_storage(lhs, lhs / self._operand(other)))

    def __rtruediv__(self, other: Any) -> Quantity[Any, Any, Any]:
        operand = self._operand(other)
        numerator = dimensionless_kind(self._semantic)
        kind = product("div", numerator, self._semantic)
        scale = _product_scale(self._system, numerator, self._semantic, kind, "div")
        raw: Any = self._value
        quotient = keep_storage(raw, operand / raw)
        return _wrap(kind, _rescaled(quotient, scale), self._system)

    def _operand(self, value: object) -> Any:
        operand = real_operand(value)
        if operand is None:
            raise TypeError(
                "Quantities scale by real numbers or numerical arrays; "
                f"received {type(value).__name__}"
            )
        return operand

    def __pow__(self, exponent: object) -> Quantity[Any, Any, Any]:
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise TypeError("Only integer quantity powers are supported")
        kind = power(self._semantic, exponent)
        raw: Any = self._value
        base = coherence(self._system, self._semantic)
        scale = base**exponent / coherence(self._system, kind)
        return _wrap(kind, _rescaled(raw**exponent, scale), self._system)

    def __neg__(self) -> Self:
        raw: Any = self._value
        return self._scaled(-raw)

    def __pos__(self) -> Self:
        return self

    def __abs__(self) -> Self:
        raw: Any = self._value
        return self._scaled(abs(raw))

    def _compare(self, other: object, compare: Callable[[Any, Any], Any]) -> Any:
        if not isinstance(other, Quantity):
            plain = self._plain(other)
            if plain is None:
                return NotImplemented
            other = plain
        rhs = cast("Quantity[Any, Any, Any]", other)
        self._same_system(rhs)
        if rhs._semantic != self._semantic:
            raise TypeError(f"Cannot compare {self.kind} and {rhs.kind}")
        return compare(self._value, rhs._value)

    @override
    def __eq__(self, other: object) -> Any:
        """Value equality within a kind and system; otherwise simply unequal.

        As with naive and aware datetimes, quantities that cannot be compared
        are unequal, while ordering them raises.
        """
        if not isinstance(other, Quantity):
            plain = self._plain(other)
            return (
                NotImplemented if plain is None else self._compare(plain, operator.eq)
            )
        rhs = cast("Quantity[Any, Any, Any]", other)
        if rhs._semantic != self._semantic or rhs._system is not self._system:
            return False
        return self._compare(rhs, operator.eq)

    @override
    def __ne__(self, other: object) -> Any:
        if not isinstance(other, Quantity):
            plain = self._plain(other)
            return (
                NotImplemented if plain is None else self._compare(plain, operator.ne)
            )
        rhs = cast("Quantity[Any, Any, Any]", other)
        if rhs._semantic != self._semantic or rhs._system is not self._system:
            return True
        return self._compare(rhs, operator.ne)

    def __bool__(self) -> bool:
        raise TypeError(
            f"The truth value of a {self.kind} quantity is ambiguous; compare it "
            "explicitly (q > 0 * u.nm), or test `q is not None`"
        )

    def __float__(self) -> float:
        raw: Any = self._value
        if not self._is_dimensionless() or getattr(raw, "ndim", 0) != 0:
            raise TypeError(
                f"Only a scalar Dimensionless quantity converts to float, not "
                f"{self.kind}; use .value or .magnitude(unit)"
            )
        return float(raw)

    @override
    def __hash__(self) -> int:
        """Scalars hash by value, consistently with ``==``; arrays are unhashable.

        Equal quantities share a kind, a system and stored value, so the display
        unit and exact echo are deliberately excluded: ``1 nm`` and ``10 Å`` agree.
        """
        raw: Any = self._value
        if not isinstance(raw, (int, float)):
            if not type(raw).__module__.startswith("numpy"):
                raise TypeError(f"unhashable array storage: {type(raw).__name__}")
            import numpy as np

            if not isinstance(raw, np.generic):
                raise TypeError(f"unhashable array storage: {type(raw).__name__}")
        return hash((self._semantic, self._system, cast("object", raw)))

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
        """``np.cos(angle)`` and friends apply quantype's unit rules.

        Supported ufuncs are those of ``quantype.numpy``, plus arithmetic and
        comparisons. Results keep physical meaning, though NumPy's static types
        cannot say so; ``quantype.numpy`` is the typed spelling.
        """
        from quantype._internal._numpy import REFLECTED, UFUNCS

        name = str(ufunc.__name__)
        function = UFUNCS.get(name)
        if method != "__call__" or kwargs or function is None:
            raise TypeError(
                f"np.{name}{'' if method == '__call__' else '.' + method} does not "
                "know the units of a quantity; use quantype.numpy, quantity "
                "methods, or .value and .magnitude(unit) explicitly"
            )
        # An array on the left (array * q) arrives here from ndarray.__mul__, so
        # hand it to the quantity's reflected operator instead of recursing.
        binary = len(inputs) == 2  # noqa: PLR2004 -- binary ufunc
        if binary and inputs[1] is self and not isinstance(inputs[0], Quantity):
            reflected = REFLECTED.get(name)
            result = (
                NotImplemented
                if reflected is None
                else getattr(self, reflected)(inputs[0])
            )
            if result is NotImplemented:
                raise TypeError(f"np.{name} of a plain array and {self.kind}")
            return result
        return function(*inputs)

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        """``np.sum(q)``, ``np.stack([...])`` and friends apply unit rules."""
        from quantype._internal._numpy import FUNCTIONS, LINALG

        name = str(func.__name__)
        table = LINALG if str(func.__module__).endswith("linalg") else FUNCTIONS
        function = table.get(name)
        if function is None:
            raise TypeError(
                f"np.{name} does not know the units of a quantity; use "
                "quantype.numpy, quantity methods, or .value and .magnitude(unit)"
            )
        return function(*args, **kwargs)

    @classmethod
    def __torch_function__(
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        """``torch.cos(angle)`` and friends apply unit rules, as NumPy's do."""
        from quantype._internal._numpy import torch_function

        return torch_function(func, args, kwargs or {})

    @property
    def dtype(self) -> Any:
        """The storage's dtype; None for Python scalars."""
        return getattr(self._value, "dtype", None)

    def reshape(self, *shape: Any) -> Self:
        from quantype._internal._numpy import reshape

        target = shape[0] if len(shape) == 1 else shape
        return cast("Self", reshape(self, target))

    def transpose(self, *axes: int) -> Self:
        from quantype._internal._numpy import transpose

        return cast("Self", transpose(self, axes or None))

    @property
    def T(self) -> Self:  # noqa: N802 -- NumPy's name
        return self.transpose()

    def squeeze(self, axis: int | None = None) -> Self:
        from quantype._internal._numpy import squeeze

        return cast("Self", squeeze(self, axis))

    def cumsum(self, axis: int | None = None) -> Self:
        from quantype._internal._numpy import cumsum

        return cast("Self", cumsum(self, axis))

    def std(
        self, axis: int | None = None, *, ddof: int = 0, keepdims: bool = False
    ) -> Quantity[Any, Any, Any]:
        """Standard deviation; of absolute temperatures, a temperature difference."""
        from quantype._internal._numpy import std

        return cast(
            "Quantity[Any, Any, Any]", std(self, axis, ddof=ddof, keepdims=keepdims)
        )

    def _symbol(self) -> str:
        unit = self.unit
        symbol = (
            unit.symbol if unit is not None else _symbol(self._semantic, self._system)
        )
        return "" if symbol == "1" else f" {symbol}"

    def _presentation(self) -> str:
        return f"{self.magnitude()!s}{self._symbol()}"

    @override
    def __format__(self, spec: str) -> str:
        """Apply a format spec to the magnitude: f"{q:.3f}" gives "2.000 nm"."""
        if not spec:
            return str(self)
        magnitude: Any = self.magnitude()
        try:
            text = format(magnitude, spec)
        except (TypeError, ValueError):
            import numpy as np

            def element(value: object) -> str:
                return format(value, spec)

            text = np.array2string(
                np.asarray(magnitude),
                formatter={"float_kind": element, "int_kind": element},
            )
        return f"{text}{self._symbol()}"

    @override
    def __str__(self) -> str:
        return self._presentation()

    @override
    def __repr__(self) -> str:
        system = "" if self._system is Atomistic else f", {self._system.__name__}"
        return f"{type(self).__name__}({self._presentation()}{system})"


class Constant[K]:
    """A physical constant: exact in every unit system, so it has none of its own.

    Arithmetic with a quantity adopts that quantity's system, so ``k_B * T`` is
    an energy in ``T``'s system. ``to_system``, ``magnitude`` and ``to`` show
    its value in a particular system or unit.
    """

    # NumPy scalars defer to the constant's reflected operators.
    __array_ufunc__: ClassVar[None] = None
    __slots__ = ("_reference", "_semantic", "name")
    #: Generated per-kind subclasses name their kind, so results keep their class.
    #: Application catalogues give the kind object itself, as their quantities do.
    _constant_kind: ClassVar[str] = ""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        semantic = cls.__dict__.get("_constant_semantic")
        if semantic is None and cls.__dict__.get("_constant_kind"):
            semantic = KINDS[cls._constant_kind]
        if semantic is not None:
            _CONSTANT_CLASSES[semantic] = cls

    def __init__(self, name: str, semantic: Semantic, reference: float) -> None:
        #: The conventional symbol, such as ``k_B``.
        self.name = name
        self._semantic = semantic
        # Coherent value in the reference (default-system) units.
        self._reference = reference

    @property
    def kind(self) -> str:
        return str(self._semantic)

    @property
    def dimensions(self) -> tuple[int, ...]:
        return self._semantic.dimensions

    def to_system(self, system: type[UnitSystem]) -> Quantity[K, float, Any]:
        """The constant's value in ``system``, as a quantity of that system."""
        target = require_system(system)
        raw = self._reference / factor(target, self._semantic)
        return cast("Quantity[K, float, Any]", _wrap(self._semantic, raw, target))

    def magnitude(self, unit: Unit[K]) -> float:
        """The constant's value in a compatible unit."""
        return self.to_system(Atomistic).magnitude(unit)

    def to(self, unit: Unit[K]) -> Quantity[K, float, Any]:
        """The constant as a default-system quantity presented in ``unit``."""
        return self.to_system(Atomistic).to(unit)

    def _combine(self, other: object, op: str, *, reflected: bool = False) -> Any:
        if isinstance(other, Quantity):
            quantity = cast("Quantity[Any, Any, Any]", other)
            own = self.to_system(quantity._system)
            left, right = (quantity, own) if reflected else (own, quantity)
            return left * right if op == "mul" else left / right
        if isinstance(other, Constant):
            constant = cast("Constant[Any]", other)
            left, right = (constant, self) if reflected else (self, constant)
            kind = product(cast("Any", op), left._semantic, right._semantic)
            value = (
                left._reference * right._reference
                if op == "mul"
                else left._reference / right._reference
            )
            symbol = " " if op == "mul" else "/"
            return _constant(f"{left.name}{symbol}{right.name}", kind, value)
        if not isinstance(other, numbers.Real) or isinstance(
            other, (bool, _numpy_bool())
        ):
            return NotImplemented
        other = float(other)
        if op == "mul":
            return _constant(
                f"{other} {self.name}", self._semantic, self._reference * other
            )
        if not reflected:
            return _constant(
                f"{self.name}/{other}", self._semantic, self._reference / other
            )
        kind = product("div", dimensionless_kind(self._semantic), self._semantic)
        return _constant(f"{other}/{self.name}", kind, other / self._reference)

    def __mul__(self, other: object) -> Any:
        return self._combine(other, "mul")

    def __rmul__(self, other: object) -> Any:
        return self._combine(other, "mul", reflected=True)

    def __truediv__(self, other: object) -> Any:
        return self._combine(other, "div")

    def __rtruediv__(self, other: object) -> Any:
        return self._combine(other, "div", reflected=True)

    def __pow__(self, exponent: object) -> Constant[Any]:
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise TypeError("Only integer constant powers are supported")
        kind = power(self._semantic, exponent)
        return _constant(f"{self.name}^{exponent}", kind, self._reference**exponent)

    def __neg__(self) -> Self:
        return type(self)(f"-{self.name}", self._semantic, -self._reference)

    @override
    def __str__(self) -> str:
        return str(self.to_system(Atomistic))

    @override
    def __repr__(self) -> str:
        return f"Constant({self.name} = {self})"


_CONSTANT_CLASSES: dict[Semantic, type[Constant[Any]]] = {}


def _constant(name: str, semantic: Semantic, reference: float) -> Constant[Any]:
    """A constant of its kind's generated class, or the structural base class."""
    return _CONSTANT_CLASSES.get(semantic, Constant)(name, semantic, reference)


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

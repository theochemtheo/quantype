"""Backend-neutral runtime. The adjacent stub is the static public contract.

Numerical operations intentionally use the stored object's own operators. Dynamic
typing is confined to this dispatch boundary, not exposed as a user's result type.

Each quantity holds raw numbers, its physical kind, its unit system, and
optionally the unit it was given in (display). In a unit system the numbers
are in the system's units; without one (``NoSystem``) they are in a storage
unit: the unit they were given in, or one composed from the operands' units.
"""

# Runtime metadata is deliberately shared by the package's boundary modules.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import numbers
import operator
import sys
from functools import cache
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NoReturn,
    Self,
    cast,
    override,
)

from quantype._internal._products import class_for
from quantype._internal._semantics import (
    DIMENSIONLESS_KINDS,
    KINDS,
    Expression,
    Kind,
    Semantic,
    addition,
    dimensionless_kind,
    kind_mismatch,
    power,
    product,
)
from quantype._internal._storage import (
    backend_name,
    is_floating,
    keep_storage,
    real_operand,
)
from quantype._internal._storage_units import (
    ComposedUnit,
    adapted,
    compose,
    raised,
    recast,
    reference_unit,
    simplest,
    system_unit,
)
from quantype._internal._systems import (
    AnySystem,
    Atomistic,
    NoSystem,
    UnitSystem,
    coherence,
    factor,
    into_system,
    overridden_kinds,
    require_any_system,
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


def _numpy_bool() -> type:
    """NumPy's boolean scalar type when NumPy is loaded; it is not a Python bool."""
    numpy = sys.modules.get("numpy")
    return bool if numpy is None else cast("type", numpy.bool_)


def _rescaled(raw: Any, scale: float) -> Any:
    """Apply a coherence factor; the usual factor of 1 leaves storage untouched."""
    return raw if scale == 1.0 else raw * scale


def _between(raw: Any, source: float, target: float) -> Any:
    """Numbers in a unit of scale ``source`` in one of scale ``target``.

    As in ``rescale``, the factor is a ratio of at least 1, so exact ratios
    such as 10 stay exact both ways.
    """
    if source == target:
        return raw
    if source > target:
        return raw * (source / target)
    return raw / (target / source)


# Array and tensor methods that keep the backend, which apply runs, and ones
# that read or leave the numbers, which need a unit first.
_STORAGE_METHODS = frozenset(
    {"detach", "cpu", "cuda", "clone", "contiguous", "requires_grad_", "astype"}
)
_READ_METHODS = frozenset({"numpy", "tolist", "backward", "block_until_ready"})


def _check_raw(value: Any, context: str) -> None:
    """Raw numbers a quantity can keep as they are: real, and floating if arrays."""
    if real_operand(value) is None:
        raise TypeError(
            f"{context} real numbers or a numerical array; "
            f"received {type(value).__name__}"
        )
    if not isinstance(value, (int, float)) and not is_floating(value):
        raise TypeError(
            f"{context} a floating dtype; received {value.dtype}. "
            "Convert first, as in raw.astype(float)"
        )


def _either(left: Any, right: Any) -> Any:
    """``or`` of Python booleans, element-wise for boolean arrays."""
    return (left or right) if isinstance(left, bool) else left | right


def _both(left: Any, right: Any) -> Any:
    """``and`` of Python booleans, element-wise for boolean arrays."""
    return (left and right) if isinstance(left, bool) else left & right


def _negated(value: Any) -> Any:
    """``not`` of a Python boolean, element-wise for boolean arrays."""
    return (not value) if isinstance(value, bool) else ~value


def _product_scale(
    system: type[AnySystem], left: Semantic, right: Semantic, result: Semantic, op: str
) -> float:
    """The factor taking raw operands' product or ratio to the result's storage."""
    overridden = overridden_kinds(system)
    if left not in overridden and right not in overridden and result not in overridden:
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
    example = "u.nm" if semantic is None else f"u.{Atomistic.unit_for(semantic).name}"
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
    elif type(unit).__module__.endswith(".units") and type(unit).__name__.endswith(
        "Namespace"
    ):
        message = (
            f"Expected a unit, such as {example}; received a namespace of units, "
            "which holds them"
        )
    else:
        message = f"Expected a unit, such as {example}; received {type(unit).__name__}."
    raise TypeError(f"{message} {hint}".rstrip())


_CLASSES: dict[Kind, type[Quantity[Any, Any, Any]]] = {}
_STRUCTURAL_CLASSES: dict[Kind, type[Quantity[Any, Any, Any]]] = {}


def _wrap(  # noqa: PLR0913 -- one keyword argument per optional field
    kind: str | Semantic,
    value: Any,
    system: type[AnySystem] | None,
    *,
    display: Unit[Any] | None = None,
    echo: float | None = None,
    storage: Unit[Any] | None = None,
) -> Quantity[Any, Any, Any]:
    """Wrap trusted raw numbers; ``system=None`` means ``NoSystem``.

    Without a system, ``storage`` is the unit the numbers are in; the default
    is the kind's reference unit (Å, eV, fs, and their products).
    """
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
    if system is None or system is NoSystem:
        result._system = NoSystem
        result._storage = reference_unit(semantic) if storage is None else storage
    else:
        result._system = system
        result._storage = None
    result._display = display
    result._echo = echo
    return cast("Quantity[Any, Any, Any]", result)


def _composite(
    kind: Semantic, raw: Any, storage: Unit[Any], display: Unit[Any]
) -> Quantity[Any, Any, Any]:
    """A product, quotient or power without a unit system.

    The numbers stay in the composed ``storage`` unit, and show in ``display``,
    composed the same way from the operands' displayed units. A dimensionless
    result holds plain numbers, so ns/fs is 1e6 and composes with nothing.
    """
    stored, shown = _result_units(kind, storage, display)
    if stored is not storage:
        raw = keep_storage(raw, _between(raw, storage.scale, stored.scale))
    return _wrap(kind, raw, NoSystem, display=shown, storage=stored)


def _product(
    operation: Literal["mul", "div"],
    kind: Semantic,
    raw: Any,
    left: Quantity[Any, Any, Any],
    right: Quantity[Any, Any, Any],
) -> Quantity[Any, Any, Any]:
    """``_composite`` for a product or quotient, in one lookup: the hot path."""
    stored, display, source, target = _product_units(
        operation,
        kind,
        cast("Unit[Any]", left._storage),
        cast("Unit[Any]", right._storage),
        cast("Unit[Any]", left._display or left.unit),
        cast("Unit[Any]", right._display or right.unit),
    )
    if source != target:
        raw = keep_storage(raw, _between(raw, source, target))
    return _wrap(kind, raw, NoSystem, display=display, storage=stored)


@cache
def _product_units(  # noqa: PLR0913, PLR0917 -- the cache key
    operation: Literal["mul", "div"],
    kind: Semantic,
    left: Unit[Any],
    right: Unit[Any],
    left_shown: Unit[Any],
    right_shown: Unit[Any],
) -> tuple[Unit[Any], Unit[Any], float, float]:
    """A product's storage and display units, and the factor ``source / target``
    taking the product of its operands' numbers into the storage unit."""
    composed, source, target = compose(operation, kind, left, right)
    shown, _, _ = compose(operation, kind, left_shown, right_shown)
    stored, display = _result_units(kind, composed, shown)
    if stored is not composed:
        source, target = source * composed.scale, target * stored.scale
    return stored, display, source, target


@cache
def _result_units(
    kind: Semantic, storage: Unit[Any], display: Unit[Any]
) -> tuple[Unit[Any], Unit[Any]]:
    """A result's storage and display units: a dimensionless one is plain."""
    if isinstance(kind, Kind) and DIMENSIONLESS_KINDS.get(kind) is kind:
        plain = reference_unit(kind)
        return plain, plain
    return storage, display


def _is_scalar(raw: object) -> bool:
    """Python numbers and NumPy scalars, as opposed to arrays and tensors."""
    if isinstance(raw, (int, float)):
        return True
    if not type(raw).__module__.startswith("numpy"):
        return False
    import numpy as np

    return isinstance(raw, np.generic)


def _delta_unit(point: Unit[Any], kind: Kind) -> Unit[Any] | None:
    """The temperature-difference unit matching a point unit (°C → Δ°C)."""
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
    if left._display is None:
        return None
    # Only a point minus a point gets here; its difference kind is the result.
    return _delta_unit(left._display, cast("Kind", kind))


class _KindName:
    """``kind`` on a named quantity class as on its instances: ``Force.kind``."""

    def __get__(self, instance: object, owner: type | None = None) -> str:
        semantic = getattr(owner if instance is None else instance, "_semantic", None)
        if semantic is None:
            raise AttributeError("Only named quantity classes have a kind")
        return str(semantic)

    def __set__(self, instance: object, value: object) -> None:
        raise AttributeError("A quantity's kind is fixed")


class _Value:
    """``q.value``: raw numbers, which only a quantity in a unit system has."""

    def __get__(self, obj: object, owner: object, /) -> Any:
        if obj is None:
            return self
        quantity = cast("Quantity[Any, Any, Any]", obj)
        if quantity._system is NoSystem:
            raise TypeError(
                f"{quantity.kind} has no raw numbers without a unit system; "
                "use .magnitude(unit), or choose a system with .to_system(...)"
            )
        return quantity._value


class Quantity[K, V, S: AnySystem]:
    """A semantic quantity: numbers, a kind, a unit system, and a display unit."""

    _kind: str = ""
    kind: ClassVar[str] = cast("str", _KindName())
    _semantic: Semantic
    _system: type[AnySystem]
    # The unit the numbers are in, without a system; None in a unit system.
    _storage: Unit[Any] | None
    _display: Unit[K] | None
    _echo: float | None
    __array_priority__: ClassVar[int] = 10000
    #: Raw numbers in the unit system's units; explicitly bypasses units. Only
    #: quantities in a unit system have them.
    value = cast("V", _Value())

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "_catalogue_dimensionless" in cls.__dict__:
            _STRUCTURAL_CLASSES[cls.__dict__["_catalogue_dimensionless"]] = cls
        if cls.__dict__.get("_kind"):
            if "_semantic" not in cls.__dict__:
                cls._semantic = KINDS[cls._kind]
            _CLASSES[cast("Kind", cls._semantic)] = cls

    @classmethod
    def __class_getitem__(cls, parameters: Any) -> GenericAlias:
        from quantype._internal._construction import StorageAlias

        return StorageAlias(cls, parameters)

    def __init__(
        self,
        value: object,
        unit: Unit[K],
        *,
        dtype: object = None,
        display: bool = True,
    ) -> None:
        """Store ``value`` in ``unit`` and show it there; no unit system.

        Units with an offset (°C) store kelvin. With ``display=False`` it shows
        in the storage unit.
        """
        if dtype is not None:
            raise TypeError("dtype requires a parameterized quantity storage type")
        if not self._kind:
            raise TypeError("Construct a named quantity or use a unit")
        self._check_unit(unit)
        built = cast("Any", unit)(value)
        self._value = cast("V", built._value)
        self._system = NoSystem
        self._storage = built._storage
        self._display = unit if display else None
        self._echo = built._echo if display else None

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
            raise ValueError(kind_mismatch(cls._semantic, reference.semantic))
        return Unit(
            name,
            reference.semantic,
            reference.scale * scale,
            reference.scale * offset + reference.offset,
            symbol,
        )

    @classmethod
    def unit_named(cls, name: str, *, units: tuple[Unit[Any], ...] = ()) -> Unit[K]:
        """The unit of this kind that ``name`` names, as ``parse`` reads it.

        A catalogue name or alias comes first, then a display symbol such as
        ``"Ha/a0"``. ``units`` adds definitions the catalogue doesn't have.
        """
        return cast("Unit[K]", _unit_named(cls, name, units, NoSystem))

    @property
    def dimensions(self) -> tuple[int, ...]:
        return self._semantic.dimensions

    @property
    def system(self) -> type[AnySystem]:
        return self._system

    @property
    def unit(self) -> Unit[K] | None:
        """The unit ``magnitude()`` and ``repr`` use.

        None for unnamed products in a unit system, which show the system's
        units composed.
        """
        if self._display is not None:
            return self._display
        if self._storage is not None:
            return cast("Unit[K]", self._storage)
        if isinstance(self._semantic, Kind):
            system = cast("type[UnitSystem]", self._system)
            return cast("Unit[K]", system.unit_for(self._semantic))
        return None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple[int, ...](getattr(self._value, "shape", ()))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def from_value[W, T: UnitSystem](
        cls, value: W, system: type[T]
    ) -> Quantity[K, W, T]:
        """Trusted wrap of raw numbers in ``system``'s units.

        ``Length.from_value(raw, SI)`` takes ``raw`` as meters. Nothing checks
        that, and the storage is kept. Without a system there are no raw
        numbers, so construct from a unit instead.
        """
        if getattr(cls, "_semantic", None) is None:
            raise TypeError("from_value requires a named quantity class")
        if isinstance(value, Quantity):
            raise TypeError(
                "from_value takes raw numbers, such as q.value, not a quantity"
            )
        _check_raw(value, "from_value needs")
        # The bound rejects NoSystem statically; untyped callers reach here.
        if cast("object", system) is NoSystem:
            raise TypeError(
                "from_value trusts raw numbers in a unit system, and NoSystem has "
                f"none; construct from a unit, as in {cls._kind}(raw, unit)"
            )
        return cast(
            "Quantity[K, W, T]", _wrap(cls._semantic, value, require_system(system))
        )

    @classmethod
    def reinterpret[W, T: AnySystem](
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
        storage = quantity._storage
        if storage is not None:
            storage = recast(target, storage)
        return cast(
            "Quantity[K, W, T]",
            _wrap(target, raw, system, display=display, storage=storage),
        )

    @classmethod
    def parse(cls, data: object, *, units: tuple[Unit[Any], ...] = ()) -> Self:
        return cast("Self", _parse(cls, data, units, NoSystem))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from quantype._internal._validation import pydantic_schema

        return pydantic_schema(cast("Any", cls), source_type, handler)

    def _check_unit(self, unit: Unit[K], hint: str = "") -> None:
        require_unit(unit, self._kind, hint)
        # Named kinds are nominal, so compared by identity; products of them,
        # as in a composed storage unit, by structure.
        semantic = unit.semantic
        if semantic is not self._semantic and not (
            isinstance(semantic, Expression) and semantic == self._semantic
        ):
            raise ValueError(
                f"{kind_mismatch(self._semantic, semantic)} (unit {unit.name!r})"
            )

    def magnitude(self, unit: Unit[K] | None = None) -> V:
        """Magnitude in ``unit``, else in ``self.unit``.

        Without ``unit``, an unnamed product in a unit system gives its raw
        numbers, in the system's units composed.
        """
        if self._echo is not None and (unit is None or unit == self._display):
            return cast("V", self._echo)
        selected: Unit[Any] | None = unit or self.unit
        if selected is None:
            return self._value
        self._check_unit(selected)
        from quantype._internal._storage import rescale, unit_conversion

        storage = self._storage
        if storage is None:
            system = cast("type[UnitSystem]", self._system)
            if not selected.offset:
                source = factor(system, self._semantic)
                return cast("V", rescale(self._value, source, selected.scale))
            scale, offset = into_system(selected, system)
        elif selected is storage:
            return self._value
        elif not selected.offset:
            return cast("V", rescale(self._value, storage.scale, selected.scale))
        else:
            scale, offset = (
                selected.scale / storage.scale,
                selected.offset / storage.scale,
            )
        return cast("V", unit_conversion(self._value, scale, offset, inverse=True))

    def to(self, unit: Unit[K] | type[UnitSystem]) -> Self:
        """Present in ``unit``, or in a system's unit; the numbers are unchanged.

        ``q.to(Atomistic)`` shows the kind's unit in that system without
        choosing it as ``q``'s system.
        """
        if isinstance(unit, type):
            shown = system_unit(require_system(unit), self._semantic)
        else:
            self._check_unit(
                unit,
                "`.to` sets the display unit; to move a tensor to another device "
                "or dtype, use `q.apply(lambda t: t.to(...))`.",
            )
            shown = unit
        return self._shown(shown)

    def simplified(self) -> Self:
        """Present in the simplest unit; the numbers are unchanged.

        Units of one dimension merge, so kcal/mol·nm/Å shows as kcal/mol. A
        named kind then shows in its unit in a display system that holds
        every remaining unit, so Da·Å^2/fs^2 shows as eV.
        """
        unit = self.unit
        if unit is None:
            return self
        return self._shown(simplest(self._semantic, unit))

    def _shown(self, display: Unit[Any]) -> Self:
        return cast(
            "Self",
            _wrap(
                self._semantic,
                self._value,
                self._system,
                display=display,
                storage=self._storage,
            ),
        )

    def to_system(self, system: type[AnySystem]) -> Quantity[K, V, Any]:
        """Rescale the numbers into another unit system; the display is kept.

        Into ``NoSystem``, nothing converts: the numbers stay in the units of
        the system they came from. An unnamed product's display is a system's
        composed units, so it gives way to the target's.
        """
        target = require_any_system(system)
        raw: Any = self._value
        storage = self._storage
        display = self._display
        structural = not isinstance(self._semantic, Kind)
        if (
            structural
            and isinstance(display, ComposedUnit)
            and target is not self._system
        ):
            display = None
        if target is NoSystem:
            if storage is None:
                source = cast("type[UnitSystem]", self._system)
                storage = system_unit(source, self._semantic)
        elif target is not self._system:
            from quantype._internal._storage import rescale, warn_if_out_of_range

            # Temperatures share kelvin-scaled bases, so there is never an offset.
            source_scale = (
                factor(cast("type[UnitSystem]", self._system), self._semantic)
                if storage is None
                else storage.scale
            )
            target_scale = factor(cast("type[UnitSystem]", target), self._semantic)
            storage = None
            converted = rescale(raw, source_scale, target_scale)
            if type(raw).__module__.startswith("numpy"):
                import numpy as np

                exact = np.asarray(raw, dtype=np.float64) * (
                    source_scale / target_scale
                )
                warn_if_out_of_range(exact, converted)
            raw = converted
        return cast(
            "Quantity[K, V, Any]",
            _wrap(
                self._semantic,
                raw,
                target,
                display=display,
                echo=self._echo,
                storage=storage,
            ),
        )

    def to_dict(self, unit: Unit[K] | None = None) -> dict[str, object]:
        from quantype.serialization import to_dict

        return to_dict(cast("Any", self), cast("Any", unit))

    def _adopted(
        self, other: Quantity[Any, Any, Any]
    ) -> tuple[Quantity[Any, Any, Any], Quantity[Any, Any, Any]]:
        """Both operands in one system: one without a system adopts the other's."""
        mine, theirs = self._system, other._system
        if mine is theirs:
            return self, other
        if mine is NoSystem:
            return self.to_system(theirs), other
        if theirs is NoSystem:
            return self, other.to_system(mine)
        raise TypeError(
            f"Cannot combine {mine.__name__} and {theirs.__name__} quantities; "
            "convert one explicitly with .to_system(...)"
        )

    def apply(self, function: Callable[[V], V], /) -> Self:
        """Apply ``function`` to the stored numbers, keeping kind, units and system.

        ``x.apply(np.sort)`` or ``x.apply(lambda t: t.detach())`` applies an
        operation quantype has no rule for. The function must give the same
        physical result whatever unit the numbers are in: sorting, slicing,
        reshaping, detaching, or moving to a device, but not ``np.log`` or
        adding a constant. Nothing checks that. It returns numbers of the same
        backend; to change backend, construct from ``.magnitude(unit)``.
        """
        result: Any = function(self._value)
        if isinstance(result, Quantity):
            raise TypeError("apply's function returns numbers, not a quantity")
        _check_raw(result, "apply's function must return")
        before, after = backend_name(self._value), backend_name(result)
        # A NumPy float64 is a float, as checkers also accept.
        if before != after and not isinstance(result, type(self._value)):
            raise TypeError(
                f"apply's function must keep {before} storage; received {after}. "
                "To change backend, construct from .magnitude(unit)"
            )
        return self._with(result)

    def _scaled(self, raw: Any) -> Self:
        """Scaling keeps the display unit, unless it has an offset (°C).

        Twice 20 °C is 586.3 K, which a Celsius display would show as 313.15 °C.
        """
        display = self._display is None or not self._display.offset
        return self._with(raw, display=display)

    def _with(self, raw: Any, *, display: bool = True) -> Self:
        """Same kind, system and storage unit; keeps the display unit by default."""
        return cast(
            "Self",
            _wrap(
                self._semantic,
                raw,
                self._system,
                display=self._display if display else None,
                storage=self._storage,
            ),
        )

    def _aligned(self, other: Quantity[Any, Any, Any]) -> Any:
        """``other``'s numbers in this quantity's storage unit, or system's units."""
        if self._storage is None or other._storage is None:
            return other._value
        return _between(other._value, other._storage.scale, self._storage.scale)

    def _reference(self) -> Any:
        """The numbers in the reference units."""
        if self._storage is None:
            return self._value
        return _between(self._value, self._storage.scale, 1.0)

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
        if left._system is not right._system:
            left, right = left._adopted(right)
        kind = addition(left._semantic, right._semantic, subtract=subtract)
        lhs: Any = left._value
        aligned: Any = left._aligned(right)
        raw = lhs - aligned if subtract else lhs + aligned
        storage = left._storage
        if storage is not None:
            storage = recast(kind, storage)
        display = _sum_display(kind, left, right)
        return _wrap(kind, raw, left._system, display=display, storage=storage)

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
            left, rhs = self, cast("Quantity[Any, Any, Any]", other)
            if left._system is not rhs._system:
                left, rhs = left._adopted(rhs)
                lhs = left._value
            kind = product("mul", left._semantic, rhs._semantic)
            if left._storage is not None and rhs._storage is not None:
                return _product("mul", kind, lhs * rhs._value, left, rhs)
            scale = _product_scale(
                left._system, left._semantic, rhs._semantic, kind, "mul"
            )
            return _wrap(kind, _rescaled(lhs * rhs._value, scale), left._system)
        if isinstance(other, Constant):
            return NotImplemented  # the constant adopts this quantity's system
        return self._scaled(keep_storage(lhs, lhs * self._operand(other)))

    def __rmul__(self, other: Any) -> Quantity[Any, Any, Any]:
        return self * other

    def __truediv__(self, other: Any) -> Quantity[Any, Any, Any]:
        lhs: Any = self._value
        if isinstance(other, Quantity):
            left, rhs = self, cast("Quantity[Any, Any, Any]", other)
            if left._system is not rhs._system:
                left, rhs = left._adopted(rhs)
                lhs = left._value
            kind = product("div", left._semantic, rhs._semantic)
            if left._storage is not None and rhs._storage is not None:
                return _product("div", kind, lhs / rhs._value, left, rhs)
            scale = _product_scale(
                left._system, left._semantic, rhs._semantic, kind, "div"
            )
            return _wrap(kind, _rescaled(lhs / rhs._value, scale), left._system)
        if isinstance(other, Constant):
            return NotImplemented  # the constant adopts this quantity's system
        return self._scaled(keep_storage(lhs, lhs / self._operand(other)))

    def __rtruediv__(self, other: Any) -> Quantity[Any, Any, Any]:
        operand = self._operand(other)
        numerator = dimensionless_kind(self._semantic)
        kind = product("div", numerator, self._semantic)
        raw: Any = self._value
        quotient = keep_storage(raw, operand / raw)
        if self._storage is not None:
            one = reference_unit(numerator)
            storage, _, _ = compose("div", kind, one, self._storage)
            display, _, _ = compose("div", kind, one, cast("Unit[Any]", self.unit))
            return _composite(kind, quotient, storage, display)
        scale = _product_scale(self._system, numerator, self._semantic, kind, "div")
        return _wrap(kind, _rescaled(quotient, scale), self._system)

    def __matmul__(self, other: Any) -> Quantity[Any, Any, Any]:
        from quantype._internal._numpy import matmul

        return cast("Quantity[Any, Any, Any]", matmul(self, other))

    def __rmatmul__(self, other: Any) -> Quantity[Any, Any, Any]:
        from quantype._internal._numpy import matmul

        return cast("Quantity[Any, Any, Any]", matmul(other, self))

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
        if self._storage is not None:
            storage = raised(kind, self._storage, exponent)
            display = raised(kind, cast("Unit[Any]", self.unit), exponent)
            return _composite(kind, raw**exponent, storage, display)
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

    def _number(
        self, other: object, *, zero: bool = False
    ) -> Quantity[Any, Any, Any] | None:
        """``other`` as a quantity to compare with, or None for a non-number.

        Dimensionless values compare with plain numbers. With ``zero``, every
        other kind but absolute temperatures equals zero or not, which means
        the same in every unit, as for ``sum()``. Other numbers raise:
        ``2 * u.nm == 2`` raises, while ``None`` and other objects are simply
        unequal.
        """
        plain = self._plain(other)
        affine = isinstance(self._semantic, Kind) and self._semantic.affine
        if plain is None and zero and type(other) is int and other == 0 and not affine:
            return _wrap(self._semantic, 0, self._system, storage=self._storage)
        if plain is None and real_operand(other) is not None:
            raise TypeError(
                f"Cannot compare {self.kind} and {type(other).__name__}; give it a unit"
            )
        return plain

    def _compare(self, other: object, compare: Callable[[Any, Any], Any]) -> Any:
        if not isinstance(other, Quantity):
            plain = self._number(other)
            if plain is None:
                return NotImplemented
            other = plain
        if other._semantic != self._semantic:
            raise TypeError(f"Cannot compare {self.kind} and {other.kind}")
        left, rhs = self._adopted(cast("Quantity[Any, Any, Any]", other))
        return left._ordered(rhs, compare)

    def _comparable(self, other: Quantity[Any, Any, Any]) -> tuple[Any, Any]:
        """Both sides' numbers in one unit: the larger, so only one side rounds."""
        mine, theirs = self._storage, other._storage
        if mine is None or theirs is None or mine.scale == theirs.scale:
            return self._value, other._value
        if mine.scale > theirs.scale:
            return self._value, _between(other._value, theirs.scale, mine.scale)
        return _between(self._value, mine.scale, theirs.scale), other._value

    def _ordered(
        self, other: Quantity[Any, Any, Any], compare: Callable[[Any, Any], Any]
    ) -> Any:
        """``compare`` of two quantities in one system, in either's unit.

        Quantities in two units are equal when either converts exactly onto the
        other, so a quantity equals itself converted into a system and back,
        whichever way that rounded. Rounding is monotonic, so the conversions
        never order a pair oppositely, and ordering agrees with equality.
        """
        mine, theirs = self._storage, other._storage
        if mine is None or theirs is None or mine.scale == theirs.scale:
            return compare(self._value, other._value)
        into_theirs = _between(self._value, mine.scale, theirs.scale)
        into_mine = _between(other._value, theirs.scale, mine.scale)
        equal = _either(into_theirs == other._value, self._value == into_mine)
        if compare is operator.eq:
            return equal
        if compare is operator.ne:
            return _negated(equal)
        ordered = compare(*self._comparable(other))
        if compare in {operator.lt, operator.gt}:
            return _both(ordered, _negated(equal))
        return _either(ordered, equal)

    def _equality(self, other: object, compare: Callable[[Any, Any], Any]) -> Any:
        if not isinstance(other, Quantity):
            plain = self._number(other, zero=True)
            return NotImplemented if plain is None else self._compare(plain, compare)
        rhs = cast("Quantity[Any, Any, Any]", other)
        if rhs._semantic != self._semantic:
            return compare is operator.ne
        systems = (self._system, rhs._system)
        if systems[0] is not systems[1] and NoSystem not in systems:
            return compare is operator.ne
        return self._compare(rhs, compare)

    @override
    def __eq__(self, other: object) -> Any:
        """Value equality within a kind, as arithmetic compares; else unequal.

        A quantity without a system adopts the other's, as for ``-`` and
        ``<``, so ``q == q.to_system(SI)``. Quantities in two unit systems are
        unequal, while ordering them raises. A plain number raises for both.
        Quantities in different units are equal when either converts exactly
        onto the other, so ``1 nm == 10 Å``.
        """
        return self._equality(other, operator.eq)

    @override
    def __ne__(self, other: object) -> Any:
        return self._equality(other, operator.ne)

    def __getattr__(self, name: str) -> NoReturn:
        """Only reached for missing attributes: name the quantity way to do it."""
        message = f"{type(self).__name__!r} object has no attribute {name!r}"
        if name in _STORAGE_METHODS:
            message += (
                "; apply it to the stored numbers, as in "
                f"q.apply(lambda raw: raw.{name}())"
            )
        elif name in _READ_METHODS:
            message += f"; take the numbers first, as in q.magnitude(unit).{name}()"
        raise AttributeError(message, name=name, obj=self)

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
                f"{self.kind}; use .magnitude(unit)"
            )
        return float(self._reference())

    # Equality adopts a system, as arithmetic does, so no hash of one side's
    # numbers could agree with it: a quantity without a system equals values
    # in several systems that differ from each other.
    __hash__ = None  # type: ignore[assignment]

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
        if _is_scalar(raw):
            raise TypeError("Scalar quantities cannot be indexed")
        selected = raw[key]
        if type(raw).__module__.startswith("numpy"):
            import numpy as np

            # Preserve ndarray storage when indexing selects a single element.
            selected = np.asarray(selected)
        return self._with(selected)

    def __len__(self) -> int:
        raw: Any = self._value
        if _is_scalar(raw):
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

    def var(
        self, axis: int | None = None, *, ddof: int = 0, keepdims: bool = False
    ) -> Quantity[Any, V, S]:
        """Variance: the square of the kind, as ``quantype.numpy.var`` gives."""
        from quantype._internal._numpy import var

        return cast(
            "Quantity[Any, V, S]", var(self, axis, ddof=ddof, keepdims=keepdims)
        )

    def copy(self) -> Self:
        """A copy whose numbers are independent of this quantity's."""
        raw: Any = self._value
        if type(raw).__module__.startswith("torch"):
            return self._with(raw.clone())
        copied = getattr(raw, "copy", None)
        return self._with(copied() if callable(copied) else raw)

    @property
    def size(self) -> int:
        """The number of elements: 1 for a scalar."""
        raw: Any = self._value
        if type(raw).__module__.startswith("torch"):
            return int(raw.numel())
        return int(getattr(raw, "size", 1))

    def ravel(self) -> Self:
        """The numbers as one dimension, keeping the kind."""
        return self.reshape(-1)

    def flatten(self) -> Self:
        """The numbers as one dimension, keeping the kind."""
        return self.reshape(-1)

    def item(self) -> Quantity[K, float, S]:
        """A one-element quantity as one with Python float storage."""
        raw: Any = self._value
        number = float(raw.item() if hasattr(raw, "item") else raw)
        return cast("Quantity[K, float, S]", self._with(number))

    def argmin(self, axis: int | None = None) -> Any:
        """The index of the smallest value, as a plain integer or array."""
        from quantype._internal._numpy import argmin

        return argmin(self, axis=axis)

    def argmax(self, axis: int | None = None) -> Any:
        """The index of the largest value, as a plain integer or array."""
        from quantype._internal._numpy import argmax

        return argmax(self, axis=axis)

    def argsort(self, axis: int = -1) -> Any:
        """The indices that sort the values, as a plain array."""
        from quantype._internal._numpy import argsort

        return argsort(self, axis=axis)

    def __array__(self, dtype: object = None, copy: object = None) -> Any:
        raise TypeError("Implicit array coercion drops units; use .magnitude(unit)")

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
                "methods, or .magnitude(unit) explicitly"
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
                "quantype.numpy, quantity methods, or .magnitude(unit)"
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
        if unit is None:
            system = cast("type[UnitSystem]", self._system)
            symbol = system_unit(system, self._semantic).symbol
        else:
            symbol = unit.symbol
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
        system = "" if self._system is NoSystem else f", {self._system.__name__}"
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
        # Coherent value in the reference units (Å, eV, fs, K, μB, e).
        self._reference = reference

    @property
    def kind(self) -> str:
        return str(self._semantic)

    @property
    def dimensions(self) -> tuple[int, ...]:
        return self._semantic.dimensions

    def to_system(self, system: type[AnySystem]) -> Quantity[K, float, Any]:
        """The constant's value in ``system``, as a quantity of that system.

        Without a system, it is stored in the reference units.
        """
        target = require_any_system(system)
        if target is NoSystem:
            raw = self._reference
        else:
            raw = self._reference / factor(
                cast("type[UnitSystem]", target), self._semantic
            )
        return cast("Quantity[K, float, Any]", _wrap(self._semantic, raw, target))

    def magnitude(self, unit: Unit[K]) -> float:
        """The constant's value in a compatible unit."""
        return self.to_system(NoSystem).magnitude(unit)

    def to(self, unit: Unit[K]) -> Quantity[K, float, Any]:
        """The constant as a quantity without a system, presented in ``unit``."""
        return self.to_system(NoSystem).to(unit)

    def _combine(self, other: object, op: str, *, reflected: bool = False) -> Any:
        if isinstance(other, Quantity):
            quantity = cast("Quantity[Any, Any, Any]", other)
            own = (
                _wrap(
                    self._semantic,
                    self._reference,
                    NoSystem,
                    display=adapted(self._semantic, cast("Unit[Any]", quantity.unit)),
                )
                if quantity._system is NoSystem
                else self.to_system(quantity._system)
            )
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
    """A constant of its kind's or product's class, or the structural base class."""
    cls = _CONSTANT_CLASSES.get(semantic)
    if cls is None and isinstance(semantic, Expression):
        from quantype._internal._products import constant_class_for

        cls = cast("type[Constant[Any]] | None", constant_class_for(semantic))
    return (cls or Constant)(name, semantic, reference)


def _unit_named(
    cls: type[Quantity[Any, Any, Any]],
    name: str,
    units: tuple[Unit[Any], ...],
    system: type[AnySystem],
) -> Unit[Any]:
    from quantype._internal._lookup import unit_of_kind

    if getattr(cls, "_semantic", None) is None:
        raise TypeError("unit_named requires a named quantity class")
    if not isinstance(name, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"Unit names are strings; received {type(name).__name__}")
    return unit_of_kind(name, cast("Kind", cls._semantic), units, system)


def _parse(
    cls: type[Quantity[Any, Any, Any]],
    data: object,
    units: tuple[Unit[Any], ...],
    system: type[AnySystem],
) -> Quantity[Any, Any, Any]:
    import numpy as np

    from quantype.serialization import parse_quantity

    result: Any = parse_quantity(cast("Any", cls), data, units=units, system=system)
    raw: Any = result._value
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
                storage=result._storage,
            )
    return result

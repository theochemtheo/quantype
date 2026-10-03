"""A standard generic alias whose call is an explicit storage conversion boundary.

``Length[V, S]`` names storage first and the unit system second (default
``Atomistic``). Calling the alias converts the input into ``S``'s units.
"""

from __future__ import annotations

from types import GenericAlias
from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
    from quantype._internal._systems import UnitSystem
    from quantype._internal._unit import Unit
    from quantype.core import Quantity

# Runtime generic arguments and array backend constructors are dynamic here.
# ruff: noqa: PLC0415
# pyright: reportPrivateUsage=false

# GenericAlias forwards every other attribute to the unparameterized class.
_OWN_ATTRIBUTES = frozenset(
    {"from_value", "parse", "storage", "unit_named", "unit_system"}
)


class StorageAlias(GenericAlias):
    @override
    def __getattribute__(self, name: str) -> Any:
        if name in _OWN_ATTRIBUTES:
            return object.__getattribute__(self, name)
        return super().__getattribute__(name)

    @property
    def storage(self) -> object:
        return self.__args__[0]

    @property
    def unit_system(self) -> type[UnitSystem]:
        from quantype._internal._systems import Atomistic, UnitSystem, require_system

        storage, *rest = self.__args__
        if isinstance(storage, type) and issubclass(storage, UnitSystem):
            raise TypeError(
                "Name the storage type first and the unit system second, "
                "for example Length[float, SI]"
            )
        if len(rest) > 1:
            raise TypeError(f"{self!r} has more than a storage type and a system")
        return require_system(rest[0]) if rest else Atomistic

    def __call__(
        self,
        value: object,
        unit: Unit[Any] | None = None,
        *,
        dtype: object = None,
        display: bool = True,
    ) -> Quantity[Any, Any, Any]:
        from quantype._internal._storage import convert, storage_origin
        from quantype._internal._systems import into_system
        from quantype.core import _wrap, require_unit

        cls = cast("Any", self.__origin__)
        system = self.unit_system
        if unit is None:
            raise TypeError(
                "Typed construction requires an input unit; "
                "use from_value for trusted raw numbers"
            )
        require_unit(unit, cls._kind)
        if unit.semantic is not cls._semantic:
            raise ValueError(f"Expected {cls._kind}; received {unit.kind}")
        scale, offset = into_system(unit, system)
        raw = convert(value, self.storage, dtype=dtype, scale=scale, offset=offset)
        if not display:
            return _wrap(cls._semantic, raw, system)
        # Python scalars are echoed exactly, so "20.1 degC" is not 20.100000000000023.
        echo = None
        if (
            storage_origin(self.storage) is float
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            echo = float(value)
        return _wrap(cls._semantic, raw, system, display=unit, echo=echo)

    def from_value(self, value: object) -> Quantity[Any, Any, Any]:
        """Trusted wrap of raw numbers already in this alias's unit system."""
        from quantype.core import _wrap

        cls = cast("Any", self.__origin__)
        if getattr(cls, "_semantic", None) is None:
            raise TypeError("from_value requires a named quantity class")
        return _wrap(cls._semantic, value, self.unit_system)

    def parse(
        self, data: object, *, units: tuple[Unit[Any], ...] = ()
    ) -> Quantity[Any, Any, Any]:
        from quantype.core import _parse

        return _parse(cast("Any", self.__origin__), data, units, self.unit_system)

    def unit_named(self, name: str, *, units: tuple[Unit[Any], ...] = ()) -> Unit[Any]:
        """Also finds the units of this alias's system, as ``parse`` does."""
        from quantype.core import _unit_named

        return _unit_named(cast("Any", self.__origin__), name, units, self.unit_system)

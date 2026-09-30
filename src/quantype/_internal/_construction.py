"""A standard generic alias whose call is an explicit storage conversion boundary."""

from __future__ import annotations

from types import GenericAlias
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from quantype._internal._unit import Unit
    from quantype.core import Quantity

# Runtime generic arguments and array backend constructors are dynamic here.
# ruff: noqa: PLC0415
# pyright: reportPrivateUsage=false


class StorageAlias(GenericAlias):
    def __call__(
        self, value: object, unit: Unit[Any] | None = None, *, dtype: object = None
    ) -> Quantity[Any, Any]:
        from quantype._internal._storage import convert
        from quantype.core import _wrap

        cls = cast("Any", self.__origin__)
        if unit is None:
            raise TypeError(
                "Typed construction requires an input unit; "
                "use from_canonical for trusted data"
            )
        if unit.semantic is not cls._semantic:
            raise ValueError(f"Expected {cls._kind}; received {unit.kind}")
        storage = self.__args__[-1]
        canonical = convert(
            value, storage, dtype=dtype, scale=unit.scale, offset=unit.offset
        )
        return _wrap(cls._semantic, canonical)

"""Quantity annotations read at runtime: kind class, storage type, unit system.

``quantity_type(Force[npt.NDArray[np.float64], SI])`` gives the class
``Force``, the storage ``npt.NDArray[np.float64]``, and the system ``SI``.
Readers and validators use it to turn an annotation into a target type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypeGuard, get_args, get_origin

from quantype._internal._construction import StorageAlias
from quantype._internal._systems import Atomistic, UnitSystem
from quantype.core import Quantity

__all__ = ["QuantityType", "quantity_type"]


@dataclass(frozen=True)
class QuantityType:
    """The parts of a quantity annotation such as ``Force[float, SI]``."""

    kind: type[Quantity[Any, Any, Any]]
    storage: object
    system: type[UnitSystem]


def quantity_type(annotation: object) -> QuantityType | None:
    """The kind, storage, and system an annotation names, or None for a non-quantity.

    ``Force[V]`` is ``Force[V, Atomistic]``, and a bare ``Force`` has no storage.
    ``Annotated`` metadata is skipped; a union such as ``Force[float] | None`` is
    not a quantity type, so look inside it first. An annotation that names a
    quantity incorrectly, such as ``Force[SI]``, raises ``TypeError``.
    """
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    kind: object = get_origin(annotation) or annotation
    if not _is_quantity_class(kind):
        return None
    if getattr(kind, "_semantic", None) is None:
        raise TypeError(
            f"{annotation!r} is not a named quantity type; "
            "annotate with a kind class such as Force[float]"
        )
    if not isinstance(annotation, StorageAlias):
        return QuantityType(kind, None, Atomistic)
    return QuantityType(kind, annotation.storage, annotation.unit_system)


def _is_quantity_class(
    candidate: object,
) -> TypeGuard[type[Quantity[Any, Any, Any]]]:
    return isinstance(candidate, type) and issubclass(candidate, Quantity)

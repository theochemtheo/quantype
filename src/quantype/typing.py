"""Typing helpers: quantity annotations read at runtime, and storage protocols.

``quantity_type(Force[npt.NDArray[np.float64], SI])`` gives the class
``Force``, the storage ``npt.NDArray[np.float64]``, and the system ``SI``.
Readers and validators use it to turn an annotation into a target type.

``SupportsToNumpy[B]`` reads a quantity type's NumPy form ``B`` in a type
checker, so ``Force[float, SI]`` maps to ``Force[npt.NDArray[np.float64], SI]``
without a table of kinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypeGuard, get_args, get_origin

from quantype._internal._construction import StorageAlias
from quantype._internal._systems import AnySystem, NoSystem
from quantype.core import Quantity

__all__ = ["QuantityType", "SupportsToNumpy", "quantity_type"]


class SupportsToNumpy[B](Protocol):
    """A value whose ``to_numpy()`` gives ``B``, such as every quantity class.

    ``def batched[B](value: SupportsToNumpy[B]) -> B`` returns
    ``Energy[npt.NDArray[np.float64], S]`` for an ``Energy[float, S]``. Every
    supported checker solves ``B``, including ty, because each quantity class
    declares ``to_numpy`` without type parameters, overloads or a typed ``self``.
    """

    def to_numpy(self) -> B: ...


@dataclass(frozen=True)
class QuantityType:
    """The parts of a quantity annotation such as ``Force[float, SI]``."""

    kind: type[Quantity[Any, Any, Any]]
    storage: object
    system: type[AnySystem]


def quantity_type(annotation: object) -> QuantityType | None:
    """The kind, storage, and system an annotation names, or None for a non-quantity.

    ``Force[V]`` is ``Force[V, NoSystem]``, and a bare ``Force`` has no storage.
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
        return QuantityType(kind, None, NoSystem)
    return QuantityType(kind, annotation.storage, annotation.unit_system)


def _is_quantity_class(
    candidate: object,
) -> TypeGuard[type[Quantity[Any, Any, Any]]]:
    return isinstance(candidate, type) and issubclass(candidate, Quantity)

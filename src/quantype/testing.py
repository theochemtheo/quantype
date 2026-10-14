"""Test assertions for quantities, in the style of ``numpy.testing``.

``assert_allclose(actual, desired)`` checks that both quantities have the same
kind and a shared unit system, then compares their numbers as ``numpy.testing``
does.
NumPy, JAX and Torch storage all work, including tensors that require gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeGuard, cast

import numpy as np

from quantype._internal._semantics import (
    TEMPERATURE_DIFFERENCES,
    Kind,
    kind_mismatch,
)
from quantype._internal._storage_units import recast
from quantype._internal._systems import NoSystem, UnitSystem
from quantype.core import Quantity

if TYPE_CHECKING:
    import numpy.typing as npt

    from quantype.core import Unit

__all__ = ["assert_allclose"]


def _numbers(value: object) -> npt.NDArray[Any]:
    """Host NumPy values from any backend's storage."""
    host: Any = value
    if callable(getattr(host, "detach", None)):
        host = host.detach().cpu()
    return np.asarray(host)


def _is_quantity(value: object) -> TypeGuard[Quantity[Any, Any, Any]]:
    return isinstance(value, Quantity)


def _quantity(value: object, role: str) -> Quantity[Any, Any, Any]:
    if not _is_quantity(value):
        raise TypeError(
            f"assert_allclose compares quantities; {role} is a "
            f"{type(value).__name__}. Compare plain numbers with numpy.testing"
        )
    return value


def _tolerance(
    atol: object, desired: Quantity[Any, Any, Any], unit: Unit[Any] | None
) -> float:
    """``atol`` in ``unit``, or in the system's units when there is none.

    Absolute temperatures take differences. A tolerance without a system
    adopts the desired quantity's.
    """
    if isinstance(atol, int) and atol == 0:
        return 0.0
    bound = _quantity(atol, "atol")
    kind = desired._semantic  # noqa: SLF001
    accepted = {kind}
    if isinstance(kind, Kind) and kind in TEMPERATURE_DIFFERENCES:
        accepted.add(TEMPERATURE_DIFFERENCES[kind])
    if bound._semantic not in accepted or bound.system not in {  # noqa: SLF001
        desired.system,
        NoSystem,
    }:
        where = (
            ""
            if desired.system is NoSystem
            else f" in {desired.system.__name__} or without a system"
        )
        raise TypeError(f"atol must be a {desired.kind}{where}, or 0")
    if unit is None:
        return float(_numbers(bound.to_system(desired.system)._value))  # noqa: SLF001
    return float(_numbers(bound.magnitude(recast(bound._semantic, unit))))  # noqa: SLF001


# One type variable for both quantities: with a shared kind parameter instead,
# mypy and Pyright infer the kind of a product expression from its left factor.
def assert_allclose[Q: Quantity[Any, Any, Any]](  # noqa: PLR0913 -- numpy.testing's
    actual: Q,
    desired: Q,
    *,
    rtol: float = 1e-7,
    atol: Quantity[Any, Any, Any] | Literal[0] = 0,
    equal_nan: bool = True,
    err_msg: str = "",
) -> None:
    """Raise AssertionError unless ``actual`` is close to ``desired``.

    Both must have the same kind. A quantity without a system adopts the
    other's, as for ``==``, while two unit systems don't mix. Their units may
    differ, so ``2 nm`` is close to ``20 Å``. ``atol`` is a quantity of the
    same kind (a temperature difference for absolute temperatures), or zero.
    The comparison uses the unit system's units, or without a system the unit
    the desired numbers are stored in, which the message names: kelvin for a
    temperature given in °C, so ``rtol`` is relative to absolute zero.
    """
    first = _quantity(actual, "actual")
    second = _quantity(desired, "desired")
    expected, received = second._semantic, first._semantic  # noqa: SLF001
    if expected != received:
        raise AssertionError(kind_mismatch(expected, received))
    systems = {first.system, second.system}
    if len(systems) > 1 and NoSystem not in systems:
        raise AssertionError(
            f"Expected a quantity in {second.system.__name__}; received one in "
            f"{first.system.__name__}. Convert it with "
            f".to_system({second.system.__name__})"
        )
    first, second = first._adopted(second)  # noqa: SLF001
    unit = second._storage  # noqa: SLF001
    where = f"{second.kind} in {second.system.__name__}"
    if unit is not None:
        where += f", compared in {unit.symbol}"
    elif isinstance(second._semantic, Kind):  # noqa: SLF001
        system = cast("type[UnitSystem]", second.system)
        where += f", compared in {system.unit_for(second._semantic).symbol}"  # noqa: SLF001
    np.testing.assert_allclose(
        _numbers(first.magnitude(unit) if unit else first._value),  # noqa: SLF001
        _numbers(second.magnitude(unit) if unit else second._value),  # noqa: SLF001
        rtol=rtol,
        atol=_tolerance(atol, second, unit),
        equal_nan=equal_nan,
        err_msg=f"{err_msg}\n{where}".strip(),
    )

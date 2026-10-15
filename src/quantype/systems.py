"""Unit systems: what a quantity's raw numbers mean.

``Length[float]`` has no unit system (``NoSystem``): it keeps the unit it was
given in, and has no raw numbers. ``Length[float, SI]`` stores meters and
``Length[float, Atomistic]`` ångströms, and ``.value`` reads them. Systems
never mix implicitly; ``.to_system(...)`` is the explicit bridge. Subclass
``UnitSystem`` to define your own.
"""

from quantype._internal._storage import StorageRangeWarning
from quantype._internal._systems import (
    CGS,
    SI,
    AnySystem,
    Atomic,
    Atomistic,
    Metal,
    NoSystem,
    RangeIssue,
    Real,
    UnitSystem,
)

__all__ = [
    "CGS",
    "SI",
    "AnySystem",
    "Atomic",
    "Atomistic",
    "Metal",
    "NoSystem",
    "RangeIssue",
    "Real",
    "StorageRangeWarning",
    "UnitSystem",
]

"""Coherent unit systems: what a quantity's raw numbers mean.

A system is a class. The same object is the static type parameter
(``Length[float, SI]``) and the runtime value (``q.to_system(SI)``). It names
one unit per base axis. Every kind's unit is the product of those units raised
to the kind's dimensions, so arithmetic inside one system never converts.

Systems are defined over the base axes, not over a list of kinds, so a kind
from a generated catalogue has a unit in every system automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast, override

from quantype._internal._registry import BASIS
from quantype._internal._semantics import KINDS, Kind
from quantype._internal._unit import Unit, get_unit, known_kinds, units_of
from quantype.kinds import (
    AtomCountKind,
    ElectronCountKind,
    EnergyKind,
    LengthKind,
    MagneticMomentKind,
    TemperatureKind,
    TimeKind,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import DTypeLike

    from quantype._internal._semantics import Semantic

# pyright: reportPrivateUsage=false

__all__ = [
    "CGS",
    "SI",
    "Atomic",
    "Atomistic",
    "Metal",
    "RangeIssue",
    "Real",
    "UnitSystem",
    "factor",
    "into_system",
    "require_system",
]

AXIS_KINDS = dict(
    zip(
        BASIS,
        (
            "Length",
            "Energy",
            "Time",
            "Temperature",
            "MagneticMoment",
            "AtomCount",
            "ElectronCount",
        ),
        strict=True,
    )
)
REQUIRED_AXES = ("length", "energy", "time")
# A derived scale names a catalogue unit when it agrees to this relative tolerance.
_TOLERANCE = 1e-9


class _CatalogueUnit:
    """A built-in base unit, resolved on first use so importing quantype stays light."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: object, owner: type | None = None) -> Unit[Any]:
        return get_unit(self.name)


def _catalogue[K](name: str, kind: type[K]) -> Unit[K]:
    """A class-body placeholder that the class attribute resolves to a unit."""
    del kind
    return cast("Unit[K]", _CatalogueUnit(name))


class _SystemUnits:
    """``System.units``: every non-catalogue unit a system defines, for decoding."""

    def __get__(
        self, instance: object, owner: type[UnitSystem] | None = None
    ) -> tuple[Unit[Any], ...]:
        if owner is None or "_derived" not in owner.__dict__:
            return ()
        defined = [
            *_resolve(owner).values(),
            *(owner.unit_for(kind) for kind in known_kinds()),
        ]
        return tuple(
            unit for unit in dict.fromkeys(defined) if not _is_catalogue_unit(unit)
        )


def _is_catalogue_unit(unit: Unit[Any]) -> bool:
    try:
        return get_unit(unit.name, kind=unit.semantic) is unit
    except ValueError:
        return False


# Built-in systems name catalogue units, which load SciPy. Their validation is
# deferred to first use and covered by the test suite instead.
_deferred_validation = False


class UnitSystem:
    """Base class for unit systems.

    Subclass it with ``name=`` and a base unit for ``length``, ``energy`` and
    ``time``. Temperature (kelvin), magnetic moment (μB) and the atom and
    electron counts have defaults. Nothing is registered globally: the class is
    usable as soon as it is defined, and it is checked when it is defined.
    """

    name: ClassVar[str]
    length: ClassVar[Unit[LengthKind]]
    energy: ClassVar[Unit[EnergyKind]]
    time: ClassVar[Unit[TimeKind]]
    temperature: ClassVar[Unit[TemperatureKind]] = _catalogue("kelvin", TemperatureKind)
    magnetic_moment: ClassVar[Unit[MagneticMomentKind]] = _catalogue(
        "bohr_magneton", MagneticMomentKind
    )
    atom: ClassVar[Unit[AtomCountKind]] = _catalogue("atom", AtomCountKind)
    electron: ClassVar[Unit[ElectronCountKind]] = _catalogue(
        "electron", ElectronCountKind
    )
    #: Units to pass to decoders (``units=``) that the catalogue cannot name.
    units = _SystemUnits()

    _bases: ClassVar[dict[str, Unit[Any]] | None] = None
    _derived: ClassVar[dict[Kind, Unit[Any]]]

    if TYPE_CHECKING:

        def __init_subclass__(cls, *, name: str, **kwargs: object) -> None: ...

    else:

        def __init_subclass__(
            cls, *, name: str | None = None, **kwargs: object
        ) -> None:
            super().__init_subclass__(**kwargs)
            _define(cls, name)

    def __new__(cls) -> Self:
        raise TypeError(
            f"Unit systems are used as classes: write {cls.__name__}, "
            f"not {cls.__name__}()"
        )

    @classmethod
    def unit_for(cls, kind: Kind | str | type[Any]) -> Unit[Any]:
        """The unit that raw numbers of ``kind`` are stored in, in this system."""
        semantic = _kind_of(kind)
        derived = cast("dict[Kind, Unit[Any]] | None", cls.__dict__.get("_derived"))
        if derived is None:
            raise TypeError(
                "UnitSystem is abstract; use a concrete system such as Atomistic"
            )
        unit = derived.get(semantic)
        if unit is None:
            unit = derived[semantic] = _derive(cls, semantic)
        return unit

    @classmethod
    def check_range(cls, dtype: DTypeLike) -> tuple[RangeIssue, ...]:
        """Report kinds whose typical magnitudes, or their squares, leave ``dtype``.

        Squares are included because variances and sums of squares are common.
        """
        import numpy as np  # noqa: PLC0415

        name = np.dtype(dtype).name
        info = np.finfo(name)
        maximum, tiny = float(info.max), float(info.tiny)
        issues: list[RangeIssue] = []
        for kind in known_kinds():
            unit = cls.unit_for(kind)
            for reference in _typical(kind):
                magnitude = reference / unit.scale
                for squared, value in (
                    (False, magnitude),
                    (True, magnitude * magnitude),
                ):
                    if abs(value) > maximum:
                        problem: Literal["overflow", "underflow"] = "overflow"
                    elif 0 < abs(value) < tiny:
                        problem = "underflow"
                    else:
                        continue
                    issues.append(
                        RangeIssue(
                            cls.__name__,
                            kind.name,
                            unit.symbol,
                            magnitude,
                            squared=squared,
                            problem=problem,
                            dtype=name,
                        )
                    )
        return tuple(issues)


@dataclass(frozen=True)
class RangeIssue:
    """A typical magnitude of ``kind`` that ``dtype`` cannot hold in ``system``."""

    system: str
    kind: str
    symbol: str
    magnitude: float
    squared: bool
    problem: Literal["overflow", "underflow"]
    dtype: str

    @override
    def __str__(self) -> str:
        subject = f"({self.magnitude:.3g} {self.symbol})"
        if self.squared:
            subject += "²"
        return f"{self.system}: {self.kind} {subject} would {self.problem} {self.dtype}"


def _define(cls: type[UnitSystem], name: object) -> None:
    if not isinstance(name, str) or not name.strip():
        raise TypeError("UnitSystem subclass requires name=...")
    for base in cls.__mro__[1:]:
        if base.__dict__.get("name") == name:
            raise TypeError(
                f"{cls.__name__} reuses the system name {name!r} of "
                f"{base.__name__}; each system in a hierarchy needs its own name"
            )
    cls.name = name
    cls._bases = None
    cls._derived = {}
    if not _deferred_validation:
        _resolve(cls)


def _resolve(cls: type[UnitSystem]) -> dict[str, Unit[Any]]:
    """Validate and return the base units: at definition, or on first use."""
    resolved: dict[str, Unit[Any]] | None = cls.__dict__.get("_bases")
    if resolved is not None:
        return resolved
    bases: dict[str, Unit[Any]] = {}
    for axis in BASIS:
        unit = getattr(cls, axis, None)
        label = axis.replace("_", " ")
        if unit is None:
            raise TypeError(f"{cls.__name__} is missing a {label} unit")
        if not isinstance(unit, Unit):
            raise TypeError(
                f"{cls.__name__}.{axis} must be a unit; received {type(unit).__name__}"
            )
        expected = AXIS_KINDS[axis]
        # Compare kind names: a generated catalogue's units are equally valid.
        if unit.kind != expected:
            article = "an" if expected[0] in "AEIOU" else "a"
            raise TypeError(
                f"{cls.__name__}.{axis} must be {article} {expected} unit; "
                f"received {unit.kind} ({unit.name})"
            )
        if unit.offset:
            raise ValueError("Temperature base must not have an offset; use kelvin")
        bases[axis] = unit
    for kind in known_kinds():
        _checked_scale(cls, bases, kind)
    cls._bases = bases
    return bases


def _scale(bases: dict[str, Unit[Any]], dimensions: Sequence[int]) -> float:
    scale = 1.0
    for axis, exponent in zip(BASIS, dimensions, strict=True):
        if exponent:
            try:
                scale *= float(bases[axis].scale ** exponent)
            except OverflowError:
                return math.inf
    return scale


def _checked_scale(
    cls: type[UnitSystem], bases: dict[str, Unit[Any]], kind: Kind
) -> float:
    scale = _scale(bases, kind.dimensions)
    if not math.isfinite(scale) or scale == 0:
        problem = "underflows" if scale == 0 else "overflows"
        raise ValueError(f"{cls.__name__} gives {kind} a scale that {problem} float64")
    return scale


def _kind_of(kind: Kind | str | type[Any]) -> Kind:
    semantic: object = kind
    if isinstance(kind, str):
        semantic = KINDS.get(kind)
    elif isinstance(kind, type):
        semantic = getattr(kind, "_semantic", None)
    if not isinstance(semantic, Kind):
        raise TypeError(
            f"Expected a named quantity kind; received {kind!r}. "
            "Structural quantities have no named unit"
        )
    return semantic


def _derive(cls: type[UnitSystem], kind: Kind) -> Unit[Any]:
    bases = _resolve(cls)
    scale = _checked_scale(cls, bases, kind)
    for base in bases.values():
        if base.semantic is kind:
            return base
    candidates = [
        unit
        for unit in units_of(kind)
        if unit.offset == 0 and math.isclose(unit.scale, scale, rel_tol=_TOLERANCE)
    ]
    for unit in candidates:
        if unit.name == kind.canonical_unit:
            return unit
    if candidates:
        return candidates[0]
    return Unit(
        f"{cls.name}:{kind}", kind, scale, symbol=_compose(bases, kind.dimensions)
    )


def _compose(bases: dict[str, Unit[Any]], dimensions: Sequence[int]) -> str:
    def term(axis: str, power: int) -> str:
        symbol = bases[axis].symbol
        return symbol if power == 1 else f"{symbol}^{power}"

    pairs = tuple(zip(BASIS, dimensions, strict=True))
    numerator = [term(axis, power) for axis, power in pairs if power > 0]
    denominator = [term(axis, -power) for axis, power in pairs if power < 0]
    text = " ".join(numerator) or "1"
    for symbol in denominator:
        text += f"/{symbol}"
    return text


def factor(system: type[UnitSystem], semantic: Semantic) -> float:
    """Reference units per stored unit of ``semantic`` in ``system``."""
    if system is Atomistic:
        return 1.0
    if isinstance(semantic, Kind):
        return system.unit_for(semantic).scale
    return _scale(_resolve(system), semantic.dimensions)


def into_system(unit: Unit[Any], system: type[UnitSystem]) -> tuple[float, float]:
    """The affine map from magnitudes in ``unit`` to raw numbers in ``system``."""
    scale = factor(system, unit.semantic)
    if scale == 1:
        return unit.scale, unit.offset
    return unit.scale / scale, unit.offset / scale


def require_system(candidate: object) -> type[UnitSystem]:
    if (
        isinstance(candidate, type)
        and issubclass(candidate, UnitSystem)
        and "_derived" in candidate.__dict__
    ):
        return candidate
    raise TypeError(f"Expected a unit system class such as SI; received {candidate!r}")


# Representative magnitudes in reference units (Å, eV, fs, K, μB), used by
# check_range. Kinds without an entry combine the per-axis ranges below.
_TYPICAL: dict[str, tuple[float, float]] = {
    "Dimensionless": (1e-6, 1e6),
    "Length": (1e-2, 1e4),
    "Area": (1e-4, 1e8),
    "Volume": (1e-1, 1e9),
    "Time": (1e-1, 1e9),
    "Velocity": (1e-6, 1.0),
    "Energy": (1e-4, 1e6),
    "EnergyPerAtom": (1e-4, 1e3),
    "Force": (1e-4, 1e2),
    "ForceConstant": (1e-3, 1e3),
    "Pressure": (1e-7, 1.0),
    "EnergyDensity": (1e-7, 1.0),
    "EnergyPerVolume": (1e-7, 1.0),
    "Temperature": (1e-2, 1e5),
    "TemperatureDifference": (1e-3, 1e4),
    "TemperatureRate": (1e-9, 1e-1),
    "MagneticMoment": (1e-3, 1e3),
    "Magnetization": (1e-5, 1.0),
    "ParticleDensity": (1e-6, 1.0),
    "ElectronDensity": (1e-6, 10.0),
    "Angle": (1e-4, 10.0),
    "Frequency": (1e-9, 1.0),
    "InverseTime": (1e-9, 1.0),
    "AtomCount": (1.0, 1e8),
    "ElectronCount": (1.0, 1e9),
}
_AXIS_TYPICAL = (
    (1e-2, 1e4),
    (1e-4, 1e6),
    (1e-1, 1e9),
    (1e-2, 1e5),
    (1e-3, 1e3),
    (1.0, 1e8),
    (1.0, 1e9),
)


def _typical(kind: Kind) -> tuple[float, float]:
    known = _TYPICAL.get(kind.name)
    if known is not None:
        return known
    low = high = 1.0
    for (axis_low, axis_high), exponent in zip(
        _AXIS_TYPICAL, kind.dimensions, strict=True
    ):
        if exponent > 0:
            low *= axis_low**exponent
            high *= axis_high**exponent
        elif exponent < 0:
            low *= axis_high**exponent
            high *= axis_low**exponent
    return low, high


def range_table(systems: Iterable[type[UnitSystem]], dtypes: Sequence[str]) -> str:
    """Markdown summary of check_range, as published in the documentation."""
    lines = [
        "| System | " + " | ".join(dtypes) + " |",
        "|---" * (len(dtypes) + 1) + "|",
    ]
    for system in systems:
        cells: list[str] = []
        for dtype in dtypes:
            flagged: dict[str, bool] = {}
            for issue in system.check_range(dtype):
                flagged[issue.kind] = flagged.get(issue.kind, True) and issue.squared
            names = [
                kind + ("²" if only_squared else "")
                for kind, only_squared in flagged.items()
            ]
            cells.append(", ".join(sorted(names)) or "safe")
        lines.append(f"| `{system.__name__}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


_deferred_validation = True


class Atomistic(UnitSystem, name="atomistic"):
    """Å, eV, fs, K, μB: the reference units, and the default system."""

    length = _catalogue("angstrom", LengthKind)
    energy = _catalogue("electron_volt", EnergyKind)
    time = _catalogue("femtosecond", TimeKind)


class Metal(UnitSystem, name="metal"):
    """LAMMPS ``units metal``: Å, eV, ps."""

    length = _catalogue("angstrom", LengthKind)
    energy = _catalogue("electron_volt", EnergyKind)
    time = _catalogue("picosecond", TimeKind)


class Real(UnitSystem, name="real"):
    """LAMMPS ``units real``: Å, kcal/mol, fs."""

    length = _catalogue("angstrom", LengthKind)
    energy = _catalogue("kcal_per_mol", EnergyKind)
    time = _catalogue("femtosecond", TimeKind)


class SI(UnitSystem, name="si"):
    """m, J, s, A·m²."""

    length = _catalogue("meter", LengthKind)
    energy = _catalogue("joule", EnergyKind)
    time = _catalogue("second", TimeKind)
    magnetic_moment = _catalogue("ampere_meter_squared", MagneticMomentKind)


class CGS(UnitSystem, name="cgs"):
    """cm, erg, s, erg/G."""

    length = _catalogue("centimeter", LengthKind)
    energy = _catalogue("erg", EnergyKind)
    time = _catalogue("second", TimeKind)
    magnetic_moment = _catalogue("erg_per_gauss", MagneticMomentKind)


class Atomic(UnitSystem, name="atomic"):
    """Hartree atomic units: a0, Ha, ħ/Eh, eħ/me."""

    length = _catalogue("bohr", LengthKind)
    energy = _catalogue("hartree", EnergyKind)
    time = _catalogue("atomic_time", TimeKind)
    magnetic_moment = _catalogue("atomic_magnetic_moment", MagneticMomentKind)


_deferred_validation = False

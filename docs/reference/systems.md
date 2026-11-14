# Unit systems

Public imports come from `quantype.systems`: `UnitSystem`, `NoSystem`,
`AnySystem`, `Atomistic`, `ASE`, `Metal`, `Real`, `SI`, `CGS`, `Atomic`,
`RangeIssue`, and `StorageRangeWarning`.

A [unit system](../explanation/glossary.md#unit-system "A convention fixing the units of raw stored numbers.")
fixes the units of raw numbers. It is the second quantity type parameter, as
in `Length[float, SI]`, and is always used as a class, never an instance. The
[physical model](../explanation/physical-model.md#unit-systems-opt-into-raw-numbers)
explains when this boundary is useful. The
[quantity conversion API](quantities.md#conversion-and-numerical-access) owns
`.value` and `.to_system` signatures; the
[construction API](quantities.md#construction-and-storage) defines constructors
and `from_value`. [Define systems](../how-to/define-systems.md) gives task-oriented
examples.

## Built-in systems

A system names one unit per base axis. Each kind's coherent storage unit is the
product of those units raised to its dimension exponents. Thus forces in
`Atomistic` use eV/Å. A system can override particular named kinds; arithmetic
producing or consuming them rescales by a constant factor. Structural unnamed
quantities are always stored coherently.

| System | Length | Energy | Time | Charge | Magnetic moment | Own units |
| --- | --- | --- | --- | --- | --- | --- |
| `Atomistic` | Å | eV | fs | e | μB | mass Da, density g/cm³ |
| `ASE` | Å | eV | Å √(amu/eV) ≈ 10.18 fs | e | μB | |
| `Metal` (LAMMPS) | Å | eV | ps | e | μB | mass g/mol, pressure bar, density g/cm³ |
| `Real` (LAMMPS) | Å | kcal/mol | fs | e | μB | mass g/mol, pressure atm, electric field V/Å, density g/cm³ |
| `SI` | m | J | s | C | A·m² | |
| `CGS` | cm | erg | s | statC | erg/G | |
| `Atomic` (Hartree) | a0 | Ha | ħ/Eₕ | e | eħ/mₑ | |

Every built-in system stores temperature in kelvin and atom and electron counts
in units of 1. System choice does not change a quantity's kind or dtype; it
changes the units of storage. See the [range table](units.md#numerical-range)
before using low-precision atomistic data in SI or CGS.

### Atomistic and ASE

With Å, eV, and fs, the coherent mass unit would be eV fs²/Å², about
0.00965 Da. `Atomistic` instead stores mass in daltons, the unit ASE uses, and
density in g/cm³.

!!! warning "Raw arithmetic omits override factors"

    In `Atomistic`, `0.5 * m * v**2` includes the factor
    1 Da Å²/fs² ≈ 103.6 eV. Multiplying the stored raw numbers alone would be wrong.
    Use quantity arithmetic to include each kind's coherence factor.

`ASE` derives time to keep mass coherent instead. Its time unit is
`u.ase_time`, Å √(amu/eV), about 10.18 fs. Velocities use `u.ase_velocity`,
√(eV/amu), and momenta `u.ase_momentum`, √(amu eV). Raw arrays from `ase.Atoms`
can be wrapped with `from_value` without conversion; arithmetic in this system
does not rescale.

```python
import numpy as np
import pytest

from quantype import Mass, Velocity, u
from quantype.systems import ASE, Atomistic

mass = Mass[float, Atomistic](12, u.Da)
speed = Velocity[float, Atomistic](0.01, u.angstrom_per_fs)
assert mass.value == 12.0
assert (0.5 * mass * speed**2).value == pytest.approx(0.0621856)

raw_masses = np.array([[1.008], [15.999]])  # atoms.get_masses()[:, None]
raw_velocities = np.array([[0.5, 0.0, 0.0], [0.0, 0.1, 0.0]])  # get_velocities()
masses = Mass.from_value(raw_masses, ASE)
velocities = Velocity.from_value(raw_velocities, ASE)
momenta = masses * velocities
np.testing.assert_array_equal(momenta.value, raw_masses * raw_velocities)
assert velocities.magnitude(u.angstrom_per_fs)[0, 0] == pytest.approx(0.0491135)
```

`ase.units` uses CODATA 2014, while Quantype defaults to 2022. Their
`u.ase_time` values differ by about 5 parts in 10⁹. Set `QUANTYPE_CODATA=2014`
before using a unit to match ASE's values; see [choose constants](../how-to/choose-constants.md).

### LAMMPS Metal and Real

`Metal` and `Real` store each supported quantity from LAMMPS `units metal` and
`units real` in the documented LAMMPS units, so raw simulator numbers can be
wrapped as they are. LAMMPS also lists torque and dynamic viscosity, which have
no Quantype kinds.

```python
import pytest

from quantype import Area, Force, Mass, Pressure, Velocity, u
from quantype.systems import Metal

assert Metal.unit_for(Pressure) is u.bar
assert Metal.unit_for(Mass) is u.gram_per_mole
stress = Force[float, Metal](1, u.eV_per_angstrom) / Area[float, Metal](
    1, u.angstrom_squared
)
assert stress.value == pytest.approx(1.602176634e6)  # bar
mass = Mass[float, Metal](1, u.gram_per_mole)
speed = Velocity[float, Metal](1, u.angstrom_per_ps)
assert (mass * speed * speed).value == pytest.approx(1.0364269e-4)  # mvv2e
```

LAMMPS converts internally using older CODATA constants rounded to about seven
digits (`boltz = 8.617343e-5` in `metal`). Derived values, such as temperature
from kinetic energy, can differ from Quantype in the seventh significant figure.
Choosing the same edition alone does not remove LAMMPS's rounding.

## System selection and adoption

`Length[float]` means `Length[float, NoSystem]`: it has per-quantity storage
units and no `.value`. `NoSystem` is not a subclass of `UnitSystem`.
`AnySystem` is the type alias `UnitSystem | NoSystem`, a bound for generic
parameters, not a concrete system. No quantity is a `Length[float, AnySystem]`,
and `.to_system(AnySystem)` is a static error.

Two different bound systems cannot participate in arithmetic or ordering:
all four supported checkers reject the operation and runtime raises `TypeError`
naming both systems. Equality instead compares them unequal. A `NoSystem`
operand adopts the other's system for arithmetic, comparisons including `==`,
and `quantype.numpy` calls: conversion happens before the operation. This does
not mutate either input.

```python
import pytest

from quantype import Energy, Length, u
from quantype.systems import SI, Metal, NoSystem, Real

length = Length[float, SI](2, u.nm)
energy = Energy[float, SI](3, u.eV)
assert (energy / length).system is SI  # SI in, SI out: newtons
metal = Length[float, Metal](5, u.angstrom)
with pytest.raises(TypeError, match="Cannot combine SI and Metal"):
    length + metal
assert (length + metal.to_system(SI)).magnitude(u.nm) == pytest.approx(2.5)
assert (length + 1 * u.nm).system is SI
kernel_input = (2 * u.nm).to_system(Real)
assert kernel_input.value == pytest.approx(20.0)
unbound = kernel_input.to_system(NoSystem)
assert unbound.magnitude(u.angstrom) == kernel_input.value
```

`.to_system(T)` is the explicit bridge: it rescales raw numbers while keeping
the display unit. `.to_system(NoSystem)` does not convert the numbers; they
remain in the source system's storage unit. System-to-system temperature
conversion has no affine offset because every system stores temperature on a
kelvin scale.

!!! warning "Adoption stops at function and validation boundaries"

    Function parameters do not adopt. A parameter typed `Length[float, SI]`
    accepts only SI quantities; `Length[float]` accepts only quantities without a
    system. [Pydantic fields and `TypeAdapter`](serialization.md#pydantic-schemas-and-target-storage)
    enforce that boundary at runtime too, with errors naming `.to_system(...)`.
    The `quantype.numpy` functions taking a list, such as `concatenate`, adopt at
    runtime, but static checkers require that list to be in one system. Convert it
    first. See [validation](../how-to/validate-data.md) and
    [numerical dispatch](numerical-operations.md#runtime-dispatch-versus-static-support).

## UnitSystem definition contract

```text
class MySystem(UnitSystem, name="my_lab:system"):
    length = ...
    energy = ...
    time = ...
    overrides = ()
```

The `length`, `energy`, and `time` axes are required. The other axes have
inheritable defaults:

| Attribute | Kind | Default |
| --- | --- | --- |
| `temperature` | `Temperature` | kelvin |
| `magnetic_moment` | `MagneticMoment` | Bohr magneton |
| `atom` | `AtomCount` | atom |
| `electron` | `ElectronCount` | electron |
| `charge` | `Charge` | elementary charge |

Base units may be custom `Kind.define_unit` objects. `overrides` is a tuple
of units replacing the derived units of particular named non-axis kinds.
Definitions are usable immediately with no global registration. Validation runs
when a custom class is defined, so mistakes appear during import:

| Rule | Error example |
| --- | --- |
| `name=` must be nonempty and unique within its inheritance hierarchy. Namespaced names are encouraged. | `UnitSystem subclass requires name=...` |
| Required axes must be present, possibly inherited. | `Gromacs is missing a time unit` |
| Each base must be a unit of its axis's nominal kind. | `Gromacs.energy must be an Energy unit; received Length (nanometer)` |
| Temperature must have no offset; affine units are not coherent bases. | `Temperature base must not have an offset; use kelvin` |
| Every derived scale for every known catalogue kind must be finite and nonzero in float64. | `Huge gives Area a scale that overflows float64` |
| Overrides must be units of named non-axis kinds, at most one per kind, without offsets. | `Gromacs overrides Length, a base axis; set its base unit instead` |

Missing names/axes, wrong unit kinds, duplicate or axis overrides, and attempts
to instantiate a system raise `TypeError`. Offset bases and nonrepresentable
derived scales raise `ValueError`. `UnitSystem` itself is abstract: choose a
concrete subclass. Renaming the Python class does not affect saved data;
changing `name=` does, because it is part of generated unit identifiers.

### Derived units and decoding

```text
System.unit_for(kind) -> Unit
System.units -> tuple[Unit, ...]
```

`unit_for` accepts a named quantity class, its semantic kind, or a built-in
kind name such as `"Pressure"`. Structural expressions have no named system
unit and raise `TypeError`. A built-in system's unit for a built-in kind is
always a named catalogue unit, for example `SI.unit_for(Pressure) is u.pascal`.
For custom systems, a matching catalogue unit is reused; otherwise the result
has a system-qualified identifier such as `gromacs:Force` and symbol `kJ/mol/nm`.
These units appear in `repr` and serialized data when no display unit is set.

Systems derive units from axes rather than a closed list of kinds. A new
`SurfaceTension` from a [generated catalogue](../how-to/custom-catalogues.md)
automatically gets a unit in every system: `J/m^2` in SI and, when unmatched by
a named unit, `gromacs:SurfaceTension` in `Gromacs`.

`System.units` contains all non-catalogue unit definitions needed for decoding,
including custom bases and derived system-qualified units. Pass it as
`units=System.units` to untyped entry points or when decoding into another
system. A target naming the system, such as `TypeAdapter(Force[float, Gromacs])`
or `load_npz(..., Force[NDArray, Gromacs])`, finds those definitions
automatically. Saved values name units, not their source system; they may be
restored into any target system. The
[define-system task](../how-to/define-systems.md#decode-system-defined-units)
shows executable parsing, JSON validation, and NPZ examples.

### Inheritance and invariance

Inheriting to vary one axis creates a separate system, even if all other units
are unchanged. The system parameter is invariant:
`Length[float, MetalFs]` cannot substitute for `Length[float, Metal]`.
Mixing a parent system and its subclass raises `TypeError` at runtime. This
is deliberately unlike ordinary Python subclass substitutability: the two
classes give raw numbers different meanings.

There is one static inference gap: mypy accepts an inline expression such as
`metal + value.to_system(MetalFs)` because it infers the conversion target from
the expected operand type, and `type[MetalFs]` is also `type[Metal]`. Pyright,
Pyrefly, and ty reject it. mypy rejects it too once the converted value is
bound to a name. Runtime rejects the mixed operation in all cases.

## Numerical range

```text
System.check_range(dtype) -> tuple[RangeIssue, ...]
```

This NumPy-backed diagnostic accepts a NumPy dtype specification such as
`np.float32` or `"float32"`. It compares representative atomistic magnitudes,
and their squares, to the dtype's normal floating-point range. It is not a
validation of your particular data. An empty tuple means those representative
values fit; a custom system may underflow float32 just as SI does. The
[built-in range table](units.md#numerical-range) uses the same calculation.

`RangeIssue` is an immutable record with `system`, `kind`, `symbol`,
`magnitude`, `squared`, `problem` (`"underflow"` or `"overflow"`), and `dtype`
fields. Its string describes the affected magnitude and, when appropriate,
its square.

```python
import numpy as np

from quantype.systems import SI

issues = SI.check_range(np.float32)
assert any(issue.kind == "Energy" and issue.squared for issue in issues)
assert SI.check_range(np.float64) == ()
for issue in issues:
    print(issue)  # e.g. SI: Area (1e-24 m^2)² would underflow float32
```

!!! warning "Range diagnostics are not data validation"

    `StorageRangeWarning` is a separate warning about actual converted values,
    emitted for float16/float32 NumPy construction and `.to_system` when values
    overflow or become subnormal. JAX and Torch traces are not checked.
    `check_range` only checks representative magnitudes, not your particular data.

## Write system-generic code

Use a type parameter bounded by `AnySystem` when a library accepts quantities
with or without a system. Bound it by `UnitSystem` if the function reads raw
`.value`.

```python
from quantype import Energy, Force, Length, u
from quantype.systems import SI, AnySystem, Metal, UnitSystem


def work[S: AnySystem](f: Force[float, S], d: Length[float, S]) -> Energy[float, S]:
    return f * d


def raw_cutoff[S: UnitSystem](cutoff: Length[float, S]) -> float:
    return cutoff.value


si = work(Force[float, SI](1, u.newton), Length[float, SI](2, u.meter))
metal = work(Force[float, Metal](1, u.eV_per_angstrom), Length[float, Metal](2, u.nm))
unbound = work(Force[float](1, u.eV_per_angstrom), Length[float](2, u.nm))
assert si.system is SI
assert metal.system is Metal
assert unbound == 2 * u.eV_per_angstrom * (1 * u.nm)
assert raw_cutoff(Length[float, Metal](2, u.nm)) == 20.0
```

Checkers accept matched SI, Metal, and `NoSystem` pairs and reject a mixed pair.
Within a coherent system, arithmetic in `work` needs no conversion; overridden
kinds still include their fixed scale factors. A literal such as `1.0 * u.nm`
has no system and adopts `S` in an operation.

Pyright and ty determine a product's kind from the right operand and cannot
adopt `S` from there. Put a literal on the left, as in
`(1.0 * u.nm) * force`, or explicitly give it the system, as in
`cutoff / (1.0 * u.fs).to_system(cutoff.system)`.

`raw_cutoff` rejects a length without a system. mypy reports
``Value of type variable "S" of "raw_cutoff" cannot be "NoSystem"``;
Pyrefly says `NoSystem` is not assignable to the upper bound `UnitSystem`.
Direct `.value` access can instead name the descriptor in the diagnostic:
``Value of type variable "T" of "__get__" of "_Value" cannot be "NoSystem"``.

Prefer system-generic annotations for libraries and explicit systems such as
`Length[npt.NDArray[np.float64], Real]` for kernels requiring particular raw
units. `ujax.grad`, `ujax.hessian`, and `utorch.grad` preserve a generic system
too: a function from `Length[Array, S]` to `Energy[Array, S]` differentiates to
`Force[Array, S]`. See [JAX](../how-to/autodiff-jax.md) and
[Torch](../how-to/autodiff-torch.md) differentiation.

### Generic over the kind

A whole-quantity type variable accepts named kinds and structural products:

```python
from typing import Any

from quantype import Quantity, u


def larger[Q: Quantity[Any, Any, Any]](a: Q, b: Q) -> Q:
    return a if a > b else b


assert larger(2 * u.nm, 3 * u.nm) == 3 * u.nm
product = larger((2 * u.nm) * (3 * u.fs), (3 * u.fs) * (2 * u.nm))
assert product == (6 * u.nm) * (1 * u.fs)
```

A shared kind parameter also works for named kinds and for product variables:

```python
from typing import Any

from quantype import Quantity, u
from quantype.systems import AnySystem


def larger[K, S: AnySystem](
    a: Quantity[K, Any, S], b: Quantity[K, Any, S]
) -> Quantity[K, Any, S]:
    return a if a > b else b


assert larger(2 * u.nm, 3 * u.nm) == 3 * u.nm
left = (2 * u.nm) * (3 * u.fs)
right = (3 * u.fs) * (2 * u.nm)
assert larger(left, right) == left
```

For the second form, mypy and Pyright reject a product expression written
directly in the call, such as `larger(nm * fs, fs * nm)`: they infer `K` from
the expression's left factor. Pyrefly and ty accept it. The whole-quantity
bound in the first form accepts direct product expressions in all four
checkers, but leaves matching kinds unchecked statically; `>` enforces that
requirement at runtime. `quantype.testing.assert_allclose` uses this same
whole-quantity pattern.

## Supported checkers

The default system parameter uses PEP 696. Stubs declare it with
`typing_extensions.TypeVar` and are tested on Python 3.12 with mypy 2.3.1,
Pyright 1.1.414, Pyrefly 1.3.2, and ty 0.0.84. Older checkers without PEP 696
support may not interpret `Length[float]` as `Length[float, NoSystem]`.
Runtime enforcement does not depend on a checker or plugin. See
[troubleshooting](../troubleshooting.md) for common system errors.

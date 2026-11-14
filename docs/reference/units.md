# Units, display units, and constants

Public imports: `from quantype import u, Unit, codata, constants` and the
quantity classes whose units you need. The [catalogue](catalogue.md) lists
built-in kinds, unit names, aliases, and relations. This page specifies units
and their presentation; [quantities](quantities.md) is authoritative for
construction, storage selection, conversion methods, comparisons, and indexing.
For the conceptual distinction, see [the physical model](../explanation/physical-model.md).

## Unit namespaces

Each kind has a namespace under its full catalogue name:
`u.length.nanometer`, `u.temperature.celsius`, and so on. Abbreviations such as
`u.nm` are directly on `u`. When a flat name clashes with a namespace, the
namespace wins: use `u.dimensionless.one`, `u.energy_density.energy_density`,
or `u.energy_per_volume.energy_per_volume`.

A unit is an immutable conversion definition with `name`, `kind`, `scale`,
`offset`, and `symbol` attributes. `scale` and `offset` describe conversion to
the reference units:

```text
reference_magnitude = magnitude * unit.scale + unit.offset
```

Units compare equal only when all their definition fields match, including the
identifier and symbol. A custom unit with the same scale but another name is
not equal to the catalogue unit. `System.unit_for(Kind) == unit` therefore tests
unit-definition equality, not just compatible dimensions or conversion factors.

Calling a unit, or multiplying a real scalar by it, constructs a quantity
without a system. For example, `2 * u.nm` and `u.nm(2)` are lengths. Use
`unit(array)` across numerical backends; `array * unit` is only guaranteed for
NumPy. Unit-first construction preserves floating storage and holds the input
array itself; Python integers become `float`. Booleans, complex values,
non-numbers, and integer arrays or backend scalars are rejected. Lists and
integer arrays need a [typed quantity constructor](quantities.md), rather than
a unit call. Offset temperature units store kelvin-scaled numbers.

## Look up a unit by name

```text
Kind.unit_named(name: str, *, units: tuple[Unit, ...] = ()) -> Unit
Kind[V, S].unit_named(name: str, *, units: tuple[Unit, ...] = ()) -> Unit
```

Lookup follows the same rules as `Kind.parse`: catalogue names and aliases
first, then the display symbols of the requested kind's units. It also reads
composed display units such as `eV/nm`. `units=` supplies explicit custom
unit definitions. A typed system target also finds units defined only by `S`.
The returned unit has the requested nominal kind; equal dimensions are not
enough. An unknown name raises `ValueError` listing that kind's units; a name
belonging to another kind raises `ValueError` naming both kinds.

```python
import pytest

from quantype import Force, Length, u

assert Force.unit_named("hartree_per_bohr") is u.hartree_per_bohr
assert Length.unit_named("nm") is u.nanometer
assert Force.unit_named("Ha/a0") is u.hartree_per_bohr
assert Force.parse("1.5 eV/nm").magnitude(u.eV_per_angstrom) == pytest.approx(0.15)
with pytest.raises(ValueError, match="Expected Force; received Length"):
    Force.unit_named("nm")
```

## Define a unit

```text
Kind.define_unit(name: str, *, reference: Unit, scale: float = 1.0,
                 offset: float = 0.0, symbol: str | None = None) -> Unit
```

The definition means `reference_magnitude = input * scale + offset`.
Composition with the reference unit gives
`new_scale = reference.scale * scale` and
`new_offset = reference.scale * offset + reference.offset`. The reference must
have the same nominal kind, not merely the same dimensions. Definitions are
immutable and checked at creation: identifiers must be nonempty, the resulting
scale finite and positive, and the resulting offset finite. Only absolute
`Temperature` units may have a nonzero offset. Invalid definitions raise
`ValueError`; a non-unit reference raises `TypeError`.

The object works immediately for construction, conversion, and serialization;
nothing is registered globally or added to `u`. Decode its identifier by passing
the definition explicitly. Duplicate definitions and reuse of built-in
identifiers are rejected when decoding; passing built-in unit objects in
`units=` is allowed. Prefer namespaced identifiers such as `my_lab:bleb`.
A temperature-difference unit needs a separate definition with a
`TemperatureDifference` reference. Follow [define units](../how-to/define-units.md)
for an executable round trip, [validation](../how-to/validate-data.md) for
Pydantic context, and [serialization](../how-to/serialize.md) for typed archives.
New physical kinds and relations require a [custom catalogue](../how-to/custom-catalogues.md),
not another unit definition.

## Display units

The [display unit](../explanation/glossary.md#display-unit "The unit used for presentation and default magnitude extraction.")
determines `.magnitude()` without an argument, `repr`, `str`, and the default
unit in serialization.

!!! info "Display selection leaves storage unchanged"

    A display unit never changes numerical storage or backend. In particular,
    `.to(System)` does not adopt that system. Use the
    [quantity conversion methods](quantities.md#conversion-and-numerical-access)
    to read numbers or change storage explicitly.

| Operation | Display rule |
| --- | --- |
| Construction, unit-first construction, `parse`, Pydantic, NPZ loading | Remember the input unit. |
| Construction with `display=False`, or `from_value` | Use the storage unit: the input unit without a system, or the system's unit. |
| Same-kind addition and subtraction | Keep the left operand's display unit. |
| Scaling, negation, `abs`, reductions, indexing | Keep the display unit, except scaling an absolute temperature drops an offset display. |
| Absolute temperature minus absolute temperature | Use the matching difference unit, such as °C → Δ°C. |
| Product, ratio, or power without a system | Compose the operands' units as described below. |
| Product, ratio, or power in a system | Use the system's unit for the result kind. |
| `.to(unit)` | Select that display unit, leaving storage unchanged. |
| `.to(System)` | Select `System.unit_for(Kind)` as display without adopting that system. |
| `.simplified()` | Simplify the display units without changing storage. |
| JAX transformations | Drop display metadata; use reference units without a system, or the system's units in one. |

`.unit` reports the unit being used; it is `None` for unnamed structural
products in a system, whose composed system units are still shown. `repr`
includes the kind and, when present, the system; `str` shows only magnitude and
unit. Format specifications apply to the magnitude: `f"{2 * u.nm:.3f}"` is
`"2.000 nm"`.

### Composed and simplified units

Without a system, products, ratios, and powers display in their operands' units:
`eV/nm`, `nm^-1`, or `Da·nm^2/ps^2`. Identical units above and below the line
cancel. A unit of the right operand that would cancel a left unit of the same
dimension converts into it: `3 eV × 2 nm / 4 Å` is `15 eV`. This unit
cancellation does not infer a new nominal physical kind.

When the composed scale matches the result kind's unit in `Atomistic`, `Metal`,
`Real`, `SI`, `CGS`, or `Atomic`, the result takes that name: `J/m` becomes `N`,
and `kcal/mol/Å` times `Å` becomes `kcal/mol`. A constant adapts to its operand's
base units, so `h·c` over `500 nm` displays in eV. Dimensionless results carry
plain numbers rather than a unit scale: `ns/fs` is `1e6`. Composed symbols parse
back, for example with `Force.parse("1.5 eV/nm")`.

`.simplified()` merges units of one dimension into the unit with the largest
power. Thus `(kcal/mol/Å)·nm` displays as `kcal/mol`, and `Pa·Å` as `J/m^2`.
A named kind takes its unit in a system containing every remaining unit:
`Da·(Å/fs)^2` becomes `eV`. `Da·nm^2/ps^2` stays composed because no built-in
system uses nm. Catalogue units such as °C are already simplest. A root, such
as `quantype.numpy.sqrt` of `nm·Å`, merges units of a dimension the same way.

```python
import pytest

from quantype import Energy, Length, Mass, Temperature, Velocity, u
from quantype.systems import Atomistic

assert repr(2 * u.nm + 5 * u.angstrom) == "Length(2.5 nm)"
assert repr(5 * u.angstrom + 2 * u.nm) == "Length(25.0 Å)"
assert repr((3 * u.eV) / (2 * u.nm)) == "Force(1.5 eV/nm)"
assert repr((2 * u.m) * (3 * u.s)) == "LengthTime(6.0 m·s)"
assert repr((1 * u.ns) / (2 * u.fs)) == "Dimensionless(500000.0)"
speed = Velocity[float](1, u.angstrom_per_fs)
kinetic = Mass[float](2, u.Da) * speed**2
assert repr(kinetic) == "Energy(2.0 Da·(Å/fs)^2)"
assert kinetic.simplified().unit is u.eV
assert kinetic.simplified().magnitude() == pytest.approx(207.2853933432161)
assert kinetic.to(Atomistic).magnitude() == kinetic.simplified().magnitude()
assert repr(30 * u.celsius - 20 * u.celsius) == "TemperatureDifference(10.0 Δ°C)"
assert repr(Length[float](2, u.nm, display=False)) == "Length(2.0 nm)"
bound = Length[float, Atomistic](2, u.nm, display=False)
assert repr(bound) == "Length(20.0 Å, Atomistic)"
assert repr(Energy[float, Atomistic](3, u.eV) / bound) == "Force(0.15 eV/Å, Atomistic)"
assert f"{2 * u.nm:.3f}" == "2.000 nm"

point = Temperature[float](20.1, u.celsius)
assert point.magnitude(u.kelvin) - 273.15 != 20.1
assert point.magnitude() == 20.1
```

A quantity built from a Python scalar remembers that scalar: `20.1 degC`
comes back as exactly `20.1`, rather than `20.100000000000023`. Any operation,
or `.to()`, forgets this input echo; `.to_system()` keeps it. It is a display
convenience, not extra numerical precision.

## Reference units and conversion precision

Conversion factors use the reference basis ångström, eV, fs, kelvin, Bohr
magneton, atom, electron, and elementary charge. Atom count, electron count, and
charge are independent [dimensions](../explanation/glossary.md#dimensions "Exponents on the physical base axes, independent of nominal meaning.").
Mass is derived as energy × time² / length²: its reference unit is eV fs²/Å²,
not the dalton used for `Atomistic` storage.

Conversions use higher working precision, then round into the requested storage;
display conversions leave the stored dtype unchanged. Lost precision in the
input cannot be recovered. The backend limits working precision, notably JAX
without x64; a factor outside its range raises `ValueError`. See
[quantity construction](quantities.md#construction-and-storage) for constructor
and copying contracts and [arrays](../how-to/arrays.md) for backend/storage choices.

JAX pytrees and the `ujax` and `utorch` derivative wrappers use reference units
without a system. Numbers in other storage units are rescaled on entry, outside
`jit`; this rounds them, but permits one trace regardless of the original input
units. See [JAX differentiation](../how-to/autodiff-jax.md) and
[Torch differentiation](../how-to/autodiff-torch.md). Importing `quantype` alone
does not import NumPy, Pydantic, JAX, or Torch.

## CODATA edition

Public import: `from quantype import codata`.

| API | Contract |
| --- | --- |
| `codata.Edition` | Literal type for `"2014"`, `"2018"`, and `"2022"`. |
| `codata.DEFAULT` | `"2022"`. |
| `codata.EDITIONS` | Included edition identifiers. |
| `codata.ENVIRONMENT_VARIABLE` | `"QUANTYPE_CODATA"`. |
| `codata.use(chosen)` | Select an edition before it is fixed; takes precedence over the environment. |
| `codata.edition()` | Return and fix the process's edition. |
| `codata.values(of=None)` | Return vendored `Codata` values; an explicit edition can be inspected without changing the process's choice. |
| `codata.is_edition(candidate)` | Test whether an edition is supported. |

!!! warning "Choose CODATA before using a unit"

    One edition applies to all units, systems, constants, and generated catalogues
    in a process. It is fixed the first time a unit is used, or `codata.edition()`
    is called. Choosing a different edition afterward raises `RuntimeError`;
    unsupported editions raise `ValueError`. Re-selecting the same edition is
    allowed.

The tables are vendored, independent of installed packages. NPZ archives record
the edition they were written with. Follow
[choose constants](../how-to/choose-constants.md) to select an edition in a fresh
process and match ASE or another code.

## Physical constants

Public import: `from quantype import constants`.

| Symbol | Meaning | Kind |
| --- | --- | --- |
| `k_B` | Boltzmann constant | `Entropy` |
| `h` | Planck constant | `Action` |
| `hbar` | Reduced Planck constant, h / 2π | `Action` |
| `e` | Elementary charge | `Charge` |
| `m_e` | Electron mass | `Mass` |
| `m_u` | Atomic mass constant, one dalton | `Mass` |
| `c` | Speed of light in vacuum | `Velocity` |
| `epsilon_0` | Vacuum electric permittivity | Structural charge² / (energy × length) |
| `k_e` | Coulomb constant, 1 / (4π ε₀) | Structural energy × length / charge² |

A `Constant` has no system or numerical backend of its own. Arithmetic with a
quantity adopts the quantity's system and numerical storage. With a quantity
without a system, the display adapts to the operand's base units. Constants
come from the chosen CODATA edition; not all physical constants are exact
measured values.

```text
constant.name                         # conventional symbol
constant.kind                         # named or structural kind
constant.dimensions                   # dimension exponents
constant.magnitude(unit) -> float
constant.to(unit) -> Quantity[..., float, NoSystem]
constant.to_system(S) -> Quantity[..., float, S]
```

With the default CODATA 2022 edition, constant representations show their symbol
and Atomistic magnitude; choosing a system produces an ordinary quantity:

```python
from quantype import constants
from quantype.systems import SI

assert repr(constants.k_B) == "Constant(k_B = 8.617333262145177e-05 eV/K)"
assert repr(constants.k_B.to_system(SI)) == "Entropy(1.380649e-23 J/K, SI)"
```

`constant.magnitude` requires a compatible unit. `to_system(NoSystem)` uses
reference units. Products, ratios, integer powers, scaling by real scalars, and
negation preserve physical algebra; noninteger powers raise `TypeError`.
Products of constants resolve named kinds using the same relations as quantities,
both at runtime and statically: `(hbar * c) / (1 * u.nm)` is `Energy`, and
`(1 * u.eV) * (2 / k_B)` is `Temperature`. Electrostatic products such as
`k_e * q1 * q2 / r` still have structural kinds, even with energy dimensions;
use `Energy.reinterpret(...)` when that is the intended meaning. See the
[constants task](../how-to/choose-constants.md) for executable calculations.

## Numerical range

Without a system, numbers stay in the units supplied, and a square is stored
in the square of its unit: a variance of energies supplied in meV is in meV².

!!! warning "System choice can underflow otherwise ordinary values"

    Choosing a system chooses the range of stored numbers too. SI underflows
    float32 for ordinary atomistic values: squared meV-scale energies are about
    2.5e-44 J², so a float32 variance comes out 1.7% wrong. Check the range before
    selecting low-precision storage.

The table comes from `System.check_range(dtype)`, which compares typical
atomistic magnitudes and their squares with the dtype's normal range.
A superscript ² means only the squares fall outside it. The section below is
owned by `quantype._internal._systems.range_table`; regenerate it rather than
editing its rows by hand.

<!-- range-table: generated by quantype._internal._systems.range_table -->
| System | float32 | float64 |
|---|---|---|
| `Atomistic` | safe | safe |
| `ASE` | safe | safe |
| `Metal` | safe | safe |
| `Real` | safe | safe |
| `SI` | Acceleration², Action², Area², Charge², DipoleMoment², ElectronDensity², EnergyPerAtom², Energy², Entropy², MagneticMoment², Mass², Momentum², ParticleDensity², Volume² | safe |
| `CGS` | Acceleration², Action², Area², DipoleMoment², ElectronDensity², MagneticMoment², Mass², Momentum², ParticleDensity², Volume² | safe |
| `Atomic` | safe | safe |

float16 cannot hold typical magnitudes of most kinds in any system. Building
float16 or float32 NumPy storage, or converting it with `.to_system`, emits
`quantype.systems.StorageRangeWarning` when a converted value overflows or
becomes subnormal. JAX and Torch traces are not checked. Use
[systems range checks](systems.md#numerical-range) for a custom system; see
[troubleshooting](../troubleshooting.md) for numerical-boundary failures.

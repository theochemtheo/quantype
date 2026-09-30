# Units, unit systems, and physical algebra

Every quantity carries three things:

| | Static | Example |
| --- | --- | --- |
| **Kind** | the quantity class | `Length`, `Pressure` |
| **Storage** | the first type parameter | `float`, `npt.NDArray[np.float64]`, `jax.Array` |
| **Unit system** | the second type parameter | `Atomistic` (default), `SI`, `Metal`, or your own |

The raw numbers in `.value` are in the coherent units of the unit system.
`Length[float]` means `Length[float, Atomistic]` and stores ångströms;
`Length[float, SI]` stores metres. Units such as `u.nm` describe input and
presentation; they never change a quantity's kind or its system.

## Construction and storage

```python
from typing import assert_type

import numpy as np
import numpy.typing as npt

from quantype import Energy, Force, Length, u
from quantype.systems import SI

length = Length[np.float64](2, u.length.nanometer)
assert_type(length, Length[np.float64])
assert length.value == 20  # ångströms, stored as a NumPy float64 scalar
assert length.magnitude() == 2  # presented in the unit it was given in

positions = Length[npt.NDArray[np.float64]]([[1, 2, 3]], u.length.nanometer)
energy = Energy[np.float64](3, u.energy.electron_volt)
assert_type(energy / length, Force[np.float64])

metres = Length[float, SI](2, u.nm)
assert metres.value == 2e-9
assert repr(metres) == "Length(2.0 nm, SI)"
```

The type arguments **convert** at construction: the storage argument selects the
numerical type, and the system argument selects the units of `.value`.
`np.float64` means a NumPy scalar; `npt.NDArray[np.float64]` means an array,
including zero-dimensional arrays. Python `float` is also supported. Scalars
reject non-scalar input; real floating storage rejects boolean, complex, and
non-numerical magnitudes.

Conversion uses wider working precision before rounding into the requested
storage; presentation conversion also widens intermediates and keeps the stored
dtype. Existing low-precision input rounding cannot be undone. Working precision
is limited by the backend (for example, JAX without x64); conversion factors
outside its range raise `ValueError`.

Units can also construct quantities directly, always in the default system:

```python
import numpy as np

from quantype import u

length = 2 * u.nm
positions = u.angstrom(np.zeros((100, 3)))
```

Unit-first construction preserves existing real floating numerical storage,
including NumPy floating scalars, where conversion allows it; Python integers
become `float`. Boolean, complex, non-numerical, and integer backend
arrays/scalars are rejected. To convert integer arrays or Python lists to
floating storage, use a typed constructor. Use `unit(array)` for reliable
cross-backend construction; backend-left multiplication is not promised outside
NumPy. For another system, write `Length[float, SI](2, u.nm)` or convert with
`.to_system(SI)`.

Hierarchical namespaces use full catalogue names, such as `u.length.nanometer`
and `u.temperature.celsius`. Flat abbreviations such as `u.nm` are also
available. Where a flat name conflicts with a namespace, the namespace wins: use
`u.dimensionless.one`, `u.energy_density.energy_density`, or
`u.energy_per_volume.energy_per_volume`.

## Unit systems

A unit system names one unit per base axis. Every kind's unit is the product of
those units raised to the kind's dimensions, so arithmetic inside one system
never converts. Import the built-in systems from `quantype.systems`:

| System | Length | Energy | Time | Magnetic moment |
| --- | --- | --- | --- | --- |
| `Atomistic` (default) | Å | eV | fs | μB |
| `Metal` (LAMMPS) | Å | eV | ps | μB |
| `Real` (LAMMPS) | Å | kcal/mol | fs | μB |
| `SI` | m | J | s | A·m² |
| `CGS` | cm | erg | s | erg/G |
| `Atomic` (Hartree) | a0 | Ha | ħ/Eₕ | eħ/mₑ |

Temperature is kelvin in every built-in system, and atom and electron counts are
1. For each built-in system and kind, `System.unit_for(Kind)` is a named
catalogue unit, such as `SI.unit_for(Pressure) is u.pascal`.

```python
import pytest

from quantype import Energy, Length, u
from quantype.systems import SI, Metal, Real

x = Length[float, SI](2, u.nm)
e = Energy[float, SI](3, u.eV)
force = e / x  # SI in, SI out: newtons
assert force.system is SI

metal = Length[float, Metal](5, u.angstrom)
with pytest.raises(TypeError, match="Cannot combine SI and Metal"):
    x + metal  # also a static type error
assert (x + metal.to_system(SI)).magnitude(u.nm) == pytest.approx(2.5)

kernel_input = (2 * u.nm).to_system(Real)  # LAMMPS real units
assert kernel_input.value == pytest.approx(20.0)
```

Systems never mix implicitly. Combining quantities from two systems is a type
error in mypy, Pyright, Pyrefly, and ty, and a `TypeError` at runtime.
`.to_system(T)` is the only bridge: it rescales the raw numbers and keeps the
display unit. Temperatures never gain an offset, because every system's
temperature base is kelvin-scaled. To define your own system, or to write code
that accepts any system, see [unit systems](unit-systems.md).

## Display units

The display unit is separate from storage. It decides what `.magnitude()`,
`repr`, and serialization show, never what `.value` means:

- Construction, unit-first construction, `parse`, Pydantic, and NPZ loading
  remember the input unit.
- Same-kind operations keep it: addition and subtraction (the left operand
  wins), scaling, negation, `abs`, reductions, and indexing.
- Point minus point gives the matching difference unit: `°C − °C → Δ°C`.
- Products, ratios, and powers fall back to the system's unit for the result.
- `.to(unit)` selects a display unit without changing the stored numbers.
- JAX transformations drop it: it is not part of the tree structure.

```python
from quantype import Temperature, u

assert repr(2 * u.nm + 5 * u.angstrom) == "Length(2.5 nm)"
assert repr(5 * u.angstrom + 2 * u.nm) == "Length(25.0 Å)"
assert repr((3 * u.eV) / (2 * u.nm)) == "Force(0.15 eV/Å)"
assert repr(30 * u.celsius - 20 * u.celsius) == "TemperatureDifference(10.0 Δ°C)"

point = Temperature[float](20.1, u.celsius)
assert point.value - 273.15 != 20.1  # 20.100000000000023 in binary64
assert point.magnitude() == 20.1  # the Python scalar it was built from
```

A quantity built from a Python scalar keeps that scalar, so `20.1 degC`
round-trips exactly instead of as `20.100000000000023`. Operations and `.to()`
drop the exact echo; `.to_system()` keeps it. `repr` names the kind, and the
system when it is not the default; `str` shows only the magnitude and unit.

## Explicit numerical boundaries

- `.value` exposes the raw numbers, in the quantity's unit system. Its static
  type says which system: a kernel annotated `Length[NDArray, Real]` receives
  `real` numbers.
- `.magnitude(unit)` converts to a compatible unit; `.magnitude()` uses the
  display unit, else the system's unit.
- `.to(unit)` selects a display unit without changing data.
- `.to_system(T)` rescales into another unit system.
- `Length[V, S].from_value(raw)` is the trusted, storage-preserving boundary for
  raw numbers already in `S`'s units. It does not inspect or convert data,
  including when used in compiled models. `Length.from_value(raw)` uses the
  default system.
- Implicit NumPy coercion is rejected rather than silently discarding meaning.

The reference units behind every conversion are those of `Atomistic`: ångström,
eV, fs, kelvin, Bohr magneton, atom, and electron. Atom and electron are
independent dimensional axes. Conversion factors come from the **installed SciPy
constants**, resolved once on first unit use. Importing `quantype` alone does not
import NumPy, SciPy, Pydantic, Torch, or JAX.

## Physical algebra

Kinds are nominal: `Pressure`, `EnergyDensity`, and `EnergyPerVolume` remain
distinct despite equal dimensions. `Force / Area` yields `Pressure`;
`Energy / Volume` yields `EnergyDensity`. Incompatible operations are rejected
both by the supported type checkers and at runtime.

Unlisted products retain structural types such as
`Quantity[Mul[LengthKind, TimeKind], float]`, not `Any`. These quantities
support arithmetic and reductions while retaining their structural kinds.
Scalar reciprocals use declared relations, such as `1 / Time` yielding
`InverseTime`, or retain a structural ratio. Structural reciprocals use the
dimensionless identity of their catalogue. For expressions mixing catalogues,
scalar reciprocals are ambiguous and raise `TypeError`; supply an explicit
dimensionless quantity numerator instead. There is no arbitrary symbolic
cancellation or inference of physical meaning from dimensions alone.

Absolute temperatures are affine points:

```python
from typing import assert_type

from quantype import Temperature, TemperatureDifference, u

cold = Temperature[float](0, u.temperature.celsius)
warm = Temperature[float](300, u.temperature.kelvin)
assert_type(warm - cold, TemperatureDifference[float])
```

Differences can be scaled; absolute temperatures cannot be added together,
multiplied, scaled, or exponentiated. Means of absolute temperatures are valid;
sums are not.

## Comparisons, indexing, and reductions

```python
import numpy as np
import numpy.typing as npt

from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([1.0, 2.0, 3.0], u.angstrom)
cutoff = Length[float](0.25, u.nm)
np.testing.assert_array_equal(positions < cutoff, [True, True, False])
assert positions[0] == 1 * u.angstrom
assert positions.shape == (3,)
assert len(positions) == 3
assert positions.max().magnitude() == 3.0
assert 1 * u.nm == 10 * u.angstrom
```

Equality compares values: it returns `bool` for scalar storage and element-wise
results for arrays, as NumPy does. Quantities of different kinds are unequal;
comparing across systems raises `TypeError`. Scalar quantities hash by value,
consistently with `==`, so they work as dict keys and in sets; quantities with
array storage are unhashable, like NumPy arrays. Ordering
comparisons require the same kind and system, statically and at runtime, and
return backend booleans.

Indexing, iteration, `.shape`, `.ndim`, `len()`, `.sum()`, `.mean()`, `.max()`,
and `.min()` keep the kind, system, and display unit. `u.sqrt(Area)`,
`u.sin(Angle)`, and `u.exp(Dimensionless)` keep the system. NumPy scalars
multiply and divide from either side. Array shape algebra and arbitrary dtype
promotion are outside the static contract. Use quantity reductions, not
`np.sum(q)` or `np.mean(q)`.

## Numerical range

The unit system is part of the type, but that does not prevent underflow. SI
storage underflows float32 for ordinary atomistic values: squared meV-scale
energies are about 2.5e-44 J², so a float32 variance comes out 1.7% wrong. This
table is generated from `System.check_range(dtype)`, which compares typical
atomistic magnitudes, and their squares, with the dtype's normal range. A
superscript ² means only the squares leave the range.

<!-- range-table: generated by quantype._internal._systems.range_table -->
| System | float32 | float64 |
|---|---|---|
| `Atomistic` | safe | safe |
| `Metal` | safe | safe |
| `Real` | safe | safe |
| `SI` | Area², ElectronDensity², EnergyPerAtom², Energy², MagneticMoment², ParticleDensity², Volume² | safe |
| `CGS` | Area², ElectronDensity², MagneticMoment², ParticleDensity², Volume² | safe |
| `Atomic` | safe | safe |

float16 cannot hold typical magnitudes of most kinds in any system. Constructing
float16 or float32 **NumPy** storage, or converting it with `.to_system`, emits a
`quantype.systems.StorageRangeWarning` when converted values overflow or become
subnormal. There is no check inside JAX or Torch traces.

## Define a unit without global registration

```python
from math import sqrt

import numpy as np

from quantype import Temperature, u

bleb = Temperature.define_unit(
    "my_lab:bleb",
    reference=u.temperature.celsius,
    scale=sqrt(2),
)
point = Temperature[np.float64](1, bleb)
assert point.magnitude(u.temperature.celsius) > 1.4

wire = point.to_dict()
restored = Temperature.parse(wire, units=(bleb,))
```

This definition means `kelvin = bleb * sqrt(2) + 273.15`: zero bleb is zero
Celsius. In general, `reference_magnitude = input * scale + offset`.
Definitions are immutable and validated once. A difference unit must be defined
explicitly using a temperature-difference reference.

A unit object can be used immediately for construction, conversion, and
serialization. Decoding its identifier requires the definition explicitly.
Namespaced identifiers avoid accidental collisions. Duplicate definitions and
shadowing built-in identifiers are rejected during decoding. Existing built-in
unit objects may also be supplied in `units=`; they do not count as shadowing.
There is no import-time mutation of the public unit namespace.

For typed restoration, see [serialization](serialization.md). To add new
physical kinds and relationships, see [custom catalogues](custom-catalogues.md).

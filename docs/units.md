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
does not convert. A system may store a few kinds in units of their own instead,
as LAMMPS does; only arithmetic producing or consuming those kinds rescales, by
a constant factor. Import the built-in systems from `quantype.systems`:

| System | Length | Energy | Time | Charge | Magnetic moment | Own units |
| --- | --- | --- | --- | --- | --- | --- |
| `Atomistic` (default) | Å | eV | fs | e | μB | |
| `Metal` (LAMMPS) | Å | eV | ps | e | μB | mass g/mol, pressure bar, density g/cm³ |
| `Real` (LAMMPS) | Å | kcal/mol | fs | e | μB | mass g/mol, pressure atm, electric field V/Å, density g/cm³ |
| `SI` | m | J | s | C | A·m² | |
| `CGS` | cm | erg | s | statC | erg/G | |
| `Atomic` (Hartree) | a0 | Ha | ħ/Eₕ | e | eħ/mₑ | |

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

`Metal` and `Real` store every quantity LAMMPS documents for `units metal` and
`units real` in LAMMPS's units, so its raw numbers wrap directly. Torque and
dynamic viscosity have no quantype kinds.

```python
import pytest

from quantype import Area, Force, Mass, Pressure, Velocity, u
from quantype.systems import Metal

assert Metal.unit_for(Pressure) is u.bar
assert Metal.unit_for(Mass) is u.gram_per_mole
stress = Force[float, Metal](1, u.eV_per_angstrom) / Area[float, Metal](
    1, u.angstrom_squared
)
assert stress.value == pytest.approx(1.602176634e6)  # bar, as LAMMPS stores it

mass = Mass[float, Metal](1, u.gram_per_mole)
speed = Velocity[float, Metal](1, u.angstrom_per_ps)
assert (mass * speed * speed).value == pytest.approx(1.0364269e-4)  # LAMMPS mvv2e
```

LAMMPS converts internally with its own constants, rounded to about seven
digits from older CODATA values (in `metal`, `boltz = 8.617343e-5`). Values it
derives with them, such as a temperature from kinetic energy, can differ from
quantype's in the seventh significant figure.

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
- `.to(unit)` selects a display unit without changing the stored numbers, and
  `.unit` reports the unit in use (`None` for unnamed products).
- Format specs apply to the magnitude: `f"{q:.3f}"` gives `2.000 nm`.
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
independent dimensional axes. Conversion factors come from **CODATA 2022** by
default, vendored with quantype, so results never depend on another installed
package. Importing `quantype` alone does not import NumPy, Pydantic, Torch, or
JAX.

CODATA 2014 and 2018 are carried too, for matching codes that use older
constants. One edition applies to the whole process, including unit systems and
generated catalogues, and it is fixed the first time a unit is used. Choose it
beforehand with `QUANTYPE_CODATA=2018` in the environment, or in code:

```python
from quantype import codata

codata.use("2022")  # before any unit is used; a later change raises
assert codata.edition() == "2022"
assert codata.values().boltzmann_constant == 1.380649e-23  # J/K
```

`codata.Edition` is the literal type of the carried editions, and
`codata.values(edition)` returns that edition's typed values. NPZ archives
record the edition they were written with.

## Physical algebra

Kinds are nominal: `Pressure`, `EnergyDensity`, and `EnergyPerVolume` remain
distinct despite equal dimensions. `Force / Area` yields `Pressure`;
`Energy / Volume` yields `EnergyDensity`. Incompatible operations are rejected
both by the supported type checkers and at runtime.

Equal dimensions never convert implicitly, but `Kind.reinterpret(q)` renames a
quantity whose dimensions match, keeping its system and storage:
`Pressure.reinterpret(energy / volume)` is a pressure, and
`Length.reinterpret(expression)` names an unnamed product. Absolute and
difference temperatures are points and vectors, so each refuses the other.

Each declared product also names the divisions that undo it: because
`Force * Length` is `Energy`, `Energy / Force` is `Length` and `Energy / Length`
is `Force`. A declared division takes precedence, so `Energy / Volume` stays
`EnergyDensity` although `Pressure * Volume` is `Energy`.

Beyond lengths, energies and times, the catalogue covers mechanics (`Mass`,
`Momentum`, `Acceleration`, `MassDensity`), electrostatics (`Charge`,
`ElectricPotential`, `ElectricField`, `DipoleMoment`), and `Entropy` and
`Action`, the kinds of the Boltzmann and Planck constants.

Unlisted products retain structural types such as
`Quantity[Mul[LengthKind, TimeKind], float]`, not `Any`. These quantities
support arithmetic and reductions while retaining their structural kinds.
Scalar reciprocals use declared relations, such as `1 / Time` yielding
`InverseTime`, or retain a structural ratio. Structural reciprocals use the
dimensionless identity of their catalogue. For expressions mixing catalogues,
scalar reciprocals are ambiguous and raise `TypeError`; supply an explicit
dimensionless quantity numerator instead. There is no arbitrary symbolic
cancellation or inference of physical meaning from dimensions alone.

Absolute temperatures are points: subtracting two gives a difference, and adding
or summing them is rejected, though their mean is valid. Every system stores
temperatures on a kelvin scale, so they multiply, divide, and scale like other
quantities:

```python
from typing import assert_type

from quantype import Energy, Temperature, TemperatureDifference, constants, u

cold = Temperature[float](0, u.temperature.celsius)
warm = Temperature[float](300, u.temperature.kelvin)
assert_type(warm - cold, TemperatureDifference[float])
assert_type(constants.k_B * warm, Energy[float])
assert repr(2 * (20 * u.celsius)) == "Temperature(586.3 K)"
```

Scaling drops a Celsius display: twice 20 °C is 586.3 K, which would read as
313.15 °C.

## Physical constants

`quantype.constants` provides `k_B`, `h`, `hbar`, `e`, `m_e`, `m_u`, `c`,
`epsilon_0`, and `k_e`, from the process's CODATA edition. A constant is exact in
every unit system, so it has none of its own: arithmetic with a quantity adopts
that quantity's system.

```python
import pytest

from quantype import Energy, Temperature, constants, u
from quantype.systems import SI, Metal

thermal = constants.k_B * Temperature[float, SI](300, u.K)
assert isinstance(thermal, Energy)
assert thermal.system is SI
assert repr(constants.k_B) == "Constant(k_B = 8.617333262145179e-05 eV/K)"
assert repr(constants.k_B.to_system(SI)) == "Entropy(1.380649e-23 J/K, SI)"
assert constants.hbar.magnitude(u.joule_second) == pytest.approx(1.0545718e-34)
assert constants.m_u.to_system(Metal).magnitude() == pytest.approx(1.0)  # g/mol
```

`epsilon_0` and `k_e` have structural kinds, as do their products:
`constants.k_e * q1 * q2 / r` has an energy's dimensions but no declared name.

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
results for arrays, as NumPy does. Quantities of different kinds or systems are
unequal, as naive and aware datetimes are. Scalar quantities hash by value,
consistently with `==`, so they work as dict keys and in sets; quantities with
array storage are unhashable, like NumPy arrays. Ordering
comparisons require the same kind and system, statically and at runtime, and
return backend booleans.

Indexing, iteration, `.shape`, `.ndim`, `len()`, `.sum()`, `.mean()`, `.max()`,
and `.min()` keep the kind, system, and display unit. `u.sqrt(Area)`,
`u.sin(Angle)`, and `u.exp(Dimensionless)` keep the system. Any real number or
numerical array multiplies and divides a quantity from either side, and the
quantity keeps its dtype; a scalar quantity scaled by an array becomes an array.
Array shape algebra and arbitrary dtype promotion are outside the static
contract. Use quantity reductions, not `np.sum(q)` or `np.mean(q)`. Builtin
`sum()` works too, because zero, alone among plain numbers, may be added to a
quantity.

`Dimensionless` values mix with plain numbers: `ratio + 1`, `ratio < 0.5`, and
`float(ratio)` work, and they print without a unit symbol.

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
| `SI` | Acceleration², Action², Area², Charge², DipoleMoment², ElectronDensity², EnergyPerAtom², Energy², Entropy², MagneticMoment², Mass², Momentum², ParticleDensity², Volume² | safe |
| `CGS` | Acceleration², Action², Area², DipoleMoment², ElectronDensity², MagneticMoment², Mass², Momentum², ParticleDensity², Volume² | safe |
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

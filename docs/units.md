# Units, unit systems, and physical algebra

Every quantity carries three things:

| | Static | Example |
| --- | --- | --- |
| **Kind** | the quantity class | `Length`, `Pressure` |
| **Storage** | the first type parameter | `float`, `npt.NDArray[np.float64]`, `jax.Array` |
| **Unit system** | the second type parameter | `Atomistic` (default), `SI`, `Metal`, or your own |

`.value` holds the raw numbers in the unit system's units: `Length[float]`,
which is `Length[float, Atomistic]`, stores ångströms, and `Length[float, SI]`
stores meters. A unit such as `u.nm` converts the input and sets how the
quantity is shown. It doesn't change the kind or the system.

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

meters = Length[float, SI](2, u.nm)
assert meters.value == 2e-9
assert repr(meters) == "Length(2.0 nm, SI)"
```

The type arguments convert the input when a quantity is built. The storage
argument chooses the numerical type, and the system argument chooses the units
of `.value`. `np.float64` gives a NumPy scalar, and `npt.NDArray[np.float64]` an
array, including zero-dimensional arrays. Python `float` works too. Scalar
storage rejects arrays, and floating storage rejects booleans, complex numbers,
and non-numbers.

Conversions run at higher precision and then round into the requested storage.
Conversions for display do the same and leave the stored dtype alone. Precision
already lost in low-precision input can't be recovered. The backend limits the
working precision (JAX without x64, for example), and a conversion factor
outside its range raises `ValueError`.

Multiplying by a unit, or calling one, also builds a quantity, always in the
default system:

```python
import numpy as np

from quantype import u

length = 2 * u.nm
positions = u.angstrom(np.zeros((100, 3)))
```

Unit-first construction keeps floating storage as it is, including NumPy
floating scalars, where the conversion allows. Python integers become `float`.
Booleans, complex numbers, non-numbers, and integer arrays or backend scalars
are rejected. For integer arrays or Python lists, use a typed constructor.
`unit(array)` works on every backend; `array * unit` is only guaranteed for
NumPy. For another system, write `Length[float, SI](2, u.nm)` or convert with
`.to_system(SI)`.

Each kind has a namespace of units under its full catalogue name, such as
`u.length.nanometer` and `u.temperature.celsius`. Abbreviations such as `u.nm`
sit directly on `u`. Where a flat name clashes with a namespace, the namespace
wins: use
`u.dimensionless.one`, `u.energy_density.energy_density`, or
`u.energy_per_volume.energy_per_volume`.

To find a unit from a name held as a string, such as one in a configuration
file, ask its kind:

```python
import pytest

from quantype import Force, Length, u

assert Force.unit_named("hartree_per_bohr") is u.hartree_per_bohr
assert Length.unit_named("nm") is u.nanometer
assert Force.unit_named("Ha/a0") is u.hartree_per_bohr  # its symbol
with pytest.raises(ValueError, match="Expected Force; received Length"):
    Force.unit_named("nm")
```

`unit_named` reads names as `parse` does. It tries catalogue names and aliases
first, then the display symbols of the kind's units, and raises `ValueError` for
a unit of another kind or a name it doesn't know, listing the kind's units in
that case. `units=` adds [custom units](#define-a-unit-without-global-registration),
and `Force[V, S].unit_named` also finds the units that only system `S` defines.

## Unit systems

A unit system is the set of units quantities store their numbers in. Storing
everything in one set gives `.value` a single meaning, keeps arithmetic free of
conversions, and lets JAX compile a function once, whatever units its inputs
were written in. The system is the second type argument. `Atomistic` is the
default, and you can choose another built-in system or
[define your own](unit-systems.md).

A system names one unit per base axis, and each kind's unit follows from its
dimensions: `Atomistic` stores forces in eV/Å because it stores energies in eV
and lengths in Å. A system can also store a few kinds in units of their own, as
`Atomistic` does for mass and LAMMPS does for pressure. Arithmetic that produces or consumes those kinds
rescales by a constant factor. The built-in systems are in `quantype.systems`:

| System | Length | Energy | Time | Charge | Magnetic moment | Own units |
| --- | --- | --- | --- | --- | --- | --- |
| `Atomistic` (default) | Å | eV | fs | e | μB | mass Da, density g/cm³ |
| `Metal` (LAMMPS) | Å | eV | ps | e | μB | mass g/mol, pressure bar, density g/cm³ |
| `Real` (LAMMPS) | Å | kcal/mol | fs | e | μB | mass g/mol, pressure atm, electric field V/Å, density g/cm³ |
| `SI` | m | J | s | C | A·m² | |
| `CGS` | cm | erg | s | statC | erg/G | |
| `Atomic` (Hartree) | a0 | Ha | ħ/Eₕ | e | eħ/mₑ | |

Temperature is in kelvin in every built-in system, and atom and electron counts
are 1. `System.unit_for(Kind)` gives the unit a system stores a kind in. For
the built-in systems it is always a named catalogue unit, as in
`SI.unit_for(Pressure) is u.pascal`.

Å, eV, and fs make the coherent mass unit eV fs²/Å², about 0.00965 Da, which
no one quotes. `Atomistic` stores masses in daltons instead, the unit ASE uses,
and densities in g/cm³:

```python
import pytest

from quantype import Mass, Velocity, u

m = Mass[float](12, u.Da)
assert m.value == 12.0  # daltons
v = Velocity[float](0.01, u.angstrom_per_fs)
kinetic = 0.5 * m * v**2  # rescaled into eV: 1 Da Å²/fs² ≈ 103.6 eV
assert kinetic.value == pytest.approx(0.0621856)
```

ASE keeps its units coherent by deriving time instead. Its time unit,
`u.ase_time` (Å √(amu/eV)), is about 10.18 fs. `atoms.get_velocities()` returns
`u.ase_velocity` (√(eV/amu)), and `atoms.get_momenta()` returns `u.ase_momentum`
(√(amu eV)), so those arrays wrap as they are:

```python
import numpy as np
import numpy.typing as npt
import pytest

from quantype import Mass, Momentum, Velocity, u

raw_velocities = np.array([[0.5, 0.0, 0.0]])  # from atoms.get_velocities()
v = Velocity[npt.NDArray[np.float64]](raw_velocities, u.ase_velocity)
assert v.magnitude(u.angstrom_per_fs)[0, 0] == pytest.approx(0.0491135)
p = Mass[float](12, u.Da) * v
assert isinstance(p, Momentum)
np.testing.assert_allclose(p.magnitude(u.ase_momentum), [[6.0, 0.0, 0.0]])
```

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

`Metal` and `Real` store each quantity that LAMMPS documents for `units metal`
and `units real` in the unit LAMMPS uses, so raw LAMMPS numbers can be wrapped
as they are. LAMMPS also lists torque and dynamic viscosity, which have no
quantype kinds.

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

LAMMPS converts internally with its own constants, which are older CODATA values
rounded to about seven digits (in `metal`, `boltz = 8.617343e-5`). Values it
derives from them, such as a temperature from kinetic energy, can differ from
quantype's in the seventh significant figure.

Combining quantities from two systems is a type error in mypy, Pyright,
Pyrefly, and ty, and a `TypeError` at runtime. Convert one with
`.to_system(T)`, which rescales the raw numbers and keeps the display unit.
Temperatures convert without an offset, because every system measures
temperature on a kelvin scale. To define your own system, or to write code that
works in any system, see [unit systems](unit-systems.md).

## Display units

The display unit sets what `.magnitude()`, `repr`, and serialization show. It
has no effect on `.value`. It is set and kept as follows:

- Construction, unit-first construction, `parse`, Pydantic, and NPZ loading
  remember the input unit. Construction with `display=False` converts in the
  same way but leaves no display unit, so the value shows in the system's unit,
  as one from `from_value` does.
- Same-kind operations keep it: addition and subtraction (the left operand
  wins), scaling, negation, `abs`, reductions, and indexing.
- Point minus point gives the matching difference unit: `°C − °C → Δ°C`.
- Products, ratios, and powers fall back to the system's unit for the result.
- `.to(unit)` selects a display unit without changing the stored numbers, and
  `.unit` reports the unit in use (`None` for unnamed products).
- Format specs apply to the magnitude: `f"{q:.3f}"` gives `2.000 nm`.
- JAX transformations drop it, because it isn't part of the pytree.

```python
from quantype import Length, Temperature, u

assert repr(2 * u.nm + 5 * u.angstrom) == "Length(2.5 nm)"
assert repr(5 * u.angstrom + 2 * u.nm) == "Length(25.0 Å)"
assert repr((3 * u.eV) / (2 * u.nm)) == "Force(0.15 eV/Å)"
assert repr(30 * u.celsius - 20 * u.celsius) == "TemperatureDifference(10.0 Δ°C)"
assert repr(Length[float](2, u.nm, display=False)) == "Length(20.0 Å)"

point = Temperature[float](20.1, u.celsius)
assert point.value - 273.15 != 20.1  # 20.100000000000023 in binary64
assert point.magnitude() == 20.1  # the Python scalar it was built from
```

A quantity built from a Python scalar remembers that scalar, so `20.1 degC`
comes back as exactly 20.1 instead of 20.100000000000023. Any operation, or
`.to()`, forgets it, and `.to_system()` keeps it. `repr` shows the kind, and
the system if it isn't the default. `str` shows only the magnitude and unit.

## Explicit numerical boundaries

- `.value` gives the raw numbers in the quantity's unit system. The static
  type says which system, so a kernel annotated `Length[NDArray, Real]` receives
  `real` numbers.
- `.magnitude(unit)` converts to a compatible unit; `.magnitude()` uses the
  display unit, else the system's unit.
- `.to(unit)` selects a display unit without changing data.
- `.to_system(T)` rescales into another unit system.
- `Length[V, S].from_value(raw)` wraps raw numbers that are already in `S`'s
  units, keeping their storage. It trusts its input and doesn't check or
  convert it. `Length.from_value(raw)` uses the default system.
- `np.asarray(q)` and other implicit conversions to arrays raise `TypeError`.

Every conversion goes through the reference units, which are `Atomistic`'s
base units: ångström, eV, fs, kelvin, Bohr magneton, atom, electron, and
elementary charge. Atom, electron, and charge are dimensions of their own. Mass
is derived, as energy time² / length², so its reference unit is eV fs²/Å², not
the dalton that `Atomistic` stores it in.
Importing `quantype` doesn't import NumPy, Pydantic, Torch, or JAX.

## CODATA edition

Conversion factors come from CODATA 2022 by default. The tables are vendored,
so results don't depend on any other installed package. CODATA 2014 and 2018
are included too, for matching codes that use older constants. One edition
applies to the whole process, including unit systems and generated catalogues,
and it is fixed the first time a unit is used. To choose another, set
`QUANTYPE_CODATA=2018` in the environment, or call `codata.use` first:

```python
from quantype import codata

codata.use("2022")  # before any unit is used; a later change raises
assert codata.edition() == "2022"
assert codata.values().boltzmann_constant == 1.380649e-23  # J/K
```

`codata.Edition` is the `Literal` type of the included editions, and
`codata.values(edition)` returns that edition's values. NPZ archives record the
edition they were written with.

## Physical algebra

Each kind is its own type, even when dimensions match. `Force / Area` is a
`Pressure`, `Energy / Volume` is an `EnergyDensity`, and one can't be used as
the other. Operations on incompatible kinds are rejected by the type checkers
and at runtime.

`Kind.reinterpret(q)` renames a quantity whose dimensions match, keeping its
system and storage: `Pressure.reinterpret(energy / volume)` is a pressure, and
`Length.reinterpret(expression)` names an unnamed product. Absolute
temperatures and temperature differences behave like points and vectors, so
neither can stand in for the other.

Each declared product also names the divisions that undo it: because
`Force * Length` is `Energy`, `Energy / Force` is `Length` and `Energy / Length`
is `Force`. A declared division takes precedence, so `Energy / Volume` stays
`EnergyDensity` although `Pressure * Volume` is `Energy`.

Beyond lengths, energies and times, the catalogue covers mechanics (`Mass`,
`Momentum`, `Acceleration`, `MassDensity`), electrostatics (`Charge`,
`ElectricPotential`, `ElectricField`, `DipoleMoment`), and `Entropy` and
`Action`, the kinds of the Boltzmann and Planck constants.

A product is named by its factors, in any order or grouping:

```python
from typing import assert_type

from quantype import Energy, Length, Mass, Momentum, Time, Velocity, u
from quantype.products import LengthTime, VelocitySquared

m = Mass[float](1, u.Da)
v = Velocity[float](2, u.angstrom_per_fs)
length = Length[float](3, u.angstrom)
t = Time[float](4, u.fs)

assert_type(0.5 * m * v**2, Energy[float])
assert_type(m * (v * v), Energy[float])
assert_type((m * v) ** 2 / m, Energy[float])
assert_type(m * length / t, Momentum[float])
assert_type(length * t, LengthTime[float])
assert_type(t * length, LengthTime[float])
assert_type(length * t / t, Length[float])
assert_type(v * v, VelocitySquared[float])
```

A product of two kinds with no relation of its own is a product class, named
after its factors in catalogue order: `LengthTime`, `EnergyPerTime`,
`VelocitySquared`, `PerLength`. `x * x` is `x ** 2` and `1 / x` is `x ** -1`, so
`t ** -1` is an `InverseTime`. A product class times or over another kind is
named when the relations that apply to its factors agree on one kind. Addition,
scaling, negation, reductions, and `to_system` keep a product class, and
`quantype.products` exports them all for annotations.

When the groupings disagree, the product stays unnamed. `force * (length /
volume)` is an `EnergyDensity` grouped as `(force * length) / volume`, and a
`Pressure` grouped as `force / (volume / length)`, so it keeps a structural type
and `reinterpret` names it. Any other unnamed product keeps a structural type
too, such as `Quantity[Mul[Mul[LengthKind, TimeKind], ChargeKind], float]` for
`length * t * charge`, and supports arithmetic and reductions like any other
quantity.

When an expression mixes catalogues, dividing a plain number by it is ambiguous
and raises `TypeError`, so divide a `Dimensionless` quantity instead.

Subtracting two absolute temperatures gives a difference. You can't add or sum
them, but you can take their mean. Every system stores temperatures on a kelvin
scale, so they multiply, divide, and scale like other quantities:

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
`epsilon_0`, and `k_e`, from the process's CODATA edition. A constant has no
unit system of its own. Combined with a quantity, it takes that quantity's
system.

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

Products of constants are named as products of quantities are, so
`(constants.hbar * constants.c) / (1 * u.nm)` is an `Energy` and
`(1 * u.eV) * (2 / constants.k_B)` a `Temperature`, at runtime and in every type
checker. `epsilon_0` and `k_e` have structural kinds, and so do their products:
`constants.k_e * q1 * q2 / r` has an energy's dimensions but no declared name.
Name it with `Energy.reinterpret(...)`.

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

`==` compares values. It returns a `bool` for scalar storage and an
element-wise result for arrays, as NumPy does. Quantities of different kinds or
systems compare unequal, as naive and aware datetimes do. Scalar quantities hash
by value, so they work as dict keys and in sets. Quantities with array storage
are unhashable, like NumPy arrays. `<` and the other orderings need the same
kind and system, statically and at runtime, and return backend booleans.
Comparing a quantity with a plain number, as in `2 * u.nm == 2`, raises
`TypeError` for `==` and the orderings alike, except for `Dimensionless`
values. `None` and other non-numbers compare unequal.

Indexing, iteration, `.shape`, `.ndim`, `.size`, `len()`, `.sum()`, `.mean()`,
`.max()`, `.min()`, `.ravel()`, and `.copy()` keep the kind, system, and display
unit. `.var()` and `qnp.square` give the squared kind, and `a @ b` multiplies
kinds as `a * b` does. With
`import quantype.numpy as qnp`, `qnp.sqrt(Area)`, `qnp.sin(Angle)`, and
`qnp.exp(Dimensionless)` keep the system. A real number or numerical array
multiplies or divides a quantity from either side, and the quantity keeps its
dtype. A scalar quantity scaled by an array becomes an array. The static types
don't track array shapes or dtype promotion. Builtin `sum()` works because zero
is the one plain number that can be added to any quantity.

NumPy's own functions apply the same unit rules at runtime, through NumPy's
dispatch. NumPy's type stubs reject quantities, so type-checked code should use
`qnp` instead. Functions with no unit rule raise:

```python
import numpy as np
import pytest

from quantype import u

positions = u.angstrom(np.array([[0.0, 0.0], [3.0, 4.0]]))
assert np.linalg.norm(positions[1]) == 5 * u.angstrom  # a Length
assert np.sum(positions) == 7 * u.angstrom
angle = u.degree(np.array(60.0))
assert float(np.cos(angle)) == pytest.approx(0.5)  # Dimensionless
with pytest.raises(TypeError, match="does not know the units"):
    np.fft.fft(positions)
```

`Dimensionless` values mix with plain numbers: `ratio + 1`, `ratio < 0.5`, and
`float(ratio)` work, and they print without a unit symbol.

## Numerical range

Choosing a unit system also chooses the range of the stored numbers. SI
underflows float32 for ordinary atomistic values: squared meV-scale energies are
about 2.5e-44 J², so a float32 variance comes out 1.7% wrong. The table below
comes from `System.check_range(dtype)`, which compares typical atomistic
magnitudes, and their squares, with the dtype's normal range. A superscript ²
means only the squares fall outside it.

<!-- range-table: generated by quantype._internal._systems.range_table -->
| System | float32 | float64 |
|---|---|---|
| `Atomistic` | safe | safe |
| `Metal` | safe | safe |
| `Real` | safe | safe |
| `SI` | Acceleration², Action², Area², Charge², DipoleMoment², ElectronDensity², EnergyPerAtom², Energy², Entropy², MagneticMoment², Mass², Momentum², ParticleDensity², Volume² | safe |
| `CGS` | Acceleration², Action², Area², DipoleMoment², ElectronDensity², MagneticMoment², Mass², Momentum², ParticleDensity², Volume² | safe |
| `Atomic` | safe | safe |

float16 can't hold typical magnitudes of most kinds in any system. Building
float16 or float32 NumPy storage, or converting it with `.to_system`, emits a
`quantype.systems.StorageRangeWarning` when a converted value overflows or
becomes subnormal. JAX and Torch traces aren't checked.

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
Definitions are immutable and checked when they're created. A difference unit
needs its own definition, with a temperature-difference reference.

A unit object can build, convert, and serialize quantities as soon as it
exists. To decode its identifier, pass the definition in. Namespaced
identifiers such as `my_lab:bleb` avoid collisions. When decoding, duplicate
definitions and definitions that reuse a built-in identifier are rejected;
passing built-in unit objects in `units=` is fine. Nothing is added to `u`.

For typed restoration, see [serialization](serialization.md). To add new
physical kinds and relationships, see [custom catalogues](custom-catalogues.md).

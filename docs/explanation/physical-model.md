# The physical model

Quantype separates **meaning**, **dimensions**, **numerical storage**, and
**presentation**. Understanding those boundaries explains why a dimensionally
valid operation can still be rejected. This page explains the model;
[quantity algebra](../reference/quantities.md#algebra-comparison-and-errors),
[display rules](../reference/units.md#display-units), and
[system adoption](../reference/systems.md#system-selection-and-adoption)
specify the public contracts.

Every quantity carries three type-level choices:

| Choice | Static representation | Example |
| --- | --- | --- |
| Kind | Quantity class or structural kind | `Length`, `Pressure` |
| Numerical storage | First parameter of a named quantity | `float`, `npt.NDArray[np.float64]`, `jax.Array` |
| Unit system | Second parameter | `NoSystem`, `Atomistic`, `ASE`, `SI`, `Metal`, or your own |

A unit such as `u.nm` describes input or display. It is not another kind, storage
backend, or system parameter. Array shapes are not part of the static type.

## Kinds are nominal meanings

`Pressure`, `EnergyDensity`, and `EnergyPerVolume` share
[dimensions](glossary.md#dimensions "Exponents on the physical base axes, independent of nominal meaning.")
but describe different
[kinds](glossary.md#kind "A nominal physical meaning, not just a dimension vector.").
`Force / Area` produces `Pressure`; `Energy / Volume` produces `EnergyDensity`.
Neither can silently stand in for the other. Declared catalogue relations, not
dimensions alone, determine result kinds. Incompatible kinds are rejected by
static checkers and at runtime.

Sometimes a different name is the intended physical interpretation:
`Pressure.reinterpret(energy / volume)` explicitly names an energy density as
pressure. Reinterpretation requires matching dimensions and preserves the
numerical storage type and system. It can also name an otherwise structural
expression, for example with `Length.reinterpret(expression)`. It is not a
general conversion between equal-dimensional meanings: absolute temperature
points and temperature-difference vectors cannot be reinterpreted as one
another.

The catalogue covers mechanics (`Mass`, `Momentum`, `Acceleration`,
`MassDensity`), electrostatics (`Charge`, `ElectricPotential`, `ElectricField`,
`DipoleMoment`), and kinds such as `Entropy` and `Action` for the Boltzmann and
Planck constants. The [generated catalogue](../reference/catalogue.md) lists
their dimensions and declared relations.

## Declared relations and structural products

A declared product also names its inverse divisions. Because
`Force * Length` is `Energy`, `Energy / Force` is `Length`, and
`Energy / Length` is `Force`. An explicitly declared division takes precedence:
`Energy / Volume` remains `EnergyDensity`, even though
`Pressure * Volume` is also `Energy`.

Factor relations can recognize a product independently of order and grouping:

```python
from typing import assert_type

from quantype import Energy, Length, Mass, Momentum, Time, Velocity, u
from quantype.products import LengthTime, VelocitySquared

mass = Mass[float](1, u.Da)
speed = Velocity[float](2, u.angstrom_per_fs)
length = Length[float](3, u.angstrom)
time = Time[float](4, u.fs)

assert_type(0.5 * mass * speed**2, Energy[float])
assert_type(mass * (speed * speed), Energy[float])
assert_type((mass * speed) ** 2 / mass, Energy[float])
assert_type(mass * length / time, Momentum[float])
assert_type(length * time, LengthTime[float])
assert_type(time * length, LengthTime[float])
assert_type(length * time / time, Length[float])
assert_type(speed * speed, VelocitySquared[float])
```

A product of two kinds without its own declared relation is a product class,
named in catalogue order: `LengthTime`, `EnergyPerTime`, `VelocitySquared`, or
`PerLength`. `x * x` is `x**2`, and `1 / x` is `x**-1`; for example,
`time**-1` is `InverseTime`. Product classes can themselves resolve to a named
kind when all applicable factor relations agree. Addition, scaling, negation,
reductions, and `.to_system` retain the product class.
`quantype.products` exports these classes for annotations.

!!! note "Ambiguous meaning stays structural"

    If groupings give different meanings, the expression stays structural.
    Consider `force * (length / volume)`. Grouping it as
    `(force * length) / volume` gives `EnergyDensity`, while
    `force / (volume / length)` gives `Pressure`. Quantype does not pick one based
    on dimensions; explicit reinterpretation supplies the intended meaning.

An otherwise unnamed expression stays structural too. A product
`length * time * charge` can have type
`Quantity[Mul[Mul[LengthKind, TimeKind], ChargeKind], float]`, retaining an
expression tree rather than inventing a kind from its dimensions. Structural
quantities support arithmetic and reductions like named quantities, but do not
have a named wire representation for serialization. There is no general
dimension-based symbolic cancellation that erases nominal meaning.

When an expression mixes generated catalogues, dividing a plain number by it
is ambiguous and raises `TypeError`. Divide a `Dimensionless` quantity instead,
so the catalogue context is explicit. See
[custom catalogues](../how-to/custom-catalogues.md) for extension boundaries.

## Presentation is not storage

Without a unit system, a quantity keeps numbers in its input unit, or a unit
composed from its operands. Addition aligns the right operand to the left
operand's storage unit. Products compose units, converting a right-hand unit
when it cancels a left-hand unit of the same dimension. Thus unit cancellation
can convert numbers without collapsing their nominal physical meanings.

The display unit is separate. `.to(unit)` changes presentation only;
`.magnitude(unit)` reads plain numbers in a requested compatible unit.
`.magnitude()` without an argument uses the display unit. This permits a result
to show as eV/nm while its numerical storage follows another valid scale.
`.simplified()` chooses simpler presentation without changing the stored data.
See [display rules](../reference/units.md#display-units) for the exact composition,
naming, scalar-echo, formatting, and JAX rules.

!!! warning "A quantity wrapper does not own a separate array"

    Numerical arrays remain mutable objects even though quantities do not support
    item assignment or in-place mutation. Construction can hold an existing array
    without copying it; reading a magnitude in its storage unit can return that
    same array. [Copy explicitly](../reference/quantities.md#construction-and-storage)
    when ownership must be separate. Implicit array coercion is rejected:
    removing physical meaning must be explicit, not an accidental effect of
    `np.asarray(q)`.

## Unit systems opt into raw numbers

`Length[float]` means `Length[float, NoSystem]`. It has no `.value`: there is
no single unit that the type could promise for its raw numbers.
`Length[float, SI]` always stores meters, so `.value` has a meaning fixed by
the static type. A
[unit system](glossary.md#unit-system "A convention fixing the units of raw stored numbers.")
is useful at a kernel or simulator boundary where raw arrays must use a
particular set of units, not a prerequisite for physical arithmetic.

A system chooses base units and derives each kind's unit from its dimensions.
Arithmetic in a coherent system needs no unit conversion. Systems can override
individual named kinds: `Atomistic` stores masses in daltons and densities in
g/cm³; LAMMPS `Metal` and `Real` store their documented simulator units, not
necessarily coherent products of the base units. Arithmetic involving an
override includes its fixed coherence factor. Multiplying raw `.value` arrays
alone would omit that factor.

`.to_system(T)` explicitly bridges storage systems. A `NoSystem` operand adopts
the system of a bound operand for arithmetic and comparisons. Two distinct
bound systems cannot be combined; convert one first. Function parameters do
not adopt: a function annotated for SI quantities does not accept an unbound
quantity without conversion. Static checkers enforce these boundaries too.
Use [system-generic functions](../reference/systems.md#write-system-generic-code)
when a library should preserve its caller's system.

Equality adopts in the same way. A quantity can equal its conversions into
several different systems, although quantities in those different bound systems
compare unequal to one another. Quantities are consequently unhashable, scalars
included: no hash could agree with that equality. Use explicit magnitudes in a
chosen unit for cache or dictionary keys. The
[quantity comparison contract](../reference/quantities.md) specifies rounding,
array booleans, zero, and ordering.

### The reference basis

The dimension basis is length, energy, time, temperature, magnetic moment,
atom count, electron count, and charge. Their reference units are ångström,
eV, fs, kelvin, Bohr magneton, atom, electron, and elementary charge.
Atom count, electron count, and charge are independent dimensions. Energy is
an axis; mass is derived as energy × time² / length². Its reference unit is
therefore eV fs²/Å², not the dalton that `Atomistic` stores.

This common reference basis makes conversions independent of display choices.
It also supplies the no-system representation for JAX pytrees and Quantype's
JAX and Torch derivatives. Inputs in other units are rescaled on entry, rounding
their numbers; for JAX this happens outside `jit`, so input presentation units
do not force separate traces. See [JAX](../how-to/autodiff-jax.md) and
[Torch](../how-to/autodiff-torch.md) differentiation.

Choosing a system also chooses the numerical range of raw values and their
squares. SI float32 can underflow for ordinary atomistic data; quantities with
no system retain their input scale instead. This is a storage tradeoff, not a
different physical algebra. Consult the [range table](../reference/units.md#numerical-range)
and backend precision cautions before choosing storage.

## Temperatures are points

Subtracting two `Temperature` points gives a `TemperatureDifference` vector.
Adding a difference shifts a point; adding or summing points is rejected.
Means are valid because they preserve the point interpretation. An offset-free
temperature unit does not make an absolute temperature a difference.

Celsius and Fahrenheit offsets affect conversion and presentation, while
storage is always kelvin-scaled. Products, ratios, powers, and scaling are
therefore meaningful: `constants.k_B * temperature` is an energy. Scaling
drops an offset display unit: twice 20 °C is 586.3 K, not 40 °C.
Follow [temperature calculations](../how-to/temperatures.md) for executable
conversions, shifts, means, and expected errors.

## Static types and runtime support

Runtime checks enforce nominal kinds and unit systems. Generated stubs express
the same algebra for mypy, Pyright, Pyrefly, and ty without plugins. Storage types
carry dtype information where the backend supports it, but not array shapes
or arbitrary dtype-promotion results. Type arguments on constructors request
real numerical conversion, not merely a label for existing data.

NumPy runtime dispatch supports quantities for operations with unit rules, but
NumPy's own stubs reject them. Use `quantype.numpy` in checked code. JAX's own
`jax.numpy` calls cannot see quantities; use the Quantype wrappers there too.
Unsupported numerical functions fail instead of silently dropping units.
See [numerical operations](../reference/numerical-operations.md) and
[arrays](../how-to/arrays.md) for supported calls and explicit backend boundaries.

!!! warning "apply does not check physical invariance"

    `q.apply(function)` is suitable only when applying the function to stored
    numbers preserves the same physical result in every unit, as sorting, slicing,
    detaching, or moving to a device does. Nothing verifies that mathematical
    invariance for you. To change backend or deliberately discard physical meaning,
    use the explicit [quantity boundaries](../reference/quantities.md#conversion-and-numerical-access).

# Troubleshooting

Start with the symptom table. Quantity errors usually identify a boundary that
needs an explicit physical decision, rather than a numerical cast. Each Python
block below runs independently; no optional backend is required.

## Common errors

| Symptom | What to do |
| --- | --- |
| `Expected Length; received Energy` | Check the input's kind and unit. An energy can't be converted to a length. See [validation](how-to/validate-data.md). |
| `Unknown unit` for a custom identifier | Pass its definition when decoding: `units=(...)` or Pydantic context. See [define units](how-to/define-units.md). |
| `'15' has no unit` | Write the unit, as in `"15 nm"`. See [wire inputs](reference/serialization.md#json-wire-objects-and-strings). |
| `Cannot add two absolute Temperatures` | Shift a temperature by a difference, as in `T + 10 * u.delta_K`. See [temperatures](how-to/temperatures.md). |
| `Gradient only defined for scalar-output functions. Output was 2.5 eV.`, or a quantity `is not a valid JAX type` | Use `ujax.grad`; native `jax.grad` can't differentiate a quantity-valued function. See [JAX differentiation](how-to/autodiff-jax.md). |
| `Units are objects, such as u.nm, not strings` | Pass a unit from `u`, or read the whole string with `Length.parse("2 nm")`. See [constructors](reference/quantities.md#construction-and-storage). |
| `u.length.nm` raises `AttributeError` | Namespaces use full names, as in `u.length.nanometer`. Abbreviations are flat, as in `u.nm`. See [units](reference/units.md). |
| `np.fft does not know the units of a quantity` | That function has no unit rule. Pass `.magnitude(unit)`; use `apply` only for a genuinely unit-invariant operation. See [unsupported calls](#unsupported-numerical-calls). |
| `Implicit array coercion drops units` | In tests use [quantype.testing.assert_allclose](how-to/test-quantities.md); elsewhere pass `.magnitude(unit)`. |
| `Length has no raw numbers without a unit system` | Read `.magnitude(unit)`, or choose a system with `.to_system(...)` first. See [raw access](#raw-value-access-without-a-system). |
| `Quantities scale by real numbers or numerical arrays` | Booleans, complex numbers, strings, and lists can't scale a quantity. Convert numerical lists with `np.asarray`; see [arrays](how-to/arrays.md). |
| `Cannot add int and Length; give it a unit` | Plain numbers can only be added to `Dimensionless` values, except integer zero is an addition identity (so `sum()` can start). Give other numbers a unit. |
| `Cannot compare Length and int; give it a unit` | Compare with a quantity, as in `q > 0 * u.nm`. Plain numbers compare with `Dimensionless`; integer zero also works with `==` / `!=` except for absolute temperatures. See [comparisons](reference/quantities.md#algebra-comparison-and-errors). |
| `The truth value of a Length quantity is ambiguous` | Compare against a value, as in `q > 0 * u.nm`, or test `q is not None`. |
| An `EnergyDensity` where a `Pressure` is expected | If the meaning really is pressure, rename it with `Pressure.reinterpret(q)`. Equal dimensions alone do not imply equal kinds; see [the physical model](explanation/physical-model.md). |
| `Cannot combine SI and Atomistic quantities` | Convert one operand with `.to_system(...)`. A quantity without a system adopts the other's. See [mixed systems](#mixed-unit-systems). |
| `Expected a quantity in SI; received one in NoSystem` from Pydantic | Fields don't adopt a system. Convert with `.to_system(SI)`, or pass a string or wire object, which decodes into the field's system. See [validation](how-to/validate-data.md#decode-into-a-chosen-unit-system). |
| `unhashable type: 'Length'` | Quantities can't be dict keys, set members, cache keys, or `jax.jit` static arguments. Use a suitable hashable representation of `q.magnitude(unit)` instead. Array magnitudes need an application-defined key too. |
| `Units only scale magnitudes` | `u.eV / u.angstrom` isn't a unit. Divide quantities, as in `(1 * u.eV) / (1 * u.angstrom)`, or use `Force.unit_named("eV/Å")`. See [units](reference/units.md). |
| `'eV/ps' is not a unit of Force` | The composed unit has other dimensions. Check its factors against [the catalogue](reference/catalogue.md). |
| `from_value trusts raw numbers in a unit system, and NoSystem has none` | Construct from a unit, as in `Length(raw, u.nm)`, or name a system: `Length.from_value(raw, SI)`. See [raw access](#raw-value-access-without-a-system). |
| An unnamed result cannot be serialized | Only named kinds have a wire representation. Name it intentionally with `reinterpret`, or declare the kind in an [application catalogue](how-to/custom-catalogues.md). |

At runtime, integer zero is an identity even for absolute temperatures:
`T + 0`, `0 + T`, and `sum([T])` return the point. This does not permit adding
two temperature points: `sum([T, T])`, `.sum()`, and `qnp.sum(T)` are rejected.
Their means are meaningful; their sums are not.

```python
from quantype import u

point = 300 * u.K
assert point + 0 is point
assert 0 + point is point
assert sum([point]) is point
try:
    sum([point, point])
except TypeError as error:
    assert "Cannot add two absolute Temperatures" in str(error)
else:
    raise AssertionError("Two absolute temperatures must not be added")
```

## Mixed unit systems

`.to(unit)` changes presentation; it does not bridge storage systems. Use
`.to_system(T)` explicitly before combining two bound systems:

```python
from quantype import Length, u
from quantype.systems import Atomistic, SI

meters = Length[float, SI](1, u.nm)
atomic = Length[float, Atomistic](1, u.nm)
try:
    meters + atomic
except TypeError as error:
    assert "Cannot combine SI and Atomistic" in str(error)
else:
    raise AssertionError("Mixed systems must fail")
assert (meters + atomic.to_system(SI)).magnitude(u.nm) == 2.0
```

A `NoSystem` literal can adopt an operand's system during arithmetic; a Pydantic
field does not adopt the system of an existing quantity instance. Choose a
boundary appropriate to your workflow:
[system contracts](reference/systems.md), [custom systems](how-to/define-systems.md),
or [validation](how-to/validate-data.md).

## Raw .value access without a system

A `NoSystem` quantity has a storage unit, not an application-wide raw numerical
convention. `.magnitude(unit)` names the unit for a consumer; `.value` requires
an explicit system:

```python
from quantype import Length, u
from quantype.systems import SI

length = Length[float](2, u.nm)
try:
    length.value
except TypeError as error:
    assert "to_system" in str(error)
else:
    raise AssertionError("Raw values require a system")
assert length.magnitude(u.nm) == 2.0
assert length.to_system(SI).value == 2e-9
```

`from_value(raw, S)` is the reverse trusted boundary, not a validator: it asserts
that `raw` is already in `S`'s units. See
[quantities](reference/quantities.md#conversion-and-numerical-access).

## Unsupported numerical calls

Use [quantype.numpy](reference/numerical-operations.md) for supported physical
rules and static types. An FFT does not have a registered unit rule, and an
implicit `np.asarray` would discard meaning:

```python
import numpy as np

from quantype import u

positions = u.angstrom(np.array([1.0, 2.0, 3.0]))
try:
    np.fft.fft(positions)
except TypeError as error:
    assert "does not know the units" in str(error)
else:
    raise AssertionError("An unsupported operation must not discard units")
raw_spectrum = np.fft.fft(positions.magnitude(u.angstrom))
assert raw_spectrum.shape == (3,)
```

The extracted result is plain numerical data. Decide its physical meaning
before constructing another quantity; complex FFT results are not supported
real floating quantity storage. For sorting, reshaping, device moves, and other
unit-invariant storage tasks, use [apply](how-to/arrays.md#apply-a-unit-invariant-operation).
Do not use `apply` to bypass an unknown physical rule.

## Unexpected named or structural kinds

Results follow declared relations, not dimension-only simplification or factor
ordering: `m * v**2` and `m * v * v` can both give `Energy`. An unnamed product
of two kinds has a product class such as `LengthTime` from `quantype.products`;
longer expressions can retain structural annotations such as
`Quantity[Mul[Mul[LengthKind, TimeKind], ChargeKind], float]`.
Read [the physical model](explanation/physical-model.md), and use
[custom catalogues](how-to/custom-catalogues.md) when the application needs a
new named meaning. For reproducible constants and units, see
[choose constants](how-to/choose-constants.md).

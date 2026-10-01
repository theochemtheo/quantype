# Getting started

This page covers building quantities, converting them, unit systems, NumPy
arrays, validating configuration, and temperatures. Each code block runs on its
own: copy the whole block into a script or REPL. None of them need JAX or Torch.

## Install

Use Python 3.12 or newer. In an existing uv project:

```bash
uv add quantype
```

Or create a virtual environment with Python's built-in tools:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install quantype
```

Run the examples with `uv run python` in a uv project, or with `python` in the
activated virtual environment.

## Construct, calculate, convert

```python
from quantype import Energy, Force, Length, u

length = Length[float](2, u.nm)
energy = Energy[float](3, u.eV)
force = energy / length

assert isinstance(force, Force)
assert length.value == 20.0  # stored in ångströms, the default system's unit
assert length.magnitude() == 2.0  # shown in the unit it was given in
assert repr(length) == "Length(2.0 nm)"
assert force.magnitude(u.eV_per_angstrom) == 0.15

shown = length.to(u.angstrom)
assert shown.value == 20.0  # .to() changes the display unit, not the stored value
assert shown.magnitude() == 20.0

wire = length.to_dict()
assert wire == {"kind": "Length", "magnitude": 2.0, "unit": "nanometer"}
restored = Length.parse(wire)
assert restored == length
```

`2 * u.nm` is a shorter way to write `Length[float](2, u.nm)`. The type argument
chooses the storage and converts the input into it: `Length[np.float64]` gives a
NumPy scalar, and `Length[npt.NDArray[np.float64]]` a NumPy array. Bare
`Length(...)` works at runtime, but strict type checkers can't tell what its
storage is. Unit-first construction takes arrays but not lists; for a list, use
a typed constructor or `u.nm(np.asarray(values))`.

Pass `.magnitude(unit)` to a library that expects a particular unit, and
`.value` to one that expects the unit system's raw numbers: ångströms, eV, and
femtoseconds by default. Either way, you get a plain float or array with no
unit attached.

## Choose a unit system

```python
from quantype import Energy, Length, u
from quantype.systems import SI, Real

x = Length[float, SI](2, u.nm)
assert x.value == 2e-9  # SI storage: metres
assert repr(x) == "Length(2.0 nm, SI)"

kernel_input = (1.0 * u.eV).to_system(Real)  # LAMMPS real units
assert round(kernel_input.value, 4) == 23.0605  # kcal/mol

try:
    x + Length[float](2, u.nm)  # SI plus the default system
except TypeError as error:
    assert "to_system" in str(error)
else:
    raise AssertionError("unit systems never mix")
```

The unit system is the second type argument. It defaults to `Atomistic`, so
`Length[float]` is `Length[float, Atomistic]`. Mixing systems is a type error
and a runtime `TypeError` whose message points to `.to_system(...)`. See
[units and unit systems](units.md#unit-systems) and
[defining your own system](unit-systems.md).

## Work with NumPy arrays

```python
import numpy as np
import numpy.typing as npt

import quantype.numpy as qnp
from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([[0, 0, 0], [3, 4, 0]], u.nm)
origin = positions[0]
displacements = positions - origin
distances = qnp.sqrt((displacements**2).sum(axis=-1))

np.testing.assert_allclose(distances.magnitude(u.nm), [0, 5])
np.testing.assert_allclose(positions.mean(axis=0).magnitude(u.nm), [1.5, 2, 0])
assert distances.shape == (2,)
np.testing.assert_array_equal(distances < 1 * u.nm, [True, False])
assert distances.max() == 5 * u.nm
```

`quantype.numpy`, imported as `qnp`, has NumPy's function names with unit rules,
and works on NumPy, JAX, and Torch arrays. `qnp.cos(angle)` is `Dimensionless`,
`qnp.arccos(ratio)` is an `Angle`, and `qnp.linalg.norm(positions, axis=-1)` is
a `Length`.

NumPy's and Torch's own functions, such as `np.cos(q)` and `np.sum(q)`, apply the
same rules at runtime. NumPy's type stubs reject quantities, so use `qnp` in
type-checked code. JAX's `jnp` functions can't see quantities at all, so use
`qnp` in JAX code too. Functions with no unit rule, such as FFTs, raise, and so
does `np.asarray(q)`. Pass `q.value` or `q.magnitude(unit)` instead.

Indexing, iteration, `len()`, `.shape`, `.max()`, `.min()`, `.reshape(...)`,
`.T`, `.squeeze()`, `.cumsum()`, and `.std()` all keep the kind. `<` and `>` need
the same kind and unit system, and return backend booleans. `==` compares
values, element-wise for arrays, and quantities of different kinds or systems
compare unequal.

For any other array operation, apply it to `.value` and give the result back
with `with_value`, which keeps the kind, unit system, and display unit:

```python
import numpy as np

from quantype import u

positions = u.nm(np.array([3.0, 1.0, 2.0]))
ordered = positions.with_value(np.sort(positions.value))
assert ordered.unit is u.nm
```

`with_value` trusts its input, as `from_value` does: the numbers must already be
in the unit system's units.

A plain number or a NumPy, JAX, or Torch array scales a quantity without
changing its dtype. Builtin `sum()`, format specs, and `reinterpret` work too:

```python
import numpy as np
import pytest

from quantype import Pressure, Time, u

step = Time[float](0.5, u.fs)
times = step * np.arange(4)
assert f"{times[-1]:.2f}" == "1.50 fs"
assert sum([1 * u.nm, 5 * u.angstrom]).magnitude(u.angstrom) == pytest.approx(15)

density = (1 * u.eV) / (1 * u.angstrom_cubed)  # an EnergyDensity
stress = Pressure.reinterpret(density)  # same dimensions, renamed
assert stress.magnitude(u.GPa) == pytest.approx(160.2176634)
```

`if q:` raises. Compare against a value, as in `q > 0 * u.nm`, or test
`q is not None`.

## Test with quantities

```python
import numpy as np

from quantype import u
from quantype.testing import assert_allclose

assert_allclose(u.nm(np.array([1.0, 2.0])), u.angstrom(np.array([10.0, 20.0])))
assert_allclose(300 * u.K, 300.05 * u.K, atol=0.1 * u.delta_K)
```

`assert_allclose` checks that both quantities have the same kind and unit
system, then compares their numbers as `numpy.testing.assert_allclose` does.
Their display units may differ. `numpy.testing` and `pytest.approx` convert
their arguments to arrays, which quantities refuse; for one number,
`q.magnitude(u.nm) == pytest.approx(2.0)` works too.

## Validate configuration and restore typed storage

```python
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ValidationError

from quantype import Length


class Configuration(BaseModel):
    cutoff: Length[float]
    positions: Length[npt.NDArray[np.float64]]


config = Configuration.model_validate(
    {
        "cutoff": "0.5 nm",
        "positions": {"kind": "Length", "magnitude": [[1, 2, 3]], "unit": "nm"},
    }
)
restored = Configuration.model_validate_json(config.model_dump_json())
assert restored.cutoff.value == 5.0
assert '"magnitude":0.5,"unit":"nanometer"' in config.model_dump_json()
assert restored.positions.value.dtype == np.float64
np.testing.assert_allclose(restored.positions.value, [[10, 20, 30]])

try:
    Configuration.model_validate({"cutoff": "1 eV", "positions": config.positions})
except ValidationError as error:
    assert "Expected Length; received Energy" in str(error)
else:
    raise AssertionError("An energy is not a length")
```

Pick an entry point by what your input is. In the table, `data` stands for your
input, and `TypeAdapter` comes from `pydantic`.

| Input and goal | Entry point |
| --- | --- |
| Numerical data with a known unit and desired storage | `Length[float](data, u.nm)` |
| The same, stored in another unit system | `Length[float, SI](data, u.nm)` |
| Trusted raw numbers, preserving storage and graphs | `Length.from_value(data)` or `Length[V, SI].from_value(data)` |
| External quantity object or scalar string, default storage | `Length.parse(data)` |
| External data with a particular storage type | `TypeAdapter(Length[np.float32]).validate_python(data)` |
| Several validated fields | A Pydantic `BaseModel` |

`Length.parse` gives a Python float or a float64 NumPy array.
`Length[float, SI].parse(...)` chooses the unit system but not the storage; for
float32, use a `TypeAdapter`. A JSON quantity object must have exactly `kind`,
`magnitude`, and `unit`. A string takes a unit's name, alias, or printed symbol,
so `"25 °C"` and `"1 eV/Å^2"` parse as printed. A number needs a unit, except for
`Dimensionless` values, which print as a bare number.

## Temperatures and custom units

```python
from math import isclose

from pydantic import TypeAdapter

from quantype import Temperature, TemperatureDifference, u

cold = Temperature[float](0, u.celsius)
warm = Temperature[float](30, u.celsius)
difference = warm - cold
assert isinstance(difference, TemperatureDifference)
assert isclose(difference.magnitude(u.delta_celsius), 30.0)

lab_degree = Temperature.define_unit("lab:degree", reference=u.celsius, scale=2.0)
point = Temperature[float](10, lab_degree)  # 20 Celsius
wire = point.to_dict()
restored = Temperature.parse(wire, units=(lab_degree,))
assert isclose(restored.magnitude(u.celsius), 20.0)

adapter = TypeAdapter(Temperature[float])
context = {"units": (lab_degree,)}
typed = adapter.validate_json(adapter.dump_json(point), context=context)
assert isclose(typed.value, point.value)
```

Subtracting two absolute temperatures gives a `TemperatureDifference`, measured
in `u.delta_celsius` or `u.delta_kelvin`. You can't add or sum absolute
temperatures, but you can take their mean, and they multiply like other
quantities: `constants.k_B * warm` is an energy.

A custom unit is an ordinary object, and nothing is registered globally. To
decode data that uses one, pass its definition again: `units=(...)` to `parse`,
or `context={"units": (...)}` to Pydantic validation, including
`model_validate_json`. Restored values keep the unit their data used.
Serialization saves values; device placement and autodiff graphs are lost.

## Common errors

| Symptom | What to do |
| --- | --- |
| `Expected Length; received Energy` | Check the input's kind and unit. An energy can't be converted to a length. |
| `Unknown unit` for a custom identifier | Pass its definition when decoding. |
| `'15' has no unit` | Write the unit, as in `"15 nm"`. |
| `Cannot add two absolute Temperatures` | Shift a temperature by a difference, as in `T + 10 * u.delta_K`. |
| `Gradient only defined for scalar-output functions. Output was 2.5 eV.`, or a quantity `is not a valid JAX type` | Use `ujax.grad`; `jax.grad` can't differentiate a quantity-valued function. |
| `Units are objects, such as u.nm, not strings` | Pass a unit from `u`, or read the whole string with `Length.parse("2 nm")`. |
| `u.length.nm` raises `AttributeError` | Namespaces use full names, as in `u.length.nanometer`. Abbreviations are flat, as in `u.nm`. |
| `np.fft does not know the units of a quantity` | That function has no unit rule. Pass `.value` or `.magnitude(unit)`. `quantype.numpy` lists the functions that have one. |
| `Implicit array coercion drops units` | In tests, compare with `quantype.testing.assert_allclose`. Elsewhere, pass `.value` or `.magnitude(unit)`. |
| `Quantities scale by real numbers or numerical arrays` | Booleans, complex numbers, strings, and lists can't scale a quantity. Convert lists with `np.asarray`. |
| `Cannot add int and Length; give it a unit` | Plain numbers can only be added to `Dimensionless` values, and zero to anything (so `sum()` works). Give the number a unit. |
| `The truth value of a Length quantity is ambiguous` | Compare against a value, as in `q > 0 * u.nm`, or test `q is not None`. |
| An `EnergyDensity` where a `Pressure` is expected | Rename it with `Pressure.reinterpret(q)`. |
| `Cannot combine SI and Atomistic quantities` | Convert one operand with `.to_system(...)`. |
| An unnamed result cannot be serialized | Only named kinds have units on the wire. Name it with `reinterpret`, or declare the kind in an [application catalogue](custom-catalogues.md). |

Result types come from declared relations, whatever the order of the factors:
`m * v**2` and `m * v * v` are both an `Energy`. A product of two kinds with no
relation is a product class such as `LengthTime` from `quantype.products`, and
a longer unnamed product keeps a structural type such as
`Quantity[Mul[Mul[LengthKind, TimeKind], ChargeKind], float]`. See
[physical algebra](units.md#physical-algebra).

## Next steps

- [Units, unit systems, and physical algebra](units.md), including physical
  constants and the CODATA edition
- [Every kind and unit, and each system's units](catalogue.md)
- [Defining unit systems and writing system-generic code](unit-systems.md)
- [JSON, Pydantic, NPZ arrays, and NPY boundaries](serialization.md)
- [JAX and Torch differentiation](autodiff.md)
  (`uv add 'quantype[jax]'` or `uv add 'quantype[torch]'`)
- [Application catalogues](custom-catalogues.md)
- [Contributor setup and checks](development.md)

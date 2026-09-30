# Getting started

quantype represents physical meaning in Python types. Units describe input and
output; arithmetic uses raw numbers in one coherent unit system, so it never
converts. It is a prototype aimed at scientific and atomistic modelling, not a
complete SI unit system.

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

Run examples with `uv run python` in a uv project, or `python` in the activated
virtual environment. Every Python block below is independent: copy the whole
block into a script or REPL. These examples need no optional backends.

## Construct, calculate, convert

```python
from quantype import Energy, Force, Length, u

length = Length[float](2, u.nm)
energy = Energy[float](3, u.eV)
force = energy / length

assert isinstance(force, Force)
assert length.value == 20.0  # raw numbers are in ångströms, the default system
assert length.magnitude() == 2.0  # shown in the unit it was given in
assert repr(length) == "Length(2.0 nm)"
assert force.magnitude(u.eV_per_angstrom) == 0.15

shown = length.to(u.angstrom)
assert shown.value == 20.0  # .to() changes presentation, not storage
assert shown.magnitude() == 20.0

wire = length.to_dict()
assert wire == {"kind": "Length", "magnitude": 2.0, "unit": "nanometer"}
restored = Length.parse(wire)
assert restored == length
```

`2 * u.nm` is a shorter construction spelling. Use `Length[float](...)` when
choosing storage explicitly; use `Length[np.float64](...)` for a NumPy scalar
and `Length[npt.NDArray[np.float64]](...)` for a NumPy array. The generic argument
converts the input into that storage. Bare `Length(...)` works at
runtime but leaves storage unspecified to strict type checkers; prefer an explicit
storage argument or unit-first construction. Unit-first construction accepts
arrays, not Python lists: use a typed constructor or `u.nm(np.asarray(values))`.

Use `.magnitude(unit)` when handing data to a library that expects a particular
unit. Use `.value` when a library expects the raw numbers of the quantity's unit
system: ångströms, eV, and femtoseconds by default. A float from either boundary
no longer carries physical meaning.

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

The unit system is the second type argument and defaults to `Atomistic`, so
`Length[float]` is `Length[float, Atomistic]`. Mixing systems is a static type
error and a runtime `TypeError`; `.to_system(...)` is the explicit bridge. See
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

`quantype.numpy`, imported as `qnp`, has NumPy's names with unit rules, and
works on NumPy, JAX, and Torch arrays: `qnp.cos(angle)` is `Dimensionless`,
`qnp.arccos(ratio)` an `Angle`, and `qnp.linalg.norm(positions, axis=-1)` a
`Length`. NumPy's and Torch's own functions apply the same rules, so `np.cos(q)`
and `np.sum(q)` work too, though their static types cannot name the result's
kind. JAX's `jnp` functions cannot see quantities; use `qnp` in JAX code.
Functions without a unit rule, such as FFTs, raise, as does `np.asarray(q)`:
use `q.value` or `q.magnitude(unit)` to cross the numerical boundary explicitly.

Indexing, iteration, `len()`, `.shape`, `.max()`, `.min()`, and comparisons keep
physical meaning. Ordering comparisons need the same kind and unit system and
return backend booleans; `==` compares values, element-wise for arrays, and
quantities of different kinds or systems are simply unequal. So do
`q.reshape(...)`, `q.T`, `q.squeeze()`, `q.cumsum()`, and `q.std()`. For other
array operations, operate on `.value` and rewrap with
`Length.from_value(...)` when the result still represents lengths in the same
system. This trusted constructor does **not** validate or convert its input.
Shape validation remains the caller's responsibility.

Any real number or numerical array (NumPy, JAX, or Torch) scales a quantity,
and the quantity keeps its dtype. Builtin `sum()`, format specs, and explicit
renaming of equal dimensions also work:

```python
import numpy as np
import pytest

from quantype import Pressure, Time, u

step = Time[float](0.5, u.fs)
times = step * np.arange(4)
assert f"{times[-1]:.2f}" == "1.50 fs"
assert sum([1 * u.nm, 5 * u.angstrom]).magnitude(u.angstrom) == pytest.approx(15)

density = (1 * u.eV) / (1 * u.angstrom_cubed)  # an EnergyDensity
stress = Pressure.reinterpret(density)  # the same dimensions, named explicitly
assert stress.magnitude(u.GPa) == pytest.approx(160.2176634)
```

`if q:` raises, because the truth of a physical value is ambiguous: compare
explicitly, or test `q is not None`.

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

Which entry point should you use? These are API patterns: `data` stands for your
input; import `TypeAdapter` from `pydantic` when using that row.

| Input and goal | Entry point |
| --- | --- |
| Numerical data with a known unit and desired storage | `Length[float](data, u.nm)` |
| The same, stored in another unit system | `Length[float, SI](data, u.nm)` |
| Trusted raw numbers, preserving storage and graphs | `Length.from_value(data)` or `Length[V, SI].from_value(data)` |
| External quantity object or scalar string, default storage | `Length.parse(data)` |
| External data with a particular storage type | `TypeAdapter(Length[np.float32]).validate_python(data)` |
| Several validated fields | A Pydantic `BaseModel` |

`Length.parse` returns Python float or NumPy float64-array storage.
`Length[float, SI].parse(...)` selects the unit system, but not the storage:
use a `TypeAdapter` to select float32 storage. A JSON quantity object must have exactly `kind`,
`magnitude`, and `unit`; a bare number does not specify an input unit.

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

Absolute temperatures are points, not differences: subtracting two gives a
`TemperatureDifference`. Use difference units (`u.delta_celsius`,
`u.delta_kelvin`) for differences. Adding or summing absolute temperatures is
rejected, but their mean is valid, and they multiply like other quantities:
`constants.k_B * warm` is an energy.

Custom units need no global registration. Supply definitions again when decoding:
`units=(...)` for direct parsing, or `context={"units": (...)}` for Pydantic
validation, including `model_validate_json`. Restored values remember the unit
their data used. Serialization preserves physical values, not device placement
or differentiation graphs.

## Common errors

| Symptom | What to do |
| --- | --- |
| `Expected Length; received Energy` | Check the input's physical kind and unit; conversion cannot turn energy into length. |
| `Unknown unit` for a custom identifier | Supply its definition at the decoding boundary. |
| `u.length.nm` raises `AttributeError` | Hierarchical namespaces use full names: `u.length.nanometer`. Abbreviations are flat: `u.nm`. |
| `np.fft does not know the units of a quantity` | That function has no unit rule; pass `.value` or `.magnitude(unit)` explicitly. `quantype.numpy` lists the functions that have one. |
| `Quantities scale by real numbers or numerical arrays` | Booleans, complex numbers, strings, and lists are not magnitudes; convert lists with `np.asarray`. |
| `Cannot add int and Length; give it a unit` | Only zero, so that `sum()` works, and `Dimensionless` values mix with plain numbers. |
| `The truth value of a Length quantity is ambiguous` | Compare explicitly, such as `q > 0 * u.nm`, or test `q is not None`. |
| An `EnergyDensity` where a `Pressure` is expected | Kinds are nominal; rename equal dimensions with `Pressure.reinterpret(q)`. |
| `Cannot combine SI and Atomistic quantities` | Convert one operand with `.to_system(...)`; quantities from different unit systems never mix. |
| An unnamed result cannot be serialized | Only named quantity kinds have wire units; keep the result in memory or define a combined application catalogue. |

`Pressure`, `EnergyDensity`, and `EnergyPerVolume` are different kinds even
though their dimensions agree. Arithmetic names only declared relationships;
it does not infer physical meaning by cancelling arbitrary dimension expressions.

## Next steps

- [Units, unit systems, and physical algebra](units.md)
- [Defining unit systems and writing system-generic code](unit-systems.md)
- [JSON, Pydantic, NPZ arrays, and NPY boundaries](serialization.md)
- [JAX and Torch differentiation](autodiff.md)
  (`uv add 'quantype[jax]'` or `uv add 'quantype[torch]'`)
- [Application catalogues](custom-catalogues.md)
- [Contributor setup and checks](development.md)

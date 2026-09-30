# Getting started

quantype represents physical meaning in Python types. Units describe input and
output; arithmetic uses canonical numerical data. It is a prototype aimed at
scientific and atomistic modelling, not a complete SI unit system.

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
assert length.value == 20.0  # canonical angstroms, not nanometers
assert length.magnitude(u.nm) == 2.0
assert force.magnitude(u.eV_per_angstrom) == 0.15

shown = length.to(u.nm)
assert shown.value == 20.0  # .to() changes presentation, not storage
assert shown.magnitude() == 2.0

wire = shown.to_dict()
assert wire == {"kind": "Length", "magnitude": 2.0, "unit": "nanometer"}
restored = Length.parse(wire)
assert restored.value == length.value
```

`2 * u.nm` is a shorter construction spelling. Use `Length[float](...)` when
choosing storage explicitly; use `Length[np.float64](...)` for a NumPy scalar
and `Length[npt.NDArray[np.float64]](...)` for a NumPy array. The generic argument
performs conversion, not just a type annotation. Bare `Length(...)` works at
runtime but leaves storage unspecified to strict type checkers; prefer an explicit
storage argument or unit-first construction. Unit-first construction accepts
arrays, not Python lists: use a typed constructor or `u.nm(np.asarray(values))`.

Use `.magnitude(unit)` when handing data to a library that expects a particular
unit. Use `.value` only when that library expects quantype's canonical units.
A float from either boundary no longer carries physical meaning.

## Work with NumPy arrays

```python
import numpy as np
import numpy.typing as npt

from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([[0, 0, 0], [3, 4, 0]], u.nm)
# Index canonical data, then explicitly restore its known physical meaning.
origin = Length.from_canonical(positions.value[0])
displacements = positions - origin
distances = u.sqrt((displacements**2).sum(axis=-1))

np.testing.assert_allclose(distances.magnitude(u.nm), [0, 5])
np.testing.assert_allclose(positions.mean(axis=0).magnitude(u.nm), [1.5, 2, 0])
assert distances.value.shape == (2,)
```

Call quantity methods such as `q.sum(axis=...)` and `q.mean(axis=...)`, not
`np.sum(q)` or `np.mean(q)`. Use `q**2` and `u.sqrt(q)`, not NumPy ufuncs on
quantities. `np.asarray(q)` is deliberately rejected: use `q.value` or
`q.magnitude(unit)` to cross the numerical boundary explicitly.

Quantities do not currently implement indexing, reshaping, or array comparison.
Operate on `.value` and rewrap with `Length.from_canonical(...)` when the result
still represents canonical lengths. This trusted constructor does **not**
validate or convert its input. For comparisons in tests, compare magnitudes in
a common unit with `np.testing.assert_allclose`; quantity `==` is not a numerical
equality check. Shape validation remains the caller's responsibility.

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
| Trusted canonical data, preserving storage and graphs | `Length.from_canonical(data)` |
| External quantity object or scalar string, default storage | `Length.parse(data)` |
| External data with a particular storage type | `TypeAdapter(Length[np.float32]).validate_python(data)` |
| Several validated fields | A Pydantic `BaseModel` |

`Length.parse` returns Python float or NumPy float64-array storage. Writing
`Length[np.float32].parse(...)` does not select float32 storage: use a
`TypeAdapter` instead. A JSON quantity object must have exactly `kind`,
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
wire = point.to(lab_degree).to_dict()
restored = Temperature.parse(wire, units=(lab_degree,))
assert isclose(restored.magnitude(u.celsius), 20.0)

adapter = TypeAdapter(Temperature[float])
context = {"units": (lab_degree,)}
typed = adapter.validate_json(adapter.dump_json(point.to(lab_degree)), context=context)
assert isclose(typed.value, point.value)
```

Absolute temperatures are points, not differences: subtracting two gives a
`TemperatureDifference`. Use difference units (`u.delta_celsius`,
`u.delta_kelvin`) for differences. Adding two absolute temperatures or scaling
one is rejected; their mean is valid but their sum is not.

Custom units need no global registration. Supply definitions again when decoding:
`units=(...)` for direct parsing, or `context={"units": (...)}` for Pydantic
validation, including `model_validate_json`. Restored values use canonical
presentation; call `.to(unit)` again when desired. Serialization preserves
physical values, not device placement or differentiation graphs.

## Common errors

| Symptom | What to do |
| --- | --- |
| `Expected Length; received Energy` | Check the input's physical kind and unit; conversion cannot turn energy into length. |
| `Unknown unit` for a custom identifier | Supply its definition at the decoding boundary. |
| `u.length.nm` raises `AttributeError` | Hierarchical namespaces use full names: `u.length.nanometer`. Abbreviations are flat: `u.nm`. |
| Implicit NumPy coercion or ufunc error | Use quantity arithmetic, `.sum()`/`.mean()`, and `u.sqrt`/`u.sin`/`u.exp`; extract magnitudes for other numerical APIs. |
| `Quantity scaling requires a real scalar` | Use a Python real scalar. Array-valued numerical operations belong at an explicit `.value` boundary. |
| An unnamed result cannot be serialized | Only named quantity kinds have wire units; keep the result in memory or define a combined application catalogue. |

`Pressure`, `EnergyDensity`, and `EnergyPerVolume` are different kinds even
though their dimensions agree. Arithmetic names only declared relationships;
it does not infer physical meaning by cancelling arbitrary dimension expressions.

## Next steps

- [Units, storage, and physical algebra](units.md)
- [JSON, Pydantic, NPZ arrays, and NPY boundaries](serialization.md)
- [JAX and Torch differentiation](autodiff.md)
  (`uv add 'quantype[jax]'` or `uv add 'quantype[torch]'`)
- [Application catalogues](custom-catalogues.md)
- [Contributor setup and checks](development.md)

# quantype

Physical meaning in Python types, with the unit system in the type too.
quantype catches invalid physical operations both statically and at runtime,
while NumPy, JAX, and Torch do the numerical work. Units describe input and
presentation; they do not change a quantity's physical kind or its storage.

## Installation and status

A prototype for scientific and atomistic modelling, not a complete SI unit system.
Requires Python 3.12+; NumPy and Pydantic v2 are core dependencies.

```bash
uv add quantype
uv add 'quantype[jax]'       # optional JAX transformations and autodiff
uv add 'quantype[torch]'     # optional Torch tensors and autograd
```

`python -m pip install quantype` also works in an activated virtual environment.

## Construction and static physical algebra

```python
from typing import assert_type
from quantype import Energy, EnergyDensity, Force, Length, Pressure, u

length = Length[float](2, u.nm)
energy = Energy[float](3, u.eV)
assert length.value == 20.0  # raw numbers in ångströms, the default system
assert length.magnitude() == 2.0  # presented in the unit it was given in

force = energy / length
assert_type(force, Force[float])
assert force.magnitude(u.eV_per_angstrom) == 0.15

pressure = force / (length**2)
density = energy / (length**3)
assert_type(pressure, Pressure[float])
assert_type(density, EnergyDensity[float])  # same dimensions, different meaning

shown = length.to(u.angstrom)
assert shown.value == length.value  # presentation only
assert shown.magnitude() == 20.0
```

The storage argument converts the input, not just its annotation. Declared
physical relationships determine result types without a type-checker plugin;
adding a length to an energy is rejected by type checkers and at runtime.

## Unit systems

```python
from typing import assert_type
from quantype import Energy, Force, Length, u
from quantype.systems import SI, Metal

x = Length[float, SI](2, u.nm)
assert x.value == 2e-9  # metres
assert_type(Energy[float, SI](3, u.eV) / x, Force[float, SI])

metal = x.to_system(Metal)  # the only bridge between systems
assert_type(metal, Length[float, Metal])
```

`Length[float]` means `Length[float, Atomistic]` (Å, eV, fs). Built-in systems
include `SI`, `CGS`, Hartree `Atomic`, and the LAMMPS `Metal` and `Real` units;
you can [define your own](docs/unit-systems.md). Systems never mix implicitly:
combining SI and metal quantities is a type error and a runtime `TypeError`.

## NumPy arrays

```python
import numpy as np
import numpy.typing as npt
from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([[0, 0, 0], [3, 4, 0]], u.nm)
distances = u.sqrt((positions**2).sum(axis=-1))
np.testing.assert_allclose(distances.magnitude(u.nm), [0, 5])
```

Use quantity arithmetic, reductions, indexing, and comparisons. For other
numerical APIs, explicitly extract `.value` (raw numbers in the unit system) or
`.magnitude(unit)`; implicit coercion is rejected.

## Differentiation with physical meaning

```python
import jax
from typing import assert_type
from quantype import Energy, Force, Length, u, ujax


def harmonic(x: Length[jax.Array]) -> Energy[jax.Array]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


x = Length[jax.Array]([1, 2, 3], u.angstrom)
force = -ujax.jit(ujax.grad(harmonic))(x)
assert_type(force, Force[jax.Array])
```

Differentiating energy with respect to length returns a **Force**, not a bare
array, including under JIT. The negative derivative is the physical force.
[Torch autograd and Hessians](docs/autodiff.md) use the same physical algebra.

## Features

- Nominal physical kinds; equal dimensions do not imply interchangeable meaning.
- Unit systems in the static type, including user-defined systems; explicit
  conversion between them and explicit numerical boundaries.
- Display units remembered from input, so `"0.5 nm"` round-trips as `0.5 nm`.
- Affine temperatures, typed reductions, and structural types for unnamed products.
- JSON, Pydantic, and pickle-free NPZ serialization.
- Local custom units and generated application catalogues, without global registration.
- Optional JAX/Torch adapters that preserve graphs during arithmetic.

## Documentation

- [Getting started](docs/getting-started.md)
- [Units, unit systems, and physical algebra](docs/units.md)
- [Defining unit systems and system-generic code](docs/unit-systems.md)
- [Serialization: JSON, Pydantic, NPZ, and NPY](docs/serialization.md)
- [JAX and Torch autodiff](docs/autodiff.md)
- [Custom catalogues](docs/custom-catalogues.md)
- [Development and conformance checks](docs/development.md)

## Development

From a checkout, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and run at the repository root. `just` and type checkers are development dependencies.

```bash
uv sync --all-extras
uv run just generate
uv run just check-generated
uv run just test
uv run just typecheck
uv run just lint
uv build
```

For core-only testing, use `uv sync` and `uv run pytest tests/runtime`.
See the [development guide](docs/development.md) for hooks and CI details.

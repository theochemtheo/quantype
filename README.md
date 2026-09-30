# quantype

Physical meaning in Python types, with canonical numerical storage. quantype
catches invalid physical operations both statically and at runtime, while NumPy,
JAX, and Torch do the numerical work. Units describe input and presentation;
they do not change a quantity's physical kind.

## Installation and status

A prototype for scientific and atomistic modelling, not a complete SI unit system.
Requires Python 3.12+; NumPy, SciPy, and Pydantic v2 are core dependencies.

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
assert length.value == 20.0  # canonical angstroms
assert length.magnitude(u.nm) == 2.0

force = energy / length
assert_type(force, Force[float])
assert force.magnitude(u.eV_per_angstrom) == 0.15

pressure = force / (length**2)
density = energy / (length**3)
assert_type(pressure, Pressure[float])
assert_type(density, EnergyDensity[float])  # same dimensions, different meaning

shown = length.to(u.nm)
assert shown.value == length.value  # presentation only
assert shown.magnitude() == 2.0
```

The storage argument converts the input, not just its annotation. Declared
physical relationships determine result types without a type-checker plugin;
adding a length to an energy is rejected by type checkers and at runtime.

## NumPy arrays

```python
import numpy as np
import numpy.typing as npt
from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([[0, 0, 0], [3, 4, 0]], u.nm)
distances = u.sqrt((positions**2).sum(axis=-1))
np.testing.assert_allclose(distances.magnitude(u.nm), [0, 5])
```

Use quantity arithmetic and reductions. For other numerical APIs, explicitly
extract `.value` (canonical data) or `.magnitude(unit)`; implicit coercion is rejected.

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
- Canonical storage with explicit conversion and numerical boundaries.
- Affine temperatures, typed reductions, and structural types for unnamed products.
- JSON, Pydantic, and pickle-free NPZ serialization.
- Local custom units and generated application catalogues, without global registration.
- Optional JAX/Torch adapters that preserve graphs during arithmetic.

## Documentation

- [Getting started](docs/getting-started.md)
- [Units, storage, and physical algebra](docs/units.md)
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

# quantype

Physical meaning in Python types, with the unit system in the type too.
quantype catches invalid physical operations both statically and at runtime,
while NumPy, JAX, and Torch do the numerical work. Units describe input and
presentation; they do not change a quantity's physical kind or its storage.

## Installation and status

Alpha, aimed at scientific and atomistic modelling: the API may change before
1.0. Requires Python 3.12+; NumPy and Pydantic v2 are the only core
dependencies.

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
assert f"{length:.1f}" == "2.0 nm"

force = energy / length
assert_type(force, Force[float])
assert force.magnitude(u.eV_per_angstrom) == 0.15
assert_type(energy / force, Length[float])  # products are undone by name

pressure = force / (length**2)
density = energy / (length**3)
assert_type(pressure, Pressure[float])
assert_type(density, EnergyDensity[float])  # same dimensions, different meaning
assert_type(Pressure.reinterpret(density), Pressure[float])  # renamed explicitly
```

The storage argument converts the input, not just its annotation. Declared
physical relationships determine result types without a type-checker plugin;
adding a length to an energy is rejected by type checkers and at runtime.

## Unit systems

```python
from typing import assert_type
from quantype import Area, Energy, Force, Length, Pressure, u
from quantype.systems import SI, Metal

x = Length[float, SI](2, u.nm)
assert x.value == 2e-9  # metres
assert_type(Energy[float, SI](3, u.eV) / x, Force[float, SI])

metal = x.to_system(Metal)  # the only bridge between systems
assert_type(metal, Length[float, Metal])

# Metal stores what LAMMPS `units metal` documents, including pressure in bar.
stress = Force[float, Metal](1, u.eV_per_angstrom) / Area[float, Metal](
    1, u.angstrom_squared
)
assert Metal.unit_for(Pressure) is u.bar
assert round(stress.value) == 1602177  # bar
```

`Length[float]` means `Length[float, Atomistic]` (Å, eV, fs). Built-in systems
include `SI`, `CGS`, Hartree `Atomic`, and LAMMPS's `Metal` and `Real`; you can
[define your own](docs/unit-systems.md). Systems never mix implicitly:
combining SI and metal quantities is a type error and a runtime `TypeError`.

## NumPy, JAX, and Torch arrays

```python
import numpy as np
import numpy.typing as npt
import quantype.numpy as qnp
from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([[0, 0, 0], [3, 4, 0]], u.nm)
distances = qnp.linalg.norm(positions, axis=-1)  # a Length
np.testing.assert_allclose(distances.magnitude(u.nm), [0, 5])
assert np.linalg.norm(positions, axis=-1).max() == 5 * u.nm  # NumPy agrees
```

`quantype.numpy` has NumPy's names with unit rules and works on NumPy, JAX, and
Torch arrays; NumPy's and Torch's own functions apply the same rules. Any real
number or array scales a quantity. For other numerical APIs, explicitly extract
`.value` (raw numbers in the unit system) or `.magnitude(unit)`; implicit
coercion is rejected.

## Differentiation with physical meaning

```python
import jax
from typing import assert_type
from quantype import Energy, Force, ForceConstant, Length, u, ujax


def harmonic(x: Length[jax.Array], k: ForceConstant[float]) -> Energy[jax.Array]:
    return 0.5 * k * (x**2).sum()


x = Length[jax.Array]([1, 2, 3], u.angstrom)
k = ForceConstant[float](2, u.eV_per_angstrom_squared)
energy, gradient = ujax.jit(ujax.value_and_grad(harmonic))(x, k)
assert_type(gradient, Force[jax.Array])
force = -gradient
```

Differentiating energy with respect to length returns a **Force**, not a bare
array, including under JIT; further arguments such as `k` pass through. The
negative derivative is the physical force. [Torch autograd, several inputs, and
Hessians](docs/autodiff.md) use the same physical algebra.

## Physical constants

```python
from typing import assert_type
from quantype import Energy, Temperature, constants, u
from quantype.systems import SI

thermal = constants.k_B * Temperature[float, SI](300, u.K)
assert_type(thermal, Energy[float, SI])  # a constant adopts its operand's system
assert repr(constants.k_B.to_system(SI)) == "Entropy(1.380649e-23 J/K, SI)"
```

Values come from CODATA 2022 by default, vendored with quantype; CODATA 2014 and
2018 are [one setting away](docs/units.md#codata-edition).

## What the catalogue covers

| Area | Kinds |
| --- | --- |
| Geometry and time | `Length`, `Area`, `Volume`, `Angle`, `Time`, `Velocity`, `Acceleration`, `Frequency`, `InverseTime` |
| Energy and stress | `Energy`, `EnergyPerAtom`, `Force`, `ForceConstant`, `Pressure`, `EnergyDensity`, `EnergyPerVolume`, `Action` |
| Mechanics | `Mass`, `MassDensity`, `Momentum` |
| Electrostatics | `Charge`, `ElectricPotential`, `ElectricField`, `DipoleMoment` |
| Thermal | `Temperature`, `TemperatureDifference`, `TemperatureRate`, `Entropy` |
| Magnetism and counts | `MagneticMoment`, `Magnetization`, `AtomCount`, `ElectronCount`, `ParticleDensity`, `ElectronDensity`, `Dimensionless` |

The [catalogue reference](docs/catalogue.md) lists every unit, the named
products and quotients, and each system's units. Application catalogues can
[add kinds and relations](docs/custom-catalogues.md).

## Features

- Nominal physical kinds; equal dimensions do not imply interchangeable meaning,
  and `Kind.reinterpret(q)` renames them explicitly.
- Unit systems in the static type, including user-defined systems and the exact
  LAMMPS `metal` and `real` units; explicit conversion between them.
- Display units remembered from input, so `"0.5 nm"` round-trips as `0.5 nm`.
- Affine temperatures that still multiply (`k_B * T`), typed reductions, and
  structural types for unnamed products.
- System-free physical constants and vendored CODATA 2014, 2018, and 2022.
- `quantype.numpy` for NumPy, JAX, and Torch, and NumPy/Torch dispatch.
- JSON, Pydantic, and pickle-free NPZ serialization.
- Local custom units and generated application catalogues, without global registration.
- JAX and Torch autodiff that returns physical kinds and preserves graphs.

## Documentation

- [Getting started](docs/getting-started.md)
- [Units, unit systems, and physical algebra](docs/units.md)
- [Kinds and units reference](docs/catalogue.md)
- [Defining unit systems and system-generic code](docs/unit-systems.md)
- [Serialization: JSON, Pydantic, NPZ, and NPY](docs/serialization.md)
- [JAX and Torch autodiff](docs/autodiff.md)
- [Custom catalogues](docs/custom-catalogues.md)
- [Development and conformance checks](docs/development.md)

## Licence

quantype is dual-licensed under the [MIT](LICENSE-MIT) and
[Apache 2.0](LICENSE-APACHE) licences, at your option.

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

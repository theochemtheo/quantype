# quantype

[![test](https://github.com/theochemtheo/quantype/actions/workflows/test.yml/badge.svg)](https://github.com/theochemtheo/quantype/actions/workflows/test.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/theochemtheo/quantype/badges/coverage.json)](https://github.com/theochemtheo/quantype/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/quantype)](https://pypi.org/project/quantype/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![SPEC 0](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CB05C)](https://scientific-python.org/specs/spec-0000/)

quantype makes physical dimensions and unit systems part of your types. mypy,
Pyright, Pyrefly, and ty check them without a plugin, and the values underneath
are still plain floats or NumPy, JAX, and Torch arrays.

```python notest
force = energy / length  # Force[float]
energy + length  # type error, and a TypeError at runtime
gradient = ujax.grad(potential)(positions)  # Force[jax.Array]
```

## Install

```bash
uv add quantype
uv add 'quantype[jax]'       # JAX transformations and autodiff
uv add 'quantype[torch]'     # Torch tensors and autograd
```

`python -m pip install quantype` also works in an activated virtual environment.
quantype needs Python 3.12 or newer, and its only core dependencies are NumPy
and Pydantic v2. It is alpha software, and the API may change before 1.0.

## Quantities and their types

```python
from typing import assert_type
from quantype import Energy, EnergyDensity, Force, Length, Pressure, u

length = Length[float](2, u.nm)
energy = Energy[float](3, u.eV)
assert repr(length + 5 * u.angstrom) == "Length(2.5 nm)"

force = energy / length
assert_type(force, Force[float])
assert repr(force) == "Force(1.5 eV/nm)"
assert force.magnitude(u.eV_per_angstrom) == 0.15
assert_type(energy / force, Length[float])

pressure = force / length**2
density = energy / length**3
assert_type(pressure, Pressure[float])
assert_type(density, EnergyDensity[float])
assert_type(Pressure.reinterpret(density), Pressure[float])
```

A quantity keeps its number in the unit it was given in. A sum converts the
right operand into the left one's unit, and a product or quotient shows its
operands' units multiplied out, such as eV/nm. `.magnitude(unit)` gives the number
in any unit you pass.

Result types come from relations declared in the catalogue. `Force * Length` is
`Energy`, so `Energy / Force` is `Length` and `Energy / Length` is `Force`. A
product is named by its factors in any order or grouping, so `m * v**2`,
`m * v * v`, and `(m * v)**2 / m` are all an `Energy`. Two kinds with no
relation multiply into a product class from `quantype.products`, such as
`LengthTime`. `Pressure` and `EnergyDensity` have the same dimensions but are
separate kinds. To turn one into the other, call `reinterpret`.

## Unit systems

A unit system is the set of units a quantity stores its numbers in, and it is
the second type argument. In a system, `.value` holds the numbers in the
system's units, so a `Length[float, SI]` always holds meters. Choose one where
you need raw numbers, such as at a kernel's boundary.

```python
from typing import assert_type
from quantype import Energy, Force, Length, Pressure, u
from quantype.systems import SI, Atomistic, Metal

length = Length[float](2, u.nm)
x = length.to_system(Atomistic)
assert_type(x, Length[float, Atomistic])
assert x.value == 20.0  # ångströms
assert repr(x) == "Length(2.0 nm, Atomistic)"  # shown in the unit it was given in

si = Length[float, SI](2, u.nm)
assert si.value == 2e-9  # meters
assert_type(Energy[float, SI](3, u.eV) / si, Force[float, SI])  # newtons
assert_type(si + length, Length[float, SI])

pressure = Pressure.from_value(1.0, Metal)  # a raw number from LAMMPS
assert pressure.magnitude(u.bar) == 1.0
```

`Length[float]` is short for `Length[float, NoSystem]`. `.to_system(...)`
converts the numbers into a system's units, and a quantity without a system
[adopts the system](https://github.com/theochemtheo/quantype/blob/main/docs/units.md#unit-systems)
of the one it's combined with, so `si + length` is in `SI`. Reading `.value`
without a system, or adding `si` to `x`, is a type error and a runtime
`TypeError`.

The built-in systems are `Atomistic` (Å, eV, and fs), `SI`, `CGS`, Hartree
`Atomic`, and LAMMPS's `Metal` and `Real`. You can also
[define your own](https://github.com/theochemtheo/quantype/blob/main/docs/unit-systems.md).
`Metal` and `Real` store each kind in the unit LAMMPS documents for it, so
`from_value` wraps raw LAMMPS numbers as they are.

## NumPy, JAX, and Torch arrays

```python
import numpy as np
import numpy.typing as npt
import quantype.numpy as qnp
from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([[0, 0, 0], [3, 4, 0]], u.nm)
distances = qnp.linalg.norm(positions, axis=-1)  # a Length
np.testing.assert_allclose(distances.magnitude(u.nm), [0, 5])
assert distances.max() == 5 * u.nm
```

`quantype.numpy`, imported as `qnp`, has NumPy's function names with unit rules,
and works on NumPy, JAX, and Torch arrays.

Any plain number or array scales a quantity. To pass data to another library,
take `.magnitude(unit)`, or `.value` in a unit system. `np.asarray(q)` raises.

## Autodiff

```python
import jax
from typing import assert_type
from quantype import Energy, Force, ForceConstant, Length, u, ujax


def spring(x: Length[jax.Array], k: ForceConstant[float]) -> Energy[jax.Array]:
    return 0.5 * k * (x**2).sum()


x = Length[jax.Array]([1, 2, 3], u.angstrom)
k = ForceConstant[float](2, u.eV_per_angstrom_squared)
energy, gradient = ujax.jit(ujax.value_and_grad(spring))(x, k)
assert_type(gradient, Force[jax.Array])
force = -gradient
```

The gradient of an `Energy` with respect to a `Length` is a `Force`, under `jit`
too. Arguments after the first, such as `k`, pass through unchanged. The
gradient is the positive derivative, so the physical force is `-gradient`.
[Torch autograd, several inputs, and Hessians](https://github.com/theochemtheo/quantype/blob/main/docs/autodiff.md) work the same
way.

## Physical constants

```python
from typing import assert_type
from quantype import Energy, Temperature, constants, u
from quantype.systems import SI

thermal = constants.k_B * Temperature[float, SI](300, u.K)
assert_type(thermal, Energy[float, SI])
assert repr(constants.k_B.to_system(SI)) == "Entropy(1.380649e-23 J/K, SI)"
```

Constants have no unit system, so they adopt the system of the quantity they
combine with: `k_B` times an SI temperature is an SI energy.
Values are from CODATA 2022, vendored with quantype. CODATA 2014 and 2018 are
[one setting away](https://github.com/theochemtheo/quantype/blob/main/docs/units.md#codata-edition).

## Built-in kinds

| Area | Kinds |
| --- | --- |
| Geometry and time | `Length`, `Area`, `Volume`, `Angle`, `Time`, `Velocity`, `Acceleration`, `Frequency`, `InverseTime` |
| Energy and stress | `Energy`, `EnergyPerAtom`, `Force`, `ForceConstant`, `Pressure`, `EnergyDensity`, `Action` |
| Mechanics | `Mass`, `MassDensity`, `Momentum` |
| Electrostatics | `Charge`, `ElectricPotential`, `ElectricField`, `DipoleMoment` |
| Thermal | `Temperature`, `TemperatureDifference`, `TemperatureRate`, `Entropy` |
| Magnetism and counts | `MagneticMoment`, `Magnetization`, `AtomCount`, `ElectronCount`, `ParticleDensity`, `ElectronDensity`, `Dimensionless` |

The [catalogue reference](https://github.com/theochemtheo/quantype/blob/main/docs/catalogue.md) lists every unit, the named
products and quotients, and each system's units. You can
[generate your own catalogue](https://github.com/theochemtheo/quantype/blob/main/docs/custom-catalogues.md) with new kinds,
relations, and units, or
[define a unit](https://github.com/theochemtheo/quantype/blob/main/docs/units.md#define-a-unit-without-global-registration)
for an existing kind without one.

## Also included

- `quantype.testing.assert_allclose`, which checks kind and unit system before
  comparing numbers.
- JSON, Pydantic, and NPZ serialization, with no pickle.
- Temperatures in °C and °F are points: subtracting two gives a
  `TemperatureDifference`, and adding two is a type error.

## Documentation

- [Getting started](https://github.com/theochemtheo/quantype/blob/main/docs/getting-started.md)
- [Units, unit systems, and physical algebra](https://github.com/theochemtheo/quantype/blob/main/docs/units.md)
- [Kinds and units reference](https://github.com/theochemtheo/quantype/blob/main/docs/catalogue.md)
- [Defining unit systems and system-generic code](https://github.com/theochemtheo/quantype/blob/main/docs/unit-systems.md)
- [Serialization: JSON, Pydantic, NPZ, and NPY](https://github.com/theochemtheo/quantype/blob/main/docs/serialization.md)
- [JAX and Torch autodiff](https://github.com/theochemtheo/quantype/blob/main/docs/autodiff.md)
- [Custom catalogues](https://github.com/theochemtheo/quantype/blob/main/docs/custom-catalogues.md)
- [Development and conformance checks](https://github.com/theochemtheo/quantype/blob/main/docs/development.md)

## License

quantype is dual-licensed under the [MIT](https://github.com/theochemtheo/quantype/blob/main/LICENSE-MIT) and
[Apache 2.0](https://github.com/theochemtheo/quantype/blob/main/LICENSE-APACHE) licenses, at your option.

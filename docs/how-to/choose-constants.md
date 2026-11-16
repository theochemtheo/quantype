# Choose constants and a CODATA edition

Use this when matching a simulator's conversion factors or calculating with
physical constants. Quantype vendors CODATA 2014, 2018, and 2022 and defaults
to 2022; results do not depend on another installed package. The
[unit reference](../reference/units.md#codata-edition) lists the exact edition
and constant APIs.

## Choose the edition before using a unit

One edition applies to the entire process, including custom generated
catalogues, unit systems, and constants. Set the environment before starting
your program:

```sh
QUANTYPE_CODATA=2018 python simulate.py
```

!!! warning "Choose once per process"
    Alternatively, make `codata.use("2018")` the first configuration step in
    Python, before any unit is used. Asking `codata.edition()` also fixes the
    choice. Changing to a different edition afterward raises `RuntimeError`;
    restart the process instead. An explicit `use` choice takes precedence
    over the environment.

This example starts a fresh interpreter so it is independent of units used by
other examples in the surrounding process:

```python
import subprocess
import sys

program = """
from quantype import codata

codata.use("2018")
from quantype import u

assert codata.edition() == "2018"
assert u.bohr.scale > 0
try:
    codata.use("2014")
except RuntimeError as error:
    assert "already fixed at 2018" in str(error)
else:
    raise AssertionError("A process must not mix CODATA editions")
"""
subprocess.run([sys.executable, "-c", program], check=True)
```

To inspect another edition without changing the active choice, pass it directly
to `codata.values`. Values in that record use the documented SI units:

```python
from quantype import codata

assert codata.is_edition("2022")
assert not codata.is_edition("2010")
assert codata.values("2022").boltzmann_constant == 1.380649e-23  # J/K
assert codata.values("2014").boltzmann_constant != 1.380649e-23
```

`ase.units` uses CODATA 2014. Choose `QUANTYPE_CODATA=2014` to match ASE's
values; the 2014 and 2022 ASE time units differ by about 5 parts in 10⁹.
LAMMPS instead uses older constants rounded to roughly seven digits, so its
derived temperatures can still differ in the seventh significant figure.
See the [built-in system reference](../reference/systems.md#lammps-metal-and-real).
NPZ archives record the CODATA edition used when written; see
[serialization](serialize.md).

## Let a constant adopt your quantity's system

Do not manually convert a physical constant into an array's raw units. A
constant has no system of its own; arithmetic adopts the quantity's system and
storage, including NumPy, JAX, or Torch storage. The same formula therefore works
for SI, a simulator system, or no system.

```python
from typing import assert_type

import pytest

from quantype import Energy, Temperature, constants, u
from quantype.systems import SI, Metal

thermal = constants.k_B * Temperature[float, SI](300, u.K)
assert_type(thermal, Energy[float, SI])
assert thermal.system is SI
assert thermal.magnitude(u.J) == pytest.approx(300 * 1.380649e-23)
assert constants.k_B.to_system(SI).magnitude(u.joule_per_kelvin) == 1.380649e-23
assert constants.hbar.magnitude(u.joule_second) == pytest.approx(1.0545718e-34)
assert constants.m_u.to_system(Metal).magnitude() == pytest.approx(1.0)  # g/mol
```

For a specific presentation use `constant.to(unit)`; for raw system numbers use
`constant.to_system(System)`. These produce ordinary quantities with `float`
storage. Constants themselves do not have a `.value` or a chosen backend.

## Convert kinetic energy to a temperature

The Atomistic system stores masses in daltons and velocities in Å/fs, but
products account for the non-coherent mass unit. Divide the resulting energy
by `k_B` rather than combining raw values with a hand-written conversion factor.

```python
from typing import assert_type

import pytest

from quantype import Energy, Mass, Temperature, Velocity, constants, u
from quantype.systems import Atomistic

mass = Mass[float, Atomistic](12, u.Da)
speed = Velocity[float, Atomistic](0.01, u.angstrom_per_fs)
kinetic = 0.5 * mass * speed**2
assert_type(kinetic, Energy[float, Atomistic])
assert kinetic.magnitude(u.eV) == pytest.approx(0.0621856)
equivalent = kinetic / constants.k_B
assert_type(equivalent, Temperature[float, Atomistic])
assert equivalent.magnitude(u.K) == pytest.approx(721.634)
```

This is an energy expressed as an equivalent temperature, not a claim about a
particular kinetic-temperature estimator or its degrees of freedom.

## Combine constants across dimensions

Products of constants follow the same declared algebra as quantities. For
example, `hbar * c` divided by a length is an energy, and energy times
`2 / k_B` is a temperature. Both runtime and the supported static checkers
recognize those named results.

```python
from typing import assert_type

import pytest

from quantype import Energy, Temperature, constants, u

photon = (constants.h * constants.c) / (500 * u.nm)
assert_type(photon, Energy[float])
assert photon.magnitude(u.eV) == pytest.approx(2.47968397)
thermal_scale = (1 * u.eV) * (2 / constants.k_B)
assert_type(thermal_scale, Temperature[float])
assert thermal_scale.magnitude(u.K) == pytest.approx(23209.036)
```

The permittivity `epsilon_0` and Coulomb constant `k_e` have structural kinds.
Their electrostatic products may have energy dimensions without a declared
energy name. Explicitly reinterpret that meaning when appropriate:

```python
import pytest

from quantype import Energy, constants, u

q1 = 1 * u.elementary_charge
q2 = 1 * u.elementary_charge
separation = 1 * u.nm
interaction = Energy.reinterpret(constants.k_e * q1 * q2 / separation)
assert interaction.magnitude(u.eV) == pytest.approx(1.4399645)
```

Reinterpretation preserves the physical dimensions, numerical storage, and
system; it does not make arbitrary kinds interchangeable. See
[the physical model](../explanation/physical-model.md#kinds-are-nominal-meanings)
and the [catalogue](../reference/catalogue.md) for the declared relations.

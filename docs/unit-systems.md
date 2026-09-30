# Unit systems

A unit system decides what a quantity's raw numbers mean. The built-in systems
and the rules for mixing them are described in
[units and unit systems](units.md#unit-systems). This page shows how to define
your own system, and how to write library code that works in any system.

## Define a unit system

Subclass `UnitSystem`, give it a `name=`, and name a base unit for each axis you
need. Nothing is registered globally: the class is usable as soon as it is
defined, as custom units are.

```python
from scipy.constants import N_A, kilo

from quantype import Energy, u
from quantype.systems import UnitSystem

# GROMACS uses kJ/mol, which is not a catalogue unit: define it first.
kJ_per_mol = Energy.define_unit(
    "gromacs:kJ_per_mol", reference=u.joule, scale=kilo / N_A, symbol="kJ/mol"
)


class Gromacs(UnitSystem, name="gromacs"):
    length = u.nanometer
    energy = kJ_per_mol
    time = u.picosecond
    # Temperature (K), magnetic moment (μB) and the atom/electron counts are
    # inherited from UnitSystem's defaults unless overridden.
```

Using it looks the same as using a built-in system:

```python
import numpy as np
import numpy.typing as npt
from scipy.constants import N_A, kilo

from quantype import Energy, Force, Length, u
from quantype.systems import Atomistic, UnitSystem

kJ_per_mol = Energy.define_unit(
    "gromacs:kJ_per_mol", reference=u.joule, scale=kilo / N_A, symbol="kJ/mol"
)


class Gromacs(UnitSystem, name="gromacs"):
    length = u.nanometer
    energy = kJ_per_mol
    time = u.picosecond


cutoff = Length[float, Gromacs](1.2, u.nm)
assert cutoff.value == 1.2  # raw numbers are in nanometres

raw = np.zeros((100, 3), dtype=np.float32)  # from a GROMACS kernel
forces = Force[npt.NDArray[np.float32], Gromacs].from_value(raw)  # kJ/mol/nm
in_ev_per_angstrom = forces.to_system(Atomistic)  # the explicit bridge
```

The class is checked when it is defined, so mistakes surface at import rather
than mid-calculation:

| Rule | Example error |
| --- | --- |
| `name=` is required, and unique within the class hierarchy. Namespaced names (`"mylab:gromacs"`) are encouraged, as for units. | `UnitSystem subclass requires name=...` |
| `length`, `energy`, and `time` are required; the other axes have defaults. | `Gromacs is missing a time unit` |
| Each base unit has the kind of its axis. | `Gromacs.energy must be an Energy unit; received Length (nanometer)` |
| The temperature base is kelvin-scaled, because affine units (°C) aren't coherent. | `Temperature base must not have an offset; use kelvin` |
| Every derived scale for every catalogue kind is finite and non-zero in float64. | `Huge gives Area a scale that overflows float64` |

A system is used as a class, never instantiated. The class name does not appear
on the wire, so renaming the class never changes stored data; only `name=`
does.

### Derived units

Each kind's unit is derived from the base units. When the result matches a named
catalogue unit, it uses that unit: `Gromacs.unit_for(Length) is u.nanometer`.
Otherwise it gets a system-qualified identifier, such as `gromacs:Force` with
symbol `kJ/mol/nm`. These appear in `repr`, and in serialized data when a value
has no display unit.

A system is defined over the base axes, not over a list of kinds. A new kind
from a [generated catalogue](custom-catalogues.md), such as `SurfaceTension`,
therefore gets a unit in every system automatically: `J/m^2` in SI, and
`gromacs:SurfaceTension` in `Gromacs`.

### Decoding

Decoding follows the rule quantype uses for custom units: supply the definitions
at the boundary. `System.units` holds every unit a system defines that the
catalogue cannot name, so one argument covers them. These lines are API
patterns: `data` and `text` stand for your input, and the imports are as above.

```python notest
Force.parse(data, units=Gromacs.units)
TypeAdapter(Force[float, Gromacs]).validate_json(text)
load_npz("frame.npz", "forces", Force[npt.NDArray[np.float32], Gromacs])
```

When the target type names the system, as in the last two lines, its units are
found automatically, so `units=` is only required for untyped entry points or
for decoding into another system. Wire data never records a system: an archive
written in `Gromacs` decodes into `Atomistic`, `SI`, or any other system.

### Inheriting to vary one axis

Inheriting gives a new, separate system:

```python
from quantype import u
from quantype.systems import Metal


class MetalFs(Metal, name="metal-fs"):
    time = u.femtosecond
```

The system parameter is invariant, so `Length[float, MetalFs]` is **not** a
`Length[float, Metal]`, and the runtime same-system check agrees. A subclass
shares definitions with its parent, not storage meaning. Python users usually
expect a subclass to be accepted where its parent is; here it is not, and mixing
the two raises `TypeError` naming both systems.

One static gap remains. mypy accepts `metal + x.to_system(MetalFs)` written
inline: it infers the conversion's target from the expected operand type, and
`type[MetalFs]` is also a `type[Metal]`. Pyright, Pyrefly, and ty reject it, and
so does mypy once the converted value is bound to a name. The runtime check
rejects the mixed operation in every case.

### Numerical range

A custom system can be as risky as SI in float32. `System.check_range(dtype)`
reports kinds whose typical atomistic magnitudes, or their squares, would
underflow or overflow:

```python
import numpy as np

from quantype.systems import SI

for issue in SI.check_range(np.float32):
    print(issue)  # SI: Energy (1.6e-23 J)² would underflow float32, ...
```

The [range table](units.md#numerical-range) for built-in systems comes from the
same check.

## Write system-generic code

Library code usually should not choose a unit system for its callers. Annotate
with a type parameter bounded by `UnitSystem`:

```python
from quantype import Energy, Force, Length, u
from quantype.systems import SI, Metal, UnitSystem


def work[S: UnitSystem](f: Force[float, S], d: Length[float, S]) -> Energy[float, S]:
    return f * d


si = work(Force[float, SI](1, u.newton), Length[float, SI](2, u.meter))
metal = work(Force[float, Metal](1, u.eV_per_angstrom), Length[float, Metal](2, u.nm))
```

The checker accepts SI pairs and metal pairs, and rejects a mixed pair. Inside
the function, arithmetic needs no conversion: every operand is in `S`.

Annotations without a system, such as `Length[float]`, mean the default system.
A function typed `Length[float]` rejects SI quantities; the error message names
`.to_system(...)` as the fix. Prefer the generic form in shared libraries, and
the explicit form (`Length[npt.NDArray[np.float64], Real]`) for kernels whose raw
numbers must be in particular units.

`ujax.grad`, `ujax.hessian`, and `utorch.grad` are generic in the same way: a
function from `Length[Array, S]` to `Energy[Array, S]` differentiates to
`Force[Array, S]`. See [autodiff](autodiff.md).

## Supported checkers

The unit-system parameter has a default (PEP 696). The stubs declare it with
`typing_extensions.TypeVar`, which is tested on Python 3.12 with mypy 2.3.1,
Pyright 1.1.414, Pyrefly 1.3.2, and ty 0.0.84. Older checkers without PEP 696
support may not treat `Length[float]` as `Length[float, Atomistic]`.

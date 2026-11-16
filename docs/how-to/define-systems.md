# Define a simulator's unit system

Use this when a kernel or simulator exposes raw numbers in a consistent
[unit system](../explanation/glossary.md#unit-system "A convention fixing the units of raw stored numbers.").
Start from the kernel's specification of those units.

First check the [built-in systems](../reference/systems.md#built-in-systems),
including ASE and LAMMPS Metal/Real. Use a custom system only when the existing
ones do not describe your raw numbers. The [system reference](../reference/systems.md)
specifies validation, derived units, invariance, and system-generic code;
[quantities](../reference/quantities.md) owns construction and conversion basics.

## Name the base units and exceptional kinds

Subclass `UnitSystem`, provide `name=`, and set `length`, `energy`, and `time`.
Other axes inherit defaults: kelvin, elementary charge, Bohr magneton, atom,
and electron. A custom `Kind.define_unit` is also a valid base unit.

GROMACS uses nanometers, kJ/mol, and picoseconds, but reports pressure in bar
rather than the coherent kJ/mol/nm³. Name that exception in `overrides`:

```python
import numpy as np

from quantype import Force, Length, Pressure, u
from quantype.systems import Atomistic, UnitSystem


class Gromacs(UnitSystem, name="gromacs"):
    length = u.nanometer
    energy = u.kJ_per_mol
    time = u.picosecond
    overrides = (u.bar,)


cutoff = Length[float, Gromacs](1.2, u.nm)
assert cutoff.value == 1.2
assert Gromacs.unit_for(Length) is u.nanometer
assert Gromacs.unit_for(Pressure) is u.bar
raw = np.zeros((100, 3), dtype=np.float32)  # from a GROMACS kernel
forces = Force.from_value(raw, Gromacs)  # raw kJ/mol/nm, not a conversion
assert forces.value is raw
in_ev_per_angstrom = forces.to_system(Atomistic)
assert in_ev_per_angstrom.system is Atomistic
assert in_ev_per_angstrom.value.dtype == np.float32
```

!!! warning "`from_value` trusts the kernel's units"
    `from_value` checks that storage is accepted raw real/floating data and the
    target is a bound `UnitSystem`. It retains the exact input without copying
    or coercion, but cannot check that the numbers have the claimed physical
    units. Use it only for numbers already in the system's units. For other
    units, use [typed construction](../reference/quantities.md#construction-and-storage)
    or [`.to_system`](../reference/quantities.md#to_system) to perform conversions.

Systems are classes, not instances, and nothing is registered globally.
Prefer a namespaced system name, such as `my_lab:gromacs`, in a shared application.

Arithmetic producing or consuming an overridden kind includes its fixed
rescaling factor. Do not multiply `.value` arrays and assume the result uses
the target kind's units when an override is involved.

## Decode system-defined units

When derived units match a catalogue unit, `unit_for` reuses it. Otherwise the
unit identifier is qualified by the system name: `gromacs:Force`, for example,
with symbol `kJ/mol/nm`. `System.units` supplies every non-catalogue definition
needed to decode those identifiers.

Pass it explicitly to untyped parsing or when decoding into another system.
A typed target that names the original system finds its definitions
automatically. This example exercises each boundary with complete inputs:

```python
import json

import numpy as np
import numpy.typing as npt
from pydantic import TypeAdapter

from quantype import Force, u
from quantype.serialization import load_npz, save_npz
from quantype.systems import Atomistic, UnitSystem


class Gromacs(UnitSystem, name="gromacs"):
    length = u.nanometer
    energy = u.kJ_per_mol
    time = u.picosecond
    overrides = (u.bar,)


# Untyped parsing needs explicit definitions.
force = Force.from_value(2.0, Gromacs)
wire = force.to_dict()
parsed = Force.parse(wire, units=Gromacs.units)
assert parsed.magnitude(Gromacs.unit_for(Force)) == 2.0

# Typed validation discovers units through its target system.
adapter = TypeAdapter(Force[float, Gromacs])
validated = adapter.validate_json(json.dumps(wire))
assert validated.value == 2.0

# Save and restore real data; paths are relative to the working directory.
raw = np.arange(6, dtype=np.float32).reshape(2, 3)
forces = Force.from_value(raw, Gromacs)
save_npz("frame.npz", forces=forces)
restored = load_npz("frame.npz", "forces", Force[npt.NDArray[np.float32], Gromacs])
np.testing.assert_array_equal(restored.value, raw)
other_system = load_npz(
    "frame.npz",
    "forces",
    Force[npt.NDArray[np.float32], Atomistic],
    units=Gromacs.units,
)
np.testing.assert_allclose(
    other_system.magnitude(Gromacs.unit_for(Force)), raw, rtol=1e-6
)
```

Saved data records units, not the source system. An archive written in Gromacs
units can therefore be read into Atomistic, SI, or another system. Renaming the
Python class has no effect on saved data, but changing `name=` changes its
system-qualified unit identifiers. For full format and backend-boundary
contracts, see [serialize quantities](serialize.md) and
[validate data](validate-data.md).

New kinds from a [generated catalogue](custom-catalogues.md) get units in every
system automatically, because systems are defined over base dimensions rather
than a closed list of kinds.

## Vary one axis through inheritance

Inheritance shares definitions, but creates a separate storage system. Choose a
new `name=` for every system in the hierarchy:

```python
import pytest

from quantype import Length, u
from quantype.systems import Metal


class MetalFs(Metal, name="metal-fs"):
    time = u.femtosecond


metal = Length[float, Metal](1, u.angstrom)
converted = metal.to_system(MetalFs)
with pytest.raises(TypeError, match="Cannot combine Metal and MetalFs"):
    metal + converted
assert (metal + converted.to_system(Metal)).value == 2.0
```

The system parameter is invariant: a subclass cannot substitute for its parent
when the raw units differ. Static checkers reject mixing them too, apart from
one [documented mypy inference gap](../reference/systems.md#inheritance-and-invariance).

## Check range before choosing low-precision storage

A system definition must give every known kind a finite, nonzero float64 scale;
that does not guarantee safe float32 storage for atomistic magnitudes or their
squares. Inspect `check_range` for the dtype you intend to use:

```python
import numpy as np

from quantype import u
from quantype.systems import UnitSystem


class Gromacs(UnitSystem, name="gromacs"):
    length = u.nanometer
    energy = u.kJ_per_mol
    time = u.picosecond
    overrides = (u.bar,)


assert Gromacs.check_range(np.float32) == ()
assert Gromacs.check_range(np.float64) == ()
```

This checks representative atomistic magnitudes, not your actual data. Read
[range diagnostics](../reference/systems.md#numerical-range) and the
[built-in range table](../reference/units.md#numerical-range) for the warning
and underflow boundaries. Base units with offsets, wrong-kind bases, repeated
names within a hierarchy, and invalid overrides are rejected when the class is
defined; see the [validation contract](../reference/systems.md#unitsystem-definition-contract).

When publishing library functions that use this system, avoid hard-coding it
unless the function actually consumes its raw numbers. Use
[system-generic annotations](../reference/systems.md#write-system-generic-code)
to preserve the caller's system, bounded by `AnySystem` for general arithmetic
or `UnitSystem` when accessing `.value`.

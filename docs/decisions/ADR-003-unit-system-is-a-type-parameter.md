# The unit system is a type parameter, and values are stored in its units

## Context and Problem Statement

Every value was stored in one fixed set of units (Å, eV, fs). Kernels written in
SI, LAMMPS, or Hartree atomic units needed per-kind conversion at every call,
and nothing recorded which units an array of raw numbers was in. How can users
choose the units `.value` is in, without the same quantity silently having
different raw numbers?

## Considered Options

* One fixed storage set, with conversion helpers at kernel boundaries
* A storage system chosen once per process
* Install extras, such as `quantype[si]`
* A system per quantity, not visible in the static type
* Storing each value in its input unit, as pint does
* The system as the last type parameter, defaulting to `Atomistic`

## Decision Outcome

Chosen option: "the system as the last type parameter", because it is the only
option where each static type gives `.value` one meaning, systems never mix
silently, and existing annotations keep their meaning.

`Length[float]` is `Length[float, Atomistic]`, and `Length[float, SI]` stores
meters. Combining systems is rejected by all four checkers and raises
`TypeError` at runtime; `.to_system(T)` is the only bridge. The prototypes
behind this are in [unit storage](../design/unit-storage.md).

### Consequences

* Good, because arithmetic inside one system doesn't convert, except for the
  kinds a system overrides
  ([ADR-007](ADR-007-lammps-systems-store-lammps-units.md)).
* Good, because code generic over `[S: UnitSystem]` works in any system, and
  JAX traces once per system.
* Bad, because a library typed `Length[float]` rejects SI quantities unless it
  is written generically over the system.
* Bad, because SI storage underflows float32 for ordinary atomistic values.
  `check_range` and `StorageRangeWarning` report it; nothing prevents it.
* Bad, because every stub, adapter, Pydantic field, and loader carries the
  system parameter. The default needs PEP 696, so the stubs declare it with
  `typing_extensions.TypeVar` on Python 3.12.

---
name: quantype-core
description: Implement or debug Quantype runtime arithmetic, nominal physical semantics, unit systems and unit conversion, display units, affine temperatures, and custom unit definitions. Use for changes to the physical engine rather than standalone backend or wire-format work.
---

# Quantype Core

Parent: read [Quantype shared guidance](../quantype/SKILL.md). Source paths below are relative to the repository root.

## Physical invariants

- The dimension basis is `(length, energy, time, temperature, magnetic_moment, atom, electron)`. Unit scales are relative to the reference units angstrom, eV, fs, kelvin, Bohr magneton, atom and electron, which are the `Atomistic` system's base units; the reference unit of mass is therefore eV fs²/Å², although `Atomistic` stores mass in Da ([ADR-013](../../../docs/decisions/ADR-013-atomistic-stores-mass-in-daltons.md)). It uses energy as an axis; do not substitute an SI mass basis.
- Each quantity stores raw numbers in its unit system `S` (`_internal/_systems.py`). A system names one unit per axis; every kind's unit is the product raised to its dimensions, so arithmetic inside one system does not convert, except for kinds a system lists in `overrides` (`Atomistic` stores mass in Da and density in g/cm³; LAMMPS `Metal`/`Real` store mass, pressure, density and, in `Real`, electric field in LAMMPS's units; [ADR-007](../../../docs/decisions/ADR-007-lammps-systems-store-lammps-units.md)). Products, ratios, powers, `sqrt` and autodiff multiply by `coherence(system, kind)` factors, which are exactly 1 for non-overridden kinds; structural quantities are always stored coherently. Operations between systems raise `TypeError`; `.to_system(T)` is the only bridge. `S.unit_for(kind)` must be a named catalogue unit for built-in systems and kinds, else a system-qualified synthetic unit.
- `Kind` identity is nominal ([ADR-001](../../../docs/decisions/ADR-001-nominal-kinds.md)). `Pressure`, `EnergyDensity` and `EnergyPerVolume` share dimensions but represent different meanings. `Force / Area` gives `Pressure`; `Energy / Volume` gives `EnergyDensity`.
- Known products and integer powers come from explicit relations. Unlisted operations retain `Expression` trees and static `Mul`, `Div` or `Pow` markers. Do not infer a named kind or introduce symbolic cancellation from dimensions alone.
- Absolute `Temperature` is affine for addition ([ADR-008](../../../docs/decisions/ADR-008-absolute-temperatures-are-points.md)): point minus point gives `TemperatureDifference`; a point may be shifted by a difference; point addition and sums are rejected; means are valid. Storage is kelvin-scaled in every system, so products, ratios, powers and scaling are allowed; scaling drops an offset (°C) display unit.
- `Constant` (in `core.py`; values in `quantype.constants`) is system-free: arithmetic with a quantity adopts that quantity's system. Generated `_{Kind}Constant` classes carry the per-relation static overloads.
- Reference conversion is `raw * scale + offset`; into a system it is `raw * scale / f + offset / f` with `f = factor(S, kind)`. `.magnitude(unit)` reverses it; `.to(unit)` only records presentation. Preserve storage when wrapping raw values.
- Display rules ([ADR-004](../../../docs/decisions/ADR-004-display-unit-is-presentation.md)): input units are remembered; same-kind operations keep the display (left operand wins for `±`); point minus point maps `°C` to `Δ°C`; products, ratios and powers fall back to `S.unit_for(result)`. Display never affects `.value` or JAX tree structure.

Trace semantic decisions in `_internal/_semantics.py`, numerical operations in `core.py`, unit validation/conversion in `_internal/_unit.py`, and unit systems in `_internal/_systems.py` (public re-exports in `systems.py`). These files live under `src/quantype/`. Their handwritten stubs and generated quantity stubs also describe the public contract.

## Custom units

`QuantityClass.define_unit` composes a reference conversion: `reference_magnitude = input * scale + offset`. The new reference scale is `reference.scale * scale`, and its reference offset is `reference.scale * offset + reference.offset`. Preserve immutable definitions, finite positive scales, compatible nominal kinds, and the restriction of nonzero offsets to absolute-temperature units.

A custom unit object works immediately for construction and presentation. Decoding its name requires explicit definitions; it does not register a new public namespace binding. Prefer namespaced identifiers in examples. For decoding changes, read [serialization](../quantype-serialization/SKILL.md).

## Route changes and verify

- New named kinds, units or declared relations, or generator behavior: read [catalogue](../quantype-catalogue/SKILL.md).
- Overloads, handwritten stubs or checker disagreements: read [typing](../quantype-typing/SKILL.md).
- Raw storage conversions or backend dispatch: read [backends](../quantype-backends/SKILL.md).

Use `tests/runtime/test_core.py` for algebra, affine behavior and explicit numerical boundaries, `test_quantities.py` for display rules, equality, comparisons and indexing, `test_systems.py` for unit systems, and `test_refactor.py` for conversions and lazy imports. A public algebra change needs both a runtime example and the corresponding static result or rejection. Consult [checks](../quantype/references/checks.md) for the appropriate conformance commands.

---
name: quantype-core
description: Implement or debug Quantype runtime arithmetic, nominal physical semantics, canonical unit conversion, affine temperatures, and custom unit definitions. Use for changes to the physical engine rather than standalone backend or wire-format work.
---

# Quantype Core

Parent: read [Quantype shared guidance](../quantype/SKILL.md). Source paths below are relative to the repository root.

## Physical invariants

- The dimension basis is `(length, energy, time, temperature, magnetic_moment, atom, electron)`, with canonical units angstrom, eV, fs, kelvin, Bohr magneton, atom and electron. It uses energy as an axis; do not substitute an SI mass basis.
- `Kind` identity is nominal. `Pressure`, `EnergyDensity` and `EnergyPerVolume` share dimensions but represent different meanings. `Force / Area` gives `Pressure`; `Energy / Volume` gives `EnergyDensity`.
- Known products and integer powers come from explicit relations. Unlisted operations retain `Expression` trees and static `Mul`, `Div` or `Pow` markers. Do not infer a named kind or introduce symbolic cancellation from dimensions alone.
- Absolute `Temperature` is affine: point minus point gives `TemperatureDifference`; a point may be shifted by a difference. Point addition, scaling, products, powers and sums are rejected. Means of points are valid.
- Canonical conversion is `raw * scale + offset`. `.magnitude(unit)` reverses it; `.to(unit)` only records presentation. Preserve storage when wrapping canonical values.

Trace semantic decisions in `_internal/_semantics.py`, numerical operations in `core.py`, and unit validation/conversion in `_internal/_unit.py`. These files live under `src/quantype/`. Their handwritten stubs and generated quantity stubs also describe the public contract.

## Custom units

`QuantityClass.define_unit` composes a reference conversion: `reference_magnitude = input * scale + offset`. The new canonical scale is `reference.scale * scale`, and its canonical offset is `reference.scale * offset + reference.offset`. Preserve immutable definitions, finite positive scales, compatible nominal kinds, and the restriction of nonzero offsets to absolute-temperature units.

A custom unit object works immediately for construction and presentation. Decoding its name requires explicit definitions; it does not register a new public namespace binding. Prefer namespaced identifiers in examples. For decoding changes, read [serialization](../quantype-serialization/SKILL.md).

## Route changes and verify

- New named kinds, units or declared relations, or generator behavior: read [catalogue](../quantype-catalogue/SKILL.md).
- Overloads, handwritten stubs or checker disagreements: read [typing](../quantype-typing/SKILL.md).
- Raw storage conversions or backend dispatch: read [backends](../quantype-backends/SKILL.md).

Use `tests/runtime/test_core.py` for algebra, affine behavior and explicit numerical boundaries, and `test_refactor.py` for canonical conversions and lazy imports. A public algebra change needs both a runtime example and the corresponding static result or rejection. Consult [checks](../quantype/references/checks.md) for the appropriate conformance commands.

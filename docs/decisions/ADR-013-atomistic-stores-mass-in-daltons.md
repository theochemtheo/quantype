# The default system stores mass in daltons

## Context and Problem Statement

`Atomistic` uses Å, eV, and fs, so its coherent mass unit is eV fs²/Å², about
0.00965 Da, and its coherent density unit is eV fs²/Å⁵. No one quotes masses or
densities in those units. ASE and most atomistic codes take masses in daltons
(amu), and densities are usually given in g/cm³. Should `Mass[float].value` be
the number people already have?

## Considered Options

* Keep `Atomistic` coherent; users convert masses at the boundary
* Override mass (Da) and mass density (g/cm³) in `Atomistic`, as
  [ADR-007](ADR-007-lammps-systems-store-lammps-units.md) does for LAMMPS
* Change `Atomistic`'s time base to ASE's, Å √(amu/eV) ≈ 10.18 fs, so that mass
  derives to the dalton

## Decision Outcome

Chosen option: "override mass and mass density in `Atomistic`", because it makes
`.value` the dalton and keeps the femtosecond as the unit of time. ASE's time
unit would make every time, velocity, and rate awkward instead.

The reference units stay Å, eV, and fs, so the reference unit of mass stays
eV fs²/Å². A canonical unit must have scale 1, and every catalogue scale is
measured against it. `Atomistic` is no longer identical to the reference units,
and code converting into it goes through `factor`, like any other system.

Momentum isn't overridden. ASE's momentum unit, √(amu eV) ≈ 10.18 eV fs/Å,
follows from its time unit, so no unit built on the femtosecond would match it.

### Consequences

* Good, because masses from ASE wrap with `Mass.from_value` and need no
  conversion, and `Atomistic` stores densities in g/cm³ like `Metal` and `Real`.
* Bad, because products that produce or consume a mass or a density rescale by
  a constant factor in the default system; 1 Da Å²/fs² is about 103.6 eV.
* Bad, because it changes the raw numbers of `Mass` and `MassDensity` in
  `Atomistic` from 0.1.0. Saved data records its unit, so it still decodes
  correctly.

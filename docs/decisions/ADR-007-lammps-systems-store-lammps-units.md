# The LAMMPS systems store LAMMPS's own units

## Context and Problem Statement

LAMMPS's `metal` and `real` unit styles aren't coherent. `metal` uses Å, eV, and
ps, but stores mass in g/mol, pressure in bar, and density in g/cm³; `real` also
uses atm for pressure and V/Å for electric field. A system built only from base
units would store pressure in eV/Å³, which a LAMMPS kernel would read as bar.
Should `Metal` and `Real` match LAMMPS exactly?

## Considered Options

* Coherent systems only, sharing LAMMPS's base units; users convert the other
  kinds at the boundary
* Systems may override the unit of individual kinds, and arithmetic rescales
  results for those kinds by a constant factor

## Decision Outcome

Chosen option: "systems may override individual kinds", because `.value` of
`Pressure[float, Metal]` should be the number LAMMPS reads and writes.

`Metal.overrides` lists g/mol, bar, and g/cm³; `Real.overrides` lists g/mol,
atm, V/Å, and g/cm³. Products, ratios, powers, and derivatives multiply by
`coherence(system, kind)`, which is exactly 1 for every kind a system doesn't
override.

### Consequences

* Good, because raw LAMMPS data wraps with `from_value` and needs no per-kind
  conversion.
* Good, because custom systems can override kinds the same way for other codes.
* Bad, because arithmetic inside one system converts when it produces or
  consumes an overridden kind, so "a system's arithmetic never converts" no
  longer holds.

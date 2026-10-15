# A quantity remembers its input unit for display only

[ADR-014](ADR-014-opt-in-unit-system.md) changes this decision for quantities
without a unit system. They store their numbers in the input unit, the third
option below, so the input unit is presentation only in a unit system. Without
one, products, ratios, and powers show their operands' units composed, and
values coming out of a JAX transformation show in the reference units.

## Context and Problem Statement

`Length[float](2, u.nm)` printed `20.0 Å`, and a configuration's `"0.5 nm"` was
written back as `5.0 angstrom`. Users expect to see the unit they gave. Where
does that unit live, and what may it affect?

## Considered Options

* Always show the unit system's unit
* Application-wide display preferences
* Store each value in its input unit
* Remember the input unit on each quantity as presentation only, outside
  storage and outside JAX pytree metadata, and keep the exact Python scalar a
  quantity was built from

## Decision Outcome

Chosen option: "remember the input unit as presentation only", because it fixes
printing, `.magnitude()`, and configuration round trips without changing
`.value` or JAX tree structure. A unit in pytree metadata made `lax.cond` fail
when branches differed only in unit, and made `jit` retrace for each input
unit. The prototypes are in [unit storage](../design/unit-storage.md).

Same-kind operations keep the display unit, with the left operand winning for
`+` and `-`. A point minus a point in °C shows Δ°C. Products, ratios, and powers
show the system's unit for the result.

### Consequences

* Good, because `repr`, `.magnitude()`, and the wire format use the unit the
  value was given in, and `20.1 °C` round-trips exactly.
* Good, because `jit` traces once, and `lax.cond` and `lax.scan` work across
  display units.
* Bad, because a value coming out of a JAX transformation shows the system's
  unit.
* Bad, because the exact echo covers Python scalars only, so arrays in offset
  units round-trip with floating-point roundoff.

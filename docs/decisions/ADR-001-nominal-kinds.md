# Kinds are nominal, and only declared relations name results

## Context and Problem Statement

Every quantity has a kind that the runtime and type checkers both check.
Different physical quantities can share dimensions: `Pressure`,
`EnergyDensity`, and `EnergyPerVolume` are all energy per length cubed, and
`Frequency` and `InverseTime` are both per time. Is a kind its dimensions or its
name, and which arithmetic results get a named kind?

## Considered Options

* A kind is its dimension vector; any result with matching dimensions has that
  kind
* Kinds are named, and a result is named whenever exactly one kind has its
  dimensions
* Kinds are named, and only declared relations name a result; other results
  keep their structure (`Mul`, `Div`, `Pow`)

## Decision Outcome

Chosen option: "only declared relations name a result", because it is the only
option that keeps kinds with equal dimensions apart, both at runtime and in the
static types, and it gives the generator one explicit table to render.

`Force / Area` is a `Pressure` and `Energy / Volume` an `EnergyDensity`,
because the catalogue declares those relations, and passing one where the other
is expected is an error. Relations are never solved as equations.

### Consequences

* Good, because quantities with the same dimensions and different meanings
  can't be confused, statically or at runtime.
* Good, because every named result comes from the catalogue's relations, which
  the runtime, the stubs, and `docs/catalogue.md` all read.
* Bad, because correct physics through an undeclared relation gives a
  structural result. Users declare the relation in an application catalogue
  ([ADR-010](ADR-010-application-catalogues-are-separate-packages.md)) or
  assert it with `reinterpret`.
* Bad, because each new kind needs its relations declared by hand.

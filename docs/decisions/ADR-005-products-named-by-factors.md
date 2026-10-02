# Products are named by their factors, whatever their order

## Context and Problem Statement

A product's kind depended on how the formula was written. `m*v*v` was an
`Energy`, but `m*v**2`, `m*(v*v)`, and `p**2/(2*m)` were unnamed, because a
result was named only when both operands were named kinds. The stubs also
widened `x**N` to `Pow[K, int]`, so `v**2 + v**3` type-checked and failed at
runtime. How should products be named, and must the runtime and the checkers
name the same ones?

## Considered Options

* Keep naming one pair of named operands at a time, and document it
* Name more products at runtime than the stubs can express
* Generate one naming table from the declared relations, used by both the
  runtime and the stubs

## Decision Outcome

Chosen option: "one naming table", because it is the only option that makes the
order, grouping, and powers of a formula irrelevant while keeping the runtime
and the checkers in exact agreement
([ADR-002](ADR-002-generated-stubs-for-four-checkers.md)).

Every product of up to three named factors is named when a declared relation
names it. Each unnamed product of two kinds, and each literal power 2, 3, −1,
and −2, has a product class, such as `LengthTime`, `VelocitySquared`, or
`PerLength`, importable from `quantype.products`. `x * x` and `x ** 2` give the
same class. A product with two possible names, such as `Force / Volume * Length`
(`Pressure` or `EnergyDensity`), stays unnamed. The table's construction is in
[naming table](../design/naming-table.md).

### Consequences

* Good, because `m*v**2`, `m*(v*v)`, and `p**2/m` are all `Energy`, in every
  checker and at runtime.
* Good, because unnamed products have readable class names in errors and
  annotations.
* Bad, because the table has 8,498 entries over 1,733 product classes, which
  checkers must load
  ([ADR-006](ADR-006-protocol-dispatch-for-product-stubs.md)).
* Bad, because products of four or more factors can still depend on grouping.

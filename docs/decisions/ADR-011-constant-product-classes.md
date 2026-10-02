# Products of constants have product classes too

## Context and Problem Statement

At runtime, a constant combined with anything acts as a float quantity of its
kind, so products of constants follow the naming table
([ADR-005](ADR-005-products-named-by-factors.md)):
`(hbar * c) / L` is an `Energy` and `(2 / k_B) * E` a `Temperature`. The stubs
typed `hbar * c` as `Constant[Mul[ActionKind, VelocityKind]]`, in written
order, so checkers saw those results as unnamed quantities, breaking the exact
agreement of [ADR-002](ADR-002-generated-stubs-for-four-checkers.md).
How should products of constants be typed?

## Considered Options

* A constant class for each product class, dispatched as quantities are
* Constants skip the naming table at runtime, so products of constants stay
  structural everywhere
* Record the disagreement as a known limit of ADR-005

## Decision Outcome

Chosen option: "a constant class for each product class", because it is the
only option that keeps the runtime and every checker in exact agreement without
naming less.

Each product module under `_products/` also holds that product's constant
class, such as `_VelocityActionConstant`, with the product class's table
entries. A product of two constants finds its result in the right constant's
`_cmul_` or `_ctruediv_` member, as `X * Y` does through `_rmul_`
([ADR-006](ADR-006-protocol-dispatch-for-product-stubs.md)), so
`hbar * c` is a `_VelocityActionConstant` and `k_B * k_B` an
`_EntropySquaredConstant`. Reciprocals and literal powers of named constants
are typed the same way. At runtime, each constant product class is created on
first use.

### Consequences

* Good, because `(hbar * c) / L`, `(2 / k_B) * E`, `m_e * c * c`, and every
  other product of constants have the same class in all four checkers as at
  runtime.
* Good, because no stub modules are added.
* Bad, because the package grows from 22 MB to 31 MB, and mypy's cold run takes
  2.4 to 2.9 s longer and about 400 MB more memory. Pyright, Pyrefly, and ty
  change by at most 0.13 s; mypy's edit loop by at most 0.13 s. The
  measurements are in [naming table](../design/naming-table.md).

# Product stubs dispatch through protocols, with one module per class

## Context and Problem Statement

The naming table ([ADR-005](ADR-005-products-named-by-factors.md)) has 8,498
entries over 1,733 product classes. Written as overloads, it made type checkers
5 to 40 times slower, because a checker tries a method's overloads in order. How
should the stubs express the table so that checking stays fast, measured with
the stubs installed as users get them?

## Considered Options

* Every entry as a flat overload on each named class
* One class per product holding its entries, all in one module
* One class per product, one module each, with named classes listing an
  overload per partner kind
* Protocol dispatch: `X * Y` finds its result in a member `_rmul_X` on `Y`, with
  one stub module per product class

## Decision Outcome

Chosen option: "protocol dispatch", because it is the only option whose cost per
operation matches the old stubs: mypy re-checked 300 products in 0.84 s against
0.74 s before, and 17.7 s with per-partner overloads. The measurements are in
[naming table](../design/naming-table.md).

`X.__mul__` has a fixed set of overloads: scalars, two protocols, and a
structural fallback. Float storage on the left has its own protocol
(`_RMulXF`), because with one protocol ty inferred `Unknown` for float times
float. Product classes inherit kind-preserving methods from the base class.
Constant classes carry protocol members too, and each has its own stub module
under `_constants/`, because together they exceed Pyright's per-module
complexity limit. The runtime table is stored as text rows and parsed lazily,
because a tuple literal took Pyright 47 s to analyze.

### Consequences

* Good, because Pyright, Pyrefly, and ty check the test suite within a second of
  the old stubs, and mypy's edit loop is 0.3 s slower.
* Bad, because loading about 1,700 stub modules is a fixed cost for any file
  that imports quantype: mypy +5.5 s cold, Pyright +0.5 s, and Pyrefly and ty
  +0.2 s.
* Bad, because the package carries about 19 MB of stubs.
* Bad, because the `_rmul_*` members exist only in the stubs, so stubtest's
  allowlist names them.

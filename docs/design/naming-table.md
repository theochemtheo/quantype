# Naming products by their factors: design and checker benchmarks

Evidence for [ADR-005](../decisions/ADR-005-products-named-by-factors.md) and
[ADR-006](../decisions/ADR-006-protocol-dispatch-for-product-stubs.md). The
prototypes and raw results were built outside the repository on 2026-10-01; this
page keeps their method and numbers.

## The problem

A product's kind depended on how the expression was written. `m*v*v` was an
`Energy`, but `m*v**2`, `m*(v*v)` and `p**2/(2*m)` were unnamed, because the
runtime named a result only when both operands were named kinds, and the stubs
mirrored that one pair at a time. The stubs also widened `x**N` to
`Pow[K, int]`, so `v**2 + v**3` type-checked and failed at runtime.

## The table

Named kinds are atoms, and an unnamed result is a monomial over them
(`{Mass: 1, Velocity: 2}`). A monomial is named when some split into two
nameable parts matches a declared relation. Relations are never treated as
equations, so `Pressure` and `EnergyDensity` stay distinct.

- Of 54,740 monomials of two or three factors, 748 are named and 3 are
  ambiguous: `Force·Length/Volume` and `Energy/(Area·Length)` (`EnergyDensity`
  or `Pressure`), and `Frequency·InverseTime·Time`. Ambiguous ones stay
  unnamed.
- As stub patterns (a named kind times or over an unnamed pair), the prototype
  had 8,949 entries over 1,675 canonical pairs. The implementation, with every
  product of two kinds and the literal powers 2, 3, −1 and −2, has 8,498
  entries over 1,733 product classes.
- 69% of entries cancel back to an operand, such as `(A*B)/B -> A`; these make
  order and bracketing irrelevant.

## Designs compared

| Variant | Design |
| --- | --- |
| `base` | the stubs before the change |
| `full` | every entry as a flat overload on each named class and on `Quantity`'s base methods |
| `dispatch` | one class per canonical pair holding its entries, all in one module; named classes return the pair class through per-partner overloads |
| `split-each` | `dispatch` with one module per pair class |
| `d2-trim-lean` | `split-each` with complete pair classes, trimmed imports, and one reflected fallback per named class |
| `pd-trim` | protocol dispatch: `X * Y` finds its result in a member `_rmul_X` on `Y`, so `X.__mul__` has a constant number of overloads |
| `pd-trim-self-merge` | `pd-trim` with pair classes inheriting kind-preserving methods from the base class, and two storage overloads instead of three |

## Method

Each variant was installed into the site-packages of a separate environment, as
users get it. With the stubs as local code, mypy type-checks the stubs too and
overstates its cost badly (80–98 s cold for `split-each`). The environment lived
outside the repository: Pyrefly finds a config by walking up from each file,
and the repository's `search-path` would have made the installed stubs import
`src/quantype`.

Workloads: `trivial` (one addition, one product), `suite` (a copy of
`tests/typing/positive`), and `stress` (300 random two- and three-factor
products over all 35 kinds). mypy ran cold (empty cache) and edit (the workload
changed, stubs cached), which is the edit-and-check loop. Apple M3 Pro, 18 GB,
medians of three runs.

## Results

| Checker, workload | base | split-each | d2-trim-lean | pd-trim | pd-trim-self-merge |
| --- | --- | --- | --- | --- | --- |
| Pyright, trivial | 0.59 s | 1.18 s | 1.15 s | 1.08 s | 1.24 s |
| Pyright, suite | 1.30 s | 2.60 s | 2.52 s | 1.86 s | 1.95 s |
| Pyright, stress | 0.91 s | 4.24 s | 4.21 s | 1.84 s | 1.67 s |
| Pyrefly, suite | 0.16 s | 0.43 s | 0.43 s | 0.34 s | 0.33 s |
| Pyrefly, stress | 0.17 s | 1.12 s | 1.13 s | 0.44 s | 0.40 s |
| ty, suite | 0.14 s | 0.41 s | 0.44 s | 0.33 s | 0.34 s |
| ty, stress | 0.18 s | 1.24 s | 1.30 s | 0.50 s | 0.47 s |
| mypy, trivial, cold | 1.66 s | 7.09 s | 7.31 s | 7.17 s | 6.27 s |
| mypy, suite, cold | 11.4 s | 16.8 s | 17.7 s | 17.2 s | 17.1 s |
| mypy, stress, cold | 2.18 s | 22.7 s | 24.1 s | 7.43 s | 6.35 s |
| mypy, trivial, edit | 0.13 s | | 0.48 s | 0.46 s | 0.43 s |
| mypy, suite, edit | 0.53 s | | 0.99 s | 0.84 s | 0.79 s |
| mypy, stress, edit | 0.74 s | | 17.7 s | 0.84 s | 0.83 s |

Peak memory, Pyright on stress: 375 MB (`base`), 1.8 GB (`d2-trim-lean`),
580 MB (`pd-trim`), 550 MB (`pd-trim-self-merge`). `full` put 14,673 overloads
on `Quantity.__mul__` and `__truediv__`; `dispatch` cost 25 s in ty for any
file.

The implementation, installed the same way:

| Check | base | implementation |
| --- | --- | --- |
| Pyright, suite / stress | 1.29 s / 0.90 s | 1.88 s / 1.84 s |
| Pyrefly, suite / stress | 0.15 s / 0.17 s | 0.36 s / 0.52 s |
| ty, suite / stress | 0.15 s / 0.18 s | 0.25 s / 0.45 s |
| mypy cold, trivial / suite | 1.65 s / 11.6 s | 7.60 s / 17.5 s |
| mypy edit, suite / stress | 0.53 s / 0.75 s | 0.85 s / 0.86 s |

## Findings

- Checkers try a method's overloads in order. A named class's `__mul__` with
  about 110 overloads cost mypy about 57 ms per product when re-checking the
  stress file; protocol dispatch removed that cost (17.7 s to 0.84 s).
- One class per module is needed. With every pair class in one module, ty took
  25 s for any file.
- Stub volume matters little. Trimming imports and fallbacks cut size by 18%
  and time by under 10%. Inheriting kind-preserving methods and merging storage
  overloads cut mypy's cold runs by 10–15%.
- What remains is a fixed cost of loading about 1,700 stub modules, paid by
  every checker for any file that imports quantype: mypy +5.5 s cold and
  +0.3 s per edit, Pyright +0.5 s, ty and Pyrefly +0.2 s. It scales with the
  number of product classes more than their size.
- With one protocol for every left operand, the member's storage overloads
  overlap for float times float, and ty infers `Unknown`. Float storage on the
  left gets its own protocol (`_RMulXF`, member `_rmul_X_f`).
- Two limits appeared during implementation. Constant classes carry a protocol
  member for every kind and operator; together they exceed Pyright's
  per-module "code is too complex to analyze" limit in `_generated.pyi`, so
  each constant class has its own stub module under `_constants/`. The runtime
  table, written as a tuple literal in `products.py`, took Pyright 47 s to
  analyse; stored as text rows and parsed lazily, the repository's Pyright run
  fell from about 130 s to 19 s.

## Constant product classes

Evidence for
[ADR-011](../decisions/ADR-011-constant-product-classes.md): each product module gained a constant class with the product's
entries, and each named constant gained `_cmul_`/`_ctruediv_` members, typed
power overloads, and a typed reciprocal. Installed the same way on 2026-10-02,
against the stubs before the change:

| Checker, workload | before | constant classes |
| --- | --- | --- |
| Pyright, trivial / suite / stress | 1.12 s / 1.95 s / 1.93 s | 1.11 s / 1.97 s / 2.06 s |
| Pyrefly, trivial / suite / stress | 0.34 s / 0.41 s / 0.59 s | 0.41 s / 0.46 s / 0.71 s |
| ty, trivial / suite / stress | 0.24 s / 0.27 s / 0.47 s | 0.24 s / 0.28 s / 0.57 s |
| mypy cold, trivial / suite / stress | 9.17 s / 19.3 s / 9.32 s | 11.5 s / 22.2 s / 12.2 s |
| mypy edit, trivial / suite / stress | 0.53 s / 0.95 s / 0.94 s | 0.63 s / 1.02 s / 1.07 s |

mypy's cold peak memory rose from 1.25–1.80 GB to 1.62–2.26 GB, and the
installed package from 22 MB to 31 MB. No modules were added, so the cost is
stub volume, which mypy alone feels on a cold run.

# Opt-in unit systems: prototype comparison

Evidence for the [opt-in unit system decision](../decisions/ADR-014-opt-in-unit-system.md),
and notes for implementing it. The prototypes were built in October 2026 as
copies of quantype 0.2.0 (commit `056c344`), changed by script. They live
outside the repository, in `/Users/tk/p/quantype-opt-in/`, which git can't
restore. Timings are best of 7 runs on one Apple-silicon Mac, with float64 data
and million-element arrays unless stated. The prototypes call the null system
`Unbound`.

## Round 1: three ways to make the system opt-in

| Variant | Default | Raw numbers |
| --- | --- | --- |
| P0 | `Atomistic` (today) | `.value` everywhere |
| P1 | a null system storing Å, eV, and fs | `.value` only in a bound system |
| P2 | `Atomistic`; `Atomistic`, `SI`, `CGS`, and `Atomic` are auto, `Metal`, `Real`, and custom systems strict | `.value` everywhere; mixing two auto systems converts into the left operand's |
| P3 | no system in the type | `x.value_in(Real)`, `Force[A].from_value(raw, Real)` |

| | P0 | P1 | P2 | P3 |
| --- | --- | --- | --- | --- |
| Scenario probe, 4 checkers | pass | pass | 3 gaps | pass |
| Stub size | 17.0 MB | 17.0 MB | 23.4 MB | 15.8 MB |
| mypy, cold | 9.2 s, 1.66 GB | 9.3 s | 13.0 s, 2.27 GB | 7.2 s, 1.48 GB |
| Pyright | 2.2 s | 2.0 s | 2.8 s | 1.9 s |
| Pyrefly | 0.53 s | 0.53 s | 0.79 s | 0.49 s |
| ty | 0.39 s | 0.41 s | 2.2 s | 0.33 s |
| Scalar `+`; array `+` | 0.49 µs; 475 µs | same | same; 1.8 µs and 1.38 ms when mixing systems | same |
| Runtime tests failing, of 1,085 | 0 | 177 | 8 | 249, and 4 errors |

Checker times are for a 400-line probe of mixed arithmetic. The P2 stubs come
from adding an auto-system overload after every same-system binary overload
and relation hook: 47,563 overloads.

Findings:

- P1 gates `.value` statically in all four checkers with a descriptor (see
  [static typing](#static-typing)), at no stub or checking cost. Most of its
  test failures read `.value` or call `from_value` on a default quantity.
- P2 is the most convenient and the most expensive. `lax.cond` over `a + b`
  and `b + a` fails because the branches land in different systems. In
  float32, (1 meV)² is 1e-6 eV² with an `Atomistic` operand on the left and
  2.6e-44 J² (subnormal) with an SI operand on the left, with no warning.
  mypy accepts `x < real` (auto against strict), and ty infers `Unknown` for
  a float SI quantity times an `Atomistic` array.
- P3 has the cleanest checker messages and the smallest stubs, but nothing
  records which system a raw array came from, every kernel call pays a
  conversion, and every system-typed annotation breaks.
- P1 still converts at construction, so it stores numbers in units nobody
  chose. That led to round 2.

## Round 2: a null system that keeps the input unit (P1b)

P1b keeps P1's stubs byte for byte and changes only the runtime. Each
null-system quantity carries a storage unit; `None` means reference units.

| | P1 | P1b |
| --- | --- | --- |
| Construct from nm | 395 µs | 3 µs |
| Read back in nm | 382 µs | 0.5 µs |
| Read back in Å | 0.46 µs | 368 µs |
| Unbind a `Real` force | 961 µs | 2 µs |
| Bind to SI | 974 µs | 955 µs |
| Add, same unit | 514 µs | 481 µs |
| Add, nm + Å | 513 µs | 1,013 µs |
| Force × length | 520 µs | 491 µs |
| Scalar eV / nm | 0.53 µs | 0.64 µs |
| `qnp.linalg.norm` of nm data | 2.94 ms | 2.98 ms |
| `==` nm against Å; same unit | | 676 µs; 218 µs |
| float64 nm round trip, values changed | 125,373 | 0 |

Construction is cheap partly because the quantity shares the caller's array
when no conversion is needed. Today that happens only when the input unit is
the system's unit.

JAX behaves as in P1: one trace for nm, Å, and bohr inputs, `lax.cond` works
across units, and `grad` is correct. The runtime suite fails 181 tests: P1's
177, two that expect products to display in Å, and two codegen typing tests
that pick up an unrelated `VIRTUAL_ENV` from the shell and fail in P1 too.

### Sharp edges found, and the fixes measured above

| Edge | First prototype | Fix |
| --- | --- | --- |
| Equality | The right operand converted into the left's unit: `0.3 nm == 3 Å` was False while `3 Å == 0.3 nm` was True (26,091 of 100,000 random pairs disagreed), and `__hash__` used stored numbers, so `1 nm == 10 Å` hashed differently | Same storage unit: compare stored numbers. Otherwise compare, and hash, in reference units. 0 asymmetric pairs, 0 hash mismatches |
| Display | `m * v**2` showed `2.0 Da·((Å/fs)^2)` | One product or quotient shows its symbol; a nested one of a named kind shows the kind's unit |
| Unnamed products | nm × ps rescaled to `59999.999999999985 Å·fs` | Composed storage unit: `6.0 nm·ps` |
| NumPy rules | A wrapper converted every operand to reference units first (norm took 3.56 ms) | Per-rule handling, below |

Two rejected equality rules: exact rational comparison is symmetric and
hash-consistent but makes `0.3 nm != 3 Å` and took 12 ms per million
elements; raising or returning False across units breaks `in`, dict lookups,
or `1 nm == 10 Å`.

The prototype showed nested compositions in `Atomistic`'s units, which gives
`2 g × (3 m/s)²` as `1.12e+17 eV`. The proposal infers the system from the
operands, with a fixed fallback; that rule isn't prototyped.

### The `quantype.numpy` rules

| Group | Rules | Storage |
| --- | --- | --- |
| Same unit out | `sum mean median max min cumsum diff std sort quantile percentile absolute negative reshape transpose squeeze expand_dims zeros_like norm`, and the `nan` versions | the operand's unit |
| Unit transformed | `sqrt square power var nanvar` | composed, as `**` |
| Need a plain number | `sin cos tan sinh cosh tanh arcsin arccos arctan exp expm1 log log2 log10 log1p` | converted to radians or a plain number |
| No unit out | `sign isnan isinf isfinite argmin argmax argsort shape ndim` | ignored: a positive scale keeps sign, order, and NaN |
| One kind, several operands | `add subtract maximum minimum hypot clip where concatenate stack linspace isclose allclose arctan2` and comparisons | aligned to the first operand; comparisons use the equality rule |
| Products | `multiply divide matmul dot outer cross trapezoid` | composed, as `*` |

A conformance test builds operands in a test unit worth 3 nm, counts every
rescaling, and compares each result with the same rule applied to
reference-unit operands. All 63 cases pass: the first and second groups, the
fourth, the products, and the multi-operand rules with shared units make no
conversion.

### Display examples (P1b)

| Expression | Shows | Stored as |
| --- | --- | --- |
| `Energy(3, eV) / Length(2, nm)` | `Force(1.5 eV/nm)` | eV/nm |
| `Length(2, nm) * Time(3, ps)` | `LengthTime(6.0 nm·ps)` | nm·ps |
| `Length(2, nm) ** 3` | `Volume(8.0 nm^3)` | nm^3 |
| `Force(1, kcal/mol/Å) * Length(2, Å)` | `Energy(2.0 kcal/mol)` | kcal/mol |
| `Mass(2, Da) * Velocity(1, Å/fs) ** 2` | `Energy(207.2853933432161 eV)` | Da·((Å/fs)^2) |
| `Mass(2, Da) * Velocity(1, Å/fs) * Velocity(1, Å/fs)` | `Energy(207.2853933432161 eV)` | (Da·(Å/fs))·(Å/fs) |

## Round 3: catalogue scales

Rerunning the catalogue's unit definitions with every input and literal as the
exact decimal it denotes: of 162 units, 6 involve π or √ and can't be checked
this way (degree, `ase_time`, `ase_velocity`, `ase_momentum`,
`atomic_momentum`, `atomic_action`). Of the other 156, 46 are not the nearest
float to their exact value.

- 5 ulp: atomic magnetization. 4 ulp: electron mass per bohr³. 3 ulp: bohr³,
  atoms and electrons per bohr³, hartree per bohr³.
- 1 ulp: among others, picosecond, second, joule, pascal, bar, kilobar, MPa,
  GPa, coulomb, statcoulomb, statvolt, bohr, atomic time, atmosphere, debye,
  cm/s, cm⁻¹, dyne, and J/K.

The coherence factors of every built-in system are exactly 1 for every kind
that the system doesn't override, so these errors don't reach arithmetic
within a bound system. They do break identities between units:

| Identity | Today | Each scale rounded once | Exact rationals |
| --- | --- | --- | --- |
| N = kg·m/s² | fails | holds | holds |
| Pa = N/m² | fails | holds | holds |
| ns = 1000 ps | fails | holds | holds |
| s = 1000 ms | fails | holds | holds |
| g = kg/1000 | fails | fails | holds |
| J = N·m | holds | fails | holds |

## Static typing

A self-typed property, `def value[T: UnitSystem](self: Quantity[V, T]) -> V`,
is rejected by Pyright and ty on a null-system quantity and silently accepted
by mypy and Pyrefly. A descriptor works in all four:

```python notest
class _Value:
    @overload
    def __get__(self, obj: None, owner: Any, /) -> _Value: ...
    @overload
    def __get__[W, T: UnitSystem](
        self, obj: Quantity[Any, W, T], owner: Any, /
    ) -> W: ...
```

The null system and `UnitSystem` both derive from `AnySystem`, which bounds
the system parameter. `x.value` on a null-system quantity reports:

| Checker | Message |
| --- | --- |
| mypy | `Value of type variable "T" of "__get__" of "_Value" cannot be "Unbound"` |
| Pyright | `Cannot access attribute "value" for class "Length[float, Unbound]"` |
| Pyrefly | ``No matching overload found for function `quantype.core._Value.__get__` `` |
| ty | ``Invalid access to descriptor attribute `value` on type `Length[float, Unbound]` `` |

Pyrefly prints a defaulted type argument as nothing, so its messages about the
null system say `Length[float]`.

## Implementing it

Suggested order; each step has a runtime test and a typing test
([ADR-002](../decisions/ADR-002-generated-stubs-for-four-checkers.md)).

1. Round each catalogue scale once, in `_internal/_registry.py`, by evaluating
   the definitions with `fractions.Fraction` and converting at the end. Add a
   test that repeats the round-3 audit. This can land on its own.
2. Add `AnySystem` and the null system in `_internal/_systems.py`. The
   prototype kept the null system a `UnitSystem` subclass at runtime and hid
   that from the checkers; the real version must match its stub, so the
   system machinery (`unit_for`, `factor`, `into_system`) needs a reference-unit
   path for the null system.
3. Change the renderers (`codegen/_render.py`, `codegen/_products.py`): the
   system type variable's bound and default, and `T: AnySystem` wherever a
   system is accepted. Replace `value` in the handwritten `core.pyi` with the
   descriptor. Regenerate.
4. Store the input unit: a `_storage` field set by `_wrap`; construction in
   `_internal/_construction.py` and `Unit.__call__`; decoding in
   `serialization.py` and `_internal/_validation.py`; `.magnitude()`, `.to()`,
   `to_system` (bind converts, unbind records the system's unit), and
   `reinterpret`. Widen `Unit`'s kind parameter to accept product trees, and
   compare product trees by structure in `_check_unit` (`cross` builds one
   without interning it).
5. Arithmetic: alignment for `+`, `-`, and comparisons; composed units for
   `*`, `/`, `**`, and reflected division; the equality and hashing rule.
6. The `quantype.numpy` groups above, in `_internal/_numpy.py`. Normalize to
   reference units in the JAX pytree flatten, in `ujax` differentiation, and in
   `testing.assert_allclose`; scale `utorch.grad` results by the output's and
   input's storage scales.
7. Display: composed symbols, and the operand-inferred system for nested
   compositions. A composed unit can carry the set of built-in systems that
   store its operands' kinds in those units: the intersection of its
   operands' sets, which doesn't depend on operand order. It displays in the
   kind's unit of the first system in the fixed order, or of the fallback
   system when the set is empty.
8. `apply(function)` on every quantity, as the null-system replacement for
   `with_value`.
9. Docs and README: the default, `.value`, `from_value`, and display rules all
   change.

### Prototype shortcuts not to copy

- The stubs were edited by regular expressions, not by changing the renderers.
- Composed units are named `unbound:<symbol>`, and catalogue lookup matches
  scales to 12 significant digits. Exact rational scales would replace the
  key.
- A nested composition is detected by a parenthesis in its symbol. Record the
  nesting when the unit is built.
- `to_dict` of a quantity stored in a composed unit wasn't tested.
- Many runtime tests fail at `.value` before they reach storage code, so the
  P1b test run says little about the paths behind them, Torch differentiation
  among them. Those paths need their own tests.

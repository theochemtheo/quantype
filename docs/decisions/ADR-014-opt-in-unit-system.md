# Quantities have no unit system until one is chosen, and keep the unit they were given in

## Context and Problem Statement

Every quantity has a unit system today. `Length[float]` is
`Length[float, Atomistic]`, so `Length[float](2, u.nm).value == 20.0`: the
numbers are in ångströms although nothing asked for them. Code that never hands
numbers to a kernel still lives in a system, a library typed `Length[float]`
rejects SI quantities, and converting into Å, eV, and fs at construction
changes data on the way through: 125,373 of 1,000,000 random float64 values
given in nm come back different from `.magnitude(u.nm)`.

Can a user opt into a unit system where raw numbers are needed, without losing
what [ADR-003](superseded/ADR-003-unit-system-is-a-type-parameter.md) gives:
`.value` has one meaning for each static type, and systems never mix silently?

The decision covers the default system, how default-system quantities store
their numbers, `.value`, `from_value`, and `with_value`, equality and hashing,
how products display, the `quantype.numpy` rules, and the JAX and Torch
boundaries. Bound systems (`Atomistic`, `SI`, `Metal`, `Real`, and custom ones)
keep their current behavior.

## Considered Options

* Keep `Atomistic` as the default (ADR-003)
* A null system that stores Å, eV, and fs and has no `.value`
* Auto and Strict systems: mixing two auto systems converts into the left
  operand's system
* No system in the type: raw numbers only through `value_in(system)` and
  `from_value(raw, system)`
* A null system that keeps each value in its input unit and has no `.value`

## Decision Outcome

Chosen option: "a null system that keeps each value in its input unit",
because it is the only option where nothing converts unless units differ or a
system is chosen, while bound systems keep every guarantee of ADR-003 and all
four checkers enforce the null system's rules. The prototypes and measurements
are in [opt-in unit systems](../design/opt-in-unit-system.md).

```python notest
x = Length[float](2, u.nm)  # Length[float, NoSystem]; stores 2.0 in nm
f = Energy[float](3, u.eV) / x  # Force; stores and shows 1.5 eV/nm
x + Length[float](3, u.angstrom)  # 2.3 nm: the right operand converts to nm
x.value  # type error, and a TypeError at runtime
x.to_system(Real).value  # 20.0: choosing a system converts once
x + Length[float, SI](1, u.m)  # Length[float, SI]: x adopts SI
Length[float, SI](1, u.m) + Length[float, Real](1, u.m)  # type error, and a TypeError
```

The rules for null-system quantities:

* Construction and decoding store the number in the unit it was given in.
  Units with an offset (°C, °F) store kelvin.
* `+`, `-`, and the multi-operand `quantype.numpy` functions convert the other
  operands into the first operand's unit.
* Products, quotients, and powers compose their units. A composed unit is a
  product of catalogue units with exponents, so an identical unit cancels
  exactly: (eV/nm)·nm is eV. A unit of the right operand that would cancel
  one of the left's dimension converts into it, as a sum's right operand
  does: eV·nm/Å is eV and nm·fs/ps is nm. A product so shows no unit its
  operands don't, and a repeated product such as `q * x / y` keeps one unit
  instead of growing nm^n/Å^n until its scale overflows. When its scale is the
  kind's unit in a display system, it takes that name, so J/m is N and
  kcal/mol/Å × Å is kcal/mol. Unnamed products, such as nm × ps, store a
  composed unit too. A dimensionless result stores plain numbers, so ns/fs is
  1e6, and it composes with nothing. Composed symbols parse back.
* `==` and ordering compare as arithmetic does: a null-system quantity adopts
  the other's system, so `q == q.to_system(S)`. Stored numbers compare
  directly when both sides share a storage unit. Otherwise the two are equal
  when either converts exactly onto the other, which keeps a round trip
  through a system equal whichever way it rounded, and ordering compares in
  the larger unit. Quantities in two unit systems are unequal. Quantities
  are unhashable.
* A product, quotient, or power displays its unit composed the same way from
  its operands' displayed units, the left operand's winning where they
  cancel: `1.5 eV/nm`,
  `9.0 Da·nm^2/ps^2`. A constant takes the base units its operand uses, so
  h·c over 500 nm shows in eV. Nothing chooses a system for it:
  `.simplified()` merges units of one dimension and shows a named kind in a
  display system that holds every unit left, and `.to(S)` shows `S`'s unit.
* Combined with a quantity in a unit system, a null-system quantity adopts
  that system, as constants do: arithmetic, comparisons, and the
  `quantype.numpy` functions convert it into the system first. Two unit systems
  still don't mix, and parameters don't adopt: a function typed
  `Length[float, SI]` rejects a null-system length.
* `.value` and `from_value(raw, system)` exist only for a unit system.
  `apply(function)` replaces `with_value` on every quantity.
  `x.to_system(S)` converts once. `q.to_system(NoSystem)` records `S`'s unit
  instead of converting.
* `quantype.numpy` rules work on stored numbers. Only functions that need
  radians or plain numbers (trigonometric, `exp`, `log`) convert, and only when
  the input is stored in another unit.
* JAX pytrees and `ujax`/`utorch` differentiation see reference units, so `jit`
  traces once and `lax.cond` works across units. As ADR-004 already accepts,
  values coming out of a transformation display in reference units.

Statically, `NoSystem` is the default system parameter and is not a
`UnitSystem`. `AnySystem` is the alias `UnitSystem | NoSystem`, the bound of
the system parameter, so it names no system of its own. `.value` is a
descriptor whose `__get__` only accepts `Quantity[Any, W, T]` with
`T: UnitSystem`, and `from_value` takes `type[T]` with `T: UnitSystem`. Code
that accepts both bound and null quantities is generic over `[S: AnySystem]`.
Each operator, and each protocol member that names a product, takes an operand
in its receiver's system or in none, and has an overload for a null-system
receiver whose result takes the operand's system. The `quantype.numpy`
functions of operands of one kind have per-kind overloads that do the same.

This supersedes ADR-003, whose system type parameter carries over unchanged,
and replaces one rule of [ADR-004](ADR-004-display-unit-is-presentation.md): products,
ratios, and powers of null-system quantities show their operands' units
composed instead of the system's unit.

### Consequences

* Good, because nothing converts by default. In the prototype, constructing a
  million-element array from nm took 3 µs instead of 395 µs, reading it back in
  nm 0.5 µs instead of 382 µs, and unbinding a `Real` force 2 µs instead of
  961 µs.
* Good, because data given in a unit round-trips exactly through
  construction, `.magnitude(unit)`, and `to_dict`, and every unit a product
  shows in parses back.
* Good, because `.value` exists only where a system is named, so it still has
  one meaning for each static type.
* Good, because mypy, Pyright, Pyrefly, and ty all reject `.value` and
  `from_value` without a system, and mixing two unit systems.
* Good, because a literal such as `1.0 * u.nm` works in code that chose a
  system: it adopts that system.
* Good, because `==`, ordering, subtraction, and `to_system` agree: a
  quantity equals itself converted into any system and back, and
  `1 nm == 10 Å`.
* Bad, because operations across units convert: adding a million nm values to
  Å values took 1,013 µs instead of 513 µs. Scalar products cost 0.72 µs
  instead of 0.58 µs, mostly to choose the unit a product shows in.
* Bad, because a composed unit can be unfamiliar: kinetic energy from Da, nm,
  and ps shows as Da·nm^2/ps^2, though that is about a kJ/mol, and
  `.magnitude()` returns numbers in it. `.simplified()`, `.to(unit)`, and
  `.to(S)` show another.
* Bad, because adoption needs a second overload for each operator and product
  relation: the stubs grew from 17 MB to 27 MB. Each named kind moved into its
  own stub module to stay under Pyright's per-module complexity limit, which
  left ty and Pyrefly faster on a user file than before (0.10 s and 0.26 s,
  from 0.25 s and 0.39 s) and Pyright and mypy about as fast. Pyright checking
  the whole repository needs a 12 GB heap.
* Bad, because quantities are unhashable, scalars included, so they can't be
  dict keys, set members, `functools.cache` arguments, or `jax.jit` static
  arguments; code keys on `q.magnitude(unit)` instead. Equality that adopts a
  system admits no hash: `1 nm` equals its SI and `Real` conversions, which
  are unequal to each other. Hashing by value in the reference units, as
  before, left `q == q.to_system(SI)` false for about a quarter of values and
  let a cache return a result in another system. Equality is also not
  transitive across systems, and is floating-point equality, so two distinct
  nm values can each equal the same Å value.
* Bad, because it breaks existing code, with no deprecation period while
  quantype is before 1.0: `.value` on default quantities, `from_value`, which
  now takes the system, `with_value`, which is gone, `.magnitude()` of a
  product, and annotations that assume `Atomistic`. `Length[float]` changes
  meaning in place. 177 of 1,085 runtime tests fail in the prototype for these
  reasons, and the docs use `.value` 45 times and `from_value` 12 times.
* Bad, because generic code chooses between two bounds: `UnitSystem` when it
  reads raw numbers, `AnySystem` when it accepts anything.
* Bad, because `apply` can't change backend, so moving NumPy numbers into Torch
  means constructing a new quantity from `.magnitude(unit)`; see
  [replacing `with_value`](#replacing-with_value).
* Bad, because checker messages for `.value` on a null-system quantity name a
  descriptor, not the fix. mypy says `Value of type variable "T" of "__get__"
  of "_Value" cannot be "NoSystem"`, and Pyright that the argument isn't
  assignable to `None`.
* Bad, because arrays share memory with their input in more cases: today only
  when the input unit is the system's unit, afterwards for any unit.
  `.magnitude(unit)` in the stored unit returns the stored array itself.
* Bad, because arrays in offset units still convert to kelvin.
* Bad, because crossing into a JAX transformation rescales numbers stored in
  another unit, outside `jit`: they round on the way, and a donated input
  donates the rescaled copy.
* Bad, because ty solves a function's system parameter as a union, such as
  `SI | Real`, where the other checkers reject the call, so
  `qnp.maximum(si, real)` passes ty. It does so before this decision too.
* Bad, because Pyright and ty don't adopt a generic system from a literal on
  the right of a product: with `d: Length[float, S]`, `d / (1.0 * u.fs)` has
  no named kind in either, as neither solves the right operand's protocol
  member against `S`. The literal goes on the left, or takes
  `.to_system(d.system)`.

### Catalogue scales round once

Catalogue scales were computed in floating point from exact definitions, and 46
of the 156 exact ones were not the nearest float to their exact value (1 to 5
ulp). The picosecond was `999.9999999999999` fs, so 3 ps was stored as
`2999.9999999999995` fs. Null-system quantities compose and compare through
these scales, so the errors would reach users more often. Each scale, and each
physical constant, is now computed exactly and rounded once. That fixes all 46
and makes N = kg·m/s², Pa = N/m², ns = 1000 ps, and s = 1000 ms hold. Atomic
momentum, which uses ħ, changed too, so 47 of the 162 scales differ.

Conversions apply the ratio of two scales as a factor of at least 1,
multiplying or dividing, so an exact ratio such as 10 stays exact both ways:
0.7 nm is 7.0 Å, not 6.999999999999999.

### Not addressed here

Exact rational scales. Rounding each scale once still leaves identities
between rounded scales that fail, such as g = kg/1000 and J = N·m, and
composed units multiply rounded scales. Carrying an exact rational scale on
each `Unit` whose definition is exact would make every such identity hold,
let composed units round once, and let catalogue lookup use exact equality.
Units involving π or √ (degree, the ASE units, atomic momentum, ħ) would keep
a float scale. This is a separate refactor of `Unit` and the catalogue.

### Replacing `with_value`

`with_value` is the escape hatch for operations quantype has no rule for:
`x.with_value(np.sort(x.value))`, `x.with_value(x.value.detach())`. It trusts
the new numbers to be in the quantity's system, so it can't work without one.
The options are:

* `x.apply(function)`: apply a function to the stored numbers, keeping the
  kind, storage unit, and display unit, as in `x.apply(np.sort)` or
  `x.apply(lambda t: t.detach())`. Nothing converts, and it works the same on
  bound quantities. The function must give the same physical result whatever
  unit the numbers are in, which holds for sorting, slicing, reshaping,
  detaching, and moving to a device, and fails for `np.log` or adding a
  constant. Nothing can check that.
* `x.with_magnitude(numbers, unit)`: new numbers in a named unit, keeping the
  kind and display unit, as in
  `x.with_magnitude(np.sort(x.magnitude(u.nm)), u.nm)`. It's always correct,
  but longer, and converts twice when `unit` isn't the storage unit.
* Bind, operate, unbind:
  `x.to_system(Atomistic).with_value(...).to_system(NoSystem)`. It needs no new
  method and converts once, on the way in.
* Methods for the common cases, such as `detach()` and a device move, with
  `quantype.numpy` covering the rest. That shrinks the need for an escape
  hatch without removing it.

Chosen: `apply`, on every quantity, documented with its contract, in place
of `with_value` everywhere. Its function returns numbers of the same backend,
which every checker can express as `Callable[[V], V]`; a generic result type
made ty and Pyrefly infer unions for overloaded functions such as `np.sort`.
`from_value(raw, system)` takes the system as an argument, so a bound on its
type parameter rejects `NoSystem` in every checker. `with_magnitude` can
follow if code turns up that has its numbers in a unit it can name.

### Names and display order

* The null system is `NoSystem`, which reads clearly in every message
  (`Length has no raw numbers without a unit system`). `Unspecified` and `Portable`
  were alternatives. The prototype used `Unbound`, which collides with
  "unbound type variable" in checker output. `Free` was rejected.
* A composed unit takes the name of the kind's unit in the first of
  `Atomistic`, `Metal`, `Real`, `SI`, `CGS`, and `Atomic` whose scale it has.
  Any catalogue unit of the same scale was rejected: nm/ps × fs would show
  as pm. Showing every product in a display system's units, with `Atomistic`
  as the fallback, was tried and rejected: eV/nm showed as eV/Å, choosing a
  system for a quantity that has none.

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
four checkers enforce the null system's rules with no change in stub size or
checking time. The prototypes and measurements are in
[opt-in unit systems](../design/opt-in-unit-system.md).

```python notest
x = Length[float](2, u.nm)  # Length[float, NoSystem]; stores 2.0 in nm
f = Energy[float](3, u.eV) / x  # Force; stores 1.5 in eV/nm, no conversion
x + Length[float](3, u.angstrom)  # 2.3 nm: the right operand converts to nm
x.value  # type error, and a TypeError at runtime
x.to_system(Real).value  # 20.0: choosing a system converts once
x + Length[float, SI](1, u.m)  # type error, and a TypeError at runtime
```

The rules for null-system quantities:

* Construction and decoding store the number in the unit it was given in.
  Units with an offset (°C, °F) store kelvin.
* `+`, `-`, and the multi-operand `quantype.numpy` functions convert the other
  operands into the first operand's unit.
* Products, quotients, and powers compose their units and never convert. A
  catalogue unit with the same scale is used when one exists, so
  kcal/mol/Å × Å stores kcal/mol. Unnamed products, such as nm × ps, store a
  composed unit too.
* `==`, ordering, and `__hash__` compare stored numbers when both sides share a
  storage unit, and values in the reference units otherwise.
* A single product or quotient displays its own symbol (`1.5 eV/nm`,
  `6.0 nm·ps`). A nested one of a named kind, such as Da·((Å/fs)^2) for an
  energy, displays in the kind's unit in a system that stores every operand's
  kind in the operand's unit (`207.29 eV`). When several systems do, the first
  in a fixed order wins; when none does, as for g × (m/s)², a fixed fallback
  system does. Neither depends on operand order, so `a * b` and `b * a`
  display alike. Storage is unchanged.
* `.value`, `from_value`, and `with_value` exist only in a bound system.
  `x.to_system(S)` converts once. `q.to_system(NoSystem)` records `S`'s unit
  instead of converting.
* `quantype.numpy` rules work on stored numbers. Only functions that need
  radians or plain numbers (trigonometric, `exp`, `log`) convert, and only when
  the input is stored in another unit.
* JAX pytrees and `ujax`/`utorch` differentiation see reference units, so `jit`
  traces once and `lax.cond` works across units. As ADR-004 already accepts,
  values coming out of a transformation display in reference units.

Statically, `NoSystem` is the default system parameter and is not a
`UnitSystem`. `.value` is a descriptor whose `__get__` only accepts
`Quantity[Any, W, T]` with `T: UnitSystem`. Code that accepts both bound and
null quantities is generic over `[S: AnySystem]`.

This supersedes ADR-003, whose system type parameter carries over unchanged,
and replaces one rule of [ADR-004](ADR-004-display-unit-is-presentation.md): products,
ratios, and powers of null-system quantities show their composed unit instead
of the system's unit.

### Consequences

* Good, because nothing converts by default. In the prototype, constructing a
  million-element array from nm took 3 µs instead of 395 µs, reading it back in
  nm 0.5 µs instead of 382 µs, and unbinding a `Real` force 2 µs instead of
  961 µs.
* Good, because data round-trips exactly through construction,
  `.magnitude(unit)`, and `to_dict`.
* Good, because `.value` exists only where a system is named, so it still has
  one meaning for each static type.
* Good, because stubs and checking time are unchanged, and mypy, Pyright,
  Pyrefly, and ty all reject `.value` on a null-system quantity and mixing it
  with a bound one.
* Good, because equality across units is symmetric and agrees with `__hash__`:
  `{Length(1, u.nm), Length(10, u.angstrom)}` has one element.
* Bad, because operations across units convert: adding a million nm values to
  Å values took 1,013 µs instead of 513 µs. Scalar products cost 0.64 µs
  instead of 0.53 µs.
* Bad, because equality across units is floating-point equality in reference
  units. Two distinct nm values can each equal the same Å value, and `a == b`
  can hold while `a - b != 0`.
* Bad, because it breaks existing code, with no deprecation period while
  quantype is before 1.0: `.value` and `from_value` on default quantities, and
  annotations that assume `Atomistic`. `Length[float]` changes meaning in
  place. 177 of 1,085 runtime
  tests fail in the prototype for these reasons, and the docs use `.value` 45
  times and `from_value` 12 times.
* Bad, because generic code chooses between two bounds: `UnitSystem` when it
  reads raw numbers, `AnySystem` when it accepts anything.
* Bad, because `with_value` trusts raw numbers in a system, so a
  null-system quantity needs another escape hatch for operations quantype has
  no rule for; see [replacing `with_value`](#replacing-with_value).
* Bad, because checker messages for `.value` on a null-system quantity name a
  descriptor, not the fix. mypy says `Value of type variable "T" of "__get__"
  of "_Value" cannot be "NoSystem"`.
* Bad, because arrays share memory with their input in more cases: today only
  when the input unit is the system's unit, afterwards for any unit.
* Bad, because arrays in offset units still convert to kelvin.
* Bad, because only `.value` is rejected statically. `from_value` and
  `with_value` raise `TypeError` at runtime without a system, but their stubs
  are unchanged, so checkers accept those calls.

### Catalogue scales round once

Catalogue scales were computed in floating point from exact definitions, and 46
of 156 were not the nearest float to their exact value (1 to 5 ulp). The
picosecond was `999.9999999999999` fs, so 3 ps was stored as
`2999.9999999999995` fs. Null-system quantities compose and compare through
these scales, so the errors would reach users more often. Each scale, and each
physical constant, is now computed exactly and rounded once. That fixes all 46
and makes N = kg·m/s², Pa = N/m², ns = 1000 ps, and s = 1000 ms hold.

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
the new numbers to be in the quantity's system, so bound quantities keep it.
For null-system quantities the options are:

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

Chosen: `apply`, on every quantity, documented with its contract.
`with_magnitude` can follow if code turns up that has its numbers in a unit it
can name.

### Names and display order

* The null system is `NoSystem`, which reads clearly in every message
  (`Cannot combine NoSystem and SI quantities`). `Unspecified` and `Portable`
  were alternatives. The prototype used `Unbound`, which collides with
  "unbound type variable" in checker output. `Free` was rejected.
* Nested compositions display in the first of `Atomistic`, `Metal`, `Real`,
  `SI`, `CGS`, and `Atomic` that stores each operand's kind in the operand's
  unit, with `SI` as the fallback, so g × (m/s)² displays in J.

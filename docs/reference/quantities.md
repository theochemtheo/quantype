# Quantities

Public imports: `from quantype import Length, Energy, Force, Quantity, u` and
`from quantype.systems import SI, Atomistic, NoSystem`.
The rules below apply to all named quantity classes. The
[generated catalogue](catalogue.md) supplies the exact
[kinds](../explanation/glossary.md#kind "A nominal physical meaning, not just a dimension vector.")
and relations; this page describes the shared handwritten API and generated
type refinements.
Each Python fence is independently executable.

## Construction and storage

The constructor form is `Length[V, S](value, unit, *, dtype=..., display=True)`;
`S` defaults to `NoSystem`. `V` is real floating storage: Python `float`, a NumPy
floating scalar or array, `jax.Array`, or `torch.Tensor`. Lists and integers can
be converted by typed constructors. Boolean, complex, and nonnumerical inputs
are rejected. Scalar targets reject nonscalar input; NumPy array storage also
supports zero-dimensional arrays. NumPy annotations encode dtype; JAX and Torch
use `dtype=`. Optional backends require their respective Quantype extras.

| Entry point | Accepted input | Result and copying boundary |
| --- | --- | --- |
| `Length[V, S](value, unit)` | Numerical scalar, array, or sequence and a Length unit | Converts storage to `V` and units to `S`; in `NoSystem`, keeps input storage units except offset temperatures |
| `Length(value, unit)` | Numerical input and a Length unit | Infers storage from floats/arrays; for integer static precision use `Length[float]` |
| `u.nm(value)` or `value * u.nm` | Real scalar or floating array, not a list or integer array | `Length` without a system; compatible floating storage is retained, Python integers become floats |
| `Length.from_value(value, system)` | Trusted numbers already in a `UnitSystem`'s units | Preserves raw storage without coercion or copying; not an external validation boundary |
| `Length.parse(data, *, units=())` | Scalar string or quantity object | Float or float64 NumPy array; choose a `TypeAdapter` for specific storage |

A NumPy array already of the requested dtype is kept when conversion is not
needed. Unit rescaling may allocate. Typed storage conversion can allocate;
Torch conversion preserves device and graph connectivity where supported.
A conflicting explicit NumPy dtype fails. JAX float64 requests fail when x64 is
unavailable instead of silently becoming float32. Float16/float32 NumPy system
conversions can emit `StorageRangeWarning` on overflow or subnormal results.

Unit-first construction rejects boolean, complex, nonnumerical and integer
array storage and backend scalars that are not supported floating storage;
use typed constructors to convert integer arrays and lists. `unit(array)` works
across backends; `array * unit` is guaranteed only for NumPy.

Conversions use higher working precision where the backend permits, then round
into the requested storage. Display conversion likewise leaves stored dtype
unchanged. Lost input precision cannot be recovered; a conversion factor outside
the working range raises `ValueError`. Float storage preserves the scalar input
for display where possible. `display=False` omits the remembered input display,
allowing the default system/storage presentation.

!!! warning "Shared arrays retain shared ownership"

    Construction and same-storage-unit `.magnitude(unit)` may expose the very same
    array, not a copy. Mutating that array mutates the quantity's numbers. Copy with
    `q.copy()` or copy the input to separate ownership. `from_value` retains its
    exact input, including a NumPy array's read-only flag, and performs no validation
    or conversion. It requires a real `UnitSystem`, not `NoSystem`.

```python
from typing import assert_type

from quantype import Energy, Force, Length, u
from quantype.systems import Atomistic

length = Length[float](2, u.nm)
assert_type(length, Length[float])
force = Energy[float](3, u.eV) / length
assert_type(force, Force[float])
bound = length.to_system(Atomistic)
assert_type(bound, Length[float, Atomistic])
assert bound.value == 20.0
```

## Conversion and numerical access

Signatures below use `V` for storage and `S` for the quantity's system:

| Operation | Result | Errors and conversion |
| --- | --- | --- |
| `q.magnitude(unit=None) -> V` | Plain numbers in the supplied or display unit | Wrong-kind unit raises `TypeError`; rescaling may allocate; same storage unit avoids conversion |
| `q.to(unit_or_system) -> Self` | New presentation, same kind, system, storage | A system here selects its display unit only; wrong-kind unit fails |
| `q.simplified() -> Self` | Simpler presentation of composed units | Does not rescale stored numbers |
| `q.to_system(T)` | Same kind/storage type, system `T` | Converts stored numbers; may allocate; the explicit bridge between systems |
| `q.value -> V` | Raw stored numbers | Available only in a bound `UnitSystem`; `NoSystem` raises `TypeError` and is rejected statically |
| `q.apply(function: Callable[[V], V]) -> Self` | Same physical metadata, transformed raw numbers | Caller must ensure unit-invariant meaning and preserve backend; no physical validation of the function |

!!! info "Presentation changes are not storage conversions"

    `.to(unit)` and `.to(System)` select a
    [display unit](../explanation/glossary.md#display-unit "The unit used for presentation and default magnitude extraction."),
    leaving raw numbers and the system unchanged. Use `.to_system(T)` to rescale
    storage for a different system; use `.magnitude(unit)` to extract plain numbers.
    The [system adoption rules](systems.md#system-selection-and-adoption) specify
    when arithmetic can do that conversion for an unbound operand.

`.unit` is the presentation unit (or `None` for a dimensionless value);
`.system` is the system class. Accessing shape/dtype and other array properties
requires appropriate numerical storage. Implicit array conversion, including
`np.asarray(q)`, raises: extract numbers explicitly.

```python
from quantype import Length, u
from quantype.systems import SI

length = Length[float](2, u.nm)
shown = length.to(u.angstrom)
assert shown == length
assert shown.magnitude() == 20.0
assert length.to_system(SI).value == 2e-9
```

## Algebra, comparison, and errors

Generated overloads determine named products, quotients, powers, and inverse
relations. For example, `Energy[float] / Length[float]` is `Force[float]`.
Same-kind addition retains the left display unit. Plain real values and arrays
scale a quantity without changing its dtype. Structural products remain typed
when no named relation exists.

Different nominal kinds cannot be added, even with matching dimensions.
`Kind.reinterpret(q)` preserves numerical meaning but explicitly renames the
kind, requiring equal dimensions. Two different bound systems cannot be
combined; `.to_system(T)` resolves that boundary. `NoSystem` adopts a bound
operand's system. These rules apply to all four supported type checkers and at
runtime.

Ordering returns backend booleans and requires compatible kinds/systems.
Equality is elementwise for arrays; different kinds or bound systems compare
unequal. In different units, equality holds when either converts exactly onto
the other, so a quantity equals its conversion into a system and back.
`Dimensionless` quantities compare with plain numbers; other kinds reject plain
numbers except integer zero with `==` / `!=` (not absolute temperatures).
`None` and other nonnumbers compare unequal. Quantities are unhashable and their
truth value raises: compare explicitly or use `q is not None`. Cache keys need
an appropriate hashable numerical representation in a named unit.

Absolute temperatures are affine points: subtraction returns a difference,
adding a difference returns a point, and point addition/sums are errors.
See [temperature calculations](../how-to/temperatures.md) and the
[physical model](../explanation/physical-model.md#temperatures-are-points).

## Arrays and static support

NumPy, JAX, and Torch arrays retain their backend through quantity arithmetic.
Use `quantype.numpy` for typed numerical calls. NumPy and Torch native dispatch
supports registered calls at runtime, but NumPy's stubs reject quantities; JAX
native `jax.numpy` does not dispatch quantities. Unsupported calls raise rather
than silently discard physical metadata. Array shape inference is not part of
the static contract.

The [numerical dispatch contract](numerical-operations.md#runtime-dispatch-versus-static-support)
and [operation tables](numerical-operations.md#dispatch-and-physical-results)
define supported functions, result kinds, and view/copy behavior. See
[array tasks](../how-to/arrays.md) for indexing, scaling, formatting, `apply`,
and backend conversion; [type annotations](type-annotations.md) for runtime
annotation discovery. For persistence contracts, see
[serialization](serialization.md).

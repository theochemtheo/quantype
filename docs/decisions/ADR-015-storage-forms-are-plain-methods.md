# Storage forms are plain methods on every quantity class

## Context and Problem Statement

A library that batches quantities needs the array form of a quantity type:
`Energy[float, S]` read from many records is an
`Energy[npt.NDArray[np.float64], S]`. Python's type system cannot map
`Kind[float, S]` to `Kind[Array, S]` once for all kinds, so the map has to come
from each class. The stubs already had array-typed results, such as the
`__mul__` overload for an integer array, but ty does not solve a protocol
through a method with a typed `self`, overloads, or its own type parameters:
reading `B` off `__mul__` gave `Unknown`. `quantype.numpy.stack` had the same
gap and typed a stack of `Energy[float, S]` as `Energy[float, S]`, although at
runtime its storage is an array. How should a quantity class name its
array form?

## Considered Options

* A generated `to_numpy()` on every quantity class, declared without type
  parameters, overloads, or a typed `self`, and a public protocol,
  `quantype.typing.SupportsToNumpy[B]`, that reads its result
* Read the form off existing array-typed operator overloads
* Make `Quantity[EnergyKind, V, S]` assignable to `Energy[V, S]`, so that one
  generic signature on the base class covers every kind
* `to_numpy()`, `to_torch()`, and `to_jax()` together

## Decision Outcome

Chosen option: "a generated `to_numpy()` and a public protocol", because it is
the only form all four checkers solve, and `quantype.numpy.stack` can use it to
type float storage correctly.

`to_numpy()` returns the quantity with float64 NumPy storage, keeping kind,
unit, and system. Float64 NumPy storage is returned as the same array, and
other storage is copied. A Torch tensor must be detached and on the CPU. The
renderers emit it beside `item()` for named kinds, product classes, and the
structural base of a generated catalogue. A storage-form method must stay plain:
a typed `self`, an overload, or a type parameter on it makes ty infer
`Unknown` for every consumer of `SupportsToNumpy`. ty needed a separate protocol
member for float storage in
[ADR-006](ADR-006-protocol-dispatch-for-product-stubs.md) for the same reason;
like the rest of the algebra, this works through the stubs alone
([ADR-002](ADR-002-generated-stubs-for-four-checkers.md)).

The Torch and JAX forms are deferred until a consumer needs them. Each kind stub
would have to import `torch` and `jax`, which only `utorch.pyi` and `ujax.pyi`
do today, so a NumPy-only user's checker would load both.

### Consequences

* Good, because typed batching libraries map any quantity type, including
  products and kinds from application catalogues, to its array form without a
  table of kinds.
* Good, because `qnp.stack` of float storage now has array storage in all four
  checkers, as at runtime.
* Bad, because the stubs grow by two lines per class, about 3,500 lines in
  all.
* Bad, because `qnp.stack` of quantities with `Any` storage now matches several
  overloads. ty infers `Unknown`, as the typing spec asks of an ambiguous `Any`
  match, where it inferred the argument type before; Pyright takes the first
  overload and claims float64 NumPy storage.
* Bad, because the renderers carry a rule that no checker enforces: changing
  `to_numpy` into a generic or overloaded method silently breaks ty consumers.
  `tests/typing/positive/storage.py` fails under ty if it does.

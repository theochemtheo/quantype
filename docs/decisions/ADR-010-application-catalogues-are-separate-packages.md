# Application catalogues generate their own package

## Context and Problem Statement

Applications need kinds and relations quantype doesn't declare, such as a
`SurfaceTension` that `Pressure * Length` gives. The static algebra lives in
generated stubs ([ADR-002](ADR-002-generated-stubs-for-four-checkers.md)), which
an application can't add overloads to: a subclass's `__rmul__` is hidden by the
left operand's generic fallback
(`tests/typing/probes/extension_operators.py`). How do applications extend the
catalogue so that type checkers see their relations?

## Considered Options

* Subclass `Quantity` in application code and write operator overloads by hand
* Register relations at runtime, with no static types
* Generate a complete package from `builtin_catalogue().extend(...)`, with its
  own classes, stubs, units, `numpy` module, products, and autodiff adapters

## Decision Outcome

Chosen option: "generate a complete package", because it is the only option
where all four checkers see the application's relations, with the same API as
quantype itself.

Unit systems are defined over base axes, so every kind in a generated catalogue
has a unit in every system without extra definitions.

### Consequences

* Good, because an application's kinds get named results, products, `qnp`
  functions, and autodiff types, checked like quantype's own.
* Bad, because `labquantities.Length` is a different type from
  `quantype.Length`. A library typed with quantype's classes rejects the
  application's quantities, and two catalogues can't exchange quantities.
* Bad, because the generated package is large and must be regenerated when
  quantype changes.

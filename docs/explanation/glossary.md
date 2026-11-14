# Glossary

Short definitions for terms used throughout the guides. Linked terms also show
a brief definition on hover; follow the link for context. The
[physical model](physical-model.md) explains how these concepts fit together.

## Kind

A declared physical meaning, such as `Length`, `Pressure`, or `EnergyDensity`.
Kinds are nominal: having the same dimensions does not make two kinds
interchangeable. See [nominal meanings](physical-model.md#kinds-are-nominal-meanings).

## Dimensions

The exponents describing how a quantity scales with the base physical axes.
They determine unit compatibility, but do not alone determine a result's kind.
The [catalogue](../reference/catalogue.md#kinds) lists each kind's dimensions.

## Storage unit

The unit in which a quantity's numerical data is actually stored. Without a
unit system, it comes from the input or from composing the operands' units.
In a unit system, the system fixes it for each kind. See
[construction and storage](../reference/quantities.md#construction-and-storage).

## Display unit

The unit used to present a quantity, including by `.magnitude()` without an
argument. It can differ from the storage unit. `.to(unit)` changes presentation,
not stored numbers or the unit system. See
[conversion and numerical access](../reference/quantities.md#conversion-and-numerical-access).

## Unit system

A convention fixing the units of raw stored numbers for each physical kind.
For example, an SI length stores meters, so `.value` has an unambiguous meaning.
The default `NoSystem` does not fix a shared convention and disallows `.value`.
See [system contracts](../reference/systems.md).

## Backend

The numerical implementation holding and operating on a quantity's data:
Python scalars, NumPy, JAX, or Torch. Backend, dtype, and device are numerical
choices, not physical kinds. See
[numerical operations](../reference/numerical-operations.md) and
[storage construction](../reference/quantities.md#construction-and-storage).

## Affine point

A value measured from an origin, rather than an increment. An absolute
`Temperature` is a point; subtracting two gives a `TemperatureDifference`.
Adding two absolute temperatures is undefined in this API. See
[temperatures as points](physical-model.md#temperatures-are-points).

## Structural product

A physical result whose declared factor structure is retained rather than
replaced by a named kind. Examples include `LengthTime` and more general
`Quantity[Mul[...], ...]` types. Equal dimensions do not supply a missing relation.
See [declared relations and structural products](physical-model.md#declared-relations-and-structural-products).

## Validation boundary

The point where external data is checked and converted into an intended
quantity type. `Length.parse`, a Pydantic field, and `TypeAdapter` are examples.
`from_value` is not a validation boundary: it trusts that raw numbers already
use the named system's units. See [validate data](../how-to/validate-data.md).

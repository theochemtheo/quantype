# Reference

Consult exact public contracts here; use [How-to guides](../how-to/index.md)
for executable task recipes. Handwritten reference covers runtime definitions
and generated stub refinements; the built-in catalogue comes from its renderer.

| Concern | Public entry points |
| --- | --- |
| [Quantities](quantities.md) | `Length[V, S]`, `Quantity`, construction, conversion, storage, and algebra |
| [Units and constants](units.md) | `u`, unit lookup/definition, presentation, physical constants, CODATA, and numerical ranges |
| [Unit systems](systems.md) | `UnitSystem`, `NoSystem`, built-ins, overrides, and generic systems |
| [Numerical operations](numerical-operations.md) | `quantype.numpy`, array methods, runtime dispatch versus static support |
| [Serialization](serialization.md) | JSON, Pydantic, NPZ/NPY, explicit custom-unit decoding |
| [Autodiff](autodiff.md) | `ujax`, `utorch`, derivative kinds, storage, and transformation restrictions |
| [Quantity annotations](type-annotations.md) | `quantype.typing.quantity_type` |
| [Built-in catalogue](catalogue.md) | Generated kinds, units, declared relations, and each system's storage units |

JAX and Torch require their respective extras; the
[autodiff adapters](autodiff.md) describe these backend contracts.
Backend and static limitations remain explicit on the relevant pages.

---
name: quantype-serialization
description: Implement Quantype JSON and Pydantic validation, custom-unit decoding, and pickle-free NPZ/NPY value boundaries. Use for quantity wire schemas, archive restoration, dtype/shape round trips or malformed-payload handling.
---

# Quantype Serialization

Parent: read [Quantype shared guidance](../quantype/SKILL.md). Public behavior is documented in `docs/serialization.md`; implementations live in `src/quantype/serialization.py` and `_internal/_validation.py`.

## Explicit value boundary

Serialization transfers values to host storage; it does not preserve graphs, device placement or model state. `_internal/_storage.host_array` intentionally detaches Torch tensors and copies them to CPU. Keep that behavior at the boundary; read [backends](../quantype-backends/SKILL.md) if changing numerical conversion shared with construction.

Serialization currently requires a named semantic kind. Structural expression quantities have no named wire representation. Selected units follow an explicit override, then display choice, then canonical units. Noncanonical conversion may introduce floating-point roundoff; these formats are not bit-exact model snapshots.

## JSON and Pydantic

Quantity objects contain exactly `kind`, `magnitude` and `unit`. Validate nominal kind as well as unit compatibility and real numerical magnitudes. Scalar strings such as `"5 angstrom"` require an expected quantity kind. JSON does not carry backend, device or dtype metadata.

`QuantityClass.parse` returns scalar/float64-array storage. Typed Pydantic fields and `TypeAdapter` convert into the annotated storage, including NumPy dtype; bare quantity annotations have a runtime schema but are not a substitute for precise static storage annotations. Preserve scalar versus zero-dimensional array schema and restoration behavior.

Custom units are explicit decoding inputs: `units=(...)` for parsing/archives and `context={"units": (...)}` for Pydantic. Reject duplicate definitions and built-in identifier shadowing. Do not install custom definitions into a global public namespace. For definition/affine conversion changes, read [core](../quantype-core/SKILL.md).

## Binary arrays

NPZ stores numerical arrays plus a JSON string under `metadata`. Current metadata is `{"version": 1, "quantities": {...}}`; each named entry requires string `kind`, `unit`, and `array` fields and optionally string `source_backend`. Check the implementation before changing schema or version behavior.

Load with `allow_pickle=False`, reject object/nonnumerical arrays, and validate metadata version, exact allowed fields, quantity identity, unit identity and referenced array existence. Preserve dtype and shape, including zero-dimensional and empty arrays, subject to the explicit requested target storage.

`load_npz` requires a parameterized quantity target and converts into that storage. `source_backend` is provenance, never an instruction to import or reconstruct its source backend. A Torch-produced archive can load into NumPy without Torch installed. Explicit unavailable dtypes and unsupported host conversions must fail rather than approximate silently.

NPY carries only numerical data: callers must supply an external unit when constructing the quantity. Do not infer physical meaning from a filename or array dtype.

## Validation

Use `tests/runtime/test_validation.py` for JSON, Pydantic, nominal distinctions and custom units; `test_binary_failures.py` for malformed archives and storage failures; and `test_refactor.py` for NPZ dtype/shape and backend round trips. For generated catalogue decoding, also use `test_codegen.py`. Public annotation changes need [typing](../quantype-typing/SKILL.md).

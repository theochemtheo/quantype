---
name: quantype-backends
description: Implement Quantype numerical storage conversion, constructors, dtype handling, reductions and lazy optional-backend integration for Python, NumPy, JAX and Torch. Use for backend dispatch or storage-preservation bugs.
---

# Quantype Backends

Parent: read [Quantype shared guidance](../quantype/SKILL.md). For a change to physical arithmetic meaning, also read [core](../quantype-core/SKILL.md).

## Construction versus numerical dispatch

`src/quantype/_internal/_construction.py` uses `StorageAlias` to make `Length[V](raw, unit)` an explicit conversion boundary. `_internal/_storage.py` resolves aliases and dispatches storage conversions. Convert the input into the requested storage, apply canonical unit conversion, and retain the requested storage after conversion. Typed construction requires an input unit.

`from_canonical` and `core._wrap` are trusted, coercion-free boundaries. Do not move constructor validation into wrapping: compiled operations and JAX tree reconstruction may carry tracers or sentinel leaves.

Unit-first construction preserves compatible existing numerical storage and promotes Python integers to floats. Use `unit(array)` in cross-backend examples; backend-left multiplication is not promised beyond NumPy. Lists should use typed constructors.

## Storage invariants

- Support real floating storage. Reject boolean, complex and nonnumerical magnitudes; integer inputs may be converted to floating storage. Python/NumPy scalar targets reject nonscalar input.
- Distinguish NumPy floating scalars from zero-dimensional arrays. `npt.NDArray[np.float64]` remains an array even after a reduction yields a scalar-shaped result.
- NumPy annotations encode dtype; an explicit conflicting `dtype=` fails. Torch/JAX storage classes need an explicit `dtype=` when requested, since their annotations do not encode dtype.
- Preserve Torch device placement and graph connectivity when converting existing tensors. List construction does not automatically enable gradients.
- JAX conversion must avoid host transfer of existing backend arrays. Unavailable explicit dtype requests, including float64 without x64 support, fail rather than silently returning a different dtype.

Arithmetic delegates to stored values' operators. `_internal/_math.py` dispatches math helpers. `core.Quantity._reduce` maps `axis`/`keepdims` to Torch's `dim`/`keepdim` and preserves NumPy array storage. Shape inference and arbitrary dtype promotion are outside the existing static contract.

Implicit NumPy coercion, ufuncs and array-function calls on quantities reject loss of physical meaning. Expose `.value` or `.magnitude(unit)` explicitly at unsupported numerical boundaries.

## Optional imports and validation

Preserve the dependency-free base import and independent optional adapters. `tests/runtime/test_refactor.py` checks lazy imports, conversion and storage. `test_optional_backends.py` blocks the other backend in a subprocess, providing stronger isolation evidence than an all-extras run alone. `test_binary_failures.py` includes invalid storage/dtype cases.

For gradients, compilation or pytree changes, read [autodiff](../quantype-autodiff/SKILL.md). For explicit host conversion via `_storage.host_array`, read [serialization](../quantype-serialization/SKILL.md); arithmetic must not reuse that detaching boundary.

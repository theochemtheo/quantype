# Quantype ownership map

Paths here are relative to the repository root. `README.md` and `docs/` describe the public contract; source and tests establish current behavior.

| Concern | Authoritative source | Relevant checks |
| --- | --- | --- |
| Built-in kinds, dimensions, conversions, relations and powers | `src/quantype/_internal/_registry.py` | `test_core.py`, `test_refactor.py`, `test_codegen.py` |
| Catalogue definition validation and immutable extension | `src/quantype/catalogue.py` | `test_codegen.py` |
| Source rendering and formatted write/check | `src/quantype/codegen/_render.py`, `_package.py`, `__init__.py`; `scripts/generate.py` | `test_codegen.py`, `scripts/generate.py --check` |
| Runtime semantic identities and expression trees | `src/quantype/_internal/_semantics.py` | `test_core.py`, `test_validation.py` |
| Quantity wrapping, operators and reductions | `src/quantype/core.py`; handwritten `core.pyi` | `test_core.py`, `test_refactor.py`, typing fixtures |
| Unit definitions and catalogue lookup | `src/quantype/_internal/_unit.py`; handwritten `_unit.pyi` | `test_core.py`, `test_validation.py` |
| Unit systems, derived units, range checks | `src/quantype/_internal/_systems.py`; public `systems.py` | `test_systems.py`, `test_scenarios.py`, typing fixtures |
| Typed constructors and backend conversion | `src/quantype/_internal/_construction.py`, `_storage.py` | `test_refactor.py`, `test_binary_failures.py` |
| Mathematical helpers | `src/quantype/_internal/_math.py`; handwritten `_math.pyi` | `test_core.py`, `test_codegen.py` |
| JSON, archive and Pydantic boundaries | `src/quantype/serialization.py`, `_internal/_validation.py` | `test_validation.py`, `test_binary_failures.py`, `test_refactor.py` |
| Optional differentiation | `src/quantype/ujax.py`, `utorch.py`; generated adjacent stubs | `test_autodiff.py`, `test_torch.py`, `test_optional_backends.py` |
| Application-catalogue runtime binding | `src/quantype/_internal/_runtime_catalogue.py`, `codegen/_package.py` | `test_codegen.py` and `tests/fixtures/generated_catalogue/` |
| Public imports | `src/quantype/__init__.py` | stubtest, positive typing, runtime imports |

Test filenames in the table live under `tests/runtime/` unless a full path is given.

## Generated versus handwritten

For the built-in package, `scripts/generate.py` uses `builtin_catalogue()` and the public generator. The renderer owns `kinds.py`, `_generated.py`, `_generated.pyi`, `units.py`, `units.pyi`, `ujax.pyi`, and `utorch.pyi`. Inspect `render(...)` for the current output set. Handwritten `core.pyi`, `_internal/_unit.pyi`, and `_internal/_math.pyi` must be maintained alongside their implementations.

Adding a built-in quantity may also require a handwritten top-level export in `src/quantype/__init__.py`. Generation does not update that file for the built-in package.

External generation emits a combined, separate nominal API with its own package scaffolding. `RuntimeCatalogue` binds its identities to shared internal lookup tables; it does not make those identities equal to the built-in kinds. Custom unit definitions for an existing kind instead remain explicit objects supplied at decoding boundaries.

## Imports and numerical boundaries

`unit_specs()` lazily resolves and caches conversion constants from installed SciPy. Public unit lookup is also lazy. Moving heavy imports into these modules' top level can break the base-import contract.

Construction converts storage and units into the target system; `_wrap` and `from_value` preserve raw objects already in a system's units. Arithmetic uses backend operators. `host_array` intentionally detaches and transfers only at serialization. A change spanning these boundaries needs checks for both values and the relevant graph, dtype or storage behavior.

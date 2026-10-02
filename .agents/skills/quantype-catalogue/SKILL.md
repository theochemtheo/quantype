---
name: quantype-catalogue
description: Extend Quantype built-in or application catalogues and maintain source generation for quantities, units, operator overloads and autodiff stubs. Use for catalogue definitions, renderer changes, generated-file drift or generated-package behavior.
---

# Quantype Catalogue

Parent: read [core physical semantics](../quantype-core/SKILL.md), including its shared guidance. Work from the repository root unless generating an application package outside this checkout.

## Select the definition boundary

- Built-in definitions live in `src/quantype/_internal/_registry.py`: `QUANTITIES`, lazily resolved `unit_specs()`, `RELATIONS` and `POWERS`.
- Application definitions use `Catalogue`, `QuantitySpec`, `UnitSpec` and `builtin_catalogue().extend(...)` from `quantype.catalogue`.
- A new unit for an existing kind may need only `QuantityClass.define_unit`; it does not require package generation.

`Catalogue` snapshots mappings and validates on construction. Preserve dimensional consistency, canonical scale 1/offset 0, valid unique identifiers and namespace names, reserved generated bindings, and offset units only for absolute temperatures. `extend` rejects replacement of existing definitions. Declared relations are kept in `Catalogue.relations`; `Catalogue.algebra` (via `close_relations` in `_internal/_registry.py`) adds multiplication's symmetry and the divisions undoing each product. Declared divisions win; two products undoing into different kinds are an error until that division is declared. Renderers and `RuntimeCatalogue` use `algebra`. Dimension tuples have eight axes, the last being charge.

## Generation ownership

`src/quantype/codegen/_render.py` produces quantity, kind, unit and autodiff contracts. `_package.py` adds application-package scaffolding. `render(...)` returns source strings without running tools. `generate(...)` formats with Ruff, minifies the private stub modules (`_products/`, `_constants/`, `_generated.pyi`; `codegen/_compact.py`), and writes changed files; `check=True` returns stale names without writing. Check mode still requires Ruff on PATH.

For built-ins, change definitions or renderers, run `uv run scripts/generate.py`, then `uv run scripts/generate.py --check`. Review the generated diff. See the [ownership map](../../../docs/architecture.md) for generated files and handwritten exports; adding a built-in class may require updating `src/quantype/__init__.py` as well.

Application generation ([ADR-010](../../../docs/decisions/ADR-010-application-catalogues-are-separate-packages.md)) currently requires a combined catalogue containing the core kinds required by its helpers. Use `builtin_catalogue().extend(...)` and a distinct package name. A generated `labquantities.Length` is a different nominal type from `quantype.Length`; use imports consistently from one API. Generation does not update the installed built-in overloads. Refer to `docs/custom-catalogues.md` for application-facing examples.

## Product classes

`codegen/_table.py` builds the naming table from `Catalogue.algebra`: which products of up to three named factors are named, and a product class for every unnamed product of two kinds and the literal powers 2, 3, -1 and -2 ([ADR-005](../../../docs/decisions/ADR-005-products-named-by-factors.md)). `codegen/_products.py` renders each product class's stub module under `_products/`, plus `products.py` (the table as text rows, parsed lazily) and `products.pyi`; `_internal/_products.py` creates the runtime classes on demand. Named classes find product results through `_rmul_*`/`_rtruediv_*` protocol members on the right operand, never through per-partner overloads ([ADR-006](../../../docs/decisions/ADR-006-protocol-dispatch-for-product-stubs.md)). Each product module also holds the product's constant class (`_LengthTimeConstant`), and products of constants dispatch through `_cmul_*`/`_ctruediv_*` members the same way ([ADR-011](../../../docs/decisions/ADR-011-constant-product-classes.md)); `_internal/_products.py` creates those classes at runtime on first use. A new kind or relation can add or rename many product classes: review the generated diff, `test_naming_table.py`, and the stubtest allowlist's `_products`/`_constants` entries. Measure checker time with the stubs installed in site-packages, not as local code.

## Validate actual consumers

`tests/runtime/test_codegen.py` checks invalid definitions, deterministic rendering, stale checks without writes, runtime behavior, optional adapters and four-checker consumer typing.

Its `tests/fixtures/generated_catalogue/` project is copied into a temporary directory and generated there. Keep generated packages out of the source fixture and parent pytest process: identities and JAX registrations are process-sensitive. Fixture runtime filenames intentionally avoid `test_*.py`; the harness invokes them explicitly in subprocesses.

For overload changes, read [typing](../quantype-typing/SKILL.md) and validate both built-in and generated consumers. For new generated autodiff behavior, read [autodiff](../quantype-autodiff/SKILL.md).

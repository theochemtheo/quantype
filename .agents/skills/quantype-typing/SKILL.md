---
name: quantype-typing
description: Maintain Quantype public stubs, operator overloads and static physical algebra across mypy, Pyright, Pyrefly and ty. Use for inferred-type errors, runtime/stub mismatches or positive and negative typing conformance.
---

# Quantype Typing

Parent: read [core physical semantics](../quantype-core/SKILL.md), including its shared guidance.

## Find the owner of the signature

The API uses Python 3.12 generic syntax and `py.typed`, without a checker plugin. Quantity stubs are generic over storage `V` and unit system `S`; `S` is a module-level `typing_extensions.TypeVar` with `default=Atomistic` (PEP 695 cannot express defaults before 3.13) and is invariant, so operands from different systems match no overload. Method-level parameters keep PEP 695 syntax. `core.py` is a dynamic runtime dispatch boundary; adjacent `core.pyi` is its precise public contract. `_internal/_unit.pyi` and `_internal/_math.pyi` are also handwritten. Quantity, namespace and autodiff stubs are generated; see [catalogue](../quantype-catalogue/SKILL.md) before changing those signatures.

Trace failures from the consumer expression to the selected overload and its generating relation or template. Preserve:

- Named result kinds for declared operations and structural `Quantity[Mul/Div/Pow[...], V, S]` results for unlisted ones, with `S` passed through unchanged.
- Static rejection of mixed unit systems, including in functions generic over `[S: UnitSystem]`.
- Nominal separation of kinds with equal dimensions.
- Storage parameters: typed construction performs conversion; annotations are more than labels. Scalar/array operator overloads must preserve the existing storage contract.
- Static rejection of incompatible addition/conversion, invalid helper inputs and missing constructor units.

Generic left-operand fallback can hide a custom right operand's `__rmul__`. `tests/typing/probes/extension_operators.py` documents this limitation and is not a passing conformance fixture. For a supported extension, use a combined generated catalogue instead of assuming subclass overloads alter the built-in API.

## Verify the public contract

Use `assert_type` in `tests/typing/positive/` to specify the actual expected result (these files are also executed by `tests/runtime/test_typing_probes.py`), and marked expressions in `tests/typing/negative/invalid.py` to specify rejection. Every negative line must end with `# error`; `scripts/check_typing.py` checks that each checker reports every marked line. It is not enough for a fixture to fail elsewhere.

`uv run just typecheck` runs ty, both mypy parsers, runtime/stub agreement via stubtest, Pyright, Pyrefly, and negative conformance. See [checks](../quantype/references/checks.md) for environment details and the dual-parser helper. A one-checker pass does not establish the supported static contract.

`tests/typing/stubtest_allowlist.txt` contains intentional runtime/stub differences. Diagnose a new mismatch before changing the allowlist; use a narrow justified entry only for an intentional difference.

Generated API changes also need `tests/runtime/test_codegen.py`, whose isolated `consumer.py` exercises all four checkers. Include corresponding runtime behavior when changing algebra so checker acceptance cannot hide a runtime disagreement.

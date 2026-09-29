---
name: design-philosophy
description: Design and review quantype's scientific Python APIs and internals. Use when adding or refactoring domain types, units, numerical backends, validation, serialization, analysis, or tests; apply the repository's semantic-boundary and simplicity principles to design decisions.
---

# Design philosophy for quantype

Use these principles when making design choices in this repository, not as a reason to rewrite unrelated code. Preserve the existing public contract unless the task calls for changing it. Read `README.md`, the relevant part of `typed-physical-units-prototype-requirements.md`, and nearby code and tests before proposing an API change. Resolve paths from the repository root; this skill lives in `.agents/skills/design-philosophy/`.

## Start with the domain boundary

- Model **physical meaning**, not just dimensions or storage. Dimensionally equal quantities can have different semantics (for example, `Pressure` and `EnergyDensity`). Name APIs for the scientific operation they provide, not for their implementation.
- Make units, conventions, reference states, coordinate systems, provenance, and definitions explicit *when relevant to the operation*. Do not silently guess when conversion is ambiguous; fail clearly and locally.
- Parse external or weakly typed input at the boundary into domain types that establish invariants once. Prefer small composable types to dictionaries, positional tuples, and stringly typed APIs. Make invalid states and invalid operations unrepresentable where practical, using static types for semantic guarantees and runtime checks for untyped inputs.
- Preserve source data and provenance when ingesting data if they matter for audit or reinterpretation; define a canonical downstream representation rather than making every consumer understand every source format.

## Keep the model deep and the interfaces small

- Follow Ousterhout's *A Philosophy of Software Design*: hide complexity behind simple interfaces, with one clear owner for each invariant and operation. Equivalent concepts should share a model even if they originate in different formats, codes, or workflows. Do not expose backend or storage details as domain concepts.
- Separate ingestion, standardisation, derivation, analysis, and presentation conceptually. Compute a result once and let rendering or serialization consume it; keep scientific logic out of renderers and storage backends. Make intermediate results typed, inspectable, and useful on their own.
- Keep convenience libraries at the edges rather than letting their types dictate domain architecture. For quantype, one physical algebra serves scalar, NumPy, JAX, and PyTorch storage; backend libraries still perform the numerical work. Do not add parallel domain hierarchies for each backend.
- Make exploration understandable: useful `__repr__` and diagnostics should expose meaning, units, and other pertinent context without requiring knowledge of internal class structure. Document non-obvious scientific conventions and metrics near the API that uses them.

## Apply this to the current prototype

- Distinguish a quantity's semantic kind from its dimensions and its presentation unit. Canonical numerical storage is distinct from display/conversion; conversion does not change physical kind. Absolute temperatures and temperature differences have different valid operations.
- Keep `src/quantype/_registry.py` the source of truth for named kinds, dimensions, canonical units, aliases, and algebra. When changing it, regenerate via `uv run just generate` and check generated outputs with `uv run just check-generated`; do not hand-edit generated code.
- Maintain explicit numerical and serialization boundaries (`.value`, `.magnitude(unit)`, `.to(unit)`, `from_canonical`, Pydantic parsing). Do not silently discard physical meaning via implicit array coercion or secretly move/detach backend tensors in arithmetic.
- Keep public static semantics usable under the project's strict type checkers without plugins. Use named relationships for scientifically meaningful results; leave unlisted results honestly typed rather than widening to `Any` or inventing a meaning from dimensions alone. Shape constraints are separate from physical kinds.
- Keep runtime operations in their narrow modules and backend adapters at their boundaries. An extension point is useful; a speculative registry, cache invalidator, factory, or orchestration framework is not.

## Decide and verify

1. Identify the caller's scientific intent, the input boundary, the invariant it establishes, and the smallest useful output type. Prefer an explicit operation to generic plumbing.
2. Put validation and failure handling at the narrowest relevant boundary. Avoid repeated checks, broad exception handlers, and silent coercion. Support independent capabilities when another capability is unavailable.
3. Choose the simplest design that preserves those semantic boundaries. Use inheritance, wrappers, factories, configuration, or abstraction only when they reduce the caller's mental model or solve a demonstrated need. Keep plausible future backends possible without building them now.
4. Test observable contracts and scientific conventions, including invalid operations and boundary failures, rather than internal structure. For type-level behavior, add positive `assert_type` cases and negative cases as appropriate. Run relevant runtime tests, `uv run just typecheck`, and `uv run just lint` when feasible; report what was not run.

When principles compete, prioritize physical correctness and explicit semantics, then a clear public interface, then implementation convenience. Do not treat these guidelines as permission to expand the requested scope.

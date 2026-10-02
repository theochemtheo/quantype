# Design principles

## Do

- Do follow John Ousterhout's *A Philosophy of Software Design*: optimise for
  deep modules and simple interfaces.
- Do follow "parse, don't validate": convert external or weakly typed data into
  stronger internal types at boundaries.
- Do make invalid states and invalid operations unrepresentable where practical.
- Do use the type system to express semantic guarantees rather than relying on
  comments or runtime assertions.
- Do prefer small, composable domain types over dictionaries, loosely structured
  tuples, and stringly typed APIs.
- Do distinguish concepts according to their meaning, not according to how or
  where they happen to be materialised.
- Do make domain semantics explicit: units, conventions, provenance, coordinate
  systems, reference states, and definitions should be represented in the model.
- Do separate workflows into clear conceptual layers.
- Do make intermediate typed, inspectable, and useful independently of the final
  layer.
- Do design `__repr__` and other exploratory interfaces so objects can be
  understood without knowing their internal class structure.
- Do prefer explicit APIs whose names communicate intent over generic plumbing.
- Do favour deterministic behaviour and stable ordering when nondeterminism
  provides no benefit.
- Do make conventions and important design decisions executable or testable
  wherever possible.
- Do write tests around behavioural contracts and semantic guarantees rather
  than implementation details.
- Do keep abstractions proportional to demonstrated needs; leave clear extension
  points without implementing speculative infrastructure.
- Do fail clearly when required invariants are absent rather than silently
  guessing.
- Do allow meaningful partial support when one unavailable capability should not
  block unrelated functionality.
- Do document non-obvious behaviour and conventions close to where users
  encounter them.
- Do prefer boring, established tools and patterns unless a new abstraction
  produces a clear reduction in complexity.

## Don't

- Don't scatter validation checks throughout the codebase when a boundary type
  can establish the invariant once.
- Don't expose implementation or storage details as domain concepts.
- Don't create parallel treatments of equivalent concepts merely because they
  originate from different file formats, codes, or workflows.
- Don't use inheritance, wrappers, factories, registries, or configuration
  layers unless they genuinely simplify the caller's mental model.
- Don't introduce abstraction in anticipation of hypothetical reuse; wait until
  the common structure is understood.
- Don't make renderers, serializers, or storage backends responsible for
  scientific logic.
- Don't build elaborate systems before the problem demonstrably requires them.
- Don't silently coerce ambiguous data into a canonical form when the conversion
  cannot be justified.
- Don't optimise for cleverness or minimal line count at the expense of
  readability, inspectability, or explicit semantics.
- Don't let convenience libraries dictate the architecture or leak their types
  throughout the domain model.
- Don't duplicate logic when a single well-named operation can own the behaviour
  and its invariants.
- Don't catch exceptions across multiple unrelated operations when the failure
  boundary can be made narrower and clearer.
- Don't treat documentation as a substitute for an API that makes the correct
  operation obvious.
- Don't overengineer: prefer the simplest design that preserves the important
  semantic boundaries and leaves room for evidence-driven evolution.
- Don't use narrative comments to explain intent - aim for 'self documenting'
  code.

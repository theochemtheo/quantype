# Generated stubs define the static API for four type checkers

## Context and Problem Statement

The kind algebra ([ADR-001](ADR-001-nominal-kinds.md)) must be visible to type
checkers: `Energy / Length` should be a `Force` in the editor, before the code
runs. The runtime is dynamic, with one `Quantity` implementation that dispatches
on kinds. Only mypy has a plugin interface; Pyright, Pyrefly, and ty have none.
How do checkers learn the algebra, and which checkers count?

## Considered Options

* A mypy plugin
* Handwritten stubs
* Stubs generated from the catalogue, checked by mypy, Pyright, Pyrefly, and
  ty, with the runtime and the stubs required to agree exactly

## Decision Outcome

Chosen option: "stubs generated from the catalogue", because it is the only
option that works in all four checkers and grows with the catalogue, and
application catalogues get the same mechanism.

Agreement is part of the decision: where the runtime names a result, the stubs
name the same one; where the runtime raises, the checkers reject. The
conformance suite enforces it with `assert_type` examples that also run, a
negative suite that every checker must flag line by line, and stubtest.

### Consequences

* Good, because users need no plugin, and each checker sees the same API.
* Good, because one source, the catalogue and its renderers, produces the
  runtime classes and the stubs; `just check-generated` catches drift.
* Bad, because the stubs are large, and loading them is a fixed cost in every
  checker ([ADR-006](ADR-006-protocol-dispatch-for-product-stubs.md)).
* Bad, because the API can only use typing features all four checkers handle
  the same way, and some signatures need workarounds for one checker.
* Bad, because the generated files are committed, so catalogue changes produce
  large diffs.

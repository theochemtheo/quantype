---
name: quantype
description: Navigate and develop this Quantype repository when a task spans modules, needs architectural context, or changes development checks. Route focused implementation work to the relevant Quantype specialization.
---

# Quantype Repository

Quantype combines nominal physical kinds, unit systems in the static type, and a generated static API. These skills describe this checkout; resolve source paths from the repository root and confirm current code before relying on a recorded detail.

## Hierarchy

The hierarchy is expressed through links, with independently discoverable skill folders:

```text
quantype
├── quantype-core
│   ├── quantype-catalogue
│   └── quantype-typing
├── quantype-backends
│   └── quantype-autodiff
└── quantype-serialization
```

Read the relevant specialization and its parent guidance. Load another branch only when the requested change crosses that boundary.

| Task | Skill |
| --- | --- |
| Arithmetic, unit conversions, affine temperatures, runtime meaning | [quantype-core](../quantype-core/SKILL.md) |
| Physical kinds, relationships, units, generated packages or renderers | [quantype-catalogue](../quantype-catalogue/SKILL.md) |
| Overloads, stubs, inferred results or checker conformance | [quantype-typing](../quantype-typing/SKILL.md) |
| Construction, storage, dtype, reductions or optional imports | [quantype-backends](../quantype-backends/SKILL.md) |
| JAX transformations, Torch gradients or physical derivative types | [quantype-autodiff](../quantype-autodiff/SKILL.md) |
| JSON, Pydantic, NPZ/NPY or custom-unit decoding | [quantype-serialization](../quantype-serialization/SKILL.md) |

## Shared contract

- Physical meaning comes from nominal kinds and declared relationships. Equal dimension vectors do not make kinds interchangeable.
- `.value` holds raw numbers in the quantity's unit system (`Atomistic` by default). Units convert input and presentation; `.to(unit)` changes presentation without changing storage; `.to_system(T)` is the only bridge between systems.
- Runtime and static contracts must agree. Generated files come from the catalogue and renderers; change those sources and regenerate rather than editing generated output alone.
- The base import stays light. NumPy, Pydantic, JAX and Torch are deferred until needed; each optional adapter works without the other backend installed.
- Existing dynamic internals support backend dispatch. Preserve precise public result types rather than widening them to `Any` to make a checker pass.

For ownership across modules, read [architecture](references/architecture.md). For setup, verification commands, and CI scope, read [checks](references/checks.md) when planning validation. These skills support local work; publishing a package or changing remote systems requires authorization for that action.

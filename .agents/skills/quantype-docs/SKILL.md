---
name: quantype-docs
description: Author, reorganize, test, and locally review Quantype documentation and maintain its Zensical site, Sybil examples, and agent exports. Use for documentation modes, executable fences, generated documentation ownership, or publication procedures.
---

# Quantype Documentation

Parent: read [repository guidance](../quantype/SKILL.md). Work from the repository root. The [documentation-site proposal](../../../docs/proposals/documentation-site.md) defines the migration and review gates; it does not authorize a PR or publication.

Read only the reference needed for the task:

- [Authoring](references/authoring.md): Diátaxis, authoritative pages, generated ownership, and agent-readable content.
- [Examples](references/examples.md): visible prerequisites, namespace boundaries, backend declarations, skips, and static claims.
- [Publishing and review](references/publishing.md): commands, development identity, local review gates, exports, and authorization boundaries.

## Route domain questions

Keep this skill procedural; do not maintain a second API manual. Consult the domain skill before changing a documented contract:

| Contract | Skill |
| --- | --- |
| Arithmetic, conversions, physical meaning, temperatures | [core](../quantype-core/SKILL.md) |
| Inferred types, overloads, static/runtime distinctions | [typing](../quantype-typing/SKILL.md) |
| Catalogue facts, renderers, generated packages | [catalogue](../quantype-catalogue/SKILL.md) |
| Construction, storage, dtype, backend availability | [backends](../quantype-backends/SKILL.md) |
| JAX transformations, Torch gradients, derivatives | [autodiff](../quantype-autodiff/SKILL.md) |
| JSON, Pydantic, NPZ/NPY, custom-unit decoding | [serialization](../quantype-serialization/SKILL.md) |

Preserve existing subject coverage and public API. Before broad content migration, complete the structural review gate. Do not move existing guides merely to fill a planned tree. Report commands actually run and all skips; a partial environment is not full example proof.

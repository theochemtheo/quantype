# Agent guide

quantype puts a physical quantity's kind and unit system in its static type.
Generated stubs carry the kind algebra to mypy, Pyright, Pyrefly, and ty, with
no plugin, and the runtime enforces the same rules.

## Where to read

Each kind of information has one home. Link to it rather than restating it.

| Question | Read |
| --- | --- |
| Why is it built this way? | [Decisions](docs/decisions/) and [design principles](docs/DESIGN_PRINCIPLES.md) |
| Where does a concern live, and what is generated? | [Architecture](docs/architecture.md) |
| How do I change an area safely? | The [repository skill](.agents/skills/quantype/SKILL.md), which routes to one skill per area |
| Which checks should I run? | [Checks](.agents/skills/quantype/references/checks.md); setup and commands are in [development](docs/development.md) |
| What do users see? | The [README](README.md) and the [guides](docs/README.md) |

## Rules

- Generated files come from the catalogue and the renderers. Change those, run
  `uv run just generate`, and commit the output; `uv run just check-generated`
  fails on drift.
- Runtime behavior and static types agree. A change to either needs a runtime
  test and a typing test
  ([ADR-002](docs/decisions/ADR-002-generated-stubs-for-four-checkers.md)).
- Prose uses American spelling, except "catalogue", which follows the API
  ([ADR-012](docs/decisions/ADR-012-american-spelling-except-catalogue.md)).
- A change that contradicts an accepted decision starts as a proposal in
  `docs/proposals/`; see [decisions](docs/decisions/README.md).
- `notes/`, where present, is local design history that git ignores and can't
  restore. Leave it alone, and scope formatters and fixers to
  `src tests scripts`.

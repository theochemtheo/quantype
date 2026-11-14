# quantype

Physical quantities whose **[kind](explanation/glossary.md#kind "A declared physical meaning, distinct from dimensions.") and [unit system](explanation/glossary.md#unit-system "A convention fixing the units of raw stored numbers.") are part of the type**.
Use ordinary floats or NumPy, JAX, and Torch arrays; check physical algebra with
mypy, Pyright, Pyrefly, or ty, without a plugin.

!!! info "Development documentation"

    This site follows `main`, not necessarily the latest PyPI release. The
    [build identity](#build-identity) records its package version and source revision.

## First calculation

Start with [Getting started](tutorials/getting-started.md). Calculate a force,
convert it, then find distances from a NumPy array.

## Find your next page

| Intention | Read |
| --- | --- |
| Learn by doing | [Getting started](tutorials/getting-started.md) |
| Complete a specific task | [How-to guides](how-to/index.md) |
| Look up exact construction and conversion rules | [Reference](reference/index.md) |
| Find a kind, unit, or system's storage units | [Built-in catalogue](reference/catalogue.md) |
| Understand kinds, dimensions, and storage | [The physical model](explanation/physical-model.md) |
| Diagnose an error | [Troubleshooting](troubleshooting.md) |
| Contribute or understand implementation choices | [Development](contributing/development.md) and [accepted decisions](https://github.com/theochemtheo/quantype/tree/main/docs/decisions) |

## Markdown and coding agents

The [agent index](llms.txt) describes when to consult each exported Markdown page.
Use **Copy as Markdown** beside a page title to copy that page from the same source.
Restrictions in these pages apply to both runtime code and static types unless a
page explicitly distinguishes them.

The [source repository](https://github.com/theochemtheo/quantype) contains the same
Markdown, contributor tooling, and historical design evidence. Start with
[Development](contributing/development.md) to change the library or these pages.

## Build identity

{{ build_identity }}.
Local previews include uncommitted working-tree changes.

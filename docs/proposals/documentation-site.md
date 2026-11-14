# Tested, agent-friendly documentation with Zensical

## Context and Problem Statement

Quantype's documentation currently lives in repository Markdown. We want a
GitHub Pages site that serves human readers and coding agents, with executable
examples checked by Sybil and doctest. Documentation must be reviewable locally
before opening a PR, including both an early structural review and a final
production-output review.

This proposal is an implementation plan, not authorization to publish the site
or open a PR. It does not change Quantype's runtime or static API.

## Considered Options

- Keep repository-only Markdown and the custom example runner.
- Adopt Zensical while retaining the custom runner.
- Adopt Zensical for rendering, Sybil for example collection and evaluation,
  and GitHub Actions for Pages deployment.

## Decision Outcome

Proposed option: Zensical, Sybil, and GitHub Pages. Markdown remains the shared
source; rendering, executable verification, and deployment stay independent.
Use Diátaxis to improve individual pages rather than impose an empty directory
structure. Add one documentation skill that routes to existing technical skills.

### Scope and defaults

- Publish one unversioned site at
  `https://theochemtheo.github.io/quantype/`, documenting `main`. Label it as
  development documentation and expose the package version and source revision
  of the build. Do not imply that it describes the latest PyPI release.
- Retain README usefulness on GitHub and PyPI. Link to the published site once
  it is available; do not duplicate the full tutorial there.
- Preserve all current user-facing subject coverage. Reorganize content without
  changing the documented API or treating the migration as an API redesign.
- Publish contributor guidance and accepted decisions in a separate navigation
  section. Exclude proposals, design-history notes, superseded decisions, and
  templates from the built site, search, and primary agent index. Keep those
  files in the repository and link to GitHub when historical evidence is useful.
- Start with built-in styling, explicit navigation, code copying, and search.
  No custom theme, remote preview service, documentation MCP server, release
  version selector, or exhaustive generated-stub listing.
- Do not run examples during rendering. Sybil checks them independently.

### Consequences

- The same source supports repository reading, a searchable website, executable
  examples, and agent-readable Markdown.
- Documentation authoring gains a small toolchain and explicit example-state
  policies. These must be documented and exercised in CI.
- Moving pages requires migrating generators, tests, attributes, and links
  together. Historical external GitHub links cannot all be preserved; avoid
  unnecessary renames and stabilize public site URLs from the first deployment.

## Information architecture and authoring

Use four reader intentions:

| Mode | Purpose | Initial Quantype content |
| --- | --- | --- |
| Tutorial | Learn through a guided, successful experience | A short physical calculation, conversion, then array operations |
| How-to | Complete a specific task with known prerequisites | Pydantic validation, serialization, NumPy integration, JAX/Torch differentiation, custom catalogues |
| Reference | Consult exact contracts | Quantities, units, systems, supported numerical operations, serialization, autodiff, generated catalogue |
| Explanation | Understand the model and tradeoffs | Nominal kinds, dimensions, presentation versus storage, unit systems, static/runtime boundaries |

The initial target is:

```text
docs/
  index.md
  tutorials/
    getting-started.md
  how-to/
    index.md
    ... task-specific pages derived from existing guides
  reference/
    index.md
    catalogue.md
    ... public API contracts grouped by concern
  explanation/
    index.md
    ... concepts extracted from existing guides
  troubleshooting.md
  contributing/
    development.md
    architecture.md
    design-principles.md
  decisions/
  design/
  proposals/
```

Only create populated pages. Choose task-specific filenames when splitting the
actual content; do not create placeholder pages to fill the tree.

- The homepage explains Quantype's purpose, offers a first-use path, and routes
  readers by task. It also links to the agent index and Markdown access.
- Narrow the existing getting-started page into a coherent tutorial. Move
  substantial conceptual detours to explanation and alternatives to how-to or
  reference pages. Keep short explanations necessary to complete each step.
- Split broad guides when distinct tasks need different prerequisites. Preserve
  coverage of temperatures, constants, custom units, backend/storage choices,
  and persistence formats, not just the headings in the new navigation.
- Reference entries identify public imports, signatures, accepted inputs,
  result kind/system/storage, backend support, errors, and relevant conversion
  or copying behavior. Clearly distinguish runtime support from static support.
- Keep one authoritative explanation of each contract. Other modes link to it
  while retaining the local prerequisites and cautions needed to use an example.
- Add troubleshooting entries for important existing failure modes, including
  mixed systems, `.value` without a system, and unsupported numerical calls.
- Keep user-facing explanation separate from ADR history. Do not rewrite accepted
  decisions merely to make them resemble current user documentation.

## Implementation phases

### 1. Record authoring and maintenance procedures

Create `quantype-docs` under the existing `.agents/skills/` convention, with a
short `SKILL.md` and focused references for authoring, examples, and site
publishing. Add routing from the repository skill.

The skill must cover the four documentation modes, example-state policies,
agent exports, generated-file ownership, version labeling, review gates, and
narrow verification commands. Route domain questions to existing core, typing,
catalogue, backend, autodiff, and serialization skills rather than duplicating
those skills' technical content. Link to authoritative documentation instead
of maintaining a separate agent manual.

### 2. Prove the toolchain and build a representative slice

- Resolve and lock compatible Zensical and Sybil versions. Verify the actual
  versions support Markdown Python fences, doctest transcripts, `llmstxt`,
  Markdown exports, and strict builds before relying on those capabilities.
- Put Sybil in the `test` dependency group so incomplete-backend CI environments
  can collect examples without installing the site builder. Put Zensical in a
  `docs` group included by `dev`; update `uv.lock`.
- Add `zensical.toml` with explicit navigation, the correct project `site_url`,
  exclusions, version/revision metadata, search, and agent exports.
- Ignore generated site/cache artifacts. Keep generated HTML out of commits.
- Build a homepage and one real page in each documentation mode. Use the
  generated catalogue as part of the reference slice. Retain remaining existing
  material until its migration is complete; the review slice is not a substitute
  for the complete migration.
- Verify the reference approach against representative public APIs with dynamic
  runtime definitions and generated stubs. Use the catalogue renderer for
  catalogue facts and concise handwritten contracts for other APIs. Do not
  assume runtime introspection alone exposes Quantype's static API correctly.

Add these repository commands; names below are proposed, not existing:

| Command | Contract |
| --- | --- |
| `uv run just docs-serve` | Serve working-tree docs on localhost with automatic reload and open a browser |
| `uv run just docs-test` | Run the collected README and published-document Python examples |
| `uv run just docs-build` | Build the production site in strict mode |
| `uv run just docs-check` | Check examples, generated reference, site build, internal links, and agent exports |
| `uv run just docs-preview` | Build and serve production output under `/quantype/` for final local review |

Site commands install/use the docs group. Example-only checks do not require
Zensical or force optional backends into an incomplete environment. Document a
full-backend invocation for `docs-check`; a partial local run must report backend
skips and cannot serve as the final all-example proof.

### 3. Structural review gate: before broad migration

Run the representative examples, build the site, inspect it in a browser, and
leave the local development server running. Provide its URL and a short route
through the homepage and four representative pages.

The user reviews navigation, terminology, page granularity, tutorial pacing,
reference usefulness, and the visual reading experience. Incorporate feedback
before migrating the remaining pages. No PR is opened at this gate.

### 4. Replace the custom example runner with Sybil

Use the Markdown `PythonCodeBlockParser` for both plain Python fences and
REPL-style doctest transcripts. Avoid overlapping parsers and double collection.

Make state policy explicit:

- Tutorial pages run in documented order with shared state within the page.
  Imports and prerequisite steps must be visible, not hidden in fixtures.
- How-to, reference, troubleshooting, and README fences are independently
  executable. Reset their namespace between fences, not between prompts in one
  transcript. Add only the evaluator adaptation required for this isolation.
- Use a temporary working directory and pytest cleanup for filesystem examples.
  Keep examples deterministic; avoid network, credentials, and GPU requirements.
- Use assertions and numerical testing helpers for arrays and approximate
  values. Use doctest output for results whose representation teaches something.
  Do not hide meaningful discrepancies with blanket ellipses or loose matching.
- Verify expected exceptions explicitly. Use documented Sybil skips with reasons
  for genuinely illustrative patterns. Migrate every existing `python notest`
  block; do not retain a second legacy skip convention.
- Replace source-text backend detection with explicit backend requirements and
  narrowly scoped skips. The full-backend run must execute those examples;
  incomplete environments skip only the unavailable-backend examples.
- Do not execute installation shell fences or historical design examples as
  ordinary Python tests.

Register collection at a scope that includes README and nested published pages.
Keep proposals, decision history, skills, and source trees out of that selection.
Update `just test`, `scripts/coverage.sh`, and relevant CI pytest invocations:
`pytest tests/runtime` alone will no longer include the Markdown examples.
Preserve optional-dependency tier checks and include examples in the full
coverage run. Ensure parallel execution never splits a sequential document
between workers; use file-based scheduling where xdist is enabled.

Remove the superseded regex/exec runner from `tests/runtime/test_docs.py`, but
retain its independent README-link contract. Add focused regression tests for
namespace isolation, tutorial sequencing, expected-output failures, and explicit
backend skips. Verify fence syntax used in the site is actually collected,
including any nested or annotated fences; avoid unsupported syntax rather than
silently dropping examples.

For important advertised inferred types and rejected expressions, reuse or add
small positive/negative consumer fixtures in `tests/typing`. Run the existing
four-checker workflow for those claims. Sybil is not a static type checker.

### 5. Complete content migration and agent access

Migrate every existing user guide into the agreed structure. Update all
repository callers and links in the same cutover, including README, package
metadata, skills, contributor documentation, generator targets, generated-output
tests, and `.gitattributes`. In particular, moving `docs/catalogue.md` requires
updating `scripts/generate_docs.py` and its generation contract; the units range
table also has an existing freshness check. Regenerate rather than hand-editing
generated sections. Remove obsolete index pages after migrating their callers.

Configure Zensical's supported `llmstxt` integration with curated sections for
getting started, tasks, exact reference, explanation, and contributor guidance.
Generate Markdown equivalents from the same pages, not a second authored set.
Enable Markdown copying where supported. Do not generate a full-site text dump
unless a concrete consumer needs it.

Agent-facing requirements:

- Descriptive titles/headings, public symbol names, stable anchors, explicit
  dependencies, and version identity.
- Critical restrictions remain visible in text, not only in diagrams, tabs,
  screenshots, or collapsed panels.
- The homepage links to the agent index. Index entries explain when to consult
  each page and link to its generated Markdown equivalent.
- Exported Markdown preserves signatures, examples, tables, and warnings.
- Generated links honor `/quantype/`; historical pages are excluded from the
  primary agent index, and development docs cannot be mistaken for release docs.

Check internal page, asset, and fragment targets against the production output.
Use Zensical validation where sufficient and add a narrow check only for gaps.
Do not make ordinary CI depend on live external websites.

### 6. Prepare CI and deployment

- On PRs, run the example checks and strict production build, including internal
  link and agent-export checks. Ensure a code change cannot bypass docs checks
  merely because no Markdown changed.
- Keep full-backend and incomplete-backend example coverage integrated with the
  existing test environments rather than create an unrelated dependency matrix.
- Add required documentation checks to the existing `test-gate` dependencies.
- Prepare a Pages workflow that builds/tests on `main` and deploys only the
  successfully checked artifact for that revision. PRs never deploy production.
- Reuse the pinned uv version, locked installs, SHA-pinned actions, and workflow
  security checks. Keep `contents: read` for build jobs; grant `pages: write`
  and `id-token: write` only to deployment. Use the `github-pages` environment
  and deployment concurrency. Do not commit `site/` or create a site branch.
- Keep repository Pages configuration and first deployment disabled until the
  user authorizes publication. Updating remote settings is not part of writing
  or locally reviewing the implementation.

### 7. Complete-site review gate: before opening a PR

Run the full-backend example checks, relevant typing fixtures, generated-output
checks, strict build, internal-link checks, and agent-export checks. Run
formatting/workflow checks scoped to changed files. Record commands and results,
including any unavailable verification; do not claim unrun checks passed.

Start the development preview for editorial iteration. Also serve the actual
production build over HTTP under `/quantype/`, with the same path semantics as
GitHub Pages. Verify the homepage, deep links, navigation, search, assets,
light/dark presentation, a narrow viewport, code copying, and Markdown access.
Confirm production exclusions rather than relying on development-server output,
which can include draft pages.

Provide the user with both local URLs, the build's version/revision identity,
and a concise review route. Incorporate feedback and rerun affected checks.
Leave the review server available while feedback is being collected. Do not
open a PR until the user approves the completed site. Publication remains a
separate authorized action.

Update contributor instructions and the changelog for the completed migration.
Remove throwaway probes and obsolete tooling; keep only useful verification.

## Completion criteria

- All current user-facing subject coverage has a clear destination, with no
  duplicated authoritative API contracts or abandoned repository links.
- Both local review gates have been completed and feedback incorporated.
- The production site works under the GitHub Pages project path and clearly
  identifies itself as development documentation.
- Runnable published examples and README examples are collected exactly once;
  tutorials preserve intended state and independent examples cannot borrow it.
- Full-backend verification executes backend examples; incomplete environments
  retain meaningful skips and dependency-isolation checks.
- Static claims have relevant typing coverage, and generated reference stays
  covered by the existing generation/freshness workflow.
- The agent index resolves to usable Markdown with essential semantics intact.
- The documentation skill and contributor commands describe the actual workflow.
- Required CI checks cover documentation, and deployment is revision-matched,
  least-privilege, and separate from PR validation.
- No PR or remote publication occurs without the corresponding user approval.

## Sources

- [Diátaxis](https://diataxis.fr/) and its
  [incremental adoption guidance](https://diataxis.fr/how-to-use-diataxis/).
- [Sybil Markdown parsers](https://sybil.readthedocs.io/en/latest/markdown.html)
  and [namespace/API semantics](https://sybil.readthedocs.io/en/latest/api.html).
- Zensical [preview](https://zensical.org/docs/usage/preview/),
  [build](https://zensical.org/docs/usage/build/),
  [supported plugins](https://zensical.org/docs/compatibility/mkdocs/plugins/),
  and [Pages deployment](https://zensical.org/docs/publish-your-site/).
- Repository [architecture](../contributing/architecture.md),
  [development workflow](../contributing/development.md), and
  [decision process](../decisions/README.md).

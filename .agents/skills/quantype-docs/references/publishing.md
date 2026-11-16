# Site commands, review, and publication

The [proposal](../../../../docs/proposals/documentation-site.md) records the implementation phases. Local tooling and a prepared deployment workflow are not permission to open a PR, enable Pages, change remote settings, or publish.

## Narrow commands

Run from the repository root; consult `justfile` for the implementation. Site commands install/use the `docs` dependency group (Zensical); example-only checks use the `test` group (Sybil), without requiring Zensical or forcing optional backends into an incomplete environment. The `dev` group includes `docs`.

| Command | Contract |
| --- | --- |
| `uv run just docs-serve` | Serve working-tree docs on localhost with automatic reload and open a browser |
| `uv run just docs-test` | Check collected README and published-document Python examples, without building the site |
| `uv run just docs-build` | Build the production site in strict mode; do not execute examples during rendering |
| `uv run just docs-check` | Check examples, generated reference freshness, strict site build, internal links, and agent exports |
| `uv run just docs-preview` | Build and serve actual production output under `/quantype/` |

Full-backend documentation proof is `uv run --all-extras just docs-check`. A partial local run must visibly report unavailable-backend skips and cannot substitute for final all-example proof. Record intentional illustrative skips separately from missing-backend coverage. Report only checks actually run, their results, and unavailable verification. Static claims additionally require relevant typing fixtures; Sybil cannot establish them.

## Build identity and output

The site is one unversioned development site documenting `main`, intended for `https://theochemtheo.github.io/quantype/`. Clearly label it development documentation and expose both the package version and source revision of the build, including in agent-facing access. Do not imply it describes the latest PyPI release or introduce a release selector.

Check internal pages, assets, and fragment targets against production output. Use supported Zensical validation first and narrow supplemental checks only for gaps; ordinary CI must not depend on live external websites. Validate curated `llms.txt`, generated Markdown equivalents, preserved contracts/warnings, and project-path links. Confirm history/proposals/templates are absent from production output, search, and the primary agent index; a development server can expose draft pages.

## Gate 1: structural review before broad migration

Build a real representative slice: homepage, one populated page in each Diátaxis mode, and generated catalogue reference. Retain unmigrated guides; the slice does not replace complete coverage.

Run representative examples, build, inspect in a browser, and leave the local development server available. Give the user its URL and a short route through the homepage and four representative pages. Ask for review of navigation, terminology, page granularity, tutorial pacing, reference usefulness, and visual reading experience. Incorporate feedback before migrating the remaining guides. Do not open a PR at this gate.

## Gate 2: complete-site review before a PR

After the approved migration, run full-backend examples, relevant typing fixtures, generated-output checks, strict build, internal-link and agent-export checks, plus formatting/workflow checks scoped to changed files. Record commands, actual outcomes, skips, and unavailable checks. Update contributor instructions and the changelog for the completed migration.

Start both an editorial development preview and an HTTP preview of actual production output under `/quantype/`. Inspect homepage, deep links, navigation, search, assets, light/dark presentation, narrow viewport, code copying, Markdown access, production exclusions, and version/revision identity. Provide both local URLs, the build identity, and a concise review route. Leave review servers available while feedback is collected. Incorporate feedback and rerun affected checks. Do not open a PR until the user approves the completed site.

## Publication is a separate authorization

Keep repository Pages configuration and first deployment disabled until the user authorizes publication. No production deployment from PRs. A prepared workflow builds/tests on `main` and deploys only the checked artifact for that revision. Reuse pinned uv, locked installs, SHA-pinned actions, and workflow security checks; build jobs retain `contents: read`, and only deployment receives `pages: write` and `id-token: write`, with the `github-pages` environment and deployment concurrency. Include docs checks in the existing test gate even for code-only changes. Do not commit `site/`, create a site branch, or change remote settings as part of local review.

# Authoring and ownership

## Choose the reader's intention

| Mode | Write for | Keep out |
| --- | --- | --- |
| Tutorial | A guided, successful first calculation, conversion, then array operations | Long conceptual detours and competing approaches |
| How-to | A specific task with explicit prerequisites and a usable outcome | A tour of the entire API |
| Reference | Exact public contracts and lookup | Tutorial pacing or undocumented introspection assumptions |
| Explanation | The model, boundaries, and tradeoffs | Step-by-step task instructions |

Use Diátaxis to improve populated pages, not to create empty directories or placeholders. Keep short explanations and cautions necessary for each task locally; link to the single authoritative contract for detail. Split broad guides only where distinct tasks warrant different prerequisites. Preserve coverage of temperatures, constants, custom units, backend/storage choices, persistence, and static typing.

Reference entries name public imports, signatures, accepted inputs, result kind/system/storage, backend support, errors, and conversion/copy behavior. Distinguish runtime support from static support. Runtime introspection alone does not describe generated stubs accurately.

## Current authoritative sources

- [Getting started](../../../../docs/tutorials/getting-started.md): sequential first-use examples.
- [Units](../../../../docs/reference/units.md) and [unit systems](../../../../docs/reference/systems.md): exact unit, temperature, storage, and system contracts.
- [Catalogue](../../../../docs/reference/catalogue.md): generated catalogue reference.
- [How-to guides](../../../../docs/how-to/index.md): application tasks.
- [Autodiff](../../../../docs/reference/autodiff.md) and [serialization](../../../../docs/reference/serialization.md): specialized user contracts.
- [Architecture](../../../../docs/contributing/architecture.md), [development](../../../../docs/contributing/development.md), and [design principles](../../../../docs/contributing/design-principles.md): contributor guidance.
- [Accepted decisions](../../../../docs/decisions/README.md): rationale, not a second user guide.

Keep README useful on GitHub and PyPI, with installation and a short first-use path. Link to the site once it is published; do not duplicate the full tutorial. Do not rewrite accepted decisions to resemble current reference material.

## Generated ownership and moves

Change generators or their inputs, then regenerate; never hand-edit generated output or generated sections alone. `scripts/generate_docs.py` owns the catalogue document; the units range table also has a freshness contract. Consult [catalogue guidance](../../quantype-catalogue/SKILL.md) for source generation and [architecture](../../../../docs/contributing/architecture.md) for ownership.

A page move is one cutover: update navigation, relative links, README, package metadata, skills, contributor guidance, generator targets, generated-output tests, and `.gitattributes` together. Preserve useful URLs and anchors where possible; remove obsolete indexes only after migrating their callers. Keep generated HTML, caches, and throwaway probes out of commits.

## Human and agent access

Use descriptive titles, public symbol names, stable anchors, explicit dependencies, and visible warnings. Critical restrictions must remain readable outside diagrams, tabs, screenshots, or collapsed panels.

Do not repeat core dependencies or minimum Python versions on every page; installation belongs in the tutorial. State genuine optional tools/backends and configuration at the task that needs them. Use selective, noncollapsible admonitions for restrictions and decision points, not every paragraph. Keep runnable fences outside callouts. Link a term's first important use to the glossary with a concise Markdown link title for a hover definition; hover content supplements visible explanations rather than replacing them. Use built-in styling and `content.tooltips`, not custom theme code.

The homepage routes readers by task and links to the agent index and Markdown access. Curate `llms.txt` sections for getting started, tasks, exact reference, explanation, and contributor guidance; each entry explains when to consult its page and links to its generated Markdown equivalent. Generate exports from the same sources, never a separately authored agent manual or an unnecessary full-site text dump. Preserve signatures, examples, tables, and warnings in Markdown exports.

Publish contributor guidance and accepted decisions separately from user navigation. Exclude proposals, design-history notes, superseded decisions, and templates from production output, search, and the primary agent index. Keep repository history available through GitHub links when useful. Production links and exports must honor `/quantype/`.

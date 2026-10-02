# Decisions

Each file records one architecture decision: the problem, the options, the
choice, and its consequences. A record's folder is its status.

| Folder | Status |
| --- | --- |
| [`../proposals/`](../proposals/) | proposed |
| `decisions/` | accepted |
| [`superseded/`](superseded/) | replaced by a later decision |

1. Copy [`_adr_template.md`](_adr_template.md) to
   `docs/proposals/<short-slug>.md` and write the proposal.
2. When it is accepted, give it the next number and move it here as
   `ADR-NNN-<short-slug>.md`.
3. When a later decision replaces it, move it to `superseded/`, keeping its
   number, and add `Superseded by [ADR-NNN](../ADR-NNN-….md).` under its title.
4. Delete a rejected proposal; git keeps its history.

To change an accepted decision, write a new proposal. Correct a fact or a link
in an accepted record with an ordinary edit.

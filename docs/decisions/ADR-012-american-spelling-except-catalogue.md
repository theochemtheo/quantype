# Prose uses American spelling, except "catalogue"

## Context and Problem Statement

quantype's prose mixed spellings: "behaviour" in the guides, "behavior" in the
agent skills, and "optimise" a few lines from "serialization". Most public
names are American: `quantype.serialization`, `Magnetization`, `u.meter`,
`u.nanometer`, as in pint. One is British: the `quantype.catalogue` module and
its `Catalogue` class. Which spelling should prose use? Prose here means the
README, guides, changelog, docstrings, comments, error messages, GitHub
templates, and agent skills.

## Considered Options

* British spelling with -ise
* British spelling with Oxford -ize
* American spelling, keeping "catalogue"
* American spelling throughout, writing "catalog" for the concept

## Decision Outcome

Chosen option: "American spelling, keeping catalogue", because prose then
matches the public names: the guides write nanometers beside `u.nanometer` and
serialization beside `quantype.serialization`. "catalogue" stays because it
names a module, a class, and the concept the guides build on, so writing
"catalog" would make every sentence about it disagree with the code.

Prose writes behavior, color, analyze, center, meter, modeling, and license.
Public names keep their spelling, and new public names are American.
Identifiers, file names, third-party names, and quoted tool output stay as
written. Units may take British spellings as aliases, as `u.metre` does.

### Consequences

* Good, because prose and public names agree, apart from one word.
* Good, because pint users find units under the names they expect.
* Bad, because "catalogue" is an exception that writers have to remember.
* Bad, because no check enforces the rule, so review has to.

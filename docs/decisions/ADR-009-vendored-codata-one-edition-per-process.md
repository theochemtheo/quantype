# CODATA values are vendored, with one edition per process

## Context and Problem Statement

Unit scales such as the electronvolt, the bohr, and the dalton come from CODATA
recommended values, which change between editions. They came from whichever
SciPy was installed (SciPy 1.13 carries CODATA 2018, 1.18 CODATA 2022), so the
same conversion gave different results in different environments, and generated
catalogues could disagree with quantype. Where should the values come from,
and can a process mix editions?

## Considered Options

* SciPy's `scipy.constants` at runtime
* One vendored edition
* Several vendored editions, with one chosen per process and fixed on first use

## Decision Outcome

Chosen option: "several vendored editions, one per process", because results
then depend only on quantype's version and the chosen edition, older results
can be reproduced, and a process never mixes editions.

quantype carries CODATA 2014, 2018, and 2022, with 2022 by default.
`QUANTYPE_CODATA` or `quantype.codata.use()` chooses another before any unit is
used; the edition is fixed on first use. `scripts/generate_codata.py` writes
`_internal/_codata.py` from the NIST tables SciPy bundles, so SciPy is a
development dependency only.

### Consequences

* Good, because conversions are reproducible across environments, and NPZ
  archives record the edition as provenance.
* Good, because the editions are a `Literal` type, so a misspelt edition is a
  type error.
* Bad, because changing the edition after the first unit is used raises
  `RuntimeError`, so a process can't compare editions directly.
* Bad, because a new CODATA edition needs a quantype release.

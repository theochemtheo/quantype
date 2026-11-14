# Getting started

Calculate a force, change its displayed units, then compute distances from an
array.

!!! note "One session, in order"

    Later steps use names from earlier blocks. Run the Python blocks in order
    in the same session.

## Install

Use Python 3.12 or newer. In an existing uv project:

```bash
uv add quantype
uv run python
```

Alternatively, run `python -m pip install quantype` in an activated virtual
environment, then start `python`.
The installed version is available as `quantype.__version__`.

## Calculate a force

```python
from quantype import Energy, Force, Length, u

length = Length[float](2, u.nm)
energy = Energy[float](3, u.eV)
force = energy / length
assert isinstance(force, Force)
assert force.magnitude(u.eV_per_angstrom) == 0.15
```

Dividing an energy by a length gives a force, not an arbitrary dimensioned
number. Its [kind](../explanation/glossary.md#kind "The declared physical meaning of a quantity.")
is `Force`; a type checker infers `Force[float]` for `force` too.

`[float]` asks the constructor to convert its input to Python floating-point
storage; it is not just a type-checking annotation.

## Change presentation, not the quantity

```python
>>> force
Force(1.5 eV/nm)
>>> force.to(u.eV_per_angstrom)
Force(0.15 eV/Å)
```

`.to(unit)` returns a new [presentation](../explanation/glossary.md#display-unit "The unit used to show a quantity, not necessarily its storage unit.")
of the same physical value. Use
[`.magnitude(unit)`](../reference/quantities.md#conversion-and-numerical-access)
when another library needs a plain number in a known unit.

```python
shown = length.to(u.angstrom)
assert shown == length
assert shown.magnitude() == 20.0
```

## Compute distances from an array

```python
import numpy as np
import numpy.typing as npt

import quantype.numpy as qnp

positions = Length[npt.NDArray[np.float64]]([[0, 0, 0], [3, 4, 0]], u.nm)
displacements = positions - positions[0]
distances = qnp.sqrt((displacements**2).sum(axis=-1))
np.testing.assert_allclose(distances.magnitude(u.nm), [0, 5])
assert distances.max() == 5 * u.nm
```

Indexing and subtraction preserve `Length`. Squaring produces `Area`, summation
preserves `Area`, and `qnp.sqrt` returns `Length`.
[`qnp`](../reference/numerical-operations.md) uses familiar NumPy names and
carries these physical result types through type checking.

!!! warning "Removing units must be explicit"

    `np.asarray(quantity)` raises `TypeError` to prevent silently discarding
    physical meaning. Use `.magnitude(unit)` to choose the numbers you pass to
    another library.

## Check your calculation

Save this standalone calculation as `calculation.py`:

```python
from typing import assert_type

from quantype import Energy, Force, Length, u

length = Length[float](2, u.nm)
energy = Energy[float](3, u.eV)
force = energy / length
assert_type(force, Force[float])
```

In your uv project, install a checker and run it from the shell, not the Python
session:

```bash
uv add --dev ty
uv run ty check calculation.py
```

The file should pass. `assert_type` asks the checker to verify the inferred
`Force[float]`; it performs no runtime check.
Now append `energy + length` to the file and rerun the checker. It rejects
adding an `Energy` to a `Length`, before you execute that expression. Remove
the invalid line before running `uv run python calculation.py`; it also raises
`TypeError` at runtime.

No plugin or Quantype-specific checker configuration is needed. If you already
use another supported checker, install that tool instead of `ty` and run one of:

| Checker | Command |
| --- | --- |
| [mypy](https://mypy.readthedocs.io/en/stable/getting_started.html) | `uv run mypy calculation.py` |
| [Pyright](https://microsoft.github.io/pyright/#/installation) | `uv run pyright calculation.py` |
| [Pyrefly](https://pyrefly.org/en/docs/installation/) | `uv run pyrefly check calculation.py` |

The development dependencies in
[`pyproject.toml`](https://github.com/theochemtheo/quantype/blob/main/pyproject.toml)
list the oldest checker versions tested by Quantype's four-checker conformance
suite. Use a current version rather than relying on an older bundled checker.

## Next steps

- [Test quantities](../how-to/test-quantities.md) across different display units.
- [Work with arrays](../how-to/arrays.md) for indexing, numerical backends,
  scaling, and unit-invariant operations.
- [Validate configuration](../how-to/validate-data.md) and
  [save quantities](../how-to/serialize.md) at application boundaries.
- [Quantities reference](../reference/quantities.md) for constructors, storage,
  and explicit unit systems.
- [The physical model](../explanation/physical-model.md) for nominal kinds and
  the difference between storage and presentation.

# Test quantities

Use `quantype.testing.assert_allclose` when comparing quantities in tests.

## Compare arrays in different units

```python
import numpy as np

from quantype import u
from quantype.testing import assert_allclose

actual = u.nm(np.array([1.0, 2.0]))
expected = u.angstrom(np.array([10.0, 20.0]))
assert_allclose(actual, expected)
```

`assert_allclose` checks the kind and shared unit system before comparing
numbers.

!!! tip "Compare quantities without stripping units"
    Display units may differ. Do not pass quantities directly to
    `numpy.testing` or `pytest.approx`: implicit array conversion is forbidden.

## Give an absolute tolerance for temperatures

```python
from quantype import u
from quantype.testing import assert_allclose

assert_allclose(300 * u.K, 300.05 * u.K, atol=0.1 * u.delta_K)
```

An absolute temperature is a point; its tolerance is a temperature **difference**.
See [the physical model](../explanation/physical-model.md#temperatures-are-points).

## Compare a scalar at a numerical boundary

```python
from math import isclose

from quantype import Length, u

result = Length[float](2, u.nm)
assert isclose(result.magnitude(u.angstrom), 20.0)
```

Extract numbers only after choosing their unit. For exact equality and
conversion behavior, consult [Quantities](../reference/quantities.md).

# Work with arrays and numerical backends

Use quantity-aware operations to keep physical meaning attached to arrays.
For supported calls and their physical results, consult
[the numerical reference](../reference/numerical-operations.md#dispatch-and-physical-results).

## Calculate distances and select values

Use an explicit array storage annotation for lists and a `quantype.numpy`
function when the operation changes physical kind:

```python
import numpy as np
import numpy.typing as npt

import quantype.numpy as qnp
from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([[0, 0, 0], [3, 4, 0]], u.nm)
displacements = positions - positions[0]
distances = qnp.sqrt((displacements**2).sum(axis=-1))
np.testing.assert_allclose(distances.magnitude(u.nm), [0, 5])
np.testing.assert_allclose(positions.mean(axis=0).magnitude(u.nm), [1.5, 2, 0])
np.testing.assert_array_equal(distances < 1 * u.nm, [True, False])
assert distances.shape == (2,)
assert distances.max() == 5 * u.nm
```

`qnp` works with NumPy, JAX, and Torch storage. Use it in type-checked code:
NumPy's native functions dispatch quantities at runtime, but NumPy's stubs
reject them. Native `jax.numpy` does not dispatch quantities. See
[JAX calculations](autodiff-jax.md) and [Torch calculations](autodiff-torch.md).

## Choose views or independent copies

Construction can retain your original array. Basic NumPy slicing shares memory;
advanced indexing copies. Make a copy when later mutation should be independent:

```python
import numpy as np
import numpy.typing as npt

from quantype import Length, u

raw = np.arange(6.0).reshape(3, 2)
positions = Length[npt.NDArray[np.float64]](raw, u.nm)
assert positions.magnitude(u.nm) is raw
assert np.shares_memory(positions[1:].magnitude(), raw)
assert not np.shares_memory(positions[np.array([0, 2])].magnitude(), raw)
assert positions[1, 0].magnitude().shape == ()
assert type(positions[1, 0].item().magnitude()) is float
independent = positions.copy()
raw[0, 0] = 10.0
assert independent[0, 0] == 0 * u.nm
assert positions[0, 0] == 10 * u.nm
```

There is no quantity item assignment or in-place arithmetic: `q += other`
creates a new quantity and rebinds `q`. A shared raw array remains mutable;
[the quantities reference](../reference/quantities.md#construction-and-storage)
describes ownership and storage conversion.

## Scale, sum, and format quantities

Real numerical arrays can scale scalar quantities. For formatting, select a
scalar; the format specification applies to its magnitude and keeps its unit:

```python
import numpy as np
from math import isclose

from quantype import Pressure, Time, u

step = Time[float](0.5, u.fs)
times = step * np.arange(4)
assert f"{times[-1]:.2f}" == "1.50 fs"
total = sum([1 * u.nm, 5 * u.angstrom], 0 * u.nm)
assert isclose(total.magnitude(u.angstrom), 15.0)

density = (1 * u.eV) / (1 * u.angstrom_cubed)
stress = Pressure.reinterpret(density)
assert isclose(stress.magnitude(u.GPa), 160.2176634)
```

`reinterpret` is an explicit change of nominal meaning, not a unit conversion;
use it only when that change is physically intended. See
[the physical model](../explanation/physical-model.md).
Do not use `if q:`; compare with a quantity or use `q is not None`.

## Apply a unit-invariant operation

For a storage operation that does not need a physical rule, apply a function to
the stored numbers:

```python
import numpy as np

from quantype import u

positions = u.nm(np.array([3.0, 1.0, 2.0]))
shifted = positions.apply(lambda raw: np.roll(raw, 1))
assert shifted.unit is u.nm
np.testing.assert_array_equal(shifted.magnitude(u.nm), [2.0, 3.0, 1.0])
```

!!! warning "Keep `apply` unit-invariant"
    The function must have the same physical meaning in any
    [storage unit](../explanation/glossary.md#storage-unit "The unit in which a quantity's raw numbers are stored.")
    and return the same backend. Sorting, slicing, reshaping, detaching, and
    moving devices qualify; adding a bare constant or applying `log` generally
    does not. Quantype cannot check this promise.

!!! tip "Name the unit at a numerical boundary"
    For unsupported physical operations, pass `q.magnitude(unit)` to the
    consumer and decide what physical meaning its result has. Never use
    `np.asarray(q)` to discard units; implicit array coercion is forbidden.

## Convert storage or move a Torch tensor

Install `quantype[torch]` for this example. Use a typed constructor at a backend
conversion boundary; `apply` is for transformations within the existing backend.

<!-- requires: torch -->
```python
import numpy as np
import torch

from quantype import Length, u

source = u.nm(np.array([1.0, 2.0]))
tensor = Length[torch.Tensor](source.magnitude(u.nm), u.nm, dtype=torch.float64)
on_cpu = tensor.apply(lambda raw: raw.to("cpu"))
assert on_cpu.magnitude(u.nm).device.type == "cpu"
assert on_cpu.dtype == torch.float64
```

For a GPU already available to your application, use
`q.apply(lambda raw: raw.to("cuda"))`; no GPU is needed above. To detach use
`q.apply(lambda raw: raw.detach())`. For host extraction from Torch, explicitly
use `q.magnitude(unit).detach().cpu().numpy()`, or
`q.apply(lambda raw: raw.detach().cpu()).to_numpy()` to keep the unit; either
loses graph connectivity.
See [serialization](serialize.md) for persistence and
[test quantities](test-quantities.md) for comparisons without dropping units.

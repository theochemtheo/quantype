# Numerical operations

Public import: `import quantype.numpy as qnp`. This is a unit-aware numerical
API, not the whole NumPy API. Functions dispatch to NumPy, JAX, or a Torch
adapter according to the
[backend](../explanation/glossary.md#backend "The numerical library that owns and operates on the stored values.");
plain numerical inputs pass through to their backend. JAX/Torch need their
respective optional extras. Backend shape, axis, and numerical-domain
restrictions still apply.

## Dispatch and physical results

| Operation | Accepted quantity inputs | Physical result |
| --- | --- | --- |
| `sin(x)`, `cos(x)`, `tan(x)` | `Angle` | `Dimensionless` |
| `arcsin(x)`, `arccos(x)`, `arctan(x)` | `Dimensionless` | `Angle` |
| `exp(x)`, `expm1(x)`, `log(x)`, `log1p(x)`, `log2(x)`, `log10(x)`, `sinh(x)`, `cosh(x)`, `tanh(x)` | `Dimensionless` | `Dimensionless` |
| `arctan2(y, x)` | Same kind and compatible systems | `Angle` |
| `sqrt(x)` | A kind/expression with a supported square-root relation | Root kind; e.g. `Area -> Length` |
| `square(x)` | Quantity | Kind as `x ** 2` |
| `abs(x)`, `absolute(x)`, `fabs(x)` | Quantity | Same kind; absolute-temperature scaling drops offset display |
| `maximum(a, b)`, `minimum(a, b)`, `hypot(a, b)` | Same kind and compatible systems; `hypot` excludes absolute temperatures | Same kind |
| `clip(x, a_min, a_max)` | Bounds of the same kind, or `None` | Same kind as `x` |
| `where(condition, x, y)` | Plain boolean condition; `x`, `y` of the same kind | Same kind as `x` |
| `dot(a, b)` | Two quantities | Product kind, as for `*` |
| `cross(a, b, axis=-1)` | Two quantities | Structural product, never implicitly a named work/torque kind |
| `matmul(a, b)`, `outer(a, b)`, `a @ b` | Two quantities, or a quantity and plain numerical operand | Product kind, or the quantity kind for a plain scaling operand |
| `linalg.norm(x, axis=None, *, keepdims=False)` | Quantity other than an absolute temperature | Same kind |
| `trapezoid(y, x=None, dx=1.0, axis=-1)` | Quantity integrand and/or coordinate/spacing | Product of integrand and coordinate kinds; plain spacing retains integrand kind |

Each unary call above takes `x` positionally. Binary elementwise/product calls
use positional operands. No operation infers nominal meaning solely from a
dimension vector: products and roots use the declared
[physical algebra](../explanation/physical-model.md). Systems and storage are
preserved, with `NoSystem` adopting a bound operand's system. Two different
bound systems raise `TypeError`.

```python
from typing import assert_type

import quantype.numpy as qnp
from quantype import Angle, Area, Dimensionless, Length, u
from quantype.systems import SI

assert_type(qnp.cos(Angle[float](60, u.deg)), Dimensionless[float])
assert_type(qnp.sqrt(Area[float, SI](4, u.angstrom_squared)), Length[float, SI])
```

## Kind-preserving array calls

| Signature | Result / restriction |
| --- | --- |
| `stack(arrays, axis=0)`, `concatenate(arrays, axis=0)` | Nonempty sequence of one kind and compatible systems; aligns units |
| `reshape(x, shape)`, `transpose(x, axes=None)`, `squeeze(x, axis=None)`, `expand_dims(x, axis)` | Same kind and system; backend reshaping conventions |
| `zeros_like(x)` | Same kind, system, and shape |
| `sum(x, axis=None, *, keepdims=False)`, `mean(...)`, `max(...)`, `min(...)` | Same kind; absolute-temperature sums rejected |
| `median(x, axis=None, *, keepdims=False)`, `nanmean(...)`, `nanmax(...)`, `nanmin(...)`, `nanmedian(...)`, `nansum(...)` | Same kind; `nansum` rejects absolute temperatures |
| `percentile(x, q, axis=None, *, keepdims=False)`, `quantile(...)` | Same kind; plain numerical `q` in 0–100 / 0–1 respectively |
| `sort(x, axis=-1)` | Same kind |
| `cumsum(x, axis=None)` | Same kind; absolute temperatures rejected |
| `diff(x, n=1, axis=-1)` | Same kind, except absolute temperatures give `TemperatureDifference` |
| `linspace(start, stop, num=50, *, endpoint=True)` | Endpoints of one kind and compatible systems; Python scalar endpoints create NumPy array storage |
| `std(x, axis=None, *, ddof=0, keepdims=False)`, `nanstd(...)` | Same kind, except absolute temperatures give `TemperatureDifference` |
| `var(x, axis=None, *, ddof=0, keepdims=False)`, `nanvar(...)` | Squared kind, or `TemperatureDifferenceSquared` for absolute temperatures |

Ellipses in this table mean the same parameters as the first signature on that
row, not support for arbitrary NumPy keywords. `out=` cannot write into a
quantity. Backend-specific optional numerical keywords are not a portable API.

## Plain numerical and boolean results

- `argmin(x, axis=None)`, `argmax(x, axis=None)`, and `argsort(x, axis=-1)`
  return plain backend indices; `sign(x)` returns plain numbers.
- `isnan(x)`, `isfinite(x)`, `isinf(x)` return plain backend booleans.
- `isclose(a, b, *, rtol=1e-5, atol=0, equal_nan=False)` returns backend
  booleans. Inputs need the same kind and compatible systems; nonzero `atol`
  must be a quantity of the compared kind, not a bare number.
- `allclose(a, b, *, rtol=1e-5, atol=0, equal_nan=False)` uses the same
  tolerances and returns a Python `bool`. Converting traced booleans to Python
  is not generally valid inside JAX transformations.

For temperature-difference tolerances and assertion diagnostics, prefer
[`quantype.testing.assert_allclose`](../how-to/test-quantities.md).

## Quantity methods, views, and storage

The shared method surface includes indexing, iteration, `len(q)`, `.shape`,
`.ndim`, `.size`, `.dtype`, `.T`, `.reshape(*shape)`, `.transpose(*axes)`,
`.squeeze(axis=None)`, `.ravel()`, `.flatten()`, `.copy()`, `.cumsum(axis=None)`,
`.argmin(axis=None)`, `.argmax(axis=None)`, and `.argsort(axis=-1)`. Reductions
`.sum`, `.mean`, `.min`, `.max` accept `axis=None, *, keepdims=False`;
`.std` and `.var` also accept keyword `ddof=0`. The physical rules above apply
to methods too.

Array methods retain kind, system, and storage type unless the operation changes
kind as described above. A NumPy array reduced or indexed to one element stays
a zero-dimensional array, not a NumPy scalar. `.item()` returns a quantity with
Python `float` storage. Basic NumPy indexing shares memory; integer/boolean
array indexing copies. `.copy()` makes an independent numerical copy. Other
view/copy behavior follows the backend. See [array tasks](../how-to/arrays.md).

## Runtime dispatch versus static support

!!! info "Native runtime dispatch is not static support"

    NumPy native ufuncs and registered array functions, and Torch registered calls
    such as `torch.cos` and `torch.linalg.vector_norm`, apply the same unit rules at
    runtime. NumPy stubs reject quantities, so use `qnp` for typed code. Native
    `jax.numpy` has no quantity-dispatch hook: use `qnp` for JAX, including traces.

Generated overloads refine named result kinds and carry storage/system
parameters; they do not track array shapes or arbitrary dtype promotion. A
real scalar or numerical array scales a quantity without changing its dtype;
scaling a scalar quantity by an array produces array storage.

!!! warning "Unsupported mathematics needs an explicit boundary"

    Wrong physical kinds, incompatible systems, implicit `np.asarray(q)`, and calls
    without a registered unit rule (such as FFTs) raise rather than drop units.
    [Extract magnitudes or apply a unit-invariant function](../how-to/arrays.md#apply-a-unit-invariant-operation)
    at an explicit boundary; [`q.apply`](quantities.md#conversion-and-numerical-access)
    is not a blanket unit rule for arbitrary math. See
    [troubleshooting](../troubleshooting.md#unsupported-numerical-calls).

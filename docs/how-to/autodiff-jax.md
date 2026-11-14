# Differentiate and compile with JAX

Install the JAX extra:

```bash
uv add 'quantype[jax]'
```

Use `ujax` for quantity-valued functions and `quantype.numpy` for numerical
operations on quantities. The examples run on CPU. The
[autodiff reference](../reference/autodiff.md#jax-api) defines derivatives,
transformations, unit systems, and static limitations.

## Calculate energy, gradient, and Hessian

Use physically typed coefficients when the model is written with quantities:

<!-- requires: jax -->
```python
import jax
import numpy as np

from quantype import Energy, Force, ForceConstant, Length, u, ujax


def harmonic(x: Length[jax.Array], k: ForceConstant[float]) -> Energy[jax.Array]:
    return 0.5 * k * (x**2).sum()


x = Length[jax.Array]([1, 2, 3], u.angstrom)
k = ForceConstant[float](2.0, u.eV_per_angstrom_squared)
energy, gradient = ujax.jit(ujax.value_and_grad(harmonic))(x, k)
hessian = ujax.hessian(harmonic)(x, k)
force = -ujax.grad(harmonic)(x, k)
assert isinstance(gradient, Force)
assert isinstance(hessian, ForceConstant)
np.testing.assert_allclose(energy.magnitude(u.eV), 14.0)
np.testing.assert_allclose(force.magnitude(u.eV_per_angstrom), [-2, -4, -6])
np.testing.assert_allclose(hessian.magnitude(u.eV_per_angstrom_squared), 2 * np.eye(3))
```

`value_and_grad` gets the energy and positive gradient from one model evaluation;
physical force is the **negative** gradient. Further arguments pass through
unchanged and keep their static types. The default first-argument derivative
has the precise static result `Force[jax.Array]`, and its Hessian
`ForceConstant[jax.Array]`.

## Differentiate several arguments

<!-- requires: jax -->
```python
import jax
import numpy as np

from quantype import Energy, Force, Length, u, ujax


def pair(x: Length[jax.Array], y: Length[jax.Array]) -> Energy[jax.Array]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * ((x - y) ** 2).sum()


x = Length[jax.Array]([1.0], u.angstrom)
y = Length[jax.Array]([3.0], u.angstrom)
on_x, on_y = ujax.grad(pair, argnums=(0, 1))(x, y)
assert isinstance(on_y, Force)
np.testing.assert_allclose(on_x.magnitude(u.eV_per_angstrom), [-4])
np.testing.assert_allclose(on_y.magnitude(u.eV_per_angstrom), [4])
```

An integer `argnums` selects one positional quantity; a tuple returns a tuple of
derivatives. Explicit nondefault `argnums` has an untyped derivative signature;
do not infer static guarantees from this runtime example.

## Use numerical functions inside a trace

<!-- requires: jax -->
```python
import jax

import quantype.numpy as qnp
from quantype import Length, u, ujax


@ujax.jit
def spread(positions: Length[jax.Array]) -> Length[jax.Array]:
    return qnp.linalg.norm(positions - positions.mean(axis=0), axis=-1).max()


positions = Length[jax.Array]([[0.0, 0.0], [3.0, 4.0]], u.angstrom)
assert float(spread(positions).magnitude(u.angstrom)) == 2.5
```

Do not use native `jnp` calls on quantities. If extracting plain numbers in a
trace, always name the unit, such as `x.magnitude(u.nm)`; a `NoSystem` quantity
is flattened into reference units, not its original display unit. See the
[pytree contract](../reference/autodiff.md#jax-pytrees-and-transformations).

## Bind the calculation to a unit system

<!-- requires: jax -->
```python
import jax
import numpy as np

from quantype import Energy, ForceConstant, Length, u, ujax
from quantype.systems import SI


def harmonic_si(x: Length[jax.Array, SI]) -> Energy[jax.Array, SI]:
    k = ForceConstant[float, SI](2.0, u.eV_per_angstrom_squared)
    return 0.5 * k * (x**2).sum()


x = Length[jax.Array, SI]([0.1, 0.2], u.nm)
force = -ujax.grad(harmonic_si)(x)
assert force.system is SI
np.testing.assert_allclose(force.magnitude(u.eV_per_angstrom), [-2, -4], rtol=1e-5)
```

The result is `Force[jax.Array, SI]`, with raw values in newtons. Inputs and
output must share a system; adapters reject mismatches. See
[systems](../reference/systems.md) and [define systems](define-systems.md).

!!! info "Enable x64 before requesting float64"
    JAX float64 construction requires x64 enabled, for example through
    `JAX_ENABLE_X64=1` before starting Python. An unavailable explicit `dtype=`
    fails rather than silently falling back.

[Serialization](serialize.md) saves values only, not device placement or
autodiff state.

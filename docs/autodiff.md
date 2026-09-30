# JAX and Torch autodiff

Optional backends use the same physical algebra as scalar and NumPy storage.
Install the corresponding extra:

```bash
uv add 'quantype[jax]'
uv add 'quantype[torch]'
```

Derivatives come back as physical kinds: an energy differentiated with respect
to a length is a `Force`, and a second derivative a `ForceConstant`. They are
positive derivatives; the physical force is `-gradient`.

## JAX

```python
import jax
from quantype import Energy, ForceConstant, Length, u, ujax


def harmonic(x: Length[jax.Array], k: ForceConstant[float]) -> Energy[jax.Array]:
    return 0.5 * k * (x**2).sum()


x = Length[jax.Array]([1, 2, 3], u.length.angstrom)
k = ForceConstant[float](2.0, u.eV_per_angstrom_squared)
energy, gradient = ujax.jit(ujax.value_and_grad(harmonic))(x, k)
hessian = ujax.hessian(harmonic)(x, k)
force = -ujax.grad(harmonic)(x, k)
```

`value_and_grad` returns the energy and its gradient from one evaluation, the
usual energy-and-forces call. `grad`, `value_and_grad`, and `hessian`
differentiate with respect to the first argument; further arguments, such as
model parameters, coefficients, or neighbour lists, pass through untouched and
keep their static types. Above, `gradient` is a `Force[jax.Array]` and
`hessian` a `ForceConstant[jax.Array]`.

`argnums` differentiates other arguments, or several at once:

```python
import jax
from quantype import Energy, Force, Length, u, ujax


def pair(x: Length[jax.Array], y: Length[jax.Array]) -> Energy[jax.Array]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * ((x - y) ** 2).sum()


x = Length[jax.Array]([1.0], u.angstrom)
y = Length[jax.Array]([3.0], u.angstrom)
on_x, on_y = ujax.grad(pair, argnums=(0, 1))(x, y)
assert isinstance(on_y, Force)
```

A tuple gives a tuple of derivatives. Types are precise for the first argument;
other `argnums` return an untyped result.

Importing `ujax` registers quantities as single-leaf pytrees for JAX `jit` and
`vmap`. The tree metadata of a quantity is its kind and unit system; the display
unit is not part of it. Values presented in different units therefore share one
trace, and `lax.cond` and `lax.scan` accept them, but results come back
presented in the system's unit. Quantities from different systems are different
tree structures, as they are different static types.

JAX's `jnp` functions cannot see quantities, and JAX offers no hook that would
let them. Inside JAX code, use `quantype.numpy`, which has NumPy's names with
unit rules and works on traced values:

```python
import jax
import quantype.numpy as qnp
from quantype import Length, u, ujax


@ujax.jit
def spread(positions: Length[jax.Array]) -> Length[jax.Array]:
    return qnp.linalg.norm(positions - positions.mean(axis=0), axis=-1).max()


positions = Length[jax.Array]([[0.0, 0.0], [3.0, 4.0]], u.angstrom)
assert float(spread(positions).value) == 2.5
```

## Torch

```python
import torch
from quantype import Length, u, utorch

x = Length[torch.Tensor](
    torch.tensor([1.0, 2.0], requires_grad=True),
    u.length.nanometer,
    dtype=torch.float64,
)
y = Length[torch.Tensor](torch.tensor([0.5], requires_grad=True), u.angstrom)
k = 2.0 * u.eV_per_angstrom_squared
energy = 0.5 * k * ((x**2).sum() + (y**2).sum())
gradient = utorch.grad(energy, x, create_graph=True)
on_x, on_y = utorch.grad(energy, [x, y])
force = -gradient
```

`utorch.grad` differentiates a scalar output with respect to one quantity, or a
sequence of them, which gives a tuple. Torch differentiation requires an input
tensor with `requires_grad=True`; constructing `Length[torch.Tensor]` from a list
does not enable gradients. `create_graph=True` keeps derivatives differentiable
for higher orders.

`torch.cos(angle)`, `torch.linalg.vector_norm(positions)`, and the other
functions of `quantype.numpy` apply their unit rules to quantities through
Torch's dispatch, as NumPy's do. To extract host values explicitly, use
`force.magnitude(u.eV_per_angstrom).detach().cpu().numpy()`. This detaches the
graph; arithmetic does not.

A raw model's output can be wrapped with `from_value`, a trusted boundary: the
caller asserts that the raw result represents, say, energy in the unit system's
eV. A model written entirely in quantity algebra, with physical coefficients as
above, needs no such assertion.

## Unit systems

```python
import jax
from quantype import Energy, ForceConstant, Length, u, ujax
from quantype.systems import SI


def harmonic_si(x: Length[jax.Array, SI]) -> Energy[jax.Array, SI]:
    k = ForceConstant[float, SI](2.0, u.eV_per_angstrom_squared)
    return 0.5 * k * (x**2).sum()


x = Length[jax.Array, SI]([0.1, 0.2], u.nm)
force = -ujax.grad(harmonic_si)(x)  # Force[jax.Array, SI], in newtons
```

Because quantities from different systems never mix, a function's output and
inputs share one system, and the raw derivative is already in that system's
coherent units. Where a system stores a kind in its own units, such as LAMMPS
pressure in bar, the derivative is rescaled accordingly. The adapters raise
`TypeError` for an output in another system. The static signatures are generic
over the system, so `Length[Array, S]` to `Energy[Array, S]` differentiates to
`Force[Array, S]`.

## Storage and dtype

Torch/JAX classes do not encode dtype, so constructors and `load_npz` accept an
explicit `dtype=`. Existing Torch tensors retain their graph and device during
conversion and scaling. JAX float64 requests require x64 support; unavailable
explicit dtypes fail rather than silently producing a different dtype.

[Serialization](serialization.md) is an explicit host boundary and does not
preserve graphs or device placement.

# JAX and Torch autodiff

JAX and Torch arrays follow the same kinds and rules as floats and NumPy
arrays. Install the extra for your backend:

```bash
uv add 'quantype[jax]'
uv add 'quantype[torch]'
```

Derivatives come back as kinds. An energy differentiated with respect to a
length is a `Force`, and its second derivative is a `ForceConstant`. The
gradient is the positive derivative, so the physical force is `-gradient`.

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

`value_and_grad` returns the energy and its gradient from one evaluation, which
is the usual way to get energies and forces together. `grad`, `value_and_grad`,
and `hessian` differentiate with respect to the first argument. Other
arguments, such as model parameters or neighbour lists, pass through unchanged
and keep their static types. Above, `gradient` is a `Force[jax.Array]` and
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

A tuple of `argnums` gives a tuple of derivatives. Only the derivative with
respect to the first argument has a precise static type; the others are
untyped.

Importing `ujax` registers quantities as single-leaf pytrees, so `jit` and
`vmap` accept them. The pytree records a quantity's kind and unit system, but
not its display unit. A function compiles once whether its inputs were given in
nm, Å, or bohr, and `lax.cond` and `lax.scan` accept branches shown in different
units. Results come back shown in the system's unit. Quantities in different
systems have different pytree structures, just as they have different static
types.

Differentiate with `ujax.grad`, `ujax.value_and_grad`, and `ujax.hessian`.
Before `ujax` is imported, JAX's own `jax.grad` rejects a quantity as "not a
valid JAX type". After, it treats a quantity as a container, so for a function
that returns an `Energy` it raises "Gradient only defined for scalar-output
functions. Output was 2.5 eV.", although the energy is a scalar.

JAX's `jnp` functions can't see quantities, and JAX has no hook to change that.
In JAX code, use `quantype.numpy`, which works on traced values:

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

`utorch.grad` differentiates a scalar output with respect to one quantity, or
with respect to a sequence of them, which gives a tuple. The input tensor needs
`requires_grad=True`: building `Length[torch.Tensor]` from a list doesn't enable
gradients. Pass `create_graph=True` to differentiate again.

Torch's own functions, such as `torch.cos(angle)` and
`torch.linalg.vector_norm(positions)`, apply the same unit rules as
`quantype.numpy`. To get host values, use
`force.magnitude(u.eV_per_angstrom).detach().cpu().numpy()`. That detaches the
graph; arithmetic on quantities keeps it.
`.to(unit)` sets the display unit. To move a quantity to another device, or
detach it, apply the tensor method to its value: `x.with_value(x.value.to("cuda"))`
or `x.with_value(x.value.detach())`. A training loop calls
`loss.value.backward()`.

To wrap a raw model's output, use `from_value`. It trusts its input:
`Energy.from_value(raw)` asserts that `raw` is an energy in the unit system's
units. A model written with quantities throughout, as above, doesn't need it.

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

A function's inputs and output are in the same system, so the raw derivative is
already in that system's units. Where a system stores a kind in a unit of its
own, such as pressure in bar for `Metal`, the derivative is rescaled to match.
The adapters raise `TypeError` if the output is in a different system from the
input. The static signatures are generic over the system: a function from
`Length[Array, S]` to `Energy[Array, S]` differentiates to `Force[Array, S]`.

## Storage and dtype

`torch.Tensor` and `jax.Array` don't say which dtype they hold, so constructors
and `load_npz` take a `dtype=` argument. Torch tensors keep their graph and
device through conversion and scaling. JAX float64 needs x64 enabled, and a
dtype the backend can't provide raises an error instead of falling back to
another.

[Serialization](serialization.md) saves values only; graphs and device placement
are lost.

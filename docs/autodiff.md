# JAX and Torch autodiff

Optional backends use the same physical algebra as scalar and NumPy storage.
Install the corresponding extra:

```bash
uv add 'quantype[jax]'
uv add 'quantype[torch]'
```

## Torch

```python
import torch
from quantype import Energy, Length, u, utorch

x = Length[torch.Tensor](
    torch.tensor([1.0, 2.0], requires_grad=True),
    u.length.nanometer,
    dtype=torch.float64,
)
energy = Energy.from_canonical((x.value**2).sum())
gradient = utorch.grad(energy, x, create_graph=True)
force = -gradient
```

`from_canonical` is a trusted boundary: the caller asserts that the raw result
represents energy in canonical eV. For a model expressed entirely in quantity
algebra, use a physical coefficient as in the JAX example below.

Torch differentiation requires an input tensor with `requires_grad=True`;
constructing `Length[torch.Tensor]` from a list does not enable gradients.
To extract host values explicitly, use
`force.magnitude(u.eV_per_angstrom).detach().cpu().numpy()`.
This detaches the graph; arithmetic does not.

## JAX

```python
import jax
from quantype import Energy, Length, u, ujax


def harmonic(x: Length[jax.Array]) -> Energy[jax.Array]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


x = Length[jax.Array]([1, 2, 3], u.length.angstrom)
gradient = ujax.jit(ujax.grad(harmonic))(x)
hessian = ujax.hessian(harmonic)(x)
```

Importing `ujax` registers quantities as single-leaf pytrees for JAX `jit` and
`vmap`. Arithmetic and autodiff use the same physical algebra: the gradient
above is a `Force[jax.Array]` and the Hessian is a
`ForceConstant[jax.Array]`. Gradients are positive derivatives; physical force
is `-gradient`. The adapters support one quantity input and scalar output, not
general JVP/VJP or multi-input models.

## Storage and dtype

Torch/JAX classes do not encode dtype, so constructors and `load_npz` accept an
explicit `dtype=`. Existing Torch tensors retain their graph and device during
conversion. JAX float64 requests require x64 support; unavailable explicit dtypes
fail rather than silently producing a different dtype.

[Serialization](serialization.md) is an explicit host boundary and does not
preserve graphs or device placement.

# Differentiate with Torch

Install the Torch extra:

```bash
uv add 'quantype[torch]'
```

!!! info "Enable gradients on the source tensor"
    Source tensors need `requires_grad=True`; constructing a quantity from a
    list does not enable gradients. The examples run on CPU.

See [the autodiff reference](../reference/autodiff.md#torch-api-and-graph-lifetime)
for exact signatures, physical derivative kinds, systems, and errors.

## Differentiate an existing scalar output

<!-- requires: torch -->
```python
import torch

from quantype import Force, Length, u, utorch

x = Length[torch.Tensor](
    torch.tensor([1.0, 2.0], requires_grad=True), u.nm, dtype=torch.float64
)
y = Length[torch.Tensor](torch.tensor([0.5], requires_grad=True), u.angstrom)
k = 2.0 * u.eV_per_angstrom_squared
energy = 0.5 * k * ((x**2).sum() + (y**2).sum())
gradient = utorch.grad(energy, x, create_graph=True)
on_x, on_y = utorch.grad(energy, [x, y])
force = -gradient
assert isinstance(force, Force)
torch.testing.assert_close(
    force.magnitude(u.eV_per_angstrom), torch.tensor([-20.0, -40.0], dtype=torch.float64)
)
assert force.magnitude(u.eV_per_angstrom).requires_grad
```

One quantity input gives one derivative; a sequence gives a tuple. The gradient
is positive, so use `-gradient` for physical force. `create_graph=True` keeps a
derivative differentiable for higher derivatives; `retain_graph` follows Torch's
usual graph-lifetime semantics.

The energy's display composes the input units, `(eV/Å^2)·nm^2`; its bare
`.magnitude()` is 100 times its magnitude in eV. Log
`energy.magnitude(u.eV)`, naming the unit explicitly.

## Keep graphs at a raw-model boundary

When wrapping a model's raw output, supply the unit explicitly. The constructor
keeps a compatible tensor and graph:

<!-- requires: torch -->
```python
import torch

from quantype import Energy, Length, u

x = Length[torch.Tensor](torch.tensor([1.0, 2.0], requires_grad=True), u.angstrom)
raw_energy = (x.magnitude(u.angstrom) ** 2).sum()
loss = Energy[torch.Tensor](raw_energy, u.eV)
loss.magnitude(u.eV).backward()
assert x.magnitude(u.angstrom).grad is not None
```

This wrapper **asserts** the raw model's output means eV; Quantype cannot verify
the raw model's physical algebra. In a unit system,
`Energy.from_value(raw, Atomistic)` instead asserts numbers already in that
system's units. A model written entirely with quantities and physically typed
coefficients, as in the first example, needs neither escape hatch.

## Use native Torch calls and extract host values

Torch's own registered calls, including `torch.cos(angle)` and
`torch.linalg.vector_norm(positions)`, use the same physical rules as
`quantype.numpy`. Prefer `quantype.numpy` where portable typed calls are useful;
see [the numerical reference](../reference/numerical-operations.md).

Arithmetic, conversion, and scaling retain graphs and tensor devices. To move
a device or detach intentionally, use `q.apply(lambda raw: raw.to("cuda"))` or
`q.apply(lambda raw: raw.detach())`. Host extraction is an explicit detaching
boundary: `force.magnitude(u.eV_per_angstrom).detach().cpu().numpy()`.
See [array tasks](arrays.md#convert-storage-or-move-a-torch-tensor).

`torch.Tensor` annotations do not encode dtype; choose `dtype=` on construction
or `load_npz` when needed. [Serialization](serialize.md) saves only values,
detaching tensors and moving them to CPU; it does not save the graph or device.

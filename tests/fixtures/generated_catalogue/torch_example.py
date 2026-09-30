"""Generated Torch adapters return quantities from the application catalogue."""

import torch
from labquantities import Energy, Force, Length, Quantity, u, utorch


def test_gradient() -> None:
    x = Length[torch.Tensor](torch.tensor([1.0, 2.0], requires_grad=True), u.angstrom)
    energy = Energy.from_value((x.value**2).sum())

    gradient = utorch.grad(energy, x)

    assert isinstance(gradient, Force)
    torch.testing.assert_close(gradient.value, torch.tensor([2.0, 4.0]))


def test_structural_gradient() -> None:
    x = Length[torch.Tensor](torch.tensor([1.0, 2.0], requires_grad=True), u.angstrom)
    q = x * (3 * u.fs)
    gradient = utorch.grad(q.sum(), x)
    assert type(q) is Quantity
    assert type(gradient) is Quantity
    torch.testing.assert_close(gradient.value, torch.tensor([3.0, 3.0]))

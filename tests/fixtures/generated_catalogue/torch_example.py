"""Generated Torch adapters return quantities from the application catalogue."""

import torch
from labquantities import Energy, Force, Length, u, utorch


def test_gradient() -> None:
    x = Length[torch.Tensor](torch.tensor([1.0, 2.0], requires_grad=True), u.angstrom)
    energy = Energy.from_canonical((x.value**2).sum())

    gradient = utorch.grad(energy, x)

    assert isinstance(gradient, Force)
    torch.testing.assert_close(gradient.value, torch.tensor([2.0, 4.0]))

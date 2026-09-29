"""Torch works independently of the JAX extra."""

# Optional dependency gate must precede backend imports.

import pytest

pytest.importorskip("torch")
import torch

from quantype import Energy, Force, ForceConstant, Length, u, utorch


def _torch_harmonic(x: Length[torch.Tensor]) -> Energy[torch.Tensor]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


def test_torch_graph_and_higher_derivative() -> None:
    raw = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x = u.angstrom(raw)
    energy = _torch_harmonic(x)
    first = utorch.grad(energy, x, create_graph=True)
    assert isinstance(first, Force)
    assert type(first.value) is torch.Tensor
    assert first.value.requires_grad
    assert first.value.grad_fn is not None
    torch.testing.assert_close(first.value, 2.0 * raw)
    second = utorch.grad(first.sum(), x)
    assert isinstance(second, ForceConstant)
    torch.testing.assert_close(second.value, torch.full_like(raw, 2.0))
    torch.testing.assert_close((-first).value, -2.0 * raw)


def test_torch_conversion_preserves_graph_and_canonical_derivative() -> None:
    raw = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    x = u.nm(raw).to(u.angstrom)
    energy = _torch_harmonic(x)
    first = utorch.grad(energy, x, retain_graph=True)
    torch.testing.assert_close(first.value, torch.tensor([2.0, 4.0, 6.0]))
    (raw_gradient,) = torch.autograd.grad(energy.value, raw)
    torch.testing.assert_close(raw_gradient, 10.0 * first.value)


def test_torch_raw_model_boundary() -> None:
    x = u.angstrom(torch.tensor([1.0, 2.0], requires_grad=True))
    energy = Energy.from_canonical(torch.sum(x.value**2))
    torch.testing.assert_close(utorch.grad(energy, x).value, 2.0 * x.value)


def test_torch_rejects_nonscalar_energy() -> None:
    x = u.angstrom(torch.tensor([1.0, 2.0], requires_grad=True))
    energy = Energy.from_canonical(x.value**2)
    with pytest.raises(ValueError, match="scalar"):
        utorch.grad(energy, x)

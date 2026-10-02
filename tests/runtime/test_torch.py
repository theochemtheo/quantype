"""Torch works independently of the JAX extra."""

from typing import Any, cast

# Optional dependency gate must precede backend imports.
import pytest

pytest.importorskip("torch")
import torch

import quantype.numpy as qnp
from quantype import (
    Angle,
    Area,
    Dimensionless,
    Energy,
    Force,
    ForceConstant,
    Length,
    Time,
    Velocity,
    u,
    utorch,
)
from quantype.systems import SI
from quantype.testing import assert_allclose


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
    energy = Energy.from_value(torch.sum(x.value**2))
    torch.testing.assert_close(utorch.grad(energy, x).value, 2.0 * x.value)


def test_torch_rejects_nonscalar_energy() -> None:
    x = u.angstrom(torch.tensor([1.0, 2.0], requires_grad=True))
    energy = Energy.from_value(x.value**2)
    with pytest.raises(ValueError, match="scalar"):
        utorch.grad(energy, x)


def test_torch_grad_in_another_system() -> None:
    raw = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    x = Length[torch.Tensor, SI](raw, u.nm)
    k = ForceConstant[float, SI](2.0, u.eV_per_angstrom_squared)
    force = utorch.grad(0.5 * k * (x**2).sum(), x)
    assert force.system is SI
    torch.testing.assert_close(
        torch.as_tensor(force.magnitude(u.eV_per_angstrom)),
        torch.tensor([2.0, 4.0], dtype=torch.float64),
    )
    untyped: Any = utorch.grad
    with pytest.raises(
        TypeError, match="output in Atomistic with respect to an input in SI"
    ):
        untyped(Energy.from_value(x.value.sum()) * 1.0, x)


def test_torch_tensors_scale_quantities() -> None:
    length = Length[torch.Tensor](torch.tensor([1.0, 2.0]), u.angstrom)
    scaled = length * torch.tensor(2.0)
    assert torch.equal(scaled.value, torch.tensor([2.0, 4.0]))
    # The quantity's dtype is kept, as unit conversion keeps it.
    doubled = torch.tensor(2.0, dtype=torch.float64) * length
    assert doubled.value.dtype == torch.float32


def test_grad_takes_several_inputs() -> None:
    x = Length[torch.Tensor](torch.tensor([1.0, 2.0], requires_grad=True), u.angstrom)
    y = Length[torch.Tensor](torch.tensor([3.0], requires_grad=True), u.angstrom)
    k = 2.0 * u.eV_per_angstrom_squared
    energy = 0.5 * k * ((x**2).sum() + (y**2).sum())
    gradients = utorch.grad(energy, [x, y])
    assert len(gradients) == 2
    assert all(isinstance(gradient, Force) for gradient in gradients)
    assert torch.allclose(gradients[1].value, torch.tensor([6.0]))


def test_torch_functions_apply_unit_rules() -> None:
    angles = Angle[torch.Tensor](torch.tensor([0.0, 90.0]), u.deg)
    untyped_torch: Any = torch
    cosines = cast("Dimensionless[torch.Tensor]", untyped_torch.cos(angles))
    assert isinstance(cosines, Dimensionless)
    assert torch.allclose(cosines.value, torch.tensor([1.0, 0.0]), atol=1e-6)
    positions = Length[torch.Tensor](torch.tensor([[3.0, 4.0]]), u.angstrom)
    norms = cast(
        "Length[torch.Tensor]", untyped_torch.linalg.vector_norm(positions, dim=-1)
    )
    assert isinstance(norms, Length)
    assert torch.allclose(norms.value, torch.tensor([5.0]))
    assert isinstance(qnp.linalg.norm(positions, axis=-1), Length)
    stacked = untyped_torch.stack([positions, positions])
    assert stacked.shape == (2, 1, 2)
    spread = untyped_torch.std(Length[torch.Tensor](torch.tensor([1.0, 3.0]), u.nm))
    assert torch.allclose(spread.magnitude(u.nm), torch.tensor(2**0.5))  # unbiased
    with pytest.raises(TypeError, match=r"torch\.fft_fft does not know|does not know"):
        untyped_torch.fft.fft(positions)


def test_quantype_numpy_on_tensors_uses_numpys_names() -> None:
    positions = Length[torch.Tensor](
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), u.angstrom
    )
    force = Force[torch.Tensor](torch.tensor([0.0, 1.0, 0.0]), u.eV_per_angstrom)
    assert qnp.transpose(positions).shape == (3, 2)
    assert qnp.expand_dims(positions, 0).shape == (1, 2, 3)
    assert qnp.squeeze(qnp.expand_dims(positions, 0)).shape == (2, 3)
    assert qnp.squeeze(qnp.expand_dims(positions, 0), 0).shape == (2, 3)
    assert qnp.concatenate([positions, positions]).shape == (4, 3)
    assert torch.equal(qnp.cumsum(positions).value[-1], torch.tensor(3.0))
    assert torch.equal(
        qnp.cumsum(positions, axis=0).value[-1], torch.tensor([1.0, 2.0, 0.0])
    )
    assert torch.equal(
        qnp.diff(positions, axis=0).value, torch.tensor([[-1.0, 2.0, 0.0]])
    )
    assert qnp.std(positions).shape == ()
    work = qnp.dot(force, positions[1])
    assert isinstance(work, Energy)
    assert float(work.value) == 2.0
    torque = qnp.cross(positions[0], force)
    assert torch.equal(torque.value, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(
        qnp.linalg.norm(positions, axis=-1).value, torch.tensor([1.0, 2.0])
    )


def test_statistics_products_and_ranges_on_tensors() -> None:
    lengths = Length[torch.Tensor](torch.tensor([3.0, 1.0, 2.0]), u.nm)
    for result, expected in (
        (qnp.sort(lengths), [1.0, 2.0, 3.0]),
        (qnp.median(lengths), 2.0),
        (qnp.quantile(lengths, 0.5), 2.0),
        (qnp.percentile(lengths, 50), 2.0),
        (qnp.nanmean(lengths), 2.0),
        (qnp.nansum(lengths), 6.0),
        (qnp.nanmedian(lengths), 2.0),
    ):
        assert isinstance(result, Length)
        assert torch.allclose(result.magnitude(), torch.tensor(expected))
    assert int(qnp.argmin(lengths)) == 1
    assert int(qnp.argmax(lengths)) == 0
    assert qnp.argsort(lengths).tolist() == [1, 2, 0]
    assert isinstance(qnp.var(lengths), Area)
    assert isinstance(qnp.square(lengths), Area)
    start = Length[torch.Tensor](torch.tensor(0.0), u.nm)
    grid = qnp.linspace(start, start + 1 * u.nm, 3)
    assert isinstance(grid, Length)
    assert torch.allclose(grid.magnitude(), torch.tensor([0.0, 0.5, 1.0]))
    times = Time[torch.Tensor](torch.tensor([0.0, 1.0, 2.0]), u.fs)
    speeds = Velocity[torch.Tensor](torch.ones(3), u.angstrom_per_fs)
    assert float(qnp.trapezoid(speeds, times).magnitude(u.angstrom)) == 2.0
    forces = Force[torch.Tensor](torch.eye(2), u.eV_per_angstrom)
    work = forces @ Length[torch.Tensor](torch.tensor([3.0, 4.0]), u.angstrom)
    assert isinstance(work, Energy)
    assert torch.equal(work.magnitude(u.eV), torch.tensor([3.0, 4.0]))
    assert lengths.size == 3
    assert lengths.copy().value is not lengths.value
    assert type(lengths[0].item().value) is float


def test_tensor_reductions_ranges_and_clipping() -> None:
    lengths = Length[torch.Tensor](torch.tensor([[1.0, 3.0], [2.0, 0.5]]), u.nm)
    assert torch.equal(lengths.max(axis=1).magnitude(), torch.tensor([3.0, 2.0]))
    assert torch.equal(lengths.min().magnitude(), torch.tensor(0.5))
    start = Length[torch.Tensor](torch.tensor(0.0), u.nm)
    grid = qnp.linspace(start, start + 1 * u.nm, 4, endpoint=False)
    assert torch.allclose(grid.magnitude(), torch.tensor([0.0, 0.25, 0.5, 0.75]))
    area = qnp.trapezoid(lengths[0], dx=2.0)
    assert float(area.magnitude()) == 4.0
    clip: Any = torch.clip
    clipped = clip(lengths, min=1 * u.nm, max=2 * u.nm)
    assert torch.equal(clipped.magnitude(), torch.tensor([[1.0, 2.0], [2.0, 1.0]]))


def test_tensors_on_the_left_and_storage_dtypes() -> None:
    narrow = Length[torch.Tensor](torch.tensor([1.0], dtype=torch.float32), u.angstrom)
    scaled = cast("Any", torch.ones(1)) * narrow
    assert isinstance(scaled, Length)
    wide = narrow * torch.tensor([2.0], dtype=torch.float64)
    assert wide.value.dtype == torch.float32
    with pytest.raises(TypeError, match="unhashable array storage: Tensor"):
        hash(narrow)
    with pytest.raises(TypeError, match="parse accepts scalar/NumPy storage"):
        Length.parse(narrow)


def test_tensor_storage_rejects_complex_and_integer_dtypes() -> None:
    with pytest.raises(ValueError, match="magnitudes must be real numbers"):
        Length[torch.Tensor](torch.tensor([1j]), u.nm)
    with pytest.raises(ValueError, match="real floating dtype"):
        Length[torch.Tensor]([1.0], u.nm, dtype=torch.int64)


def test_assert_allclose_accepts_tensors_that_require_gradients() -> None:
    tracked = torch.tensor([1.0, 2.0], requires_grad=True)
    assert_allclose(Length[torch.Tensor](tracked, u.nm), u.nm(torch.tensor([1.0, 2.0])))

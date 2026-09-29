"""Each adapter must work when importing the other backend is impossible."""

import importlib.util
import subprocess
import sys

import pytest


@pytest.mark.parametrize(("backend", "blocked"), [("jax", "torch"), ("torch", "jax")])
def test_independent_backend(backend: str, blocked: str) -> None:
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"{backend} is not installed")
    program = f"""
import importlib.abc
import sys
class BlockOtherBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == {blocked!r}:
            raise ModuleNotFoundError(name={blocked!r})
sys.meta_path.insert(0, BlockOtherBackend())
from quantype import Length, Energy, Force, u
"""
    if backend == "jax":
        program += """
import jax
from quantype import ujax
x = Length[jax.Array]([1, 2], u.length.nanometer)
def energy(x):
    return Energy.from_canonical((x.value**2).sum())
assert isinstance(ujax.grad(energy)(x), Force)
"""
    else:
        program += """
import torch
from quantype import utorch
x = Length[torch.Tensor](torch.tensor([1., 2.], requires_grad=True), u.length.nanometer)
energy = Energy.from_canonical((x.value**2).sum())
assert isinstance(utorch.grad(energy, x), Force)
"""
    program += f"\nassert {blocked!r} not in sys.modules\n"
    subprocess.run([sys.executable, "-c", program], check=True)  # noqa: S603

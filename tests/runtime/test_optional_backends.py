"""Each adapter must work when importing the other backend is impossible."""

import importlib.util
import os
import shutil
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
    return Energy.from_value((x.value**2).sum())
assert isinstance(ujax.grad(energy)(x), Force)
"""
    else:
        program += """
import torch
from quantype import utorch
x = Length[torch.Tensor](torch.tensor([1., 2.], requires_grad=True), u.length.nanometer)
energy = Energy.from_value((x.value**2).sum())
assert isinstance(utorch.grad(energy, x), Force)
"""
    program += f"\nassert {blocked!r} not in sys.modules\n"
    subprocess.run([sys.executable, "-c", program], check=True)  # noqa: S603


# The dependency tiers CI builds, as the optional backends each should provide.
# QUANTYPE_TEST_TIER is set by every test job in check.yml; unset (a bare local
# run) skips, because there is no declared expectation to check against.
TIERS: dict[str, set[str]] = {
    "base": set(),
    "jax": {"jax"},
    "torch": {"torch"},
    "full": {"jax", "torch"},
}
# The full tier also runs the generated-catalogue checks, which skip without these.
FULL_TOOLS = ("ruff", "mypy", "pyright", "pyrefly", "ty")


def test_tier_environment_is_the_one_ci_declared() -> None:
    tier = os.environ.get("QUANTYPE_TEST_TIER")
    if tier is None:
        pytest.skip("QUANTYPE_TEST_TIER unset; not a tiered run")
    assert tier in TIERS, f"unknown tier {tier!r}; expected one of {sorted(TIERS)}"
    optional = {name for names in TIERS.values() for name in names}
    present = {name for name in optional if importlib.util.find_spec(name) is not None}
    assert present == TIERS[tier], (
        f"tier {tier!r} resolved to an unexpected environment; "
        f"missing {sorted(TIERS[tier] - present)}, "
        f"unexpected {sorted(present - TIERS[tier])}. Either the install step "
        "drifted from this table, or a dependency group changed what it pulls in."
    )
    if tier == "full":
        missing = [tool for tool in FULL_TOOLS if shutil.which(tool) is None]
        assert not missing, f"the full tier lacks {missing}"

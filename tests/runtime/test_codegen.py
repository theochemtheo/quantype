"""The public generator uses the same model/rendering as the builtin catalogue."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from quantype._registry import QuantitySpec, UnitSpec
from quantype.catalogue import builtin_catalogue
from quantype.codegen import generate, render

if TYPE_CHECKING:
    from pathlib import Path


def test_invalid_catalogue_extension() -> None:
    builtin = builtin_catalogue()
    with pytest.raises(ValueError, match="replace"):
        builtin.extend(quantities={"Length": builtin.quantities["Length"]}, units={})
    with pytest.raises(ValueError, match="inconsistent"):
        builtin.extend(
            quantities={}, units={}, relations={("mul", "Pressure", "Length"): "Energy"}
        )


@pytest.mark.skipif(
    any(
        shutil.which(tool) is None
        for tool in ("ruff", "mypy", "pyright", "pyrefly", "ty")
    ),
    reason="Generation conformance requires development tools",
)
def test_combined_generated_catalogue(tmp_path: Path) -> None:
    catalogue = builtin_catalogue().extend(
        quantities={
            "SurfaceTension": QuantitySpec((-2, 1, 0, 0, 0, 0, 0), "surface_tension")
        },
        units={"surface_tension": UnitSpec("SurfaceTension")},
        relations={("mul", "Pressure", "Length"): "SurfaceTension"},
    )
    package = tmp_path / "labquantities"
    assert generate(catalogue, package, package="labquantities")
    assert not generate(catalogue, package, package="labquantities", check=True)
    assert render(catalogue, package="labquantities") == render(
        catalogue, package="labquantities"
    )
    (tmp_path / "pyproject.toml").write_text("""[tool.pyright]
typeCheckingMode = "strict"
include = ["consumer.py"]
[tool.pyrefly]
preset = "strict"
project-includes = ["consumer.py"]
[tool.ty.rules]
all = "error"
""")
    program = tmp_path / "consumer.py"
    program.write_text("""from typing import assert_type
from labquantities import Length, Pressure, SurfaceTension, u
from quantype import Length as OriginalLength

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)
assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])
assert isinstance(length * pressure, SurfaceTension)
assert isinstance(pressure * length, SurfaceTension)
assert not isinstance(length, OriginalLength)
assert Length.parse(length.to_dict()).value == length.value
assert u.sqrt(length**2).value == length.value
cold = u.celsius(0)
assert (u.kelvin(300) - cold).kind == "TemperatureDifference"
""")
    subprocess.run([sys.executable, str(program)], check=True)  # noqa: S603
    backend_program = tmp_path / "backends.py"
    backend_program.write_text("""import importlib.util
from labquantities import Length, Energy, Force, ForceConstant, u
if importlib.util.find_spec("jax"):
    import jax
    from labquantities import ujax
    def energy(x):
        return Energy.from_canonical((x.value**2).sum())
    x = Length[jax.Array]([1, 2], u.angstrom)
    assert isinstance(ujax.jit(ujax.grad(energy))(x), Force)
    assert isinstance(ujax.hessian(energy)(x), ForceConstant)
if importlib.util.find_spec("torch"):
    import torch
    from labquantities import utorch
    x = Length[torch.Tensor](torch.tensor([1.0, 2.0], requires_grad=True), u.angstrom)
    result = Energy.from_canonical((x.value**2).sum())
    assert isinstance(utorch.grad(result, x), Force)
""")
    subprocess.run([sys.executable, str(backend_program)], check=True)  # noqa: S603
    # Static verification is also reproducible without committing generated
    # application packages. Full checker commands are exercised by this test.
    for command in (
        ["mypy", "--strict", "--follow-imports=silent"],
        ["pyright"],
        ["pyrefly", "check"],
        ["ty", "check"],
    ):
        run = subprocess.run(  # noqa: S603 -- fixed checker commands
            [*command, str(program)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0, run.stdout + run.stderr

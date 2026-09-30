"""Generate a real application project; test its contracts in isolated processes."""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from quantype.catalogue import QuantitySpec, UnitSpec, builtin_catalogue
from quantype.codegen import generate, render

if TYPE_CHECKING:
    from quantype.catalogue import Catalogue

PROJECT_FIXTURE = Path(__file__).parents[1] / "fixtures" / "generated_catalogue"


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    return builtin_catalogue().extend(
        quantities={
            "SurfaceTension": QuantitySpec((-2, 1, 0, 0, 0, 0, 0), "surface_tension")
        },
        units={"surface_tension": UnitSpec("SurfaceTension")},
        relations={("mul", "Pressure", "Length"): "SurfaceTension"},
    )


@pytest.fixture(scope="module")
def generated_project(
    tmp_path_factory: pytest.TempPathFactory, catalogue: Catalogue
) -> Path:
    if shutil.which("ruff") is None:
        pytest.skip("Generating an application package requires Ruff")
    project = tmp_path_factory.mktemp("generated_catalogue")
    shutil.copytree(PROJECT_FIXTURE, project, dirs_exist_ok=True)
    generate(catalogue, project / "labquantities", package="labquantities")
    return project


def _run(project: Path, *command: str) -> None:
    result = subprocess.run(  # noqa: S603 -- fixed test/checker commands, no shell
        command,
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command: {shlex.join(command)}\n"
        f"Working directory: {project}\n"
        f"Exit code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_invalid_catalogue_extension() -> None:
    builtin = builtin_catalogue()
    with pytest.raises(ValueError, match="replace"):
        builtin.extend(quantities={"Length": builtin.quantities["Length"]}, units={})
    with pytest.raises(ValueError, match="inconsistent"):
        builtin.extend(
            quantities={}, units={}, relations={("mul", "Pressure", "Length"): "Energy"}
        )


@pytest.mark.parametrize(
    "name",
    [
        "sqrt",
        "sin",
        "exp",
        "get_unit",
        "runtime",
        "Unit",
        "cast",
        "Any",
        "globals",
        "__name__",
        "Length",
        "LengthKind",
        "_LengthNamespace",
    ],
)
@pytest.mark.parametrize("as_alias", [False, True])
def test_unit_identifiers_cannot_shadow_generated_bindings(
    name: str, *, as_alias: bool
) -> None:
    units = (
        {"lab_length": UnitSpec("Length", aliases=(name,))}
        if as_alias
        else {name: UnitSpec("Length")}
    )
    with pytest.raises(
        ValueError, match=f"{name!r} conflicts with a generated API binding"
    ):
        builtin_catalogue().extend(quantities={}, units=units)


@pytest.mark.parametrize("kind", ["Sqrt", "Sin", "Exp", "GetUnit", "Runtime"])
def test_quantity_namespaces_cannot_shadow_generated_bindings(kind: str) -> None:
    with pytest.raises(ValueError, match="Conflicting quantity namespace"):
        builtin_catalogue().extend(
            quantities={kind: QuantitySpec((1, 0, 0, 0, 0, 0, 0), "lab_unit")},
            units={"lab_unit": UnitSpec(kind)},
        )


def test_quantity_namespaces_are_unique() -> None:
    with pytest.raises(ValueError, match="Conflicting quantity namespace 'length'"):
        builtin_catalogue().extend(
            quantities={"length": QuantitySpec((1, 0, 0, 0, 0, 0, 0), "lab_unit")},
            units={"lab_unit": UnitSpec("length")},
        )


def test_rendering_is_deterministic(catalogue: Catalogue) -> None:
    assert render(catalogue, package="labquantities") == render(
        catalogue, package="labquantities"
    )


def test_generated_files_are_current(
    generated_project: Path, catalogue: Catalogue
) -> None:
    assert not generate(
        catalogue,
        generated_project / "labquantities",
        package="labquantities",
        check=True,
    )


def test_stale_check_does_not_rewrite_files(
    tmp_path: Path, generated_project: Path, catalogue: Catalogue
) -> None:
    # Modify a private copy so this test cannot affect runtime/checker consumers.
    package = tmp_path / "labquantities"
    shutil.copytree(generated_project / "labquantities", package)
    stale_file = package / "units.pyi"
    stale_source = "# Deliberately stale unit definitions.\n"
    stale_file.write_text(stale_source)

    stale = generate(catalogue, package, package="labquantities", check=True)

    assert stale == ["units.pyi"]
    assert stale_file.read_text() == stale_source


@pytest.mark.parametrize(
    "case",
    [
        "physical_algebra",
        "serialization_roundtrip",
        "math_preserves_catalogue",
        "temperature_difference",
    ],
)
def test_generated_runtime(generated_project: Path, case: str) -> None:
    _run(
        generated_project,
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"runtime_cases.py::test_{case}",
    )


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_generated_backend(generated_project: Path, backend: str) -> None:
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"{backend} is not installed")
    _run(
        generated_project,
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"{backend}_example.py",
    )


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(("mypy", "--strict", "--follow-imports=silent"), id="mypy"),
        pytest.param(("pyright",), id="pyright"),
        pytest.param(("pyrefly", "check"), id="pyrefly"),
        pytest.param(("ty", "check"), id="ty"),
    ],
)
def test_generated_typing(generated_project: Path, command: tuple[str, ...]) -> None:
    if shutil.which(command[0]) is None:
        pytest.skip(f"{command[0]} is not installed")
    _run(generated_project, *command, "consumer.py")

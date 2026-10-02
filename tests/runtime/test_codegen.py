"""Generate a real application project; test its contracts in isolated processes."""

from __future__ import annotations

import ast
import importlib.util
import io
import re
import shlex
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path

import numpy as np
import pytest

from quantype.catalogue import Catalogue, QuantitySpec, UnitSpec, builtin_catalogue
from quantype.codegen import generate, render
from quantype.codegen._compact import minify

PROJECT_FIXTURE = Path(__file__).parents[1] / "fixtures" / "generated_catalogue"


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    return builtin_catalogue().extend(
        quantities={
            "SurfaceTension": QuantitySpec((-2, 1, 0, 0, 0, 0, 0, 0), "surface_tension")
        },
        units={
            "surface_tension": UnitSpec("SurfaceTension"),
            "lab_sqrt2": UnitSpec("Length", scale=np.sqrt(2.0)),
            "lab_point": UnitSpec("Temperature", offset=np.float64(1.5)),
        },
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
        "_typing",
        "_get_unit",
        "_runtime",
        "_NAMES",
        "overload",
        "Unit",
        "np",
        "Any",
        "globals",
        "__name__",
        "__init__",
        "__new__",
        "__getattribute__",
        "__setattr__",
        "__class__",
        "__dict__",
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


@pytest.mark.parametrize("kind", ["Overload", "Np", "Globals"])
def test_quantity_namespaces_cannot_shadow_generated_bindings(kind: str) -> None:
    with pytest.raises(ValueError, match="Conflicting quantity namespace"):
        builtin_catalogue().extend(
            quantities={kind: QuantitySpec((1, 0, 0, 0, 0, 0, 0, 0), "lab_unit")},
            units={"lab_unit": UnitSpec(kind)},
        )


@pytest.mark.parametrize(
    "kind",
    [
        "u",
        "LengthKind",
        "Any",
        "Literal",
        "override",
        "np",
        "npt",
        "_units",
        "_BaseQuantity",
        "_StructuralQuantity",
        "Quantity",
        "units",
        "kinds",
        "ujax",
        "utorch",
        "_generated",
        "_catalogue",
        "numpy",
        "Callable",
        "Array",
        "Tensor",
        "__name__",
        "__init__",
        "V",
        "W",
        "_LengthUnit",
        "_LengthNamespace",
    ],
)
def test_quantity_names_cannot_shadow_generated_bindings(kind: str) -> None:
    with pytest.raises(ValueError, match="Invalid quantity definition"):
        builtin_catalogue().extend(
            quantities={kind: QuantitySpec((1, 0, 0, 0, 0, 0, 0, 0), "lab_unit")},
            units={"lab_unit": UnitSpec(kind)},
        )


def test_quantity_marker_collision_is_order_independent() -> None:
    for names in (("Sample", "SampleKind"), ("SampleKind", "Sample")):
        with pytest.raises(
            ValueError, match="Invalid quantity definition 'SampleKind'"
        ):
            builtin_catalogue().extend(
                quantities={
                    name: QuantitySpec((1, 0, 0, 0, 0, 0, 0, 0), f"lab_{name}")
                    for name in names
                },
                units={f"lab_{name}": UnitSpec(name) for name in names},
            )


def test_quantity_namespaces_are_unique() -> None:
    with pytest.raises(ValueError, match="Conflicting quantity namespace 'length'"):
        builtin_catalogue().extend(
            quantities={"length": QuantitySpec((1, 0, 0, 0, 0, 0, 0, 0), "lab_unit")},
            units={"lab_unit": UnitSpec("length")},
        )


def test_canonical_unit_cannot_override_namespace_constructor() -> None:
    with pytest.raises(ValueError, match="conflicts with a generated API binding"):
        builtin_catalogue().extend(
            quantities={"Sample": QuantitySpec((0, 0, 0, 0, 0, 0, 0, 0), "__init__")},
            units={"__init__": UnitSpec("Sample")},
        )


@pytest.mark.parametrize(
    ("scale", "offset"),
    [(True, 0.0), (1.0, True), (float("inf"), 0.0), (1.0, float("nan"))],
)
def test_unit_factors_require_portable_real_numbers(
    scale: float, offset: float
) -> None:
    with pytest.raises(ValueError, match="Invalid conversion"):
        builtin_catalogue().extend(
            quantities={},
            units={"lab_point": UnitSpec("Temperature", scale=scale, offset=offset)},
        )


def test_rendering_is_deterministic(catalogue: Catalogue) -> None:
    assert render(catalogue, package="labquantities") == render(
        catalogue, package="labquantities"
    )


def _checker_comments(source: str) -> list[str]:
    return [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
        and token.string.startswith(("# mypy:", "# pyright:", "# pyrefly:", "# ty:"))
    ]


def _unminified(tree: ast.Module) -> ast.Module:
    """Undo minification's short names for ``overload`` and ``override``."""
    names = {"_o": "overload", "_v": "override"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = names.get(node.id, node.id)
        elif isinstance(node, ast.alias) and node.asname in names:
            node.asname = None
    return tree


def test_minifying_keeps_each_private_stub_module() -> None:
    sources = render(builtin_catalogue())
    private = [
        name
        for name in sources
        if name.startswith(("_products/", "_constants/")) or name == "_generated.pyi"
    ]
    assert len(private) > 1700
    for name in private:
        source = sources[name]
        minified = minify(source)
        assert len(minified) <= len(source)
        assert ast.dump(_unminified(ast.parse(minified))) == ast.dump(
            ast.parse(source)
        ), name
        assert _checker_comments(minified) == _checker_comments(source), name


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


def test_unrendered_product_modules_are_stale_and_removed(
    tmp_path: Path, generated_project: Path, catalogue: Catalogue
) -> None:
    package = tmp_path / "labquantities"
    shutil.copytree(generated_project / "labquantities", package)
    leftover = package / "_products" / "RenamedQuantityTime.pyi"
    leftover.write_text("# A product class from an older catalogue.\n")

    assert generate(catalogue, package, package="labquantities", check=True) == [
        "_products/RenamedQuantityTime.pyi"
    ]
    assert leftover.exists()
    generate(catalogue, package, package="labquantities")
    assert not leftover.exists()


@pytest.mark.parametrize(
    "case",
    [
        "physical_algebra",
        "serialization_roundtrip",
        "math_preserves_catalogue",
        "temperature_difference",
        "structural_reciprocals",
        "portable_unit_factors",
        "unit_systems_cover_new_kinds",
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

    # Negative generated consumers need per-line diagnostics, not just a
    # nonzero checker exit that could hide a different generation failure.
    expected = {
        index
        for index, line in enumerate(
            (generated_project / "invalid.py").read_text().splitlines(), start=1
        )
        if line.endswith("# error")
    }
    result = subprocess.run(  # noqa: S603 -- fixed checker commands, no shell
        (*command, "invalid.py"),
        cwd=generated_project,
        capture_output=True,
        text=True,
        check=False,
    )
    # Checkers color their output when FORCE_COLOR is set, even into a pipe.
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
    reported = {int(line) for line in re.findall(r"invalid\.py:(\d+):", output)}
    assert result.returncode == 1, output
    assert expected <= reported, output


_needs_ruff = pytest.mark.skipif(
    shutil.which("ruff") is None, reason="Generation requires Ruff"
)


@_needs_ruff
def test_the_builtin_package_is_current() -> None:
    package = Path(__file__).parents[2] / "src" / "quantype"
    assert generate(builtin_catalogue(), package, check=True) == []


@_needs_ruff
def test_checking_an_empty_directory_reports_every_file(tmp_path: Path) -> None:
    stale = generate(builtin_catalogue(), tmp_path, check=True)
    assert {"_generated.pyi", "products.py", "numpy.pyi"} <= set(stale)
    assert not any(tmp_path.iterdir())


def _missing(name: str) -> None:
    del name


def test_generation_needs_a_package_name_ruff_and_the_core_kinds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Expected a Python package name"):
        render(builtin_catalogue(), package="lab-quantities")
    alone = Catalogue(
        quantities={"Length": QuantitySpec((1, 0, 0, 0, 0, 0, 0, 0), "angstrom")},
        units={"angstrom": UnitSpec("Length")},
    )
    with pytest.raises(ValueError, match="combined builtin catalogue"):
        render(alone, package="lengths")
    monkeypatch.setattr(shutil, "which", _missing)
    with pytest.raises(RuntimeError, match="Ruff on PATH"):
        generate(builtin_catalogue(), tmp_path)

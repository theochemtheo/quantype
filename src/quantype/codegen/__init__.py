"""Deterministic source generation shared by quantype and application catalogues.

``render`` is pure. ``generate`` additionally formats and writes files and needs
Ruff on PATH. External catalogues generate a complete, separate nominal API;
import its quantities consistently rather than mixing them with quantype's.
"""

from __future__ import annotations

import keyword
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from quantype.codegen._package import package_outputs
from quantype.codegen._render import outputs

if TYPE_CHECKING:
    from quantype.catalogue import Catalogue


def render(catalogue: Catalogue, *, package: str = "quantype") -> dict[str, str]:
    if not all(
        part.isidentifier() and not keyword.iskeyword(part)
        for part in package.split(".")
    ):
        raise ValueError("Expected a Python package name")
    catalogue.validate()
    sources = {
        Path(path).name: source for path, source in outputs(catalogue, package).items()
    }
    if package != "quantype":
        required = {
            "Length",
            "Area",
            "Angle",
            "Dimensionless",
            "Temperature",
            "TemperatureDifference",
        }
        if not required <= catalogue.quantities.keys():
            raise ValueError(
                "External generation currently requires a combined builtin catalogue"
            )
        sources.update(package_outputs(catalogue, package))
    return sources


def generate(
    catalogue: Catalogue,
    output: str | Path,
    *,
    package: str = "quantype",
    check: bool = False,
) -> list[str]:
    """Write/check a package directory, returning names of changed/stale files."""
    formatter = shutil.which("ruff")
    if formatter is None:
        raise RuntimeError("Code generation requires Ruff on PATH")
    stale: list[str] = []
    directory = Path(output)
    for name, source in render(catalogue, package=package).items():
        content = source
        if name.endswith((".py", ".pyi")):
            content = subprocess.run(  # noqa: S603 -- resolved formatter, no shell
                [
                    formatter,
                    "check",
                    "--fix",
                    "--select",
                    "I",
                    "--stdin-filename",
                    name,
                    "-",
                ],
                input=content,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            content = subprocess.run(  # noqa: S603 -- resolved formatter, no shell
                [formatter, "format", "--stdin-filename", name, "-"],
                input=content,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        path = directory / name
        if not path.exists() or path.read_text() != content:
            stale.append(name)
            if not check:
                directory.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
    return stale

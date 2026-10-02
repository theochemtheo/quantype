"""Deterministic source generation shared by quantype and application catalogues.

``render`` is pure. ``generate`` additionally formats and writes files and needs
Ruff on PATH. External catalogues generate a complete, separate nominal API;
import its quantities consistently rather than mixing them with quantype's.
"""

from __future__ import annotations

import keyword
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from quantype.catalogue import (
    _QUANTITY_BINDINGS,  # pyright: ignore[reportPrivateUsage]
)
from quantype.codegen._compact import minify
from quantype.codegen._package import package_outputs
from quantype.codegen._products import product_outputs
from quantype.codegen._render import outputs
from quantype.codegen._table import naming_table

if TYPE_CHECKING:
    from collections.abc import Collection

    from quantype.catalogue import Catalogue


def render(catalogue: Catalogue, *, package: str = "quantype") -> dict[str, str]:
    if not all(
        part.isidentifier() and not keyword.iskeyword(part)
        for part in package.split(".")
    ):
        raise ValueError("Expected a Python package name")
    catalogue.validate()
    table = naming_table(catalogue, reserved=_QUANTITY_BINDINGS)
    sources = outputs(catalogue, package, table)
    sources.update(product_outputs(catalogue, package, table))
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


# Paths per Ruff run: every platform caps the length of a command line.
_BATCH = 200


def _private_stub(name: str) -> bool:
    """A module that is minified (see ``quantype.codegen._compact``)."""
    return name.startswith(("_products/", "_constants/")) or name == "_generated.pyi"


def _formatted(formatter: str, package: str, sources: dict[str, str]) -> dict[str, str]:
    """Sort imports and format the Python sources, many files per Ruff run.

    The private stub modules are then minified.

    Starting Ruff once per file dominated generation. The files are written
    under the working directory, so Ruff reads the configuration it would read
    for files there, as it did for ``--stdin-filename``.
    """
    # Ruff otherwise classifies imports partly by which modules exist on disk,
    # which changes as generation adds product modules.
    first_party = package.split(".", maxsplit=1)[0]
    python = [name for name in sources if name.endswith((".py", ".pyi"))]
    formatted = dict(sources)
    with tempfile.TemporaryDirectory(prefix=".quantype-generate-", dir=".") as scratch:
        root = Path(scratch).resolve()
        for name in python:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(sources[name], encoding="utf-8")
        paths = [str(root / name) for name in python]
        for start in range(0, len(paths), _BATCH):
            batch = paths[start : start + _BATCH]
            for command in (
                [
                    "check",
                    "--fix",
                    "--select",
                    "I,RUF022",
                    "--config",
                    f"lint.isort.known-first-party = [{first_party!r}]",
                ],
                ["format"],
            ):
                subprocess.run(  # noqa: S603 -- resolved formatter, no shell
                    [formatter, *command, "--quiet", *batch],
                    capture_output=True,
                    check=True,
                )
        for name in python:
            text = (root / name).read_text(encoding="utf-8")
            formatted[name] = minify(text) if _private_stub(name) else text
    return formatted


def generate(
    catalogue: Catalogue,
    output: str | Path,
    *,
    package: str = "quantype",
    check: bool = False,
) -> list[str]:
    """Write/check a package directory, returning names of changed/stale files.

    The generator owns ``_products/`` and ``_constants/``: a module there that it
    no longer renders is stale, and is removed.
    """
    formatter = shutil.which("ruff")
    if formatter is None:
        raise RuntimeError("Code generation requires Ruff on PATH")
    stale: list[str] = []
    directory = Path(output)
    sources = render(catalogue, package=package)

    formatted = _formatted(formatter, package, sources)
    for name, content in formatted.items():
        path = directory / name
        if not path.exists() or path.read_text() != content:
            stale.append(name)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
    return stale + _unrendered(directory, formatted.keys(), check=check)


def _unrendered(
    directory: Path, rendered: Collection[str], *, check: bool
) -> list[str]:
    """Modules in the generator's own folders that it no longer renders."""
    stale: list[str] = []
    for owned in ("_products", "_constants"):
        folder = directory / owned
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            name = f"{owned}/{path.name}"
            if path.is_file() and name not in rendered:
                stale.append(name)
                if not check:
                    path.unlink()
    return stale

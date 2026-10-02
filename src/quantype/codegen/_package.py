"""Package scaffolding for a combined application catalogue."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantype._internal._registry import unit_specs

if TYPE_CHECKING:
    from quantype.catalogue import Catalogue

# Source templates deliberately keep each generated statement together.
# ruff: noqa: E501


def package_outputs(catalogue: Catalogue, package: str) -> dict[str, str]:
    builtin_units = unit_specs()
    units = ", ".join(
        f"{name!r}: "
        + (
            f"unit_specs()[{name!r}]"
            if builtin_units.get(name) == spec
            else (
                f"UnitSpec({spec.kind!r}, scale={float(spec.scale)!r}, "
                f"offset={float(spec.offset)!r}, symbol={spec.symbol!r}, "
                f"aliases={spec.aliases!r})"
            )
        )
        for name, spec in catalogue.units.items()
    )
    definitions = (
        "# Generated catalogue; do not edit.\n"
        "from quantype.catalogue import Catalogue\n"
        "from quantype._internal._registry import QuantitySpec, UnitSpec, unit_specs\n"
        "from quantype._internal._runtime_catalogue import RuntimeCatalogue\n"
        f"runtime = RuntimeCatalogue(Catalogue({dict(catalogue.quantities)!r}, {{{units}}}, {dict(catalogue.relations)!r}, {dict(catalogue.powers)!r}))\n"
    )
    jax_runtime = (
        "from quantype.ujax import grad as grad, hessian as hessian, jit as jit, value_and_grad as value_and_grad, vmap as vmap, _register_quantity\n"
        f"from {package} import _generated\n"
    )
    for name in ("Quantity", *catalogue.quantities):
        jax_runtime += f"_register_quantity(_generated.{name})\n"
    return {
        "_catalogue.py": definitions,
        "__init__.py": f"from {package}._generated import (Quantity as Quantity, "
        + ", ".join(f"{name} as {name}" for name in catalogue.quantities)
        + f")\nfrom {package} import units as u\n",
        "ujax.py": jax_runtime,
        "utorch.py": "from quantype.utorch import grad as grad\n",
        "py.typed": "",
    }

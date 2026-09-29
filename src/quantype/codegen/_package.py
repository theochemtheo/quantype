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
        + (f"unit_specs()[{name!r}]" if builtin_units.get(name) == spec else repr(spec))
        for name, spec in catalogue.units.items()
    )
    definitions = (
        "# Generated catalogue; do not edit.\n"
        "from quantype.catalogue import Catalogue\n"
        "from quantype._internal._registry import QuantitySpec, UnitSpec, unit_specs\n"
        "from quantype._internal._runtime_catalogue import RuntimeCatalogue\n"
        f"runtime = RuntimeCatalogue(Catalogue({dict(catalogue.quantities)!r}, {{{units}}}, {dict(catalogue.relations)!r}, {dict(catalogue.powers)!r}))\n"
    )
    math_runtime = (
        "from typing import Any\nfrom quantype.core import Quantity, _wrap\n"
        "from quantype._internal._math import _unary\n"
        f"from {package}._catalogue import runtime\n"
    )
    math_stub = f"from {package}._generated import Area, Length, Angle, Dimensionless\n"
    for name, source, result in (
        ("sqrt", "Area", "Length"),
        ("sin", "Angle", "Dimensionless"),
        ("exp", "Dimensionless", "Dimensionless"),
    ):
        math_runtime += (
            f"def {name}(quantity: Quantity[Any, Any]) -> Quantity[Any, Any]:\n"
            f"    if quantity._semantic is not runtime.kinds[{source!r}]:\n"
            f"        raise TypeError('Expected {source}')\n"
            f"    return _wrap(runtime.kinds[{result!r}], _unary({name!r}, quantity.value))\n"
        )
        math_stub += f"def {name}[V](quantity: {source}[V]) -> {result}[V]: ...\n"
    jax_runtime = (
        "from quantype.ujax import grad as grad, hessian as hessian, jit as jit, vmap as vmap, _register_quantity\n"
        f"from {package} import _generated\n"
    )
    for name in catalogue.quantities:
        jax_runtime += f"_register_quantity(_generated.{name})\n"
    return {
        "_catalogue.py": definitions,
        "__init__.py": f"from {package}._generated import ("
        + ", ".join(f"{name} as {name}" for name in catalogue.quantities)
        + f")\nfrom {package} import units as u\n",
        "_math.py": math_runtime,
        "_math.pyi": math_stub,
        "ujax.py": jax_runtime,
        "utorch.py": "from quantype.utorch import grad as grad\n",
        "py.typed": "",
    }

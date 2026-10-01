"""Renderers for a catalogue's product classes and naming table.

Each product class gets its own stub module under ``_products/``, so checkers
that load modules lazily load only the classes a file reaches. ``products``
re-exports them; at runtime it holds the naming table and creates each class
on first use.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from quantype.codegen._render import (
    HEADER,
    OVERLAPS,
    PROTOCOL_MEMBERS,
    PROTOCOLS,
    QUANTITY_OVERRIDES,
    binary,
    method_overloads,
    overloads,
    scaling,
    structural_fallback,
)

if TYPE_CHECKING:
    from quantype.catalogue import Catalogue
    from quantype.codegen._table import Pair, Table

# Source templates keep individual generated signatures on one line.
# ruff: noqa: E501

_TYPING = ("Any", "Literal", "overload", "override")
_CORE = ("_Numerical", "_Scalar")


def _entry_signatures(method: str, pair: str, right: str, result: str) -> list[str]:
    """A product class times or over a named kind: storage as for ``binary``.

    The kind's constant is float-like, so it takes the merged overload.
    """
    first, second = binary(method, pair, right, result)
    return [
        first,
        second.replace(
            f"{right}[V, S] | {right}[float, S]",
            f"{right}[V, S] | {right}[float, S] | _{right}Constant",
        ),
    ]


def pair_class(pair: Pair, table: Table) -> str:
    """The stub for one product class."""
    name = pair.name
    text = f"class {name}(Quantity[{pair.marker}, V, S]):\n"
    for operation, method, expression, _, member in PROTOCOLS:
        signatures: list[str] = []
        for (op, left, right), result in table.entries.items():
            if op == operation and left == name:
                signatures += _entry_signatures(method, name, right, result)
        if operation == "div":
            # A product over itself is Dimensionless, as K / K is.
            signatures += binary(method, name, name, "Dimensionless")
        signatures += scaling(method, name, lambda storage: f"{name}[{storage}, S]")
        signatures += structural_fallback(
            method, name, f"Quantity[{expression}[{pair.marker}, K], {{}}, S]"
        )
        text += method_overloads(method, signatures)
        for (op, left, right), result in table.entries.items():
            if op == operation and right == name:
                text += f"    def {member}{left}_f(self, other: {left}[float, S], /) -> {result}[V, S]: ...\n"
                text += overloads(
                    [
                        f"def {member}{left}[W](self: {name}[float, S], other: {left}[W, S], /) -> {result}[W, S]: ...",
                        f"def {member}{left}(self, other: {left}[V, S], /) -> {result}[V, S]: ...",
                    ]
                )
    # `a @ b` multiplies kinds as `a * b` does; scalars have no matrix product.
    signatures = [
        signature
        for (op, left, right), result in table.entries.items()
        if op == "mul" and left == name
        for signature in binary("__matmul__", name, right, result)
    ]
    signatures += scaling("__matmul__", name, lambda storage: f"{name}[{storage}, S]")[
        1:
    ]
    signatures += structural_fallback(
        "__matmul__", name, f"Quantity[Mul[{pair.marker}, K], {{}}, S]"
    )
    text += method_overloads("__matmul__", signatures)
    for method in ("__rmul__", "__rmatmul__"):
        scaled = scaling(method, name, lambda storage: f"{name}[{storage}, S]")
        text += overloads(
            scaled[1:] if method == "__rmatmul__" else scaled, overrides=True
        )
    text += f"    @override\n    def item(self) -> {name}[float, S]: ...\n"
    text += overloads(
        scaling(
            "__rtruediv__",
            name,
            lambda storage: (
                f"Quantity[Div[DimensionlessKind, {pair.marker}], {storage}, S]"
            ),
        ),
        overrides=True,
    )
    # Any quantity of the same structure adds, as for the base class, and the
    # sum keeps this product class.
    structure = f"Quantity[{pair.marker}, {{}}, S]"
    for method in ("__add__", "__sub__"):
        text += overloads(
            [
                f"def {method}[W](self: {name}[float, S], other: {structure.format('W')}, /) -> {name}[W, S]: ...",
                f"def {method}(self, other: {structure.format('V')} | {structure.format('float')}, /) -> {name}[V, S]: ...",
            ],
            overrides=True,
        )
    text += f"    @override\n    def to_system[T: UnitSystem](self, system: type[T]) -> {name}[V, T]: ...\n"
    text += f"    @classmethod\n    @override\n    def from_value[W](cls, value: W) -> {name}[W, S]: ...\n"
    return text


def _pair_module(pair: Pair, catalogue: Catalogue, package: str, table: Table) -> str:
    body = pair_class(pair, table)
    used: set[str] = set(re.findall(r"\b\w+\b", body))
    text = (
        HEADER
        + OVERLAPS
        + QUANTITY_OVERRIDES
        + "# pyright: reportPrivateUsage=false\n"
        + (PROTOCOL_MEMBERS if "    def _r" in body else "")
    )
    text += "from typing import " + ", ".join(n for n in _TYPING if n in used) + "\n"
    text += "import numpy as np\nimport numpy.typing as npt\n"
    text += "from typing_extensions import TypeVar\n"
    text += "from quantype.systems import Atomistic, UnitSystem\n"
    text += (
        "from quantype.core import " + ", ".join(n for n in _CORE if n in used) + "\n"
    )
    text += (
        "from quantype.core import Quantity\n"
        if package == "quantype"
        else f"from {package}._generated import Quantity\n"
    )
    markers = sorted(
        word for word in used if word.endswith("Kind") or word in {"Mul", "Div", "Pow"}
    )
    text += f"from {package}.kinds import " + ", ".join(markers) + "\n"
    named = [kind for kind in catalogue.quantities if kind in used]
    text += f"from {package}._generated import " + ", ".join(named) + "\n"
    for kind in catalogue.quantities:
        if f"_{kind}Constant" in used:
            text += f"from {package}._constants.{kind} import _{kind}Constant\n"
    text += "# Public parameter names read better in diagnostics than private ones.\n"
    text += "# ruff: noqa: PYI001\n"
    text += (
        "V = TypeVar('V')\nS = TypeVar('S', bound=UnitSystem, default=Atomistic)\n\n"
    )
    return text + body


def _runtime(package: str, table: Table) -> str:
    text = HEADER + (
        '"""Classes for unnamed products of two named kinds, such as ``LengthTime``.\n\n'
        "``Length * Time`` and ``Time * Length`` are both a ``LengthTime``, and a\n"
        "product class times or over a named kind is named wherever the\n"
        "catalogue's relations agree on one kind. Annotate with these classes when\n"
        "a function takes or returns such a product.\n"
        '"""\n\n'
    )
    text += "# Product classes resolve lazily through __getattr__.\n"
    text += "# ruff: noqa: F822\n"
    text += "# pyright: reportUnsupportedDunderAll=false\n"
    text += "from typing import Any\n"
    text += "from quantype._internal._products import pair_class as _pair_class\n"
    text += "from quantype._internal._products import register as _register\n"
    if package == "quantype":
        text += "from quantype._internal._semantics import KINDS as _KINDS\n"
        text += "from quantype.core import Quantity as _Quantity\n"
    else:
        text += f"from {package}._catalogue import runtime as _runtime\n"
        text += f"from {package}._generated import Quantity as _Quantity\n"
        text += "_KINDS = _runtime.kinds\n"
    # One row per line, as text: type checkers read a literal of this size far
    # faster than the equivalent tuple of tuples.
    pairs = "".join(
        f"{name} {pair.operation} {pair.left} {pair.right}\n"
        for name, pair in sorted(table.pairs.items())
    )
    entries = "".join(
        f"{operation} {left} {right} {result}\n"
        for (operation, left, right), result in sorted(table.entries.items())
    )
    text += f'# Product classes: name, operation, left, right.\n_PAIRS = """\\\n{pairs}"""\n'
    text += f'# Named results: operation, left, right, result.\n_ENTRIES = """\\\n{entries}"""\n'
    text += "_register(_KINDS, _PAIRS, _ENTRIES, __name__)\n\n"
    text += "def __getattr__(name: str) -> type[_Quantity[Any, Any, Any]]:\n"
    text += "    return _pair_class(__name__, name, _Quantity)\n\n"
    text += "def __dir__() -> list[str]:\n    return __all__\n\n"
    text += f"__all__ = {sorted(table.pairs)!r}\n"
    return text


def _stub(package: str, table: Table) -> str:
    text = HEADER + "# Re-exports each product class from its own module.\n"
    text += "# ruff: noqa: E501\n"
    text += "".join(
        f"from {package}._products.{name} import {name} as {name}\n"
        for name in sorted(table.pairs)
    )
    return text + f"__all__ = {sorted(table.pairs)!r}\n"


def product_outputs(catalogue: Catalogue, package: str, table: Table) -> dict[str, str]:
    outputs = {
        "products.py": _runtime(package, table),
        "products.pyi": _stub(package, table),
        "_products/__init__.pyi": HEADER,
    }
    for name, pair in table.pairs.items():
        outputs[f"_products/{name}.pyi"] = _pair_module(pair, catalogue, package, table)
    return outputs

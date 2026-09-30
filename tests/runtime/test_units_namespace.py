"""Every unit is reachable, by its flat name and through its kind's namespace."""

from __future__ import annotations

from quantype import u
from quantype._internal._registry import unit_specs
from quantype.catalogue import builtin_catalogue
from quantype.codegen._render import unit_names
from quantype.core import get_unit


def test_every_namespaced_unit_is_the_catalogue_unit() -> None:
    catalogue = builtin_catalogue()
    for kind in catalogue.quantities:
        namespace = getattr(u, _namespace(kind))
        for name, spec in unit_specs().items():
            if spec.kind == kind:
                unit = getattr(namespace, name)
                assert unit is get_unit(name), (kind, name)
                assert unit.kind == kind


def test_every_flat_name_is_the_catalogue_unit() -> None:
    namespaces = {_namespace(kind) for kind in builtin_catalogue().quantities}
    for identifier, name in unit_names(builtin_catalogue()).items():
        if identifier not in namespaces:
            assert getattr(u, identifier) is get_unit(name), identifier


def test_units_expose_no_helpers() -> None:
    public = {name for name in dir(u) if not name.startswith("_")}
    assert public <= set(u.__all__)
    for helper in ("Any", "cast", "Unit", "get_unit", "LengthKind", "sqrt"):
        assert helper not in public


def _namespace(kind: str) -> str:
    return "".join(
        ("_" + char.lower()) if char.isupper() else char for char in kind
    ).lstrip("_")

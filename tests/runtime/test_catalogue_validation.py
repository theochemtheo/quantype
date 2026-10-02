"""Application catalogues are validated when defined, with a specific error."""

from __future__ import annotations

from typing import Any

import pytest

from quantype.catalogue import QuantitySpec, UnitSpec, builtin_catalogue

LINE_TENSION = QuantitySpec((-1, 1, 0, 0, 0, 0, 0, 0), "line_tension")
# Deliberately invalid: dimensions must be integer exponents.
HALF: Any = (0.5, 0, 0, 0, 0, 0, 0, 0)


def _extend(**changes: object) -> None:
    arguments: dict[str, Any] = {
        "quantities": {"LineTension": LINE_TENSION},
        "units": {"line_tension": UnitSpec("LineTension")},
    }
    arguments.update(changes)
    builtin_catalogue().extend(**arguments)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"quantities": {"Half": QuantitySpec(HALF, "x")}},
            "integer exponents",
        ),
        ({"units": {}}, "Invalid canonical unit for LineTension"),
        (
            {"units": {"line_tension": UnitSpec("LineTension", scale=2.0)}},
            "Invalid canonical unit",
        ),
        (
            {
                "units": {
                    "line_tension": UnitSpec("LineTension"),
                    "bad unit": UnitSpec("LineTension"),
                }
            },
            "Invalid unit definition 'bad unit'",
        ),
        (
            {
                "units": {
                    "line_tension": UnitSpec("LineTension"),
                    "other": UnitSpec("NoSuchKind"),
                }
            },
            "Invalid unit definition 'other'",
        ),
        (
            {
                "units": {
                    "line_tension": UnitSpec("LineTension"),
                    "negative": UnitSpec("LineTension", scale=-1.0),
                }
            },
            "Invalid conversion for negative",
        ),
        (
            {
                "units": {
                    "line_tension": UnitSpec("LineTension"),
                    "shifted": UnitSpec("LineTension", offset=1.0),
                }
            },
            "Only absolute-temperature units may have offsets",
        ),
        (
            {"units": {"line_tension": UnitSpec("LineTension", aliases=("",))}},
            "Empty or duplicate unit name ''",
        ),
        (
            {"relations": {("mul", "LineTension", "Nothing"): "Energy"}},
            "Unknown quantity in relation",
        ),
        (
            {"relations": {("pow", "LineTension", "Length"): "Energy"}},
            "Invalid relation operation 'pow'",
        ),
        (
            {"relations": {("mul", "LineTension", "Time"): "Energy"}},
            "Dimensionally inconsistent relation",
        ),
        (
            {"powers": {("LineTension", 2.0): "Area"}},
            "invalid exponent in power",
        ),
        ({"powers": {("LineTension", 2): "Area"}}, r"Invalid power LineTension \*\* 2"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_invalid_definitions_are_rejected(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _extend(**changes)


def test_commutative_relations_must_agree() -> None:
    with pytest.raises(ValueError, match="Conflicting commutative relationship"):
        _extend(
            relations={
                ("mul", "LineTension", "Length"): "Energy",
                ("mul", "Length", "LineTension"): "Force",
            }
        )


def test_extensions_cannot_replace_definitions() -> None:
    with pytest.raises(ValueError, match="cannot replace existing definitions"):
        builtin_catalogue().extend(quantities={"Length": LINE_TENSION}, units={})


def test_ambiguous_divisions_are_resolved_by_declaring_them() -> None:
    # LineTension * Length and Force * Length are both Energy, so Energy / Length
    # could be either; declaring it settles which.
    relations = {("mul", "LineTension", "Length"): "Energy"}
    with pytest.raises(ValueError, match="Ambiguous division Energy / Length"):
        _extend(relations=relations)
    catalogue = builtin_catalogue().extend(
        quantities={"LineTension": LINE_TENSION},
        units={"line_tension": UnitSpec("LineTension")},
        relations={**relations, ("div", "Energy", "Length"): "Force"},
    )
    assert catalogue.algebra["div", "Energy", "Length"] == "Force"
    assert catalogue.algebra["div", "Energy", "LineTension"] == "Length"

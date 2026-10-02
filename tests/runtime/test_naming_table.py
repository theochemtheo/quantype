"""The naming table that runtime arithmetic and the generated stubs share."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

import quantype._generated as named
from quantype import products
from quantype.catalogue import QuantitySpec, UnitSpec, builtin_catalogue
from quantype.codegen._table import Operation, Product, Table, naming_table

if TYPE_CHECKING:
    from quantype.core import Quantity


@pytest.fixture(scope="module")
def table() -> Table:
    return naming_table(builtin_catalogue())


@pytest.mark.parametrize(
    ("key", "result"),
    [
        (("mul", "Mass", "VelocitySquared"), "Energy"),
        (("mul", "VelocitySquared", "Mass"), "Energy"),
        (("div", "MomentumSquared", "Mass"), "Energy"),
        (("div", "LengthMass", "Time"), "Momentum"),
        (("div", "Length", "TimeSquared"), "Acceleration"),
        (("div", "LengthTime", "Time"), "Length"),
        (("mul", "Length", "PerLength"), "Dimensionless"),
    ],
)
def test_entries_name_products_of_three_factors(
    table: Table, key: tuple[Product, str, str], result: str
) -> None:
    assert table.entries[key] == result


def test_two_groupings_naming_different_kinds_stay_unnamed(table: Table) -> None:
    # Force * Length / Volume is EnergyDensity one way and Pressure the other.
    assert table.pair("div", "Length", "Volume") is not None
    assert ("mul", "Force", "LengthPerVolume") not in table.entries
    # Force * Length is Energy, so this grouping is named one step at a time.
    assert table.pair("mul", "Force", "Length") is None


@pytest.mark.parametrize(
    ("spelling", "name"),
    [
        (("mul", "Length", "Time"), "LengthTime"),
        (("mul", "Time", "Length"), "LengthTime"),
        (("mul", "Velocity", "Velocity"), "VelocitySquared"),
        (("pow", "Velocity", 2), "VelocitySquared"),
        (("div", "Dimensionless", "Length"), "PerLength"),
        (("pow", "Length", -1), "PerLength"),
        (("div", "Energy", "Time"), "EnergyPerTime"),
    ],
)
def test_every_spelling_finds_one_canonical_pair(
    table: Table, spelling: tuple[Operation, str, str | int], name: str
) -> None:
    pair = table.pair(*spelling)
    assert pair is not None
    assert pair.name == name


def test_named_powers_and_reciprocals_are_not_pairs(table: Table) -> None:
    assert table.pair("pow", "Length", 2) is None  # Area
    assert table.pair("pow", "Time", -1) is None  # InverseTime, as 1 / Time is
    assert table.pair("mul", "Mass", "Velocity") is None  # Momentum


def test_a_pair_name_clashing_with_a_quantity_is_rejected() -> None:
    catalogue = builtin_catalogue().extend(
        quantities={"LengthTime": QuantitySpec((1, 0, 1, 0, 0, 0, 0, 0), "lt")},
        units={"lt": UnitSpec("LengthTime")},
    )
    with pytest.raises(ValueError, match="LengthTime"):
        naming_table(catalogue)


def _operand(name: str) -> Quantity[Any, float, Any]:
    """A value of the named kind or product class."""
    cls = getattr(named, name, None) or getattr(products, name)
    return cast("Quantity[Any, float, Any]", cls.from_value(2.0))


def _apply(operation: str, left: object, right: object) -> object:
    lhs, rhs = cast("Any", left), cast("Any", right)
    if operation == "mul":
        return cast("object", lhs * rhs)
    if operation == "div":
        return cast("object", lhs / rhs)
    return cast("object", lhs**rhs)


def test_runtime_arithmetic_follows_every_entry(table: Table) -> None:
    """Exact agreement: each entry's operands multiply or divide to its result."""
    wrong = [
        (key, result, type(outcome).__name__)
        for key, result in table.entries.items()
        if type(outcome := _apply(key[0], _operand(key[1]), _operand(key[2]))).__name__
        != result
    ]
    assert wrong == []


def test_runtime_arithmetic_finds_the_product_class_of_every_spelling(
    table: Table,
) -> None:
    wrong: list[tuple[tuple[str, str, str | int], str, str]] = []
    for (operation, left, right), name in table.spellings.items():
        operand = right if isinstance(right, int) else _operand(right)
        outcome = _apply(operation, _operand(left), operand)
        if type(outcome).__name__ != name:
            wrong.append(((operation, left, right), name, type(outcome).__name__))
    assert wrong == []


def test_two_products_with_the_same_name_are_rejected() -> None:
    names = ["Ab", "AbCd", "CdEf", "Ef"]
    catalogue = builtin_catalogue().extend(
        quantities={
            name: QuantitySpec((0, 0, 0, 0, 0, index + 1, 0, 0), name.lower())
            for index, name in enumerate(names)
        },
        units={name.lower(): UnitSpec(name) for name in names},
    )
    with pytest.raises(ValueError, match="would both be named AbCdEf"):
        naming_table(catalogue)

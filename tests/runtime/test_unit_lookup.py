"""A kind class finds its units by name, alias, or symbol, as parse does."""

from __future__ import annotations

import subprocess
import sys
from typing import Any, cast

import pytest

import quantype._generated as kinds
from quantype import Energy, Force, Length, Mass, Pressure, Quantity, Temperature, u
from quantype._internal._registry import unit_specs
from quantype.core import get_unit
from quantype.products import LengthTime
from quantype.systems import UnitSystem


def _kind_class(kind: str) -> type[Quantity[Any, Any, Any]]:
    cls: type[Quantity[Any, Any, Any]] = getattr(kinds, kind)
    return cls


@pytest.mark.parametrize("name", list(unit_specs()))
def test_every_catalogue_name_and_alias_resolves(name: str) -> None:
    spec = unit_specs()[name]
    cls = _kind_class(spec.kind)
    unit = cls.unit_named(name)
    for alias in spec.aliases:
        assert cls.unit_named(alias) is unit
    assert unit is get_unit(name)
    assert cls.unit_named(unit.symbol) is unit


def test_documented_aliases_are_the_flat_units() -> None:
    assert Length.unit_named("nanometer") is u.nm
    assert Length.unit_named("nm") is u.nm
    assert Mass.unit_named("dalton") is u.Da
    assert Mass.unit_named("amu") is u.Da
    assert Force.unit_named("hartree_per_bohr") is u.hartree_per_bohr
    assert Force.unit_named("Ha/a0") is u.hartree_per_bohr
    assert Force.unit_named("eV/Å") is u.eV_per_angstrom


def test_a_name_of_another_kind_names_both_kinds() -> None:
    message = r"Expected Force; received Length \(unit 'nm'\)"
    with pytest.raises(ValueError, match=message):
        Force.unit_named("nm")
    with pytest.raises(ValueError, match="Expected Pressure; received EnergyDensity"):
        Pressure.unit_named("energy_density_hartree_per_bohr_cubed")


def test_an_unknown_name_lists_the_kinds_units() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown unit 'furlong' for Length; use one of angstrom, meter",
    ):
        Length.unit_named("furlong")
    with pytest.raises(TypeError, match="Unit names are strings"):
        Length.unit_named(cast("Any", 1))


def test_custom_units_are_found_when_supplied() -> None:
    bleb = Temperature.define_unit("lab:bleb", reference=u.celsius, scale=2.0)
    with pytest.raises(ValueError, match="Unknown unit 'lab:bleb'"):
        Temperature.unit_named("lab:bleb")
    assert Temperature.unit_named("lab:bleb", units=(bleb,)) is bleb
    spark = Energy.define_unit("lab:spark", reference=u.eV, scale=2.0, symbol="sp")
    assert Energy.unit_named("sp", units=(spark,)) is spark
    with pytest.raises(ValueError, match="Expected Length; received Energy"):
        Length.unit_named("lab:spark", units=(spark,))
    shadow = Length.define_unit("angstrom", reference=u.angstrom, scale=2.0)
    with pytest.raises(ValueError, match="shadows builtin"):
        Length.unit_named("angstrom", units=(shadow,))


def test_names_take_precedence_over_symbols() -> None:
    odd = Length.define_unit("lab:odd", reference=u.nm, symbol="angstrom")
    assert Length.unit_named("angstrom", units=(odd,)) is u.angstrom


def test_a_symbol_of_the_kind_beats_a_name_of_another_kind() -> None:
    volt_like = Length.define_unit("lab:v", reference=u.nm, symbol="eV")
    assert Length.unit_named("eV", units=(volt_like,)) is volt_like


def test_a_symbol_shared_within_a_kind_is_ambiguous() -> None:
    first = Length.define_unit("lab:first", reference=u.nm, symbol="lu")
    second = Length.define_unit("lab:second", reference=u.nm, scale=2.0, symbol="lu")
    with pytest.raises(ValueError, match="'lu' is ambiguous; use one of lab:first"):
        Length.unit_named("lu", units=(first, second))


class _Gromacs(UnitSystem, name="test-unit-lookup:gromacs"):
    length = u.nanometer
    energy = u.kJ_per_mol
    time = u.picosecond


def test_an_alias_naming_a_system_finds_its_units() -> None:
    unit = _Gromacs.unit_for(Force)
    assert Force[float, _Gromacs].unit_named(unit.name) is unit
    assert Force[float, _Gromacs].unit_named(unit.symbol) is unit
    assert Force[float].unit_named("newton") is u.newton
    with pytest.raises(ValueError, match="Unknown unit"):
        Force.unit_named(unit.name)


def test_unnamed_products_have_no_catalogue_units() -> None:
    with pytest.raises(ValueError, match=r"Unknown unit 'x' for Mul\[Length,Time\]$"):
        LengthTime.unit_named("x")
    with pytest.raises(TypeError, match="named quantity class"):
        Quantity.unit_named("nm")


def test_lookup_does_not_import_numpy() -> None:
    code = (
        "import sys; from quantype import Force; "
        "Force.unit_named('Ha/a0'); Force.unit_named('newton'); "
        "assert 'numpy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603

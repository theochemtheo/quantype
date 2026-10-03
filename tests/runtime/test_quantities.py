"""The runtime model: display rules, exact echo, comparisons, and indexing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

import quantype
import quantype.numpy as qnp
from quantype import (
    Area,
    Dimensionless,
    Energy,
    Force,
    Length,
    Quantity,
    Temperature,
    TemperatureDifference,
    Time,
    u,
)
from quantype.catalogue import builtin_catalogue
from quantype.systems import SI, Atomistic, Metal

if TYPE_CHECKING:
    from quantype.core import Unit
    from quantype.kinds import ForceKind


def _dynamic(cls: object) -> Any:  # noqa: ANN401 -- deliberately invalid calls
    """Hide deliberately invalid parameters from the type checkers."""
    return cls


def test_a_kind_class_names_its_kind() -> None:
    for name in builtin_catalogue().quantities:
        cls: Any = getattr(quantype, name)
        assert cls.kind == name
        assert cls[float].kind == name
        assert cls.from_value(1.0).kind == name
    product = Length[float](2, u.nm) * Time[float](1, u.fs)
    assert type(product).kind == product.kind == "Mul[Length,Time]"
    with pytest.raises(AttributeError, match="named quantity classes"):
        getattr(Quantity, "kind")  # noqa: B009 -- the access is the test
    with pytest.raises(AttributeError, match="kind is fixed"):
        _dynamic(product).kind = "Energy"


def test_construction_remembers_the_input_unit() -> None:
    typed = Length[float, SI](2, u.nm)
    assert typed.value == pytest.approx(2e-9)
    assert typed.magnitude() == 2
    assert str(typed) == "2.0 nm"
    assert repr(2 * u.nm) == "Length(2.0 nm)"
    bare: Length[float] = Length(2, u.nm)
    assert repr(bare) == "Length(2.0 nm)"
    assert bare.system is Atomistic
    assert repr(Length.parse("3 bohr")) == "Length(3.0 a0)"
    assert repr(Length[float, SI].from_value(2.0)) == "Length(2.0 m, SI)"
    assert repr(Length.from_value(2.0)) == "Length(2.0 Å)"


@pytest.mark.parametrize(
    ("unit", "storage", "raw"),
    [
        (
            u.hartree_per_bohr,
            npt.NDArray[np.float32],
            np.array([[1.0, -2.0, 3.0]], dtype=np.float32),
        ),
        (u.eV_per_angstrom, npt.NDArray[np.float64], np.array([0.5, 1.5])),
    ],
)
def test_construction_can_leave_the_display_unit_unset(
    unit: Unit[ForceKind], storage: object, raw: npt.NDArray[np.floating[Any]]
) -> None:
    alias = _dynamic(Force)[storage, SI]
    shown = alias(raw, unit)
    stored = alias(raw, unit, display=False)
    assert stored.value.dtype == raw.dtype
    np.testing.assert_array_equal(stored.value, shown.value)
    assert stored.unit is SI.unit_for(Force)
    assert str(stored) == str(alias.from_value(shown.value))
    assert stored.to_dict()["unit"] == "newton"
    assert shown.to_dict()["unit"] == unit.name


def test_construction_without_a_display_unit_converts_offsets() -> None:
    point = Temperature[float](20.1, u.celsius, display=False)
    assert point.value == Temperature[float](20.1, u.celsius).value
    assert point.unit is u.kelvin
    assert point.magnitude() == point.value
    assert repr(point) == f"Temperature({point.value} K)"
    bare = Length(2.0, u.nm, display=False)
    assert bare.value == 20.0
    assert repr(bare) == "Length(20.0 Å)"
    assert repr(Length[float](2, u.nm, display=True)) == "Length(2.0 nm)"


def test_same_kind_operations_keep_the_display_unit() -> None:
    q = Length[npt.NDArray[np.float64], SI]([1.0, -2.0, 3.0], u.nm)
    for result in (
        q * 2,
        2 * q,
        q / 2,
        -q,
        abs(q),
        q.sum(),
        q.mean(),
        q.max(),
        q.min(),
        q[1:],
        q[0],
    ):
        assert repr(result).startswith("Length(")
        assert repr(result).endswith(" nm, SI)")
    assert q.max().magnitude() == pytest.approx(3.0)
    assert q.min().magnitude() == pytest.approx(-2.0)


def test_addition_prefers_the_left_display_unit() -> None:
    nm, angstrom = 2 * u.nm, 5 * u.angstrom
    assert repr(nm + angstrom) == "Length(2.5 nm)"
    assert repr(angstrom + nm) == "Length(25.0 Å)"
    assert repr(Length.from_value(5.0) + nm) == "Length(2.5 nm)"
    assert repr(nm - angstrom) == "Length(1.5 nm)"


def test_products_ratios_and_powers_fall_back_to_the_system() -> None:
    x = Length[float, SI](2, u.nm)
    assert repr(x * x) == "Area(4e-18 m^2, SI)"
    assert repr(x**2) == "Area(4e-18 m^2, SI)"
    assert repr(Energy[float, SI](3, u.eV) / x).startswith("Force(")
    assert repr((2 * u.nm) * (3 * u.nm)) == "Area(600.0 Å^2)"
    structural = (2 * u.angstrom) * (3 * u.fs)
    assert repr(structural) == "LengthTime(6.0 (Å * fs))"
    assert "Mul" not in repr(structural)


def test_temperature_points_and_differences() -> None:
    warm, cold = 30 * u.celsius, 20 * u.celsius
    assert repr(warm - cold) == "TemperatureDifference(10.0 Δ°C)"
    assert repr(300 * u.K - cold) == "TemperatureDifference(6.850000000000023 ΔK)"
    shifted = cold + TemperatureDifference[float](5, u.delta_kelvin)
    assert repr(shifted) == "Temperature(25.0 °C)"
    custom = Temperature.define_unit("lab:bleb", reference=u.celsius, scale=2.0)
    difference = Temperature[float](3, custom) - Temperature[float](1, custom)
    assert difference.magnitude(u.delta_kelvin) == pytest.approx(4.0)
    assert repr(difference) == "TemperatureDifference(4.0 ΔK)"


def test_exact_echo_of_python_scalars() -> None:
    point = Temperature[float](20.1, u.celsius)
    assert point.value == pytest.approx(293.25)
    assert point.magnitude() == 20.1
    assert point.magnitude(u.celsius) == 20.1
    assert point.to_dict()["magnitude"] == 20.1
    assert (20.1 * u.celsius).magnitude() == 20.1
    assert Temperature.parse("20.1 degC").to_dict()["magnitude"] == 20.1
    # The echo is the input, so it survives only presentation-preserving steps.
    assert point.to_system(SI).magnitude() == 20.1
    assert point.mean().magnitude() == pytest.approx(20.1)
    assert point.mean().magnitude() != 20.1
    assert point.to(u.kelvin).magnitude() == pytest.approx(293.25)
    array = Temperature[npt.NDArray[np.float64]]([20.1], u.celsius)
    assert array.magnitude()[0] == pytest.approx(20.1)
    assert Temperature[np.float64](20.1, u.celsius).magnitude() != 20.1


def test_storage_alias_names_storage_then_system() -> None:
    swapped: Any = _dynamic(Length)[SI, float]
    with pytest.raises(TypeError, match=r"storage type first.*Length\[float, SI\]"):
        swapped(1, u.nm)
    bad: Any = _dynamic(Length)[float, int]
    with pytest.raises(TypeError, match="unit system class"):
        bad(1, u.nm)
    abstract: Any = Length[float, Any]
    with pytest.raises(TypeError, match="unit system class"):
        abstract.from_value(1.0)
    with pytest.raises(TypeError, match="from_value"):
        _dynamic(Length)[float, SI](1)
    parsed = Length[float, Metal].parse("1 nm")
    assert parsed.system is Metal
    assert parsed.value == pytest.approx(10.0)


def test_value_equality() -> None:
    a, b = Length[float](1, u.nm), Length[float](10, u.angstrom)
    assert a == b
    assert a != 2 * u.nm
    assert a != 1 * u.eV
    with pytest.raises(TypeError, match="give it a unit"):
        _ = a != 1.0
    arrays = Length[npt.NDArray[np.float64]]([1.0, 2.0], u.nm)
    np.testing.assert_array_equal(arrays == arrays.to(u.angstrom), [True, True])
    np.testing.assert_array_equal(arrays != 1 * u.nm, [False, True])
    with pytest.raises(TypeError, match="unhashable"):
        hash(arrays)
    # Like naive and aware datetimes: unequal across systems, unordered.
    mixed: Any = a.to_system(SI)
    assert a != mixed
    assert (a == mixed) is False
    assert a not in [mixed, 2 * u.nm]
    with pytest.raises(TypeError, match="Cannot combine Atomistic and SI"):
        _ = a < mixed


def test_scalar_quantities_hash_by_value() -> None:
    a, b = Length[float](1, u.nm), Length[float](10, u.angstrom)
    assert hash(a) == hash(b)
    assert hash(a) == hash(Length[np.float64](1, u.nm))
    assert hash(-(0 * u.nm)) == hash(0 * u.nm)
    assert len({a, b, 2 * u.nm}) == 2
    assert {a: "cutoff"}[b] == "cutoff"
    structural = (2 * u.angstrom) * (3 * u.fs)
    assert hash(structural) == hash((6 * u.angstrom) * (1 * u.fs))
    for unhashable in (
        Length[npt.NDArray[np.float64]]([1.0], u.nm),
        Length[npt.NDArray[np.float64]](1.0, u.nm),
    ):
        with pytest.raises(TypeError, match="unhashable array storage"):
            hash(unhashable)


def test_ordering_comparisons() -> None:
    positions = Length[npt.NDArray[np.float64]]([1.0, 2.0, 3.0], u.angstrom)
    cutoff = Length[float](0.25, u.nm)
    np.testing.assert_array_equal(positions < cutoff, [True, True, False])
    np.testing.assert_array_equal(positions >= cutoff, [False, False, True])
    assert 1 * u.nm <= 10 * u.angstrom
    assert 20 * u.celsius < 300 * u.K
    wrong: Any = positions
    with pytest.raises(TypeError, match="Cannot compare Length and Energy"):
        _ = wrong < 1 * u.eV
    with pytest.raises(TypeError, match="Cannot compare Length and float"):
        _ = wrong < 1.0


def test_indexing_shape_and_length() -> None:
    positions = Length[npt.NDArray[np.float64]]([[1.0, 2.0], [3.0, 4.0]], u.nm)
    assert positions.shape == (2, 2)
    assert positions.ndim == 2
    assert len(positions) == 2
    row = positions[1]
    assert isinstance(row, Length)
    np.testing.assert_array_equal(row.magnitude(), [3.0, 4.0])
    element = positions[1, 0]
    assert isinstance(element.value, np.ndarray)
    assert element.shape == ()
    assert [float(q.magnitude()) for q in positions[0]] == [1.0, 2.0]
    scalar = 1 * u.nm
    assert scalar.shape == ()
    assert scalar.ndim == 0
    untyped: Any = scalar
    with pytest.raises(TypeError, match="cannot be indexed"):
        _ = untyped[0]
    with pytest.raises(TypeError, match="no length"):
        len(untyped)


def test_reductions_along_an_axis() -> None:
    q = Length[npt.NDArray[np.float64]]([[1.0, 5.0], [3.0, 2.0]], u.angstrom)
    np.testing.assert_array_equal(q.max(axis=0).value, [3.0, 5.0])
    np.testing.assert_array_equal(q.min(axis=1, keepdims=True).value, [[1.0], [2.0]])
    points = Temperature[npt.NDArray[np.float64]]([300.0, 310.0], u.K)
    assert points.max().value == 310
    assert points.min().value == 300


def test_reflected_numpy_scalars() -> None:
    q = Length[float](1, u.nm)
    for result in (np.float64(2) * q, q * np.float64(2)):
        assert isinstance(result, Length)
        assert result.magnitude() == 2
    inverse = np.float64(2) / (2 * u.fs)
    assert inverse.kind == "InverseTime"
    assert inverse.value == 1.0
    untyped: Any = q
    with pytest.raises(TypeError, match="Cannot add float64 and Length"):
        np.add(np.float64(1), untyped)


def test_math_helpers_keep_the_system() -> None:
    area = Area[float, SI](4, u.square_meter)
    assert qnp.sqrt(area).system is SI
    assert qnp.sqrt(area).value == pytest.approx(2.0)
    assert qnp.exp(Dimensionless[float, SI](0, u.one)).system is SI


def test_derived_kinds_of_one_system_multiply_coherently() -> None:
    force = Force[float, SI](1, u.newton)
    energy = force * Length[float, SI](1, u.meter)
    assert type(energy) is Energy
    assert energy.value == pytest.approx(1.0)
    assert energy.magnitude(u.joule) == pytest.approx(1.0)
    ratio = energy / energy
    assert type(ratio) is Dimensionless
    assert isinstance(ratio, Quantity)

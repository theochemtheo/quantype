"""quantype.testing, and apply for operations quantype has no rule for."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from quantype import Energy, Length, Temperature, u
from quantype.products import LengthTime
from quantype.systems import SI, Atomistic
from quantype.testing import assert_allclose

Array = npt.NDArray[np.float64]


def test_close_quantities_pass_whatever_their_display_units() -> None:
    assert_allclose(u.nm(np.array([1.0, 2.0])), u.angstrom(np.array([10.0, 20.0])))
    assert_allclose(2 * u.nm, 20.0000001 * u.angstrom, rtol=1e-6)
    assert_allclose(300 * u.K, 300.05 * u.K, atol=0.1 * u.delta_K)
    assert_allclose(300 * u.K, 300.05 * u.K, atol=0.1 * u.K)
    assert_allclose((2 * u.nm) * (3 * u.fs), (3 * u.fs) * (2 * u.nm))


def test_differences_name_the_kind_system_and_unit() -> None:
    with pytest.raises(
        AssertionError, match=r"(?s)compared in nm\n.*ACTUAL: array\(2\.\)"
    ):
        assert_allclose(2 * u.nm, 3 * u.nm)
    with pytest.raises(AssertionError, match="Length in NoSystem, compared in Å"):
        assert_allclose(2 * u.nm, 3 * u.angstrom)
    with pytest.raises(AssertionError, match="Length in SI, compared in m"):
        assert_allclose(Length[float, SI](2, u.nm), Length[float, SI](3, u.nm))
    with pytest.raises(AssertionError, match="Expected Energy; received Length"):
        assert_allclose(cast("Any", 2 * u.nm), 2 * u.eV)
    with pytest.raises(AssertionError, match="Expected a quantity in SI"):
        assert_allclose(cast("Any", 2 * u.nm), Length[float, SI](2, u.nm))


def test_plain_numbers_and_mismatched_tolerances_are_type_errors() -> None:
    with pytest.raises(TypeError, match=r"Compare plain numbers with numpy\.testing"):
        assert_allclose(cast("Any", 2.0), 2 * u.nm)
    with pytest.raises(TypeError, match="atol must be a Length in NoSystem"):
        assert_allclose(2 * u.nm, 2 * u.nm, atol=cast("Any", 1 * u.eV))


def test_apply_keeps_kind_system_storage_and_display_unit() -> None:
    lengths = Length[Array](np.array([3.0, 1.0, 2.0]), u.nm).to(u.angstrom)
    ordered = lengths.apply(np.sort)
    assert isinstance(ordered, Length)
    assert ordered.unit is u.angstrom
    np.testing.assert_array_equal(ordered.magnitude(u.nm), [1, 2, 3])
    bound = Length[Array, SI](np.array([3.0, 1.0]), u.nm).apply(np.sort)
    assert bound.system is SI
    np.testing.assert_allclose(bound.value, [1e-9, 3e-9])
    product = (2 * u.nm) * (3 * u.fs)
    assert type(product.apply(lambda raw: raw * 2)) is LengthTime
    with pytest.raises(TypeError, match="returns numbers, not a quantity"):
        lengths.apply(lambda _: cast("Any", lengths))
    with pytest.raises(TypeError, match="numerical array; received str"):
        lengths.apply(cast("Any", str))
    with pytest.raises(TypeError, match="received list"):
        lengths.apply(cast("Any", list))
    with pytest.raises(TypeError, match="keep numpy storage; received python"):
        lengths.apply(lambda raw: cast("Any", float(raw[0])))
    assert (2 * u.nm).apply(np.abs).magnitude() == 2.0


def test_temperature_kinds_round_trip_through_apply() -> None:
    warm = Temperature[float, Atomistic](20, u.celsius)
    assert warm.apply(lambda raw: raw + 1).magnitude() == pytest.approx(21)
    assert isinstance(Energy[float, Atomistic](1, u.eV).apply(abs), Energy)

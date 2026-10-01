"""quantype.testing, and with_value for operations quantype has no rule for."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from quantype import Energy, Length, Temperature, u
from quantype.products import LengthTime
from quantype.systems import SI
from quantype.testing import assert_allclose

Array = npt.NDArray[np.float64]


def test_close_quantities_pass_whatever_their_display_units() -> None:
    assert_allclose(u.nm(np.array([1.0, 2.0])), u.angstrom(np.array([10.0, 20.0])))
    assert_allclose(2 * u.nm, 20.0000001 * u.angstrom, rtol=1e-6)
    assert_allclose(300 * u.K, 300.05 * u.K, atol=0.1 * u.delta_K)
    assert_allclose(300 * u.K, 300.05 * u.K, atol=0.1 * u.K)
    assert_allclose((2 * u.nm) * (3 * u.fs), (3 * u.fs) * (2 * u.nm))


def test_differences_name_the_kind_system_and_unit() -> None:
    with pytest.raises(AssertionError, match="Length in Atomistic, compared in Å"):
        assert_allclose(2 * u.nm, 3 * u.nm)
    with pytest.raises(AssertionError, match="Expected Energy; received Length"):
        assert_allclose(cast("Any", 2 * u.nm), 2 * u.eV)
    with pytest.raises(AssertionError, match="Expected a quantity in SI"):
        assert_allclose(cast("Any", 2 * u.nm), Length[float, SI](2, u.nm))


def test_plain_numbers_and_mismatched_tolerances_are_type_errors() -> None:
    with pytest.raises(TypeError, match=r"Compare plain numbers with numpy\.testing"):
        assert_allclose(cast("Any", 2.0), 2 * u.nm)
    with pytest.raises(TypeError, match="atol must be a Length in Atomistic"):
        assert_allclose(2 * u.nm, 2 * u.nm, atol=cast("Any", 1 * u.eV))


def test_with_value_keeps_kind_system_and_display_unit() -> None:
    lengths = Length[Array, SI](np.array([3.0, 1.0, 2.0]), u.nm)
    ordered = lengths.with_value(np.sort(lengths.value))
    assert isinstance(ordered, Length)
    assert ordered.system is SI
    assert ordered.unit is u.nm
    np.testing.assert_allclose(ordered.magnitude(), [1, 2, 3])
    product = (2 * u.nm) * (3 * u.fs)
    assert type(product.with_value(1.0)) is LengthTime
    with pytest.raises(TypeError, match="with_value takes raw numbers"):
        lengths.with_value(cast("Any", lengths))


def test_temperature_kinds_round_trip_through_with_value() -> None:
    warm = Temperature[float](20, u.celsius)
    assert warm.with_value(warm.value + 1).magnitude() == pytest.approx(21)
    assert isinstance(Energy[float](1, u.eV).with_value(2.0), Energy)

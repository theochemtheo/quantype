"""Unit conversions come from a vendored CODATA table, one edition per process."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import fields
from typing import TYPE_CHECKING, assert_type, cast

import numpy as np
import numpy.typing as npt
import pytest

from quantype import Length, codata, u
from quantype._internal._registry import unit_specs
from quantype.serialization import load_npz, save_npz

if TYPE_CHECKING:
    from pathlib import Path


def _run(code: str, **environment: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key != codata.ENVIRONMENT_VARIABLE
    }
    return subprocess.run(  # noqa: S603 -- a fixed interpreter and test source
        [sys.executable, "-c", code],
        env={**env, **environment},
        capture_output=True,
        text=True,
        check=False,
    )


def test_editions_are_typed() -> None:
    assert codata.EDITIONS == ("2014", "2018", "2022")
    assert codata.DEFAULT == "2022"
    values = codata.values("2018")
    assert_type(values, codata.Codata)
    assert_type(values.edition, codata.Edition)
    assert values.edition == "2018"
    assert values.boltzmann_constant == 1.380649e-23


def test_units_use_the_process_edition() -> None:
    values = codata.values()
    assert values.edition == codata.edition() == codata.DEFAULT
    assert u.bohr.scale == values.bohr_radius / 1e-10
    assert u.hartree.scale == values.hartree_energy_in_ev
    assert u.joule.scale == 1 / values.elementary_charge
    assert u.celsius.offset == 273.15


def test_editions_differ_where_codata_changed() -> None:
    bohr = {edition: unit_specs(edition)["bohr"].scale for edition in codata.EDITIONS}
    assert len(set(bohr.values())) == len(bohr)
    # Exact since the 2019 SI redefinition, so 2018 and 2022 agree.
    joule = {edition: unit_specs(edition)["joule"].scale for edition in codata.EDITIONS}
    assert joule["2018"] == joule["2022"] != joule["2014"]


@pytest.mark.parametrize("requested", ["2010", "latest", ""])
def test_unknown_edition_is_rejected(requested: str) -> None:
    assert not codata.is_edition(requested)
    # Statically rejected; this checks the runtime guard for untyped callers.
    edition = cast("codata.Edition", requested)
    with pytest.raises(ValueError, match="Unknown CODATA edition"):
        codata.values(edition)


def test_every_edition_has_every_value() -> None:
    for edition in codata.EDITIONS:
        values = codata.values(edition)
        for field in fields(values):
            value = getattr(values, field.name)
            assert field.name == "edition" or (math.isfinite(value) and value > 0)


def test_exact_values_are_computed_not_truncated() -> None:
    # NIST prints the pre-2019 electric constant as 8.854 187 817... e-12.
    values = codata.values("2014")
    c = values.speed_of_light_in_vacuum
    assert values.vacuum_electric_permittivity == 1 / (4e-7 * math.pi * c**2)


def test_default_edition_matches_scipy_when_scipy_carries_it() -> None:
    constants = pytest.importorskip("scipy.constants")
    scipy_codata = pytest.importorskip("scipy.constants._codata")
    if getattr(scipy_codata, "_current_codata", None) != f"CODATA {codata.DEFAULT}":
        pytest.skip("the installed SciPy uses another CODATA edition")
    values = codata.values(codata.DEFAULT)
    physical = constants.physical_constants
    assert values.bohr_radius == physical["Bohr radius"][0]
    assert values.hartree_energy_in_ev == physical["Hartree energy in eV"][0]
    assert values.bohr_magneton == physical["Bohr magneton"][0]


def test_environment_chooses_the_edition() -> None:
    code = (
        "from quantype import codata, u\n"
        "assert codata.edition() == '2014'\n"
        "assert u.bohr.scale == codata.values('2014').bohr_radius / 1e-10\n"
    )
    result = _run(code, QUANTYPE_CODATA="2014")
    assert result.returncode == 0, result.stderr


def test_environment_rejects_unknown_editions() -> None:
    result = _run("from quantype import u; u.bohr.scale", QUANTYPE_CODATA="2010")
    assert result.returncode != 0
    assert "Unknown CODATA edition '2010' from QUANTYPE_CODATA" in result.stderr


def test_use_chooses_the_edition_before_first_unit_use() -> None:
    code = (
        "from quantype import codata, u\n"
        "codata.use('2018')\n"
        "assert u.bohr.scale == codata.values('2018').bohr_radius / 1e-10\n"
        "codata.use('2018')  # repeating the same choice is harmless\n"
        "try:\n"
        "    codata.use('2022')\n"
        "except RuntimeError as error:\n"
        "    assert 'already fixed at 2018' in str(error)\n"
        "else:\n"
        "    raise AssertionError('the edition changed after use')\n"
    )
    result = _run(code, QUANTYPE_CODATA="2014")
    assert result.returncode == 0, result.stderr


def test_edition_cannot_change_after_units_are_used() -> None:
    code = "from quantype import codata, u\nu.bohr.scale\ncodata.use('2014')\n"
    result = _run(code)
    assert result.returncode != 0
    assert "already fixed at 2022" in result.stderr


def test_conversions_never_import_scipy() -> None:
    code = (
        "import sys; from quantype import u; u.bohr.scale; "
        "assert 'scipy' not in sys.modules"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_archives_record_the_edition(tmp_path: Path) -> None:
    path = tmp_path / "frame.npz"
    save_npz(path, cutoff=Length[npt.NDArray[np.float64]]([1.0], u.bohr))
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    assert metadata["codata"] == codata.edition()
    restored = load_npz(path, "cutoff", Length[npt.NDArray[np.float64]])
    np.testing.assert_array_equal(restored.magnitude(u.bohr), [1.0])


@pytest.mark.parametrize(("recorded", "loads"), [(None, True), (2022, False)])
def test_archive_edition_is_optional_provenance(
    tmp_path: Path, recorded: object, *, loads: bool
) -> None:
    metadata: dict[str, object] = {
        "version": 1,
        "quantities": {"x": {"kind": "Length", "unit": "angstrom", "array": "a"}},
    }
    if recorded is not None:
        metadata["codata"] = recorded
    path = tmp_path / "frame.npz"
    np.savez(path, a=np.ones(2), metadata=np.asarray(json.dumps(metadata)))
    if loads:
        assert load_npz(path, "x", Length[npt.NDArray[np.float64]]).shape == (2,)
    else:
        with pytest.raises(ValueError, match="metadata fields"):
            load_npz(path, "x", Length[npt.NDArray[np.float64]])

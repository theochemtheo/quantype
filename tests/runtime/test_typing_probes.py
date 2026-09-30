"""The positive typing probes also run, so the stubs cannot drift from runtime."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import numpy as np
import pytest

from quantype import Length, u
from quantype.serialization import save_npz

PROBES = sorted((Path(__file__).parents[1] / "typing" / "positive").glob("*.py"))


@pytest.mark.parametrize("probe", PROBES, ids=lambda path: path.stem)
def test_positive_probe_runs(
    probe: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for backend in ("jax", "torch"):
        if importlib.util.find_spec(backend) is None:
            pytest.skip(f"{backend} is not installed")
    # construction.py restores this archive.
    monkeypatch.chdir(tmp_path)
    save_npz("positions.npz", positions=u.angstrom(np.zeros(3)))
    assert Length.from_value(0.0).value == 0
    runpy.run_path(str(probe), run_name="probe")

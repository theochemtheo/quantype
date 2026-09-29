"""Cold-process startup and steady-state conversion measurements (no thresholds)."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import timeit


def main() -> None:
    imports = {
        "empty": ("pass", "pass"),
        "numpy": ("pass", "import numpy"),
        "scipy.constants": ("pass", "from scipy import constants"),
        "quantype": ("pass", "import quantype"),
        "first_unit": ("from quantype import u", "u.hartree(2)"),
        "first_typed_constructor": (
            "from quantype import Length, u; from numpy import float64",
            "Length[float64](2, u.length.nanometer)",
        ),
    }
    results: dict[str, object] = {"python": sys.version}
    for label, (setup, statement) in imports.items():
        samples = []
        for _ in range(7):
            command = (
                f"import time; {setup}; t=time.perf_counter(); "
                f"{statement}; print(time.perf_counter()-t)"
            )
            run = subprocess.run(  # noqa: S603
                [sys.executable, "-c", command],
                capture_output=True,
                text=True,
                check=True,
            )
            samples.append(float(run.stdout))
        results[f"cold_seconds/{label}"] = statistics.median(samples)
    setup = (
        "from scipy import constants as c; "
        "scale=c.physical_constants['Hartree energy in eV'][0]; "
        "from quantype import Length,u; import numpy as np; a=np.ones(1000)"
    )
    for label, statement in {
        "cached_factor": "2.0 * scale",
        "scipy_lookup": "2.0 * c.physical_constants['Hartree energy in eV'][0]",
        "scalar_quantity": "u.hartree(2.0)",
        "array_quantity_1000": "u.hartree(a)",
        "typed_scalar": "Length[np.float64](2, u.length.nanometer)",
        "trusted_boundary": "Length.from_canonical(a)",
    }.items():
        results[f"seconds_per_call/{label}"] = (
            min(timeit.repeat(statement, setup, number=10000, repeat=5)) / 10000
        )
    print(json.dumps(results, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()

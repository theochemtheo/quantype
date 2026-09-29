"""Each marked line must produce a diagnostic in every supported checker."""

from quantype import Energy, EnergyPerAtom, Pressure
from quantype import units as u


def energy_only(value: Energy[float]) -> None:
    pass


def pressure_only(value: Pressure[float]) -> None:
    pass


def per_atom_only(value: EnergyPerAtom[float]) -> None:
    pass


energy_only(1.0 * u.angstrom)  # error
bad_addition = (1.0 * u.eV) + (1.0 * u.angstrom)  # error
bad_conversion = (1.0 * u.angstrom).to(u.eV)  # error
pressure_only((1.0 * u.eV) / ((1.0 * u.angstrom) ** 3))  # error
per_atom_only(1.0 * u.eV)  # error
bad_temperature = (300 * u.K) + (280 * u.K)  # error
bad_sine = u.sin(2.0 * u.angstrom)  # error
bad_exponential = u.exp(1.0 * u.eV)  # error
bad_root = u.sqrt(1.0 * u.eV)  # error

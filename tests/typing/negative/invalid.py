"""Each marked line must produce a diagnostic in every supported checker."""

from quantype import Energy, EnergyPerAtom, Length, Pressure, Temperature
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
bad_constructor = Length[float](2, u.energy.hartree)  # error
bad_definition = Temperature.define_unit("bleb", reference=u.nm)  # error
missing_unit = Length[float](2)  # error

point = 300 * u.K
bad_point_scale = point * 2  # error
bad_point_reverse_scale = 2 * point  # error
bad_point_division = point / 2  # error
bad_point_ratio = point / point  # error
bad_point_inverse = 1.0 / point  # error
bad_point_power = point**2  # error
bad_point_sum = point.sum()  # error
bad_point_negation = -point  # error
bad_point_absolute = abs(point)  # error
bad_point_left_product = point * (2 * u.angstrom)  # error
bad_point_right_product = (2 * u.angstrom) * point  # error
bad_point_right_ratio = (2 * u.angstrom) / point  # error

product = (2 * u.angstrom) * (3 * u.fs)
ratio = (2 * u.eV) / (3 * u.fs)
bad_structural_sum = product + ratio  # error
bad_structural_nominal_sum = product + ((3 * u.fs) * (2 * u.angstrom))  # error
bad_structural_point_product = product * point  # error
bad_structural_point_ratio = product / point  # error

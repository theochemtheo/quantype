"""Each marked line must produce a diagnostic in every supported checker."""

from quantype import Energy, EnergyPerAtom, Force, Length, Pressure, Temperature
from quantype import units as u
from quantype.systems import SI, Metal, UnitSystem


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
bad_point_sum = point.sum()  # error

product = (2 * u.angstrom) * (3 * u.fs)
ratio = (2 * u.eV) / (3 * u.fs)
bad_structural_sum = product + ratio  # error
bad_structural_nominal_sum = product + ((3 * u.fs) * (2 * u.angstrom))  # error

# Unit systems never mix; .to_system(...) is the explicit bridge.
x_si = Length[float, SI](2.0, u.nm)
x_metal = Length[float, Metal](20.0, u.angstrom)
e_si = Energy[float, SI](3.0, u.eV)
mixed_addition = x_si + x_metal  # error
mixed_ratio = e_si / x_metal  # error
mixed_comparison = x_si < x_metal  # error
energy_only(e_si)  # error
default_length: Length[float] = x_si  # error
not_a_system = Length[float, int]  # error
swapped_parameters = Length[SI, float]  # error
converted = x_si.to_system(Metal)
converted_is_metal = x_si + converted  # error


def work[S: UnitSystem](f: Force[float, S], d: Length[float, S]) -> Energy[float, S]:
    return f * d


mixed_generic = work(e_si / x_si, x_metal)  # error

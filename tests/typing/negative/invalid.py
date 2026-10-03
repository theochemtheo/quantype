"""Each marked line must produce a diagnostic in every supported checker."""

import quantype.numpy as qnp
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
bad_sine = qnp.sin(2.0 * u.angstrom)  # error
bad_exponential = qnp.exp(1.0 * u.eV)  # error
bad_root = qnp.sqrt(1.0 * u.eV)  # error
bad_constructor = Length[float](2, u.energy.hartree)  # error
bad_definition = Temperature.define_unit("bleb", reference=u.nm)  # error
missing_unit = Length[float](2)  # error
wrong_named_unit = Length[float](2, Force.unit_named("newton"))  # error
unit_name_not_string = Length.unit_named(u.nm)  # error
Length[float](2, u.nm).kind = "Energy"  # error

point = 300 * u.K
bad_point_sum = point.sum()  # error
bad_builtin_point_sum = sum([point, point])  # error
bad_point_from_zero = 0 - point  # error

product = (2 * u.angstrom) * (3 * u.fs)
ratio = (2 * u.eV) / (3 * u.fs)
bad_structural_sum = product + ratio  # error
# Length * Time and Time * Length are one product class, but products of
# more factors keep the order they were written in.
bad_structural_nominal_sum = product * ratio + ratio * product  # error
speed = 2 * u.angstrom_per_fs
bad_power_sum = speed**2 + speed**3  # error

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

# Plain numbers are only added to Dimensionless values; only those convert to float.
bad_plain_addition = (2 * u.nm) + 1  # error
bad_float = float(2 * u.nm)  # error
bad_text_scale = (2 * u.nm) * "2"  # error
bad_reinterpret = Length.reinterpret(2.0)  # error
bad_point_from_zero = 0 - (300 * u.K)  # error

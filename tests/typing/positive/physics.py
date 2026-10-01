"""Static types of the mechanics, electrostatics and thermal algebra."""

from typing import assert_type

from quantype import (
    Acceleration,
    Action,
    Charge,
    Dimensionless,
    DipoleMoment,
    ElectricField,
    ElectricPotential,
    Energy,
    Entropy,
    Force,
    Length,
    Mass,
    MassDensity,
    Momentum,
    Pressure,
    Temperature,
    Time,
    Volume,
    u,
)
from quantype.products import TemperatureSquared
from quantype.systems import SI, Metal

energy = 2 * u.eV
force = 1 * u.eV_per_angstrom
length = 2 * u.angstrom
time = 3 * u.fs
velocity = 1 * u.angstrom_per_fs
mass = 1 * u.Da
charge = 1 * u.e

# Each multiplication also names the divisions that undo it.
assert_type(energy / force, Length[float])
assert_type(energy / (1 * u.GPa), Volume[float])
assert_type(length / velocity, Time[float])
assert_type(force / mass, Acceleration[float])
assert_type(1.0 / (1 * u.THz), Time[float])

# Mechanics.
assert_type(mass * velocity, Momentum[float])
assert_type(mass * velocity * velocity, Energy[float])
assert_type(force * time, Momentum[float])
assert_type(mass * (1 * u.angstrom_per_fs2), Force[float])
assert_type(mass / (1 * u.angstrom_cubed), MassDensity[float])
assert_type(1 * u.gram_per_cubic_centimeter, MassDensity[float])

# Electrostatics.
assert_type(charge * (1 * u.V), Energy[float])
assert_type(charge * (1 * u.volt_per_angstrom), Force[float])
assert_type(charge * length, DipoleMoment[float])
assert_type(energy / charge, ElectricPotential[float])
assert_type(force / charge, ElectricField[float])
assert_type((1 * u.debye) * (1 * u.volt_per_angstrom), Energy[float])
assert_type(1 * u.coulomb, Charge[float])

# Absolute temperatures multiply and scale; only sums are rejected.
point = 300 * u.K
k = Entropy[float](8.617333262e-5, u.eV_per_kelvin)
assert_type(k * point, Energy[float])
assert_type(point * k, Energy[float])
assert_type(energy / k, Temperature[float])
assert_type(energy / point, Entropy[float])
assert_type(2 * point, Temperature[float])
assert_type(point / 2, Temperature[float])
assert_type(-point, Temperature[float])
assert_type(abs(point), Temperature[float])
assert_type(point / point, Dimensionless[float])
assert_type(point**2, TemperatureSquared[float])

# Planck's constant relates energy and angular frequency.
assert_type(energy * time, Action[float])
assert_type((energy * time) * (1.0 / time), Energy[float])

# Overridden kinds keep their static types; only raw numbers rescale.
bar = Pressure[float, Metal](1, u.bar)
assert_type(bar * Volume[float, Metal](1, u.angstrom_cubed), Energy[float, Metal])
assert_type(Mass[float, SI](1, u.kg), Mass[float, SI])

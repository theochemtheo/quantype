"""Physical meaning in Python types, with the unit system in the type too."""

from quantype import units as u
from quantype import units as units
from quantype._generated import (
    Acceleration,
    Action,
    Angle,
    Area,
    AtomCount,
    Charge,
    Dimensionless,
    DipoleMoment,
    ElectricField,
    ElectricPotential,
    ElectronCount,
    ElectronDensity,
    Energy,
    EnergyDensity,
    EnergyPerAtom,
    EnergyPerVolume,
    Entropy,
    Force,
    ForceConstant,
    Frequency,
    InverseTime,
    Length,
    MagneticMoment,
    Magnetization,
    Mass,
    MassDensity,
    Momentum,
    ParticleDensity,
    Pressure,
    Temperature,
    TemperatureDifference,
    TemperatureRate,
    Time,
    Velocity,
    Volume,
)
from quantype.core import Quantity, Unit

__all__ = [
    "Acceleration",
    "Action",
    "Angle",
    "Area",
    "AtomCount",
    "Charge",
    "Dimensionless",
    "DipoleMoment",
    "ElectricField",
    "ElectricPotential",
    "ElectronCount",
    "ElectronDensity",
    "Energy",
    "EnergyDensity",
    "EnergyPerAtom",
    "EnergyPerVolume",
    "Entropy",
    "Force",
    "ForceConstant",
    "Frequency",
    "InverseTime",
    "Length",
    "MagneticMoment",
    "Magnetization",
    "Mass",
    "MassDensity",
    "Momentum",
    "ParticleDensity",
    "Pressure",
    "Quantity",
    "Temperature",
    "TemperatureDifference",
    "TemperatureRate",
    "Time",
    "Unit",
    "Velocity",
    "Volume",
    "u",
    "units",
]


def __getattr__(name: str) -> str:
    # The installed version, read on first use so that importing stays fast.
    if name == "__version__":
        from importlib.metadata import version  # noqa: PLC0415

        return version("quantype")
    raise AttributeError(f"module 'quantype' has no attribute {name!r}")

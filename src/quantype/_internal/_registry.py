"""Declarative semantic catalogue; scales convert into the reference units.

The reference units (Å, eV, fs, K, μB, e) are the base units of the default
``Atomistic`` system, and every kind's reference unit is their coherent product.
``Atomistic`` stores mass and mass density in units of their own (see
``ATOMISTIC_OVERRIDES``). Every unit system is defined relative to the reference
units.

Dimensions use the named basis below, not SI mass dimensions. Equal dimension
vectors never imply semantic equality. Relations are deliberately explicit.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

from quantype import codata
from quantype._internal._codata import Codata, Edition

BASIS = (
    "length",
    "energy",
    "time",
    "temperature",
    "magnetic_moment",
    "atom",
    "electron",
    "charge",
)
type Dimensions = tuple[int, int, int, int, int, int, int, int]

# Units the default Atomistic system stores in place of the coherent eV fs²/Å²
# and eV fs²/Å⁵. The code generator also reads them, for kind docstrings.
ATOMISTIC_OVERRIDES = ("dalton", "gram_per_cubic_centimeter")


@dataclass(frozen=True)
class QuantitySpec:
    dimensions: Dimensions
    canonical_unit: str


@dataclass(frozen=True)
class UnitSpec:
    kind: str
    scale: float = 1.0
    offset: float = 0.0
    symbol: str | None = None
    aliases: tuple[str, ...] = ()


QUANTITIES = {
    "Dimensionless": QuantitySpec((0, 0, 0, 0, 0, 0, 0, 0), "one"),
    "Length": QuantitySpec((1, 0, 0, 0, 0, 0, 0, 0), "angstrom"),
    "Area": QuantitySpec((2, 0, 0, 0, 0, 0, 0, 0), "angstrom_squared"),
    "Volume": QuantitySpec((3, 0, 0, 0, 0, 0, 0, 0), "angstrom_cubed"),
    "Time": QuantitySpec((0, 0, 1, 0, 0, 0, 0, 0), "femtosecond"),
    "Velocity": QuantitySpec((1, 0, -1, 0, 0, 0, 0, 0), "angstrom_per_fs"),
    "Energy": QuantitySpec((0, 1, 0, 0, 0, 0, 0, 0), "electron_volt"),
    "EnergyPerAtom": QuantitySpec((0, 1, 0, 0, 0, -1, 0, 0), "eV_per_atom"),
    "Force": QuantitySpec((-1, 1, 0, 0, 0, 0, 0, 0), "eV_per_angstrom"),
    "ForceConstant": QuantitySpec((-2, 1, 0, 0, 0, 0, 0, 0), "eV_per_angstrom_squared"),
    "Pressure": QuantitySpec((-3, 1, 0, 0, 0, 0, 0, 0), "eV_per_angstrom_cubed"),
    "EnergyDensity": QuantitySpec((-3, 1, 0, 0, 0, 0, 0, 0), "energy_density"),
    "EnergyPerVolume": QuantitySpec((-3, 1, 0, 0, 0, 0, 0, 0), "energy_per_volume"),
    "Temperature": QuantitySpec((0, 0, 0, 1, 0, 0, 0, 0), "kelvin"),
    "TemperatureDifference": QuantitySpec((0, 0, 0, 1, 0, 0, 0, 0), "delta_kelvin"),
    "TemperatureRate": QuantitySpec((0, 0, -1, 1, 0, 0, 0, 0), "kelvin_per_fs"),
    "MagneticMoment": QuantitySpec((0, 0, 0, 0, 1, 0, 0, 0), "bohr_magneton"),
    "Magnetization": QuantitySpec(
        (-3, 0, 0, 0, 1, 0, 0, 0), "bohr_magneton_per_angstrom_cubed"
    ),
    "ParticleDensity": QuantitySpec(
        (-3, 0, 0, 0, 0, 1, 0, 0), "atom_per_angstrom_cubed"
    ),
    "ElectronDensity": QuantitySpec(
        (-3, 0, 0, 0, 0, 0, 1, 0), "electron_per_angstrom_cubed"
    ),
    "Angle": QuantitySpec((0, 0, 0, 0, 0, 0, 0, 0), "radian"),
    "Frequency": QuantitySpec((0, 0, -1, 0, 0, 0, 0, 0), "frequency_per_fs"),
    "InverseTime": QuantitySpec((0, 0, -1, 0, 0, 0, 0, 0), "per_fs"),
    "AtomCount": QuantitySpec((0, 0, 0, 0, 0, 1, 0, 0), "atom"),
    "ElectronCount": QuantitySpec((0, 0, 0, 0, 0, 0, 1, 0), "electron"),
    # Mass is derived, not a base axis: energy time² / length².
    "Mass": QuantitySpec((-2, 1, 2, 0, 0, 0, 0, 0), "eV_fs2_per_angstrom2"),
    "MassDensity": QuantitySpec((-5, 1, 2, 0, 0, 0, 0, 0), "eV_fs2_per_angstrom5"),
    "Momentum": QuantitySpec((-1, 1, 1, 0, 0, 0, 0, 0), "eV_fs_per_angstrom"),
    "Acceleration": QuantitySpec((1, 0, -2, 0, 0, 0, 0, 0), "angstrom_per_fs2"),
    "Charge": QuantitySpec((0, 0, 0, 0, 0, 0, 0, 1), "elementary_charge"),
    "ElectricPotential": QuantitySpec((0, 1, 0, 0, 0, 0, 0, -1), "volt"),
    "ElectricField": QuantitySpec((-1, 1, 0, 0, 0, 0, 0, -1), "volt_per_angstrom"),
    "DipoleMoment": QuantitySpec((1, 0, 0, 0, 0, 0, 0, 1), "e_angstrom"),
    "Entropy": QuantitySpec((0, 1, 0, -1, 0, 0, 0, 0), "eV_per_kelvin"),
    "Action": QuantitySpec((0, 1, 1, 0, 0, 0, 0, 0), "eV_fs"),
}


@dataclass(frozen=True)
class _Constants:
    """SI values that unit definitions use: exact definitions plus one edition."""

    codata: Codata
    # Exact by definition, whichever CODATA edition is in use.
    angstrom = 1e-10
    kilo, centi, milli, nano, pico, femto = 1e3, 1e-2, 1e-3, 1e-9, 1e-12, 1e-15
    giga, tera = 1e9, 1e12
    calorie = 4.184  # thermochemical
    erg = 1e-7
    zero_Celsius = 273.15  # noqa: N815 -- the conventional symbol
    degree = math.pi / 180

    @property
    def electron_volt(self) -> float:
        return self.codata.elementary_charge

    @property
    def N_A(self) -> float:  # noqa: N802 -- the conventional symbol
        return self.codata.avogadro_constant


def unit_specs(edition: Edition | None = None) -> MappingProxyType[str, UnitSpec]:
    """Unit definitions from one CODATA edition; by default, the process's."""
    return _unit_specs(codata.values(edition))


@cache
def _unit_specs(values: Codata) -> MappingProxyType[str, UnitSpec]:
    c = _Constants(values)
    return MappingProxyType(
        {**_common_units(c), **_system_units(c), **_electromechanical_units(c)}
    )


def _common_units(c: _Constants) -> dict[str, UnitSpec]:
    ev_joule = c.electron_volt
    bohr_angstrom = c.codata.bohr_radius / c.angstrom
    hartree_ev = c.codata.hartree_energy_in_ev
    bohr_magneton_si = c.codata.bohr_magneton
    return {
        "one": UnitSpec("Dimensionless", symbol="1", aliases=("dimensionless",)),
        "angstrom": UnitSpec("Length", symbol="Å", aliases=("Å", "Angstrom", "Ang")),
        "meter": UnitSpec("Length", 1 / c.angstrom, symbol="m", aliases=("m", "metre")),
        "centimeter": UnitSpec(
            "Length", c.centi / c.angstrom, symbol="cm", aliases=("cm",)
        ),
        "nanometer": UnitSpec(
            "Length", c.nano / c.angstrom, symbol="nm", aliases=("nm",)
        ),
        "bohr": UnitSpec("Length", bohr_angstrom, symbol="a0"),
        "angstrom_squared": UnitSpec("Area", symbol="Å^2", aliases=("angstrom2",)),
        "angstrom_cubed": UnitSpec("Volume", symbol="Å^3", aliases=("angstrom3",)),
        "femtosecond": UnitSpec("Time", symbol="fs", aliases=("fs",)),
        "picosecond": UnitSpec("Time", c.pico / c.femto, symbol="ps", aliases=("ps",)),
        "second": UnitSpec("Time", 1 / c.femto, symbol="s", aliases=("s",)),
        "angstrom_per_fs": UnitSpec("Velocity", symbol="Å/fs"),
        "electron_volt": UnitSpec("Energy", symbol="eV", aliases=("eV",)),
        "millielectron_volt": UnitSpec(
            "Energy", c.milli, symbol="meV", aliases=("meV",)
        ),
        "joule": UnitSpec("Energy", 1 / ev_joule, symbol="J", aliases=("J",)),
        "hartree": UnitSpec("Energy", hartree_ev, symbol="Ha", aliases=("Ha",)),
        "rydberg": UnitSpec("Energy", hartree_ev / 2, symbol="Ry", aliases=("Ry",)),
        "eV_per_atom": UnitSpec(
            "EnergyPerAtom", symbol="eV/atom", aliases=("eV / atom", "eV/atom")
        ),
        "hartree_per_atom": UnitSpec(
            "EnergyPerAtom", hartree_ev, symbol="Ha/atom", aliases=("hartree / atom",)
        ),
        "J_per_atom": UnitSpec(
            "EnergyPerAtom", 1 / ev_joule, symbol="J/atom", aliases=("J / atom",)
        ),
        "eV_per_angstrom": UnitSpec(
            "Force",
            symbol="eV/Å",
            aliases=("eV / angstrom", "eV/angstrom", "eV/Å", "eV/Ang"),
        ),
        "newton": UnitSpec("Force", c.angstrom / ev_joule, symbol="N", aliases=("N",)),
        "hartree_per_bohr": UnitSpec(
            "Force",
            hartree_ev / bohr_angstrom,
            symbol="Ha/a0",
            aliases=("hartree / bohr",),
        ),
        "eV_per_angstrom_squared": UnitSpec("ForceConstant", symbol="eV/Å^2"),
        "eV_per_angstrom_cubed": UnitSpec(
            "Pressure", symbol="eV/Å^3", aliases=("eV / angstrom^3",)
        ),
        "pascal": UnitSpec(
            "Pressure", c.angstrom**3 / ev_joule, symbol="Pa", aliases=("Pa",)
        ),
        "gigapascal": UnitSpec(
            "Pressure",
            c.giga * c.angstrom**3 / ev_joule,
            symbol="GPa",
            aliases=("GPa",),
        ),
        "hartree_per_bohr_cubed": UnitSpec(
            "Pressure", hartree_ev / bohr_angstrom**3, symbol="Ha/a0^3"
        ),
        "energy_density": UnitSpec("EnergyDensity", symbol="eV/Å^3"),
        "energy_per_volume": UnitSpec("EnergyPerVolume", symbol="eV/Å^3"),
        "kelvin": UnitSpec("Temperature", symbol="K", aliases=("K",)),
        "celsius": UnitSpec(
            "Temperature", offset=c.zero_Celsius, symbol="°C", aliases=("degC",)
        ),
        "delta_kelvin": UnitSpec(
            "TemperatureDifference", symbol="ΔK", aliases=("delta_K",)
        ),
        "delta_celsius": UnitSpec(
            "TemperatureDifference", symbol="Δ°C", aliases=("delta_degC",)
        ),
        "kelvin_per_fs": UnitSpec("TemperatureRate", symbol="K/fs"),
        "bohr_magneton": UnitSpec("MagneticMoment", symbol="μB", aliases=("mu_B",)),
        "ampere_meter_squared": UnitSpec(
            "MagneticMoment", 1 / bohr_magneton_si, symbol="A m^2"
        ),
        "bohr_magneton_per_angstrom_cubed": UnitSpec("Magnetization", symbol="μB/Å^3"),
        "atom_per_angstrom_cubed": UnitSpec("ParticleDensity", symbol="atom/Å^3"),
        "electron_per_angstrom_cubed": UnitSpec(
            "ElectronDensity", symbol="electron/Å^3"
        ),
        "radian": UnitSpec("Angle", symbol="rad", aliases=("rad",)),
        "degree": UnitSpec("Angle", c.degree, symbol="°", aliases=("deg",)),
        "frequency_per_fs": UnitSpec("Frequency", symbol="fs^-1"),
        "terahertz": UnitSpec(
            "Frequency", c.tera * c.femto, symbol="THz", aliases=("THz",)
        ),
        "hertz": UnitSpec("Frequency", c.femto, symbol="Hz", aliases=("Hz",)),
        "per_fs": UnitSpec("InverseTime", symbol="fs^-1"),
        "atom": UnitSpec("AtomCount", symbol="atom"),
        "electron": UnitSpec("ElectronCount", symbol="electron"),
    }


def _system_units(c: _Constants) -> dict[str, UnitSpec]:
    """Named derived units, so every built-in system's kinds have catalogue names.

    Scales are products of each system's base scales, keeping the systems coherent.
    """
    ev_joule = c.electron_volt
    au_time = c.codata.atomic_unit_of_time / c.femto
    au_moment = c.codata.atomic_unit_of_magnetic_dipole_moment / c.codata.bohr_magneton
    bohr = c.codata.bohr_radius / c.angstrom
    hartree = c.codata.hartree_energy_in_ev
    kcal_mol = c.kilo * c.calorie / c.N_A / ev_joule
    metre, centimetre = 1 / c.angstrom, c.centi / c.angstrom
    second, picosecond = 1 / c.femto, c.pico / c.femto
    joule, erg = 1 / ev_joule, c.erg / ev_joule
    ampere_metre2 = 1 / c.codata.bohr_magneton
    erg_per_gauss = c.milli * ampere_metre2
    units = {
        # LAMMPS metal: Å, eV, ps.
        "angstrom_per_ps": UnitSpec("Velocity", 1 / picosecond, symbol="Å/ps"),
        "kelvin_per_ps": UnitSpec("TemperatureRate", 1 / picosecond, symbol="K/ps"),
        "per_ps": UnitSpec("InverseTime", 1 / picosecond, symbol="ps^-1"),
        # LAMMPS real: Å, kcal/mol, fs.
        "kcal_per_mol": UnitSpec(
            "Energy", kcal_mol, symbol="kcal/mol", aliases=("kcal/mol",)
        ),
        "kcal_per_mol_per_atom": UnitSpec(
            "EnergyPerAtom", kcal_mol, symbol="kcal/mol/atom"
        ),
        "kcal_per_mol_per_angstrom": UnitSpec(
            "Force", kcal_mol, symbol="kcal/mol/Å", aliases=("kcal/mol/Å",)
        ),
        "kcal_per_mol_per_angstrom_squared": UnitSpec(
            "ForceConstant", kcal_mol, symbol="kcal/mol/Å^2"
        ),
        "kcal_per_mol_per_angstrom_cubed": UnitSpec(
            "Pressure", kcal_mol, symbol="kcal/mol/Å^3"
        ),
        # SI: m, J, s, A·m².
        "square_meter": UnitSpec("Area", metre**2, symbol="m^2"),
        "cubic_meter": UnitSpec("Volume", metre**3, symbol="m^3"),
        "meter_per_second": UnitSpec(
            "Velocity", metre / second, symbol="m/s", aliases=("m/s",)
        ),
        "newton_per_meter": UnitSpec("ForceConstant", joule / metre**2, symbol="N/m"),
        "kelvin_per_second": UnitSpec("TemperatureRate", 1 / second, symbol="K/s"),
        "ampere_per_meter": UnitSpec(
            "Magnetization", ampere_metre2 / metre**3, symbol="A/m"
        ),
        "atom_per_cubic_meter": UnitSpec(
            "ParticleDensity", metre**-3, symbol="atom/m^3"
        ),
        "electron_per_cubic_meter": UnitSpec(
            "ElectronDensity", metre**-3, symbol="electron/m^3"
        ),
        "per_second": UnitSpec("InverseTime", 1 / second, symbol="s^-1"),
        # CGS: cm, erg, s, erg/G.
        "square_centimeter": UnitSpec("Area", centimetre**2, symbol="cm^2"),
        "cubic_centimeter": UnitSpec("Volume", centimetre**3, symbol="cm^3"),
        "centimeter_per_second": UnitSpec(
            "Velocity", centimetre / second, symbol="cm/s"
        ),
        "erg": UnitSpec("Energy", erg, symbol="erg"),
        "erg_per_atom": UnitSpec("EnergyPerAtom", erg, symbol="erg/atom"),
        "dyne": UnitSpec("Force", erg / centimetre, symbol="dyn", aliases=("dyn",)),
        "dyne_per_centimeter": UnitSpec(
            "ForceConstant", erg / centimetre**2, symbol="dyn/cm"
        ),
        "barye": UnitSpec("Pressure", erg / centimetre**3, symbol="Ba"),
        "erg_per_gauss": UnitSpec("MagneticMoment", erg_per_gauss, symbol="erg/G"),
        "emu_per_cubic_centimeter": UnitSpec(
            "Magnetization", erg_per_gauss / centimetre**3, symbol="emu/cm^3"
        ),
        "atom_per_cubic_centimeter": UnitSpec(
            "ParticleDensity", centimetre**-3, symbol="atom/cm^3"
        ),
        "electron_per_cubic_centimeter": UnitSpec(
            "ElectronDensity", centimetre**-3, symbol="electron/cm^3"
        ),
        # Hartree atomic units: a0, Ha, ħ/Eh, eħ/me.
        "bohr_squared": UnitSpec("Area", bohr**2, symbol="a0^2"),
        "bohr_cubed": UnitSpec("Volume", bohr**3, symbol="a0^3"),
        "atomic_time": UnitSpec("Time", au_time, symbol="ħ/Eh"),
        "atomic_velocity": UnitSpec("Velocity", bohr / au_time, symbol="a0 Eh/ħ"),
        "hartree_per_bohr_squared": UnitSpec(
            "ForceConstant", hartree / bohr**2, symbol="Ha/a0^2"
        ),
        "kelvin_per_atomic_time": UnitSpec(
            "TemperatureRate", 1 / au_time, symbol="K Eh/ħ"
        ),
        "atomic_magnetic_moment": UnitSpec("MagneticMoment", au_moment, symbol="eħ/me"),
        "atomic_magnetization": UnitSpec(
            "Magnetization", au_moment / bohr**3, symbol="eħ/me/a0^3"
        ),
        "atom_per_bohr_cubed": UnitSpec(
            "ParticleDensity", bohr**-3, symbol="atom/a0^3"
        ),
        "electron_per_bohr_cubed": UnitSpec(
            "ElectronDensity", bohr**-3, symbol="electron/a0^3"
        ),
        "frequency_per_atomic_time": UnitSpec("Frequency", 1 / au_time, symbol="Eh/ħ"),
        "per_atomic_time": UnitSpec("InverseTime", 1 / au_time, symbol="Eh/ħ"),
    }
    # Pressure, EnergyDensity and EnergyPerVolume share dimensions but not kinds,
    # so each needs its own named unit in every system.
    for prefix, kind in (
        ("energy_density", "EnergyDensity"),
        ("energy_per_volume", "EnergyPerVolume"),
    ):
        for suffix, scale, symbol in (
            ("kcal_per_mol_per_angstrom_cubed", kcal_mol, "kcal/mol/Å^3"),
            ("joule_per_cubic_meter", joule / metre**3, "J/m^3"),
            ("erg_per_cubic_centimeter", erg / centimetre**3, "erg/cm^3"),
            ("hartree_per_bohr_cubed", hartree / bohr**3, "Ha/a0^3"),
        ):
            units[f"{prefix}_{suffix}"] = UnitSpec(kind, scale, symbol=symbol)
    return units


def _electromechanical_units(c: _Constants) -> dict[str, UnitSpec]:
    """Mass, charge and their relatives, with a named unit in every built-in system.

    Also the everyday units that the catalogue otherwise lacks. Mass is derived
    (eV fs²/Å² in the reference units), so its scales divide by that unit in kg.
    """
    ev = c.electron_volt  # J per eV, and C per elementary charge
    mass = ev * 1e-10  # kg per eV fs²/Å²
    per_atom_mole = 1 / c.N_A
    kcal_mol = c.kilo * c.calorie * per_atom_mole / ev
    bohr = c.codata.bohr_radius / c.angstrom
    hartree = c.codata.hartree_energy_in_ev
    au_time = c.codata.atomic_unit_of_time / c.femto
    hbar = c.codata.planck_constant / (2 * math.pi) / ev / c.femto  # eV fs
    statcoulomb = 1 / (10 * c.codata.speed_of_light_in_vacuum)  # C
    statvolt = 1e-7 / statcoulomb  # V, which is eV per e
    pascal = c.angstrom**3 / ev
    return {
        # Everyday units of existing kinds.
        "millimeter": UnitSpec("Length", 1e7, symbol="mm", aliases=("mm",)),
        "micrometer": UnitSpec("Length", 1e4, symbol="µm", aliases=("um", "µm", "μm")),
        "picometer": UnitSpec("Length", 1e-2, symbol="pm", aliases=("pm",)),
        "nanosecond": UnitSpec("Time", 1e6, symbol="ns", aliases=("ns",)),
        "microsecond": UnitSpec("Time", 1e9, symbol="µs", aliases=("us", "µs", "μs")),
        "millisecond": UnitSpec("Time", 1e12, symbol="ms", aliases=("ms",)),
        "kilojoule": UnitSpec("Energy", c.kilo / ev, symbol="kJ", aliases=("kJ",)),
        "kilocalorie": UnitSpec(
            "Energy", c.kilo * c.calorie / ev, symbol="kcal", aliases=("kcal",)
        ),
        "kJ_per_mol": UnitSpec(
            "Energy", c.kilo * per_atom_mole / ev, symbol="kJ/mol", aliases=("kJ/mol",)
        ),
        "bar": UnitSpec("Pressure", 1e5 * pascal, symbol="bar"),
        "kilobar": UnitSpec("Pressure", 1e8 * pascal, symbol="kbar", aliases=("kbar",)),
        "atmosphere": UnitSpec(
            "Pressure", 101325 * pascal, symbol="atm", aliases=("atm",)
        ),
        "megapascal": UnitSpec(
            "Pressure", 1e6 * pascal, symbol="MPa", aliases=("MPa",)
        ),
        # Spectroscopic wavenumbers, as the frequency c times the wavenumber.
        "inverse_centimeter": UnitSpec(
            "Frequency",
            c.codata.speed_of_light_in_vacuum / c.centi * c.femto,
            symbol="cm^-1",
            aliases=("cm^-1", "cm-1"),
        ),
        # Mass. LAMMPS writes per-particle masses in g/mol.
        "eV_fs2_per_angstrom2": UnitSpec("Mass", symbol="eV fs^2/Å^2"),
        "dalton": UnitSpec(
            "Mass",
            c.codata.atomic_mass_constant / mass,
            symbol="Da",
            aliases=("Da", "amu"),
        ),
        "gram_per_mole": UnitSpec(
            "Mass", 1e-3 * per_atom_mole / mass, symbol="g/mol", aliases=("g/mol",)
        ),
        "kilogram": UnitSpec("Mass", 1 / mass, symbol="kg", aliases=("kg",)),
        "gram": UnitSpec("Mass", 1e-3 / mass, symbol="g", aliases=("g",)),
        "electron_mass": UnitSpec(
            "Mass", c.codata.electron_mass / mass, symbol="m_e", aliases=("m_e",)
        ),
        "eV_fs2_per_angstrom5": UnitSpec("MassDensity", symbol="eV fs^2/Å^5"),
        "gram_per_cubic_centimeter": UnitSpec(
            "MassDensity", 1e-27 / mass, symbol="g/cm^3", aliases=("g/cm^3",)
        ),
        "kilogram_per_cubic_meter": UnitSpec(
            "MassDensity", 1e-30 / mass, symbol="kg/m^3", aliases=("kg/m^3",)
        ),
        "electron_mass_per_bohr_cubed": UnitSpec(
            "MassDensity", c.codata.electron_mass / mass / bohr**3, symbol="m_e/a0^3"
        ),
        "eV_fs_per_angstrom": UnitSpec("Momentum", symbol="eV fs/Å"),
        "eV_ps_per_angstrom": UnitSpec("Momentum", 1e3, symbol="eV ps/Å"),
        "kcal_per_mol_fs_per_angstrom": UnitSpec(
            "Momentum", kcal_mol, symbol="kcal/mol fs/Å"
        ),
        "kilogram_meter_per_second": UnitSpec("Momentum", 1e-5 / mass, symbol="kg m/s"),
        "gram_centimeter_per_second": UnitSpec(
            "Momentum", 1e-10 / mass, symbol="g cm/s"
        ),
        "atomic_momentum": UnitSpec("Momentum", hbar / bohr, symbol="ħ/a0"),
        "angstrom_per_fs2": UnitSpec("Acceleration", symbol="Å/fs^2"),
        "angstrom_per_ps2": UnitSpec("Acceleration", 1e-6, symbol="Å/ps^2"),
        "meter_per_second_squared": UnitSpec(
            "Acceleration", 1e-20, symbol="m/s^2", aliases=("m/s^2",)
        ),
        "centimeter_per_second_squared": UnitSpec(
            "Acceleration", 1e-22, symbol="cm/s^2"
        ),
        "atomic_acceleration": UnitSpec(
            "Acceleration", bohr / au_time**2, symbol="a0 Eh^2/ħ^2"
        ),
        # Charge and electrostatics. An elementary charge is the reference.
        "elementary_charge": UnitSpec("Charge", symbol="e", aliases=("e",)),
        "coulomb": UnitSpec("Charge", 1 / ev, symbol="C", aliases=("C",)),
        "statcoulomb": UnitSpec(
            "Charge", statcoulomb / ev, symbol="statC", aliases=("statC",)
        ),
        "volt": UnitSpec("ElectricPotential", symbol="V", aliases=("V",)),
        "kcal_per_mol_per_e": UnitSpec(
            "ElectricPotential", kcal_mol, symbol="kcal/mol/e"
        ),
        "statvolt": UnitSpec(
            "ElectricPotential", statvolt, symbol="statV", aliases=("statV",)
        ),
        "hartree_per_e": UnitSpec("ElectricPotential", hartree, symbol="Ha/e"),
        "volt_per_angstrom": UnitSpec(
            "ElectricField", symbol="V/Å", aliases=("V/Å", "V/angstrom", "V/Ang")
        ),
        "volt_per_meter": UnitSpec(
            "ElectricField", 1e-10, symbol="V/m", aliases=("V/m",)
        ),
        "statvolt_per_centimeter": UnitSpec(
            "ElectricField", statvolt * 1e-8, symbol="statV/cm"
        ),
        "atomic_electric_field": UnitSpec(
            "ElectricField", hartree / bohr, symbol="Ha/(e a0)"
        ),
        "e_angstrom": UnitSpec("DipoleMoment", symbol="e Å"),
        "debye": UnitSpec(
            "DipoleMoment",
            1e-21 / c.codata.speed_of_light_in_vacuum / ev / c.angstrom,
            symbol="D",
            aliases=("D",),
        ),
        "coulomb_meter": UnitSpec("DipoleMoment", 1 / ev / c.angstrom, symbol="C m"),
        "statcoulomb_centimeter": UnitSpec(
            "DipoleMoment", statcoulomb * c.centi / ev / c.angstrom, symbol="statC cm"
        ),
        "e_bohr": UnitSpec("DipoleMoment", bohr, symbol="e a0"),
        # Entropy is the kind of the Boltzmann constant; action, of ħ.
        "eV_per_kelvin": UnitSpec("Entropy", symbol="eV/K", aliases=("eV/K",)),
        "joule_per_kelvin": UnitSpec("Entropy", 1 / ev, symbol="J/K", aliases=("J/K",)),
        "erg_per_kelvin": UnitSpec("Entropy", c.erg / ev, symbol="erg/K"),
        "kcal_per_mol_per_kelvin": UnitSpec("Entropy", kcal_mol, symbol="kcal/mol/K"),
        "hartree_per_kelvin": UnitSpec("Entropy", hartree, symbol="Ha/K"),
        "eV_fs": UnitSpec("Action", symbol="eV fs"),
        "eV_ps": UnitSpec("Action", 1e3, symbol="eV ps"),
        "kcal_per_mol_fs": UnitSpec("Action", kcal_mol, symbol="kcal/mol fs"),
        "joule_second": UnitSpec("Action", 1 / ev / c.femto, symbol="J s"),
        "erg_second": UnitSpec("Action", c.erg / ev / c.femto, symbol="erg s"),
        "atomic_action": UnitSpec("Action", hbar, symbol="ħ"),
    }


def __getattr__(name: str) -> MappingProxyType[str, UnitSpec]:
    if name == "UNITS":
        return unit_specs()
    raise AttributeError(name)


type Relations = dict[tuple[str, str, str], str]

# Multiplication entries are symmetric, and each one also names the divisions
# that undo it (see close_relations). Nothing is inferred from dimensions alone:
# Pressure and EnergyDensity stay distinct although their vectors match.
RELATIONS: Relations = {
    ("mul", "Length", "Length"): "Area",
    ("mul", "Area", "Length"): "Volume",
    ("mul", "Force", "Length"): "Energy",
    ("mul", "ForceConstant", "Length"): "Force",
    ("mul", "ForceConstant", "Area"): "Energy",
    ("mul", "Pressure", "Volume"): "Energy",
    ("mul", "Velocity", "Time"): "Length",
    ("mul", "Acceleration", "Time"): "Velocity",
    ("mul", "Mass", "Velocity"): "Momentum",
    ("mul", "Mass", "Acceleration"): "Force",
    ("mul", "Momentum", "Velocity"): "Energy",
    ("mul", "Force", "Time"): "Momentum",
    ("mul", "MassDensity", "Volume"): "Mass",
    ("mul", "EnergyPerAtom", "AtomCount"): "Energy",
    ("mul", "ParticleDensity", "Volume"): "AtomCount",
    ("mul", "ElectronDensity", "Volume"): "ElectronCount",
    ("mul", "Magnetization", "Volume"): "MagneticMoment",
    ("mul", "TemperatureRate", "Time"): "TemperatureDifference",
    ("mul", "Entropy", "Temperature"): "Energy",
    ("mul", "Entropy", "TemperatureDifference"): "Energy",
    ("mul", "Charge", "ElectricPotential"): "Energy",
    ("mul", "Charge", "ElectricField"): "Force",
    ("mul", "ElectricField", "Length"): "ElectricPotential",
    ("mul", "Charge", "Length"): "DipoleMoment",
    ("mul", "DipoleMoment", "ElectricField"): "Energy",
    ("mul", "Energy", "Time"): "Action",
    ("mul", "Action", "InverseTime"): "Energy",
    ("mul", "Frequency", "Time"): "Dimensionless",
    ("mul", "InverseTime", "Time"): "Dimensionless",
    ("div", "Energy", "Area"): "ForceConstant",
    ("div", "Force", "Length"): "ForceConstant",
    ("div", "Force", "Area"): "Pressure",
    ("div", "Energy", "Volume"): "EnergyDensity",
    ("div", "MagneticMoment", "Volume"): "Magnetization",
    ("div", "Energy", "AtomCount"): "EnergyPerAtom",
    ("div", "AtomCount", "Volume"): "ParticleDensity",
    ("div", "ElectronCount", "Volume"): "ElectronDensity",
    ("div", "TemperatureDifference", "Time"): "TemperatureRate",
    ("div", "Dimensionless", "Time"): "InverseTime",
    # k_B T / k_B: both temperature kinds multiply entropy, so choose one.
    ("div", "Energy", "Entropy"): "Temperature",
}
for (_op, _left, _right), _result in tuple(RELATIONS.items()):
    if _op == "mul":
        RELATIONS[(_op, _right, _left)] = _result
for _kind in QUANTITIES:
    RELATIONS[("div", _kind, _kind)] = "Dimensionless"
    RELATIONS[("mul", _kind, "Dimensionless")] = _kind
    RELATIONS[("mul", "Dimensionless", _kind)] = _kind
    RELATIONS[("div", _kind, "Dimensionless")] = _kind


def close_relations(declared: Mapping[tuple[str, str, str], str]) -> Relations:
    """Declared relations, plus multiplication's symmetry and the divisions undoing it.

    ``A * B -> C`` also gives ``B * A -> C``, ``C / A -> B`` and ``C / B -> A``.
    A declared division always wins. Two multiplications that would undo into
    different results make that division ambiguous; declare it explicitly.
    """
    closed = dict(declared)
    for (operation, left, right), result in declared.items():
        if operation == "mul":
            closed.setdefault(("mul", right, left), result)
    derived: Relations = {}
    ambiguous: set[tuple[str, str, str]] = set()
    for (operation, left, right), result in closed.items():
        if operation != "mul":
            continue
        for key, quotient in (
            (("div", result, left), right),
            (("div", result, right), left),
        ):
            if key in declared:
                continue
            if derived.setdefault(key, quotient) != quotient:
                ambiguous.add(key)
    if ambiguous:
        described = ", ".join(
            sorted(f"{left} / {right}" for _, left, right in ambiguous)
        )
        raise ValueError(
            f"Ambiguous division {described}: several multiplications undo into "
            "different kinds. Declare the division explicitly"
        )
    return closed | derived


POWERS: dict[tuple[str, int], str] = {
    ("Length", 2): "Area",
    ("Length", 3): "Volume",
}

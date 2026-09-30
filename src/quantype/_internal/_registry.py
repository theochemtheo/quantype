"""Declarative semantic catalogue; scales convert into the reference units.

The reference units (Å, eV, fs, K, μB) are the coherent units of the default
``Atomistic`` system. Every other unit system is defined relative to them.

Dimensions use the named basis below, not SI mass dimensions. Equal dimension
vectors never imply semantic equality. Relations are deliberately explicit.
"""

from dataclasses import dataclass
from functools import cache
from types import ModuleType

BASIS = (
    "length",
    "energy",
    "time",
    "temperature",
    "magnetic_moment",
    "atom",
    "electron",
)
type Dimensions = tuple[int, int, int, int, int, int, int]


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
    "Dimensionless": QuantitySpec((0, 0, 0, 0, 0, 0, 0), "one"),
    "Length": QuantitySpec((1, 0, 0, 0, 0, 0, 0), "angstrom"),
    "Area": QuantitySpec((2, 0, 0, 0, 0, 0, 0), "angstrom_squared"),
    "Volume": QuantitySpec((3, 0, 0, 0, 0, 0, 0), "angstrom_cubed"),
    "Time": QuantitySpec((0, 0, 1, 0, 0, 0, 0), "femtosecond"),
    "Velocity": QuantitySpec((1, 0, -1, 0, 0, 0, 0), "angstrom_per_fs"),
    "Energy": QuantitySpec((0, 1, 0, 0, 0, 0, 0), "electron_volt"),
    "EnergyPerAtom": QuantitySpec((0, 1, 0, 0, 0, -1, 0), "eV_per_atom"),
    "Force": QuantitySpec((-1, 1, 0, 0, 0, 0, 0), "eV_per_angstrom"),
    "ForceConstant": QuantitySpec((-2, 1, 0, 0, 0, 0, 0), "eV_per_angstrom_squared"),
    "Pressure": QuantitySpec((-3, 1, 0, 0, 0, 0, 0), "eV_per_angstrom_cubed"),
    "EnergyDensity": QuantitySpec((-3, 1, 0, 0, 0, 0, 0), "energy_density"),
    "EnergyPerVolume": QuantitySpec((-3, 1, 0, 0, 0, 0, 0), "energy_per_volume"),
    "Temperature": QuantitySpec((0, 0, 0, 1, 0, 0, 0), "kelvin"),
    "TemperatureDifference": QuantitySpec((0, 0, 0, 1, 0, 0, 0), "delta_kelvin"),
    "TemperatureRate": QuantitySpec((0, 0, -1, 1, 0, 0, 0), "kelvin_per_fs"),
    "MagneticMoment": QuantitySpec((0, 0, 0, 0, 1, 0, 0), "bohr_magneton"),
    "Magnetization": QuantitySpec(
        (-3, 0, 0, 0, 1, 0, 0), "bohr_magneton_per_angstrom_cubed"
    ),
    "ParticleDensity": QuantitySpec((-3, 0, 0, 0, 0, 1, 0), "atom_per_angstrom_cubed"),
    "ElectronDensity": QuantitySpec(
        (-3, 0, 0, 0, 0, 0, 1), "electron_per_angstrom_cubed"
    ),
    "Angle": QuantitySpec((0, 0, 0, 0, 0, 0, 0), "radian"),
    "Frequency": QuantitySpec((0, 0, -1, 0, 0, 0, 0), "frequency_per_fs"),
    "InverseTime": QuantitySpec((0, 0, -1, 0, 0, 0, 0), "per_fs"),
    "AtomCount": QuantitySpec((0, 0, 0, 0, 0, 1, 0), "atom"),
    "ElectronCount": QuantitySpec((0, 0, 0, 0, 0, 0, 1), "electron"),
}


@cache
def unit_specs() -> dict[str, UnitSpec]:
    """Resolve installed SciPy constants once, on first use of a unit."""
    from scipy import constants as c  # noqa: PLC0415

    return {**_common_units(c), **_system_units(c)}


def _common_units(c: ModuleType) -> dict[str, UnitSpec]:
    ev_joule = c.electron_volt
    bohr_angstrom = c.physical_constants["Bohr radius"][0] / c.angstrom
    hartree_ev = c.physical_constants["Hartree energy in eV"][0]
    bohr_magneton_si = c.physical_constants["Bohr magneton"][0]
    return {
        "one": UnitSpec("Dimensionless", symbol="1", aliases=("dimensionless",)),
        "angstrom": UnitSpec("Length", symbol="Å", aliases=("Å",)),
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
            "Force", symbol="eV/Å", aliases=("eV / angstrom", "eV/angstrom", "eV/Å")
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


def _system_units(c: ModuleType) -> dict[str, UnitSpec]:
    """Named derived units, so every built-in system's kinds have catalogue names.

    Scales are products of each system's base scales, keeping the systems coherent.
    """
    ev_joule = c.electron_volt
    au_time = c.physical_constants["atomic unit of time"][0] / c.femto
    au_moment = (
        c.physical_constants["atomic unit of mag. dipole mom."][0]
        / c.physical_constants["Bohr magneton"][0]
    )
    bohr = c.physical_constants["Bohr radius"][0] / c.angstrom
    hartree = c.physical_constants["Hartree energy in eV"][0]
    kcal_mol = c.kilo * c.calorie / c.N_A / ev_joule
    metre, centimetre = 1 / c.angstrom, c.centi / c.angstrom
    second, picosecond = 1 / c.femto, c.pico / c.femto
    joule, erg = 1 / ev_joule, c.erg / ev_joule
    ampere_metre2 = 1 / c.physical_constants["Bohr magneton"][0]
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


def __getattr__(name: str) -> dict[str, UnitSpec]:
    if name == "UNITS":
        return unit_specs()
    raise AttributeError(name)


# Multiplication entries are symmetric. Division is not inferred from dimensions:
# Pressure and EnergyDensity must remain distinct even though their vectors match.
RELATIONS: dict[tuple[str, str, str], str] = {
    ("mul", "Length", "Length"): "Area",
    ("mul", "Area", "Length"): "Volume",
    ("mul", "Force", "Length"): "Energy",
    ("mul", "ForceConstant", "Length"): "Force",
    ("mul", "ForceConstant", "Area"): "Energy",
    ("mul", "Velocity", "Time"): "Length",
    ("mul", "EnergyPerAtom", "AtomCount"): "Energy",
    ("mul", "ParticleDensity", "Volume"): "AtomCount",
    ("mul", "ElectronDensity", "Volume"): "ElectronCount",
    ("mul", "Magnetization", "Volume"): "MagneticMoment",
    ("mul", "TemperatureRate", "Time"): "TemperatureDifference",
    ("div", "Length", "Time"): "Velocity",
    ("div", "Area", "Length"): "Length",
    ("div", "Volume", "Length"): "Area",
    ("div", "Volume", "Area"): "Length",
    ("div", "Energy", "Length"): "Force",
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
}
for (_op, _left, _right), _result in tuple(RELATIONS.items()):
    if _op == "mul":
        RELATIONS[(_op, _right, _left)] = _result
for _kind in QUANTITIES:
    if _kind != "Temperature":
        RELATIONS[("div", _kind, _kind)] = "Dimensionless"
        RELATIONS[("mul", _kind, "Dimensionless")] = _kind
        RELATIONS[("mul", "Dimensionless", _kind)] = _kind
        RELATIONS[("div", _kind, "Dimensionless")] = _kind

POWERS: dict[tuple[str, int], str] = {
    ("Length", 2): "Area",
    ("Length", 3): "Volume",
}

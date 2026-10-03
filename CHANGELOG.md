# Changelog

All notable changes to quantype are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-03

`Atomistic` now stores masses in daltons and mass densities in g/cm³, so code
that reads the `.value` of a mass or density, or wraps raw masses with
`from_value`, needs updating; see Changed. The release also adds what a file
reader needs to fill typed quantities: unit lookup by name, annotation
inspection, and construction without a display unit.

### Added

- ASE's units: `u.ase_time` (Å √(amu/eV), about 10.18 fs), and
  `u.ase_velocity` and `u.ase_momentum`, the units of `atoms.get_velocities()`
  and `atoms.get_momenta()`.
- `Kind.unit_named(name)` finds a unit of that kind by catalogue name, alias,
  or display symbol, as `parse` reads them: `Force.unit_named("Ha/a0") is
  u.hartree_per_bohr`. `units=` adds custom units, and `Force[V, S].unit_named`
  also finds the units only `S` defines.
- `quantype.typing.quantity_type(annotation)` reads the kind class, storage
  type, and unit system from an annotation such as `Force[float, SI]`, and
  returns `None` for anything else.
- `Force.kind` gives the kind's name on the class, as `.kind` does on a
  quantity.
- Typed construction takes `display=False` to convert without remembering the
  input unit, so the value shows in the system's unit, as with `from_value`.

### Changed

- `Atomistic`, the default system, stores masses in daltons and mass densities
  in g/cm³, instead of eV fs²/Å² and eV fs²/Å⁵. `Mass[float](12, u.Da).value`
  is now `12.0`, and `Mass.from_value` takes daltons. Products that involve a
  mass or a density rescale, as they do in `Metal` and `Real`. See
  [ADR-013](https://github.com/theochemtheo/quantype/blob/v0.2.0/docs/decisions/ADR-013-atomistic-stores-mass-in-daltons.md).
- Unknown unit names in `parse`, Pydantic validation, and `load_npz` list the
  kind's units, and a display symbol of the requested kind takes precedence
  over a name that belongs to another kind.
- A third type argument, as in `Length[float, SI, int]`, raises `TypeError`
  when the alias is called or read.

## [0.1.0] - 2026-10-03

Initial release of `quantype`, which makes physical dimensions and unit systems
part of your types. mypy, Pyright, Pyrefly, and ty check quantity arithmetic
without a plugin: `Energy / Length` is a `Force`, and `Energy + Length` is a
type error and a runtime `TypeError`. Values are plain floats or NumPy, JAX, and
Torch arrays, stored in the `Atomistic`, `SI`, `CGS`, `Atomic`, `Metal`, or
`Real` unit system, or one you define. The release also includes CODATA
constants, JAX and Torch autodiff that types a gradient by its physical kind,
JSON, Pydantic, and NPZ serialization, and custom catalogues of kinds and
relations.

[Unreleased]: https://github.com/theochemtheo/quantype/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/theochemtheo/quantype/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/theochemtheo/quantype/releases/tag/v0.1.0

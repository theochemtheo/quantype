# Changelog

All notable changes to quantype are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-15

Quantities no longer have a unit system unless you choose one.
`Length[float]` is now `Length[float, NoSystem]`: it keeps the unit it was
given in, and has no `.value` until `.to_system(...)` picks a system. Code that
reads `.value` or calls `from_value` on default quantities, or calls
`with_value`, needs updating; see Changed and
[ADR-014](https://github.com/theochemtheo/quantype/blob/v0.3.0/docs/decisions/ADR-014-opt-in-unit-system.md).

### Added

- `NoSystem`, the default system, and `AnySystem`, the alias
  `UnitSystem | NoSystem` that bounds the system parameter. Code that accepts
  quantities with or without a system is generic over `[S: AnySystem]`.
- `q.apply(function)` maps the stored numbers and keeps the kind, units, and
  storage backend. It works with or without a system.
- `q.simplified()` merges units of one dimension and shows a named kind in a
  display system that holds every unit left, so `(kcal/mol/Å)·nm` is
  `kcal/mol` and `Da·(Å/fs)^2` is `eV`.
- `q.to(System)` shows a quantity in a system's unit for its kind without
  converting it into that system.

### Changed

- Construction and decoding without a system store the number in the unit it
  was given in, so `Length[float](2, u.nm).magnitude(u.nm)` returns the input
  exactly. Units with an offset, such as °C, still store kelvin.
- `.value` and `from_value` need a unit system, and all four checkers reject
  them without one. `from_value(raw, system)` takes the system:
  `Length.from_value(2.0, SI)`.
- `+` and `-` convert the right operand into the left one's unit:
  `2 nm + 3 Å` is `2.3 nm`.
- Products, quotients, and powers without a system compose their operands'
  units instead of converting: `Energy[float](3, u.eV) / Length[float](2, u.nm)`
  shows `1.5 eV/nm`, and `.magnitude()` returns numbers in that unit. A
  composed unit takes a catalogue name when the scales agree, so J/m is N.
  Composed units parse back through `parse`, `unit_named`, and `to_dict`.
- A quantity without a system adopts the other operand's system in
  arithmetic, comparisons, `==`, `quantype.numpy`, and `assert_allclose`, as
  constants do. Two unit systems still don't mix, and a parameter typed
  `Length[float, SI]` rejects a quantity without a system.
- `==` and ordering compare as arithmetic does, so `1 nm == 10 Å` and
  `q == q.to_system(SI)`. `q == 0` works for every kind except absolute
  temperatures.
- Quantities are unhashable, scalars included. Key dicts, sets, caches, and
  `jax.jit` static arguments on `q.magnitude(unit)`.
- `Length.parse`, Pydantic fields, `load_npz` targets, and
  `quantype.typing.quantity_type` read an annotation without a system, such as
  `Force[float]`, as `NoSystem` instead of `Atomistic`.
- JAX pytrees and `ujax`/`utorch` derivatives see reference units, so `jit`
  traces once across input units. Values that come out of a transformation
  show in reference units.
- `quantype.numpy` functions that take several quantities of one kind have
  per-kind overloads, so checkers reject mixed kinds.
- Each catalogue unit scale and physical constant is computed exactly and
  rounded to a float once. 46 scales move by 1 to 5 ulp, so a picosecond is
  exactly 1000 fs and N = kg·m/s² holds.
- Error messages for mismatched kinds and systems, unit arithmetic, and array
  methods called on a quantity name the fix.
- `x**1` keeps its kind, at runtime and in the stubs.
- `from_value` and `apply` reject booleans, lists, and integer arrays.
- `assert_allclose` compares in the desired quantity's unit, compares
  temperatures in kelvin, and takes an `atol` without a system.

### Removed

- `with_value`. Use `apply` to change numbers in place, or construct a new
  quantity from `.magnitude(unit)` to change backend.

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

[Unreleased]: https://github.com/theochemtheo/quantype/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/theochemtheo/quantype/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/theochemtheo/quantype/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/theochemtheo/quantype/releases/tag/v0.1.0

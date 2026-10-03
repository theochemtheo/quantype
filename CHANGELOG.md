# Changelog

All notable changes to quantype are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `Atomistic`, the default system, stores masses in daltons and mass densities
  in g/cm³, instead of eV fs²/Å² and eV fs²/Å⁵. `Mass[float](12, u.Da).value`
  is now `12.0`, and `Mass.from_value` takes daltons. Products that involve a
  mass or a density rescale, as they do in `Metal` and `Real`. See
  [ADR-013](docs/decisions/ADR-013-atomistic-stores-mass-in-daltons.md).

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

[Unreleased]: https://github.com/theochemtheo/quantype/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/theochemtheo/quantype/releases/tag/v0.1.0

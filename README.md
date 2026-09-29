# quantype

Physical meaning in Python types, with canonical numerical storage. Python 3.12+
with NumPy, SciPy, and Pydantic v2; JAX and Torch are optional extras.

```bash
uv add quantype
uv add 'quantype[jax]'       # JAX transformations and autodiff
uv add 'quantype[torch]'     # Torch tensors and autograd
```

## Construction and storage

```python
import numpy as np
import numpy.typing as npt
from typing import assert_type
from quantype import Energy, Force, Length, u

length = Length[np.float64](2, u.length.nanometer)
assert_type(length, Length[np.float64])
assert length.value == 20  # canonical angstroms, stored as a NumPy float64 scalar

positions = Length[npt.NDArray[np.float64]](
    [[1, 2, 3]],
    u.length.nanometer,
)
energy = Energy[np.float64](3, u.energy.electron_volt)
assert_type(energy / length, Force[np.float64])
```

The type argument **converts storage** at construction. `np.float64` means a
NumPy scalar; `npt.NDArray[np.float64]` means an array, including zero-dimensional
arrays. Python `float` is also supported. Scalars reject non-scalar input; real
floating storage rejects boolean, complex, and non-numerical magnitudes.

Units describe input magnitudes, not the resulting quantity type. Construction
converts into canonical units and defaults to canonical presentation. Shape and
symmetry are the caller's responsibility; the old `quantype.shapes` helpers have
been removed.

The previous unit-first API remains available:

```python
length = 2 * u.nm
positions = u.angstrom(np.zeros((100, 3)))
```

Unit-first construction preserves existing numerical storage except that Python
integers normalize to `float`. Use `unit(array)` for reliable cross-backend
construction; backend-left multiplication is not promised outside NumPy.

Hierarchical namespaces use full catalogue names, such as `u.length.nanometer`
and `u.temperature.celsius`. Flat abbreviations remain conveniences. Where a
flat name conflicts with a namespace, the namespace wins: use
`u.dimensionless.one`, `u.energy_density.energy_density`, or
`u.energy_per_volume.energy_per_volume`.

### Explicit numerical boundaries

- `.value` exposes canonical numerical data.
- `.magnitude(unit)` converts to a compatible unit.
- `.to(unit)` selects presentation/serialization units without changing data.
- `.magnitude()` uses the chosen presentation unit, or canonical units.
- `Length.from_canonical(raw)` is the trusted, storage-preserving boundary. It
  does not inspect or convert data, including when used in compiled models.
- Implicit NumPy coercion is rejected rather than silently discarding meaning.

Canonical units are angstrom, eV, fs, kelvin, Bohr magneton, atom, and electron.
Atom and electron are independent dimensional axes. Conversion factors come
from the **installed SciPy constants**, resolved once on first unit use. Importing
`quantype` alone does not import NumPy, SciPy, Pydantic, Torch, or JAX.

## Physical algebra

Kinds are nominal: `Pressure`, `EnergyDensity`, and `EnergyPerVolume` remain
distinct despite equal dimensions. `Force / Area` yields `Pressure`;
`Energy / Volume` yields `EnergyDensity`. Incompatible operations are rejected
both by the supported type checkers and at runtime.

Unlisted products retain structural types such as
`Quantity[Mul[LengthKind, TimeKind], float]`, not `Any`. Runtime expressions are
structured trees, not parsed strings. There is no arbitrary symbolic cancellation
or inference of physical meaning from dimensions alone.

Absolute temperatures are affine points:

```python
from quantype import Temperature, TemperatureDifference

cold = Temperature[float](0, u.temperature.celsius)
warm = Temperature[float](300, u.temperature.kelvin)
assert_type(warm - cold, TemperatureDifference[float])
```

Differences can be scaled; absolute temperatures cannot be added together,
multiplied, scaled, or exponentiated. Means of absolute temperatures are valid;
sums are not.

`sum`, `mean`, `u.sqrt(Area)`, `u.sin(Angle)`, and `u.exp(Dimensionless)` delegate
to numerical backends. Array shape algebra and arbitrary dtype promotion are
outside the static contract.

## Define a unit without global registration

```python
from math import sqrt
from quantype import Temperature

bleb = Temperature.define_unit(
    "my_lab:bleb",
    reference=u.temperature.celsius,
    scale=sqrt(2),
)
point = Temperature[np.float64](1, bleb)
assert point.magnitude(u.temperature.celsius) > 1.4
```

This definition means `kelvin = bleb * sqrt(2) + 273.15`: zero bleb is zero
Celsius. In general, `reference_magnitude = input * scale + offset`.
Definitions are immutable and validated once. A difference unit must be defined
explicitly using a temperature-difference reference.

A unit object can be used immediately for construction, conversion, and
serialization. Decoding its identifier requires the definition explicitly:

```python
wire = point.to(bleb).to_dict()
restored = Temperature.parse(wire, units=(bleb,))
```

Namespaced identifiers avoid accidental collisions. Duplicate definitions and
shadowing built-in identifiers are rejected during decoding. There is no
import-time mutation of the public unit namespace.

## JSON and Pydantic

Discrete values use an explicit physical kind:

```json
{"kind": "Length", "magnitude": 5.0, "unit": "angstrom"}
```

The old `value`/`units` object format is no longer accepted. Scalar strings such
as `"5 angstrom"` remain accepted when the expected quantity kind is supplied.

```python
from pydantic import BaseModel, TypeAdapter


class Configuration(BaseModel):
    cutoff: Length[float]
    positions: Length[npt.NDArray[np.float64]]


config = Configuration.model_validate(
    {
        "cutoff": "0.5 nm",
        "positions": {"kind": "Length", "magnitude": [[1, 2, 3]], "unit": "nm"},
    }
)
restored = Configuration.model_validate_json(config.model_dump_json())

adapter = TypeAdapter(Temperature[np.float64])
restored_point = adapter.validate_python(wire, context={"units": (bleb,)})
```

Pydantic restoration converts to the annotated storage, including NumPy dtype.
Bare quantity annotations work at runtime; strict static code should specify
storage. `Length.parse(...)` intentionally returns scalar-or-float64-array
storage; use a typed constructor or `TypeAdapter` for a particular target.

JSON stores values, not backend/device/dtype metadata. Serialization is an
explicit host boundary: Torch tensors are detached and copied to CPU there,
never inside arithmetic or differentiation.

## Binary arrays: NPZ and NPY

NPZ archives store numerical arrays plus a versioned JSON metadata entry, with
no pickle dependency:

```python
from quantype.serialization import load_npz, save_npz

save_npz("frame.npz", positions=positions, energy=energy, record_backend=True)
restored_positions = load_npz(
    "frame.npz",
    "positions",
    Length[npt.NDArray[np.float64]],
)
```

Arrays carry dtype and shape, including zero-dimensional and empty arrays.
Metadata identifies each quantity's kind, unit, and array key. With
`record_backend=True`, it also records `source_backend`, for example `"numpy"`
or `"torch"`. That field is **provenance**, not an instruction to import a backend.
The target type controls restoration; a Torch-produced archive can be read into
NumPy without Torch installed. Supply `units=(bleb,)` for custom-unit decoding.

```python
with np.load("frame.npz", allow_pickle=False) as archive:
    metadata_json = str(archive["metadata"])
```

Graphs and device placement are not serialized. Unsupported host dtypes, such as
Torch bfloat16 through NumPy's normal conversion, are not silently approximated.
JSON and NPZ are value-oriented formats, not bit-exact snapshots of model state;
noncanonical unit conversion can introduce floating-point roundoff.

NPY deliberately carries only numerical data. Use an explicit external unit:

```python
np.save("positions.npy", positions.magnitude(u.nm), allow_pickle=False)
restored_positions = Length[npt.NDArray[np.float64]](
    np.load("positions.npy", allow_pickle=False),
    u.nm,
)
```

## Optional numerical backends

```python
import torch
from quantype import utorch

x = Length[torch.Tensor](
    torch.tensor([1.0, 2.0], requires_grad=True),
    u.length.nanometer,
    dtype=torch.float64,
)
energy = Energy.from_canonical((x.value**2).sum())
gradient = utorch.grad(energy, x, create_graph=True)
```

Torch/JAX classes do not encode dtype, so constructors and `load_npz` accept an
explicit `dtype=`. Existing Torch tensors retain their graph and device during
conversion. JAX float64 requests require x64 support; unavailable explicit dtypes
fail rather than silently producing a different dtype.

```python
import jax
from quantype import ujax


def harmonic(x: Length[jax.Array]) -> Energy[jax.Array]:
    k = 2.0 * u.eV_per_angstrom_squared
    return 0.5 * k * (x**2).sum()


x = Length[jax.Array]([1, 2, 3], u.length.angstrom)
gradient = ujax.jit(ujax.grad(harmonic))(x)
hessian = ujax.hessian(harmonic)(x)
```

Importing `ujax` registers quantities as single-leaf pytrees for JAX `jit` and
`vmap`. Arithmetic and autodiff use the same physical algebra. Gradients are
positive derivatives; physical force is `-gradient`. Adapters currently support
one quantity input and scalar output, not general JVP/VJP or multi-input models.

## Generate an application catalogue

The public generator uses the same definition model, validation, and renderers
as the built-in API. It requires Ruff for formatting, not a type-checker plugin.

```python
from quantype.catalogue import QuantitySpec, UnitSpec, builtin_catalogue
from quantype.codegen import generate

catalogue = builtin_catalogue().extend(
    quantities={
        "SurfaceTension": QuantitySpec((-2, 1, 0, 0, 0, 0, 0), "surface_tension"),
    },
    units={"surface_tension": UnitSpec("SurfaceTension")},
    relations={("mul", "Pressure", "Length"): "SurfaceTension"},
)
generate(catalogue, "src/labquantities", package="labquantities")
```

Then import consistently from that generated package:

```python
from labquantities import Length, Pressure, SurfaceTension, u

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)
assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])
```

Multiplication relations are symmetric; division relations are explicit.
Generation includes runtime classes, stubs, units, numerical helpers, and
optional autodiff adapters. `generate(..., check=True)` returns stale file names
without writing; `render(...)` returns source strings without invoking tools.

**This is a combined catalogue, not a patch to the installed built-ins.** Its
`Length` is a distinct nominal type from `quantype.Length`. Independent extension
stubs cannot add overloads to existing built-in classes, and reflected operators
do not override their structural typing fallback. See [the refactor report](docs/refactor.md)
for the probe results and tradeoffs. Custom storage backends remain out of scope.

## Development

```bash
uv sync --all-extras
uv run just generate
uv run just check-generated
uv run just test
uv run just typecheck
uv run just lint
uv build
```

`just lint` runs [zizmor](https://github.com/zizmorcore/zizmor) offline against the
GitHub Actions workflows; the pre-commit hook checks workflow changes too.

The top-level `quantype` import provides quantities and `u`; `quantype.units`
holds the unit namespaces. Use `quantype.serialization` for JSON/NPZ boundaries,
`quantype.ujax` or `quantype.utorch` for optional autodiff, and
`quantype.catalogue` with `quantype.codegen` for application catalogues.
`quantype.core` and `quantype.kinds` expose the base types and structural markers
for advanced annotations. The generated quantity classes and stubs live beside
them because the catalogue generator uses the same package layout externally.

Implementation helpers live under `quantype._internal`: `_registry.py` owns the
built-in definitions, `_semantics.py` the algebra, `_unit.py` conversions,
`_construction.py` and `_storage.py` storage boundaries, and `_validation.py`
the Pydantic adapter. Generation separates catalogue validation, source rendering,
and formatting/file output.

The conformance suite checks positive `assert_type` examples and marked negative
examples with strict mypy (both parsers), Pyright, Pyrefly, and ty. Stubtest checks
runtime/stub agreement. Generated overloads narrowly suppress intentional
structural-fallback overlaps. CI is configured to exercise core-only, JAX-only,
Torch-only, and minimum-core-dependency environments.

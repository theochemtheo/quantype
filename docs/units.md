# Units, storage, and physical algebra

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
symmetry are the caller's responsibility. Noncanonical conversion uses wider
working precision before rounding into the requested storage; presentation
conversion also widens intermediates and retains the stored dtype. Existing
low-precision input rounding cannot be undone. Working precision is limited by
the backend (for example, JAX without x64); conversion factors outside its range
raise `ValueError`.

Units can also construct quantities directly:

```python
import numpy as np
from quantype import u

length = 2 * u.nm
positions = u.angstrom(np.zeros((100, 3)))
```

Unit-first construction preserves existing real floating numerical storage,
including NumPy floating scalars, where conversion allows it; Python integers
become `float`. Boolean, complex, non-numerical, and integer backend
arrays/scalars are rejected. To convert integer arrays or Python lists to floating
storage, use a typed constructor. Use `unit(array)` for reliable cross-backend construction;
backend-left multiplication is not promised outside NumPy.

Hierarchical namespaces use full catalogue names, such as `u.length.nanometer`
and `u.temperature.celsius`. Flat abbreviations such as `u.nm` are also available.
Where a flat name conflicts with a namespace, the namespace wins: use
`u.dimensionless.one`, `u.energy_density.energy_density`, or
`u.energy_per_volume.energy_per_volume`.

## Explicit numerical boundaries

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
structured trees, not parsed strings. These quantities support arithmetic and
reductions while retaining their structural kinds. Scalar reciprocals use declared
relations, such as `1 / Time` yielding `InverseTime`, or retain a structural ratio.
Structural reciprocals use the dimensionless identity of their catalogue.
For expressions mixing catalogues, scalar reciprocals are ambiguous and raise
`TypeError`; supply an explicit dimensionless quantity numerator instead.
There is no arbitrary symbolic cancellation or inference of physical meaning from
dimensions alone.

Absolute temperatures are affine points:

```python
from typing import assert_type
from quantype import Temperature, TemperatureDifference, u

cold = Temperature[float](0, u.temperature.celsius)
warm = Temperature[float](300, u.temperature.kelvin)
assert_type(warm - cold, TemperatureDifference[float])
```

Differences can be scaled; absolute temperatures cannot be added together,
multiplied, scaled, or exponentiated. Means of absolute temperatures are valid;
sums are not.

Quantity methods `.sum()` and `.mean()`, plus `u.sqrt(Area)`, `u.sin(Angle)`, and
`u.exp(Dimensionless)`, delegate to numerical backends. Array shape algebra and
arbitrary dtype promotion are outside the static contract. Use quantity
reductions, not `np.sum(q)` or `np.mean(q)`. For indexing and other unsupported
array operations, see the
[explicit NumPy boundary example](getting-started.md#work-with-numpy-arrays).

## Define a unit without global registration

```python
from math import sqrt
import numpy as np
from quantype import Temperature, u

bleb = Temperature.define_unit(
    "my_lab:bleb",
    reference=u.temperature.celsius,
    scale=sqrt(2),
)
point = Temperature[np.float64](1, bleb)
assert point.magnitude(u.temperature.celsius) > 1.4

wire = point.to(bleb).to_dict()
restored = Temperature.parse(wire, units=(bleb,))
```

This definition means `kelvin = bleb * sqrt(2) + 273.15`: zero bleb is zero
Celsius. In general, `reference_magnitude = input * scale + offset`.
Definitions are immutable and validated once. A difference unit must be defined
explicitly using a temperature-difference reference.

A unit object can be used immediately for construction, conversion, and
serialization. Decoding its identifier requires the definition explicitly.
Namespaced identifiers avoid accidental collisions. Duplicate definitions and
shadowing built-in identifiers are rejected during decoding. Existing built-in
unit objects may also be supplied in `units=`; they do not count as shadowing.
There is no import-time mutation of the public unit namespace.

For typed restoration, see [serialization](serialization.md). To add new
physical kinds and relationships, see [custom catalogues](custom-catalogues.md).

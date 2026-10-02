# Serialization

quantype writes quantities to JSON, Pydantic models, and NPZ archives. Only
values are saved: autodiff graphs, device placement, and model state are lost.
Data that uses a custom unit needs the unit's definition when it is read back;
see [custom units](units.md#define-a-unit-without-global-registration).

Saved data records a unit but not a unit system. The type you decode into
chooses the system, so any JSON document or archive can be read into any
system.

## JSON and Pydantic

A serialized quantity names its kind:

```json
{"kind": "Length", "magnitude": 0.5, "unit": "nanometer"}
```

`to_dict` writes the value in its display unit: the unit it was given in, or
the one chosen with `.to(unit)`. A value with no display unit is written in its
system's unit for the kind. That is a catalogue name for built-in systems, or a
system-qualified identifier such as `gromacs:Force` for
[custom systems](unit-systems.md#decoding). A configuration's `"0.5 nm"` is
written back as `0.5 nanometer` in every system, and `"20.1 degC"` as exactly
`20.1 celsius`.

A quantity object has `magnitude` and `unit`, and `kind` if it names its kind,
which must then match: written objects always include it, and hand-written ones
can leave it out where the target kind is known, as in a Pydantic field or
`Length.parse`. A string such as `"5 angstrom"` is accepted there too.
`model_dump()` keeps quantities, as it keeps datetimes, and
`model_dump(mode="json")` and `model_dump_json()` write the objects.

```python
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from quantype import Length


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
```

A field's unit system comes from its annotation, so `cutoff: Length[float, SI]`
reads `"0.5 nm"` as meters. A quantity object from another system raises;
convert it with `.to_system(...)` first.

Pass custom units through the validation context:

```python
from math import sqrt
import numpy as np
from pydantic import TypeAdapter
from quantype import Temperature, u

bleb = Temperature.define_unit(
    "my_lab:bleb",
    reference=u.temperature.celsius,
    scale=sqrt(2),
)
point = Temperature[np.float64](1, bleb)
wire = point.to_dict()
adapter = TypeAdapter(Temperature[np.float64])
restored_point = adapter.validate_python(wire, context={"units": (bleb,)})
```

Pydantic converts to the annotated storage, including the NumPy dtype, and to
the annotated unit system. A bare annotation such as `Length` works at runtime,
but strict type checkers want the storage spelled out. `Length.parse(...)`
returns a float or a float64 array in the default system, and
`Length[float, SI].parse(...)` chooses the system. For a particular storage
type, use a typed constructor or a `TypeAdapter`. When the target type names a
custom system, that system's units are found without `units=`.

JSON numbers are written as Python floats (binary64), including values from
NumPy `longdouble` storage. Extra precision is rounded off, and a finite value
that would become infinity or zero raises `ValueError`. Use NPZ to keep the
dtype and extended precision.

JSON doesn't record the backend or dtype. Torch tensors are detached and copied
to the CPU when they are serialized, and at no other time.

## Binary arrays: NPZ and NPY

An NPZ archive stores the numerical arrays and a versioned JSON metadata entry,
and needs no pickle:

```python
import numpy as np
import numpy.typing as npt
from quantype import Energy, Length, u
from quantype.serialization import load_npz, save_npz
from quantype.systems import SI

positions = Length[npt.NDArray[np.float64]]([[1, 2, 3]], u.nm)
energy = Energy[np.float64](3, u.eV)
save_npz("frame.npz", positions=positions, energy=energy, record_backend=True)
restored_positions = load_npz(
    "frame.npz",
    "positions",
    Length[npt.NDArray[np.float64]],
)
in_metres = load_npz("frame.npz", "positions", Length[npt.NDArray[np.float64], SI])

with np.load("frame.npz", allow_pickle=False) as archive:
    metadata_json = str(archive["metadata"])
```

Arrays keep their dtype and shape, including zero-dimensional and empty arrays.
The metadata gives each quantity's kind, unit, and array key, and the CODATA
edition the units came from (for example, `"2022"`). With `record_backend=True`,
it also records `source_backend`, such as `"numpy"` or `"torch"`. That field is
only a record, and loading never imports the backend. The target type decides
the storage and unit system, so a Torch-produced archive can be read into NumPy
without Torch installed, and an archive written from SI quantities can be read
into `Atomistic`. Restored quantities keep the archived unit for display. Pass
`units=(...)` to decode custom units.

Dtypes NumPy can't represent, such as Torch's bfloat16, raise `TypeError`.
JSON and NPZ store values, and unit conversion on the way can add
floating-point round-off. Low-precision storage shown in a very different unit
can lose more: float16 ångströms written as meters are subnormal. Switch such
values to a nearby unit with `.to(...)` before saving.

An NPY file holds only numbers, so you choose the unit when saving and when
loading:

```python
import numpy as np
import numpy.typing as npt
from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([[1, 2, 3]], u.nm)
np.save("positions.npy", positions.magnitude(u.nm), allow_pickle=False)
restored_positions = Length[npt.NDArray[np.float64]](
    np.load("positions.npy", allow_pickle=False),
    u.nm,
)
```

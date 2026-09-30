# Serialization

Serialization is an explicit host boundary. Values are stored, not graphs,
device placement, or model state. Custom-unit decoding requires the unit
definitions at the boundary; see [custom units](units.md#define-a-unit-without-global-registration).

## JSON and Pydantic

Serialized quantities carry an explicit physical kind:

```json
{"kind": "Length", "magnitude": 5.0, "unit": "angstrom"}
```

Quantity objects require exactly `kind`, `magnitude`, and `unit`. Scalar strings
such as `"5 angstrom"` are accepted when the expected quantity kind is supplied.

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

Custom units are supplied through validation context:

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
wire = point.to(bleb).to_dict()
adapter = TypeAdapter(Temperature[np.float64])
restored_point = adapter.validate_python(wire, context={"units": (bleb,)})
```

Pydantic restoration converts to the annotated storage, including NumPy dtype.
Bare quantity annotations work at runtime; strict static code should specify
storage. `Length.parse(...)` intentionally returns scalar-or-float64-array
storage; use a typed constructor or `TypeAdapter` for a particular target.

JSON number leaves are Python floats (binary64), including values from NumPy
`longdouble` storage. Extra precision is rounded; finite extended-range values
that would become infinity or round to zero are rejected with `ValueError`.
Use NPZ when dtype and extended precision must be preserved.

JSON stores values, not backend/device/dtype metadata. Torch tensors are detached
and copied to CPU at serialization, never inside arithmetic or differentiation.

## Binary arrays: NPZ and NPY

NPZ archives store numerical arrays plus a versioned JSON metadata entry, with
no pickle dependency:

```python
import numpy as np
import numpy.typing as npt
from quantype import Energy, Length, u
from quantype.serialization import load_npz, save_npz

positions = Length[npt.NDArray[np.float64]]([[1, 2, 3]], u.nm)
energy = Energy[np.float64](3, u.eV)
save_npz("frame.npz", positions=positions, energy=energy, record_backend=True)
restored_positions = load_npz(
    "frame.npz",
    "positions",
    Length[npt.NDArray[np.float64]],
)

with np.load("frame.npz", allow_pickle=False) as archive:
    metadata_json = str(archive["metadata"])
```

Arrays carry dtype and shape, including zero-dimensional and empty arrays.
Metadata identifies each quantity's kind, unit, and array key. With
`record_backend=True`, it also records `source_backend`, for example `"numpy"`
or `"torch"`. That field is **provenance**, not an instruction to import a backend.
The target type controls restoration; a Torch-produced archive can be read into
NumPy without Torch installed. Supply `units=(...)` for custom-unit decoding.

Graphs and device placement are not serialized. Unsupported host dtypes, such as
Torch bfloat16 through NumPy's normal conversion, are not silently approximated.
JSON and NPZ are value-oriented formats, not bit-exact snapshots of model state;
noncanonical unit conversion can introduce floating-point roundoff.

NPY deliberately carries only numerical data. Use an explicit external unit:

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

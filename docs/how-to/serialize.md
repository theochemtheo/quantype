# Save and restore quantities

Choose JSON for configuration, NPZ for typed arrays, and NPY only when the
unit is supplied by an external agreement. File examples write to the current
working directory.

!!! warning "Persistence saves values, not execution state"
    Serialization does not preserve autodiff graphs, device placement, or model
    state. Torch tensors are detached and moved to CPU; the
    [value-boundary reference](../reference/serialization.md#value-boundary-and-unit-selection)
    describes backend and precision handling.

## Write a JSON quantity

Use `to_dict` and standard JSON tools; parse into a known kind on read:

```python
import json
from pathlib import Path

from quantype import Length, u
from quantype.systems import SI

length = Length[float](0.5, u.nm)
path = Path("cutoff.json")
path.write_text(json.dumps(length.to_dict()))
wire = json.loads(path.read_text())
assert wire == {"kind": "Length", "magnitude": 0.5, "unit": "nanometer"}
restored = Length[float, SI].parse(wire)
assert restored.value == 5e-10
assert restored.unit is u.nm
```

For a Pydantic model, use `model_dump_json()` or `model_dump(mode="json")`;
plain `model_dump()` keeps quantity instances, as it keeps datetimes. See
[validate data](validate-data.md) for complete model round trips and custom-unit
validation context.

`q.to_dict(unit)` explicitly selects the written unit; otherwise the current
presentation is used. Before saving low-precision values, choose a nearby unit
with `.to(...)`: float16 ångströms written as meters can become subnormal.
JSON rounds extended precision to binary64. Use NPZ when dtype and extended
precision matter. The [format reference](../reference/serialization.md) defines
unit identifiers, accepted input, errors, and precision limits.

## Save several quantities in one NPZ archive

```python
import json

import numpy as np
import numpy.typing as npt

from quantype import Energy, Length, u
from quantype.serialization import load_npz, save_npz
from quantype.systems import SI

positions = Length[npt.NDArray[np.float64]]([[1, 2, 3]], u.nm)
energy = Energy[np.float64](3, u.eV)
save_npz("frame.npz", positions=positions, energy=energy, record_backend=True)
restored = load_npz("frame.npz", "positions", Length[npt.NDArray[np.float64]])
in_meters = load_npz("frame.npz", "positions", Length[npt.NDArray[np.float64], SI])
np.testing.assert_allclose(restored.magnitude(u.nm), [[1, 2, 3]])
np.testing.assert_allclose(in_meters.value, [[1e-9, 2e-9, 3e-9]])
assert restored.dtype == np.float64
assert restored.shape == (1, 3)
with np.load("frame.npz", allow_pickle=False) as archive:
    metadata = json.loads(str(archive["metadata"]))
assert metadata["version"] == 1
assert metadata["quantities"]["positions"]["source_backend"] == "numpy"
```

The target type chooses storage and system; source backend metadata is
provenance only. A Torch-produced archive can load into NumPy without Torch
installed. Zero-dimensional and empty arrays retain their shape when the target
storage supports it. Unit conversion can introduce roundoff; this is not a
bit-exact model snapshot.

## Restore a custom unit from NPZ

```python
import numpy as np
import numpy.typing as npt

from quantype import Length, u
from quantype.serialization import load_npz, save_npz

pixel = Length.define_unit("camera:pixel", reference=u.nm, scale=2.0)
positions = Length[npt.NDArray[np.float64]]([1, 2], pixel)
save_npz("pixels.npz", positions=positions)
restored = load_npz(
    "pixels.npz", "positions", Length[npt.NDArray[np.float64]], units=(pixel,)
)
np.testing.assert_allclose(restored.magnitude(u.nm), [2, 4])
assert restored.unit is pixel
```

Typed custom-system targets find their own units automatically. Otherwise
provide custom definitions explicitly; see [define units](define-units.md).

## Use NPY with an explicit unit agreement

!!! warning "NPY needs an external unit agreement"
    NPY contains no physical metadata. Record the unit in your file format or
    application schema, then use the same agreement to reconstruct:

```python
import numpy as np
import numpy.typing as npt

from quantype import Length, u

positions = Length[npt.NDArray[np.float64]]([[1, 2, 3]], u.nm)
np.save("positions.npy", positions.magnitude(u.nm), allow_pickle=False)
restored = Length[npt.NDArray[np.float64]](
    np.load("positions.npy", allow_pickle=False), u.nm
)
np.testing.assert_allclose(restored.magnitude(u.nm), [[1, 2, 3]])
```

Never infer a unit from a filename or dtype. For annotation-driven readers, use
[quantity_type](../reference/type-annotations.md) to obtain a field's target.

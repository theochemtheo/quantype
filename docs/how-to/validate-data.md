# Validate configuration and external data

Start with a known quantity
[kind](../explanation/glossary.md#kind "A nominal physical meaning, such as Length or Energy."):
a bare numerical value does not tell a validator its physical meaning.

## Validate several fields with Pydantic

Annotate each field with the desired kind and storage. Validation checks the
kind and unit, and restores the annotated numerical dtype:

```python
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ValidationError

from quantype import Length, u


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
assert restored.cutoff.magnitude(u.angstrom) == 5.0
assert '"magnitude":0.5,"unit":"nanometer"' in config.model_dump_json()
assert restored.positions.dtype == np.float64
np.testing.assert_allclose(restored.positions.magnitude(u.angstrom), [[10, 20, 30]])
try:
    Configuration.model_validate({"cutoff": "1 eV", "positions": config.positions})
except ValidationError as error:
    assert "Expected Length; received Energy" in str(error)
else:
    raise AssertionError("An energy is not a length")
```

## Choose a decoding entry point

| Input and goal | Entry point |
| --- | --- |
| Numerical data with a known unit and desired storage | `Length[float](data, u.nm)` |
| The same, stored in another unit system | `Length[float, SI](data, u.nm)` |
| Trusted raw numbers in a system's units, preserving storage and graphs | `Length.from_value(data, Atomistic)` |
| External quantity object or scalar string, default storage | `Length.parse(data)` |
| External data with a particular storage type | `TypeAdapter(Length[np.float32]).validate_python(data)` |
| Several validated fields | A Pydantic `BaseModel` |

`TypeAdapter` comes from `pydantic`. See
[constructors](../reference/quantities.md#construction-and-storage) for numerical
inputs and [wire objects and strings](../reference/serialization.md#json-wire-objects-and-strings)
for accepted fields, syntax, and errors.

## Decode into a chosen unit system

Strings and wire objects carry units but no system, so they decode into the
field's system. Existing quantity instances must already have that system;
Pydantic does not adopt or convert them implicitly:

```python
from pydantic import TypeAdapter, ValidationError

from quantype import Length, u
from quantype.systems import SI

adapter = TypeAdapter(Length[float, SI])
assert adapter.validate_python("0.5 nm").value == 5e-10
literal = Length[float](0.5, u.nm)
try:
    adapter.validate_python(literal)
except ValidationError as error:
    assert "Expected a quantity in SI; received one in NoSystem" in str(error)
else:
    raise AssertionError("Fields must not adopt a system")
assert adapter.validate_python(literal.to_system(SI)).value == 5e-10
```

Choose a system at your application's boundary; see
[systems reference](../reference/systems.md) and
[define systems](define-systems.md).

## Supply custom unit definitions

!!! info "Pass custom units in validation context"
    A custom unit works immediately as an object, but decoding its name needs
    its definition. Pass `context={"units": (bleb,)}` to `TypeAdapter` or
    `BaseModel.model_validate` / `model_validate_json`; no global registration
    occurs. See the [custom-unit decoding contract](../reference/serialization.md#custom-unit-decoding).

For example:

```python
from math import isclose, sqrt

import numpy as np
from pydantic import TypeAdapter

from quantype import Temperature, u

bleb = Temperature.define_unit("my_lab:bleb", reference=u.celsius, scale=sqrt(2))
point = Temperature[np.float64](1, bleb)
adapter = TypeAdapter(Temperature[np.float64])
restored = adapter.validate_json(adapter.dump_json(point), context={"units": (bleb,)})
assert isclose(restored.magnitude(bleb), 1.0)
assert restored.dtype == np.float64
```

See [define units](define-units.md) and [temperatures](temperatures.md) for affine
unit definitions. To save a validated configuration or array, continue with
[persistence](serialize.md).

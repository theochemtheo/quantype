# Define units without global registration

Use this when an existing physical kind needs an application-specific unit or
calibration. To introduce a new kind or relation, use
[a custom catalogue](custom-catalogues.md) instead.

The [unit reference](../reference/units.md#define-a-unit) specifies the full
contract. A unit definition does not change constructor behavior; see
[quantities](../reference/quantities.md) for storage and conversion choices.

## Define a scale relative to an existing unit

Choose a namespaced identifier and a compatible reference unit. The scale is
in reference-unit magnitudes: one pixel below is 0.25 nm.

```python
import pytest

from quantype import Length, u

pixel = Length.define_unit("my_lab:pixel", reference=u.nm, scale=0.25, symbol="px")
width = Length[float](8, pixel)
assert width.magnitude(u.nm) == pytest.approx(2)
assert width.unit is pixel
assert Length.unit_named("my_lab:pixel", units=(pixel,)) is pixel
```

The object is usable immediately, but nothing is added to `u`. Definitions are
immutable. Namespaces reduce collisions when several applications provide
units. Use the object to construct or display a quantity; use its identifier
plus explicit definitions to decode saved data.

## Compose an affine temperature calibration

`reference_magnitude = input * scale + offset`. The reference's own offset
is composed with that map, so using Celsius below means
`kelvin = bleb * sqrt(2) + 273.15`. Zero bleb is zero Celsius.

```python
from math import sqrt

import numpy as np
import pytest

from quantype import Temperature, TemperatureDifference, u

bleb = Temperature.define_unit(
    "my_lab:bleb", reference=u.temperature.celsius, scale=sqrt(2)
)
point = Temperature[np.float64](1, bleb)
assert point.magnitude(u.temperature.celsius) > 1.4
assert point.magnitude(u.K) == pytest.approx(273.15 + sqrt(2))

# Differences need a separate, offset-free definition.
delta_bleb = TemperatureDifference.define_unit(
    "my_lab:delta_bleb", reference=u.delta_kelvin, scale=sqrt(2)
)
shifted = point + TemperatureDifference[float](2, delta_bleb)
assert shifted.magnitude(bleb) == pytest.approx(3)
```

!!! note "Define increments separately from absolute temperatures"
    An explicit `offset=` is measured in the reference unit before that
    reference's offset is applied. Only absolute-temperature units may have
    a nonzero offset. Use [temperature differences](temperatures.md#subtract-points-and-shift-by-a-difference),
    not absolute temperatures, for increments and rates.

## Supply definitions when decoding

Serialization records the unit identifier, not a globally registered object.
Pass your definitions to parsing or loading. This self-contained round trip
works without changing a public namespace:

```python
from math import sqrt

import numpy as np
import pytest

from quantype import Temperature, u

bleb = Temperature.define_unit(
    "my_lab:bleb", reference=u.temperature.celsius, scale=sqrt(2)
)
point = Temperature[np.float64](1, bleb)
wire = point.to_dict()
restored = Temperature.parse(wire, units=(bleb,))
assert restored.magnitude(bleb) == pytest.approx(point.magnitude(bleb))
assert restored.unit is bleb
```

Duplicate definitions or definitions reusing a built-in identifier are rejected
when decoding. Passing built-in unit objects in `units=` is allowed. For typed
JSON validation, supply `context={"units": (bleb,)}`; see
[validate data](validate-data.md). For typed NPZ restoration and backend/dtype
selection, see [serialize quantities](serialize.md). A custom system can expose
all of its non-catalogue definitions through
[`System.units`](../reference/systems.md#derived-units-and-decoding).

## Catch invalid definitions at creation

Use a finite positive scale and a reference of the same nominal kind. Matching
dimensions alone do not permit a different kind's reference unit.

```python
import pytest

from quantype import Length, u

with pytest.raises(ValueError, match="scale must be finite and positive"):
    Length.define_unit("my_lab:invalid_scale", reference=u.nm, scale=0)
with pytest.raises(ValueError, match="Only absolute-temperature units"):
    Length.define_unit("my_lab:invalid_offset", reference=u.nm, offset=1)
```

These errors concern the immutable definition. A valid unit can still require
more precision or numerical range than a chosen backend supports; consult
[range and precision](../reference/units.md#reference-units-and-conversion-precision)
before storing extreme conversion factors in low-precision arrays.

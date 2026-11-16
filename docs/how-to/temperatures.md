# Calculate with temperatures and differences

Convert an absolute temperature, shift it, or calculate a temperature-dependent
energy. Absolute `Temperature` values are
[affine points](../explanation/glossary.md#affine-point "An absolute value shifted by differences, rather than added to other points."),
while `TemperatureDifference` values are increments. The
[physical model](../explanation/physical-model.md#temperatures-are-points)
explains that distinction; [quantities](../reference/quantities.md#conversion-and-numerical-access)
defines conversion methods.

## Convert an absolute temperature

Provide an absolute-temperature unit such as Celsius or kelvin. Fahrenheit is
not built in; define its conversion explicitly when needed:

```python
import pytest

from quantype import Temperature, u

fahrenheit = Temperature.define_unit(
    "lab:fahrenheit",
    reference=u.celsius,
    scale=5 / 9,
    offset=-160 / 9,
    symbol="°F",
)
freezing = Temperature[float](0, u.temperature.celsius)
assert freezing.magnitude(u.K) == pytest.approx(273.15)
assert freezing.magnitude(fahrenheit) == pytest.approx(32)
assert Temperature[float](32, fahrenheit).magnitude(u.celsius) == pytest.approx(0)
assert freezing.to(u.K).magnitude() == pytest.approx(273.15)
```

Offsets affect input and presentation. Storage is always kelvin-scaled, with
or without a system. `.to(unit)` selects presentation only; choosing a unit
system with `.to_system(System)` changes the storage scale but never adds a
Celsius or Fahrenheit offset. All built-in systems store temperature in kelvin.

## Subtract points and shift by a difference

Use a difference unit for an increment, not an absolute-temperature unit.
Point minus point produces a difference; adding that difference to a point
produces a point.

```python
from typing import assert_type

import pytest

from quantype import Temperature, TemperatureDifference, u

cold = Temperature[float](0, u.celsius)
warm = Temperature[float](30, u.celsius)
difference = warm - cold
assert_type(difference, TemperatureDifference[float])
assert difference.magnitude(u.delta_celsius) == pytest.approx(30)
assert difference.magnitude(u.delta_kelvin) == pytest.approx(30)
assert repr(difference) == "TemperatureDifference(30.0 Δ°C)"
shifted = cold + TemperatureDifference[float](10, u.delta_K)
assert shifted.magnitude(u.celsius) == pytest.approx(10)
assert (warm - difference).magnitude(u.celsius) == pytest.approx(0)

with pytest.raises(TypeError, match="Cannot add two absolute Temperatures"):
    cold + warm
```

!!! warning "Points and differences are not interchangeable"
    Do not reinterpret a temperature as a difference, or the reverse, even
    though their dimensions match. Subtract a reference point or add an
    increment to one. Custom affine units need a separate difference-unit
    definition; see [define units](define-units.md#compose-an-affine-temperature-calibration).

## Average samples rather than summing points

A mean is meaningful for absolute temperatures; a sum is not. This is enforced
at runtime and in the static API:

```python
import numpy as np
import numpy.typing as npt
import pytest

from quantype import Temperature, u

samples = Temperature[npt.NDArray[np.float64]]([280, 300], u.K)
assert samples.mean().magnitude(u.K) == pytest.approx(290)
with pytest.raises(TypeError, match="Cannot sum absolute Temperatures"):
    samples.sum()
```

For more reductions and backend dispatch, see
[numerical operations](../reference/numerical-operations.md) and
[working with arrays](arrays.md).

## Multiply and scale on the kelvin scale

Multiplication, division, powers, and scaling are allowed. For example,
Boltzmann's constant times an absolute temperature is an energy:

```python
from typing import assert_type

import pytest

from quantype import Energy, Temperature, constants, u

warm = Temperature[float](300, u.K)
thermal = constants.k_B * warm
assert_type(thermal, Energy[float])
assert thermal.magnitude(u.eV) == pytest.approx(0.025852, rel=1e-4)
assert float(warm / warm) == 1
assert (warm**2).magnitude() == 90000
assert repr(2 * (20 * u.celsius)) == "Temperature(586.3 K)"
```

!!! note "Scaling uses kelvin, not the offset display"
    Twice 20 °C is 586.3 K, not 40 °C. Showing that result as 313.15 °C would
    obscure what was scaled, so the default display becomes kelvin-scaled.
    Difference units have no affine offset and keep their display when scaled.

See [choose constants](choose-constants.md) for CODATA selection and
system-preserving thermal calculations, or
[troubleshooting](../troubleshooting.md) for invalid point arithmetic.

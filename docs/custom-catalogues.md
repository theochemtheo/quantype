# Custom catalogues

An application catalogue adds your own kinds and relations to the built-in
ones. quantype generates it as a package with the same API as quantype itself:
runtime classes, stubs, units, a `numpy` module, and optionally `ujax` and
`utorch`. No type-checker plugin is needed.

## Generate an application catalogue

This declares a `SurfaceTension` kind, and says that `Pressure * Length` gives
one:

```python notest
from quantype.catalogue import QuantitySpec, UnitSpec, builtin_catalogue
from quantype.codegen import generate

catalogue = builtin_catalogue().extend(
    quantities={
        "SurfaceTension": QuantitySpec((-2, 1, 0, 0, 0, 0, 0, 0), "surface_tension"),
    },
    units={"surface_tension": UnitSpec("SurfaceTension")},
    relations={("mul", "Pressure", "Length"): "SurfaceTension"},
)
generate(catalogue, "src/labquantities", package="labquantities")
```

Generation formats its output with Ruff, so install Ruff where you run it
(`uv add --dev ruff` in an application project, or `python -m pip install ruff`).
The private stubs, `_products/`, `_constants/`, and `_generated.pyi`, are
written for type checkers in long lines, so exclude them from your own Ruff
formatting and line-length checks.

A `QuantitySpec`'s dimensions are exponents over quantype's eight base axes, in
order: length, energy, time, temperature, magnetic moment, atom, electron, and
charge. `SurfaceTension` above is energy / length². The
[catalogue reference](catalogue.md) shows every built-in kind's dimensions.

Run generation from your application project, not from quantype's source tree.
For the `src/` layout above, install your application in editable mode (for example,
`uv pip install -e .`) so Python can import `labquantities`. Alternatively, generate
into `"labquantities"` beside a script for a standalone experiment.

Then import everything from the generated package:

```python notest
from typing import assert_type
import labquantities.numpy as qnp
from labquantities import Length, Pressure, SurfaceTension, u

length = Length[float](2, u.length.angstrom)
pressure = Pressure[float](3, u.pressure.pascal)
assert_type(length * pressure, SurfaceTension[float])
assert_type(pressure * length, SurfaceTension[float])
assert_type(qnp.sqrt(length * length), Length[float])
```

Multiplication relations work in either order, and each also defines the
divisions that undo it, so the catalogue above gives
`SurfaceTension / Length -> Pressure`. A declared division takes precedence. If
two multiplications would undo to different results, declare the division
yourself.

`generate(..., check=True)` returns the names of stale files without writing
anything, and `render(...)` returns the source as strings without running any
tools.

Names and aliases that clash with names the generated package uses, such as
`Unit` or `np`, are rejected, and so are clashing quantity namespaces. Quantity
names also can't reuse imported typing names, package aliases such as `u`, or
generated kind-marker names such as `LengthKind`.

Built-in units in the package follow the process's CODATA edition, as
quantype's do. Units you define are written out as Python float literals, fixed
when the package is generated. NumPy-derived factors work, rounded to binary64,
and factors outside binary64's range are rejected.

A generated catalogue is a separate package that includes the built-in kinds;
it doesn't extend quantype's own classes. Its `Length` is a different type from
`quantype.Length`, so import all quantities from the generated package.
quantype's own operators are unchanged. Structural results use the package's
own `Quantity` subclass, so import that class for structural annotations, and
reciprocals keep the catalogue's `DimensionlessKind`. It is still a subclass of
`quantype.core.Quantity`. The package has its own `products` module, with a
product class for each unnamed product of two of its kinds, named as quantype
names them (`LengthTime`, `SurfaceTensionPerTime`); generation fails if one of
those names is also a quantity's name.

Generated quantities take a unit system parameter like the built-in ones, and
`Length[float, SI]` works with no extra definitions. Systems are defined by base
axes, so every kind in the catalogue has a unit in every system. A kind with no
named catalogue unit in a system gets a system-qualified identifier, such as `si:SurfaceTension` (symbol `J/m^2`); see
[unit systems](unit-systems.md#derived-units).

To define a new unit for an existing physical kind instead, see
[custom units](units.md#define-a-unit-without-global-registration).
